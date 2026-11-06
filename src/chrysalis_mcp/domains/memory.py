"""``memory`` domain: knowledge ingest and associative recall over the shared substrate.

``ingest`` (triples -> graph + FlyHash codes + dense vectors), ``recall`` (``ppr`` default: seeds +
associative boost + Personalized PageRank; ``seeds``: plain KNN). The seed backend is selectable:
``seeds="cosine"`` (default, exact cosine over the n-gram/embedding vectors; H-a' showed FlyHash
KNN reaches only 0.38-0.58 x cosine precision) or ``seeds="flyhash"`` (experimental). FlyHash
codes always feed the associative ``assoc`` learning (H-b) and the learning/forgetting
operations ``update`` (reinforce active KCs), ``forget`` (soft-invalidate a source) and
``consolidate`` (decay + prune ``assoc``). ``update`` also keeps the teach query as a bounded
example per node; ``recall`` (ppr) turns it into ``learned_as`` facts for boosted nodes, so a
learned alias is explainable to the model. Examples follow ``assoc`` (consolidate), not edge sources
(``forget``).
The learned boost is normalised to the overlap fraction ``BOOST_SCALE * sum(assoc) / k`` so
``lambda`` acts on a [0, 1] scale comparable to the cosine seeds: unrelated ``update`` calls cannot
swamp the cosine seeds of other queries (measured: plain-KB recall@5 stays 1.0 after 300 updates).
Depends on ``substrate``, ``algorithms`` and ``core`` only.
"""

import math
import os
import re
import threading
from typing import Any

import numpy as np

from ..algorithms import FlyHash, features, personalized_pagerank
from ..core.registry import Context, Params, Payload, Registry
from ..substrate import BackupError, open_substrate

MAX_TRIPLES = 500
MAX_TEXT_CHARS = 512
MAX_REL_CHARS = 128
MAX_K = 50
DEFAULT_K = 5
MAX_ABS_NUMBER = 1e300  # keeps sums/norms far below float64 overflow
RECALL_MODES = ("ppr", "seeds")
DEFAULT_MODE = "ppr"
SEED_BACKENDS = ("cosine", "flyhash")
DEFAULT_BACKEND = "cosine"
BACKFILL_BATCH = 500
MAX_SEEDS = 50
DEFAULT_SEEDS = 5
MAX_HOPS = 3
DEFAULT_HOPS = 2
MAX_NODES = 20000
DEFAULT_NODES = 5000
MAX_LAMBDA = 10.0
DEFAULT_LAMBDA = 0.5
# Normalised boost = lambda * BOOST_SCALE * b / k (b: summed assoc over the query's active KCs, k:
# FlyHash KCs per code). 4.0 is the narrow safe window measured by the T19 bench: x1 does not teach
# an alias, x8 lets unrelated updates leak into other queries' recall.
BOOST_SCALE = 4.0
MAX_DELTA = 100.0
ETA = 1.0  # learning rate of ``update``
MAX_PRUNE = 1e6
# A returned node is "learned" (its teach examples are explained as ``learned_as`` facts) only when
# its learned boost is at least this share of its final seed score: boost >= 0.25 * seed score.
LEARNED_SHARE = 0.25
MAX_LEARNED_EXAMPLES = 2  # teach examples explained per learned node (most similar to the query)
# An example is only quoted when it actually resembles the current query (cosine of the n-gram
# features): the learned boost is a coarse KC overlap and can touch unrelated wording (measured:
# an unrelated query scores 0.19 against a taught one, a paraphrase 0.4-0.9).
MIN_EXAMPLE_SIMILARITY = 0.3
# ...and it must share a RARE word with the query: shared question templates ("what is the ... of")
# give unrelated pairs a similar cosine, but a taught alias is a word no other node's examples use.
# Rare = present in at most RARE_TOKEN_SHARE of the BACKGROUND_EXAMPLES most recent examples taught
# to other nodes (words of < MIN_TOKEN_CHARS characters are ignored).
RARE_TOKEN_SHARE = 0.1
MIN_TOKEN_CHARS = 3
BACKGROUND_EXAMPLES = 64
LEARNED_REL = "learned_as"
TRIPLE_KEYS = frozenset(("s", "r", "o", "weight", "source"))


class _State:
    def __init__(self, path: str | os.PathLike[str] | None) -> None:
        self.read, self.write = open_substrate(path)
        p = self.read.flyhash_params()
        self.params = p
        self.hasher = FlyHash(p.d, p.m, p.k, p.c, p.seed)
        self._backfill_dense()

    def _backfill_dense(self) -> None:
        """v2 -> v3 data-preserving migration: dense vectors for nodes that predate them.

        Recomputed from the node text (n-gram features). Vectors that were supplied as external
        embeddings before v3 were never stored and cannot be recovered; those nodes fall back to
        the text features.
        """
        while batch := self.read.nodes_without_dense(BACKFILL_BATCH):
            self.write.backfill_dense(
                {row["id"]: features(row["text"], self.params.d).tolist() for row in batch}
            )


_lock = threading.Lock()
_state: _State | None = None
_configured_path: str | os.PathLike[str] | None = None


def configure(path: str | os.PathLike[str] | None) -> None:
    """Test/embedding hook: point the domain at ``path`` (``None`` = env/XDG default)."""
    global _state, _configured_path
    with _lock:
        _state, _configured_path = None, path


def _get_state() -> _State:
    global _state
    with _lock:
        if _state is None:
            _state = _State(_configured_path)
        return _state


def _error(message: str, **extra: Any) -> Payload:
    return {"status": "ERROR", "error": message, **extra}


def _text(value: Any, label: str, limit: int) -> str | None:
    """Return an error message when ``value`` is not a non-blank string within ``limit``."""
    if not isinstance(value, str) or not value.strip():
        return f"{label} must be a non-empty string"
    if len(value) > limit:
        return f"{label} exceeds {limit} characters"
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        return f"{label} contains an unpaired surrogate (invalid Unicode)"
    return None


def _number_ok(value: Any) -> bool:
    """Real number, not bool, finite and |x| <= MAX_ABS_NUMBER (huge ints never reach float())."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    if isinstance(value, float) and not math.isfinite(value):
        return False
    return abs(value) <= MAX_ABS_NUMBER


def _validate_triples(triples: Any) -> tuple[list[dict[str, Any]] | None, str | None]:
    if not isinstance(triples, list) or not triples:
        return None, "triples must be a non-empty list"
    if len(triples) > MAX_TRIPLES:
        return None, f"too many triples: {len(triples)} > {MAX_TRIPLES}"
    for i, t in enumerate(triples):
        if not isinstance(t, dict):
            return None, f"triples[{i}] must be an object"
        missing = [k for k in ("s", "r", "o") if k not in t]
        unknown = sorted(set(t) - TRIPLE_KEYS)
        if missing or unknown:
            return None, f"triples[{i}]: missing {missing}, unknown {unknown}"
        for key, limit in (("s", MAX_TEXT_CHARS), ("r", MAX_REL_CHARS), ("o", MAX_TEXT_CHARS)):
            if (msg := _text(t[key], f"triples[{i}].{key}", limit)) is not None:
                return None, msg
        w = t.get("weight", 1.0)
        if not _number_ok(w):
            return None, f"triples[{i}].weight must be a finite number with |x| <= 1e300"
        src = t.get("source")
        if src is not None and (msg := _text(src, f"triples[{i}].source", MAX_TEXT_CHARS)):
            return None, msg
    return triples, None


def _validate_embeddings(
    embeddings: Any, nodes: set[str], d: int
) -> tuple[dict[str, np.ndarray], str | None]:
    if embeddings is None:
        return {}, None
    if not isinstance(embeddings, dict):
        return {}, "embeddings must be an object {node_text: [floats]}"
    out: dict[str, np.ndarray] = {}
    for text, vec in embeddings.items():
        if text not in nodes:
            return {}, f"embeddings key {text!r} is not a node of the given triples"
        if not (isinstance(vec, list) and all(_number_ok(x) for x in vec)):
            return {}, f"embedding for {text!r} must be a list of finite numbers (|x| <= 1e300)"
        if len(vec) != d:
            return {}, f"embedding for {text!r} has dimension {len(vec)}, expected {d}"
        arr = np.asarray(vec, dtype=np.float64)
        peak = float(np.max(np.abs(arr)))
        if peak == 0.0:
            return {}, f"embedding for {text!r} must be non-zero"
        out[text] = arr / peak  # rescale first: squared norms of 1e300-sized values overflow
        # (WTA hashing is invariant to positive scaling, so codes are unchanged)
    return out, None


def _ingest(params: Params, ctx: Context) -> Payload:
    triples, problem = _validate_triples(params["triples"])
    if problem or triples is None:
        return _error(problem or "invalid triples")
    default_source = params.get("source")
    if default_source is not None and (msg := _text(default_source, "source", MAX_TEXT_CHARS)):
        return _error(msg)
    st = _get_state()
    nodes = {t[key] for t in triples for key in ("s", "o")}
    vectors, problem = _validate_embeddings(params.get("embeddings"), nodes, st.params.d)
    if problem:
        return _error(problem)

    def vector_for(text: str) -> np.ndarray:
        vec = vectors.get(text)
        return vec if vec is not None else features(text, st.params.d)

    def code_for(text: str) -> list[int]:
        return st.hasher.hash(vector_for(text)).tolist()

    def dense_for(text: str) -> list[float]:
        return vector_for(text).tolist()

    # Exactly one write call: nodes, edges, codes, dense vectors and log commit together.
    result = st.write.ingest_batch(triples, code_for, default_source, dense_for)
    nodes_added = len(result.nodes) - len(result.preexisting)
    codes_added = len(result.coded)
    ignored = [text for text in vectors if text not in result.coded]
    return {
        "status": "OK",
        "summary": f"{len(triples)} edges, {nodes_added} new nodes, {codes_added} codes",
        "nodes_added": nodes_added,
        "edges_added": result.edges,
        "codes_added": codes_added,
        "vec_index": st.read.vec_available(),
        "embeddings_ignored": ignored,
        "terse": {"edges_added": result.edges},
    }


def _int_param(params: Params, name: str, default: int, lo: int, hi: int) -> int | str:
    value = params.get(name, default)
    if isinstance(value, bool) or not isinstance(value, int) or not lo <= value <= hi:
        return f"{name} must be an integer in [{lo}, {hi}]"
    return value


def _active(st: _State, query: str) -> list[int]:
    return st.hasher.hash(features(query, st.params.d)).tolist()


def _live_nearest(st: _State, active: list[int], want: int) -> list[tuple[int, int]]:
    """Top-``want`` nearest nodes that still have a valid edge; widens the KNN window if needed."""
    total = st.read.counts()["nodes"]
    fetch = want
    while True:
        rows = st.read.nearest_codes(active, fetch)
        live = st.read.live_nodes(n for n, _ in rows)
        kept = [(n, o) for n, o in rows if n in live]
        if len(kept) >= want or len(rows) < fetch or fetch >= total:
            return kept[:want]
        fetch = min(fetch * 4, total)


def _live_nearest_dense(st: _State, query: str, want: int) -> list[tuple[int, float]]:
    """Top-``want`` live nodes by exact cosine to the query features; widens like the code KNN."""
    vec = features(query, st.params.d)
    total = st.read.counts()["nodes"]
    fetch = want
    while True:
        rows = st.read.nearest_dense(vec, fetch)
        live = st.read.live_nodes(n for n, _ in rows)
        kept = [(n, c) for n, c in rows if n in live]
        if len(kept) >= want or len(rows) < fetch or fetch >= total:
            return kept[:want]
        fetch = min(fetch * 4, total)


def _seed_scores(st: _State, query: str, backend: str, want: int) -> list[tuple[int, float]]:
    """``[(node_id, score)]``: cosine similarity or FlyHash code overlap, best first."""
    if backend == "cosine":
        return _live_nearest_dense(st, query, want)
    return [(n, float(o)) for n, o in _live_nearest(st, _active(st, query), want)]


def _recall(params: Params, ctx: Context) -> Payload:
    mode = params.get("mode", DEFAULT_MODE)
    if mode not in RECALL_MODES:
        return _error(f"unknown mode {mode!r}", valid=list(RECALL_MODES))
    backend = params.get("seeds", DEFAULT_BACKEND)
    if backend not in SEED_BACKENDS:
        return _error(f"unknown seeds backend {backend!r}", valid=list(SEED_BACKENDS))
    if (msg := _text(params["query"], "query", MAX_TEXT_CHARS)) is not None:
        return _error(msg)
    ints = {
        "k": _int_param(params, "k", DEFAULT_K, 1, MAX_K),
        "n_seeds": _int_param(params, "n_seeds", DEFAULT_SEEDS, 1, MAX_SEEDS),
        "hops": _int_param(params, "hops", DEFAULT_HOPS, 0, MAX_HOPS),
        "max_nodes": _int_param(params, "max_nodes", DEFAULT_NODES, 1, MAX_NODES),
    }
    for value in ints.values():
        if isinstance(value, str):
            return _error(value)
    k, n_seeds, hops, max_nodes = (int(ints[n]) for n in ("k", "n_seeds", "hops", "max_nodes"))
    lam = params.get("lambda", DEFAULT_LAMBDA)
    if not _number_ok(lam) or not 0 <= lam <= MAX_LAMBDA:
        return _error(f"lambda must be a number in [0, {MAX_LAMBDA:g}]")
    st = _get_state()
    if mode == "seeds":
        results = []
        score_key = "similarity" if backend == "cosine" else "overlap"
        for node_id, score in _seed_scores(st, params["query"], backend, k):
            node = st.read.get_node(node_id)
            assert node is not None
            results.append({"text": node["text"], score_key: score})
        return {
            "status": "OK",
            "summary": f"{len(results)} seed(s) for mode={mode}, seeds={backend}",
            "mode": mode,
            "seeds": backend,
            "results": results,
            "terse": {"results": [r["text"] for r in results]},
        }
    return _recall_ppr(st, params["query"], backend, k, n_seeds, hops, max_nodes, float(lam))


def _recall_ppr(
    st: _State,
    query: str,
    backend: str,
    k: int,
    n_seeds: int,
    hops: int,
    max_nodes: int,
    lam: float,
) -> Payload:
    active = _active(st, query)  # FlyHash codes always drive the learned associative boost
    seeds: dict[int, float] = dict(_seed_scores(st, query, backend, n_seeds))
    # Associative boost learned by ``update``, as an overlap fraction on the cosine seeds' scale:
    # s_j += lambda * BOOST_SCALE * (sum_{kc in active} assoc[kc, j]) / k.
    boost: dict[int, float] = {}
    contrib: dict[int, float] = {}  # boost actually added to each node's seed score
    for row in st.read.assoc_for(active):
        boost[row["node_id"]] = boost.get(row["node_id"], 0.0) + float(row["weight"])
    if lam > 0.0:
        alive = st.read.live_nodes(boost)
        top = sorted(
            ((j, b) for j, b in boost.items() if b > 0.0 and j in alive),
            key=lambda t: (-t[1], t[0]),
        )
        for node_id, b in top[:n_seeds]:
            contrib[node_id] = lam * BOOST_SCALE * b / st.params.k
            seeds[node_id] = seeds.get(node_id, 0.0) + contrib[node_id]
    seeds = {n: s for n, s in seeds.items() if s > 0.0}
    node_ids, edges = st.read.subgraph(sorted(seeds), hops, max_nodes)
    index = {n: i for i, n in enumerate(node_ids)}
    s_vec = np.zeros(len(node_ids))
    for n, s in seeds.items():
        if n in index:
            s_vec[index[n]] = s
    scores = personalized_pagerank(
        len(node_ids), [(index[u], index[v], w) for u, v, w in edges], s_vec
    )
    order = sorted(range(len(node_ids)), key=lambda i: (-scores[i], node_ids[i]))
    order = [i for i in order if scores[i] > 0.0][:k]
    texts = {}
    for i in order:
        node = st.read.get_node(node_ids[i])
        assert node is not None
        texts[node_ids[i]] = node["text"]
    results = [{"text": texts[node_ids[i]], "score": float(scores[i])} for i in order]
    top_ids = [node_ids[i] for i in order]
    learned_ids = [n for n in top_ids if _is_learned(n, contrib, seeds)]
    learned = _learned_facts(st, query, learned_ids, texts)
    # Learned facts come first (they bridge the user's wording to the node); the total stays <= 2k.
    facts = (
        [*learned, *_facts(st, top_ids, 2 * k)][: 2 * k] if learned else _facts(st, top_ids, 2 * k)
    )
    flagged = {texts[n] for n in learned_ids if any(f[2] == texts[n] for f in learned)}
    for r in results:
        if r["text"] in flagged:
            r["learned"] = True
    return {
        "status": "OK",
        "summary": f"{len(results)} node(s), {len(facts)} fact(s) for mode=ppr",
        "mode": "ppr",
        "seeds": backend,
        "results": results,
        "facts": facts,
        "terse": {"results": [r["text"] for r in results]},
    }


def _tokens(text: str) -> frozenset[str]:
    return frozenset(w for w in re.findall(r"\w+", text.casefold()) if len(w) >= MIN_TOKEN_CHARS)


def _is_learned(node_id: int, contrib: dict[int, float], seeds: dict[int, float]) -> bool:
    """The boost is a meaningful share of the node's final seed score (``LEARNED_SHARE``)."""
    c = contrib.get(node_id, 0.0)
    return c > 0.0 and c >= LEARNED_SHARE * seeds.get(node_id, 0.0)


def _learned_facts(
    st: _State, query: str, node_ids: list[int], texts: dict[int, str]
) -> list[list[str]]:
    """``[example_query, "learned_as", node_text]`` for the stored teach examples of learned nodes.

    Per node the ``MAX_LEARNED_EXAMPLES`` examples most similar to ``query`` (cosine of the n-gram
    features, ties by recency) that reach ``MIN_EXAMPLE_SIMILARITY`` and share a rare word with the
    query are kept; nodes without qualifying examples emit nothing.
    """
    if not node_ids:
        return []
    qvec = features(query, st.params.d)
    background = [_tokens(q) for q in st.read.recent_examples(node_ids, BACKGROUND_EXAMPLES)]
    rare_cap = RARE_TOKEN_SHARE * len(background)
    q_rare = {t for t in _tokens(query) if sum(t in b for b in background) <= rare_cap}
    out: list[list[str]] = []
    for node_id, rows in st.read.examples_for(node_ids).items():
        scored = [(float(features(r["query"], st.params.d) @ qvec), r) for r in rows]
        scored = [
            (sim, r)
            for sim, r in scored
            if sim >= MIN_EXAMPLE_SIMILARITY and q_rare & _tokens(r["query"])
        ]
        scored.sort(key=lambda t: (-t[0], -t[1]["ts"]))
        out.extend(
            [r["query"], LEARNED_REL, texts[node_id]] for _, r in scored[:MAX_LEARNED_EXAMPLES]
        )
    return out


def _facts(st: _State, top_ids: list[int], limit: int) -> list[list[str]]:
    """Up to ``limit`` valid triples ``[s, r, o]`` touching the top nodes (inside-top first)."""
    inside = set(top_ids)
    seen: dict[int, dict[str, Any]] = {}
    for node_id in top_ids:
        for edge in st.read.neighbors(node_id, valid_only=True):
            seen.setdefault(edge["id"], edge)
    ranked = sorted(
        seen.values(),
        key=lambda e: (-int(e["subj"] in inside and e["obj"] in inside), e["id"]),
    )[:limit]
    names: dict[int, str] = {}
    for e in ranked:
        for nid in (e["subj"], e["obj"]):
            if nid not in names:
                node = st.read.get_node(nid)
                assert node is not None
                names[nid] = node["text"]
    return [[names[e["subj"]], e["rel"], names[e["obj"]]] for e in ranked]


def _update(params: Params, ctx: Context) -> Payload:
    if (msg := _text(params["query"], "query", MAX_TEXT_CHARS)) is not None:
        return _error(msg)
    node_key = params["node"]
    if isinstance(node_key, bool) or not isinstance(node_key, (str, int)):
        return _error("node must be a node text (string) or id (integer)")
    if isinstance(node_key, str) and (msg := _text(node_key, "node", MAX_TEXT_CHARS)):
        return _error(msg)
    delta = params.get("delta", 1.0)
    if not _number_ok(delta) or abs(delta) > MAX_DELTA:
        return _error(f"delta must be a finite number with |delta| <= {MAX_DELTA:g}")
    st = _get_state()
    node = st.read.get_node(node_key)
    if node is None:
        return _error(f"unknown node {node_key!r}")
    active = _active(st, params["query"])
    try:
        n_kcs = st.write.reinforce(active, node["id"], ETA * float(delta), example=params["query"])
    except ValueError as exc:  # node vanished between lookup and write
        return _error(str(exc))
    return {
        "status": "OK",
        "summary": f"reinforced {n_kcs} KC(s) -> node {node['text']!r}",
        "node": node["text"],
        "kcs_reinforced": n_kcs,
        "terse": {"kcs_reinforced": n_kcs},
    }


def _forget(params: Params, ctx: Context) -> Payload:
    if (msg := _text(params["source"], "source", MAX_TEXT_CHARS)) is not None:
        return _error(msg)
    version = params.get("version")
    if version is not None and (isinstance(version, bool) or not isinstance(version, int)):
        return _error("version must be an integer")
    st = _get_state()
    try:
        backup = st.write.backup("pre-forget")  # lossy write: snapshot first, fail closed
    except BackupError as exc:
        return _error(f"backup before forget failed, nothing written: {exc}")
    edge_ids = st.write.invalidate_ids(params["source"], version)
    return {
        "status": "OK",
        "summary": f"invalidated {len(edge_ids)} edge(s)",
        "invalidated": len(edge_ids),
        "backup_slot": backup.get("slot"),
        "edge_ids": edge_ids,
        "terse": {"invalidated": len(edge_ids)},
    }


def _consolidate(params: Params, ctx: Context) -> Payload:
    decay = params.get("decay", 0.9)
    if not _number_ok(decay) or not 0 < decay <= 1:
        return _error("decay must be a number in (0, 1]")
    prune = params.get("prune_below", 1e-3)
    if not _number_ok(prune) or not 0 <= prune <= MAX_PRUNE:
        return _error(f"prune_below must be a number in [0, {MAX_PRUNE:g}]")
    st = _get_state()
    try:
        backup = st.write.backup("pre-consolidate")  # lossy write: snapshot first, fail closed
    except BackupError as exc:
        return _error(f"backup before consolidate failed, nothing written: {exc}")
    decayed, pruned = st.write.decay_assoc(float(decay), float(prune))
    return {
        "status": "OK",
        "summary": f"decayed {decayed}, pruned {pruned}",
        "backup_slot": backup.get("slot"),
        "decayed": decayed,
        "pruned": pruned,
        "terse": {"decayed": decayed, "pruned": pruned},
    }


def _restore(params: Params, ctx: Context) -> Payload:
    st = _get_state()
    slot = params.get("slot")
    if slot is None:
        backups = st.write.list_backups()
        return {
            "status": "OK",
            "summary": f"{len(backups)} backup(s)",
            "backups": backups,
            "terse": {"slots": [b["slot"] for b in backups]},
        }
    try:
        done = st.write.restore(slot)
    except BackupError as exc:
        return _error(str(exc))
    return {
        "status": "OK",
        "summary": f"restored slot {done['restored_slot']}; previous state in slot "
        f"{done['pre_restore_slot']}",
        **done,
        "terse": {"restored_slot": done["restored_slot"]},
    }


def register(registry: Registry) -> None:
    registry.register(
        "memory",
        "ingest",
        _ingest,
        params_schema={
            "triples": {
                "type": "array",
                "required": True,
                "max_items": MAX_TRIPLES,
                "items": "{s, r, o, weight?, source?}",
            },
            "embeddings": {"type": "object", "required": False},
            "source": {"type": "string", "required": False},
        },
        description=(
            "Ingest knowledge triples: adds nodes/edges plus a FlyHash sparse code and a dense "
            "vector per new node (character 3-gram features unless 'embeddings' "
            "{node_text: [floats]} are given)."
        ),
        example={
            "triples": [{"s": "París", "r": "capital_de", "o": "Francia"}],
            "source": "wiki",
        },
    )
    int_opt = {"type": "integer", "required": False}
    registry.register(
        "memory",
        "recall",
        _recall,
        params_schema={
            "query": {"type": "string", "required": True},
            "k": {**int_opt, "min": 1, "max": MAX_K, "default": DEFAULT_K},
            "mode": {"type": "string", "required": False, "enum": list(RECALL_MODES)},
            "seeds": {
                "type": "string",
                "required": False,
                "enum": list(SEED_BACKENDS),
                "default": DEFAULT_BACKEND,
            },
            "n_seeds": {**int_opt, "min": 1, "max": MAX_SEEDS, "default": DEFAULT_SEEDS},
            "hops": {**int_opt, "min": 0, "max": MAX_HOPS, "default": DEFAULT_HOPS},
            "max_nodes": {**int_opt, "min": 1, "max": MAX_NODES, "default": DEFAULT_NODES},
            "lambda": {
                "type": "number",
                "required": False,
                "min": 0,
                "max": MAX_LAMBDA,
                "default": DEFAULT_LAMBDA,
            },
        },
        description=(
            "Associative recall. mode='ppr' (default): seeds + learned boost -> "
            "Personalized PageRank over the graph, returns top-k nodes and supporting facts. "
            "lambda scales the learned boost on a [0, 1] range comparable to the cosine seeds. "
            "When a returned node ranks mainly because of what 'update' taught it, the facts "
            "include up to 2 [taught_query, 'learned_as', node] entries (the taught queries "
            "most similar to yours) and the result item carries learned=true: read it as "
            "'queries worded like taught_query refer to node'. "
            "mode='seeds': nearest nodes only (no learned boost, no learned_as). "
            "seeds= selects the seed backend, distinct from "
            "mode='seeds': seeds='cosine' (default): exact cosine over the "
            "dense vectors; seeds='flyhash' (experimental): FlyHash code overlap."
        ),
        example={"query": "paris", "k": 5, "mode": "ppr"},
    )
    registry.register(
        "memory",
        "update",
        _update,
        params_schema={
            "query": {"type": "string", "required": True},
            "node": {"type": "string", "required": True},
            "delta": {
                "type": "number",
                "required": False,
                "min": -MAX_DELTA,
                "max": MAX_DELTA,
                "default": 1.0,
            },
        },
        description=(
            "Reinforce the association between the query's active KCs and a node "
            "(Hebbian-style); later ppr recalls of similar queries rank that node higher. "
            "The boost is bounded (recall lambda acts on a [0, 1] scale), so it does not "
            "affect unrelated queries. A positive delta also stores the query as a teach "
            "example of the node (3 most recent per node, deduped) so recall can explain the "
            "association with a 'learned_as' fact; a negative delta that cancels the node's "
            "association for this query removes that example."
        ),
        example={"query": "capital of france", "node": "París", "delta": 1.0},
    )
    registry.register(
        "memory",
        "forget",
        _forget,
        params_schema={
            "source": {"type": "string", "required": True},
            "version": {"type": "integer", "required": False},
        },
        description="Soft-invalidate every valid edge of a source (optionally one version).",
        example={"source": "wiki"},
    )
    registry.register(
        "memory",
        "consolidate",
        _consolidate,
        params_schema={
            "decay": {"type": "number", "required": False, "min": 0, "max": 1, "default": 0.9},
            "prune_below": {"type": "number", "required": False, "default": 1e-3},
        },
        description=(
            "Decay learned associations (assoc *= decay) and prune |w| < prune_below; teach "
            "examples of nodes left without any association are dropped. 'forget' acts on edge "
            "sources only: examples follow assoc, not sources."
        ),
        example={"decay": 0.9, "prune_below": 0.001},
    )
    registry.register(
        "memory",
        "restore",
        _restore,
        params_schema={"slot": {"type": "integer", "required": False, "min": 0, "max": 2}},
        description=(
            "Without 'slot': list the rotating backups (taken before forget/consolidate). "
            "With 'slot': verify it, snapshot the current state (reversible), then restore it."
        ),
        example={},
    )
