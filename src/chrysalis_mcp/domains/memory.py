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

Bio layer (substrate v5; each mechanism has a biological analogue and an explicit trade-off):

* Negative selection (thymus). ``learned_as`` used to ask "does the query share a RARE word with
  the example?" over a window of the 64 latest examples: order dependent and fooled by common
  words. Now the *self* profile is a count: per Kenyon cell (KC), the number of DISTINCT nodes whose
  teach queries activate it (``node_kcs``). KCs that fire for ``SELF_FRACTION`` of the taught nodes
  (at least ``SELF_MIN_NODES``) are non-specific "self" (question templates, filler words) and are
  ignored when matching a query to an example; at least ``MIN_SPECIFIC_KCS`` shared non-self KCs
  are required. The decision depends on the SET of taught examples, never on their order.
* Quorum sensing (bacteria). Each node counts its DISTINCT teach queries (near duplicates, cosine
  >= ``TEACH_DEDUPE_SIM``, count once). A ``learned_as`` explanation is ``confirmed`` at
  ``quorum`` (default 2) or more, else ``provisional``. The ranking boost is never gated by it;
  ``learned_min_status="confirmed"`` additionally hides the provisional explanations.
* Replay (hippocampal sleep). ``consolidate(replay=True)`` re-applies the stored examples at
  ``replay_rate`` before decaying, so what the user taught survives pruning better than residue.
* Pheromone / Physarum. ``recall(track_usage=True)`` counts (bounded) the edges around the returned
  nodes; ``consolidate(usage_weighting=True)`` lets used nodes decay less and unused ones more,
  and evaporates the counters. Only learned ``assoc`` and the usage counters change: source
  facts (edges) are never deleted or reweighted.

Low-priority bio tools (substrate v6; each opt-in, defaults keep the previous behaviour):

* Synaptic tagging & capture (Frey & Morris 1997). ``update`` stamps the associations it touches
  with a logical update counter (a "tag"). ``consolidate(capture=True, capture_window=N)`` exempts
  from decay and prune the associations tagged within the last ``N`` updates and those of nodes
  with at least ``capture_confirmations`` distinct teach phrasings ("captured" to long-term);
  untagged residue decays as usual. The result reports the counts.
* Lateral inhibition (fly APL neuron, Lin et al. 2014). ``recall(inhibition="adaptive", z=1.0)``
  keeps, for the QUERY code only, the Kenyon cells above ``mean + z * std`` (clamped to
  ``[k/2, 2k]``) instead of a fixed ``k``; stored codes never change. The learned boost is
  normalised by the query's actual number of cells, so its scale is unchanged.
* Sparse distributed memory (Kanerva 1988). ``recall(mode="sdm", radius=r)`` reads every stored node
  code within Hamming distance ``r`` of the query code, ranked by graded activation
  ``(r + 1 - hamming) / (r + 1)``; deterministic, paginated by ``cursor``. No learned boost.
* FlyVec-lite (Liang et al. 2021). ``flyvec_train`` learns per-token Kenyon-cell codes from the
  knowledge base itself; ``recall(encoder="flyvec")`` retrieves the seeds with them instead of
  character n-grams (limits in ``algorithms/flyvec.py``: no generalisation to unseen words).
* ``learned_legacy=True`` restores the pre-v5 ``learned_as`` gate (rare shared word over the 64 most
  recent examples; order dependent, fooled by common words) for callers that relied on it.

Depends on ``substrate``, ``algorithms`` and ``core`` only.
"""

import math
import os
import re
import threading
from dataclasses import dataclass
from collections.abc import Callable, Iterable
from typing import Any

import numpy as np

from ..algorithms import FlyHash, features, flyvec, personalized_pagerank
from ..core.registry import Context, Params, Payload, Registry
from ..substrate import BackupError, open_substrate

MAX_TRIPLES = 500
MAX_TEXT_CHARS = 512
MAX_REL_CHARS = 128
MAX_K = 50
DEFAULT_K = 5
MAX_ABS_NUMBER = 1e300  # keeps sums/norms far below float64 overflow
RECALL_MODES = ("ppr", "seeds", "sdm")
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
# ...and it must share SPECIFIC cells with the query (negative selection, see module docstring).
# A KC is "self" when it fires for >= max(SELF_MIN_NODES, ceil(SELF_FRACTION * taught nodes))
# distinct nodes. Defaults come from a sweep on seeds disjoint from the bench seeds (report).
SELF_FRACTION = 0.02
SELF_MIN_NODES = 3
MIN_SPECIFIC_KCS = 4
# ...and cover this share of the example's own specific cells: "Amber Well" shares the word Amber
# with the taught "Amber Hive" but not its distinctive PAIR, so it covers only part of it.
MIN_COVERAGE = 0.2
# Quorum sensing: a node's teach queries closer than this (cosine of the n-gram features) are the
# same confirmation (typo, casing); a paraphrase (measured 0.4-0.8) is a new one.
TEACH_DEDUPE_SIM = 0.85
DEFAULT_QUORUM = 2
MAX_QUORUM = 16
LEARNED_STATUSES = ("provisional", "confirmed")  # ascending strength
DEFAULT_LEARNED_STATUS = "provisional"
# Replay: assoc[kc, node] += rate * ETA * min(example weight, REPLAY_WEIGHT_CAP) per stored example.
DEFAULT_REPLAY_RATE = 0.25
REPLAY_WEIGHT_CAP = 3.0
# Usage (pheromone): +1 per recall on the edges around the returned nodes, capped; a node's usage is
# the sum over its edges, saturating at USAGE_SATURATION. Per node the consolidate factor is
# interpolated between decay ** UNUSED_DECAY_EXPONENT (never used) and
# 1 - (1 - decay) * (1 - USAGE_PROTECT) (saturated): protection only, never growth above 1.
USAGE_CAP = 10.0
USAGE_SATURATION = 4.0
USAGE_PROTECT = 0.75
UNUSED_DECAY_EXPONENT = 1.5
USAGE_EVAPORATION = 0.9
USAGE_FLOOR = 0.05
MAX_USAGE_EDGES = 256
# Legacy (pre-v5) learned_as gate, kept behind ``learned_legacy=True``.
RARE_TOKEN_SHARE = 0.1
MIN_TOKEN_CHARS = 3
BACKGROUND_EXAMPLES = 64
# Low-priority bio tools.
INHIBITIONS = ("fixed", "adaptive")
DEFAULT_INHIBITION = "fixed"
DEFAULT_Z = 1.0
MAX_Z = 5.0
ENCODERS = ("ngram", "flyvec")
DEFAULT_ENCODER = "ngram"
DEFAULT_CAPTURE_WINDOW = 50
MAX_CAPTURE_WINDOW = 1_000_000
DEFAULT_CAPTURE_CONFIRMATIONS = 2
SDM_MAX_PAGE = MAX_K
FLYVEC_MAX_EDGES = 20000
FLYVEC_CACHE_LIVE_FACTOR = 4  # widen factor when the best FlyVec neighbours are not live
LEARNED_REL = "learned_as"
TRIPLE_KEYS = frozenset(("s", "r", "o", "weight", "source"))


class _State:
    def __init__(self, path: str | os.PathLike[str] | None) -> None:
        self.read, self.write = open_substrate(path)
        p = self.read.flyhash_params()
        self.params = p
        self.hasher = FlyHash(p.d, p.m, p.k, p.c, p.seed)
        self._backfill_dense()
        self._backfill_teach()

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

    def _backfill_teach(self) -> None:
        """v4 -> v5 data-preserving migration: teach queries (with codes) of the old examples."""
        while batch := self.read.examples_without_teach(BACKFILL_BATCH):
            self.write.backfill_teach(
                [
                    (
                        row["node_id"],
                        row["query"],
                        self.hasher.hash(features(row["query"], self.params.d)).tolist(),
                        row["ts"],
                    )
                    for row in batch
                ]
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


def _active(st: _State, query: str, inhibition: str = DEFAULT_INHIBITION, z: float = DEFAULT_Z):
    """Active KCs of a text; ``inhibition="adaptive"`` is a QUERY-time option (stored codes: fixed)."""
    vec = features(query, st.params.d)
    if inhibition == "adaptive":
        return st.hasher.hash_adaptive(vec, z).tolist()
    return st.hasher.hash(vec).tolist()


@dataclass(frozen=True)
class _Opts:
    """Opt-in recall options of the low-priority bio tools (defaults = previous behaviour)."""

    inhibition: str = DEFAULT_INHIBITION
    z: float = DEFAULT_Z
    encoder: str = DEFAULT_ENCODER
    legacy: bool = False


def _flyvec_scores(st: _State, query: str, want: int) -> list[tuple[int, float]] | None:
    """Top live nodes by cosine of FlyVec vectors, or ``None`` when no query token is known."""
    info = st.read.flyvec_info()
    assert info is not None
    codes = st.read.flyvec_tokens(flyvec.tokenize(query))
    qvec = flyvec.encode(flyvec.tokenize(query), codes, int(info["hidden"]))
    if not qvec.any():
        return None
    ids, matrix = st.read.flyvec_node_vectors()
    if not ids:
        return []
    sims = np.round(matrix @ qvec, 6)
    order = np.lexsort((np.asarray(ids), -sims))
    out: list[tuple[int, float]] = []
    fetch = want
    while True:
        window = [(ids[i], float(sims[i])) for i in order[:fetch] if sims[i] > 0.0]
        live = st.read.live_nodes(n for n, _ in window)
        out = [(n, c) for n, c in window if n in live]
        if len(out) >= want or fetch >= len(ids) or len(window) < fetch:
            return out[:want]
        fetch = min(fetch * FLYVEC_CACHE_LIVE_FACTOR, len(ids))


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


def _seed_scores(
    st: _State,
    query: str,
    backend: str,
    want: int,
    opts: _Opts = _Opts(),  # noqa: B008 - frozen dataclass, immutable default
    notes: dict[str, Any] | None = None,
) -> list[tuple[int, float]]:
    """``[(node_id, score)]``: cosine similarity or FlyHash code overlap, best first.

    ``encoder="flyvec"`` swaps the cosine backend's vectors for the learned FlyVec ones; when no
    token of the query is known it falls back to the n-gram backend and says so in ``notes``.
    """
    if backend == "cosine" and opts.encoder == "flyvec":
        found = _flyvec_scores(st, query, want)
        if found is not None:
            return found
        if notes is not None:
            notes["encoder_fallback"] = "ngram"
    if backend == "cosine":
        return _live_nearest_dense(st, query, want)
    return [
        (n, float(o))
        for n, o in _live_nearest(st, _active(st, query, opts.inhibition, opts.z), want)
    ]


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
    quorum = _int_param(params, "quorum", DEFAULT_QUORUM, 1, MAX_QUORUM)
    if isinstance(quorum, str):
        return _error(quorum)
    status = params.get("learned_min_status", DEFAULT_LEARNED_STATUS)
    if status not in LEARNED_STATUSES:
        return _error(f"unknown learned_min_status {status!r}", valid=list(LEARNED_STATUSES))
    track = params.get("track_usage", False)
    if not isinstance(track, bool):
        return _error("track_usage must be a boolean")
    inhibition = params.get("inhibition", DEFAULT_INHIBITION)
    if inhibition not in INHIBITIONS:
        return _error(f"unknown inhibition {inhibition!r}", valid=list(INHIBITIONS))
    z = params.get("z", DEFAULT_Z)
    if not _number_ok(z) or not 0 <= z <= MAX_Z:
        return _error(f"z must be a number in [0, {MAX_Z:g}]")
    encoder = params.get("encoder", DEFAULT_ENCODER)
    if encoder not in ENCODERS:
        return _error(f"unknown encoder {encoder!r}", valid=list(ENCODERS))
    legacy = params.get("learned_legacy", False)
    if not isinstance(legacy, bool):
        return _error("learned_legacy must be a boolean")
    opts = _Opts(inhibition, float(z), encoder, legacy)
    st = _get_state()
    if encoder == "flyvec" and st.read.flyvec_info() is None:
        return _error("no FlyVec model: run memory.flyvec_train first", next_action="flyvec_train")
    if mode == "sdm":
        return _recall_sdm(st, params, k, opts)
    notes: dict[str, Any] = {}
    if mode == "seeds":
        results = []
        score_key = "similarity" if backend == "cosine" else "overlap"
        for node_id, score in _seed_scores(st, params["query"], backend, k, opts, notes):
            node = st.read.get_node(node_id)
            assert node is not None
            results.append({"text": node["text"], score_key: score})
        return {
            "status": "OK",
            "summary": f"{len(results)} seed(s) for mode={mode}, seeds={backend}",
            "mode": mode,
            "seeds": backend,
            "results": results,
            **notes,
            "terse": {"results": [r["text"] for r in results]},
        }
    return _recall_ppr(
        st, params["query"], backend, k, n_seeds, hops, max_nodes, float(lam), quorum, status,
        track, opts,
    )  # fmt: skip


def _recall_sdm(st: _State, params: Params, k: int, opts: _Opts) -> Payload:
    """Kanerva read: every live node code within Hamming ``radius`` of the query code, ranked."""
    p = st.params
    radius = params.get("radius", p.k)
    cursor = params.get("cursor", "0")
    if isinstance(radius, bool) or not isinstance(radius, int) or not 0 <= radius <= 4 * p.k:
        return _error(f"radius must be an integer in [0, {4 * p.k}] (Hamming bits)")
    if not isinstance(cursor, str) or not cursor.isdigit():
        return _error("cursor must be the string returned by the previous page")
    offset = int(cursor)
    active = _active(st, params["query"], opts.inhibition, opts.z)
    total = st.read.counts()["nodes"]
    fetch = min(max(offset + k, 64), max(total, 1))
    while True:
        rows = st.read.nearest_codes(active, fetch)
        # hamming = |query| + k_node - 2 * overlap (every stored code has exactly p.k cells)
        inside = [(n, len(active) + p.k - 2 * o) for n, o in rows]
        done = len(rows) < fetch or fetch >= total or (inside and inside[-1][1] > radius)
        if done:
            break
        fetch = min(fetch * 4, total)
    inside = [(n, h) for n, h in inside if h <= radius]
    live = st.read.live_nodes(n for n, _ in inside)
    hits = [(n, h) for n, h in inside if n in live]
    page = hits[offset : offset + k]
    more = offset + len(page) < len(hits)
    results = []
    for node_id, ham in page:
        node = st.read.get_node(node_id)
        assert node is not None
        results.append(
            {
                "text": node["text"],
                "score": round((radius + 1 - ham) / (radius + 1), 6),
                "hamming": ham,
            }
        )
    return {
        "status": "OK",
        "summary": f"{len(results)} of {len(hits)} node(s) within Hamming radius {radius}",
        "mode": "sdm",
        "radius": radius,
        "in_radius": len(hits),
        "results": results,
        "cursor": str(offset + len(page)) if more else None,
        "next_action": "Pass cursor to read the next page." if more else "Done.",
        "terse": {"results": [r["text"] for r in results]},
    }


def _recall_ppr(
    st: _State,
    query: str,
    backend: str,
    k: int,
    n_seeds: int,
    hops: int,
    max_nodes: int,
    lam: float,
    quorum: int = DEFAULT_QUORUM,
    min_status: str = DEFAULT_LEARNED_STATUS,
    track_usage: bool = False,
    opts: _Opts = _Opts(),  # noqa: B008 - frozen dataclass, immutable default
) -> Payload:
    # FlyHash codes always drive the learned associative boost (query side: optional inhibition)
    active = _active(st, query, opts.inhibition, opts.z)
    notes: dict[str, Any] = {}
    seeds: dict[int, float] = dict(_seed_scores(st, query, backend, n_seeds, opts, notes))
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
            contrib[node_id] = lam * BOOST_SCALE * b / len(active)  # = k unless adaptive
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
    learned, details = _learned_facts(
        st, query, active, learned_ids, texts, quorum, min_status, opts.legacy
    )
    # Learned facts come first (they bridge the user's wording to the node); the total stays <= 2k.
    facts = (
        [*learned, *_facts(st, top_ids, 2 * k)][: 2 * k] if learned else _facts(st, top_ids, 2 * k)
    )
    by_text = {texts[n]: d for n, d in details.items()}
    for r in results:
        if r["text"] in by_text:
            r["learned"] = True
            r["learned_status"] = by_text[r["text"]]["status"]
    if track_usage:
        _track_usage(st, top_ids)
    payload: Payload = {
        "status": "OK",
        "summary": f"{len(results)} node(s), {len(facts)} fact(s) for mode=ppr",
        "mode": "ppr",
        "seeds": backend,
        "results": results,
        "facts": facts,
        **notes,
        "terse": {"results": [r["text"] for r in results]},
    }
    if by_text:
        payload["learned"] = [{"node": t, **d} for t, d in by_text.items()]
    return payload


def _track_usage(st: _State, top_ids: list[int]) -> None:
    """Pheromone: +1 (bounded) on the valid edges touching the returned nodes."""
    edge_ids: dict[int, None] = {}
    for node_id in top_ids:
        for edge in st.read.neighbors(node_id, valid_only=True):
            edge_ids.setdefault(edge["id"])
    st.write.bump_usage(list(edge_ids)[:MAX_USAGE_EDGES], USAGE_CAP)


def _is_learned(node_id: int, contrib: dict[int, float], seeds: dict[int, float]) -> bool:
    """The boost is a meaningful share of the node's final seed score (``LEARNED_SHARE``)."""
    c = contrib.get(node_id, 0.0)
    return c > 0.0 and c >= LEARNED_SHARE * seeds.get(node_id, 0.0)


def _confirmations(st: _State, teach_queries: list[str]) -> int:
    """Distinct teach queries of a node: connected components under ``TEACH_DEDUPE_SIM``.

    Components (not a greedy scan) so the count does not depend on the order of the queries.
    """
    n = len(teach_queries)
    if n <= 1:
        return n
    vecs = np.vstack([features(q, st.params.d) for q in teach_queries])
    close = (vecs @ vecs.T) >= TEACH_DEDUPE_SIM
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        for j in range(i + 1, n):
            if close[i, j]:
                parent[find(i)] = find(j)
    return len({find(i) for i in range(n)})


def _tokens(text: str) -> frozenset[str]:
    return frozenset(w for w in re.findall(r"\w+", text.casefold()) if len(w) >= MIN_TOKEN_CHARS)


def _specific_filter(st: _State) -> tuple[int, Callable[[Iterable[int]], dict[int, int]]]:
    """``(cut, df_of)``: the self cut over the taught nodes and a document-frequency lookup."""
    taught = st.read.teach_node_count()
    cut = max(SELF_MIN_NODES, math.ceil(SELF_FRACTION * taught))
    return cut, st.read.kc_node_counts


def _specific_kcs(st: _State, active: list[int]) -> set[int]:
    """Cells of ``active`` that are not "self" (fire for fewer than the cut of distinct nodes)."""
    cut, df_of = _specific_filter(st)
    df = df_of(active)
    return {kc for kc in active if df.get(kc, 0) < cut}


def _learned_facts(
    st: _State,
    query: str,
    active: list[int],
    node_ids: list[int],
    texts: dict[int, str],
    quorum: int = DEFAULT_QUORUM,
    min_status: str = DEFAULT_LEARNED_STATUS,
    legacy: bool = False,
) -> tuple[list[list[str]], dict[int, dict[str, Any]]]:
    """``([example_query, "learned_as", node_text], {node_id: {status, confirmations}})``.

    Per node the ``MAX_LEARNED_EXAMPLES`` stored examples most similar to ``query`` (cosine of the
    n-gram features, ties by recency) are kept when they reach ``MIN_EXAMPLE_SIMILARITY`` and share
    at least ``MIN_SPECIFIC_KCS`` non-self cells with the query (negative selection). A node with no
    qualifying example emits nothing; its ``status`` is ``confirmed`` at ``quorum`` distinct teach
    queries, else ``provisional`` (hidden when ``min_status="confirmed"``).
    """
    if not node_ids:
        return [], {}
    qvec = features(query, st.params.d)
    specific = _specific_kcs(st, active)
    q_rare: set[str] = set()
    if legacy:  # pre-v5 gate: a word of the query that few recent examples of other nodes use
        background = [_tokens(q) for q in st.read.recent_examples(node_ids, BACKGROUND_EXAMPLES)]
        cap = RARE_TOKEN_SHARE * len(background)
        q_rare = {t for t in _tokens(query) if sum(t in b for b in background) <= cap}
    teach = st.read.teach_for(node_ids)
    out: list[list[str]] = []
    details: dict[int, dict[str, Any]] = {}
    for node_id, rows in st.read.examples_for(node_ids).items():
        codes = {t["query"]: set(t["active"]) for t in teach.get(node_id, [])}
        scored = []
        for r in rows:
            sim = float(features(r["query"], st.params.d) @ qvec)
            if sim < MIN_EXAMPLE_SIMILARITY:
                continue
            if legacy:
                if q_rare & _tokens(r["query"]):
                    scored.append((sim, r))
                continue
            code = codes.get(r["query"]) or set(_active(st, r["query"]))
            own = _specific_kcs(st, sorted(code))  # what makes THIS example distinctive
            shared = len(specific & own)
            if shared >= MIN_SPECIFIC_KCS and shared >= MIN_COVERAGE * len(own):
                scored.append((sim, r))
        if not scored:
            continue
        confirmations = max(1, _confirmations(st, [t["query"] for t in teach.get(node_id, [])]))
        status = "confirmed" if confirmations >= quorum else "provisional"
        if min_status == "confirmed" and status != "confirmed":
            continue
        scored.sort(key=lambda t: (-t[0], -t[1]["ts"]))
        out.extend(
            [r["query"], LEARNED_REL, texts[node_id]] for _, r in scored[:MAX_LEARNED_EXAMPLES]
        )
        details[node_id] = {"status": status, "confirmations": confirmations}
    return out, details


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


def _usage_factors(st: _State, decay: float) -> tuple[dict[int, float], float] | None:
    """Per-node consolidate factors from the usage counters, or ``None`` when nothing was used."""
    usage = st.read.node_usage()
    if not usage:
        return None
    unused = decay**UNUSED_DECAY_EXPONENT
    used = 1.0 - (1.0 - decay) * (1.0 - USAGE_PROTECT)
    factors = {
        node: unused + min(1.0, u / USAGE_SATURATION) * (used - unused) for node, u in usage.items()
    }
    return factors, unused


def _replay_items(st: _State, rate: float) -> list[tuple[list[int], int, float]]:
    """``(kcs, node_id, delta)`` per stored example, re-applied at ``rate``."""
    teach = {
        (node, t["query"]): t["active"]
        for node, rows in st.read.teach_for({r["node_id"] for r in st.read.all_examples()}).items()
        for t in rows
    }
    items = []
    for row in st.read.all_examples():
        code = teach.get((row["node_id"], row["query"])) or _active(st, row["query"])
        delta = rate * ETA * min(float(row["weight"]), REPLAY_WEIGHT_CAP)
        items.append((sorted(set(code)), row["node_id"], delta))
    return items


def _consolidate(params: Params, ctx: Context) -> Payload:
    decay = params.get("decay", 0.9)
    if not _number_ok(decay) or not 0 < decay <= 1:
        return _error("decay must be a number in (0, 1]")
    prune = params.get("prune_below", 1e-3)
    if not _number_ok(prune) or not 0 <= prune <= MAX_PRUNE:
        return _error(f"prune_below must be a number in [0, {MAX_PRUNE:g}]")
    replay = params.get("replay", False)
    usage_weighting = params.get("usage_weighting", False)
    if not isinstance(replay, bool) or not isinstance(usage_weighting, bool):
        return _error("replay and usage_weighting must be booleans")
    rate = params.get("replay_rate", DEFAULT_REPLAY_RATE)
    if not _number_ok(rate) or not 0 < rate <= 1:
        return _error("replay_rate must be a number in (0, 1]")
    capture = params.get("capture", False)
    if not isinstance(capture, bool):
        return _error("capture must be a boolean")
    window = _int_param(params, "capture_window", DEFAULT_CAPTURE_WINDOW, 1, MAX_CAPTURE_WINDOW)
    confirmations = _int_param(
        params, "capture_confirmations", DEFAULT_CAPTURE_CONFIRMATIONS, 1, MAX_QUORUM
    )
    if isinstance(window, str) or isinstance(confirmations, str):
        return _error(window if isinstance(window, str) else str(confirmations))
    st = _get_state()
    stats: dict[str, int] | None = None
    confirmed: list[int] = []
    if capture:
        taught = {r["node_id"] for r in st.read.all_examples()}
        confirmed = sorted(
            n
            for n, rows in st.read.teach_for(taught).items()
            if _confirmations(st, [t["query"] for t in rows]) >= confirmations
        )
        stats = st.read.capture_stats(window, confirmed)
    try:
        backup = st.write.backup("pre-consolidate")  # lossy write: snapshot first, fail closed
    except BackupError as exc:
        return _error(f"backup before consolidate failed, nothing written: {exc}")
    items = _replay_items(st, float(rate)) if replay else None
    plan = _usage_factors(st, float(decay)) if usage_weighting else None
    decayed, pruned = st.write.decay_assoc(
        float(decay),
        float(prune),
        replay=items,
        node_factors=plan[0] if plan else None,
        default_factor=plan[1] if plan else None,
        usage_evaporation=(USAGE_EVAPORATION, USAGE_FLOOR) if usage_weighting else None,
        capture_window=window if capture else None,
        capture_nodes=confirmed,
    )
    out: Payload = {
        "status": "OK",
        "summary": f"decayed {decayed}, pruned {pruned}",
        "backup_slot": backup.get("slot"),
        "decayed": decayed,
        "pruned": pruned,
        "terse": {"decayed": decayed, "pruned": pruned},
    }
    if replay:
        out["replayed"] = len(items or [])
    if usage_weighting:
        out["usage_applied"] = plan is not None
    if stats is not None:
        out["captured"] = stats["captured"]
        out["captured_tagged"] = stats["tagged"]
        out["captured_confirmed"] = stats["confirmed"]
        out["decayable"] = stats["assoc"] - stats["captured"]
    return out


def _flyvec_train(params: Params, ctx: Context) -> Payload:
    """Learn per-token Kenyon-cell codes from the KB triples and encode every node text."""
    ints = {
        "epochs": _int_param(params, "epochs", 5, 1, flyvec.MAX_EPOCHS),
        "hidden": _int_param(params, "hidden", 256, 8, flyvec.MAX_HIDDEN),
        "active": _int_param(params, "active", 16, 1, flyvec.MAX_HIDDEN),
        "seed": _int_param(params, "seed", 1729, 0, 2**31 - 1),
    }
    for value in ints.values():
        if isinstance(value, str):
            return _error(value)
    n = {key: int(value) for key, value in ints.items()}
    if n["active"] > n["hidden"]:
        return _error("active must be <= hidden")
    st = _get_state()
    edges = st.read.training_sentences(FLYVEC_MAX_EDGES)
    if not edges:
        return _error("nothing to train on: ingest triples first")
    sentences = [flyvec.tokenize(f"{a} {r} {b}") for a, r, b in edges]
    codes, info = flyvec.train(sentences, **n)
    nodes = st.read.node_texts(MAX_NODES)
    vectors = {}
    for node_id, text in nodes:
        vec = flyvec.encode(flyvec.tokenize(text), codes, n["hidden"])
        if vec.any():
            vectors[node_id] = vec.tolist()
    info = {**info, "nodes_encoded": len(vectors), "nodes_total": len(nodes)}
    st.write.flyvec_replace(codes, vectors, info)
    return {
        "status": "OK",
        "summary": f"FlyVec: {len(codes)} token(s), {len(vectors)}/{len(nodes)} node(s) encoded",
        "model": info,
        "next_action": "Use memory.recall(encoder='flyvec'); nodes ingested later need a retrain.",
        "terse": {"tokens": len(codes), "nodes_encoded": len(vectors)},
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
            "quorum": {**int_opt, "min": 1, "max": MAX_QUORUM, "default": DEFAULT_QUORUM},
            "learned_min_status": {
                "type": "string",
                "required": False,
                "enum": list(LEARNED_STATUSES),
                "default": DEFAULT_LEARNED_STATUS,
            },
            "track_usage": {"type": "boolean", "required": False, "default": False},
            "inhibition": {
                "type": "string",
                "required": False,
                "enum": list(INHIBITIONS),
                "default": DEFAULT_INHIBITION,
            },
            "z": {
                "type": "number",
                "required": False,
                "min": 0,
                "max": MAX_Z,
                "default": DEFAULT_Z,
            },
            "encoder": {
                "type": "string",
                "required": False,
                "enum": list(ENCODERS),
                "default": DEFAULT_ENCODER,
            },
            "learned_legacy": {"type": "boolean", "required": False, "default": False},
            "radius": {"type": "integer", "required": False, "min": 0, "max": 256},
            "cursor": {"type": "string", "required": False},
        },
        description=(
            "Associative recall. mode='ppr' (default): seeds + learned boost -> "
            "Personalized PageRank over the graph, returns top-k nodes and supporting facts. "
            "lambda scales the learned boost on a [0, 1] range comparable to the cosine seeds. "
            "When a returned node ranks mainly because of what 'update' taught it, the facts "
            "include up to 2 [taught_query, 'learned_as', node] entries (the taught queries "
            "most similar to yours) and the result item carries learned=true: read it as "
            "'queries worded like taught_query refer to node'. "
            "A learned node carries learned_status: 'confirmed' when its teach queries are >= "
            "quorum (default 2) DISTINCT phrasings (near duplicates count once), else "
            "'provisional'; the boost is not gated by it. learned_min_status='confirmed' hides "
            "the provisional learned_as explanations (precision over recall: a single-teach "
            "alias is not explained). track_usage=true counts the edges around the returned "
            "nodes (pheromone) for consolidate(usage_weighting=true); off by default. "
            "mode='seeds': nearest nodes only (no learned boost, no learned_as). "
            "seeds= selects the seed backend, distinct from "
            "mode='seeds': seeds='cosine' (default): exact cosine over the "
            "dense vectors; seeds='flyhash' (experimental): FlyHash code overlap. "
            "Opt-in: inhibition='adaptive' (query-time APL inhibition, z), mode='sdm'+radius "
            "(Kanerva Hamming read, cursor pages), encoder='flyvec' (after flyvec_train), "
            "learned_legacy=true (pre-v5 learned_as gate)."
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
            "replay": {"type": "boolean", "required": False, "default": False},
            "replay_rate": {
                "type": "number",
                "required": False,
                "min": 0,
                "max": 1,
                "default": DEFAULT_REPLAY_RATE,
            },
            "usage_weighting": {"type": "boolean", "required": False, "default": False},
            "capture": {"type": "boolean", "required": False, "default": False},
            "capture_window": {
                **int_opt,
                "min": 1,
                "max": MAX_CAPTURE_WINDOW,
                "default": DEFAULT_CAPTURE_WINDOW,
            },
            "capture_confirmations": {
                **int_opt,
                "min": 1,
                "max": MAX_QUORUM,
                "default": DEFAULT_CAPTURE_CONFIRMATIONS,
            },
        },
        description=(
            "Decay learned associations (assoc *= decay) and prune |w| < prune_below; teach "
            "examples of nodes left without any association are dropped. 'forget' acts on edge "
            "sources only: examples follow assoc, not sources. replay=true first re-applies the "
            "stored teach examples at replay_rate (sleep replay) so taught associations survive "
            "decay; usage_weighting=true makes nodes used by recall(track_usage=true) decay less "
            "and unused ones more (usage counters evaporate; source facts are never touched). "
            "capture=true: tagged-recently (capture_window updates) or confirmed associations "
            "are exempt from decay/prune; reports captured counts."
        ),
        example={"decay": 0.9, "prune_below": 0.001, "replay": True},
    )
    registry.register(
        "memory",
        "flyvec_train",
        _flyvec_train,
        params_schema={
            "epochs": {**int_opt, "min": 1, "max": flyvec.MAX_EPOCHS, "default": 5},
            "hidden": {**int_opt, "min": 8, "max": flyvec.MAX_HIDDEN, "default": 256},
            "active": {**int_opt, "min": 1, "max": flyvec.MAX_HIDDEN, "default": 16},
            "seed": {**int_opt, "min": 0, "max": 2**31 - 1, "default": 1729},
        },
        description=(
            "FlyVec-lite: learn per-token codes from the KB triples (bounded, seeded); enables "
            "recall(encoder='flyvec'). No generalisation to unseen words; retrain after ingest."
        ),
        example={"epochs": 5},
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
