"""H-b end to end, retrieval level (T19): does ``memory.update`` teach the system a user alias?

Scenario: an invented entity (``Kestrel Station``) has numeric facts in the KB; the user calls it by
an alias (``the Blue Hive``) that appears nowhere in the KB. Arms (a fresh substrate each):
  no_learn                KB only
  fly_update              ``memory.update(teach_query, node=entity)`` per taught item (the product)
  alias_ingest            the classic RAG way: ingest ``alias alias_of entity`` (STRONG baseline)
  fly_update_consolidate  fly_update plus one ``consolidate`` before the final test (informational)
  alias_ingest_clean      alias_ingest WITHOUT the interference updates (informational): separates
                          the alias mechanism from the effect of the interference updates
Metric: recall@k (k = DEFAULT_K, ppr mode) of the entity node for the set-A ``test`` questions, at
two times: right after teaching set A ("taught") and after teaching set B plus N_INTERFERENCE
unrelated updates ("retained"). The same operation lists are replayed by ``eval_llm`` (group
``hbe2e``) through the stdio server, so both levels measure the same prepared substrate.
"""

import json
import random
import sys
import tempfile
from pathlib import Path
from typing import Any

from chrysalis_mcp.core.session import SessionCache
from chrysalis_mcp.domains import build_registry, memory
from chrysalis_mcp.domains.memory import DEFAULT_K
from chrysalis_mcp.server import handle_call
from validation import RESULTS_DIR
from validation.datasets import ALIAS_SEED, AliasDataset, AliasItem, alias_dataset

ARMS = ("no_learn", "fly_update", "alias_ingest")
EXTRA_ARMS = ("alias_ingest_clean", "fly_update_consolidate")  # informational, never in a verdict
BENCH_ARMS = (*ARMS, *EXTRA_ARMS)
SEEDS = tuple(ALIAS_SEED + i for i in range(5))
INGEST_CHUNK = 400  # below memory.MAX_TRIPLES
RESULT_FILE = "hb_e2e.json"

Op = tuple[str, dict[str, Any]]  # (memory action, params)


def _ingest_ops(triples: list[tuple[str, str, str]]) -> list[Op]:
    rows = [{"s": s, "r": r, "o": o} for s, r, o in triples]
    return [
        ("ingest", {"triples": rows[i : i + INGEST_CHUNK]})
        for i in range(0, len(rows), INGEST_CHUNK)
    ]


def kb_ops(data: AliasDataset) -> list[Op]:
    """The knowledge base every arm shares (entity facts + unrelated filler nodes)."""
    return _ingest_ops(data.kb_triples)


def teach_ops(arm: str, items: tuple[AliasItem, ...]) -> list[Op]:
    """How an arm absorbs the user's alias corrections for ``items``."""
    if arm.startswith("fly_update"):
        return [("update", {"query": i.teach, "node": i.entity}) for i in items]
    if arm.startswith("alias_ingest"):
        return _ingest_ops([(i.alias, "alias_of", i.entity) for i in items])
    if arm == "no_learn":
        return []
    raise ValueError(f"unknown arm {arm!r}")


def retention_ops(arm: str, data: AliasDataset) -> list[Op]:
    """After set A: teach set B, then the unrelated interference updates (same for every arm)."""
    ops = teach_ops(arm, data.set_b)
    if arm != "alias_ingest_clean":
        ops += [("update", {"query": q, "node": n}) for q, n in data.interference_pairs]
    if arm == "fly_update_consolidate":
        ops.append(("consolidate", {}))
    return ops


def prepare_ops(arm: str, data: AliasDataset) -> list[Op]:
    """The full harness-side preparation: KB, teach A, teach B, interference (final state)."""
    return [*kb_ops(data), *teach_ops(arm, data.set_a), *retention_ops(arm, data)]


class _Run:
    """One fresh substrate; operations replayed in-process through the real dispatcher."""

    def __init__(self, path: str) -> None:
        memory.configure(path)
        self.registry, self.cache = build_registry(), SessionCache()

    def call(self, action: str, **params: Any) -> dict[str, Any]:
        return json.loads(handle_call(self.registry, self.cache, "memory", action, params))

    def apply(self, ops: list[Op]) -> None:
        for action, params in ops:
            out = self.call(action, **params)
            assert out["status"] == "OK", (action, out)

    def explained_rate(self, items: tuple[AliasItem, ...], k: int) -> float:
        """Share of ``test`` questions whose recall has a ``learned_as`` fact quoting the alias."""
        hits = 0
        for item in items:
            facts = self.call("recall", query=item.test, k=k, mode="ppr")["facts"]
            hits += any(f[1] == "learned_as" and item.alias in f[0] for f in facts)
        return hits / len(items)

    def unrelated_learned_rate(self, items: tuple[AliasItem, ...], k: int) -> float:
        """Share of plain (alias-free) questions with any ``learned_as`` fact (should be ~0)."""
        hits = 0
        for item in items:
            facts = self.call("recall", query=plain_question(item), k=k, mode="ppr")["facts"]
            hits += any(f[1] == "learned_as" for f in facts)
        return hits / len(items)

    def recall_rates(self, items: tuple[AliasItem, ...], k: int) -> tuple[float, float]:
        """(recall@k, recall@1) of the entity node over the ``test`` questions."""
        at_k = at_1 = 0
        for item in items:
            res = self.call("recall", query=item.test, k=k, mode="ppr")["results"]
            texts = [r["text"] for r in res]
            at_k += item.entity in texts
            at_1 += bool(texts) and texts[0] == item.entity
        return at_k / len(items), at_1 / len(items)


def plain_question(item: AliasItem) -> str:
    """A question about the entity by its real name: no alias, nothing was taught for it."""
    rel = item.triples[0][1].replace("has_", "").replace("_", " ")
    return f"What is the {rel} of {item.entity}?"


def _run_arm(arm: str, data: AliasDataset, k: int) -> dict[str, float]:
    with tempfile.TemporaryDirectory() as tmp:
        run = _Run(str(Path(tmp) / "hb_e2e.db"))
        try:
            run.apply(kb_ops(data))
            run.apply(teach_ops(arm, data.set_a))
            taught, taught_1 = run.recall_rates(data.set_a, k)
            explained = run.explained_rate(data.set_a, k)
            run.apply(retention_ops(arm, data))
            retained, retained_1 = run.recall_rates(data.set_a, k)
            explained_retained = run.explained_rate(data.set_a, k)
            # Only meaningful where nothing was taught: interference updates alone must not spray
            # learned_as onto plain questions (in the other arms those entities ARE taught).
            unrelated = run.unrelated_learned_rate(data.items, k) if arm == "no_learn" else None
        finally:
            memory.configure(None)
    out: dict[str, float] = {
        "taught": taught,
        "retained": retained,
        "taught@1": taught_1,
        "retained@1": retained_1,
        "explained": explained,
        "explained_retained": explained_retained,
    }
    if unrelated is not None:
        out["unrelated_learned_as"] = unrelated
    return out


def run(
    seeds: tuple[int, ...] = SEEDS,
    arms: tuple[str, ...] = BENCH_ARMS,
    k: int = DEFAULT_K,
    n_entities: int | None = None,
    n_interference: int | None = None,
) -> dict[str, Any]:
    opts = {
        key: v
        for key, v in (("n_entities", n_entities), ("n_interference", n_interference))
        if v is not None
    }
    datasets = [alias_dataset(seed, **opts) for seed in seeds]
    first = datasets[0]
    data: dict[str, Any] = {}
    for arm in arms:
        rows = [_run_arm(arm, d, k) for d in datasets]
        mean = {key: round(sum(r[key] for r in rows) / len(rows), 4) for key in rows[0]}
        mean["retention"] = round(mean["retained"] / mean["taught"], 4) if mean["taught"] else None
        data[arm] = {**mean, "per_seed": [[r["taught"], r["retained"]] for r in rows]}
    return {
        "hypothesis": "H-b-e2e",
        "config": {
            "seeds": list(seeds),
            "k": k,
            "n_set_a": len(first.set_a),
            "n_set_b": len(first.set_b),
            "n_interference": len(first.interference_pairs),
        },
        "data": data,
    }


# --- bio layer (substrate v5): negative selection, quorum, replay, usage ----------
# Arms (each on a fresh substrate, ``fly_update`` teaching as the product does it):
#   single       one teach phrasing per alias (default recall)
#   x2           two differently worded teach phrasings per alias (a quorum of 2)
#   decay        after interference, DECAY_CYCLES strong consolidate cycles (plain)
#   decay_replay the same cycles with ``replay=True``
#   usage_off    half of set A is RECALLED (used) before the cycles; plain consolidate
#   usage_on     the same, consolidate(usage_weighting=True)
# Variants: ``standard`` (unique colour + noun aliases) and ``common`` (pairs of ordinary words
# reused across entities, "the big apple"). ``false_learned_as`` = share of probe questions (an
# alias that was NEVER taught, built from taught words) that get any ``learned_as`` fact.
BIO_ARMS = ("single", "x2", "decay", "decay_replay", "usage_off", "usage_on")
BIO_VARIANTS = ("standard", "common")
BIO_SEEDS = tuple(ALIAS_SEED + i for i in range(5))
DECAY_CYCLES = 4
DECAY_PARAMS = {"decay": 0.5, "prune_below": 0.1}
USAGE_RECALLS = 3
ORDER_PERMUTATIONS = 3
BIO_RESULT_FILE = "hb_e2e_bio.json"


def _teach_pairs(arm: str, items: tuple[AliasItem, ...]) -> list[Op]:
    ops = [("update", {"query": i.teach, "node": i.entity}) for i in items]
    if arm == "x2":
        ops += [("update", {"query": i.teach2, "node": i.entity}) for i in items]
    return ops


class _BioRun(_Run):
    """``_Run`` plus calls that probe optional (v5) parameters without breaking on older code."""

    def supports(self, action: str, param: str) -> bool:
        spec = self.registry.schema()["domains"]["memory"][action]
        return param in spec["params"]

    def has_learned_as(self, query: str, k: int, **extra: Any) -> list[list[str]]:
        facts = self.call("recall", query=query, k=k, mode="ppr", **extra)["facts"]
        return [f for f in facts if f[1] == "learned_as"]

    def explained(self, items: tuple[AliasItem, ...], k: int, **extra: Any) -> float:
        hits = sum(
            any(i.alias in f[0] for f in self.has_learned_as(i.test, k, **extra)) for i in items
        )
        return hits / len(items)

    def false_rate(self, probes: tuple[tuple[str, str], ...], k: int, **extra: Any) -> float:
        return sum(bool(self.has_learned_as(q, k, **extra)) for _, q in probes) / len(probes)


def _bio_measure(run: _BioRun, data: AliasDataset, k: int, suffix: str = "") -> dict[str, float]:
    out = {
        f"explained{suffix}": run.explained(data.set_a, k),
        f"false_learned_as{suffix}": run.false_rate(data.probes, k),
    }
    if run.supports("recall", "learned_min_status"):
        strict = {"learned_min_status": "confirmed"}
        out[f"explained_confirmed{suffix}"] = run.explained(data.set_a, k, **strict)
        out[f"false_confirmed{suffix}"] = run.false_rate(data.probes, k, **strict)
    return out


def _bio_arm(arm: str, data: AliasDataset, k: int) -> dict[str, float] | None:
    with tempfile.TemporaryDirectory() as tmp:
        run = _BioRun(str(Path(tmp) / "bio.db"))
        try:
            if arm == "decay_replay" and not run.supports("consolidate", "replay"):
                return None
            if arm == "usage_on" and not run.supports("consolidate", "usage_weighting"):
                return None
            run.apply(kb_ops(data))
            run.apply(_teach_pairs(arm, data.set_a))
            out = _bio_measure(run, data, k)
            out["taught"], _ = run.recall_rates(data.set_a, k)
            run.apply(_teach_pairs(arm, data.set_b))
            run.apply([("update", {"query": q, "node": n}) for q, n in data.interference_pairs])
            out["retained"], _ = run.recall_rates(data.set_a, k)
            out.update(_bio_measure(run, data, k, "_retained"))
            if arm.startswith(("decay", "usage")):
                used = data.set_a[: len(data.set_a) // 2]
                unused = data.set_a[len(data.set_a) // 2 :]
                track = {"track_usage": True} if run.supports("recall", "track_usage") else {}
                if arm.startswith("usage"):
                    for _ in range(USAGE_RECALLS):
                        for item in used:
                            run.call("recall", query=item.test, k=k, mode="ppr", **track)
                flags: dict[str, Any] = {}
                if arm == "decay_replay":
                    flags["replay"] = True
                if arm == "usage_on":
                    flags["usage_weighting"] = True
                for _ in range(DECAY_CYCLES):
                    run.apply([("consolidate", {**DECAY_PARAMS, **flags})])
                out["after_cycles"], _ = run.recall_rates(data.set_a, k)
                out["after_cycles_used"], _ = run.recall_rates(used, k)
                out["after_cycles_unused"], _ = run.recall_rates(unused, k)
                out["explained_after_cycles"] = run.explained(data.set_a, k)
        finally:
            memory.configure(None)
    return out


def order_agreement(
    data: AliasDataset, k: int = DEFAULT_K, n_perm: int = ORDER_PERMUTATIONS
) -> float:
    """Share of queries (set-A tests + probes) with the SAME ``learned_as`` facts under every one of
    ``n_perm`` update orders (permutation 0 = natural order). 1.0 = order independent (D2)."""
    updates = [
        *_teach_pairs("single", data.set_a),
        *_teach_pairs("single", data.set_b),
        *[("update", {"query": q, "node": n}) for q, n in data.interference_pairs],
    ]
    queries = [i.test for i in data.set_a] + [q for _, q in data.probes]
    decisions: list[list[frozenset[tuple[str, str]]]] = []
    for p in range(n_perm):
        order = list(updates)
        if p:
            random.Random(p).shuffle(order)
        with tempfile.TemporaryDirectory() as tmp:
            run = _BioRun(str(Path(tmp) / "order.db"))
            try:
                run.apply(kb_ops(data))
                run.apply(order)
                decisions.append(
                    [frozenset((f[0], f[2]) for f in run.has_learned_as(q, k)) for q in queries]
                )
            finally:
                memory.configure(None)
    same = sum(len({d[i] for d in decisions}) == 1 for i in range(len(queries)))
    return same / len(queries)


def run_bio(
    seeds: tuple[int, ...] = BIO_SEEDS,
    arms: tuple[str, ...] = BIO_ARMS,
    variants: tuple[str, ...] = BIO_VARIANTS,
    k: int = DEFAULT_K,
    n_entities: int | None = None,
    n_interference: int | None = None,
    order: bool = True,
) -> dict[str, Any]:
    """Per variant and arm: mean over seeds of the D1/D2/D3 metrics (``None`` = arm unsupported)."""
    opts = {
        key: v
        for key, v in (("n_entities", n_entities), ("n_interference", n_interference))
        if v is not None
    }
    data: dict[str, Any] = {}
    for variant in variants:
        datasets = [alias_dataset(seed, common_words=variant == "common", **opts) for seed in seeds]
        per_arm: dict[str, Any] = {}
        for arm in arms:
            rows = [r for d in datasets if (r := _bio_arm(arm, d, k)) is not None]
            if not rows:
                per_arm[arm] = None
                continue
            per_arm[arm] = {key: round(sum(r[key] for r in rows) / len(rows), 4) for key in rows[0]}
        if order:
            agree = [order_agreement(d, k) for d in datasets]
            per_arm["order_agreement"] = round(sum(agree) / len(agree), 4)
        data[variant] = per_arm
    return {
        "hypothesis": "H-b-bio",
        "config": {"seeds": list(seeds), "k": k, "cycles": DECAY_CYCLES, "decay": DECAY_PARAMS},
        "data": data,
    }


def main(argv: list[str] | None = None) -> None:
    """``python -m validation.bench_hb_e2e`` (classic bench) or ``... bio [--out FILE]``."""
    args = list(sys.argv[1:] if argv is None else argv)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if args[:1] == ["bio"]:
        out = (
            Path(args[args.index("--out") + 1])
            if "--out" in args
            else RESULTS_DIR / BIO_RESULT_FILE
        )
        out.write_text(json.dumps(run_bio(), indent=2))
        return
    (RESULTS_DIR / RESULT_FILE).write_text(json.dumps(run(), indent=2))


if __name__ == "__main__":
    main()
