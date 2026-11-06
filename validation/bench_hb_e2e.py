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


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / RESULT_FILE).write_text(json.dumps(run(), indent=2))


if __name__ == "__main__":
    main()
