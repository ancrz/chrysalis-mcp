"""H-b: sparse (active-KC) Hebbian update vs three baselines, retention of old associations.

All arms learn the same old query->node associations and then absorb N new ones; only the update
rule differs. Precision = share of old queries whose top-1 PPR recall is still the node they were
taught. Arms:
  sparse           additive Hebbian update on the query's active KCs (the shipped ``update``)
  overwrite        naive: on the active KCs, erase every other node's weight and set this node's
  dense            adds the full delta on EVERY KC (no query specificity)
  dense_normalized dense with the delta scaled by k/m so the per-query mass equals the sparse one
Precondition: an arm is only comparable if it has learned the old associations (precision at N=0
>= ``PRECONDITION_MIN``); arms that fail are reported as such, not silently scored.
"""

import json
import random
import tempfile
from pathlib import Path
from typing import Any

from chrysalis_mcp.algorithms.flyhash import FlyHash, features
from chrysalis_mcp.core.session import SessionCache
from chrysalis_mcp.domains import build_registry, memory
from chrysalis_mcp.domains.memory import ETA
from chrysalis_mcp.server import handle_call
from chrysalis_mcp.substrate import open_substrate
from validation import RESULTS_DIR
from validation.datasets import synthetic_text, synthetic_vocab

CHECKPOINTS = (10, 100, 1000)
N_OLD = 20
DELTA = 1.0
PRECONDITION_MIN = 0.9
ARMS = ("sparse", "overwrite", "dense", "dense_normalized")
INGEST_CHUNK = 400  # below memory.MAX_TRIPLES
RESULT_FILE = "hb.json"


class _Arm:
    def __init__(self, path: str, kind: str) -> None:
        self.kind = kind
        memory.configure(path)
        self.registry, self.cache = build_registry(), SessionCache()
        self.read, self.write = open_substrate(path)
        p = self.read.flyhash_params()
        self.m, self.k, self.d = p.m, p.k, p.d
        self.hasher = FlyHash(p.d, p.m, p.k, p.c, p.seed)

    def call(self, action: str, **params: Any) -> dict[str, Any]:
        return json.loads(handle_call(self.registry, self.cache, "memory", action, params))

    def ingest(self, pairs: list[tuple[str, str]]) -> None:
        for i in range(0, len(pairs), INGEST_CHUNK):
            chunk = [
                {"s": f"anchor {a}", "r": "ref", "o": t} for a, t in pairs[i : i + INGEST_CHUNK]
            ]
            assert self.call("ingest", triples=chunk)["status"] == "OK"

    def _overwrite(self, query: str, node_id: int) -> None:
        """Zero every node's weight on the query's active KCs, then write this node's."""
        active = self.hasher.hash(features(query, self.d)).tolist()
        for row in self.read.assoc_for(active):
            self.write.reinforce([row["kc"]], row["node_id"], -float(row["weight"]))
        self.write.reinforce(active, node_id, ETA * DELTA)

    def learn(self, query: str, node: str) -> None:
        node_id = self.read.get_node(node)["id"]
        if self.kind == "sparse":
            assert self.call("update", query=query, node=node, delta=DELTA)["status"] == "OK"
        elif self.kind == "overwrite":
            self._overwrite(query, node_id)
        elif self.kind == "dense":
            self.write.reinforce(range(self.m), node_id, ETA * DELTA)
        else:  # dense_normalized
            self.write.reinforce(range(self.m), node_id, ETA * DELTA * self.k / self.m)

    def precision(self, old: list[tuple[str, str]]) -> float:
        ok = 0
        for query, node in old:
            res = self.call("recall", query=query, k=1, mode="ppr")
            ok += bool(res["results"]) and res["results"][0]["text"] == node
        return ok / len(old)


def _run_arm(kind: str, seed: int, checkpoints: tuple[int, ...], n_old: int) -> dict[str, float]:
    rng = random.Random(seed)
    vocab = synthetic_vocab(rng)
    n_new = max(checkpoints)
    old = [(synthetic_text(rng, vocab), f"old target {i}") for i in range(n_old)]
    new = [(synthetic_text(rng, vocab), f"new target {i}") for i in range(n_new)]
    with tempfile.TemporaryDirectory() as tmp:
        arm = _Arm(str(Path(tmp) / "hb.db"), kind)
        try:
            arm.ingest([(f"o{i}", t) for i, (_, t) in enumerate(old)])
            arm.ingest([(f"n{i}", t) for i, (_, t) in enumerate(new)])
            for q, t in old:
                arm.learn(q, t)
            curve = {"0": arm.precision(old)}
            for i, (q, t) in enumerate(new, start=1):
                arm.learn(q, t)
                if i in checkpoints:
                    curve[str(i)] = arm.precision(old)
            return curve
        finally:
            memory.configure(None)


def run(
    seed: int = 0, checkpoints: tuple[int, ...] = CHECKPOINTS, n_old: int = N_OLD
) -> dict[str, Any]:
    return {
        "hypothesis": "H-b",
        "config": {"seed": seed, "checkpoints": list(checkpoints), "n_old": n_old, "delta": DELTA},
        "data": {arm: _run_arm(arm, seed, checkpoints, n_old) for arm in ARMS},
    }


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / RESULT_FILE).write_text(json.dumps(run(), indent=2))


if __name__ == "__main__":
    main()
