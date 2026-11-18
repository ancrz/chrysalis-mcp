"""Low-priority bio tools (substrate v6): does each mechanism help, and where does it not?

Same alias scenario as ``bench_hb_e2e`` (invented entities, user aliases taught with
``memory.update``), one fresh substrate per (arm, seed), operations through the real dispatcher.

* L1 synaptic tagging & capture: ``noise`` updates, then set A, then set B, then ``DECAY_CYCLES``
  strong consolidations. Arms: ``plain``; ``capture_tag`` (window = the A+B updates, confirmations
  off); ``capture_confirmed`` (A taught with two phrasings, tiny window: only confirmation protects
  A). Metrics: recall@k of A and B after the cycles, ``assoc_rows`` left (= what capture costs).
* L2 lateral inhibition: ``fixed`` vs ``adaptive`` (z in ``Z_GRID``) at QUERY time over the same
  retained substrate: recall@k of the taught aliases, plain (alias-free) entity questions and
  ``false_learned_as`` on untaught-alias probes.
* L3 sparse distributed memory: ``ppr`` vs ``sdm`` (radii in ``SDM_RADII``) on alias questions and
  plain questions; SDM has no learned boost by design.
* L4 FlyVec-lite: ``ngram`` vs ``flyvec`` encoder (modes ``seeds`` and ``ppr``) on plain questions
  and on the same questions with a typo in the entity name.
"""

import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from chrysalis_mcp.domains import memory
from chrysalis_mcp.domains.memory import DEFAULT_K
from validation import RESULTS_DIR
from validation.bench_hb_e2e import (
    BIO_SEEDS,
    DECAY_CYCLES,
    DECAY_PARAMS,
    Op,
    _BioRun,
    kb_ops,
    plain_question,
)
from validation.datasets import AliasDataset, AliasItem, alias_dataset

RESULT_FILE = "bio_low.json"
Z_GRID = (0.5, 1.0, 2.0)
SDM_RADII = (64, 96, 128)
L1_ARMS = ("plain", "capture_tag", "capture_confirmed")
FLYVEC_PARAMS = {"epochs": 5, "hidden": 256, "active": 16}


def _updates(items: tuple[AliasItem, ...], second: bool = False) -> list[Op]:
    ops: list[Op] = [("update", {"query": i.teach, "node": i.entity}) for i in items]
    if second:
        ops += [("update", {"query": i.teach2, "node": i.entity}) for i in items]
    return ops


def _noise(data: AliasDataset) -> list[Op]:
    return [("update", {"query": q, "node": n}) for q, n in data.interference_pairs]


def _typo(item: AliasItem) -> str:
    """The plain question with one letter dropped from the entity name's first word."""
    first = item.entity.split()[0]
    return plain_question(item).replace(
        item.entity, item.entity.replace(first, first[:2] + first[3:])
    )


def _hit_rate(run: _BioRun, queries: list[tuple[str, str]], k: int, **extra: Any) -> float:
    """Share of ``(query, expected_node)`` whose node is in the top-``k`` results."""
    hits = 0
    for query, node in queries:
        res = run.call("recall", query=query, k=k, **extra)
        hits += node in [r["text"] for r in res["results"]]
    return hits / len(queries)


def _alias_queries(items: tuple[AliasItem, ...]) -> list[tuple[str, str]]:
    return [(i.test, i.entity) for i in items]


def _plain_queries(items: tuple[AliasItem, ...]) -> list[tuple[str, str]]:
    return [(plain_question(i), i.entity) for i in items]


def _l1(arm: str, data: AliasDataset, k: int) -> dict[str, float]:
    window = len(data.set_a) + len(data.set_b)
    flags: dict[str, Any] = {}
    if arm == "capture_tag":
        flags = {"capture": True, "capture_window": window, "capture_confirmations": 16}
    elif arm == "capture_confirmed":
        flags = {"capture": True, "capture_window": 1, "capture_confirmations": 2}
    with tempfile.TemporaryDirectory() as tmp:
        run = _BioRun(str(Path(tmp) / "l1.db"))
        try:
            run.apply(kb_ops(data))
            run.apply(_noise(data))
            run.apply(_updates(data.set_a, second=arm == "capture_confirmed"))
            run.apply(_updates(data.set_b))
            rows_before = memory._get_state().read.counts()["assoc"]
            captured = 0
            for _ in range(DECAY_CYCLES):
                out = run.call("consolidate", **DECAY_PARAMS, **flags)
                assert out["status"] == "OK", out
                captured = out.get("captured", 0)
            rows_after = memory._get_state().read.counts()["assoc"]
            return {
                "recall_a": _hit_rate(run, _alias_queries(data.set_a), k, mode="ppr"),
                "recall_b": _hit_rate(run, _alias_queries(data.set_b), k, mode="ppr"),
                "assoc_rows_before": rows_before,
                "assoc_rows_after": rows_after,
                "captured_last_cycle": captured,
            }
        finally:
            memory.configure(None)


def _retained_run(tmp: str, data: AliasDataset) -> _BioRun:
    run = _BioRun(str(Path(tmp) / "retained.db"))
    run.apply(kb_ops(data))
    run.apply(_updates(data.set_a))
    run.apply(_updates(data.set_b))
    run.apply(_noise(data))
    return run


def _l2(data: AliasDataset, k: int) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    with tempfile.TemporaryDirectory() as tmp:
        run = _retained_run(tmp, data)
        try:
            arms: dict[str, dict[str, Any]] = {"fixed": {}}
            arms.update({f"adaptive_z{z}": {"inhibition": "adaptive", "z": z} for z in Z_GRID})
            for name, extra in arms.items():
                out[name] = {
                    "alias_recall": _hit_rate(run, _alias_queries(data.set_a), k, **extra),
                    "plain_recall": _hit_rate(run, _plain_queries(data.items), k, **extra),
                    "explained": run.explained(data.set_a, k, **extra),
                    "false_learned_as": run.false_rate(data.probes, k, **extra),
                }
        finally:
            memory.configure(None)
    return out


def _l3(data: AliasDataset, k: int) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    with tempfile.TemporaryDirectory() as tmp:
        run = _retained_run(tmp, data)
        try:
            arms: dict[str, dict[str, Any]] = {"ppr": {"mode": "ppr"}}
            arms.update({f"sdm_r{r}": {"mode": "sdm", "radius": r} for r in SDM_RADII})
            for name, extra in arms.items():
                alias_q, plain_q = _alias_queries(data.set_a), _plain_queries(data.items)
                start = time.perf_counter()
                alias = _hit_rate(run, alias_q, k, **extra)
                ms = (time.perf_counter() - start) * 1000.0
                out[name] = {
                    "alias_recall": alias,
                    "plain_recall": _hit_rate(run, plain_q, k, **extra),
                    "ms_per_alias_query": round(ms / len(alias_q), 3),
                }
        finally:
            memory.configure(None)
    return out


def _l4(data: AliasDataset, k: int) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    with tempfile.TemporaryDirectory() as tmp:
        run = _retained_run(tmp, data)
        try:
            trained = run.call("flyvec_train", **FLYVEC_PARAMS)
            assert trained["status"] == "OK", trained
            typo = [(_typo(i), i.entity) for i in data.items]
            arms = {
                "ngram_seeds": {"mode": "seeds"},
                "flyvec_seeds": {"mode": "seeds", "encoder": "flyvec"},
                "ngram_ppr": {"mode": "ppr"},
                "flyvec_ppr": {"mode": "ppr", "encoder": "flyvec"},
            }
            for name, extra in arms.items():
                out[name] = {
                    "plain_recall": _hit_rate(run, _plain_queries(data.items), k, **extra),
                    "typo_recall": _hit_rate(run, typo, k, **extra),
                    "alias_recall": _hit_rate(run, _alias_queries(data.set_a), k, **extra),
                }
            out["model"] = {
                key: trained["model"][key] for key in ("vocab", "samples_used", "nodes_encoded")
            }
        finally:
            memory.configure(None)
    return out


def _mean(rows: list[dict[str, Any]]) -> dict[str, Any]:
    keys = rows[0].keys()
    return {
        key: round(sum(r[key] for r in rows) / len(rows), 4)
        if not isinstance(rows[0][key], dict)
        else _mean([r[key] for r in rows])
        for key in keys
    }


def run(
    seeds: tuple[int, ...] = BIO_SEEDS,
    k: int = DEFAULT_K,
    n_entities: int | None = None,
    n_interference: int | None = None,
    only: tuple[str, ...] = ("l1", "l2", "l3", "l4"),
) -> dict[str, Any]:
    opts = {
        key: v
        for key, v in (("n_entities", n_entities), ("n_interference", n_interference))
        if v is not None
    }
    datasets = [alias_dataset(seed, **opts) for seed in seeds]
    data: dict[str, Any] = {}
    if "l1" in only:
        data["l1_capture"] = {a: _mean([_l1(a, d, k) for d in datasets]) for a in L1_ARMS}
    for key, fn in (("l2_inhibition", _l2), ("l3_sdm", _l3), ("l4_flyvec", _l4)):
        if key[:2] in only:
            data[key] = _mean([fn(d, k) for d in datasets])
    first = datasets[0]
    return {
        "hypothesis": "bio-low",
        "config": {
            "seeds": list(seeds),
            "k": k,
            "n_set_a": len(first.set_a),
            "n_set_b": len(first.set_b),
            "n_noise": len(first.interference_pairs),
            "cycles": DECAY_CYCLES,
            "decay": DECAY_PARAMS,
            "flyvec": FLYVEC_PARAMS,
        },
        "data": data,
    }


def main(argv: list[str] | None = None) -> None:
    """``python -m validation.bench_bio_low [--out FILE]``."""
    args = list(sys.argv[1:] if argv is None else argv)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = Path(args[args.index("--out") + 1]) if "--out" in args else RESULTS_DIR / RESULT_FILE
    out.write_text(json.dumps(run(), indent=2))


if __name__ == "__main__":
    main()
