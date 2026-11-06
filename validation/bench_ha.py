"""H-a (retrieval half): PPR recall vs FlyHash seeds vs a 2-round iterative baseline.

recall@k = share of questions whose gold answer node is among the top-k results. Cost = mean
latency, response bytes and tool calls per question. Arms:
  seeds      one plain FlyHash-KNN call
  iterative  two seeds calls: round 2 re-queries with the question plus the top ``EXPAND_TOP``
             round-1 node texts (what an LLM would do to hop without a graph); results are
             round 1 then new round 2 hits
  ppr        one personalized-PageRank call
The end-to-end LLM half lives in ``eval_llm``.
"""

import json
import tempfile
import time
from pathlib import Path
from typing import Any

from chrysalis_mcp.core.session import SessionCache
from chrysalis_mcp.domains import build_registry, memory
from chrysalis_mcp.server import handle_call
from validation import RESULTS_DIR
from validation.datasets import QA_ITEMS

KS = (1, 5)
EXPAND_TOP = 2
RESULT_FILE = "ha.json"


def load_facts(call) -> None:
    triples = [{"s": s, "r": r, "o": o} for item in QA_ITEMS for s, r, o in item.triples]
    assert json.loads(call("memory", "ingest", {"triples": triples}))["status"] == "OK"


def _recall(call, query: str, mode: str) -> tuple[list[str], int]:
    text = call("memory", "recall", {"query": query, "k": max(KS), "mode": mode})
    return [r["text"] for r in json.loads(text)["results"]], len(text.encode("utf-8"))


def _ask(call, arm: str, question: str) -> tuple[list[str], int]:
    if arm != "iterative":
        return _recall(call, question, arm)
    first, b1 = _recall(call, question, "seeds")
    second, b2 = _recall(call, " ".join([question, *first[:EXPAND_TOP]]), "seeds")
    return first + [t for t in second if t not in first], b1 + b2


def run() -> dict[str, Any]:
    registry, cache = build_registry(), SessionCache()

    def call(domain: str, action: str, params: dict[str, Any]) -> str:
        return handle_call(registry, cache, domain, action, params, "summary")

    out: dict[str, Any] = {}
    with tempfile.TemporaryDirectory() as tmp:
        memory.configure(str(Path(tmp) / "ha.db"))
        try:
            load_facts(call)
            for arm in ("seeds", "iterative", "ppr"):
                hits = dict.fromkeys(KS, 0)
                total_ms = total_bytes = 0.0
                for item in QA_ITEMS:
                    t0 = time.perf_counter()
                    ranked, nbytes = _ask(call, arm, item.question)
                    total_ms += 1000 * (time.perf_counter() - t0)
                    total_bytes += nbytes
                    for k in KS:
                        hits[k] += item.answer in ranked[:k]
                n = len(QA_ITEMS)
                out[arm] = {
                    **{f"recall@{k}": round(hits[k] / n, 4) for k in KS},
                    "query_ms": round(total_ms / n, 3),
                    "response_bytes": round(total_bytes / n, 1),
                    "tool_calls": 2 if arm == "iterative" else 1,
                }
        finally:
            memory.configure(None)
    return {"hypothesis": "H-a", "config": {"n_items": len(QA_ITEMS), "ks": list(KS)}, "data": out}


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / RESULT_FILE).write_text(json.dumps(run(), indent=2))


if __name__ == "__main__":
    main()
