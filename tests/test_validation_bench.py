"""Full benchmark runs (``-m bench``): assert measurement integrity, write results + RESULTS.md."""

import json

import pytest

from validation import (
    RESULTS_DIR,
    bench_ha,
    bench_ha_prime,
    bench_hb,
    bench_hb_e2e,
    bench_hmcp,
    report,
)

pytestmark = pytest.mark.bench


def _write(name: str, result: dict) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / name).write_text(json.dumps(result, indent=2))


def test_ha_prime_full():
    out = bench_ha_prime.run()
    assert set(out["data"]) == {"1000", "10000"}
    for row in out["data"].values():
        assert set(row["fly"]) == {
            f"k{k}_c{c}" for k in bench_ha_prime.SWEEP_K for c in bench_ha_prime.SWEEP_C
        }
        assert all(0.0 <= v["precision"] <= 1.0 for v in [row["cosine"], *row["fly"].values()])
    _write(bench_ha_prime.RESULT_FILE, out)


def test_ha_retrieval_full():
    out = bench_ha.run()
    assert set(out["data"]) == {"seeds", "iterative", "ppr"}
    _write(bench_ha.RESULT_FILE, out)


def test_hb_full():
    out = bench_hb.run()
    assert set(out["data"]) == set(bench_hb.ARMS)
    assert set(out["data"]["sparse"]) == {"0", "10", "100", "1000"}
    assert all(0.0 <= v <= 1.0 for arm in out["data"].values() for v in arm.values())
    _write(bench_hb.RESULT_FILE, out)


def test_hb_e2e_full():
    out = bench_hb_e2e.run()
    assert set(out["data"]) == set(bench_hb_e2e.BENCH_ARMS)
    assert out["config"]["n_interference"] == 300 and len(out["config"]["seeds"]) == 5
    assert all(
        0.0 <= a["taught"] <= 1.0 and 0.0 <= a["retained"] <= 1.0 for a in out["data"].values()
    )
    _write(bench_hb_e2e.RESULT_FILE, out)


def test_hmcp_full():
    out = bench_hmcp.run()
    assert out["data"]["batch"]["tool_calls"] == 1
    _write(bench_hmcp.RESULT_FILE, out)


def test_report_is_written_from_the_results():
    path = report.write()
    text = path.read_text()
    for h in ("H-a'", "H-b", "H-b-e2e", "H-mcp", "H-a (retrieval)"):
        assert h in text
