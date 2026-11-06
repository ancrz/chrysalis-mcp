"""Harness integrity: datasets, scoring, import graph, benchmark measurement integrity, report."""

import ast
import json
from pathlib import Path

import numpy as np
import pytest

from chrysalis_mcp.core.session import SessionCache
from chrysalis_mcp.domains import build_registry
from chrysalis_mcp.server import handle_call
from tests.fake_llm import FakeLlm, reply_text
from validation import bench_ha, bench_ha_prime, bench_hb, bench_hmcp, eval_llm, report
from validation.datasets import PUZZLES, QA_ITEMS, score_puzzle, score_qa
from validation.llm_client import LlmClient

SRC = Path(__file__).resolve().parent.parent / "src" / "chrysalis_mcp"


def test_chrysalis_never_imports_the_validation_harness():
    for path in SRC.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            names = (
                [a.name for a in node.names]
                if isinstance(node, ast.Import)
                else [node.module or ""]
                if isinstance(node, ast.ImportFrom)
                else []
            )
            assert not any(n.split(".")[0] == "validation" for n in names), path


def test_the_datasets_have_the_declared_shape():
    assert len(QA_ITEMS) == 20 and len(PUZZLES) == 20
    entities = [t[0] for i in QA_ITEMS for t in i.triples]
    assert len({i.answer for i in QA_ITEMS}) == 20
    for item in QA_ITEMS:  # two hops chained through a shared middle entity
        assert item.triples[0][2] == item.triples[1][0] and item.triples[1][2] == item.answer
    assert len(set(entities)) == len(entities)


def test_every_puzzle_has_exactly_the_declared_unique_solution():
    registry, cache = build_registry(), SessionCache()
    for puzzle in PUZZLES:
        out = json.loads(
            handle_call(
                registry,
                cache,
                "reason",
                "solve",
                {"smt2": puzzle.smt2, "max_models": 2},
                "summary",
            )
        )
        assert out["result"] == "sat", puzzle.text
        assert len(out["models"]) == 1, f"not unique: {puzzle.text}"
        assert {k: int(v) for k, v in out["models"][0].items()} == puzzle.answer, puzzle.text


@pytest.mark.parametrize(
    "answer, gold, ok",
    [
        ("Tallin Reach", "Tallin Reach", True),
        ("  the tallin reach. ", "Tallin Reach", True),
        ("It is Tallin Reach", "Tallin Reach", False),
        (None, "x", False),
        ("", "x", False),
    ],
)
def test_qa_scoring_is_exact_after_folding(answer, gold, ok):
    assert score_qa(answer, gold) is ok


@pytest.mark.parametrize(
    "answer, ok",
    [
        ('{"x": 6, "y": 4}', True),
        ('so {"x": 1} no, final: {"x": "6", "y": 4}', True),
        ('{"x": 6, "y": 5}', False),
        ('{"x": "six"}', False),
        ("no json", False),
        (None, False),
    ],
)
def test_puzzle_scoring_reads_the_last_json_object(answer, ok):
    assert score_puzzle(answer, {"x": 6, "y": 4}) is ok


def _strip_ms(row):
    return {
        "cosine": {k: v for k, v in row["cosine"].items() if k != "query_ms"},
        "fly": {s: {k: v for k, v in r.items() if k != "query_ms"} for s, r in row["fly"].items()},
    }


_HAP = {"sizes": (200,), "n_queries": 10, "seed": 3, "sweep_k": (64, 256), "sweep_c": (12,)}


def test_ha_prime_measurements_are_in_range_and_reproducible():
    a, b = bench_ha_prime.run(**_HAP), bench_ha_prime.run(**_HAP)
    row = a["data"]["200"]
    assert set(row["fly"]) == {"k64_c12", "k256_c12"}
    assert 0.0 <= row["cosine"]["precision"] <= 1.0
    assert (row["fly"]["k64_c12"]["bytes_per_node"], row["cosine"]["bytes_per_node"]) == (
        1024,
        4096,
    )
    for r in row["fly"].values():
        assert 0.0 <= r["precision"] <= 1.0 and 0.0 <= r["agreement_with_cosine"] <= 1.0
        assert r["query_ms"] > 0
    assert _strip_ms(row) == _strip_ms(b["data"]["200"])


def test_ha_prime_ranking_is_independent_of_corpus_order():
    fwd = bench_ha_prime.run(**_HAP)
    rev = bench_ha_prime.run(**_HAP, reverse_corpus=True)
    assert _strip_ms(fwd["data"]["200"]) == _strip_ms(rev["data"]["200"])


def test_ha_prime_ties_are_broken_by_item_key_not_position():
    scores = np.zeros(5)  # everything tied
    key = np.array([0.5, 0.1, 0.4, 0.2, 0.3])
    ids = np.arange(5)
    got = bench_ha_prime.top_neighbours(scores, key, ids, self_id=1, k=3)
    assert got == [3, 4, 2]  # ascending key, self (id 1) excluded
    perm = ids[::-1]
    assert bench_ha_prime.top_neighbours(scores[perm], key[perm], perm, 1, 3) == got


def test_hb_precision_is_a_probability_per_checkpoint_and_reproducible():
    kw = {"seed": 1, "checkpoints": (3, 6), "n_old": 5}
    a, b = bench_hb.run(**kw), bench_hb.run(**kw)
    assert a == b
    assert set(a["data"]) == set(bench_hb.ARMS)
    for arm in bench_hb.ARMS:
        assert set(a["data"][arm]) == {"0", "3", "6"}
        assert all(0.0 <= v <= 1.0 for v in a["data"][arm].values())


def test_ha_retrieval_reports_both_modes():
    data = bench_ha.run()["data"]
    assert set(data) == {"seeds", "iterative", "ppr"}
    assert all(0.0 <= v["recall@5"] <= 1.0 and v["response_bytes"] > 0 for v in data.values())
    assert (data["seeds"]["tool_calls"], data["iterative"]["tool_calls"]) == (1, 2)
    assert data["iterative"]["response_bytes"] > data["seeds"]["response_bytes"]


def test_hmcp_counts_one_call_for_the_batch_and_four_otherwise():
    d = bench_hmcp.run()["data"]
    assert (
        d["batch"]["tool_calls"],
        d["sequential"]["tool_calls"],
        d["granular"]["tool_calls"],
    ) == (1, 4, 4)
    assert d["granular"]["tool_list_bytes"] > d["sequential"]["tool_list_bytes"] > 0
    assert all(v["est_tokens"] > 0 for v in d.values())


def _hap(ratio, fly_b=1, cos_b=4):
    return {
        "config": {"flyhash_default": {"k": 64, "c": 12}},
        "data": {
            "1": {
                "cosine": {"bytes_per_node": cos_b},
                "fly": {
                    "k64_c12": {"fly_over_cosine": ratio, "bytes_per_node": fly_b},
                    "k410_c24": {"fly_over_cosine": 0.99, "bytes_per_node": fly_b},
                },
            }
        },
    }


@pytest.mark.parametrize(
    "builder, expected",
    [
        (lambda: _hap(0.95), report.SUPPORTED),
        (lambda: _hap(0.56), report.NOT_SUPPORTED),
        (lambda: _hap(0.95, fly_b=8), report.NOT_SUPPORTED),
        (lambda: None, report.INCONCLUSIVE),
    ],
)
def test_ha_prime_verdict_follows_the_ninety_percent_rule(builder, expected):
    assert report.verdict_ha_prime(builder())[0] == expected


def test_ha_prime_verdict_reports_the_best_swept_setting():
    assert "k410_c24" in report.verdict_ha_prime(_hap(0.5))[1]


def test_hb_verdict_applies_precondition_and_retention():
    def hb(**arms):
        return {"data": {"sparse": {"0": 1.0, "9": 1.0}, **arms}}

    ok = {"0": 1.0, "9": 0.2}
    assert report.verdict_hb(hb(overwrite=ok))[0] == report.SUPPORTED
    assert report.verdict_hb(hb(overwrite={"0": 1.0, "9": 1.0}))[0] == report.NOT_SUPPORTED
    # a baseline that never learned (N=0 < 0.9) is excluded, not counted as a win
    assert report.verdict_hb(hb(dense={"0": 0.05, "9": 0.05}))[0] == report.INCONCLUSIVE
    assert report.verdict_hb(hb(dense={"0": 0.05, "9": 0.05}, overwrite=ok))[0] == report.SUPPORTED
    assert "dense" in report.verdict_hb(hb(dense={"0": 0.05, "9": 0.05}, overwrite=ok))[1]
    bad = {"data": {"sparse": {"0": 0.5, "9": 0.5}, "overwrite": ok}}
    assert report.verdict_hb(bad)[0] == report.INCONCLUSIVE


def test_ha_verdict_compares_with_the_iterative_baseline():
    def mk(ppr5, it5, ppr_ms, it_ms, ppr_b=800.0, it_b=900.0):
        return {
            "data": {
                "seeds": {"recall@5": 0.3, "query_ms": 0.5, "response_bytes": 400.0},
                "iterative": {"recall@5": it5, "query_ms": it_ms, "response_bytes": it_b},
                "ppr": {"recall@5": ppr5, "query_ms": ppr_ms, "response_bytes": ppr_b},
            }
        }

    assert report.verdict_ha(mk(0.45, 0.3, 1.0, 1.3), None)[0] == report.SUPPORTED
    assert report.verdict_ha(mk(0.45, 0.5, 1.0, 1.3), None)[0] == report.NOT_SUPPORTED
    assert report.verdict_ha(mk(0.45, 0.3, 2.0, 1.3), None)[0] == report.NOT_SUPPORTED
    assert report.verdict_ha(mk(0.45, 0.3, 1.0, 1.3, ppr_b=950.0), None)[0] == report.NOT_SUPPORTED
    assert report.verdict_ha({"data": {"seeds": {}, "ppr": {}}}, None)[0] == report.INCONCLUSIVE

    def mk(b, s, g):
        return {
            "data": {
                "batch": {"est_tokens": b},
                "sequential": {"est_tokens": s},  # noqa: E731
                "granular": {"est_tokens": g},
            }
        }

    assert report.verdict_hmcp(mk(1, 2, 3))[0] == report.SUPPORTED
    assert report.verdict_hmcp(mk(5, 2, 3))[0] == report.NOT_SUPPORTED


def test_the_report_is_inconclusive_without_data_and_llm_unreachable(tmp_path):
    text = report.render(tmp_path)
    assert text.count(report.INCONCLUSIVE) == 11
    (tmp_path / "llm.json").write_text(json.dumps({"status": "UNREACHABLE", "data": {}}))
    assert (
        report.verdict_llm(json.loads((tmp_path / "llm.json").read_text()), "H-a", "memory")[0]
        == report.INCONCLUSIVE
    )


def test_eval_without_a_server_is_unreachable_not_an_exception():
    out = eval_llm.run(LlmClient(base_url="http://127.0.0.1:9/v1", timeout_s=1))
    assert out["status"] == "UNREACHABLE" and out["data"] == {}


def test_eval_runs_end_to_end_against_the_fake_server(monkeypatch):
    monkeypatch.setattr(eval_llm, "QA_ITEMS", QA_ITEMS[:2])
    monkeypatch.setattr(eval_llm, "PUZZLES", PUZZLES[:2])
    with FakeLlm() as fake:
        fake.script = [reply_text("?", reasoning="thinking...")] * 40
        out = eval_llm.run(LlmClient(base_url=fake.url, model="fake", timeout_s=10, thinking=True))
        requests = fake.requests
    assert out["status"] == "OK"
    assert set(out["data"]) == {"H-a", "H-c"}
    assert out["data"]["H-a"]["alone"]["accuracy"] == 0.0
    assert set(out["modes"]) == {"off", "on"}
    assert "H-a" in out["modes"]["off"] and "H-a" not in out["modes"]["on"]
    assert out["modes"]["on"]["H-c"]["alone"]["mean_reasoning_chars"] == len("thinking...")
    # per-mode budget and thinking switch actually reach the wire (regardless of the base client)
    by_budget = {(r["max_tokens"], "chat_template_kwargs" in r) for r in requests}
    assert by_budget == {(512, True), (2048, False)}


def test_eval_writes_one_result_file_per_mode(tmp_path, monkeypatch):
    monkeypatch.setattr(eval_llm, "RESULTS_DIR", tmp_path)
    result = {
        "hypothesis": "H-a/H-c",
        "config": {},
        "status": "OK",
        "data": {},
        "modes": {"off": {"H-c": {}}, "on": {"H-c": {}}},
    }
    eval_llm.write(result)
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "llm.json",
        "llm_thinking_off.json",
        "llm_thinking_on.json",
    ]
    assert json.loads((tmp_path / "llm_thinking_on.json").read_text())["thinking"] == "on"
