"""T19 H-b end to end: alias dataset, retrieval bench, eval group wiring, report verdict."""

import asyncio
import json
from pathlib import Path

import pytest

from chrysalis_mcp.domains import memory
from tests.fake_llm import FakeLlm, reply_text
from validation import bench_hb_e2e, eval_llm, report
from validation.datasets import N_INTERFERENCE, alias_dataset
from validation.llm_client import LlmClient


def test_dataset_aliases_are_absent_from_the_kb_and_questions_carry_them():
    data = alias_dataset()
    kb_text = " ".join(x for t in data.kb_triples for x in t).casefold()
    assert len(data.items) == 24 and len(data.interference_pairs) == N_INTERFERENCE
    assert len({i.alias for i in data.items}) == 24 and len({i.entity for i in data.items}) == 24
    for item in data.items:
        assert item.alias.casefold() not in kb_text
        assert item.alias in item.test and item.alias in item.teach
        assert item.entity not in item.test and item.entity not in item.teach
        assert item.teach != item.test
        assert item.answer == item.triples[0][2] and 1 <= len(item.triples) <= 2
        assert item.answer.isdigit() and all(t[0] == item.entity for t in item.triples)


def test_dataset_sets_are_disjoint_complete_and_deterministic():
    data = alias_dataset()
    a, b = {i.entity for i in data.set_a}, {i.entity for i in data.set_b}
    assert a.isdisjoint(b) and a | b == {i.entity for i in data.items} and len(a) == len(b) == 12
    assert alias_dataset() == data and alias_dataset(seed=1) != data
    queries = {q for q, _ in data.interference_pairs}
    assert not queries & {i.teach for i in data.items} and len(queries) == N_INTERFERENCE
    with pytest.raises(ValueError):
        alias_dataset(n_entities=1)


def test_arm_operations_differ_only_in_how_the_alias_is_learned():
    data = alias_dataset(n_interference=5)
    assert bench_hb_e2e.teach_ops("no_learn", data.set_a) == []
    fly = bench_hb_e2e.teach_ops("fly_update", data.set_a)
    assert [a for a, _ in fly] == ["update"] * 12 and fly[0][1]["node"] == data.set_a[0].entity
    alias = bench_hb_e2e.teach_ops("alias_ingest", data.set_a)
    row = alias[0][1]["triples"][0]
    assert row == {"s": data.set_a[0].alias, "r": "alias_of", "o": data.set_a[0].entity}
    tail = [a for a, _ in bench_hb_e2e.retention_ops("fly_update_consolidate", data)]
    assert tail[-1] == "consolidate" and tail.count("update") == 12 + 5
    clean = [a for a, _ in bench_hb_e2e.retention_ops("alias_ingest_clean", data)]
    assert "update" not in clean  # no interference updates in the clean baseline
    with pytest.raises(ValueError):
        bench_hb_e2e.teach_ops("nope", data.set_a)


def test_tiny_retrieval_run_is_deterministic_and_fly_update_beats_no_learn():
    kw = {"seeds": (19, 20), "arms": ("no_learn", "fly_update"), "n_entities": 8}
    first = bench_hb_e2e.run(n_interference=20, **kw)
    assert first == bench_hb_e2e.run(n_interference=20, **kw)
    assert first["config"]["n_set_a"] == 4 and first["config"]["n_interference"] == 20
    no, fly = first["data"]["no_learn"], first["data"]["fly_update"]
    assert fly["taught"] > no["taught"] and fly["retained"] > no["retained"]
    assert all(0.0 <= v <= 1.0 for a in first["data"].values() for k, v in a.items() if "@" in k)


def test_eval_group_is_registered_and_prepares_every_arm_substrate(tmp_path):
    assert "hbe2e" in eval_llm.ALL_GROUPS and eval_llm.GROUP_NAMES["hbe2e"] == "H-b-e2e"
    assert "hbe2e" not in eval_llm.LONG_GROUPS  # runs in thinking on and off
    assert {"no_learn", "fly_update", "alias_ingest"} <= set(eval_llm.HBE2E_ARMS)
    with FakeLlm() as fake:
        fake.script = [reply_text("no idea")] * len(eval_llm.HBE2E_ARMS)
        llm = LlmClient(base_url=fake.url, model="fake", timeout_s=10, thinking=False)
        out = asyncio.run(eval_llm._group("hbe2e", llm, 64, 1, 4.0, str(tmp_path)))
        system = fake.requests[0]["messages"][0]["content"]
        assert "tools" in fake.requests[0] and "recall" in system
    assert list(out) == list(eval_llm.HBE2E_ARMS)
    for arm in out.values():
        assert arm["n"] == 1 and arm["accuracy"] == 0.0 and arm["memory_use_rate"] == 0.0
        assert len(arm["items"]) == 1 and "mean_prompt_tokens" in arm
    # the substrates were really prepared by the harness: fly arm knows the aliases, no_learn not
    # Measured over all of set A (rates, not one item): with the bounded boost a single taught
    # alias can legitimately miss (default seed: 11/12 taught), so a one-item assertion is brittle.
    data = alias_dataset()
    rates = {}
    for arm in ("no_learn", "fly_update"):
        db = next(Path(tmp_path).glob(f"hbe2e_{arm}_0.db"))
        try:
            run = bench_hb_e2e._Run(str(db))  # configures the memory domain for this db
            rates[arm] = run.recall_rates(data.set_a, 5)[0]
        finally:
            memory.configure(None)
    assert rates["no_learn"] <= 0.1 and rates["fly_update"] >= 0.9


def _llm_arm(acc: float, use: float = 1.0) -> dict:
    return {
        "accuracy": acc,
        "mean_prompt_tokens": 100.0,
        "mean_completion_tokens": 10.0,
        "mean_tool_calls": 1.0,
        "memory_use_rate": use,
    }


def _hb_e2e(retained: float = 1.0) -> dict:
    row = {"taught": 1.0, "retained": retained, "taught@1": 1.0, "retained@1": 1.0}
    return {
        "config": {"k": 5, "seeds": [19], "n_set_a": 12, "n_set_b": 12, "n_interference": 300},
        "data": {
            "no_learn": {**row, "taught": 0.1, "retained": 0.0, "retention": 0.0},
            "fly_update": {**row, "retention": round(retained, 4)},
            "alias_ingest": {**row, "retained": 0.0, "retention": 0.0},
        },
    }


def _llm(off: tuple[float, float], on: tuple[float, float] | None = None) -> dict:
    def mode(no: float, fly: float) -> dict:
        return {
            "H-b-e2e": {
                "no_learn": _llm_arm(no, 0.9),
                "fly_update": _llm_arm(fly),
                "alias_ingest": _llm_arm(0.9),
            }
        }

    modes = {"off": mode(*off), **({"on": mode(*on)} if on else {})}
    return {"status": "OK", "data": modes["off"], "modes": modes}


def test_verdict_requires_the_gain_in_every_mode_and_retention():
    hb = _hb_e2e()
    assert report.verdict_hb_e2e(hb, _llm((0.1, 0.4), (0.0, 0.3)))[0] == report.SUPPORTED
    assert report.verdict_hb_e2e(hb, _llm((0.1, 0.4), (0.2, 0.3)))[0] == report.NOT_SUPPORTED
    assert report.verdict_hb_e2e(hb, _llm((0.1, 0.29)))[0] == report.NOT_SUPPORTED
    verdict, text = report.verdict_hb_e2e(_hb_e2e(retained=0.85), _llm((0.1, 0.9)))
    assert verdict == report.NOT_SUPPORTED and "retention >= 0.9: False" in text


def test_alias_baseline_is_informational_only():
    hb, llm = _hb_e2e(), _llm((0.1, 0.4))
    verdict, text = report.verdict_hb_e2e(hb, llm)  # alias_ingest 0.9 beats fly 0.4
    assert verdict == report.SUPPORTED and "alias_ingest 0.9" in text
    llm["modes"]["off"]["H-b-e2e"]["alias_ingest"]["accuracy"] = 0.0
    assert report.verdict_hb_e2e(hb, llm)[0] == report.SUPPORTED


def test_retrieval_only_results_render_without_llm_results(tmp_path):
    verdict, text = report.verdict_hb_e2e(_hb_e2e(), None)
    assert verdict == report.INCONCLUSIVE and "LLM arms not run" in text and "taught 1.0" in text
    assert report.verdict_hb_e2e(None, None)[0] == report.INCONCLUSIVE
    (tmp_path / "hb_e2e.json").write_text(json.dumps(_hb_e2e()))
    rendered = report.render(tmp_path)
    assert "H-b-e2e (learned aliases" in rendered and "## H-b-e2e retrieval" in rendered
    (tmp_path / "llm.json").write_text(json.dumps(_llm((0.1, 0.4))))
    assert "SUPPORTED" in report.render(tmp_path)


# --- regression: the learned boost must stay bounded (no "boost leak") ---------------------------

_TOPICS = ("weather", "cooking", "football", "python", "sailing", "music", "tax", "gardening")
_VERBS = ("how do I get started with", "what is the best way to learn", "tips for", "history of")


def _natural_query(i: int) -> str:
    return f"{_VERBS[i % 4]} {_TOPICS[i % 8]} number {i} in the region of Lima"


def _plain_kb_recall(run, data) -> float:
    hits = 0
    for item in data.items:
        res = run.call("recall", query=bench_hb_e2e.plain_question(item), k=5, mode="ppr")
        hits += item.entity in [r["text"] for r in res["results"]]
    return hits / len(data.items)


@pytest.mark.parametrize("kind", ["synthetic", "natural"])
def test_unrelated_updates_do_not_degrade_plain_kb_recall(tmp_path, kind):
    data = alias_dataset(19, n_interference=300)
    nodes = [n for _, n in data.interference_pairs]
    pairs = (
        list(data.interference_pairs)
        if kind == "synthetic"
        else [(_natural_query(i), nodes[i]) for i in range(300)]
    )
    run = bench_hb_e2e._Run(str(tmp_path / "leak.db"))
    try:
        run.apply(bench_hb_e2e.kb_ops(data))
        assert _plain_kb_recall(run, data) == 1.0
        assert run.unrelated_learned_rate(data.items, 5) == 0.0
        run.apply([("update", {"query": q, "node": n}) for q, n in pairs])
        assert _plain_kb_recall(run, data) == 1.0
        # interference updates must not spray learned_as facts onto unrelated plain questions
        assert run.unrelated_learned_rate(data.items, 5) <= 0.1
    finally:
        memory.configure(None)


def test_update_still_teaches_alias_paraphrases(tmp_path):
    data = alias_dataset(19)
    run = bench_hb_e2e._Run(str(tmp_path / "teach.db"))
    try:
        run.apply(bench_hb_e2e.kb_ops(data))
        run.apply(bench_hb_e2e.teach_ops("fly_update", data.set_a))
        taught, _ = run.recall_rates(data.set_a, 5)
    finally:
        memory.configure(None)
    assert taught >= 0.9


def test_fly_arm_explains_the_alias_and_no_learn_never_does(tmp_path):
    data = alias_dataset(19)
    rates = {}
    for arm in ("no_learn", "fly_update"):
        run = bench_hb_e2e._Run(str(tmp_path / f"{arm}.db"))
        try:
            run.apply(bench_hb_e2e.kb_ops(data))
            run.apply(bench_hb_e2e.teach_ops(arm, data.set_a))
            rates[arm] = run.explained_rate(data.set_a, 5)
            if arm == "fly_update":
                res = [run.call("recall", query=i.test, k=5, mode="ppr") for i in data.set_a]
                assert any(
                    f[1] == "learned_as" and f[2] == i.entity
                    for i, r in zip(data.set_a, res, strict=True)
                    for f in r["facts"]
                )
                assert any(x.get("learned") for r in res for x in r["results"])
            else:
                res = run.call("recall", query=data.set_a[0].test, k=5, mode="ppr")
                assert not [f for f in res["facts"] if f[1] == "learned_as"]
        finally:
            memory.configure(None)
    assert rates["no_learn"] == 0.0 and rates["fly_update"] >= 0.9


def test_boost_is_an_overlap_fraction_on_the_cosine_scale():
    assert memory.BOOST_SCALE == 4.0 and memory.DEFAULT_LAMBDA == 0.5
