"""T12: long-context baseline, hard DSL set, advised arm, calibration loop, report sections."""

import asyncio
import json
import re
from typing import Any

import pytest

from chrysalis_mcp.core.session import SessionCache
from chrysalis_mcp.domains import build_registry
from chrysalis_mcp.server import handle_call
from tests.fake_llm import FakeLlm, reply_text, reply_tool
from validation import eval_llm, report
from validation.bridge import ChrysalisSession
from validation.datasets import HARD_PUZZLES, QA_ITEMS, score_puzzle, score_qa
from validation.llm_client import LlmClient

pytest.importorskip("z3")


def _solve(constraints: dict[str, Any], max_models: int = 2) -> dict[str, Any]:
    out = handle_call(
        build_registry(),
        SessionCache(),
        "reason",
        "solve",
        {"constraints": constraints, "max_models": max_models},
        "summary",
    )
    return json.loads(out)


def test_hard_set_shape():
    assert len(HARD_PUZZLES) >= 15
    assert len({p.text for p in HARD_PUZZLES}) == len(HARD_PUZZLES)
    assert sum(p.goal is not None for p in HARD_PUZZLES) >= 4
    assert sum(len(p.answer) >= 3 for p in HARD_PUZZLES) >= 10
    assert all(2 <= len(p.answer) <= 5 for p in HARD_PUZZLES)
    assert all(score_puzzle(json.dumps(p.answer), p.answer) for p in HARD_PUZZLES)


@pytest.mark.parametrize("puzzle", HARD_PUZZLES, ids=lambda p: p.text[:40])
def test_hard_puzzle_answer_is_the_unique_z3_solution(puzzle):
    if puzzle.goal:  # the optimum is the declared answer ...
        out = _solve(puzzle.constraints, max_models=1)
        assert out["result"] == "sat", puzzle.text
        assert {k: int(v) for k, v in out["model"].items()} == puzzle.answer
        kind, var = puzzle.goal.split(":")
        pinned = {
            "vars": puzzle.constraints["vars"],
            "assert": [*puzzle.assert_, f"{var} == {puzzle.answer[var]}"],
        }
    else:
        pinned = {"vars": puzzle.constraints["vars"], "assert": list(puzzle.assert_)}
    out = _solve(pinned, max_models=2)  # ... and nothing else attains it
    assert out["result"] == "sat" and len(out["models"]) == 1, f"not unique: {puzzle.text}"
    assert {k: int(v) for k, v in out["models"][0].items()} == puzzle.answer


def test_brief_schema_is_compact_and_advisory_is_level_only():
    registry, cache = build_registry(), SessionCache()
    handle_call(registry, cache, "meta", "handshake", {"model": "m", "params_b": 4})
    brief = json.loads(handle_call(registry, cache, "meta", "schema", {}, "brief"))
    assert len(json.dumps(brief).encode()) <= 8000
    assert brief["domains"]["memory"]["recall"]["advisory"] == "recommended"
    assert "dis:n_atoms<=6&goal=sat" in brief["rules"]["reason.check,reason.solve"]
    summary = json.loads(handle_call(registry, cache, "meta", "schema", {}, "summary"))
    assert "why" in summary["domains"]["memory"]["recall"]["advisory"]


def test_handshake_returns_the_advisory_view_at_every_verbosity():
    registry, cache = build_registry(), SessionCache()
    p = {"model": "gemma-4-e4b", "params_b": 4, "thinking": False}
    full = json.loads(handle_call(registry, cache, "meta", "handshake", p, "summary"))
    assert full["advisories"]["memory.recall"]["level"] == "recommended"
    assert "why" in full["advisories"]["memory.recall"]
    assert "rec:" in full["rules"]["reason.check,reason.solve"]
    brief = json.loads(handle_call(registry, cache, "meta", "handshake", p, "brief"))
    assert brief["advisories"]["memory.recall"] == "recommended"
    assert "rules" in brief
    assert "why" not in json.dumps(brief["advisories"])


def _run(coro):
    return asyncio.run(coro)


def test_context_arm_carries_every_fact_and_no_tools():
    facts = eval_llm.context_facts().splitlines()
    assert len(facts) == 2 * len(QA_ITEMS) and len(set(facts)) == len(facts)
    assert facts == eval_llm.context_facts().splitlines()  # fixed seed
    unshuffled = [f"{s} {r.replace('_', ' ')} {o}." for i in QA_ITEMS for s, r, o in i.triples]
    assert sorted(facts) == sorted(unshuffled) and facts != unshuffled
    with FakeLlm() as fake:
        fake.script = [reply_text(QA_ITEMS[0].answer)]
        llm = LlmClient(base_url=fake.url, model="fake", timeout_s=10, thinking=False)
        out = _run(eval_llm._group("ctx", llm, 64, 1, 4.0, "."))
        req = fake.requests[0]
    assert out["context"]["accuracy"] == 1.0
    system = req["messages"][0]["content"]
    assert all(f in system for f in facts) and "tools" not in req


def test_advised_arm_calls_handshake_first_and_sees_the_advisories():
    who = {"model": "fake", "family": "fake", "params_b": 4.0, "thinking": False}
    handshake = {"domain": "meta", "action": "handshake", "params": who}
    puzzle = HARD_PUZZLES[0]
    with FakeLlm() as fake:
        fake.script = [reply_tool(handshake), reply_text(json.dumps(puzzle.answer))]
        llm = LlmClient(base_url=fake.url, model="fake", timeout_s=10, thinking=False)
        out = _run(eval_llm._group("adv", llm, 64, 1, 4.0, "."))
        first, second = fake.requests
    arm = out["advised"]
    assert arm["accuracy"] == 1.0
    assert arm["handshake_first_rate"] == 1.0 and arm["reason_use_rate"] == 0.0
    assert "handshake" in first["messages"][0]["content"] and "tools" in first
    tool_msg = next(m for m in second["messages"] if m["role"] == "tool")
    assert "advisories" in tool_msg["content"]


def test_tool_arm_documents_the_dsl_and_records_use():
    solve = {
        "domain": "reason",
        "action": "solve",
        "params": {"constraints": HARD_PUZZLES[0].constraints},
    }
    with FakeLlm() as fake:
        answer = reply_text(json.dumps(HARD_PUZZLES[0].answer))
        fake.script = [reply_text("no idea"), reply_tool(solve), answer]  # alone, then tool arm
        llm = LlmClient(base_url=fake.url, model="fake", timeout_s=10, thinking=False)
        out = _run(eval_llm._group("hard", llm, 64, 1, 4.0, "."))
        sys_prompt = fake.requests[1]["messages"][0]["content"]
    assert out["tool"]["reason_use_rate"] == 1.0 and out["tool"]["accuracy"] == 1.0
    assert out["alone"]["reason_use_rate"] == 0.0 and out["alone"]["accuracy"] == 0.0
    assert "goal" in sys_prompt and "min:x" in sys_prompt


class _ProbeSolver:
    """Duck-typed model: reads the probes in the prompt, answers them right or wrong."""

    model, thinking = "duck", False

    def __init__(self, right: bool) -> None:
        self.right, self.calls = right, 0

    def chat(self, messages, tools=None, **_):
        self.calls += 1
        if self.calls > 1:
            return {"status": "OK", "message": {"content": "done"}, "usage": _USAGE}
        answers = self._answers(messages[-1]["content"]) if self.right else {"reason-1": "?"}
        call = {
            "id": "c1",
            "type": "function",
            "function": {
                "name": "chrysalis",
                "arguments": json.dumps(
                    {"domain": "meta", "action": "calibrate", "params": {"answers": answers}}
                ),
            },
        }
        return {
            "status": "OK",
            "message": {"content": None, "tool_calls": [call]},
            "usage": _USAGE,
        }

    @staticmethod
    def _answers(text: str) -> dict[str, Any]:
        def num(pattern: str) -> re.Match[str]:
            m = re.search(pattern, text)
            assert m, pattern
            return m

        a, b = num(r"plus (\d+) equals (\d+)").groups()
        total = num(r"their sum is (\d+)").group(1)
        verb, var = num(r"to (minimize|maximize) variable (\w)").groups()
        hits = json.loads("[" + num(r"recall hits \[(.*?)\], answer").group(1) + "]")
        best = max(hits, key=lambda h: h["score"])
        city = num(r"lives in (\w+)\"").group(1)
        return {
            "reason-1": [f"x+{a}=={b}"],
            "reason-2": ["x>y", f"x+y=={total}"],
            "reason-3": f"{verb[:3]}:{var}",
            "memory-1": f"{best['id']}:{round(best['score'] * 10)}",
            "memory-2": city[::-1],
        }


_USAGE = {"prompt_tokens": 5, "completion_tokens": 3}
_WHO = {"model": "duck", "family": "duck", "params_b": 4.0, "thinking": False}


def test_calibration_loop_scores_right_and_wrong_answers():
    right = _run(eval_llm.calibrate_once(_ProbeSolver(True), 64, _WHO))
    assert right["submitted"] and right["scores"] == {"reason": 1.0, "memory": 1.0}
    wrong = _run(eval_llm.calibrate_once(_ProbeSolver(False), 64, _WHO))
    assert wrong["submitted"] and wrong["scores"] == {"reason": 0.0}


def test_calibration_without_an_answer_records_no_scores():
    with FakeLlm() as fake:
        fake.script = [reply_text("I will not use the tool")]
        llm = LlmClient(base_url=fake.url, model="fake", timeout_s=10, thinking=False)
        out = _run(eval_llm.calibrate_once(llm, 64, _WHO))
    assert out["submitted"] is False and out["scores"] == {}


def test_calibration_probe_prompt_lists_every_probe():
    async def probes():
        async with ChrysalisSession(None) as cs:
            issued = json.loads(await cs.call(domain="meta", action="calibrate"))
        return issued["probes"]

    probes_ = _run(probes())
    text = eval_llm.probe_prompt(probes_)
    assert all(p["id"] in text and p["prompt"] in text for p in probes_)


def _arm(acc: float, prompt: float = 100.0, use: float | None = 0.0, hs: float | None = 0.0):
    return {
        "n": 17,
        "accuracy": acc,
        "mean_prompt_tokens": prompt,
        "mean_completion_tokens": 10.0,
        "mean_tool_calls": 1.0,
        "reason_use_rate": use,
        "handshake_first_rate": hs,
    }


def _fixture() -> dict[str, Any]:
    off = {
        "H-a": {"alone": _arm(0.0), "memory": _arm(0.65, 900.0)},
        "H-a-ctx": {"context": _arm(0.7, 1500.0)},
        "H-c-hard": {"alone": _arm(0.2), "tool": _arm(0.5, use=0.9)},
        "H-adv": {"advised": _arm(0.45, use=0.2, hs=1.0)},
        "calibration": {"status": "OK", "scores": {"reason": 0.9, "memory": 0.9}},
    }
    return {"status": "OK", "data": off, "modes": {"off": off}}


def test_report_renders_the_new_sections_and_verdicts(tmp_path):
    (tmp_path / "llm.json").write_text(json.dumps(_fixture()))
    text = report.render(tmp_path)
    assert "H-a-ctx (memory vs full context)" in text and "## Calibration vs measured" in text
    llm = _fixture()
    assert report.verdict_ctx(llm)[0] == report.SUPPORTED  # 0.65 >= 0.7-0.05, 900 < 1500 tokens
    assert report.verdict_hard(llm)[0] == report.SUPPORTED  # gain 0.3
    verdict, text = report.verdict_adv(llm)  # 0.45 >= 0.5-0.05, but fitted on HARD_PUZZLES
    assert verdict == report.INCONCLUSIVE and report.IN_SAMPLE_NOTE in text
    rows = report.calibration_rows(llm)
    assert [r[1:3] for r in rows] == [["reason", 0.9], ["memory", 0.9]]
    assert all(r[4] is True for r in rows)  # high score and the tool helped, both families
    llm["modes"]["off"]["H-adv"]["advised"]["accuracy"] = 0.2
    verdict, text = report.verdict_adv(llm)
    assert verdict == report.NOT_SUPPORTED and report.IN_SAMPLE_NOTE in text
    llm["modes"]["off"]["H-c-hard"]["tool"]["accuracy"] = 0.25
    assert report.verdict_hard(llm)[0] == report.NOT_SUPPORTED


def test_report_is_inconclusive_when_new_arms_were_not_run():
    for verdict in (report.verdict_ctx, report.verdict_hard, report.verdict_adv):
        assert verdict(None)[0] == report.INCONCLUSIVE
        assert verdict({"status": "OK", "modes": {"off": {}}})[0] == report.INCONCLUSIVE


def test_results_merge_per_mode_and_group_without_losing_data():
    old = {"status": "OK", "config": {}, "data": {}, "modes": {"off": {"H-a": {"x": 1}}}}
    new = {
        "status": "OK",
        "config": {},
        "data": {},
        "modes": {"off": {"H-c-hard": {"y": 2}}, "on": {"H-c-hard": {"y": 3}}},
    }
    merged = eval_llm.merge(old, new)
    assert set(merged["modes"]["off"]) == {"H-a", "H-c-hard"} and "on" in merged["modes"]
    assert set(merged["data"]) == {"H-a", "H-c-hard"}
    dead = {"status": "UNREACHABLE", "data": {}}
    assert eval_llm.merge(old, dead) is old


def test_run_rejects_unknown_groups_and_is_unreachable_without_a_server():
    with pytest.raises(ValueError):
        eval_llm.run(LlmClient(base_url="http://127.0.0.1:9/v1", timeout_s=1), groups=("nope",))
    out = eval_llm.run(
        LlmClient(base_url="http://127.0.0.1:9/v1", timeout_s=1), groups=eval_llm.ALL_GROUPS
    )
    assert out["status"] == "UNREACHABLE"


def test_default_groups_skip_the_long_prompt_arms_in_thinking_mode():
    with FakeLlm() as fake:
        fake.script = [reply_text("?")] * 80
        llm = LlmClient(base_url=fake.url, model="fake", timeout_s=10, thinking=False)
        out = eval_llm.run(llm, groups=("ctx", "hard"), limit=1)
    assert "H-a-ctx" in out["modes"]["off"] and "H-a-ctx" not in out["modes"]["on"]
    assert set(out["modes"]["on"]) == {"config", "H-c-hard"}
    assert score_qa("x", "y") is False
