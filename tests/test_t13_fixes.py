"""T13 fix cycle: hostile task params, held-out set, circularity-safe H-adv verdict, probe reuse."""

import asyncio
import json
import re
from typing import Any

import pytest

from chrysalis_mcp.core import task_features
from chrysalis_mcp.core.advisory import Advisor
from chrysalis_mcp.core.profile import Profile
from chrysalis_mcp.core.session import SessionCache
from chrysalis_mcp.domains import build_registry
from chrysalis_mcp.server import handle_call
from tests.fake_llm import FakeLlm, reply_text
from tests.test_validation_t12 import _solve
from validation import datasets, eval_llm, report
from validation.datasets import HARD_PUZZLES, HELDOUT_EASY, HELDOUT_HARD, HELDOUT_PUZZLES
from validation.llm_client import LlmClient

_DSL_CMP = re.compile(r"==|!=|<=|>=|<|>")


# --- 1: hostile params never escape dispatch ----------------------------------------------------


@pytest.mark.parametrize("bad", [5, None, "a<b", {"x": 1}, [1, None, {"y": 2}], True])
@pytest.mark.parametrize("action", ["solve", "check"])
def test_non_list_assert_never_raises_out_of_dispatch(bad, action):
    registry, cache = build_registry(), SessionCache()
    handle_call(registry, cache, "meta", "handshake", {"model": "m", "params_b": 4}, "summary")
    params = {"constraints": {"vars": {"x": "int"}, "assert": bad}}
    out = json.loads(handle_call(registry, cache, "reason", action, params, "summary"))
    assert isinstance(out, dict) and "status" in out
    assert task_features.extract(f"reason.{action}", params)["n_asserts"] == 0 or isinstance(
        bad, list
    )


def test_lookup_survives_a_feature_extractor_failure(monkeypatch):
    def boom(*_a: Any, **_k: Any) -> dict[str, Any]:
        raise RuntimeError("boom")

    monkeypatch.setattr(task_features, "extract", boom)
    got = Advisor.default().lookup("reason.solve", Profile(params_b=4), {"constraints": {}})
    assert got["level"] == "neutral"


# --- 2: held-out set -----------------------------------------------------------------------------


def _n_atoms(p: datasets.HardPuzzle) -> int:
    return sum(len(_DSL_CMP.findall(a)) for a in p.assert_)


def test_heldout_shape_is_disjoint_and_split_by_the_rule_boundary():
    assert (len(HELDOUT_HARD), len(HELDOUT_EASY)) == (8, 8)
    assert not {p.text for p in HELDOUT_PUZZLES} & {p.text for p in HARD_PUZZLES}
    assert len({p.text for p in HELDOUT_PUZZLES}) == 16
    assert all(7 <= _n_atoms(p) <= 12 and 3 <= len(p.answer) <= 5 for p in HELDOUT_HARD)
    assert all(
        _n_atoms(p) <= 6 and 2 <= len(p.answer) <= 3 and p.goal is None for p in HELDOUT_EASY
    )
    assert {p.goal is None for p in HELDOUT_HARD} == {True, False}  # mixed sat / opt


@pytest.mark.parametrize("puzzle", HELDOUT_PUZZLES, ids=lambda p: p.text[:40])
def test_heldout_answer_is_the_unique_z3_solution(puzzle):
    if puzzle.goal:
        out = _solve(puzzle.constraints, max_models=1)
        assert out["result"] == "sat"
        assert {k: int(v) for k, v in out["model"].items()} == puzzle.answer
        var = puzzle.goal.split(":")[1]
        pinned = {
            "vars": puzzle.constraints["vars"],
            "assert": [*puzzle.assert_, f"{var} == {puzzle.answer[var]}"],
        }
    else:
        pinned = {"vars": puzzle.constraints["vars"], "assert": list(puzzle.assert_)}
    out = _solve(pinned, max_models=2)
    assert out["result"] == "sat" and len(out["models"]) == 1, f"not unique: {puzzle.text}"
    assert {k: int(v) for k, v in out["models"][0].items()} == puzzle.answer


def test_the_advisor_splits_the_heldout_set_by_its_rule():
    adv, small = Advisor.default(), Profile(params_b=4, thinking=False)
    level = lambda p: adv.lookup("reason.solve", small, {"constraints": p.constraints})["level"]  # noqa: E731
    assert {level(p) for p in HELDOUT_HARD} == {"recommended"}
    assert {level(p) for p in HELDOUT_EASY} == {"discouraged"}


def test_heldout_group_runs_three_arms_and_reports_per_subset_tool_use():
    easy, hard = HELDOUT_EASY[0], HELDOUT_HARD[0]
    answers = [json.dumps(hard.answer), json.dumps(easy.answer)]
    with FakeLlm() as fake:
        fake.script = [reply_text(a) for _ in range(3) for a in answers]  # alone, tool, advised
        llm = LlmClient(base_url=fake.url, model="fake", timeout_s=10, thinking=False)
        out = asyncio.run(eval_llm._group("heldout", llm, 64, 1, 4.0, "."))
    assert set(out) == {"alone", "tool", "advised"}
    sub = out["advised"]["by_subset"]
    assert set(sub) == {"hard", "easy"} and sub["easy"]["n"] == sub["hard"]["n"] == 1
    assert sub["easy"]["reason_use_rate"] == 0.0 and out["advised"]["accuracy"] == 1.0
    assert "by_subset" not in out["alone"]
    legacy = asyncio.run(_adv_group())
    assert legacy["advised"]["in_sample"] is True


async def _adv_group() -> dict[str, Any]:
    with FakeLlm() as fake:
        fake.script = [reply_text(json.dumps(HARD_PUZZLES[0].answer))]
        llm = LlmClient(base_url=fake.url, model="fake", timeout_s=10, thinking=False)
        return await eval_llm._group("adv", llm, 64, 1, 4.0, ".")


def _arm(acc: float, use: float = 0.5, subset: bool = False) -> dict[str, Any]:
    arm = {"accuracy": acc, "handshake_first_rate": 1.0, "reason_use_rate": use}
    if subset:
        arm["by_subset"] = {
            "easy": {"n": 8, "accuracy": acc, "reason_use_rate": 0.1},
            "hard": {"n": 8, "accuracy": acc, "reason_use_rate": 0.9},
        }
    return arm


def _llm(**groups: Any) -> dict[str, Any]:
    off = {"H-c-hard": {"alone": _arm(0.2), "tool": _arm(0.5)}, **groups}
    return {"status": "OK", "modes": {"off": off}, "data": off}


def test_adv_verdict_on_heldout_compares_the_same_set_and_can_be_supported():
    held = {"alone": _arm(0.4), "tool": _arm(0.3), "advised": _arm(0.36, subset=True)}
    verdict, text = report.verdict_adv(_llm(**{"H-adv-heldout": held}))
    assert verdict == report.SUPPORTED and "held-out" in text and "easy n=8" in text
    assert "in-sample" not in text  # H-c-hard (0.5) is NOT the comparator here
    held["advised"]["accuracy"] = 0.2
    assert report.verdict_adv(_llm(**{"H-adv-heldout": held}))[0] == report.NOT_SUPPORTED


@pytest.mark.parametrize("flag", [{"in_sample": True}, {}])  # legacy results carry no flag
def test_adv_verdict_on_hard_puzzles_is_never_plain_supported(flag):
    llm = _llm(**{"H-adv": {"advised": {**_arm(0.5), **flag}}})
    verdict, text = report.verdict_adv(llm)
    assert verdict != report.SUPPORTED and text.endswith(report.IN_SAMPLE_NOTE)
    assert report.IN_SAMPLE_NOTE == "(in-sample; thresholds fitted on this set)"


# --- 3: one chars-per-token budget ---------------------------------------------------------------


def test_tokens_per_fact_derives_from_the_shared_chars_per_token_budget():
    assert datasets.CHARS_PER_TOKEN is task_features.CHARS_PER_TOKEN == 2.5
    lengths = [len(datasets.fact_sentence(*t)) for i in datasets.QA_ITEMS for t in i.triples]
    assert sum(lengths) / len(lengths) - 1 <= task_features.AVG_FACT_CHARS
    assert task_features.TOKENS_PER_FACT * task_features.CHARS_PER_TOKEN >= (
        task_features.AVG_FACT_CHARS
    )


# --- 4: a scored probe set is spent ---


def test_scored_probes_cannot_be_resubmitted_until_reissued():
    registry, cache = build_registry(), SessionCache()

    def call(params: dict[str, Any] | None = None) -> dict[str, Any]:
        return json.loads(
            handle_call(
                registry, cache, "meta", "calibrate", params or {}, "summary", session_id="s"
            )
        )

    probes = call()["probes"]
    assert probes
    first = call({"answers": {"memory-1": "x"}})
    assert first["status"] == "OK"
    again = call({"answers": {"memory-1": "y"}})
    assert again["status"] == "ERROR" and "already scored" in again["error"]
    assert call({"reissue": True})["probes"]  # a fresh set re-opens scoring
    assert call({"answers": {"memory-1": "z"}})["status"] == "OK"
