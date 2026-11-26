"""E1-E6: tool_choice, the fewshot / forced / oracle_ctx arms, H-bio, feedback hardening."""

import asyncio
import json
import tempfile
from typing import Any

import pytest

from tests.fake_llm import FakeLlm, reply_text, reply_tool
from validation import bio_steps, eval_llm, report
from validation.bridge import ChrysalisSession
from validation.datasets import HARD_PUZZLES, HELDOUT_EASY, HELDOUT_HARD, PUZZLES
from validation.llm_client import LlmClient

WHO = {"model": "m-4b", "family": "m", "params_b": 4.0, "thinking": False}
TOOLS = [{"type": "function", "function": {"name": "chrysalis", "parameters": {}}}]


def _llm(fake: FakeLlm, thinking: bool = False) -> LlmClient:
    return LlmClient(base_url=fake.url, model="fake", timeout_s=20, thinking=thinking)


def solve_call(puzzle: Any) -> dict[str, Any]:
    return {"domain": "reason", "action": "solve", "params": {"constraints": puzzle.constraints}}


# --- E1: tool_choice -------------------------------------------------------------------------


@pytest.mark.parametrize("choice", ["auto", "required", "none"])
def test_tool_choice_reaches_the_request(choice):
    with FakeLlm() as fake:
        fake.script = [reply_text("ok")]
        _llm(fake).chat([{"role": "user", "content": "x"}], tools=TOOLS, tool_choice=choice)
    assert fake.requests[0]["tool_choice"] == choice


def test_tool_choice_default_is_absent_and_needs_tools():
    with FakeLlm() as fake:
        fake.script = [reply_text("ok"), reply_text("ok")]
        _llm(fake).chat([{"role": "user", "content": "x"}], tools=TOOLS)
        _llm(fake).chat([{"role": "user", "content": "x"}], tool_choice="required")
    assert all("tool_choice" not in r for r in fake.requests)


def test_tool_choice_rejects_unknown_values():
    with pytest.raises(ValueError, match="tool_choice"):
        LlmClient("http://x/v1", model="m").chat([], tool_choice="always")


# --- E2: fewshot / forced / oracle_ctx -------------------------------------------------------


def test_fewshot_examples_are_not_eval_items_and_are_correct():
    texts = " ".join(p.text for p in (*HARD_PUZZLES, *HELDOUT_HARD, *HELDOUT_EASY, *PUZZLES))
    for needle in ("p + q = 17", "p - q = 5", "m + n <= 11", "m - n >= 2", "Maximize m."):
        assert needle not in texts

    async def solve(constraints: dict[str, Any]) -> dict[str, Any]:
        async with ChrysalisSession(None) as cs:
            raw = await cs.call(
                domain="reason", action="solve", params={"constraints": constraints}
            )
        return json.loads(raw)["model"]

    ex1 = {
        "vars": {"p": "int", "q": "int"},
        "assert": ["p >= 0", "p <= 30", "q >= 0", "q <= 30", "p + q == 17", "p - q == 5"],
        "goal": "sat",
    }
    ex2 = {
        "vars": {"m": "int", "n": "int"},
        "assert": ["m >= 0", "m <= 9", "n >= 0", "n <= 9", "m + n <= 11", "m - n >= 2", "n >= 2"],
        "goal": "max:m",
    }
    assert asyncio.run(solve(ex1)) == {"p": "11", "q": "6"}
    assert asyncio.run(solve(ex2)) == {"m": "9", "n": "2"}
    assert '{"p": "11", "q": "6"}' in eval_llm.FEWSHOT_EXAMPLES
    assert '{"m": "9", "n": "2"}' in eval_llm.FEWSHOT_EXAMPLES
    for call in ("'p + q == 17'", "'m + n <= 11'", "'goal': 'max:m'"):
        assert call in eval_llm.FEWSHOT_EXAMPLES


def test_advised_system_unchanged_without_examples_and_reminder_stays_last():
    plain = eval_llm.advised_system(WHO)
    shots = eval_llm.advised_system(WHO, "", eval_llm.FEWSHOT_EXAMPLES)
    assert plain.endswith(eval_llm.SYS_FORMAT_REMINDER) and shots.endswith(
        eval_llm.SYS_FORMAT_REMINDER
    )
    assert eval_llm.FEWSHOT_EXAMPLES in shots and eval_llm.FEWSHOT_EXAMPLES not in plain


def test_forced_choice_only_on_the_first_step():
    puzzle = HARD_PUZZLES[0]
    with FakeLlm() as fake:
        fake.script = [reply_tool(solve_call(puzzle)), reply_text("{}")]
        asyncio.run(
            eval_llm._arm(
                _llm(fake),
                [(puzzle.text, puzzle.answer)],
                eval_llm.SYS_HARD_TOOL,
                True,
                lambda a, g: False,
                None,
                None,
                64,
                first_tool_choice="required",
            )
        )
    assert fake.requests[0]["tool_choice"] == "required"
    assert "tool_choice" not in fake.requests[1]


def test_extra_arms_forced_oracle_fewshot_requests_and_labels():
    puzzles = list(HELDOUT_HARD[:2])
    with FakeLlm() as fake:
        fake.script = [reply_text("{}") for _ in range(6)]
        arms = asyncio.run(
            eval_llm._extra_adv_arms(_llm(fake), puzzles, ["hard", "hard"], WHO, 64, False)
        )
    few, forced, oracle = fake.requests[0:2], fake.requests[2:4], fake.requests[4:6]
    assert all(
        "tools" in r and eval_llm.FEWSHOT_EXAMPLES in r["messages"][0]["content"] for r in few
    )
    assert all(r["tool_choice"] == "required" and "tools" in r for r in forced)
    assert all("tools" not in r and "tool_choice" not in r for r in oracle)  # no tool exposed
    assert all("Solver result" in r["messages"][-1]["content"] for r in oracle)
    assert '"model"' in oracle[0]["messages"][-1]["content"]
    assert arms["oracle_ctx"]["upper_bound"] is True and arms["oracle_ctx"]["fair"] is False
    assert arms["oracle_ctx"]["oracle_solved"] == 2 and "UPPER BOUND" in arms["oracle_ctx"]["note"]
    assert arms["forced"]["first_tool_choice"] == "required"
    assert "upper_bound" not in arms["fewshot"] and arms["fewshot"]["prompt_hint_used"] is not None


def test_new_arms_in_both_advised_groups_and_in_all(tmp_path, monkeypatch):
    assert "hbio" in eval_llm.ALL_GROUPS and eval_llm.GROUP_NAMES["hbio"] == "H-bio"
    monkeypatch.setattr(eval_llm, "RESULTS_DIR", tmp_path)
    monkeypatch.delenv(eval_llm.ENV_ADVICE_DB, raising=False)
    with FakeLlm() as fake:
        res = eval_llm.run(_llm(fake), ("adv", "heldout"), ("off",), limit=1, params_b=4.0)
    data = res["modes"]["off"]
    assert {"advised", "fewshot", "forced", "oracle_ctx"} <= data["H-adv"].keys()
    assert {"alone", "tool", "advised", "fewshot", "forced", "oracle_ctx"} <= (
        data["H-adv-heldout"].keys()
    )
    assert data["H-adv"]["fewshot"]["in_sample"] is True
    assert "oracle_ctx" in data["config"]["extra_arms"]
    # the report still renders (unknown arms handled) and flags the upper bound
    eval_llm.write(res)
    md = report.render(tmp_path)
    assert "oracle_ctx [UPPER BOUND, not a fair arm]" in md and "fewshot" in md


# --- E5: H-bio -------------------------------------------------------------------------------

CORRECT: dict[str, dict[str, Any]] = {
    "handshake": {"domain": "meta", "action": "handshake", "params": {"model": "m-4b"}},
    "ingest": {
        "domain": "memory",
        "action": "ingest",
        "params": {"triples": [{"s": "Lumen", "r": "is_a", "o": "telescope"}]},
    },
    "teach1": {"domain": "memory", "action": "teach", "params": bio_steps.TEACH_1},
    "teach2": {"domain": "memory", "action": "update", "params": bio_steps.TEACH_2},
    "recall_quorum": {
        "domain": "memory",
        "action": "recall",
        "params": {"query": "space eye", "quorum": 2, "learned_min_status": "confirmed"},
    },
    "recall_usage": {
        "domain": "memory",
        "action": "recall",
        "params": {"query": "Kestrel", "track_usage": True},
    },
    "recall_inhib": {
        "domain": "memory",
        "action": "recall",
        "params": {"query": "Quito", "inhibition": "adaptive"},
    },
    "recall_sdm": {
        "domain": "memory",
        "action": "recall",
        "params": {"query": "lunar rover", "mode": "sdm"},
    },
    "flyvec_train": {"domain": "memory", "action": "flyvec_train", "params": {"epochs": 3}},
    "recall_flyvec": {
        "domain": "memory",
        "action": "recall",
        "params": {"query": "telescope", "encoder": "flyvec"},
    },
    "consolidate": {
        "domain": "memory",
        "action": "consolidate",
        "params": {"decay": 0.8, "replay": True, "usage_weighting": True, "capture": True},
    },
    "advise": {
        "domain": "meta",
        "action": "advise",
        "params": {"domain": "reason", "action": "solve", "task": {"n_vars": 6, "goal": "sat"}},
    },
    "feedback": {
        "domain": "meta",
        "action": "feedback",
        "params": {"action": "reason.solve", "used_tool": True, "outcome": "success"},
    },
    "advice_stats": {
        "domain": "meta",
        "action": "advice_stats",
        "params": {"action": "reason.solve"},
    },
    "advice_reset": {
        "domain": "meta",
        "action": "advice_reset",
        "params": {"action": "reason.solve", "confirm": True},
    },
}


def _bio(fake: FakeLlm, limit: int | None = None) -> dict[str, Any]:
    with tempfile.TemporaryDirectory():
        return asyncio.run(eval_llm._bio_group(_llm(fake), 64, limit, WHO))


def test_bio_steps_cover_the_fifteen_smoke_steps():
    assert [s.sid for s in bio_steps.STEPS] == list(CORRECT)
    assert len(bio_steps.STEPS) == 15


def test_every_step_seeds_cleanly_on_a_fresh_session(tmp_path):
    async def go() -> None:
        for step in bio_steps.STEPS:
            async with ChrysalisSession(
                str(tmp_path / f"{step.sid}.db"), str(tmp_path / f"{step.sid}_a.db")
            ) as cs:
                await bio_steps.seed_step(cs, step, WHO)

    asyncio.run(go())


def test_bio_all_steps_pass_with_correct_calls(monkeypatch):
    monkeypatch.setenv(bio_steps.ENV_REPS, "1")
    with FakeLlm() as fake:
        fake.script = [r for sid in CORRECT for r in (reply_tool(CORRECT[sid]), reply_text("done"))]
        out = _bio(fake)
    assert out["total"] == {"k": 15, "n": 15} and out["reps"] == 1
    assert all(v == {"k": 1, "n": 1, "called": 1, "server_ok": 1} for v in out["per_step"].values())


def test_bio_failing_step_does_not_affect_the_next_and_reps(monkeypatch):
    monkeypatch.setenv(bio_steps.ENV_REPS, "2")
    bad_ingest = {"domain": "memory", "action": "ingest", "params": {"bogus": 1}}
    with FakeLlm() as fake:
        fake.script = [
            reply_text("I will not call it"),  # handshake rep 0: never calls the tool
            reply_text("nope"),  # handshake rep 1
            reply_tool(bad_ingest),  # ingest rep 0: server ERROR
            reply_text("gave up"),
            reply_tool(bad_ingest),  # ingest rep 1
            reply_text("gave up"),
            reply_tool(CORRECT["teach1"]),  # teach1 rep 0 needs facts: harness seeded them
            reply_text("done"),
            reply_tool(CORRECT["teach1"]),  # teach1 rep 1
            reply_text("done"),
        ]
        out = _bio(fake, limit=3)
    ps = out["per_step"]
    assert out["total"] == {"k": 2, "n": 6} and out["reps"] == 2
    assert ps["handshake"] == {"k": 0, "n": 2, "called": 0, "server_ok": 0}
    assert ps["ingest"] == {"k": 0, "n": 2, "called": 0, "server_ok": 0}  # wrong params: no hit
    assert ps["teach1"] == {"k": 2, "n": 2, "called": 2, "server_ok": 2}  # memory.teach accepted


def test_bio_default_reps_and_bad_env(monkeypatch):
    monkeypatch.delenv(bio_steps.ENV_REPS, raising=False)
    assert bio_steps.reps_from_env() == 3
    monkeypatch.setenv(bio_steps.ENV_REPS, "0")
    with pytest.raises(ValueError, match=bio_steps.ENV_REPS):
        bio_steps.reps_from_env()


def test_bio_scoring_params_subset_and_server_status():
    step = next(s for s in bio_steps.STEPS if s.sid == "recall_usage")
    wrong = {"domain": "memory", "action": "recall", "params": {"query": "x", "track_usage": False}}
    right = {
        "domain": "memory",
        "action": "recall",
        "params": json.dumps({"query": "x", "track_usage": True}),
    }
    ok = {"status": "OK"}
    assert not bio_steps.score_step([(wrong, ok)], step)["called"]
    assert bio_steps.score_step([(right, ok)], step)["passed"]
    assert not bio_steps.score_step([(right, {"status": "ERROR"})], step)["passed"]
    fb = next(s for s in bio_steps.STEPS if s.sid == "feedback")
    syn = {
        "domain": "meta",
        "action": "feedback",
        "params": {"action": "reason.solve", "used_tool": True, "outcome": "success"},
    }
    assert bio_steps.score_step([(syn, ok)], fb)["passed"]  # curated synonym == server rename


def test_bio_scores_bare_list_params_like_the_server_wraps_them():
    ingest = next(s for s in bio_steps.STEPS if s.sid == "ingest")
    fb = next(s for s in bio_steps.STEPS if s.sid == "feedback")
    ok = {"status": "OK"}
    triples = [{"subject": "Lumen", "predicate": "is_a", "object": "telescope"}]
    as_list = {"domain": "memory", "action": "ingest", "params": triples}
    as_str = {"domain": "memory", "action": "ingest", "params": json.dumps(triples)}
    assert bio_steps.score_step([(as_list, ok)], ingest)["passed"]
    assert bio_steps.score_step([(as_str, ok)], ingest)["passed"]
    # meta.feedback has no sole required array param: the server rejects a list, so no hit
    no_wrap = {"domain": "meta", "action": "feedback", "params": json.dumps(triples)}
    assert not bio_steps.score_step([(no_wrap, ok)], fb)["called"]


def test_bio_batch_counts_only_when_a_step_is_the_target():
    fb = next(s for s in bio_steps.STEPS if s.sid == "feedback")
    ok = {"status": "OK"}

    def batch(steps, *, as_string=False):
        params = {"steps": steps}
        return {
            "domain": "meta",
            "action": "batch",
            "params": json.dumps(params) if as_string else params,
        }

    hit = [{"id": "a", "domain": "meta", "action": "feedback", "params": {"action": "x"}}]
    miss = [
        {"id": "a", "domain": "memory", "action": "ingest", "params": {"note": "meta feedback"}}
    ]
    assert bio_steps.score_step([(batch(hit), ok)], fb)["called"]
    assert bio_steps.score_step([(batch(hit, as_string=True), ok)], fb)["passed"]
    assert not bio_steps.score_step([(batch(miss), ok)], fb)["called"]  # substring no longer hits
    assert not bio_steps.score_step([(batch("nope"), ok)], fb)["called"]  # malformed steps
    assert not bio_steps.score_step([(batch(["meta.feedback"]), ok)], fb)["called"]


def test_bio_runs_concurrently_and_renders_in_report(tmp_path, monkeypatch):
    monkeypatch.setenv(bio_steps.ENV_REPS, "2")
    monkeypatch.setattr(eval_llm, "RESULTS_DIR", tmp_path)
    monkeypatch.delenv(eval_llm.ENV_ADVICE_DB, raising=False)
    with FakeLlm() as fake:
        res = eval_llm.run(
            _llm(fake), ("hbio",), ("off", "on"), limit=2, concurrency=2, params_b=4.0
        )
    off = res["modes"]["off"]
    assert off["H-bio"]["bio"]["total"]["n"] == 4
    assert off["config"]["bio_reps"] == 2 and off["config"]["bio_defaults"] in ("off", "all")
    assert res["modes"]["on"]["H-bio"]["bio"]["total"]["n"] == 4
    eval_llm.write(res)
    md = report.render(tmp_path)
    assert "H-bio: bio actions" in md and "TOTAL" in md


# --- E6: feedback hardening ------------------------------------------------------------------


def _loop_arm(fake: FakeLlm, puzzles: list[Any], golds_win: list[bool], monkeypatch) -> list[Any]:
    sent: list[dict[str, Any]] = []
    real = ChrysalisSession.call

    async def spy(self, /, **args: Any) -> str:
        if (
            args.get("action") == "feedback"
            and args.get("domain") == "meta"
            and "outcome" in args.get("params", {})
        ):
            sent.append(args["params"])
        return await real(self, **args)

    monkeypatch.setattr(ChrysalisSession, "call", spy)
    monkeypatch.setenv(eval_llm.ENV_FEEDBACK, "1")
    asyncio.run(
        eval_llm._arm(
            _llm(fake),
            [(p.text, w) for p, w in zip(puzzles, golds_win, strict=True)],
            "sys",
            True,
            lambda ans, gold: ans == "win" and gold,
            None,
            None,
            64,
            WHO,
            features=[eval_llm._features(p) for p in puzzles],
        )
    )
    return sent


def test_feedback_maps_used_outcome_and_features_with_mixed_values(monkeypatch):
    puzzles = list(HARD_PUZZLES[:4])
    with FakeLlm() as fake:
        fake.script = [
            reply_tool(solve_call(puzzles[0])),  # item 0: tool used, answers win
            reply_text("win"),
            reply_text("lose"),  # item 1: no tool, wrong
            reply_tool(solve_call(puzzles[2])),  # item 2: tool used, wrong
            reply_text("lose"),
            reply_text("win"),  # item 3: no tool, right
        ]
        sent = _loop_arm(fake, puzzles, [True, True, True, True], monkeypatch)
    assert [(s["used"], s["outcome"]) for s in sent] == [
        (True, "success"),
        (False, "failure"),
        (True, "failure"),
        (False, "success"),
    ]
    assert [s["task"] for s in sent] == [eval_llm._features(p) for p in puzzles]
    assert len({json.dumps(s["task"], sort_keys=True) for s in sent}) > 1  # features differ


def test_non_json_feedback_reply_counts_as_error_not_crash(monkeypatch):
    real = ChrysalisSession.call

    async def garbled(self, /, **args: Any) -> str:
        if args.get("action") in ("feedback", "advice_stats") and args.get("domain") == "meta":
            return "not json at all"
        return await real(self, **args)

    monkeypatch.setattr(ChrysalisSession, "call", garbled)
    monkeypatch.setenv(eval_llm.ENV_FEEDBACK, "1")
    puzzles = list(HARD_PUZZLES[:2])
    with FakeLlm() as fake:
        fake.script = [reply_text("x"), reply_text("y")]
        arm = asyncio.run(
            eval_llm._arm(
                _llm(fake),
                [(p.text, p.answer) for p in puzzles],
                "sys",
                False,
                lambda a, g: False,
                None,
                None,
                64,
                WHO,
                features=[eval_llm._features(p) for p in puzzles],
            )
        )
    assert arm["feedback_sent"] == 0 and arm["feedback_errors"] == 2
    assert arm["advice_stats"] is None and arm["n"] == 2
