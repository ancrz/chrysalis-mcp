"""T13: task-conditioned advisories, calibration with a step-hungry thinking model, H-a-scale."""

import asyncio
import json
import re
from typing import Any

import pytest

from chrysalis_mcp.core import task_features
from chrysalis_mcp.core.advisory import Advisor, load_evidence, validate_entry
from chrysalis_mcp.core.profile import Profile, ProfileStore, score
from chrysalis_mcp.core.session import SessionCache
from chrysalis_mcp.domains import build_registry
from chrysalis_mcp.server import handle_call
from tests.fake_llm import FakeLlm, reply_text
from tests.test_validation_t12 import _USAGE, _WHO, _ProbeSolver
from validation import eval_llm, report
from validation.bridge import ChrysalisSession
from validation.datasets import (
    HARD_PUZZLES,
    PUZZLES,
    answerable_in,
    fit_context,
    scaled_qa,
)
from validation.llm_client import LlmClient


def _run(coro: Any) -> Any:
    return asyncio.run(coro)


def _entry(**kw: Any) -> dict[str, Any]:
    base = {
        "action": "reason.solve", "model_class": "any", "level": "recommended",
        "metric": "m", "value": "v", "source": "s", "date": "2026-11-05",
    }  # fmt: skip
    return {**base, **kw}


SMALL = Profile(params_b=4, thinking=False)

# --- features ----------------------------------------------------------------------------------


def test_features_from_the_dsl_and_from_smt2():
    dsl = {
        "constraints": {
            "vars": {"x": "int", "y": "int"},
            "assert": ["x >= 1", "x <= 9", "x + y == 10"],
            "goal": "max:x",
        }
    }
    assert task_features.extract("reason.solve", dsl) == {
        "n_vars": 2, "n_asserts": 3, "n_atoms": 3, "goal": "opt",
    }  # fmt: skip
    smt2 = (
        "(declare-const x Int)(declare-const y Int)(assert (and (> x 0) (<= y 3)))(assert (= x y))"
    )
    assert task_features.extract("reason.check", {"smt2": smt2}) == {
        "n_vars": 2, "n_asserts": 2, "n_atoms": 3, "goal": "sat",
    }  # fmt: skip


def test_features_are_empty_when_unknown_or_hostile():
    assert task_features.extract("reason.solve", None) == {}
    assert task_features.extract("memory.recall", {"query": "q"}) == {}
    assert task_features.extract("reason.solve", {"constraints": "nope"}) == {}
    bad = {"constraints": {"vars": 3, "assert": [1, None, "x > 1"], "goal": 5}}
    assert task_features.extract("reason.solve", bad)["n_atoms"] == 1


def test_facts_exceed_ctx_is_derived_from_facts_and_window():
    f = task_features.derive({"facts": 40}, 8192)
    assert f["facts_exceed_ctx"] is False
    assert task_features.derive({"facts": 800}, 8192)["facts_exceed_ctx"] is True
    assert (
        task_features.derive({"facts": 800, "ctx_tokens": 100_000}, 8192)["facts_exceed_ctx"]
        is False
    )
    assert "facts_exceed_ctx" not in task_features.derive({"facts": 800}, None)
    assert "facts_exceed_ctx" not in task_features.derive({"facts": True}, 8192)


# --- matching table ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("entries", "task", "expected"),
    [
        # a task-conditioned entry beats the model-level one when its condition holds
        (
            [_entry(level="neutral"), _entry(level="discouraged", task={"n_atoms": {"max": 6}})],
            {"n_atoms": 3},
            "discouraged",
        ),
        # ... and the model-level one is the fallback when it does not
        (
            [_entry(level="neutral"), _entry(level="discouraged", task={"n_atoms": {"max": 6}})],
            {"n_atoms": 9},
            "neutral",
        ),
        # missing features -> the task entry cannot match -> fallback
        (
            [_entry(level="neutral"), _entry(level="discouraged", task={"n_atoms": {"max": 6}})],
            {},
            "neutral",
        ),
        # more conditions = more specific
        (
            [
                _entry(level="recommended", task={"goal": "opt"}),
                _entry(level="discouraged", task={"goal": "opt", "n_atoms": {"max": 2}}),
            ],
            {"goal": "opt", "n_atoms": 1},
            "discouraged",
        ),
        # tie on specificity -> most recent date wins, regardless of severity
        (
            [
                _entry(level="discouraged", task={"goal": "sat"}, date="2026-01-01"),
                _entry(level="recommended", task={"goal": "sat"}, date="2026-11-05"),
            ],
            {"goal": "sat"},
            "recommended",
        ),
        # list condition, inclusive bounds, bool condition
        (
            [_entry(level="discouraged", task={"goal": ["sat", "opt"]})],
            {"goal": "opt"},
            "discouraged",
        ),
        (
            [_entry(level="discouraged", task={"n_vars": {"min": 3, "max": 3}})],
            {"n_vars": 3},
            "discouraged",
        ),
        (
            [_entry(level="discouraged", task={"facts_exceed_ctx": False})],
            {"facts": 1},
            "neutral",
        ),  # no window known -> feature absent
        # pending entries never advise
        (
            [_entry(level="discouraged", task={"goal": "sat"}, status="pending")],
            {"goal": "sat"},
            "neutral",
        ),
    ],
)
def test_task_matching_table(entries, task, expected):
    assert Advisor(entries).lookup("reason.solve", SMALL, task=task)["level"] == expected


def test_bool_condition_does_not_match_int_and_window_comes_from_the_profile():
    adv = Advisor(
        [_entry(action="memory.recall", level="discouraged", task={"facts_exceed_ctx": True})]
    )
    wide = Profile(params_b=4, context=8192)
    assert adv.lookup("memory.recall", wide, task={"facts": 800})["level"] == "discouraged"
    assert adv.lookup("memory.recall", wide, task={"facts": 10})["level"] == "neutral"
    assert adv.lookup("memory.recall", wide, task={"facts_exceed_ctx": 1})["level"] == "neutral"


def test_lookup_measures_features_from_call_params():
    adv = Advisor.default()
    easy = {"constraints": {"vars": {"x": "int"}, "assert": ["x > 1", "x < 3"]}}
    hard = {"constraints": {"vars": {"x": "int"}, "assert": ["x > 1"] * 8}}
    assert adv.lookup("reason.solve", SMALL, easy)["level"] == "discouraged"
    assert adv.lookup("reason.solve", SMALL, hard)["level"] == "recommended"


def test_evidence_task_entries_validate():
    assert validate_entry(_entry(task={"nope": 1})) == ["unknown task feature 'nope'"]
    assert validate_entry(_entry(task={})) == ["task must be a non-empty object"]
    assert validate_entry(_entry(task={"n_atoms": {"lt": 3}})) == [
        "task.n_atoms: bounds are ('min', 'max')"
    ]
    assert validate_entry(_entry(status="soon")) == [
        "status must be one of ('measured', 'pending')"
    ]
    assert load_evidence()  # the shipped file is valid


def test_h_a_scale_evidence_is_measured_and_recommends_memory_when_facts_exceed_ctx():
    adv = Advisor.default()
    wide = Profile(params_b=4, context=8192)
    assert adv.lookup("memory.recall", wide, task={"facts": 800})["level"] == "recommended"  # H-a
    assert adv.lookup("memory.recall", wide, task={"facts": 40})["level"] == "neutral"  # H-a-ctx
    assert not [e for e in adv.evidence if e.get("status") == "pending"]
    big = adv.lookup("memory.recall", wide, task={"facts_exceed_ctx": True})
    assert big["level"] == "recommended" and "job 1098" in big["why"]


# --- thresholds are the ones the datasets give -------------------------------------------------


def _levels(profile: Profile, params_list: list[dict[str, Any]]) -> list[str]:
    adv = Advisor.default()
    return [adv.lookup("reason.solve", profile, p)["level"] for p in params_list]


def test_rule_separates_the_easy_set_from_the_hard_set():
    easy = _levels(SMALL, [{"smt2": p.smt2} for p in PUZZLES])
    hard = _levels(SMALL, [{"constraints": p.constraints} for p in HARD_PUZZLES])
    assert set(hard) == {"recommended"}  # every hard item
    assert easy.count("discouraged") >= len(easy) - 1  # all but the one 9-comparison outlier
    assert easy.count("recommended") <= 1


# --- rules and the model-facing surfaces -------------------------------------------------------


def test_rules_render_from_evidence_and_share_identical_actions():
    adv = Advisor(
        [
            _entry(action="reason.solve", task={"goal": "opt"}),
            _entry(action="reason.check", task={"goal": "opt"}),
            _entry(
                action="reason.solve",
                level="discouraged",
                task={"n_atoms": {"max": 6}, "goal": "sat"},
            ),
            _entry(
                action="reason.check",
                level="discouraged",
                task={"n_atoms": {"max": 6}, "goal": "sat"},
            ),
            _entry(action="reason.solve", task={"goal": "x"}, status="pending"),
            _entry(action="reason.solve", level="neutral", task={"goal": "y"}),
        ]
    )
    assert adv.rules(SMALL) == {"reason.check,reason.solve": "dis:n_atoms<=6&goal=sat rec:goal=opt"}
    # large models: no reason evidence; the model-class "any" memory rule applies to everyone
    assert Advisor.default().rules(Profile(params_b=70)) == {
        "memory.recall": "rec:facts_exceed_ctx=true"
    }


def _call(registry, cache, action, params=None, verbosity="summary", session="s"):
    out = handle_call(
        registry, cache, "meta", action, params, verbosity=verbosity, session_id=session
    )
    return json.loads(out)


def test_meta_advise_returns_the_level_for_the_task():
    registry, cache = build_registry(), SessionCache()
    _call(registry, cache, "handshake", {"model": "m", "params_b": 4, "thinking": False})

    def ask(t):
        return _call(registry, cache, "advise", {"domain": "reason", "action": "solve", "task": t})

    assert ask({"n_atoms": 2, "goal": "sat"})["advisory"]["level"] == "discouraged"
    assert ask({"n_atoms": 12, "goal": "sat"})["advisory"]["level"] == "recommended"
    assert ask({"goal": "opt"})["advisory"]["level"] == "recommended"
    assert ask({"bogus": 1})["status"] == "ERROR"
    assert (
        _call(registry, cache, "advise", {"domain": "meta", "action": "stats"})["status"] == "ERROR"
    )
    assert _call(registry, cache, "advise", {"domain": "reason"})["status"] == "ERROR"


def test_budgets_hold_with_rules():
    registry, cache = build_registry(), SessionCache()
    _call(registry, cache, "handshake", {"model": "gemma", "params_b": 4, "thinking": False})
    brief = _call(registry, cache, "schema", verbosity="brief")
    terse = _call(registry, cache, "schema", verbosity="terse")
    assert len(json.dumps(brief).encode()) <= 8000
    assert len(json.dumps(terse).encode()) <= 1500
    assert "rules" in brief and "rules" in terse
    disc = _call(registry, cache, "discover", verbosity="terse")
    assert len(json.dumps(disc).encode()) <= 1500 and "rules" in disc


def test_tool_description_carries_the_rules_without_a_handshake_call():
    async def description(declare: bool) -> str:
        async with ChrysalisSession(None) as cs:
            if declare:
                await cs.call(domain="meta", action="handshake", params=_WHO)
            return (await cs.openai_tool())["function"]["description"]

    declared = _run(description(True))
    assert "Advisory reason.check,reason.solve: dis:n_atoms<=6&goal=sat" in declared
    undeclared = _run(description(False))  # unknown model: no model-class claim (reason)
    assert "reason." not in undeclared


def test_advised_arm_declares_the_model_so_the_model_need_not_handshake():
    puzzle = HARD_PUZZLES[0]
    with FakeLlm() as fake:
        fake.script = [reply_text(json.dumps(puzzle.answer))]  # never calls the tool
        llm = LlmClient(base_url=fake.url, model="gemma-fake", timeout_s=10, thinking=False)
        out = _run(eval_llm._group("adv", llm, 64, 1, 4.0, "."))
        tool = fake.requests[0]["tools"][0]["function"]["description"]
    assert out["advised"]["handshake_first_rate"] == 0.0
    assert "rec:" in tool and "dis:" in tool


# --- B: calibration ---------------------------------------------------------------------------


class _HungryThinker(_ProbeSolver):
    """Thinking-model fingerprint: schema first, re-asks for probes, decorated answer keys."""

    def __init__(self) -> None:
        super().__init__(True)
        self.step = 0

    def _tool(self, **args: Any) -> dict[str, Any]:
        call = {"id": f"c{self.step}", "type": "function",
                "function": {"name": "chrysalis", "arguments": json.dumps(args)}}  # fmt: skip
        return {"status": "OK", "message": {"content": None, "tool_calls": [call]}, "usage": _USAGE}

    def chat(self, messages, tools=None, **_):
        self.step += 1
        if self.step == 1:
            return self._tool(domain="meta", action="schema", verbosity="terse")
        if self.step == 2:  # asks again: must not invalidate the probes already in the prompt
            return self._tool(domain="meta", action="calibrate")
        if self.step == 3:
            prompt = messages[1]["content"]
            answers = {f"{k} (reason)" if k.startswith("reason") else k: v
                       for k, v in self._answers(prompt).items()}  # fmt: skip
            return self._tool(domain="meta", action="calibrate", params={"answers": answers})
        return {"status": "OK", "message": {"content": "done"}, "usage": _USAGE}


def test_calibration_survives_a_step_hungry_thinking_model():
    out = _run(eval_llm.calibrate_once(_HungryThinker(), 64, _WHO))
    assert out["scores"] == {"reason": 1.0, "memory": 1.0}, out
    assert out["actions"] == ["meta.schema", "meta.calibrate", "meta.calibrate"]


def test_calibrate_reissue_is_idempotent_unless_asked():
    store = ProfileStore()
    first = store.issue("s")
    assert store.issue("s") is first
    assert store.issue("s", fresh=True) is not first
    registry, cache = build_registry(), SessionCache()
    a = _call(registry, cache, "calibrate")["probes"]
    assert _call(registry, cache, "calibrate")["probes"] == a
    assert _call(registry, cache, "calibrate", {"reissue": True})["probes"] != a


def test_unknown_probe_ids_get_the_valid_keys_back():
    registry, cache = build_registry(), SessionCache()
    _call(registry, cache, "calibrate")
    out = _call(registry, cache, "calibrate", {"answers": {"nope": "x"}})
    assert out["status"] == "ERROR" and "reason-1" in out["error"] and "reason-1" in out["valid"]


def test_answer_keys_and_equality_spelling_are_tolerated():
    store = ProfileStore()
    issued = store.issue("s")
    # whatever the nonce, a decorated key and a lone '=' must score like the canonical form
    ids = sorted(issued.expected)
    assert score({f"Probe {i}:": "?" for i in ids}, issued)[1] == []  # keys resolved to ids
    from chrysalis_mcp.core import profile as prof

    built = {pid: ans for pid, _, _, ans in prof._build("n1")}
    probes = prof.issue_probes("n1")
    sloppy = {
        "reason-1": [re.sub("==", "=", built["reason-1"][0])],
        "reason-2 (reason)": list(built["reason-2"]),
    }
    assert score(sloppy, probes)[0] == {"reason": round(2 / 3, 4)}


def test_reasoning_only_replies_end_calibration_cleanly():
    with FakeLlm() as fake:
        fake.script = [reply_text("", reasoning="long thought, no answer")]
        llm = LlmClient(base_url=fake.url, model="fake", timeout_s=10, thinking=True)
        out = _run(eval_llm.calibrate_once(llm, 64, _WHO))
        sent = fake.requests[0]["max_tokens"]
    assert out["scores"] == {} and out["actions"] == [] and out["submitted"] is False
    assert sent == 64 * eval_llm.CAL_TOKEN_FACTOR


def test_reason_probes_state_the_dsl_syntax():
    probes = ProfileStore().issue("s").public
    text = " ".join(p["prompt"] for p in probes if p["family"] == "reason")
    assert "==" in text and "min:<var>" in text


# --- D: H-a-scale -----------------------------------------------------------------------------


def test_scaled_qa_is_deterministic_and_exceeds_the_window():
    a, b = scaled_qa(), scaled_qa()
    assert a == b and scaled_qa(seed=99) != a
    assert len(a.facts) == 2 * 400 and len(a.items) == 20
    assert len({i.question for i in a.items}) == 20
    assert sum(len(f) + 1 for f in a.facts) / 2.5 > 8192  # more than the whole window
    names = {s for s, _, _ in a.triples}
    assert len(names) >= 400  # fictional, unique subjects
    with pytest.raises(ValueError):
        scaled_qa(n_chains=5, n_questions=6)


def test_fit_context_keeps_the_tail_and_accounts_for_the_drop():
    data = scaled_qa()
    kept, dropped = fit_context(data.facts, 8192 - 1024)
    assert kept == list(data.facts[len(data.facts) - len(kept) :])
    assert sum(len(f) + 1 for f in kept) <= (8192 - 1024) * 2.5
    assert dropped == pytest.approx(1 - len(kept) / len(data.facts), abs=1e-4) and 0 < dropped < 1
    assert 0 <= answerable_in(data.items, kept) < 1
    assert (
        fit_context(data.facts, 10**9)[1] == 0.0
        and answerable_in(data.items, list(data.facts)) == 1.0
    )
    assert fit_context((), 100) == ([], 0.0)


def test_scale_group_runs_both_arms_and_records_the_truncation(tmp_path):
    item = scaled_qa().items[0]
    with FakeLlm() as fake:
        fake.script = [reply_text(item.answer), reply_text(item.answer)]  # context, then memory
        llm = LlmClient(base_url=fake.url, model="fake", timeout_s=10, thinking=False)
        out = _run(eval_llm._group("hascale", llm, 64, 1, 4.0, str(tmp_path)))
        system = fake.requests[0]["messages"][0]["content"]
    ctx = out["context"]
    assert (
        ctx["total_facts"] == 800
        and ctx["facts_in_context"] < 800
        and 0 < ctx["dropped_fraction"] < 1
    )
    assert ctx["accuracy"] == 1.0 and "tools" not in fake.requests[0]
    assert system.count("\n") == ctx["facts_in_context"]  # header line + n-1 separators
    assert "tools" in fake.requests[1] and out["memory"]["n"] == 1


def test_scale_group_runs_in_thinking_off_only():
    assert "hascale" in eval_llm.ALL_GROUPS and "hascale" in eval_llm.LONG_GROUPS
    assert eval_llm.GROUP_NAMES["hascale"] == "H-a-scale"


def _scale(mem: float, ctx: float) -> dict[str, Any]:
    arm = {"n": 20, "mean_prompt_tokens": 100.0}
    return {
        "status": "OK",
        "modes": {"off": {"H-a-scale": {
            "memory": {**arm, "accuracy": mem},
            "context": {**arm, "accuracy": ctx, "total_facts": 800, "facts_in_context": 266,
                        "dropped_fraction": 0.6675, "answerable_in_context": 0.15},
        }}},
    }  # fmt: skip


def test_scale_verdict_needs_a_gain_of_at_least_a_tenth():
    assert report.verdict_scale(_scale(0.6, 0.45))[0] == report.SUPPORTED
    assert report.verdict_scale(_scale(0.5, 0.45))[0] == report.NOT_SUPPORTED
    assert report.verdict_scale(None)[0] == report.INCONCLUSIVE
    assert report.verdict_scale({"status": "OK", "modes": {"off": {}}})[0] == report.INCONCLUSIVE
    assert "266/800" in report.verdict_scale(_scale(0.6, 0.45))[1]


def test_facts_exceed_ctx_lookup_is_recommended_for_recall_and_derived_from_facts():
    adv = Advisor.default()
    wide = Profile(params_b=4, context=8192)
    task = task_features.derive({"facts": 800}, wide.context)
    assert task["facts_exceed_ctx"] is True
    assert adv.lookup("memory.recall", wide, task=task)["level"] == "recommended"
    assert adv.lookup("memory.recall", wide)["level"] == "recommended"  # model level (H-a)
