"""T17: plain-language hints rendered from the rule conditions; conditional format reminder."""

import asyncio
import json
from typing import Any

import pytest

from chrysalis_mcp.core import task_features
from chrysalis_mcp.core.advisory import Advisor, hint_for_tags, hint_for_task
from chrysalis_mcp.core.profile import Profile
from chrysalis_mcp.core.session import SessionCache
from chrysalis_mcp.domains import build_registry
from chrysalis_mcp.server import handle_call
from tests.test_task_advisory import _entry
from tests.test_validation_t12 import _WHO
from validation import eval_llm
from validation.bridge import ChrysalisSession

SMALL = Profile(params_b=4, thinking=False)


@pytest.mark.parametrize(
    ("feature", "cond", "expected"),
    [
        ("n_atoms", {"min": 7}, "about 7+ comparisons, counting each lower/upper bound"),
        ("n_atoms", {"max": 6}, "about 6 or fewer comparisons counting bounds"),
        ("n_atoms", {"min": 3, "max": 6}, "about 3 to 6 comparisons counting bounds"),
        ("n_vars", {"min": 4}, "4+ variables"),
        ("n_asserts", {"max": 2}, "2 or fewer listed conditions"),
        ("goal", "opt", "asks for the smallest/largest value"),
        ("goal", "sat", "only asks for any valid values"),
        (
            "goal",
            ["sat", "opt"],
            "only asks for any valid values or asks for the smallest/largest value",
        ),
        ("facts_exceed_ctx", True, "the facts do not fit in your context window"),
        ("facts_exceed_ctx", False, "the facts fit in your context window"),
        ("facts", {"min": 100}, "100+ stored facts"),
        ("ctx_tokens", {"min": 1}, None),  # no phrasing: no hint, no crash
        ("bogus", "x", None),
        ("n_atoms", "weird", None),
        ("goal", "mystery", None),
    ],
)
def test_hint_text_table(feature, cond, expected):
    assert task_features.hint_text(feature, cond) == expected


def test_hint_combines_with_and_or_and_never_crashes_on_unknown():
    both = {"n_atoms": {"max": 6}, "goal": "sat"}
    assert hint_for_task(both) == (
        "about 6 or fewer comparisons counting bounds and only asks for any valid values"
    )
    assert hint_for_task({"n_atoms": {"max": 6}, "bogus": 1}) is None  # partial phrasing is a lie
    assert hint_for_task({}) is None
    tags = {"rec": [{"n_atoms": {"min": 7}}, {"goal": "opt"}, {"bogus": 1}], "dis": [both]}
    text = hint_for_tags(tags)
    assert text.startswith("not worth using, do it yourself, when about 6 or fewer")
    assert "worth using when about 7+ comparisons, counting each lower/upper bound or asks" in text
    assert hint_for_tags({"rec": [{"bogus": 1}]}) == ""


def test_hints_share_keys_with_rules_and_skip_unphrasable_rules():
    adv = Advisor(
        [
            _entry(action="reason.solve", task={"goal": "opt"}),
            _entry(action="reason.check", task={"goal": "opt"}),
            _entry(action="memory.recall", task={"bogus_feature": 1}),
        ]
    )
    assert adv.rules(SMALL) == {
        "memory.recall": "rec:bogus_feature=1",
        "reason.check,reason.solve": "rec:goal=opt",
    }
    assert adv.hints(SMALL) == {
        "reason.check,reason.solve": "worth using when asks for the smallest/largest value"
    }


def _call(registry, cache, action, params=None, verbosity="summary"):
    out = handle_call(registry, cache, "meta", action, params, verbosity=verbosity, session_id="s")
    return json.loads(out)


def test_hint_reaches_handshake_schema_discover_advise_and_budgets_hold():
    registry, cache = build_registry(), SessionCache()
    hs = _call(registry, cache, "handshake", {"model": "g", "params_b": 4, "thinking": False})
    assert hs["hints"] and set(hs["hints"]) == set(hs["rules"])
    for verbosity in ("summary", "brief", "terse"):
        for action in ("schema", "discover"):
            out = _call(registry, cache, action, verbosity=verbosity)
            assert out["hints"], (action, verbosity)
    for action in ("schema", "discover"):
        assert len(json.dumps(_call(registry, cache, action, verbosity="brief")).encode()) <= 8000
    assert len(json.dumps(_call(registry, cache, "schema", verbosity="terse")).encode()) <= 1500
    assert len(json.dumps(_call(registry, cache, "discover", verbosity="terse")).encode()) <= 1500
    adv = _call(registry, cache, "advise", {"domain": "reason", "action": "solve",
                                            "task": {"n_atoms": 12, "goal": "sat"}})  # fmt: skip
    assert "7+ comparisons" in adv["advisory"]["hint"]


def test_tool_description_carries_the_hint_for_a_small_thinking_off_profile():
    async def description() -> str:
        async with ChrysalisSession(None) as cs:
            await cs.call(domain="meta", action="handshake", params=_WHO)
            return (await cs.openai_tool())["function"]["description"]

    text = asyncio.run(description())
    assert "In words: not worth using, do it yourself, when about 6 or fewer comparisons" in text
    assert "worth using when" in text


def test_eval_prompt_starts_with_tool_prompt_and_ends_with_conditional_reminder():
    who: dict[str, Any] = {"model": "m", "family": "gemma", "params_b": 4.0, "thinking": False}
    s = eval_llm.advised_system(who)
    assert s.startswith(eval_llm.SYS_HARD_TOOL)
    assert s.endswith(eval_llm.SYS_FORMAT_REMINDER)
    assert eval_llm.SYS_FORMAT_REMINDER.strip().startswith("When you have the values")
    assert "In words" in s


def test_render_advisory_line_is_the_single_renderer():
    from chrysalis_mcp.core.advisory import render_advisory_line

    assert render_advisory_line({"a.b": "rec:goal=opt"}, {"a.b": "x"}) == (
        " Advisory a.b: rec:goal=opt. In words: x."
    )
    assert render_advisory_line({"a.b": "rec:goal=opt"}) == " Advisory a.b: rec:goal=opt."
    assert render_advisory_line({}) == ""


def test_bridge_description_lines_equal_the_renderer_output():
    from chrysalis_mcp.core.advisory import render_advisory_line

    async def go() -> tuple[str, dict[str, Any]]:
        async with ChrysalisSession(None) as cs:
            await cs.call(domain="meta", action="handshake", params=_WHO)
            schema = json.loads(await cs.call(domain="meta", action="schema", verbosity="terse"))
            return (await cs.openai_tool())["function"]["description"], schema

    text, schema = asyncio.run(go())
    for key, rule in schema["rules"].items():
        assert render_advisory_line({key: rule}, schema["hints"]) in text


def test_server_list_tools_carries_the_startup_profile_line(monkeypatch):
    from chrysalis_mcp.server import TOOL_DESCRIPTION, create_app

    async def desc() -> str:
        return (await create_app().list_tools())[0].description

    monkeypatch.delenv("CHRYSALIS_MODEL_PROFILE", raising=False)
    plain = asyncio.run(desc())
    assert plain.startswith(TOOL_DESCRIPTION) and "Advisory memory.recall:" in plain
    assert "reason." not in plain
    monkeypatch.setenv("CHRYSALIS_MODEL_PROFILE", json.dumps({"params_b": 4, "thinking": False}))
    forced = asyncio.run(desc())
    assert "Advisory reason.check,reason.solve:" in forced and "In words:" in forced
    assert len(forced.encode()) <= 1500


# --- T18: system-prompt channels ---------------------------------------------------------


def test_render_prompt_hint_empty_and_bounded():
    from chrysalis_mcp.core.advisory import render_prompt_hint

    assert render_prompt_hint({}) == ""
    assert render_prompt_hint({"a.b": "x"}) == "Tool advice: For a.b: x."
    hints = Advisor.default().hints(SMALL)
    assert len(render_prompt_hint(hints).encode()) <= 1000


def _instructions(monkeypatch, forced: str | None) -> str:
    from chrysalis_mcp.server import create_app

    if forced:
        monkeypatch.setenv("CHRYSALIS_MODEL_PROFILE", forced)
    else:
        monkeypatch.delenv("CHRYSALIS_MODEL_PROFILE", raising=False)
    text = create_app(build_registry(), SessionCache()).instructions
    assert text
    return text


def test_server_instructions_forced_and_unforced_profile(monkeypatch):
    plain = _instructions(monkeypatch, None)
    forced = _instructions(
        monkeypatch, json.dumps({"model": "m", "params_b": 4, "thinking": False})
    )
    assert "chrysalis(domain, action, params)" in plain and "memory.recall" in plain
    assert "reason.solve" not in plain
    assert "reason.solve" in forced and "do it yourself" in forced
    assert max(len(plain.encode()), len(forced.encode())) <= 1000


def test_handshake_prompt_hint_for_small_thinking_off_and_budgets():
    registry, cache = build_registry(), SessionCache()
    hs = _call(registry, cache, "handshake", {"model": "g", "params_b": 4, "thinking": False})
    assert "reason.solve" in hs["prompt_hint"] and "do it yourself" in hs["prompt_hint"]
    assert "6 or fewer comparisons" in hs["prompt_hint"]
    for verbosity in ("brief", "terse"):
        out = _call(registry, cache, "handshake", {"model": "g", "params_b": 4, "thinking": False},
                    verbosity=verbosity)  # fmt: skip
        assert "prompt_hint" not in out and out["hints"]  # clipped views never carry a cut hint
    assert len(json.dumps(_call(registry, cache, "handshake", {"model": "g", "params_b": 4},
                                verbosity="brief")).encode()) <= 8000  # fmt: skip
    assert len(json.dumps(_call(registry, cache, "handshake", {"model": "g", "params_b": 4},
                                verbosity="terse")).encode()) <= 1500  # fmt: skip


def test_bridge_reads_server_instructions(monkeypatch):
    monkeypatch.delenv("CHRYSALIS_MODEL_PROFILE", raising=False)

    async def read() -> str:
        async with ChrysalisSession(None) as cs:
            return cs.instructions

    assert "chrysalis(domain, action, params)" in asyncio.run(read())


def test_advised_prompt_order_with_prompt_hint():
    who: dict[str, Any] = {"model": "m", "family": "gemma", "params_b": 4.0, "thinking": False}
    hint = "Tool advice: For reason.solve: worth using when X."
    s = eval_llm.advised_system(who, hint)
    assert s.startswith(eval_llm.SYS_HARD_TOOL) and s.endswith(eval_llm.SYS_FORMAT_REMINDER)
    assert hint in s and s.index(hint) < s.index(eval_llm.SYS_FORMAT_REMINDER)
    assert hint not in eval_llm.advised_system(who)


def test_advised_arm_injects_handshake_prompt_hint(tmp_path):
    seen: list[str] = []

    class _Llm:
        model, thinking = "duck", False

        def chat(self, messages, **_):
            seen.append(messages[0]["content"])
            return {
                "status": "OK",
                "message": {"content": "{}"},
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            }

    arm = asyncio.run(
        eval_llm._arm(
            _Llm(),
            [("q", {})],
            lambda h: eval_llm.advised_system(_WHO, h),
            True,
            lambda a, g: True,
            None,
            None,
            64,
            _WHO,
        )  # fmt: skip
    )
    assert arm["prompt_hint_used"] is True
    assert "reason.solve" in seen[0] and seen[0].endswith(eval_llm.SYS_FORMAT_REMINDER)
