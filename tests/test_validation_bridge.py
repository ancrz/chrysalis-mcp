"""Gemma<->Chrysalis bridge against a fake OpenAI-compatible server (no live model needed)."""

import json

import pytest

from tests.fake_llm import FakeLlm, dead_url, reply_text, reply_tool
from validation.bridge import ChrysalisSession, run_agent
from validation.llm_client import LlmClient

pytestmark = [pytest.mark.anyio, pytest.mark.contract]


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
def fake():
    with FakeLlm() as server:
        yield server


@pytest.fixture
def llm(fake):
    return LlmClient(base_url=fake.url, model="fake", timeout_s=10)


async def test_a_tool_call_reaches_chrysalis_and_the_result_returns_to_the_model(
    fake, llm, tmp_path
):
    fake.script = [
        reply_tool({"domain": "meta", "action": "schema", "verbosity": "terse"}),
        reply_text("done"),
    ]
    async with ChrysalisSession(str(tmp_path / "b.db")) as cs:
        out = await run_agent(llm, "what can you do?", cs, system="be brief")
    assert (out["status"], out["answer"]) == ("OK", "done")
    first, second = fake.requests
    assert [t["function"]["name"] for t in first["tools"]] == ["chrysalis"]
    assert first["messages"][0] == {"role": "system", "content": "be brief"}
    tool_msg = second["messages"][-1]
    assert tool_msg["role"] == "tool" and tool_msg["tool_call_id"] == "call_1"
    assert json.loads(tool_msg["content"])["status"] == "OK"


async def test_usage_and_tool_traffic_are_accounted(fake, llm, tmp_path):
    fake.script = [
        reply_tool({"domain": "meta", "action": "schema"}, prompt=20, completion=8),
        reply_text("ok", prompt=30, completion=2),
    ]
    async with ChrysalisSession(str(tmp_path / "b.db")) as cs:
        m = (await run_agent(llm, "q", cs))["metrics"]
    assert (m["llm_prompt_tokens"], m["llm_completion_tokens"]) == (50, 10)
    assert (m["llm_calls"], m["tool_calls"], m["malformed_tool_calls"]) == (2, 1, 0)
    assert m["tool_bytes"] > 0 and m["wall_s"] >= 0


async def test_max_steps_stops_a_model_that_never_answers(fake, llm, tmp_path):
    fake.script = [reply_tool({"domain": "meta", "action": "schema"}) for _ in range(5)]
    async with ChrysalisSession(str(tmp_path / "b.db")) as cs:
        out = await run_agent(llm, "q", cs, max_steps=3)
    assert out["status"] == "MAX_STEPS" and out["answer"] is None
    assert out["metrics"]["llm_calls"] == 3


@pytest.mark.parametrize(
    "arguments, why",
    [
        ("{not json", "invalid JSON"),
        ('["a", "list"]', "not an object"),
    ],
)
async def test_malformed_tool_arguments_are_reported_to_the_model_not_raised(
    fake, llm, tmp_path, arguments, why
):
    fake.script = [reply_tool(arguments), reply_text("recovered")]
    async with ChrysalisSession(str(tmp_path / "b.db")) as cs:
        out = await run_agent(llm, "q", cs)
    assert out["answer"] == "recovered" and out["metrics"]["malformed_tool_calls"] == 1, why
    assert "malformed tool arguments" in fake.requests[1]["messages"][-1]["content"]


async def test_an_unknown_tool_name_is_an_error_result(fake, llm, tmp_path):
    fake.script = [reply_tool({}, name="rm_rf"), reply_text("fine")]
    async with ChrysalisSession(str(tmp_path / "b.db")) as cs:
        out = await run_agent(llm, "q", cs)
    assert out["metrics"]["malformed_tool_calls"] == 1
    assert "unknown tool" in fake.requests[1]["messages"][-1]["content"]


async def test_invalid_chrysalis_arguments_come_back_as_an_error_envelope(fake, llm, tmp_path):
    fake.script = [reply_tool({"domain": "nope", "action": "x"}), reply_text("ok")]
    async with ChrysalisSession(str(tmp_path / "b.db")) as cs:
        out = await run_agent(llm, "q", cs)
    assert out["status"] == "OK"
    assert json.loads(fake.requests[1]["messages"][-1]["content"])["status"] == "ERROR"


async def test_the_baseline_arm_sends_no_tools(fake, llm):
    fake.script = [reply_text("42")]
    out = await run_agent(llm, "q", None)
    assert out["answer"] == "42" and "tools" not in fake.requests[0]


async def test_an_unreachable_model_is_reported_not_raised(tmp_path):
    llm = LlmClient(base_url=dead_url(), model="fake", timeout_s=2)
    async with ChrysalisSession(str(tmp_path / "b.db")) as cs:
        out = await run_agent(llm, "q", cs)
    assert out["status"] == "UNREACHABLE" and out["answer"] is None


@pytest.mark.parametrize(
    "script, status",
    [
        ([500], "ERROR"),
        (['{"choices": []}'], "ERROR"),
        (["not json at all"], "ERROR"),
    ],
)
async def test_bad_server_replies_are_structured_errors(fake, llm, script, status):
    fake.script = list(script)
    assert (await run_agent(llm, "q", None))["status"] == status


def test_availability_probe(fake):
    assert LlmClient(base_url=fake.url).available() is True
    assert LlmClient(base_url=dead_url()).available() is False


async def test_thinking_off_sends_the_template_switch_and_on_omits_it(fake):
    fake.script = [reply_text("a"), reply_text("b")]
    LlmClient(base_url=fake.url, model="m", thinking=False).chat([{"role": "user", "content": "q"}])
    LlmClient(base_url=fake.url, model="m", thinking=True).chat([{"role": "user", "content": "q"}])
    assert fake.requests[0]["chat_template_kwargs"] == {"enable_thinking": False}
    assert "chat_template_kwargs" not in fake.requests[1]


async def test_thinking_default_comes_from_the_environment(monkeypatch):
    monkeypatch.delenv("CHRYSALIS_LLM_THINKING", raising=False)
    assert LlmClient(base_url="http://x/v1").thinking is False
    monkeypatch.setenv("CHRYSALIS_LLM_THINKING", "on")
    assert LlmClient(base_url="http://x/v1").thinking is True
    assert LlmClient(base_url="http://x/v1", thinking=False).thinking is False
    monkeypatch.setenv("CHRYSALIS_LLM_THINKING", "maybe")
    with pytest.raises(ValueError):
        LlmClient(base_url="http://x/v1")


async def test_reasoning_is_counted_and_never_fed_back(fake, llm, tmp_path):
    call = reply_tool({"domain": "meta", "action": "schema", "verbosity": "terse"})
    call["choices"][0]["message"]["reasoning_content"] = "xyz"
    fake.script = [call, reply_text("done", reasoning="abcde")]
    async with ChrysalisSession(str(tmp_path / "r.db")) as cs:
        out = await run_agent(llm, "q", cs, max_tokens=77)
    assert out["metrics"]["reasoning_chars"] == 3 + 5
    assert all(r["max_tokens"] == 77 for r in fake.requests)
    assert "reasoning_content" not in json.dumps(fake.requests[1]["messages"])
