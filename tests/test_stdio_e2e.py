"""End-to-end over real stdio with the official MCP client; asserts on what the CLIENT receives."""

import json
import os
import sys

import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


def _params(db: str | None = None) -> StdioServerParameters:
    env = {**os.environ, **({"CHRYSALIS_DB": db} if db else {})}
    return StdioServerParameters(
        command=sys.executable, args=["-m", "chrysalis_mcp.server"], env=env
    )


async def _call(session: ClientSession, **args):
    return await session.call_tool("chrysalis", args)


async def test_stdio_roundtrip_single_encoding():
    async with stdio_client(_params()) as (r, w), ClientSession(r, w) as s:
        await s.initialize()
        tools = (await s.list_tools()).tools
        assert [t.name for t in tools] == ["chrysalis"]

        res = await _call(s, domain="meta", action="schema", verbosity="terse")
        assert not res.isError
        # CLAUDE.md s4 anti-chatty: payload must arrive ONCE, as compact JSON, not duplicated
        # in structuredContent={"result": "<escaped json>"} plus TextContent.
        assert res.structuredContent is None, "payload duplicated in structuredContent"
        assert len(res.content) == 1
        body = json.loads(res.content[0].text)
        assert body["status"] == "OK" and body["domains"]["meta"]
        assert not isinstance(body.get("result"), str)

        res = await _call(s, domain="meta", action="schema")
        schema = json.loads(res.content[0].text)
        assert "meta" in schema["domains"]

        batch = {
            "steps": [
                {"id": "a", "domain": "meta", "action": "discover", "store_as": "d"},
                {
                    "id": "b",
                    "domain": "meta",
                    "action": "discover",
                    "params": {"domain": "$cache.d.domains.meta.schema.description"},
                },
            ],
            "return_only": "summary",
        }
        res = await _call(s, domain="meta", action="batch", params=batch, verbosity="brief")
        out = json.loads(res.content[0].text)
        assert out["steps_run"] == 2
        assert [x["id"] for x in out["results"]] == ["a", "b"]


async def test_stdio_missing_cache_ref_and_bad_inputs():
    async with stdio_client(_params()) as (r, w), ClientSession(r, w) as s:
        await s.initialize()
        miss = {
            "steps": [
                {"domain": "meta", "action": "discover", "params": {"domain": "$cache.nope.x"}}
            ]
        }
        out = json.loads(
            (await _call(s, domain="meta", action="batch", params=miss)).content[0].text
        )
        assert out["status"] == "ERROR" and "unresolved" in out["failed_steps"][0]["error"]

        out = json.loads(
            (await _call(s, domain="meta", action="schema", verbosity="loud")).content[0].text
        )
        assert out["status"] == "ERROR" and "valid" in out

        out = json.loads(
            (await _call(s, domain="meta", action="batch", params={"steps": "x"})).content[0].text
        )
        assert out["status"] == "ERROR"


async def test_stdio_reason_rejected_command_then_normal_call(tmp_path):
    target = tmp_path / "trace.txt"
    bad = f'(set-option :trace_file_name "{target}")(set-option :memory_max_size 1)'
    good = "(declare-const x Int)(assert (> x 3))"
    async with stdio_client(_params()) as (r, w), ClientSession(r, w) as s:
        await s.initialize()
        res = await _call(s, domain="reason", action="solve", params={"smt2": bad + good})
        out = json.loads(res.content[0].text)
        assert out["status"] == "ERROR" and "allowed" in out
        assert not target.exists()
        for _ in range(2):  # server alive, worker healthy
            res = await _call(s, domain="reason", action="solve", params={"smt2": good})
            out = json.loads(res.content[0].text)
            assert out["status"] == "OK" and out["result"] == "sat"
        assert (await s.list_tools()).tools[0].name == "chrysalis"


async def test_stdio_memory_batch_ingest_recall(tmp_path):
    triples = [
        {"s": "París", "r": "capital_de", "o": "Francia"},
        {"s": "paris", "r": "es", "o": "ciudad"},
        {"s": "mitocondria", "r": "produce", "o": "ATP"},
    ]
    async with stdio_client(_params(str(tmp_path / "e2e.db"))) as (r, w), ClientSession(r, w) as s:
        await s.initialize()
        assert [t.name for t in (await s.list_tools()).tools] == ["chrysalis"]
        batch = {
            "steps": [
                {
                    "id": "i",
                    "domain": "memory",
                    "action": "ingest",
                    "params": {"triples": triples},
                    "store_as": "ing",
                },
                {
                    "id": "q",
                    "domain": "memory",
                    "action": "recall",
                    "params": {"query": "parís", "k": 2, "mode": "seeds"},
                },
            ],
            "return_only": "last",
        }
        res = await _call(s, domain="meta", action="batch", params=batch, verbosity="brief")
        assert not res.isError and res.structuredContent is None and len(res.content) == 1
        out = json.loads(res.content[0].text)
        assert out["status"] == "OK" and out["steps_run"] == 2
        assert {x["text"] for x in out["results"]} == {"París", "paris"}
        schema = json.loads((await _call(s, domain="meta", action="schema")).content[0].text)
        assert set(schema["domains"]["memory"]) == {
            "ingest",
            "recall",
            "update",
            "forget",
            "consolidate",
            "restore",
        }


async def test_stdio_memory_ppr_update_batch(tmp_path):
    triples = [
        {"s": "zorblax", "r": "founded", "o": "quuxcorp"},
        {"s": "quuxcorp", "r": "located_in", "o": "wibbleton"},
        {"s": "mitocondria", "r": "produce", "o": "ATP"},
    ]
    q = {"query": "zorblax", "k": 6}
    steps = [
        {"id": "i", "domain": "memory", "action": "ingest", "params": {"triples": triples}},
        {"id": "r1", "domain": "memory", "action": "recall", "params": q},
        {
            "id": "u",
            "domain": "memory",
            "action": "update",
            "params": {"query": "zorblax", "node": "ATP"},
        },
        {"id": "r2", "domain": "memory", "action": "recall", "params": q},
    ]
    async with stdio_client(_params(str(tmp_path / "e2e2.db"))) as (r, w), ClientSession(r, w) as s:
        await s.initialize()
        batch = {"steps": steps, "return_only": "all"}
        res = await _call(s, domain="meta", action="batch", params=batch, verbosity="brief")
        assert not res.isError and len(res.content) == 1
        out = json.loads(res.content[0].text)
        assert out["status"] == "OK", out
        assert "wibbleton" in json.dumps(out)
        assert "ATP" in json.dumps(out)


async def test_stdio_discover_not_stale_and_agent_stats(tmp_path):
    async with stdio_client(_params(str(tmp_path / "v.db"))) as (r, w), ClientSession(r, w) as s:
        await s.initialize()
        res = await _call(s, domain="meta", action="discover", verbosity="terse")
        assert res.structuredContent is None and len(res.content) == 1
        body = json.loads(res.content[0].text)
        assert body["stale"] is False and body["schema_hash"] and body["loaded_commit"]
        batch = {
            "steps": [
                {"domain": "meta", "action": "schema"},
                {"domain": "meta", "action": "stats", "params": {"group_by": "agent"}},
            ],
            "return_only": "last",
        }
        res = await _call(s, domain="meta", action="batch", params=batch, agent_id="e2e-agent")
        out = json.loads(res.content[0].text)
        assert out["agents"]["e2e-agent"]["calls"] >= 2 and "anonymous" in out["agents"]
