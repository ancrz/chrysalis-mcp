import asyncio
import json

import pytest

from chrysalis_mcp.core.envelope import (
    MAX_SAFE_JSON_BYTES,
    Telemetry,
    build_response,
    infer_severity,
)
from chrysalis_mcp.core.session import SessionCache
from chrysalis_mcp.domains import build_registry
from chrysalis_mcp.server import create_app, handle_call
from tests.conftest import ENVELOPE_KEYS

VERBS = ["detail", "summary", "brief", "terse"]


@pytest.mark.parametrize("verbosity", VERBS)
def test_top_level_envelope_no_nested(call, verbosity):
    out = call("meta", "schema", verbosity=verbosity)
    assert out.keys() >= ENVELOPE_KEYS
    assert "envelope" not in out


def test_size_ordering(raw_call):
    size = {v: len(raw_call("meta", "schema", verbosity=v).encode()) for v in VERBS}
    assert size["terse"] <= size["brief"] < size["summary"] <= size["detail"]


@pytest.mark.parametrize("verbosity", ["brief", "terse"])
def test_no_telemetry_in_brief_terse_by_default(call, verbosity):
    assert "telemetry" not in call("meta", "schema", verbosity=verbosity)
    forced = call("meta", "schema", verbosity=verbosity, telemetry=True)
    assert {"request_id", "ms", "bytes", "est_tokens"} <= forced["telemetry"].keys()


@pytest.mark.parametrize("verbosity", ["detail", "summary"])
def test_telemetry_default_and_opt_out(call, verbosity):
    assert "telemetry" in call("meta", "schema", verbosity=verbosity)
    assert "telemetry" not in call("meta", "schema", verbosity=verbosity, telemetry=False)


def test_invalid_verbosity_lists_valid(call):
    out = call("meta", "schema", verbosity="loud")
    assert out["severity"] == "ERROR"
    assert out["valid"] == VERBS


def test_terse_keeps_error_details(call):
    out = call("meta", "nope", verbosity="terse")
    assert out["valid"]


def test_severity_inference():
    assert infer_severity("OK") == "INFO"
    assert infer_severity("PARTIAL") == "WARN"
    assert infer_severity("BOOM") == "ERROR"


def test_clamp_oversized_sets_truncated():
    payload = {"status": "OK", "items": list(range(200_000))}
    out = json.loads(build_response(payload, Telemetry("d", "a"), verbosity="summary"))
    assert out["truncated"] is True
    assert len(out["items"]) == 50
    assert out["items_total_count"] == 200_000
    assert len(json.dumps(out)) < MAX_SAFE_JSON_BYTES


def test_brief_caps_lists():
    payload = {"status": "OK", "items": list(range(30))}
    out = json.loads(build_response(payload, Telemetry("d", "a"), verbosity="brief"))
    assert len(out["items"]) == 10
    assert out["truncated"] is True


def test_server_exposes_exactly_one_tool():
    app = create_app(build_registry(), SessionCache())
    tools = asyncio.run(app.list_tools())
    assert [t.name for t in tools] == ["chrysalis"]


def test_handle_call_records_stats():
    registry, cache = build_registry(), SessionCache()
    handle_call(registry, cache, "meta", "schema", session_id="s1")
    stats = json.loads(handle_call(registry, cache, "meta", "stats", session_id="s1"))
    assert stats["actions"]["meta.schema"]["calls"] == 1
    assert stats["actions"]["meta.schema"]["bytes"] > 0


def test_clamp_recurses_into_nested_values():
    payload = {"status": "OK", "nest": {"z": "q" * 400_000}}
    text = build_response(payload, Telemetry("d", "a"), verbosity="brief")
    out = json.loads(text)
    assert len(text.encode()) <= MAX_SAFE_JSON_BYTES
    assert out["truncated"] is True


def test_clamp_replaces_largest_value_when_still_too_big():
    payload = {"status": "OK", "wide": {str(i): "x" * 30_000 for i in range(50)}}
    text = build_response(payload, Telemetry("d", "a"), verbosity="summary")
    out = json.loads(text)
    assert len(text.encode()) <= MAX_SAFE_JSON_BYTES
    assert out["truncated"] is True
    assert out["wide"]["_clamped"] is True


def test_top_level_store_as_validated(call):
    out = call("meta", "schema", store_as="a.b")
    assert out["severity"] == "ERROR"
    assert call("meta", "schema", store_as="  ")["severity"] == "ERROR"


def test_select_fields_must_be_list(call):
    out = call(
        "meta",
        "batch",
        params={"steps": [{"domain": "meta", "action": "discover"}], "select_fields": "status"},
    )
    assert out["status"] == "ERROR"
