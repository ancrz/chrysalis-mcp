import json

from chrysalis_mcp.core.registry import Context, Registry
from chrysalis_mcp.core.session import SessionCache
from chrysalis_mcp.domains import build_registry
from chrysalis_mcp.server import create_app, handle_call


def test_invalid_action_returns_alternatives(call):
    out = call("meta", "bogus")
    assert out["status"] == "ERROR"
    assert out["valid"] == [
        "advise",
        "batch",
        "calibrate",
        "discover",
        "handshake",
        "schema",
        "stats",
    ]


def test_invalid_domain_returns_alternatives(call):
    out = call("nope", "x")
    assert out["status"] == "ERROR"
    assert "meta" in out["valid"]


def test_param_validation(call):
    out = call("meta", "batch", params={})
    assert out["missing"] == ["steps"]
    out = call("meta", "schema", params={"zzz": 1})
    assert out["unknown"] == ["zzz"]


def test_schema_lists_exactly_registered_handlers_without_server_edit():
    registry = build_registry()
    cache = SessionCache()

    def hello(params, ctx):
        return {"status": "OK", "greeting": "hi"}

    registry.register("dummy", "hello", hello, description="says hi")
    # A server built around this registry sees the new domain with no server.py change.
    create_app(registry, cache)
    out = json.loads(handle_call(registry, cache, "meta", "schema"))
    assert {"meta", "dummy"} <= set(out["domains"])
    assert set(out["domains"]["meta"]) == set(registry.actions("meta"))
    assert set(out["domains"]["dummy"]) == {"hello"}
    assert json.loads(handle_call(registry, cache, "dummy", "hello"))["greeting"] == "hi"


def test_schema_is_generated_from_registry():
    registry = Registry()
    registry.register("a", "x", lambda p, c: {}, params_schema={"q": {"required": True}})
    assert registry.schema()["domains"] == {
        "a": {
            "x": {
                "description": "",
                "params": {"q": {"required": True}},
                "example": {"domain": "a", "action": "x", "params": {}},
            }
        }
    }


def test_handler_exception_becomes_error():
    registry = Registry()

    def boom(params, ctx):
        raise RuntimeError("bad")

    registry.register("a", "boom", boom)
    out = registry.dispatch("a", "boom", {}, Context(None, SessionCache(), registry))
    assert out["status"] == "ERROR"
    assert "RuntimeError: bad" in out["error"]


def test_duplicate_registration_rejected():
    registry = build_registry()
    try:
        registry.register("meta", "schema", lambda p, c: {})
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_discover_terse_is_names_only(call):
    out = call("meta", "discover", verbosity="terse")
    # terse caps lists at 5 (truncated=True); calibrate (one-off, named by handshake) is hidden
    assert out["domains"]["meta"] == ["batch", "discover", "schema", "stats", "handshake"]
    assert out["truncated"] is True
