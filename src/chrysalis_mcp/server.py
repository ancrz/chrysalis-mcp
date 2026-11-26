"""FastMCP app exposing ONE tool: ``chrysalis(domain, action, ...)``."""

import ast
import json
from typing import Any

from mcp.server.fastmcp import FastMCP

from .core.advisory import render_advisory_line, render_prompt_hint
from .core.envelope import VERBOSITIES, Telemetry, build_response
from .core.registry import AGENT_ID_RE, ANONYMOUS, Context, Registry
from .core.session import SessionCache
from .domains import build_registry

TOOL_DESCRIPTION = (
    "Single dispatcher: chrysalis(domain, action, params). Call domain='meta', action='schema' "
    "for the full contract, or action='discover' for a compact map. Chain steps in one call with "
    "meta.batch. verbosity: detail|summary|brief|terse. Self-report: meta.handshake (declare "
    "model/family/params_b), meta.advise (should I use a tool for this task?), meta.feedback "
    "(record a tool outcome)."
)


def tool_description(registry: Registry) -> str:
    """Static base + the advisory rules of the startup profile.

    The profile is the forced ``CHRYSALIS_MODEL_PROFILE`` if set, else unknown (only the
    model-agnostic rules). Per-session profiles (``meta.handshake``/``calibrate``) are keyed by a
    tool argument, not by the MCP transport session, so ``list_tools`` cannot see them.
    """
    profile = registry.profiles.resolve(None)
    line = render_advisory_line(registry.advisor.rules(profile), registry.advisor.hints(profile))
    return TOOL_DESCRIPTION + line


INSTRUCTIONS_BASE = (
    "One tool, chrysalis(domain, action, params); call meta.schema for the contract, "
    "meta.batch to chain steps."
)


def server_instructions(registry: Registry) -> str:
    """MCP ``instructions`` (clients put them in the system prompt): dispatcher usage line plus
    the plain-language tool advice of the startup profile (forced profile, else model-agnostic)."""
    profile = registry.profiles.resolve(None)
    hint = render_prompt_hint(registry.advisor.hints(profile))
    return f"{INSTRUCTIONS_BASE} {hint}" if hint else INSTRUCTIONS_BASE


def _params_error(registry: Registry, domain: str, action: str, error: str) -> dict[str, Any]:
    out: dict[str, Any] = {
        "status": "ERROR",
        "error": error,
        "next_action": "Send params as a JSON object, e.g. copy the shape of 'example'.",
    }
    if (example := registry.example_call(domain, action)) is not None:
        out["example"] = example
    return out


PY_LITERAL_MAX_CHARS = 64 * 1024  # size cap before ast.literal_eval (also bounds its recursion)


def _jsonable(value: Any) -> Any:
    """Python literal -> JSON-shaped value (tuples become lists); ``ValueError`` otherwise."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        if not all(isinstance(k, str) for k in value):
            raise ValueError("non-string key")
        return {k: _jsonable(v) for k, v in value.items()}
    raise ValueError(f"unsupported literal {type(value).__name__}")


def decode_params_string(text: str) -> tuple[Any, str | None]:
    """``(value, note)``: JSON first, then a Python literal (tuples, single quotes, True/None).

    ``ast.literal_eval`` only evaluates literals (no names, calls or attributes) and the input is
    size-capped. ``(None, None)`` when neither parses to an object or list.
    """
    if len(text) > PY_LITERAL_MAX_CHARS:  # cap BEFORE any parser: json.loads recurses too
        return None, None
    try:
        decoded = json.loads(text)
    except (ValueError, RecursionError, MemoryError):  # fmt: skip
        pass
    else:
        if isinstance(decoded, (dict, list)):
            return decoded, "params_from_json_string"
        return None, None
    try:
        literal = _jsonable(ast.literal_eval(text.strip()))
    except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):  # fmt: skip
        return None, None
    if isinstance(literal, (dict, list)):
        return literal, "params_from_python_literal"
    return None, None


def coerce_params(
    registry: Registry, domain: str, action: str, params: Any
) -> tuple[dict[str, Any] | None, str | None, dict[str, Any] | None]:
    """``(params, note, error)``: small models send params as a JSON/Python-literal string or a
    bare list (a list is wrapped into the action's sole required array param)."""
    if params is None or isinstance(params, dict):
        return params, None, None
    note: str | None = None
    if isinstance(params, str):
        if not params.strip():
            return None, None, None
        params, note = decode_params_string(params)
        if params is None:
            return (
                None,
                None,
                _params_error(registry, domain, action, "params string must be a JSON object"),
            )
        if isinstance(params, dict):
            return params, note, None
    if isinstance(params, list):
        key = registry.sole_required_array(domain, action)
        if key is None:
            return (
                None,
                None,
                _params_error(registry, domain, action, "params must be an object, not a list"),
            )
        return {key: params}, (note or "params_from_list"), None
    return None, None, _params_error(registry, domain, action, "params must be an object")


def handle_call(
    registry: Registry,
    cache: SessionCache,
    domain: str,
    action: str,
    params: dict[str, Any] | str | list[Any] | None = None,
    verbosity: str = "summary",
    session_id: str | None = None,
    store_as: str | None = None,
    telemetry: bool | None = None,
    agent_id: str | None = None,
) -> str:
    tracker = Telemetry(domain, action)
    who = ANONYMOUS if agent_id is None else agent_id
    if not AGENT_ID_RE.match(who):
        payload: dict[str, Any] = {
            "status": "ERROR",
            "error": f"agent_id must match {AGENT_ID_RE.pattern}",
        }
        who = ANONYMOUS
    elif verbosity not in VERBOSITIES:
        payload = {
            "status": "ERROR",
            "error": f"unknown verbosity '{verbosity}'",
            "valid": list(VERBOSITIES),
        }
    else:
        ctx = Context(session_id, cache, registry, verbosity, who)
        if store_as is not None and (not store_as.strip() or "." in store_as):
            payload = {"status": "ERROR", "error": "store_as must be non-empty and contain no '.'"}
        else:
            fixed, note, bad = coerce_params(registry, domain, action, params)
            if bad is not None:
                payload = bad
            else:
                payload = registry.dispatch(domain, action, fixed, ctx)
                if note:
                    payload["coerced"] = [note, *payload.get("coerced", [])]
            if store_as and str(payload.get("status", "OK")) != "ERROR":
                cache.put(session_id, store_as.strip(), payload)
    text = build_response(payload, tracker, verbosity=verbosity, telemetry=telemetry)
    registry.add_bytes(session_id, domain, action, len(text.encode("utf-8")), who)
    return text


def create_app(registry: Registry | None = None, cache: SessionCache | None = None) -> FastMCP:
    registry = registry or build_registry()
    cache = cache or SessionCache()
    app = FastMCP("chrysalis", instructions=server_instructions(registry))

    # structured_output=False: the payload must reach the client once (TextContent), not duplicated
    # in structuredContent as an escaped string.
    @app.tool(name="chrysalis", description=tool_description(registry), structured_output=False)
    def chrysalis(
        domain: str,
        action: str,
        params: dict[str, Any] | str | list[Any] | None = None,
        verbosity: str = "summary",
        session_id: str | None = None,
        store_as: str | None = None,
        telemetry: bool | None = None,
        agent_id: str | None = None,
    ) -> str:
        return handle_call(
            registry,
            cache,
            domain,
            action,
            params,
            verbosity,
            session_id,
            store_as,
            telemetry,
            agent_id,
        )

    return app


def main() -> None:
    create_app().run(transport="stdio")


if __name__ == "__main__":
    main()
