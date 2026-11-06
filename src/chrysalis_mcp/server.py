"""FastMCP app exposing ONE tool: ``chrysalis(domain, action, ...)``."""

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
    "meta.batch. verbosity: detail|summary|brief|terse."
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


def handle_call(
    registry: Registry,
    cache: SessionCache,
    domain: str,
    action: str,
    params: dict[str, Any] | None = None,
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
            payload = registry.dispatch(domain, action, params, ctx)
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
        params: dict[str, Any] | None = None,
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
