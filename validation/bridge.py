"""Gemma <-> Chrysalis agent loop over real stdio MCP (official client)."""

import json
import os
import sys
import time
from contextlib import AsyncExitStack
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from chrysalis_mcp.core.advisory import render_advisory_line
from validation.llm_client import LlmClient

TOOL_NAME = "chrysalis"
DEFAULT_MAX_STEPS = 8
MAX_ERR_CHARS = 200
# Declared identity and advisory mode come from the model handshake, never from shell leftovers.
_UNINHERITED_ENV = ("CHRYSALIS_ENFORCE_ADVISORY", "CHRYSALIS_MODEL_PROFILE")


class ChrysalisSession:
    """A running chrysalis-mcp over stdio. ``call`` returns the text the client received."""

    def __init__(self, db_path: str | None = None) -> None:
        env = {k: v for k, v in os.environ.items() if k not in _UNINHERITED_ENV}
        env.update({"CHRYSALIS_DB": db_path} if db_path else {})
        self._params = StdioServerParameters(
            command=sys.executable, args=["-m", "chrysalis_mcp.server"], env=env
        )
        self._stack = AsyncExitStack()
        self._session: ClientSession | None = None
        self.tool_def: dict[str, Any] = {}
        self.instructions: str = ""  # server ``instructions`` from the initialize result

    async def __aenter__(self) -> ChrysalisSession:
        read, write = await self._stack.enter_async_context(stdio_client(self._params))
        self._session = await self._stack.enter_async_context(ClientSession(read, write))
        init = await self._session.initialize()  # InitializeResult
        self.instructions = init.instructions or ""
        tools = (await self._session.list_tools()).tools
        tool = next(t for t in tools if t.name == TOOL_NAME)
        self.tool_def = {
            "name": tool.name,
            "description": tool.description or "",
            "inputSchema": tool.inputSchema,
        }
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self._stack.aclose()

    async def call(self, /, **args: Any) -> str:
        assert self._session is not None
        res = await self._session.call_tool(TOOL_NAME, args)
        return "".join(getattr(c, "text", "") for c in res.content)

    async def openai_tool(self) -> dict[str, Any]:
        """The single tool in OpenAI format; description enriched from ``meta.schema`` (terse)."""
        schema = json.loads(await self.call(domain="meta", action="schema", verbosity="terse"))
        catalog = "; ".join(
            f"{dom}: {', '.join(sorted(actions))}"  # terse: list of names; iterating a dict too
            for dom, actions in schema.get("domains", {}).items()
        )
        # Conditional advisory rules (evidence-based, per declared profile) ride in the description:
        # they reach the model whether or not it ever calls handshake.
        # The server description already carries the startup-profile lines: add only the others.
        base = self.tool_def["description"]
        hints = schema.get("hints") or {}
        rules = "".join(
            line
            for key, rule in (schema.get("rules") or {}).items()
            if (line := render_advisory_line({key: rule}, hints)) not in base
        )
        return {
            "type": "function",
            "function": {
                "name": TOOL_NAME,
                "description": f"{self.tool_def['description']} Domains/actions - {catalog}."
                f"{rules} Call domain='meta', action='schema' for exact parameters.",
                "parameters": self.tool_def["inputSchema"],
            },
        }


def _new_metrics() -> dict[str, Any]:
    return {
        "llm_prompt_tokens": 0,
        "llm_completion_tokens": 0,
        "llm_calls": 0,
        "reasoning_chars": 0,
        "tool_calls": 0,
        "malformed_tool_calls": 0,
        "tool_bytes": 0,
        "wall_s": 0.0,
    }


async def run_agent(
    llm: LlmClient,
    question: str,
    chrysalis: ChrysalisSession | None,
    system: str | None = None,
    max_steps: int = DEFAULT_MAX_STEPS,
    max_tokens: int | None = None,
) -> dict[str, Any]:
    """Loop tool_calls -> chrysalis -> tool results until a final answer or ``max_steps``.

    Returns ``{"status": OK|EMPTY|MAX_STEPS|UNREACHABLE|ERROR, "answer", "metrics", ...}``;
    never raises for model failures. Without ``chrysalis`` the model runs with no tools
    (baseline arm).
    """
    metrics = _new_metrics()
    actions: list[str] = []  # "domain.action" of every well-formed call, in order
    started = time.perf_counter()
    tools = [await chrysalis.openai_tool()] if chrysalis else None
    messages: list[dict[str, Any]] = ([{"role": "system", "content": system}] if system else []) + [
        {"role": "user", "content": question}
    ]

    def done(status: str, answer: str | None = None, **extra: Any) -> dict[str, Any]:
        metrics["wall_s"] = round(time.perf_counter() - started, 3)
        return {
            "status": status,
            "answer": answer,
            "metrics": metrics,
            "actions": actions,
            **extra,
        }

    for _ in range(max_steps):
        reply = llm.chat(
            messages, tools=tools, **({"max_tokens": max_tokens} if max_tokens else {})
        )
        metrics["llm_calls"] += 1
        if reply["status"] != "OK":
            return done(reply["status"], error=reply.get("error"))
        metrics["llm_prompt_tokens"] += reply["usage"]["prompt_tokens"]
        metrics["llm_completion_tokens"] += reply["usage"]["completion_tokens"]
        metrics["reasoning_chars"] += reply.get("reasoning_chars", 0)
        message = reply["message"]
        calls = message.get("tool_calls") or []
        if not calls or chrysalis is None:
            text = (message.get("content") or "").strip()
            # an empty final message (e.g. thinking consumed the budget) is not an answer
            return done("OK" if text else "EMPTY", text)
        messages.append(
            {"role": "assistant", "content": message.get("content") or "", "tool_calls": calls}
        )
        for call in calls:
            text = await _dispatch(chrysalis, call, metrics, actions)
            metrics["tool_bytes"] += len(text.encode("utf-8"))
            messages.append({"role": "tool", "tool_call_id": call.get("id", ""), "content": text})
    return done("MAX_STEPS")


async def _dispatch(
    chrysalis: ChrysalisSession,
    call: dict[str, Any],
    metrics: dict[str, Any],
    actions: list[str],
) -> str:
    metrics["tool_calls"] += 1
    fn = call.get("function") or {}
    if fn.get("name") != TOOL_NAME:
        metrics["malformed_tool_calls"] += 1
        return json.dumps({"status": "ERROR", "error": f"unknown tool {fn.get('name')!r}"})
    try:
        args = json.loads(fn.get("arguments") or "{}")
        if not isinstance(args, dict):
            raise ValueError("arguments must be a JSON object")
    except ValueError as exc:
        metrics["malformed_tool_calls"] += 1
        return json.dumps(
            {"status": "ERROR", "error": f"malformed tool arguments: {str(exc)[:MAX_ERR_CHARS]}"}
        )
    actions.append(f"{args.get('domain')}.{args.get('action')}")
    return await chrysalis.call(**args)
