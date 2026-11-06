"""H-mcp: tokens of a composed task (ingest -> recall -> update -> recall) over real stdio.

Cost model (what a tool-calling LLM actually pays in context): every LLM request carries the whole
tool list, so  tokens = est(list_bytes * requests + request_bytes + response_bytes)  with
requests = tool_calls + 1 (the final answer). Compared designs:
  batch       one `meta.batch` call on the single `chrysalis` tool (measured on a real server)
  sequential  four calls on the single `chrysalis` tool (measured)
  granular    SIMULATED: four calls on a one-tool-per-action server (tool list built from
              ``meta.schema``; responses reuse the sequential ones since the payloads are equal)
The single-tool list size is that of the enriched OpenAI-format tool the LLM harness really sends
(``ChrysalisSession.openai_tool``); simulated granular tools use the same wrapper.
"""

import asyncio
import json
import tempfile
from pathlib import Path
from typing import Any

from validation import RESULTS_DIR, est_tokens
from validation.bridge import ChrysalisSession

RESULT_FILE = "hmcp.json"
JSON_TYPES = {"string", "integer", "number", "boolean", "object", "array"}
_TRIPLES = [
    {"s": "Ana", "r": "works_at", "o": "Bolt Labs"},
    {"s": "Bolt Labs", "r": "located_in", "o": "Quito"},
    {"s": "Quito", "r": "lies_in", "o": "Ecuador"},
]
_QUERY = "where does Ana work"


def _steps() -> list[tuple[str, str, dict[str, Any]]]:
    return [
        ("memory", "ingest", {"triples": _TRIPLES}),
        ("memory", "recall", {"query": _QUERY, "k": 3}),
        ("memory", "update", {"query": _QUERY, "node": "Bolt Labs"}),
        ("memory", "recall", {"query": _QUERY, "k": 3}),
    ]


def _size(obj: Any) -> int:
    return len(json.dumps(obj, separators=(",", ":")).encode("utf-8"))


def _granular_tools(schema: dict[str, Any]) -> list[dict[str, Any]]:
    """One OpenAI-format function per domain.action, same wrapper as ``openai_tool``."""
    tools = []
    for domain, actions in schema["domains"].items():
        for action, spec in actions.items():
            params = spec.get("params", {})
            props = {
                name: {"type": p["type"] if p.get("type") in JSON_TYPES else "string"}
                for name, p in params.items()
            }
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": f"{domain}_{action}",
                        "description": spec.get("description", ""),
                        "parameters": {
                            "type": "object",
                            "properties": props,
                            "required": [n for n, p in params.items() if p.get("required")],
                        },
                    },
                }
            )
    return tools


def _tokens(list_bytes: int, calls: int, req_bytes: int, resp_bytes: int) -> dict[str, int]:
    total = list_bytes * (calls + 1) + req_bytes + resp_bytes
    return {
        "tool_calls": calls,
        "tool_list_bytes": list_bytes,
        "request_bytes": req_bytes,
        "response_bytes": resp_bytes,
        "est_tokens": est_tokens(total),
    }


async def _measure() -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as tmp:
        async with ChrysalisSession(str(Path(tmp) / "seq.db")) as seq:
            schema = json.loads(await seq.call(domain="meta", action="schema"))
            list_bytes = _size(await seq.openai_tool())
            responses, requests = [], []
            for domain, action, params in _steps():
                args = {"domain": domain, "action": action, "params": params}
                requests.append(_size(args))
                responses.append(len((await seq.call(**args)).encode("utf-8")))
        async with ChrysalisSession(str(Path(tmp) / "batch.db")) as bat:
            batch_args = {
                "domain": "meta",
                "action": "batch",
                "params": {
                    "steps": [
                        {"id": f"s{i}", "domain": d, "action": a, "params": p}
                        for i, (d, a, p) in enumerate(_steps())
                    ],
                    "return_only": "last",
                },
            }
            batch_resp = len((await bat.call(**batch_args)).encode("utf-8"))
    granular_list = _size(_granular_tools(schema))
    granular_req = sum(_size({"name": f"{d}_{a}", "arguments": p}) for d, a, p in _steps())
    return {
        "batch": _tokens(list_bytes, 1, _size(batch_args), batch_resp),
        "sequential": _tokens(list_bytes, len(requests), sum(requests), sum(responses)),
        "granular": {
            **_tokens(granular_list, len(requests), granular_req, sum(responses)),
            "simulated": True,
        },
    }


def run() -> dict[str, Any]:
    return {
        "hypothesis": "H-mcp",
        "config": {"task": [f"{d}.{a}" for d, a, _ in _steps()], "bytes_per_token": 4},
        "data": asyncio.run(_measure()),
    }


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / RESULT_FILE).write_text(json.dumps(run(), indent=2))


if __name__ == "__main__":
    main()
