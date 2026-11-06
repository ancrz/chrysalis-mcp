"""Typed response envelope, severity inference and payload clamping.

Adapted from agent-execution-plane ``engine/telemetry.py`` (Apache-2.0, see NOTICE).
Fixes GAP-OM-16: envelope fields are emitted once at the top level (no nested ``envelope``
duplicate) and telemetry is omitted in ``brief``/``terse`` unless explicitly requested.
"""

import json
import time
import uuid
from typing import Any

VERBOSITIES = ("detail", "summary", "brief", "terse")
MAX_SAFE_JSON_BYTES = 180_000
BRIEF_LIST_CAP = 10
BRIEF_STR_CAP = 256
TERSE_LIST_CAP = 5
TERSE_STR_CAP = 64
# error text and the advisory hints (bounded by construction) must stay readable at every verbosity
_UNCAPPED_KEYS = frozenset(("error", "hints"))
_CLAMP_LIST_CAP = 50
_CLAMP_STR_CAP = 32_000
_BYTES_PER_TOKEN = 4

_INFO = frozenset({"OK", "SUCCESS", "CREATED", "UPDATED", "DELETED", "CACHE_HIT", "CLEAN"})
_WARN = frozenset({"DEGRADED", "PARTIAL", "CACHE_MISS", "UNSUPPORTED", "HOLD", "DISCOURAGED"})


class Telemetry:
    """Wall-clock latency plus serialized size and estimated tokens for one request."""

    def __init__(self, domain: str, action: str) -> None:
        self.request_id = f"req_{uuid.uuid4().hex[:10]}"
        self.domain = domain
        self.action = action
        self._start_ns = time.perf_counter_ns()

    def finish(self, serialized_bytes: int) -> dict[str, Any]:
        elapsed_ms = round((time.perf_counter_ns() - self._start_ns) / 1_000_000, 2)
        return {
            "request_id": self.request_id,
            "ms": elapsed_ms,
            "bytes": serialized_bytes,
            "est_tokens": max(1, round(serialized_bytes / _BYTES_PER_TOKEN)),
        }


def infer_severity(status: str) -> str:
    st = (status or "OK").upper()
    if st in _INFO:
        return "INFO"
    if st in _WARN:
        return "WARN"
    return "ERROR"


def _clamp_value(value: Any) -> tuple[Any, bool]:
    """Recursively clamp huge strings, lists and dicts; returns (new_value, changed)."""
    if isinstance(value, str) and len(value) > _CLAMP_STR_CAP:
        return value[-_CLAMP_STR_CAP:], True
    if isinstance(value, list):
        changed = len(value) > _CLAMP_LIST_CAP
        items = []
        for item in value[:_CLAMP_LIST_CAP]:
            new, did = _clamp_value(item)
            items.append(new)
            changed = changed or did
        return items, changed
    if isinstance(value, dict):
        changed = len(value) > _CLAMP_LIST_CAP
        out = {}
        for key, item in list(value.items())[:_CLAMP_LIST_CAP]:
            out[key], did = _clamp_value(item)
            changed = changed or did
        return out, changed
    return value, False


def clamp_oversized(payload: dict[str, Any], max_bytes: int = MAX_SAFE_JSON_BYTES) -> bool:
    """Clamp the payload in place (protects the stdio JSON-RPC frame).

    Pass 1 recurses into nested values. If the result is still above ``max_bytes``, the largest
    values are replaced by ``{"_clamped": true, "bytes": N}`` until it fits.
    """
    clamped = False
    for key, value in list(payload.items()):
        if isinstance(value, list) and len(value) > _CLAMP_LIST_CAP:
            payload[f"{key}_total_count"] = len(value)
        elif isinstance(value, dict) and len(value) > _CLAMP_LIST_CAP:
            payload[f"{key}_total_keys"] = len(value)
        payload[key], did = _clamp_value(value)
        clamped = clamped or did
    while len(_dumps(payload, None).encode("utf-8")) > max_bytes:
        sizes = {k: len(_dumps(v, None).encode("utf-8")) for k, v in payload.items()}
        biggest = max(sizes, key=lambda k: sizes[k])
        payload[biggest] = {"_clamped": True, "bytes": sizes[biggest]}
        clamped = True
    return clamped


def _dumps(obj: Any, indent: int | None) -> str:
    return json.dumps(obj, indent=indent, ensure_ascii=False, default=str)


def _cap_lists(payload: dict[str, Any], cap: int) -> bool:
    capped = False
    for key, value in list(payload.items()):
        if isinstance(value, list) and len(value) > cap:
            payload[f"{key}_total_count"] = len(value)
            payload[key] = value[:cap]
            capped = True
    return capped


def _cap_nested(value: Any, str_cap: int, list_cap: int) -> tuple[Any, bool]:
    """Recursively cap string length and list length; returns (new_value, changed)."""
    if isinstance(value, str):
        return (value[:str_cap] + "...", True) if len(value) > str_cap else (value, False)
    if isinstance(value, list):
        changed = len(value) > list_cap
        items = []
        for item in value[:list_cap]:
            new, did = _cap_nested(item, str_cap, list_cap)
            items.append(new)
            changed = changed or did
        return items, changed
    if isinstance(value, dict):
        changed = False
        out = {}
        for key, item in value.items():
            out[key], did = _cap_nested(item, str_cap, list_cap)
            changed = changed or did
        return out, changed
    return value, False


def _cap_view(rest: dict[str, Any], str_cap: int, list_cap: int) -> bool:
    """Cap ``rest`` in place (top-level lists record ``<key>_total_count``)."""
    capped = _cap_lists(rest, list_cap)
    for key, value in list(rest.items()):
        if key in _UNCAPPED_KEYS or key.endswith("_total_count"):
            continue
        rest[key], did = _cap_nested(value, str_cap, list_cap)
        capped = capped or did
    return capped


def build_response(
    payload: dict[str, Any],
    tracker: Telemetry,
    *,
    verbosity: str = "summary",
    telemetry: bool | None = None,
) -> str:
    """Serialize a handler payload as the single typed envelope.

    Top-level keys are always: status, severity, summary, next_action, truncated, cursor.
    ``telemetry=None`` means default per verbosity (on for detail/summary, off for brief/terse).
    """
    body = dict(payload)
    terse_view = body.pop("terse", None)
    brief_view = body.pop("brief", None)
    body.pop("telemetry", None)
    body.pop("envelope", None)
    status = str(body.pop("status", "OK"))
    severity = infer_severity(status)
    truncated = bool(body.pop("truncated", False))
    cursor = body.pop("cursor", None)
    summary = (
        body.pop("summary", None)
        or body.get("error")
        or (f"{tracker.domain}.{tracker.action}: {status}")
    )
    next_action = body.pop("next_action", None) or _default_next_action(severity, truncated)

    if verbosity == "terse" and severity != "ERROR":
        rest = dict(terse_view) if isinstance(terse_view, dict) else {}
        truncated = _cap_view(rest, TERSE_STR_CAP, TERSE_LIST_CAP) or truncated
    else:
        rest = dict(brief_view) if verbosity == "brief" and isinstance(brief_view, dict) else body
        if verbosity == "brief" and _cap_view(rest, BRIEF_STR_CAP, BRIEF_LIST_CAP):
            truncated = True
        elif verbosity == "terse":  # terse + ERROR: details kept, but still bounded
            truncated = _cap_view(rest, TERSE_STR_CAP, TERSE_LIST_CAP) or truncated

    out: dict[str, Any] = {
        "status": status,
        "severity": severity,
        "summary": summary,
        "next_action": next_action,
        "truncated": truncated,
        "cursor": cursor,
        **rest,
    }
    indent = 2 if verbosity == "detail" else None
    text = _dumps(out, indent)
    if len(text.encode("utf-8")) > MAX_SAFE_JSON_BYTES and clamp_oversized(out):
        out["truncated"] = True
        text = _dumps(out, indent)
    with_telemetry = telemetry if telemetry is not None else verbosity in ("detail", "summary")
    if with_telemetry:
        out["telemetry"] = tracker.finish(len(text.encode("utf-8")))
        text = _dumps(out, indent)
    return text


def _default_next_action(severity: str, truncated: bool) -> str:
    if severity == "ERROR":
        return "Inspect 'error' or 'valid' and retry with corrected parameters."
    if truncated:
        return "Use verbosity='detail' or 'cursor' to inspect the remaining items."
    return "Proceed with the next step."
