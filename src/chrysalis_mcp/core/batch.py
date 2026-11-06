"""Server-side step chaining with ``$cache`` references and output projection.

Adapted from AEP ``playbook_run(action="batch")`` (see NOTICE). Only internal chrysalis steps
run here: no shell execution. Results of intermediate steps never leave the server unless the
caller asks for them via ``return_only`` / ``select_fields``.
"""

import re
from typing import Any

from .envelope import infer_severity
from .registry import Context, Payload, Registry
from .session import MISSING, SessionCache

MAX_STEPS = 25
RETURN_MODES = ("all", "last", "summary")
_REF = re.compile(r"^\$cache\.([^.\s]+)((?:\.[^.\s]+)*)$")


class BatchError(Exception):
    """A batch is malformed or a reference cannot be resolved."""


def lookup(obj: Any, dotted: str) -> Any:
    """Follow a dot-path through dicts (keys) and lists (integer indexes)."""
    for part in dotted.split("."):
        try:
            obj = obj[int(part)] if isinstance(obj, list) else obj[part]
        except (KeyError, IndexError, ValueError, TypeError) as exc:
            raise LookupError(dotted) from exc
    return obj


def project_fields(payload: Payload, paths: list[str]) -> Payload:
    """Keep only the dot-paths requested (missing paths are skipped)."""
    out: Payload = {}
    for path in paths:
        try:
            value = lookup(payload, path)
        except LookupError:
            continue
        node = out
        *head, last = path.split(".")
        for part in head:
            node = node.setdefault(part, {})
        node[last] = value
    return out


def _resolve(value: Any, local: dict[str, Any], cache: SessionCache, sid: str | None) -> Any:
    if isinstance(value, dict):
        return {k: _resolve(v, local, cache, sid) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve(v, local, cache, sid) for v in value]
    if not isinstance(value, str) or not (match := _REF.match(value)):
        return value
    key, rest = match.group(1), match.group(2)[1:]
    root = local.get(key, MISSING)
    if root is MISSING:
        root = cache.get(sid, key)
    if root is MISSING:
        raise BatchError(f"unresolved reference {value!r}: no cache key '{key}'")
    try:
        return lookup(root, rest) if rest else root
    except LookupError as exc:
        raise BatchError(f"unresolved reference {value!r}: path not found") from exc


def _validate(steps: Any, max_steps: int) -> list[dict[str, Any]]:
    if not isinstance(steps, list) or not steps:
        raise BatchError("'steps' must be a non-empty list")
    if len(steps) > max_steps:
        raise BatchError(f"too many steps: {len(steps)} > max_steps={max_steps}")
    seen: set[str] = set()
    normalized: list[dict[str, Any]] = []
    for index, step in enumerate(steps):
        if not isinstance(step, dict) or not step.get("domain") or not step.get("action"):
            raise BatchError(f"step {index}: needs 'domain' and 'action'")
        if (step["domain"], step["action"]) == ("meta", "batch"):
            raise BatchError(f"step {index}: nested meta.batch is not allowed")
        step_id = str(step.get("id") or f"s{index}")
        if step_id in seen or "." in step_id:
            raise BatchError(f"step {index}: id '{step_id}' is duplicated or contains '.'")
        seen.add(step_id)
        store_as = step.get("store_as")
        if store_as is not None and (not str(store_as).strip() or "." in str(store_as)):
            raise BatchError(f"step {index}: invalid store_as")
        normalized.append({**step, "id": step_id})
    return normalized


def run_batch(
    steps: Any,
    registry: Registry,
    ctx: Context,
    *,
    return_only: str = "all",
    select_fields: list[str] | None = None,
    continue_on_error: bool = False,
    max_steps: int = MAX_STEPS,
) -> Payload:
    if return_only not in RETURN_MODES:
        return _error(f"return_only must be one of {list(RETURN_MODES)}")
    if select_fields is not None and not (
        isinstance(select_fields, list) and all(isinstance(f, str) for f in select_fields)
    ):
        return _error("select_fields must be a list of dot-path strings")
    try:
        plan = _validate(steps, max_steps)
    except BatchError as exc:
        return _error(str(exc))

    local: dict[str, Any] = {}
    results: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    for step in plan:
        try:
            params = _resolve(step.get("params") or {}, local, ctx.cache, ctx.session_id)
        except BatchError as exc:
            output: Payload = {"status": "ERROR", "error": str(exc)}
        else:
            output = registry.dispatch(step["domain"], step["action"], params, ctx)
        status = str(output.get("status", "OK"))
        results.append({"id": step["id"], "status": status, "output": output})
        if infer_severity(status) == "ERROR" or status == "DISCOURAGED":
            failed.append({"id": step["id"], "status": status, "error": output.get("error")})
            if not continue_on_error:
                break
            continue
        local[step["id"]] = output
        if step.get("store_as"):
            key = str(step["store_as"]).strip()
            local[key] = output
            ctx.cache.put(ctx.session_id, key, output)

    status = "OK" if not failed else ("PARTIAL" if continue_on_error else "ERROR")
    if return_only == "last":
        body: Payload = {**results[-1]["output"]}
    elif return_only == "summary":
        body = {"results": [{"id": r["id"], "status": r["status"]} for r in results]}
    else:
        body = {"results": results}
    body["steps_run"] = len(results)
    if select_fields:
        body = project_fields(body, select_fields)
    # Status and failed steps are never projected away: a failure must stay visible.
    body["status"] = status if failed else str(body.get("status", status))
    body["failed_steps"] = failed
    return body


def _error(message: str) -> Payload:
    return {"status": "ERROR", "error": message, "failed_steps": []}
