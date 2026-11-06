"""Z3 worker process: the only place that imports ``z3``.

Runs in a spawned child with an address-space limit, so a Z3 crash, a runaway allocation or a
hang can only kill the worker, never the MCP server. The parent talks to it over a ``Pipe``
with plain dicts. Each job uses a fresh ``z3.Context``.
"""

import os
import resource
import time
from multiprocessing.connection import Connection
from typing import Any

Job = dict[str, Any]
Payload = dict[str, Any]


def _error(message: str, **extra: Any) -> Payload:
    return {"status": "ERROR", "error": message, **extra}


def _constants(z3: Any, assertions: list[Any]) -> dict[str, Any]:
    """Declared (uninterpreted, arity-0) constants reachable from the assertions."""
    found: dict[str, Any] = {}
    seen: set[int] = set()
    stack = list(assertions)
    while stack:
        node = stack.pop()
        if node.get_id() in seen:
            continue
        seen.add(node.get_id())
        if z3.is_const(node) and node.decl().kind() == z3.Z3_OP_UNINTERPRETED:
            found[node.decl().name()] = node
        stack.extend(node.children())
    return found


def _model_dict(model: Any) -> dict[str, str]:
    return {d.name(): str(model[d]) for d in model.decls() if d.arity() == 0}


def _block(z3: Any, solver: Any, model: Any) -> bool:
    """Add a clause excluding ``model``; False when there is nothing to block."""
    diffs = [d() != model[d] for d in model.decls() if d.arity() == 0]
    if not diffs:
        return False
    solver.add(z3.Or(*diffs))
    return True


def _solve(z3: Any, job: Job) -> Payload:
    started = time.perf_counter()
    z3ctx = z3.Context()
    try:
        solver = z3.Solver(ctx=z3ctx)
        solver.set("timeout", job["timeout_ms"])
        solver.add(*z3.parse_smt2_string(job["smt2"], ctx=z3ctx))
        verdict = solver.check()
        models: list[dict[str, str]] = []
        while verdict == z3.sat:
            model = solver.model()
            models.append(_model_dict(model))
            if len(models) >= job["max_models"] or not _block(z3, solver, model):
                break
            verdict = solver.check()
        reason_unknown = str(solver.reason_unknown()) if verdict == z3.unknown else None
    except z3.Z3Exception as exc:
        return _error(f"SMT-LIB error: {exc}")

    result = "sat" if models else str(verdict)
    payload: Payload = {
        "status": "OK",
        "summary": result + (f" ({len(models)} model(s))" if models else ""),
        "result": result,
        "model": models[0] if models else {},
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
        "terse": {"result": result},
    }
    if job["max_models"] > 1:
        payload["models"] = models
    if reason_unknown is not None:
        payload["reason_unknown"] = reason_unknown
    return payload


def _coerce(z3: Any, const: Any, value: Any) -> Any:
    """Build a Z3 value of the constant's sort from a JSON-friendly scalar."""
    kind = const.sort().kind()
    z3ctx = const.ctx
    if kind == z3.Z3_BOOL_SORT:
        if isinstance(value, str):
            if value.lower() not in ("true", "false"):
                raise ValueError(f"expected true/false, got {value!r}")
            value = value.lower() == "true"
        if not isinstance(value, bool):
            raise ValueError(f"expected bool, got {value!r}")
        return z3.BoolVal(value, z3ctx)
    if isinstance(value, bool):
        raise ValueError(f"expected {const.sort()}, got bool")
    if kind == z3.Z3_INT_SORT:
        return z3.IntVal(int(value), z3ctx)
    if kind == z3.Z3_REAL_SORT:
        return z3.RealVal(str(value), z3ctx)
    if kind == z3.Z3_BV_SORT:
        return z3.BitVecVal(int(value), const.sort().size(), z3ctx)
    if kind == z3.Z3_SEQ_SORT and const.sort().is_string():
        return z3.StringVal(str(value), z3ctx)
    raise ValueError(f"unsupported sort {const.sort()}")


def _check(z3: Any, job: Job) -> Payload:
    started = time.perf_counter()
    assignment = job["assignment"]
    z3ctx = z3.Context()
    try:
        assertions = list(z3.parse_smt2_string(job["smt2"], ctx=z3ctx))
        declared = _constants(z3, assertions)
        unknown = sorted(set(assignment) - set(declared))
        if unknown:
            return _error(
                f"unknown constant(s): {unknown}",
                valid=sorted(declared),
                next_action="Use only names listed in 'valid'.",
            )
        solver = z3.Solver(ctx=z3ctx)
        solver.set("timeout", job["timeout_ms"])
        solver.add(*assertions)
        try:
            for name, value in assignment.items():
                solver.add(declared[name] == _coerce(z3, declared[name], value))
        except (ValueError, TypeError) as exc:
            return _error(f"bad assignment value: {exc}")
        verdict = solver.check()
        reason_unknown = str(solver.reason_unknown()) if verdict == z3.unknown else None
    except z3.Z3Exception as exc:
        return _error(f"SMT-LIB error: {exc}")

    valid = verdict == z3.sat
    payload: Payload = {
        "status": "OK",
        "summary": f"assignment {'valid' if valid else 'not valid'} ({verdict})",
        "valid": valid,
        "result": str(verdict),
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
        "terse": {"valid": valid},
    }
    if reason_unknown is not None:
        payload["reason_unknown"] = reason_unknown
    return payload


def serve(conn: Connection, memory_limit_bytes: int) -> None:
    """Worker entry point: answer jobs until the pipe closes."""
    # The parent's stdin/stdout carry the MCP JSON-RPC stream; never touch them from here.
    devnull = os.open(os.devnull, os.O_RDWR)
    os.dup2(devnull, 0)
    os.dup2(devnull, 1)
    resource.setrlimit(resource.RLIMIT_AS, (memory_limit_bytes, memory_limit_bytes))
    import z3

    conn.send({"ready": True})
    while True:
        try:
            job = conn.recv()
        except EOFError, OSError:
            return
        crash = job.get("test_crash")  # set only by tests through the parent module attribute
        if crash == "exit":
            os._exit(70)
        if crash == "hang":
            time.sleep(3600)
        handler = _solve if job["op"] == "solve" else _check
        try:
            result = handler(z3, job)
        except MemoryError:
            result = _error(
                "worker_memory: Z3 exceeded the worker memory limit", code="worker_memory"
            )
        conn.send(result)
