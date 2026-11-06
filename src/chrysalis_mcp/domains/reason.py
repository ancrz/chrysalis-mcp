"""``reason`` domain: SMT-LIB solving and assignment checking backed by Z3.

The problem is given either as ``smt2`` text or as a ``constraints`` JSON object (a safe infix
DSL, see ``constraint_dsl``) that is compiled to SMT-LIB and then follows the exact same path:
allow-list, then the isolated worker. ``constraints.goal`` ``min:x``/``max:x`` is answered by
iterated tightening over the solver (no ``Optimize`` engine, no new commands).

Independent of the memory substrate. Two defence layers keep untrusted SMT-LIB harmless:

1. An allow-list of top-level commands is enforced *before* Z3 sees the text, so commands such as
   ``set-option`` (process-global Z3 parameters, trace/output files) or ``push N`` never run.
2. Parsing and solving happen in a separate worker process (``_z3_worker``) with an address-space
   limit and a wall-clock kill, so a crash or runaway call cannot take the MCP server down.
"""

import atexit
import multiprocessing as mp
import threading
import time
from fractions import Fraction
from typing import Any

from ..core.registry import Context, Params, Payload, Registry
from . import _z3_worker
from .constraint_dsl import Compiled, DslError, bound_assertion, compile_constraints

MAX_SMT2_BYTES = 64 * 1024
MIN_TIMEOUT_MS, MAX_TIMEOUT_MS, DEFAULT_TIMEOUT_MS = 1, 10_000, 2_000
MIN_MODELS, MAX_MODELS = 1, 10
MAX_NESTING_DEPTH = 512
WORKER_MEMORY_BYTES = 1024**3
WORKER_GRACE_S = 1.0
WORKER_START_TIMEOUT_S = 30.0
MAX_OPT_ITERATIONS = 40
OPT_BUDGET_S = 20.0

ALLOWED_COMMANDS = (
    "assert",
    "check-sat",
    "declare-const",
    "declare-datatypes",
    "declare-fun",
    "declare-sort",
    "define-fun",
    "define-sort",
    "set-logic",
)
ALLOWED_LOGICS = frozenset(
    (
        "ALL",
        "AUFLIA",
        "AUFLIRA",
        "AUFNIRA",
        "LIA",
        "LRA",
        "NIA",
        "NRA",
        "UF",
        "UFBV",
        "UFIDL",
        "UFLIA",
        "UFLRA",
        "UFNIA",
        "BV",
        "QF_ABV",
        "QF_ALIA",
        "QF_AUFBV",
        "QF_AUFLIA",
        "QF_AX",
        "QF_BV",
        "QF_DT",
        "QF_FP",
        "QF_BVFP",
        "QF_IDL",
        "QF_LIA",
        "QF_LIRA",
        "QF_LRA",
        "QF_NIA",
        "QF_NIRA",
        "QF_NRA",
        "QF_RDL",
        "QF_S",
        "QF_SLIA",
        "QF_UF",
        "QF_UFBV",
        "QF_UFIDL",
        "QF_UFLIA",
        "QF_UFLRA",
        "QF_UFNIA",
    )
)

# Test-only crash injection ("exit" | "hang"); not reachable from MCP parameters.
_test_crash: str | None = None


def _error(message: str, **extra: Any) -> Payload:
    return {"status": "ERROR", "error": message, **extra}


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


class _Rejected(Exception):
    pass


_ATOM_END = frozenset('()";|')


def _top_level_commands(text: str) -> list[list[str]]:
    """Tokenize top-level s-expressions; return each one's depth-1 tokens (command, first arg...).

    Skips ``;`` comments, "string literals" and |quoted symbols|. Linear time, iterative.
    """
    n, i, depth = len(text), 0, 0
    commands: list[list[str]] = []
    current: list[str] = []
    while i < n:
        c = text[i]
        if c == ";":
            end = text.find("\n", i)
            i = n if end == -1 else end + 1
        elif c.isspace():
            i += 1
        elif c == "(":
            if depth == 0:
                current = []
            depth += 1
            if depth > MAX_NESTING_DEPTH:
                raise _Rejected(f"nesting deeper than {MAX_NESTING_DEPTH}")
            i += 1
        elif c == ")":
            if depth == 0:
                raise _Rejected("unbalanced parentheses: unexpected ')'")
            depth -= 1
            if depth == 0:
                if not current:
                    raise _Rejected("empty command")
                commands.append(current)
            i += 1
        else:
            if c in ('"', "|"):
                j = i + 1
                while True:
                    j = text.find(c, j)
                    if j == -1:
                        raise _Rejected("unterminated string or quoted symbol")
                    if c == '"' and text[j + 1 : j + 2] == '"':  # "" escape inside a string
                        j += 2
                        continue
                    break
                token, i = text[i : j + 1], j + 1
            else:
                j = i
                while j < n and not text[j].isspace() and text[j] not in _ATOM_END:
                    j += 1
                token, i = text[i:j], j
            if depth == 0:
                raise _Rejected(f"unexpected top-level token {token[:40]!r}")
            if depth == 1:
                current.append(token)
    if depth != 0:
        raise _Rejected("unbalanced parentheses: missing ')'")
    return commands


def _allow_list(smt2: str) -> Payload | None:
    try:
        commands = _top_level_commands(smt2)
    except _Rejected as exc:
        return _error(f"SMT-LIB rejected: {exc}", allowed=list(ALLOWED_COMMANDS))
    for tokens in commands:
        name = tokens[0]
        if name not in ALLOWED_COMMANDS:
            return _error(
                f"SMT-LIB command {name[:40]!r} is not allowed",
                allowed=list(ALLOWED_COMMANDS),
                next_action="Remove the command; only declarations, definitions and asserts run.",
            )
        if name == "set-logic" and (len(tokens) != 2 or tokens[1] not in ALLOWED_LOGICS):
            return _error(
                f"unknown logic {tokens[1:2]!r}",
                allowed_logics=sorted(ALLOWED_LOGICS),
            )
    return None


class _WorkerHandle:
    """Lazily started, persistent worker; restarted after it dies or is killed."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._proc: Any = None
        self._conn: Any = None

    def _stop(self) -> None:
        if self._proc is not None:
            self._proc.kill()
            self._proc.join(2)
        if self._conn is not None:
            self._conn.close()
        self._proc = self._conn = None

    def _start(self) -> bool:
        ctx = mp.get_context("spawn")
        parent_conn, child_conn = ctx.Pipe()
        proc = ctx.Process(
            target=_z3_worker.serve, args=(child_conn, WORKER_MEMORY_BYTES), daemon=True
        )
        proc.start()
        child_conn.close()
        self._proc, self._conn = proc, parent_conn
        try:
            if parent_conn.poll(WORKER_START_TIMEOUT_S) and parent_conn.recv().get("ready"):
                return True
        except EOFError, OSError:
            pass
        self._stop()
        return False

    def run(self, job: dict[str, Any], timeout_ms: int) -> Payload:
        with self._lock:
            if (self._proc is None or not self._proc.is_alive()) and not (
                self._stop() or self._start()
            ):
                return _error(
                    "worker_crashed: could not start the Z3 worker", code="worker_crashed"
                )
            try:
                self._conn.send({**job, "test_crash": _test_crash})
                if not self._conn.poll(timeout_ms / 1000 + WORKER_GRACE_S):
                    self._stop()
                    return _error(
                        f"worker_timeout: no answer within {timeout_ms} ms + grace; worker killed",
                        code="worker_timeout",
                    )
                return self._conn.recv()
            except EOFError, OSError, BrokenPipeError:
                code = self._proc.exitcode if self._proc is not None else None
                self._stop()
                return _error(
                    f"worker_crashed: Z3 worker died (exit code {code}); restarted on next call",
                    code="worker_crashed",
                )

    def shutdown(self) -> None:
        with self._lock:
            self._stop()


_worker = _WorkerHandle()
atexit.register(_worker.shutdown)


def _validate_common(params: Params) -> tuple[str, int, Compiled | None] | Payload:
    smt2, spec = params.get("smt2"), params.get("constraints")
    compiled: Compiled | None = None
    if (smt2 is None) == (spec is None):
        return _error("give exactly one of 'smt2' (SMT-LIB text) or 'constraints' (JSON DSL)")
    if spec is not None:
        try:
            compiled = compile_constraints(spec)
        except DslError as exc:
            return _error(f"constraints rejected: {exc}", code="dsl_rejected")
        smt2 = compiled.smt2
    if not isinstance(smt2, str) or not smt2.strip():
        return _error("'smt2' must be a non-empty string")
    if len(smt2.encode("utf-8")) > MAX_SMT2_BYTES:
        return _error(f"'smt2' exceeds {MAX_SMT2_BYTES} bytes")
    timeout_ms = params.get("timeout_ms", DEFAULT_TIMEOUT_MS)
    if not _is_int(timeout_ms) or not MIN_TIMEOUT_MS <= timeout_ms <= MAX_TIMEOUT_MS:
        return _error(f"'timeout_ms' must be an integer in [{MIN_TIMEOUT_MS}, {MAX_TIMEOUT_MS}]")
    rejected = _allow_list(smt2)  # also applied to compiled output: defence in depth
    if rejected is not None:
        return rejected
    return smt2, timeout_ms, compiled


def _optimize(compiled: Compiled, timeout_ms: int) -> Payload:
    """min/max by tightening: solve, then require a strictly better goal value, until unsat.

    Integers gallop (step doubles on sat, halves on unsat) so wide ranges converge in O(log) calls;
    the proof of optimality is an unsat at step 1. Reals use strict bounds and may not terminate
    (e.g. ``x > 0``), so the loop is capped by ``MAX_OPT_ITERATIONS`` and ``OPT_BUDGET_S`` and the
    answer then says ``optimal: false`` with the best value found.
    """
    assert compiled.goal_var is not None
    var, kind = compiled.goal_var, compiled.var_types[compiled.goal_var]
    minimize = compiled.goal == "min"
    best_payload = _worker.run(
        {"op": "solve", "smt2": compiled.smt2, "timeout_ms": timeout_ms, "max_models": 1},
        timeout_ms,
    )
    if best_payload.get("status") != "OK" or best_payload.get("result") != "sat":
        return best_payload
    try:
        best = Fraction(best_payload["model"][var])
    except KeyError, ValueError, ZeroDivisionError:
        return {**best_payload, "optimal": False, "note": f"goal variable {var!r} has no value"}
    step, iterations, optimal = 1, 0, False
    deadline = time.monotonic() + OPT_BUDGET_S
    while iterations < MAX_OPT_ITERATIONS and time.monotonic() < deadline:
        iterations += 1
        if kind == "int":
            op, target = ("<=", best - step) if minimize else (">=", best + step)
        else:
            op, target = ("<" if minimize else ">"), best
        job = {
            "op": "solve",
            "smt2": compiled.smt2 + "\n" + bound_assertion(var, kind, op, target),
            "timeout_ms": timeout_ms,
            "max_models": 1,
        }
        res = _worker.run(job, timeout_ms)
        if res.get("status") != "OK" or res.get("result") not in ("sat", "unsat"):
            break
        if res["result"] == "unsat":
            if kind == "real" or step == 1:
                optimal = True
                break
            step = max(1, step // 2)
            continue
        try:
            best = Fraction(res["model"][var])
        except KeyError, ValueError, ZeroDivisionError:
            break
        best_payload = res
        if kind == "int":
            step *= 2
    sense = "minimum" if minimize else "maximum"
    state = "optimal" if optimal else "best found, optimality not proven"
    return {
        **best_payload,
        "summary": f"sat; {sense} of {var} = {best} ({state})",
        "objective": {
            "var": var,
            "sense": compiled.goal,
            "value": str(best),
            "optimal": optimal,
            "iterations": iterations,
        },
        "terse": {"result": "sat", "objective": str(best), "optimal": optimal},
    }


def _solve(params: Params, ctx: Context) -> Payload:
    common = _validate_common(params)
    if isinstance(common, dict):
        return common
    smt2, timeout_ms, compiled = common
    max_models = params.get("max_models", 1)
    if not _is_int(max_models) or not MIN_MODELS <= max_models <= MAX_MODELS:
        return _error(f"'max_models' must be an integer in [{MIN_MODELS}, {MAX_MODELS}]")
    if compiled is not None and compiled.goal != "sat":
        return _optimize(compiled, timeout_ms)
    job = {"op": "solve", "smt2": smt2, "timeout_ms": timeout_ms, "max_models": max_models}
    return _worker.run(job, timeout_ms)


def _check(params: Params, ctx: Context) -> Payload:
    common = _validate_common(params)
    if isinstance(common, dict):
        return common
    smt2, timeout_ms, _ = common
    assignment = params.get("assignment")
    if not isinstance(assignment, dict):
        return _error("'assignment' must be an object of name -> int|bool|str")
    bad = [k for k, v in assignment.items() if not isinstance(v, (int, bool, str))]
    if bad:
        return _error("assignment values must be int, bool or str", invalid=sorted(bad))
    job = {"op": "check", "smt2": smt2, "timeout_ms": timeout_ms, "assignment": assignment}
    return _worker.run(job, timeout_ms)


def register(registry: Registry) -> None:
    common = {
        "smt2": {"type": "string", "required": False, "max_bytes": MAX_SMT2_BYTES},
        "constraints": {
            "type": "object",
            "required": False,
            "description": (
                "Alternative to smt2: {vars:{x:'int'|'bool'|'real'}, assert:['x + y == 10', "
                "'x > y'], goal?:'sat'|'min:x'|'max:x'}. Operators: + - * == != < <= > >= "
                "and or not, parentheses, int/decimal literals, true/false."
            ),
        },
        "timeout_ms": {
            "type": "integer",
            "required": False,
            "min": MIN_TIMEOUT_MS,
            "max": MAX_TIMEOUT_MS,
            "default": DEFAULT_TIMEOUT_MS,
        },
    }
    example_smt2 = (
        "(declare-const x Int)(declare-const y Int)"
        "(assert (= (+ x y) 10))(assert (> x y))(assert (> y 0))"
    )
    allowed = ", ".join(ALLOWED_COMMANDS)
    registry.register(
        "reason",
        "solve",
        _solve,
        params_schema={
            **common,
            "max_models": {
                "type": "integer",
                "required": False,
                "min": MIN_MODELS,
                "max": MAX_MODELS,
                "default": 1,
            },
        },
        description=(
            "Solve an SMT-LIB 2 problem ('smt2') or a safe JSON constraint problem "
            "('constraints') with Z3 in an isolated worker: sat/unsat/unknown plus model(s); "
            f"goal min:x/max:x optimizes. Allowed SMT-LIB commands: {allowed}."
        ),
        example={
            "constraints": {
                "vars": {"x": "int", "y": "int"},
                "assert": ["x + y == 10", "x > y", "y > 0"],
                "goal": "max:x",
            },
            "timeout_ms": 2000,
        },
    )
    registry.register(
        "reason",
        "check",
        _check,
        params_schema={**common, "assignment": {"type": "object", "required": True}},
        description="Check whether an assignment of declared constants satisfies the problem.",
        example={"smt2": example_smt2, "assignment": {"x": 6, "y": 4}},
    )
