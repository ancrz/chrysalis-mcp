import ast
import time
from pathlib import Path

import pytest
import z3

import chrysalis_mcp.domains.reason as reason_module

XY = (
    "(declare-const x Int)(declare-const y Int)"
    "(assert (= (+ x y) 10))(assert (> x y))(assert (> y 0))"
)


def solve(call, smt2=XY, **params):
    return call("reason", "solve", params={"smt2": smt2, **params})


def test_sat_model_satisfies_constraints(call):
    out = solve(call)
    assert out["status"] == "OK" and out["result"] == "sat"
    x, y = int(out["model"]["x"]), int(out["model"]["y"])
    assert x + y == 10 and x > y and y > 0
    assert "elapsed_ms" in out


def test_unsat_has_empty_model(call):
    out = solve(call, "(declare-const x Int)(assert (> x 1))(assert (< x 1))")
    assert out["result"] == "unsat" and out["model"] == {}


def test_timeout_path_does_not_raise(call):
    hard = (
        "(declare-const a Int)(declare-const b Int)(declare-const c Int)"
        "(assert (> a 1))(assert (> b 1))(assert (> c 1))"
        "(assert (= (+ (* a a a) (* b b b)) (* c c c)))"
    )
    out = solve(call, hard, timeout_ms=1)
    assert out["status"] == "OK" and out["result"] in ("unknown", "sat", "unsat")
    if out["result"] == "unknown":
        assert out["reason_unknown"]


def test_max_models_distinct(call):
    out = solve(call, XY, max_models=3)
    models = out["models"]
    assert len(models) == 3
    assert len({tuple(sorted(m.items())) for m in models}) == 3
    for m in models:
        assert int(m["x"]) + int(m["y"]) == 10


def test_max_models_fewer_than_requested(call):
    out = solve(call, "(declare-const b Bool)(assert (or b (not b)))", max_models=5)
    assert len(out["models"]) <= 2


def test_check_valid_invalid_and_unknown(call):
    ok = call("reason", "check", params={"smt2": XY, "assignment": {"x": 6, "y": 4}})
    assert ok["valid"] is True
    bad = call("reason", "check", params={"smt2": XY, "assignment": {"x": 1, "y": 9}})
    assert bad["valid"] is False and bad["result"] == "unsat"
    err = call("reason", "check", params={"smt2": XY, "assignment": {"zzz": 1}})
    assert err["status"] == "ERROR" and err["valid"] == ["x", "y"]


def test_check_bool_and_string_values(call):
    smt2 = '(declare-const p Bool)(declare-const s String)(assert p)(assert (= s "hi"))'
    out = call("reason", "check", params={"smt2": smt2, "assignment": {"p": True, "s": "hi"}})
    assert out["valid"] is True
    out = call("reason", "check", params={"smt2": smt2, "assignment": {"p": False}})
    assert out["valid"] is False
    out = call("reason", "check", params={"smt2": XY, "assignment": {"x": True}})
    assert out["status"] == "ERROR"


def test_limits_are_structured_errors(call):
    assert solve(call, "(assert true)" + " " * 65 * 1024)["status"] == "ERROR"
    assert solve(call, timeout_ms=0)["status"] == "ERROR"
    assert solve(call, timeout_ms=20000)["status"] == "ERROR"
    assert solve(call, max_models=11)["status"] == "ERROR"
    assert solve(call, max_models=0)["status"] == "ERROR"
    assert solve(call, timeout_ms=True)["status"] == "ERROR"
    assert solve(call, "   ")["status"] == "ERROR"


def test_malformed_smt2_is_error(call):
    out = solve(call, "(assert (> x 1)")
    assert out["status"] == "ERROR" and "SMT-LIB" in out["error"]


def test_terse_projection(call):
    out = call("reason", "solve", params={"smt2": XY}, verbosity="terse")
    assert out["result"] == "sat" and "model" not in out
    out = call("reason", "check", params={"smt2": XY, "assignment": {"x": 6}}, verbosity="terse")
    assert out["valid"] is True


def test_schema_lists_reason_actions(call, registry):
    out = call("meta", "schema")
    assert set(out["domains"]["reason"]) == {"solve", "check"}
    assert registry.actions("reason") == ["check", "solve"]


def test_batch_solve_then_check_with_cache_ref(call):
    out = call(
        "meta",
        "batch",
        params={
            "steps": [
                {"id": "s", "domain": "reason", "action": "solve", "params": {"smt2": XY}},
                {
                    "id": "c",
                    "domain": "reason",
                    "action": "check",
                    "params": {"smt2": XY, "assignment": "$cache.s.model"},
                },
            ],
            "return_only": "last",
            "select_fields": ["valid"],
        },
    )
    assert out["status"] == "OK" and out["valid"] is True


def test_reason_does_not_import_substrate():
    tree = ast.parse(Path(reason_module.__file__).read_text())
    modules = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            modules.append(("." * node.level) + (node.module or ""))
        elif isinstance(node, ast.Import):
            modules.extend(a.name for a in node.names)
    assert not any("substrate" in m for m in modules)


# --- security regressions (Dokimos T6 loop 1): allow-list + worker isolation -----------------


def _ok(call):
    out = solve(call)
    assert out["status"] == "OK" and out["result"] == "sat"


def test_memory_max_size_rejected_and_next_solve_works(call):
    before = z3.get_param("memory_max_size")
    out = solve(call, "(set-option :memory_max_size 1)" + XY)
    assert out["status"] == "ERROR" and "set-option" in out["error"]
    assert "assert" in out["allowed"] and "set-option" not in out["allowed"]
    _ok(call)
    assert z3.get_param("memory_max_size") == before


def test_global_params_unchanged_in_parent(call):
    before = (z3.get_param("pp.decimal"), z3.get_param("memory_max_size"))
    for opt in ("(set-option :pp.decimal true)", "(set-option :memory_max_size 1)"):
        assert solve(call, opt + XY)["status"] == "ERROR"
    _ok(call)
    assert (z3.get_param("pp.decimal"), z3.get_param("memory_max_size")) == before


@pytest.mark.parametrize(
    "opt", [":trace_file_name", ":diagnostic-output-channel", ":regular-output-channel"]
)
def test_file_options_rejected_and_no_file_created(call, tmp_path, opt):
    target = tmp_path / "pwned.txt"
    out = solve(call, f'(set-option {opt} "{target}")' + XY)
    assert out["status"] == "ERROR"
    assert not target.exists() and list(tmp_path.iterdir()) == []


def test_push_huge_rejected_fast(call):
    started = time.perf_counter()
    out = solve(call, "(push 200000000)" + XY)
    assert out["status"] == "ERROR" and time.perf_counter() - started < 1.0


@pytest.mark.parametrize(
    "smt2",
    [
        "(get-model)",
        '(echo "x")',
        "(eval 1)",
        "(reset)",
        "(|set-option| :a 1)",
        "(declare-const x Int) stray",
        "(assert true))",
        '(assert (= "unterminated true))',
    ],
)
def test_other_commands_and_shapes_rejected(call, smt2):
    assert solve(call, smt2)["status"] == "ERROR"


def test_command_hidden_in_comment_or_string_is_not_a_command(call):
    smt2 = (
        '; (set-option :memory_max_size 1)\n(declare-const s String)(assert (= s "(push 1)"))'
        "(declare-const |(pop 1)| Int)(check-sat)(set-logic QF_S)"
    )
    out = solve(call, smt2)
    assert out["status"] == "OK" and out["result"] == "sat"


def test_unknown_logic_rejected_and_known_accepted(call):
    assert solve(call, "(set-logic NOPE)" + XY)["status"] == "ERROR"
    assert solve(call, "(set-logic QF_LIA)" + XY)["result"] == "sat"


def test_deep_nesting_rejected(call):
    deep = "(declare-const p Bool)(assert " + "(not " * 600 + "p" + ")" * 600 + ")"
    out = solve(call, deep)
    assert out["status"] == "ERROR" and "nesting" in out["error"]


def test_undeclared_symbol_is_z3_error(call):
    out = solve(call, "(assert (> x 1))")
    assert out["status"] == "ERROR" and "SMT-LIB error" in out["error"]


def test_worker_crash_returns_error_and_recovers(call, monkeypatch):
    _ok(call)
    monkeypatch.setattr(reason_module, "_test_crash", "exit")
    out = solve(call)
    assert out["status"] == "ERROR" and out["code"] == "worker_crashed"
    monkeypatch.setattr(reason_module, "_test_crash", None)
    _ok(call)


def test_worker_hang_is_killed_and_recovers(call, monkeypatch):
    _ok(call)
    monkeypatch.setattr(reason_module, "_test_crash", "hang")
    started = time.perf_counter()
    out = solve(call, timeout_ms=1)
    assert out["status"] == "ERROR" and out["code"] == "worker_timeout"
    assert time.perf_counter() - started < 5.0
    monkeypatch.setattr(reason_module, "_test_crash", None)
    _ok(call)


def test_worker_killed_externally_recovers(call):
    _ok(call)
    reason_module._worker._proc.kill()
    reason_module._worker._proc.join()
    _ok(call)


def test_reason_module_does_not_import_z3():
    tree = ast.parse(Path(reason_module.__file__).read_text())
    names = {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    assert "z3" not in names
