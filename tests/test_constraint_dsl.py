import ast
import string
from pathlib import Path

import pytest

import chrysalis_mcp.domains.constraint_dsl as dsl_module
from chrysalis_mcp.domains.constraint_dsl import DslError, bound_assertion, compile_constraints

V = {"x": "int", "y": "int", "r": "real", "b": "bool"}


def asserts(*exprs, goal=None):
    spec = {"vars": V, "assert": list(exprs)}
    if goal:
        spec["goal"] = goal
    return compile_constraints(spec)


def solve(call, exprs, goal=None, **params):
    spec = {"vars": V, "assert": exprs}
    if goal:
        spec["goal"] = goal
    return call("reason", "solve", params={"constraints": spec, **params})


@pytest.mark.parametrize(
    ("expr", "smt"),
    [
        ("x + y == 10", "(assert (= (+ x y) 10))"),
        ("x != y", "(assert (not (= x y)))"),
        ("x + 2 * 3 >= y", "(assert (>= (+ x (* 2 3)) y))"),
        ("-x < 0", "(assert (< (- x) 0))"),
        ("x < 1 or y < 1 and b", "(assert (or (< x 1) (and (< y 1) b)))"),
        ("not (x < 1)", "(assert (not (< x 1)))"),
        ("not not b", "(assert (not (not b)))"),
        ("b == (x < 2)", "(assert (= b (< x 2)))"),
        ("b != true", "(assert (not (= b true)))"),
        ("r > x", "(assert (> r (to_real x)))"),
        ("r + 1 > 2.5", "(assert (> (+ r (to_real 1)) 2.5))"),
        ("x - y - 1 == 0", "(assert (= (- (- x y) 1) 0))"),
    ],
)
def test_valid_expressions_compile(expr, smt):
    assert asserts(expr).smt2.splitlines()[-1] == smt


def test_declarations_and_goal():
    c = asserts("x > 0", goal="min:x")
    assert c.goal == "min" and c.goal_var == "x" and c.var_types == V
    assert c.smt2.splitlines()[:4] == [
        "(declare-const x Int)",
        "(declare-const y Int)",
        "(declare-const r Real)",
        "(declare-const b Bool)",
    ]
    assert asserts("x > 0").goal == "sat"


@pytest.mark.parametrize(
    "expr",
    [
        "x +",  # dangling operator
        "x ++ y == 1",
        "x == == y",
        "x < y < 3",  # chained
        "(x + y",  # missing paren
        "x + y)",  # extra paren
        "z > 1",  # undeclared
        "x + y",  # not boolean
        "b + 1 == 2",  # bool in arithmetic
        "x and y",  # int in connective
        "not x",
        "b < b",
        "x == b",
        "x / 2 == 1",  # division is not whitelisted
        "x % 2 == 1",
        "x ** 2 == 4",
        "x & 1 == 1",
        "x => y",
        "x == 1;",
        "",
        "   ",
        "99999999999999999999 == x",  # literal too long
        "x == 1e3",
        "x == 0x10",
    ],
)
def test_invalid_expressions_rejected(expr):
    with pytest.raises(DslError):
        asserts(expr)


@pytest.mark.parametrize(
    "expr",
    [
        "__import__('os').system('id') == 0",
        "x == 1) (set-option :print-success true",
        "x == 1) (assert true",
        "(assert true)",
        "open('/etc/passwd')",
        "x.__class__ == 1",
        "x[0] == 1",
        "lambda: 1",
        "x == 1 # comment",
        "x == 1\n(check-sat)",
        "x == `id`",
        "x == $(id)",
        "x == '1'",
        'x == "1"',
        "{x} == 1",
        "x := 1",
        "exec('1') == 1",
        "eval(x) == 1",
        "x |foo| 1",
        "(declare-const q Int) x == 1",
    ],
)
def test_injection_like_strings_rejected(expr):
    with pytest.raises(DslError):
        asserts(expr)


def test_depth_and_size_limits():
    with pytest.raises(DslError, match="nested"):
        asserts("(" * 60 + "x > 1" + ")" * 60)
    with pytest.raises(DslError, match="nested"):
        asserts("not " * 60 + "b")
    with pytest.raises(DslError, match="exceeds"):
        asserts("x > 1 " + "and x > 1 " * 200)
    with pytest.raises(DslError, match="too many assertions"):
        compile_constraints({"vars": V, "assert": ["x > 1"] * 101})
    with pytest.raises(DslError, match="too many variables"):
        compile_constraints({"vars": {f"v{i}": "int" for i in range(51)}, "assert": ["v0 > 1"]})


@pytest.mark.parametrize(
    "spec",
    [
        None,
        [],
        "x > 1",
        {"vars": V},
        {"assert": ["x > 1"]},
        {"vars": {}, "assert": ["x > 1"]},
        {"vars": V, "assert": []},
        {"vars": V, "assert": "x > 1"},
        {"vars": V, "assert": [1]},
        {"vars": V, "assert": ["x > 1"], "extra": 1},
        {"vars": {"x": "string"}, "assert": ["x > 1"]},
        {"vars": {"x y": "int"}, "assert": ["x > 1"]},
        {"vars": {"1x": "int"}, "assert": ["1x > 1"]},
        {"vars": {"x)": "int"}, "assert": ["x > 1"]},
        {"vars": {"ite": "int"}, "assert": ["ite > 1"]},
        {"vars": {"_": "int"}, "assert": ["_ > 1"]},
        {"vars": {"x" * 33: "int"}, "assert": ["x > 1"]},
        {"vars": V, "assert": ["x > 1"], "goal": "min:b"},
        {"vars": V, "assert": ["x > 1"], "goal": "min:nope"},
        {"vars": V, "assert": ["x > 1"], "goal": "minimize x"},
        {"vars": V, "assert": ["x > 1"], "goal": 5},
    ],
)
def test_invalid_specs_rejected(spec):
    with pytest.raises(DslError):
        compile_constraints(spec)


def test_bound_assertion_numerals():
    assert bound_assertion("x", "int", "<=", -3) == "(assert (<= x (- 3)))"
    assert bound_assertion("x", "int", ">=", 7) == "(assert (>= x 7))"
    assert bound_assertion("r", "real", "<", 5) == "(assert (< r 5.0))"
    assert bound_assertion("r", "real", ">", -0.5) == "(assert (> r (- (/ 1.0 2.0))))"
    with pytest.raises(ValueError):
        bound_assertion("x", "int", "==", 1)


def test_module_never_evaluates_code():
    tree = ast.parse(Path(dsl_module.__file__).read_text())
    called = {
        n.func.id
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
    }
    imported = {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    imported |= {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    assert not called & {"eval", "exec", "compile", "__import__", "getattr", "open"}
    assert not imported & {"subprocess", "os", "importlib", "pickle"}
    assert set(string.printable) >= set("+-*")  # sanity: whitelist is plain ASCII


# --- end to end through reason.solve / reason.check ---------------------------------------


def test_dsl_solve_sat_model(call):
    out = solve(call, ["x + y == 10", "x > y", "y > 0"])
    assert out["status"] == "OK" and out["result"] == "sat"
    x, y = int(out["model"]["x"]), int(out["model"]["y"])
    assert x + y == 10 and x > y and y > 0


def test_dsl_solve_unsat(call):
    out = solve(call, ["x > 5", "x < 3"])
    assert out["status"] == "OK" and out["result"] == "unsat"


def test_dsl_mixed_real_int_and_bool(call):
    out = solve(call, ["r * 2 == x + 1", "x == 4", "b == (r > 2)"])
    assert out["result"] == "sat" and out["model"]["b"].lower() == "true"


@pytest.mark.parametrize(
    ("goal", "expected"),
    [("max:x", 9), ("min:x", 6), ("max:y", 4), ("min:y", 1)],
)
def test_dsl_goal_optimum(call, goal, expected):
    out = solve(call, ["x + y == 10", "x > y", "y > 0"], goal=goal)
    assert out["result"] == "sat"
    assert out["objective"]["optimal"] is True and out["objective"]["value"] == str(expected)
    assert out["model"][goal[-1]] == str(expected)


def test_dsl_goal_wide_range_converges_fast(call):
    out = solve(call, ["x >= 0", "x <= 1000000", "y == x"], goal="max:x")
    assert out["objective"]["optimal"] is True and out["objective"]["value"] == "1000000"
    assert out["objective"]["iterations"] < 40


def test_dsl_goal_unbounded_reports_not_optimal(call):
    out = solve(call, ["x > 0"], goal="max:x")
    assert out["status"] == "OK" and out["objective"]["optimal"] is False


def test_dsl_goal_real_optimum(call):
    out = solve(call, ["r >= 1.5", "r <= 7.25"], goal="min:r")
    assert out["objective"]["optimal"] is True and out["objective"]["value"] == "3/2"


def test_dsl_goal_unsat_passthrough(call):
    out = solve(call, ["x > 5", "x < 3"], goal="max:x")
    assert out["result"] == "unsat" and "objective" not in out


def test_dsl_check_assignment(call):
    spec = {"vars": V, "assert": ["x + y == 10", "x > y"]}
    ok = call("reason", "check", params={"constraints": spec, "assignment": {"x": 6, "y": 4}})
    bad = call("reason", "check", params={"constraints": spec, "assignment": {"x": 4, "y": 6}})
    assert ok["status"] == "OK" and ok["valid"] is True
    assert bad["valid"] is False


def test_dsl_and_smt2_are_exclusive(call):
    spec = {"vars": V, "assert": ["x > 1"]}
    both = call("reason", "solve", params={"constraints": spec, "smt2": "(assert true)"})
    neither = call("reason", "solve", params={})
    assert both["status"] == "ERROR" and neither["status"] == "ERROR"


def test_dsl_rejection_is_structured_error(call):
    out = solve(call, ["x == 1) (set-option :print-success true"])
    assert out["status"] == "ERROR" and out.get("code") == "dsl_rejected"


def test_smt2_input_still_supported(call):
    out = call("reason", "solve", params={"smt2": "(declare-const x Int)(assert (> x 3))"})
    assert out["result"] == "sat"


def test_non_ascii_digits_are_rejected_by_the_dsl():
    with pytest.raises(DslError):
        compile_constraints({"vars": {"x": "int"}, "assert": ["x > ٣"]})
