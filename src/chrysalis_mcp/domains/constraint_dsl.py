"""Safe constraint DSL for ``reason``: a JSON problem compiled to SMT-LIB 2 text.

The point is to let a small model state a problem as plain infix text instead of writing
SMT-LIB. Nothing here evaluates anything: the assertion strings go through a tokenizer and a
recursive-descent parser over a closed grammar (no ``eval``/``exec``/``ast`` evaluation, no
attribute access, no calls), type-check against the declared variables, and are emitted as
s-expressions built only from the operator whitelist below. The result still passes through the
``reason`` command allow-list and runs in the isolated Z3 worker.

Problem shape::

    {"vars": {"x": "int", "b": "bool", "r": "real"},
     "assert": ["x + y == 10", "x > y", "b and not (x < 0)"],
     "goal": "sat" | "min:x" | "max:x"}

Grammar (lowest to highest precedence; comparison is non-associative)::

    expr  := and ("or" and)*          and := not ("and" not)*        not := "not" not | cmp
    cmp   := sum (("==" | "!=" | "<" | "<=" | ">" | ">=") sum)?
    sum   := prod (("+" | "-") prod)*        prod := unary ("*" unary)*
    unary := "-" unary | atom
    atom  := INT | DECIMAL | true | false | NAME | "(" expr ")"

Types are ``int``, ``bool`` and ``real``; ``int`` operands mixed with ``real`` are promoted.
"""

import re
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

VAR_TYPES = ("int", "bool", "real")
GOALS_HELP = "'sat', 'min:<var>' or 'max:<var>' (numeric var)"
MAX_VARS = 50
MAX_ASSERTS = 100
MAX_EXPR_CHARS = 1000
MAX_DEPTH = 48
MAX_LITERAL_DIGITS = 18
MAX_NAME_CHARS = 32

_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
# SMT-LIB / Z3 symbols that a declared constant must not shadow (hyphenated commands cannot match
# the identifier pattern anyway).
_RESERVED = frozenset(
    [
        "_",
        "as",
        "let",
        "par",
        "forall",
        "exists",
        "ite",
        "div",
        "mod",
        "abs",
        "to_real",
        "to_int",
        "is_int",
        "distinct",
        "xor",
        "true",
        "false",
        "and",
        "or",
        "not",
        "Int",
        "Real",
        "Bool",
        "assert",
    ]
)
_CMP = frozenset(("==", "!=", "<", "<=", ">", ">="))
_WORDS = frozenset(("and", "or", "not", "true", "false"))
_TOKEN = re.compile(
    r"""\s*(?:
        (?P<num>[0-9]+(?:\.[0-9]+)?)
      | (?P<name>[A-Za-z_][A-Za-z0-9_]*)
      | (?P<op>==|!=|<=|>=|<|>|\+|-|\*|\(|\))
    )""",
    re.VERBOSE | re.ASCII,
)


class DslError(ValueError):
    """The constraint problem is malformed; the message is safe to return to the caller."""


@dataclass(frozen=True)
class Compiled:
    smt2: str
    var_types: dict[str, str]
    goal: str  # "sat" | "min" | "max"
    goal_var: str | None


@dataclass(frozen=True)
class _Typed:
    smt: str
    type: str


def _tokenize(text: str) -> list[tuple[str, str]]:
    tokens: list[tuple[str, str]] = []
    pos = 0
    text = text.rstrip()
    while pos < len(text):
        m = _TOKEN.match(text, pos)
        if m is None or m.end() == pos:
            raise DslError(f"unexpected character {text[pos:].lstrip()[:1]!r} in {text[:60]!r}")
        pos = m.end()
        if m.group("num") is not None:
            tokens.append(("num", m.group("num")))
        elif m.group("name") is not None:
            word = m.group("name")
            tokens.append(("kw" if word in _WORDS else "name", word))
        else:
            tokens.append(("op", m.group("op")))
    return tokens


class _Parser:
    def __init__(self, tokens: list[tuple[str, str]], var_types: dict[str, str]) -> None:
        self._tokens, self._vars, self._i, self._depth = tokens, var_types, 0, 0

    def _peek(self) -> tuple[str, str] | None:
        return self._tokens[self._i] if self._i < len(self._tokens) else None

    def _take(self, value: str) -> bool:
        tok = self._peek()
        if tok is not None and tok[0] in ("op", "kw") and tok[1] == value:
            self._i += 1
            return True
        return False

    def parse(self) -> _Typed:
        node = self._or()
        if self._peek() is not None:
            raise DslError(f"unexpected token {self._peek()[1]!r}")  # type: ignore[index]
        if node.type != "bool":
            raise DslError("an assertion must be a boolean expression")
        return node

    def _enter(self) -> None:
        self._depth += 1
        if self._depth > MAX_DEPTH:
            raise DslError(f"expression nested deeper than {MAX_DEPTH}")

    def _or(self) -> _Typed:
        terms = [self._and()]
        while self._take("or"):
            terms.append(self._and())
        return self._connective("or", terms)

    def _and(self) -> _Typed:
        terms = [self._not()]
        while self._take("and"):
            terms.append(self._not())
        return self._connective("and", terms)

    @staticmethod
    def _connective(op: str, terms: list[_Typed]) -> _Typed:
        if len(terms) == 1:
            return terms[0]
        if any(t.type != "bool" for t in terms):
            raise DslError(f"'{op}' needs boolean operands")
        return _Typed(f"({op} {' '.join(t.smt for t in terms)})", "bool")

    def _not(self) -> _Typed:
        if self._take("not"):
            self._enter()
            inner = self._not()
            self._depth -= 1
            if inner.type != "bool":
                raise DslError("'not' needs a boolean operand")
            return _Typed(f"(not {inner.smt})", "bool")
        return self._cmp()

    def _cmp(self) -> _Typed:
        left = self._sum()
        tok = self._peek()
        if tok is None or tok[0] != "op" or tok[1] not in _CMP:
            return left
        self._i += 1
        right = self._sum()
        nxt = self._peek()
        if nxt is not None and nxt[0] == "op" and nxt[1] in _CMP:
            raise DslError("chained comparisons are not supported; join them with 'and'")
        op = tok[1]
        if left.type == "bool" or right.type == "bool":
            if left.type != right.type or op not in ("==", "!="):
                raise DslError(f"cannot apply '{op}' to {left.type} and {right.type}")
            core = f"(= {left.smt} {right.smt})"
        else:
            a, b = self._unify(left, right)
            smt_op = {"==": "=", "!=": "=", "<": "<", "<=": "<=", ">": ">", ">=": ">="}[op]
            core = f"({smt_op} {a.smt} {b.smt})"
        return _Typed(f"(not {core})" if op == "!=" else core, "bool")

    @staticmethod
    def _promote(node: _Typed) -> _Typed:
        if node.type == "real":
            return node
        return _Typed(f"(to_real {node.smt})", "real")

    def _unify(self, left: _Typed, right: _Typed) -> tuple[_Typed, _Typed]:
        if left.type == right.type:
            return left, right
        return self._promote(left), self._promote(right)

    def _arith(self, op: str, items: list[_Typed]) -> _Typed:
        if any(t.type == "bool" for t in items):
            raise DslError(f"'{op}' needs numeric operands")
        kind = "real" if any(t.type == "real" for t in items) else "int"
        if kind == "real":
            items = [self._promote(t) for t in items]
        return _Typed(f"({op} {' '.join(t.smt for t in items)})", kind)

    def _sum(self) -> _Typed:
        node = self._prod()
        while (tok := self._peek()) is not None and tok[0] == "op" and tok[1] in ("+", "-"):
            self._i += 1
            node = self._arith(tok[1], [node, self._prod()])
        return node

    def _prod(self) -> _Typed:
        node = self._unary()
        while self._take("*"):
            node = self._arith("*", [node, self._unary()])
        return node

    def _unary(self) -> _Typed:
        if self._take("-"):
            self._enter()
            inner = self._unary()
            self._depth -= 1
            if inner.type == "bool":
                raise DslError("unary '-' needs a numeric operand")
            return _Typed(f"(- {inner.smt})", inner.type)
        return self._atom()

    def _atom(self) -> _Typed:
        tok = self._peek()
        if tok is None:
            raise DslError("unexpected end of expression")
        kind, value = tok
        self._i += 1
        if kind == "num":
            digits = value.replace(".", "")
            if len(digits) > MAX_LITERAL_DIGITS:
                raise DslError(f"numeric literal longer than {MAX_LITERAL_DIGITS} digits")
            return _Typed(value, "real" if "." in value else "int")
        if kind == "kw" and value in ("true", "false"):
            return _Typed(value, "bool")
        if kind == "name":
            if value not in self._vars:
                raise DslError(f"undeclared variable {value!r}")
            return _Typed(value, self._vars[value])
        if kind == "op" and value == "(":
            self._enter()
            inner = self._or()
            self._depth -= 1
            if not self._take(")"):
                raise DslError("missing ')'")
            return inner
        raise DslError(f"unexpected token {value!r}")


def _check_vars(raw: Any) -> dict[str, str]:
    if not isinstance(raw, dict) or not raw:
        raise DslError("'vars' must be a non-empty object {name: 'int'|'bool'|'real'}")
    if len(raw) > MAX_VARS:
        raise DslError(f"too many variables: {len(raw)} > {MAX_VARS}")
    out: dict[str, str] = {}
    for name, typ in raw.items():
        if not isinstance(name, str) or not _NAME.match(name) or len(name) > MAX_NAME_CHARS:
            raise DslError(f"invalid variable name {str(name)[:40]!r}")
        if name in _RESERVED:
            raise DslError(f"variable name {name!r} is reserved")
        if typ not in VAR_TYPES:
            raise DslError(f"variable {name!r}: type must be one of {list(VAR_TYPES)}")
        out[name] = typ
    return out


def _check_goal(goal: Any, var_types: dict[str, str]) -> tuple[str, str | None]:
    if goal is None or goal == "sat":
        return "sat", None
    m = re.fullmatch(r"(min|max):([A-Za-z_][A-Za-z0-9_]*)", goal) if isinstance(goal, str) else None
    if m is None:
        raise DslError(f"goal must be {GOALS_HELP}")
    kind, var = m.groups()
    if var_types.get(var) not in ("int", "real"):
        raise DslError(f"goal variable {var!r} must be a declared int or real")
    return kind, var


def compile_constraints(spec: Any) -> Compiled:
    """Validate and compile a constraint problem; raises ``DslError`` on any violation."""
    if not isinstance(spec, dict):
        raise DslError("'constraints' must be an object {vars, assert, goal?}")
    unknown = sorted(set(spec) - {"vars", "assert", "goal"})
    if unknown:
        raise DslError(f"unknown constraint keys: {unknown}")
    var_types = _check_vars(spec.get("vars"))
    asserts = spec.get("assert")
    if not isinstance(asserts, list) or not asserts:
        raise DslError("'assert' must be a non-empty list of strings")
    if len(asserts) > MAX_ASSERTS:
        raise DslError(f"too many assertions: {len(asserts)} > {MAX_ASSERTS}")
    goal, goal_var = _check_goal(spec.get("goal"), var_types)
    lines = [f"(declare-const {n} {t.capitalize()})" for n, t in var_types.items()]
    for i, text in enumerate(asserts):
        if not isinstance(text, str) or not text.strip():
            raise DslError(f"assert[{i}] must be a non-empty string")
        if len(text) > MAX_EXPR_CHARS:
            raise DslError(f"assert[{i}] exceeds {MAX_EXPR_CHARS} characters")
        try:
            node = _Parser(_tokenize(text), var_types).parse()
        except DslError as exc:
            raise DslError(f"assert[{i}]: {exc}") from None
        lines.append(f"(assert {node.smt})")
    return Compiled("\n".join(lines), var_types, goal, goal_var)


def bound_assertion(var: str, var_type: str, op: str, value: Fraction | int) -> str:
    """``(assert (<op> var value))`` with ``op`` in ``<``, ``<=``, ``>``, ``>=``; exact numerals."""
    if op not in ("<", "<=", ">", ">="):
        raise ValueError(f"unsupported bound operator {op!r}")
    return f"(assert ({op} {var} {_numeral(value, var_type)}))"


def _numeral(value: Fraction | int, var_type: str) -> str:
    frac = Fraction(value)
    if var_type == "int":
        if frac.denominator != 1:
            raise ValueError("integer bound must be integral")
        n = frac.numerator
        return str(n) if n >= 0 else f"(- {-n})"
    num = f"{abs(frac.numerator)}.0"
    text = num if frac.denominator == 1 else f"(/ {num} {frac.denominator}.0)"
    return text if frac >= 0 else f"(- {text})"
