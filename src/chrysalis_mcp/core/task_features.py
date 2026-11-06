"""Task features: cheap, side-effect-free measurements of one concrete call.

An advisory can depend on the *task*, not only on the model (a solver that hurts on a two-variable
puzzle helps on a ten-constraint one). Features are computed from the call params with regexes
only (nothing is parsed with ``eval`` or sent to Z3) and are the vocabulary of the ``task``
condition in ``data/evidence.json``:

- ``n_vars``, ``n_atoms`` (comparison operators), ``n_asserts``, ``goal`` (``sat`` | ``opt``) for
  ``reason.solve`` / ``reason.check`` from either ``constraints`` (DSL) or ``smt2``.
- ``facts`` / ``ctx_tokens`` -> derived ``facts_exceed_ctx`` for memory (caller-supplied through
  ``meta.advise``: the store size is not a call parameter).
"""

import math
import re
from typing import Any

FEATURES = ("n_vars", "n_atoms", "n_asserts", "goal", "facts", "ctx_tokens", "facts_exceed_ctx")
REASON_ACTIONS = ("reason.solve", "reason.check")
CHARS_PER_TOKEN = 2.5  # conservative for fictional names (real ratio ~3); one source of truth
AVG_FACT_CHARS = 39  # mean length of the one-line facts of the validation sets (H-a, H-a-scale)
TOKENS_PER_FACT = math.ceil(AVG_FACT_CHARS / CHARS_PER_TOKEN)  # 16: never underestimates the window
CTX_FILL = 0.75  # share of the window the facts may take before the question and answer need room
_DSL_CMP = re.compile(r"==|!=|<=|>=|<|>")
_SMT_CMP = re.compile(r"\((?:=|<=|>=|<|>|distinct)\s")
_SMT_DECL = re.compile(r"\((?:declare-const|declare-fun)\s")
_SMT_ASSERT = re.compile(r"\(assert\s")
_SMT_OPT = re.compile(r"\((?:minimize|maximize)\s")
_MAX_ASSERTS = 1000  # cap scanned constraints, same reason
_MAX_TEXT = 20_000  # cap scanned text: features must stay O(1)-ish on hostile input


def _reason(params: dict[str, Any]) -> dict[str, Any]:
    spec = params.get("constraints")
    if isinstance(spec, dict):
        raw = spec.get("assert")  # untrusted: any JSON type may arrive here
        asserts = (
            [a for a in raw[:_MAX_ASSERTS] if isinstance(a, str)]
            if isinstance(raw, list | tuple)
            else []
        )
        goal = spec.get("goal")
        variables = spec.get("vars")
        return {
            "n_vars": len(variables) if isinstance(variables, dict) else 0,
            "n_asserts": len(asserts),
            "n_atoms": sum(len(_DSL_CMP.findall(a[:_MAX_TEXT])) for a in asserts),
            "goal": "opt" if isinstance(goal, str) and goal.startswith(("min:", "max:")) else "sat",
        }
    smt2 = params.get("smt2")
    if isinstance(smt2, str):
        text = smt2[:_MAX_TEXT]
        return {
            "n_vars": len(_SMT_DECL.findall(text)),
            "n_asserts": len(_SMT_ASSERT.findall(text)),
            "n_atoms": len(_SMT_CMP.findall(text)),
            "goal": "opt" if _SMT_OPT.search(text) else "sat",
        }
    return {}


def derive(features: dict[str, Any], context: int | None = None) -> dict[str, Any]:
    """Add ``facts_exceed_ctx`` when ``facts`` and a context size (feature or profile) are known."""
    out = dict(features)
    ctx_tokens = out.get("ctx_tokens", context)
    facts = out.get("facts")
    numeric = (isinstance(facts, int | float) and not isinstance(facts, bool)) and (
        isinstance(ctx_tokens, int | float) and not isinstance(ctx_tokens, bool)
    )
    if "facts_exceed_ctx" not in out and numeric:
        out["facts_exceed_ctx"] = facts * TOKENS_PER_FACT > ctx_tokens * CTX_FILL
    return out


def extract(action: str, params: dict[str, Any] | None) -> dict[str, Any]:
    """Features of one call; empty when the action has none or the params carry none."""
    if params and action in REASON_ACTIONS:
        return _reason(params)
    return {}


def _bound_hint(cond: Any, lo: str, hi: str, both: str) -> str | None:
    if not isinstance(cond, dict):
        return None
    low, high = cond.get("min"), cond.get("max")
    if low is not None and high is not None:
        return both.format(lo=f"{low:g}", hi=f"{high:g}")
    if low is not None:
        return lo.format(n=f"{low:g}")
    if high is not None:
        return hi.format(n=f"{high:g}")
    return None


_GOAL_WORDS = {
    "opt": "asks for the smallest/largest value",
    "sat": "only asks for any valid values",
}
_BOUND_PHRASES = {  # feature -> (min only, max only, min and max); model-countable wording
    "n_atoms": (
        "about {n}+ comparisons, counting each lower/upper bound",
        "about {n} or fewer comparisons counting bounds",
        "about {lo} to {hi} comparisons counting bounds",
    ),
    "n_vars": ("{n}+ variables", "{n} or fewer variables", "{lo} to {hi} variables"),
    "n_asserts": (
        "{n}+ listed conditions",
        "{n} or fewer listed conditions",
        "{lo} to {hi} listed conditions",
    ),
    "facts": ("{n}+ stored facts", "{n} or fewer stored facts", "{lo} to {hi} stored facts"),
}


def hint_text(feature: str, cond: Any) -> str | None:
    """Plain-language phrase for one task condition; ``None`` when the feature has no phrasing.

    The wording is model-countable (variables, listed conditions, min/max) on purpose: the DSL
    form (``n_atoms>=7``) is not evaluable by a small model without thinking.
    """
    if feature in _BOUND_PHRASES:
        return _bound_hint(cond, *_BOUND_PHRASES[feature])
    if feature == "goal":
        values = cond if isinstance(cond, list) else [cond]
        words = [_GOAL_WORDS[v] for v in values if isinstance(v, str) and v in _GOAL_WORDS]
        return " or ".join(words) or None
    if feature == "facts_exceed_ctx" and isinstance(cond, bool):
        return (
            "the facts do not fit in your context window"
            if cond
            else "the facts fit in your context window"
        )
    return None
