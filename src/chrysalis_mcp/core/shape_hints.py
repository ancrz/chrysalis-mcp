"""Shape-based redirect hints for FAILED parameter validation (data table, one place).

Small models sometimes send the params of a *meta* verb to a domain action (a task descriptor to
``reason.solve``, a feedback record to ``memory.ingest``). When validation already failed and the
supplied key set matches a row below, the error envelope gains ``did_you_mean_action`` /
``did_you_mean_example`` (same contract as the in-domain redirect in ``registry._check_params``)
plus a ``hint``. Valid calls are never touched. Domains do not import this module; only the
registry consumes it.
"""

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from . import learning

Params = dict[str, Any]


@dataclass(frozen=True)
class ShapeRule:
    """``domain``/``actions``: where it applies (``None`` actions = any action of the domain)."""

    domain: str
    actions: frozenset[str] | None
    matches: Callable[[Params], bool]
    target: str  # "<domain>.<action>" suggested instead
    example: Callable[[str, str, Params], Params]  # (domain, action, params) -> target params
    hint: str


_TASK_KEYS = ("n_vars", "goal")
_SOLVER_INPUTS = ("constraints", "smt2")


def _is_task_descriptor(params: Params) -> bool:
    return any(k in params for k in _TASK_KEYS) and not any(k in params for k in _SOLVER_INPUTS)


MAX_ECHO_CHARS = 64  # every echoed string is bounded; the hint never grows with the input
_ACTION_RE = re.compile(r"^[a-z_]{1,32}\.[a-z_]{1,32}$")
_PLACEHOLDER_ACTION = "reason.solve"


def _bounded_scalar(value: Any) -> bool:
    if isinstance(value, bool):
        return True
    if isinstance(value, str):
        return len(value) <= MAX_ECHO_CHARS
    return isinstance(value, (int, float)) and abs(value) < 10**12


def _advise_example(domain: str, action: str, params: Params) -> Params:
    # advise.task takes features: n_vars (number) and goal (short string); anything else is dropped
    task: Params = {}
    n_vars = params.get("n_vars")
    if isinstance(n_vars, int) and not isinstance(n_vars, bool) and _bounded_scalar(n_vars):
        task["n_vars"] = n_vars
    goal = params.get("goal")
    if isinstance(goal, str) and _bounded_scalar(goal):
        task["goal"] = goal
    return {"domain": domain, "action": action, "task": task}


def _is_feedback_record(params: Params) -> bool:
    return (
        "action" in params
        and ("outcome" in params or "used" in params)
        and not any(k in params for k in ("triples", "text"))
    )


def _feedback_example(_domain: str, _action: str, params: Params) -> Params:
    """A call valid for meta.feedback: bad or oversized values give way to placeholders."""
    action = params.get("action")
    used = params.get("used")
    outcome = params.get("outcome")
    task = params.get("task")
    out: Params = {
        "action": action
        if isinstance(action, str) and _ACTION_RE.match(action)
        else _PLACEHOLDER_ACTION,
        "used": used if isinstance(used, bool) else True,
        "outcome": outcome if outcome in learning.OUTCOMES else learning.OUTCOMES[0],
    }
    if isinstance(task, dict) and len(task) <= 6:
        small = {
            k: v for k, v in task.items() if len(str(k)) <= MAX_ECHO_CHARS and _bounded_scalar(v)
        }
        if len(small) == len(task):
            out["task"] = small
    return out


SHAPE_RULES: tuple[ShapeRule, ...] = (
    ShapeRule(
        domain="reason",
        actions=None,
        matches=_is_task_descriptor,
        target="meta.advise",
        example=_advise_example,
        hint=(
            "n_vars/goal describe a task, not a problem: ask meta.advise whether to use this "
            "action (params: domain, action, task), solve with 'constraints' or 'smt2', then "
            "record the outcome with meta.feedback."
        ),
    ),
    ShapeRule(
        domain="memory",
        actions=frozenset({"ingest"}),
        matches=_is_feedback_record,
        target="meta.feedback",
        example=_feedback_example,
        hint=(
            "This is a tool-outcome record, not knowledge (memory.ingest stores 'triples'): "
            "call meta.handshake first (or pass model) so meta.feedback knows the model."
        ),
    ),
)


def shape_hint(domain: str, action: str, params: Params) -> Params | None:
    """Fields to merge into a failing envelope, or ``None`` when no rule matches."""
    for rule in SHAPE_RULES:
        if rule.domain != domain or (rule.actions is not None and action not in rule.actions):
            continue
        if rule.matches(params):
            target_domain, target_action = rule.target.split(".")
            return {
                "did_you_mean_action": rule.target,
                "did_you_mean_example": {
                    "domain": target_domain,
                    "action": target_action,
                    "params": rule.example(domain, action, params),
                },
                "hint": rule.hint,
            }
    return None
