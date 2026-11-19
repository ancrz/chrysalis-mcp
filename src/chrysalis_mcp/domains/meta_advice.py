"""``meta`` actions that make the advisory learn from outcomes: feedback, advice_stats,
advice_reset.

Also the blend used by ``meta.advise`` and the handshake prompt hint (see ``core/learning.py``).
Reward: either the caller reports ``outcome``, or passes ``problem`` + ``assignment`` and the
server derives it with ``reason.check`` (verification is a free reward signal). For ``goal``
min/max the reward is success only when the check passes AND the assignment reaches the optimum
proven by ``reason.solve``; an optimum that cannot be proven (timeout, unbounded) or an ``smt2``
problem with minimize/maximize cannot be judged and is rejected without recording anything.
"""

import sqlite3
from dataclasses import replace
from fractions import Fraction
from typing import Any

from ..core import learning, task_features
from ..core.profile import Profile
from ..core.registry import Context, Params, Payload, Registry

ADVICE_RETRY_AFTER_S = 2
MAX_PAGE = 200
DEFAULT_PAGE = 20


def parse_task(params: Params) -> tuple[dict[str, Any], Payload | None]:
    """Validated ``task`` features (or an ERROR payload), shared by advise and feedback."""
    task = params.get("task")
    if task is not None and not isinstance(task, dict):
        return {}, {"status": "ERROR", "error": "task must be an object of task features"}
    unknown = sorted(set(task or {}) - set(task_features.FEATURES))
    if unknown:
        return {}, {
            "status": "ERROR",
            "error": f"unknown task features {unknown}",
            "valid": list(task_features.FEATURES),
        }
    return dict(task or {}), None


def nest_task(params: Params) -> tuple[Params, list[str]]:
    """Lenient coercion for advise/feedback: top-level task features move into ``task``."""
    flat = [k for k in task_features.FEATURES if k in params]
    if not flat or ("task" in params and not isinstance(params["task"], dict)):
        return params, []
    rest = {k: v for k, v in params.items() if k not in flat}
    task = {**(params.get("task") or {}), **{k: params[k] for k in flat}}
    return {**rest, "task": task}, ["task_features_nested"]


def identity(profile: Profile) -> tuple[str, str] | None:
    """``(model key, thinking key)`` of a declared profile; ``None`` before any handshake."""
    key = learning.model_key(profile.model, profile.family, profile.params_b)
    return None if key is None else (key, learning.thinking_key(profile.thinking))


def blend(
    ctx: Context,
    name: str,
    features: dict[str, Any],
    rule_advisory: dict[str, Any],
    *,
    explore: Any = None,
    min_n: Any = None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """``(advisory, evidence)``: the rule advisory until the bucket has ``min_n`` outcomes, then
    the arm with the better posterior. ``evidence`` is ``None`` when the session has no model."""
    who = identity(ctx.registry.profiles.resolve(ctx.session_id))
    if who is None:
        return rule_advisory, None
    model, thinking = who
    bucket = learning.bucket_key(features)
    threshold, sampling = learning.min_n_value(min_n), learning.explore_enabled(explore)
    used, own = ctx.registry.advice.cells(model, thinking, name, bucket)
    seed = learning.seed_for(
        ctx.session_id or "", (model, thinking, name, bucket), sum(used) + sum(own)
    )
    choice = learning.decide(used, own, min_n=threshold, explore=sampling, seed=seed)
    if choice is None:
        ev = learning.evidence("rule", used, own, bucket=bucket, min_n=threshold, explore=sampling)
        return rule_advisory, ev
    ev = learning.evidence("learned", used, own, bucket=bucket, min_n=threshold, explore=sampling)
    tool = "using the tool" if choice == "used" else "answering yourself"
    advisory = {
        "level": "recommended" if choice == "used" else "discouraged",
        "why": (
            f"learned: {tool} succeeded more for this model on this bucket "
            f"(used {ev['success_rate_used']} n={ev['n_used']}, "
            f"self {ev['success_rate_self']} n={ev['n_self']})"
        ),
        "source": "learned",
        "hint": (
            "for tasks like this, the tool has worked better for you"
            if choice == "used"
            else "for tasks like this, answering yourself has worked better for you"
        ),
    }
    return advisory, ev


def learned_hint_text(ctx: Context) -> str:
    """Learned plain-language lines for the session's model (``""`` if none or no handshake)."""
    who = identity(ctx.registry.profiles.resolve(ctx.session_id))
    if who is None:
        return ""
    lines = learning.learned_lines(ctx.registry.advice, who[0], who[1], learning.min_n_value())
    return " ".join(lines)


# --- reward ------------------------------------------------------------------------------------


def _goal_var(problem: dict[str, Any]) -> tuple[str, str] | None:
    spec = problem.get("constraints")
    goal = spec.get("goal") if isinstance(spec, dict) else None
    if isinstance(goal, str) and goal.startswith(("min:", "max:")):
        sense, var = goal.split(":", 1)
        return sense, var.strip()
    return None


def _err(message: str, **extra: Any) -> Payload:
    return {"status": "ERROR", "error": message, **extra}


def derive_outcome(
    problem: dict[str, Any], assignment: dict[str, Any], ctx: Context
) -> tuple[str, str] | Payload:
    """``(outcome, reason)`` from the solver, or an ERROR payload when it cannot be judged."""
    allowed = {"smt2", "constraints", "timeout_ms"}
    extra = sorted(set(problem) - allowed)
    if extra:
        return _err(f"problem has unknown keys {extra}", valid=sorted(allowed))
    reg = ctx.registry
    checked = reg.dispatch(
        "reason", "check", {**problem, "assignment": assignment, "override": True}, ctx
    )
    if checked.get("status") != "OK" or not isinstance(checked.get("valid"), bool):
        return _err(
            "reason.check could not judge the assignment; nothing recorded "
            "(pass outcome='failure' explicitly if the answer was malformed)",
            check=checked.get("error") or checked.get("summary"),
        )
    if not checked["valid"]:
        return "failure", "assignment violates the constraints"
    goal = _goal_var(problem)
    smt2 = problem.get("smt2")
    if (
        goal is None
        and isinstance(smt2, str)
        and task_features.extract("reason.check", problem)["goal"] == "opt"
    ):
        return _err("smt2 with minimize/maximize cannot be judged for optimality; pass outcome")
    if goal is None:
        return "success", "assignment satisfies the constraints"
    sense, var = goal
    solved = reg.dispatch("reason", "solve", {**problem, "override": True}, ctx)
    objective = solved.get("objective")
    if (
        solved.get("status") != "OK"
        or not isinstance(objective, dict)
        or not objective.get("optimal")
    ):
        return _err(
            "optimum not proven by reason.solve; nothing recorded", solve=solved.get("summary")
        )
    if var not in assignment:
        return "failure", f"assignment lacks the goal variable {var!r}"
    try:
        reached = Fraction(str(assignment[var])) == Fraction(str(objective["value"]))
    except ValueError, ZeroDivisionError:
        return "failure", f"goal variable {var!r} is not numeric"
    verdict = "optimal" if reached else "valid but not optimal"
    return ("success" if reached else "failure"), f"{sense} {var}: assignment {verdict}"


# --- actions -----------------------------------------------------------------------------------


def _feedback(params: Params, ctx: Context) -> Payload:
    who, implicit, problem = _session_identity(params, ctx)
    if problem is not None or who is None:
        return problem or _err("no model identity")
    name, used = params.get("action"), params.get("used")
    known = {f"{d}.{a}" for d, acts in ctx.registry.schema()["domains"].items() for a in acts}
    if not isinstance(name, str) or name not in known or name.startswith("meta."):
        return _err(f"unknown or non-advisable action {name!r}", valid=sorted(known))
    if not isinstance(used, bool):
        return _err("used must be a boolean (did the model use the tool)")
    features, problem = parse_task(params)
    if problem is not None:
        return problem
    outcome, spec, assignment = (
        params.get("outcome"),
        params.get("problem"),
        params.get("assignment"),
    )
    reason = "reported by caller"
    if outcome is not None:
        if spec is not None or assignment is not None:
            return _err("give outcome OR problem+assignment, not both")
        if outcome not in learning.OUTCOMES:
            return _err("outcome must be 'success' or 'failure'", valid=list(learning.OUTCOMES))
    else:
        if not isinstance(spec, dict) or not isinstance(assignment, dict):
            return _err("without outcome give problem (object) and assignment (object)")
        if not name.startswith("reason."):
            return _err("reward derivation is only available for reason.* actions; pass outcome")
        derived = derive_outcome(spec, assignment, ctx)
        if isinstance(derived, dict):
            return derived
        outcome, reason = derived
        if not features:
            features = task_features.derive(task_features.extract(name, spec))
    model, thinking = who
    bucket = learning.bucket_key(features)
    arm = "used" if used else "self"
    try:
        cell = ctx.registry.advice.record(model, thinking, name, bucket, arm, outcome == "success")
    except sqlite3.Error as exc:  # typically "database is locked" under concurrent writers
        return _err(
            f"advice store unavailable ({exc}); nothing was recorded",
            retryable=True,
            retry_after_s=ADVICE_RETRY_AFTER_S,
            next_action=f"Retry meta.feedback in about {ADVICE_RETRY_AFTER_S} seconds.",
        )
    used_cell, own_cell = ctx.registry.advice.cells(model, thinking, name, bucket)
    ev = learning.evidence(
        "recorded", used_cell, own_cell, bucket=bucket,
        min_n=learning.min_n_value(), explore=learning.explore_enabled(),
    )  # fmt: skip
    return {
        **({"implicit_handshake": True} if implicit else {}),
        "status": "OK",
        "summary": f"{name} {arm} {outcome} recorded for {model} ({bucket})",
        "model": model,
        "thinking": thinking,
        "bucket": bucket,
        "arm": arm,
        "outcome": outcome,
        "reward_reason": reason,
        "cell": {"successes": cell[0], "failures": cell[1]},
        "evidence": ev,
        "next_action": "Keep sending feedback; meta.advise switches to learned advice at min_n.",
        "terse": {"outcome": outcome, "bucket": bucket, "arm": arm},
    }


HANDSHAKE_EXAMPLE = {
    "domain": "meta",
    "action": "handshake",
    "params": {"model": "<your model name>"},
}


def _no_model(what: str) -> Payload:
    return _err(
        f"no model: pass model or call meta.handshake first ({what})",
        example=HANDSHAKE_EXAMPLE,
        next_action="Send the handshake 'example' once per session, or pass params.model.",
    )


def _session_identity(
    params: Params, ctx: Context
) -> tuple[tuple[str, str] | None, bool, Payload | None]:
    """``(identity, implicit, error)``. Order: the session's declared/forced profile; else the
    call's own ``model`` (declared for the session as a minimal handshake); else an error."""
    profile = ctx.registry.profiles.resolve(ctx.session_id)
    who = identity(profile)
    if who is not None:
        return who, False, None
    given = params.get("model")
    if given is None:
        return None, False, _no_model("feedback")
    key = learning.model_key(given) if isinstance(given, str) else None
    if key is None:
        return None, False, _err("model must be a non-empty string")
    # Only model is declared (no params_b); client stays unset so clientInfo audit is untouched.
    declared = replace(profile, source="declared", model=given.strip(), scores={}, note=None)
    ctx.registry.profiles.declare(ctx.session_id, declared)
    return (key, learning.thinking_key(declared.thinking)), True, None


def _target_model(params: Params, ctx: Context) -> tuple[str | None, Payload | None]:
    given = params.get("model")
    if given is not None:
        key = learning.model_key(given) if isinstance(given, str) else None
        return (key, None) if key else (None, _err("model must be a non-empty string"))
    who = identity(ctx.registry.profiles.resolve(ctx.session_id))
    if who is None:
        return None, _no_model("this action")
    return who[0], None


def _advice_stats(params: Params, ctx: Context) -> Payload:
    model, problem = _target_model(params, ctx)
    if problem is not None or model is None:
        return problem or _err("no model")
    limit, cursor = params.get("limit", DEFAULT_PAGE), params.get("cursor", "0")
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= MAX_PAGE:
        return _err(f"limit must be an integer in [1, {MAX_PAGE}]")
    if not isinstance(cursor, str) or not cursor.isdigit():
        return _err("cursor must be the string returned by the previous page")
    offset = int(cursor)
    rows, total = ctx.registry.advice.rows(model, params.get("action"), offset, limit)
    nxt = offset + len(rows)
    more = nxt < total
    return {
        "status": "OK",
        "summary": f"{len(rows)} of {total} learned rows for {model}",
        "model": model,
        "rows": rows,
        "total": total,
        "cursor": str(nxt) if more else None,
        "truncated": more,
        "next_action": "Pass cursor to read the next page."
        if more
        else "Learning is read-only here.",
        "terse": {"model": model, "total": total},
    }


def _advice_reset(params: Params, ctx: Context) -> Payload:
    if params.get("confirm") is not True:
        return _err(
            "reset requires confirm=true (irreversible: deletes learned counts)",
            next_action="Retry with confirm=true to delete the model's learned counts.",
        )
    model, problem = _target_model(params, ctx)
    if problem is not None or model is None:
        return problem or _err("no model")
    deleted = ctx.registry.advice.reset(model, params.get("action"))
    return {
        "status": "OK",
        "summary": f"{deleted} learned rows deleted for {model}",
        "model": model,
        "deleted": deleted,
        "terse": {"model": model, "deleted": deleted},
    }


def register(registry: Registry) -> None:
    registry.register(
        "meta",
        "feedback",
        _feedback,
        params_schema={
            "action": {"type": "string", "required": True},
            "used": {"type": "boolean", "required": True},
            "task": {"type": "object", "required": False},
            "outcome": {"type": "string", "required": False, "enum": list(learning.OUTCOMES)},
            "problem": {"type": "object", "required": False},
            "assignment": {"type": "object", "required": False},
            "model": {"type": "string", "required": False},
        },
        coerce=nest_task,
        description=(
            "Record one outcome so meta.advise learns per model (handshake or 'model'). 'used': "
            "did the model call the tool. Give 'outcome' (success|failure), or 'problem' "
            "({constraints|smt2}) + 'assignment' and the server derives the reward with "
            "reason.check (min/max goals: success only if also optimal, proven by reason.solve). "
            "Stored in a separate db (CHRYSALIS_ADVICE_DB)."
        ),
        example={
            "action": "reason.solve",
            "used": True,
            "task": {"n_atoms": 9, "goal": "sat"},
            "outcome": "success",
        },
    )
    registry.register(
        "meta",
        "advice_stats",
        _advice_stats,
        params_schema={
            "model": {"type": "string", "required": False},
            "action": {"type": "string", "required": False},
            "limit": {"type": "integer", "required": False, "min": 1, "max": MAX_PAGE},
            "cursor": {"type": "string", "required": False},
        },
        description=(
            "Read-only: learned success/failure counts per (thinking, action, bucket, arm) for "
            "the session's model (or 'model'); paginated with cursor."
        ),
        example={"action": "reason.solve", "limit": 10},
    )
    registry.register(
        "meta",
        "advice_reset",
        _advice_reset,
        params_schema={
            "confirm": {"type": "boolean", "required": True},
            "model": {"type": "string", "required": False},
            "action": {"type": "string", "required": False},
        },
        description=(
            "Delete the learned counts of ONE model (the session's, or 'model'; optionally one "
            "'action'). Requires confirm=true; advice returns to the rules."
        ),
        example={"confirm": True, "action": "reason.solve"},
    )
