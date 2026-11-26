"""H-bio: the bio-actions smoke made fair. 15 independent steps, each on a fresh substrate.

The model gets one natural-language task per step and must call the target ``domain.action``
(with the required params) through the single ``chrysalis`` tool. The harness pre-seeds whatever
the step needs (facts, teaches, a trained FlyVec, a handshake, one feedback), so a failing step
never contaminates the next. Scoring per step: the target was called with its required params
subset (curated param synonyms accepted, as the server does) AND the server answered OK/PARTIAL.
"""

import json
import os
from dataclasses import dataclass
from typing import Any

from chrysalis_mcp.core.registry import AUTO_RENAMES, Registry
from chrysalis_mcp.domains import build_registry
from chrysalis_mcp.server import coerce_params

ENV_REPS = "CHRYSALIS_EVAL_BIO_REPS"
DEFAULT_REPS = 3
BIO_MAX_STEPS = 6  # agent loop cap per step (as the original smoke)
OK_STATUSES = ("OK", "PARTIAL")

SYSTEM = (
    "You operate the 'chrysalis' tool. Do exactly the requested operation with the tool, "
    "using the exact domain, action and params it needs, then answer in one short sentence."
)
FACTS = (
    ("Lumen", "is_a", "orbital telescope"),
    ("Lumen", "launched_in", "2031"),
    ("Lumen", "operated_by", "Andes Space Agency"),
    ("Andes Space Agency", "headquartered_in", "Quito"),
    ("Kestrel", "is_a", "lunar rover"),
    ("Kestrel", "operated_by", "Andes Space Agency"),
)
TEACH_1 = {"query": "the space eye of the Andes", "node": "Lumen", "delta": 1}
TEACH_2 = {"query": "Andean sky observatory in orbit", "node": "Lumen", "delta": 1}
FEEDBACK_SEED = {
    "action": "reason.solve",
    "used": True,
    "task": {"n_vars": 6, "goal": "sat"},
    "outcome": "success",
}


@dataclass(frozen=True)
class BioStep:
    sid: str
    task: str  # may contain {model}, {family}, {params_b}
    targets: tuple[str, ...]  # any of these "domain.action" counts
    want: dict[str, Any]  # required params: key -> value (None = any value)
    seed: tuple[str, ...]  # prerequisite ids, applied in order (see ``_SEEDERS``)


STEPS: tuple[BioStep, ...] = (
    BioStep(
        "handshake",
        "Declare yourself to the tool: you are model '{model}', family '{family}', "
        "params_b {params_b}.",
        ("meta.handshake",),
        {},
        (),
    ),
    BioStep(
        "ingest",
        "Store these facts in memory as triples: "
        + "; ".join(f"{s} {r} {o}" for s, r, o in FACTS)
        + ".",
        ("memory.ingest",),
        {"triples": None},
        ("handshake",),
    ),
    BioStep(
        "teach1",
        "Teach memory that the query 'the space eye of the Andes' refers to the node 'Lumen' "
        "(memory update, delta 1).",
        ("memory.update", "memory.teach"),
        {"node": "Lumen"},
        ("handshake", "facts"),
    ),
    BioStep(
        "teach2",
        "Teach memory again, with a different wording: the query 'Andean sky observatory in "
        "orbit' refers to node 'Lumen' (memory update, delta 1).",
        ("memory.update", "memory.teach"),
        {"node": "Lumen"},
        ("handshake", "facts", "teach1"),
    ),
    BioStep(
        "recall_quorum",
        "Recall from memory 'space eye of the Andes' with quorum 2 and learned_min_status "
        "'confirmed'. Tell me if Lumen is learned and its learned_status.",
        ("memory.recall",),
        {"learned_min_status": "confirmed"},
        ("handshake", "facts", "teach1", "teach2"),
    ),
    BioStep(
        "recall_usage",
        "Recall 'who operates Kestrel' from memory with track_usage true.",
        ("memory.recall",),
        {"track_usage": True},
        ("handshake", "facts"),
    ),
    BioStep(
        "recall_inhib",
        "Recall 'Quito agency' from memory using adaptive inhibition (inhibition 'adaptive').",
        ("memory.recall",),
        {"inhibition": "adaptive"},
        ("handshake", "facts"),
    ),
    BioStep(
        "recall_sdm",
        "Recall 'lunar rover' from memory using mode 'sdm'.",
        ("memory.recall",),
        {"mode": "sdm"},
        ("handshake", "facts"),
    ),
    BioStep(
        "flyvec_train",
        "Train the FlyVec encoder on the memory (memory flyvec_train, 3 epochs).",
        ("memory.flyvec_train",),
        {},
        ("handshake", "facts"),
    ),
    BioStep(
        "recall_flyvec",
        "Recall 'telescope' from memory with encoder 'flyvec'.",
        ("memory.recall",),
        {"encoder": "flyvec"},
        ("handshake", "facts", "flyvec"),
    ),
    BioStep(
        "consolidate",
        "Consolidate memory with decay 0.8, replay true, usage_weighting true and capture true. "
        "Report what it returned.",
        ("memory.consolidate",),
        {"replay": True, "usage_weighting": True, "capture": True},
        ("handshake", "facts", "teach1", "teach2"),
    ),
    BioStep(
        "advise",
        "Ask the tool for advice (meta advise) on whether to use reason.solve for a task with "
        "n_vars 6 and goal 'sat'.",
        ("meta.advise",),
        {"domain": "reason", "action": "solve"},
        ("handshake",),
    ),
    BioStep(
        "feedback",
        "Record feedback (meta feedback): for action 'reason.solve' you used the tool (used true) "
        "on a task with n_vars 6 and goal 'sat', and the outcome was success.",
        ("meta.feedback",),
        {"action": "reason.solve", "used": True, "outcome": "success"},
        ("handshake",),
    ),
    BioStep(
        "advice_stats",
        "Show the learned advice statistics (meta advice_stats) for action 'reason.solve'.",
        ("meta.advice_stats",),
        {},
        ("handshake", "feedback"),
    ),
    BioStep(
        "advice_reset",
        "Reset the learned advice (meta advice_reset) for action 'reason.solve'; I confirm it, "
        "so pass confirm true.",
        ("meta.advice_reset",),
        {"confirm": True},
        ("handshake", "feedback"),
    ),
)


def reps_from_env() -> int:
    raw = os.environ.get(ENV_REPS, "").strip()
    if not raw:
        return DEFAULT_REPS
    try:
        value = int(raw)
    except ValueError:
        value = 0
    if value < 1:
        raise ValueError(f"{ENV_REPS} must be a positive integer, got {raw!r}")
    return value


def task_text(step: BioStep, who: dict[str, Any]) -> str:
    return step.task.format(model=who["model"], family=who["family"], params_b=who["params_b"])


def _seed_calls(prereq: str, who: dict[str, Any]) -> list[dict[str, Any]]:
    """Tool-call kwargs (``domain``, ``action``, ``params``) that establish one prerequisite."""
    if prereq == "handshake":
        return [{"domain": "meta", "action": "handshake", "params": who}]
    if prereq == "facts":
        triples = [{"s": s, "r": r, "o": o} for s, r, o in FACTS]
        return [{"domain": "memory", "action": "ingest", "params": {"triples": triples}}]
    if prereq in ("teach1", "teach2"):
        params = TEACH_1 if prereq == "teach1" else TEACH_2
        return [{"domain": "memory", "action": "update", "params": params}]
    if prereq == "flyvec":
        return [{"domain": "memory", "action": "flyvec_train", "params": {"epochs": 3}}]
    if prereq == "feedback":
        return [{"domain": "meta", "action": "feedback", "params": FEEDBACK_SEED}]
    raise ValueError(f"unknown prerequisite {prereq!r}")


async def seed_step(cs: Any, step: BioStep, who: dict[str, Any]) -> None:
    """Pre-seed one step on its fresh session; a failing prerequisite is a harness bug (raises)."""
    for prereq in step.seed:
        for call in _seed_calls(prereq, who):
            out = json.loads(await cs.call(**call))
            if out.get("status") not in OK_STATUSES:
                raise RuntimeError(f"H-bio seed {prereq} for {step.sid} failed: {out.get('error')}")


_REGISTRY: Registry | None = None


def _registry() -> Registry:
    """Lazily built (module-level cache): the same actions the server coerces params against."""
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = build_registry()
    return _REGISTRY


def _params(args: dict[str, Any]) -> dict[str, Any]:
    """The params the server would act on: JSON/literal strings decoded, a bare list wrapped into
    the action's sole required array param (``server.coerce_params``, not re-implemented here)."""
    coerced, _note, error = coerce_params(
        _registry(), str(args.get("domain")), str(args.get("action")), args.get("params")
    )
    return coerced if error is None and isinstance(coerced, dict) else {}


def _has(params: dict[str, Any], key: str, value: Any) -> bool:
    """The key (or a curated synonym the server renames) is present, with ``value`` unless any."""
    for name, got in params.items():
        if name == key or AUTO_RENAMES.get(str(name).lower()) == key:
            return value is None or got == value
    return False


def matches(args: dict[str, Any], step: BioStep) -> bool:
    """Does this raw tool call hit the step's target with its required params subset?"""
    target = f"{args.get('domain')}.{args.get('action')}"
    if target not in step.targets:
        return False
    params = _params(args)
    return all(_has(params, k, v) for k, v in step.want.items())


def _batch_hits(args: dict[str, Any], step: BioStep) -> bool:
    """A meta.batch counts only if one of its steps IS the target (domain + action equal)."""
    if args.get("domain") != "meta" or args.get("action") != "batch":
        return False
    steps = _params(args).get("steps")
    if not isinstance(steps, list):
        return False
    for inner in steps:
        if not isinstance(inner, dict):
            continue
        if f"{inner.get('domain')}.{inner.get('action')}" in step.targets:
            return True
    return False


def score_step(calls: list[tuple[dict[str, Any], dict[str, Any]]], step: BioStep) -> dict[str, Any]:
    """``calls`` = (raw args, server envelope) of the model's calls for this step, in order."""
    hits = [(a, e) for a, e in calls if matches(a, step)]
    if not hits:
        hits = [(a, e) for a, e in calls if _batch_hits(a, step)]
    ok = [e for _, e in hits if str(e.get("status", "")).upper() in OK_STATUSES]
    return {
        "called": bool(hits),
        "server_ok": bool(ok),
        "passed": bool(hits) and bool(ok),
        "n_calls": len(calls),
    }
