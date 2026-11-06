"""Transversal ``meta`` domain: schema, discover, stats, batch."""

from typing import Any

from ..core import task_features, version
from ..core.advisory import render_prompt_hint
from ..core.batch import MAX_STEPS, RETURN_MODES, run_batch
from ..core.envelope import VERBOSITIES
from ..core.profile import from_fields, score
from ..core.registry import AGENT_ID_RE, Context, Params, Payload, Registry

# Terse views cap lists at 5 entries: keep the everyday meta actions first and let the one-off
# calibrate (named by handshake's next_action) fall off the end.
_META_TERSE_ORDER = ("batch", "discover", "schema", "stats", "handshake", "advise", "calibrate")


def _terse_actions(domain: str, actions: Any) -> list[str]:
    names = sorted(actions)
    if domain != "meta":
        return names
    rank = {n: i for i, n in enumerate(_META_TERSE_ORDER)}
    return sorted(names, key=lambda n: (rank.get(n, len(rank)), n))


def _load_error(ctx: Context) -> dict[str, Any]:
    error = ctx.registry.advisor.load_error
    return {"load_error": error} if error else {}


def _annotate(domains: dict[str, Any], ctx: Context) -> tuple[dict[str, Any], dict[str, str]]:
    """Add ``advisory`` to every non-meta action for the session's profile.

    Only non-neutral advisories are attached; an absent ``advisory`` means neutral. Returns the
    profile view and the non-neutral levels.
    """
    profile = ctx.registry.profiles.resolve(ctx.session_id)
    advisor = ctx.registry.advisor
    non_neutral: dict[str, str] = {}
    for domain, actions in domains.items():
        if domain == "meta":
            continue
        for action, spec in actions.items():
            name = f"{domain}.{action}"
            advisory = advisor.lookup(name, profile)
            if advisory["level"] != "neutral":  # neutral is the default: omitted to save tokens
                spec["advisory"] = {"level": advisory["level"], "why": advisory["why"]}
                non_neutral[name] = advisory["level"]
            variants = advisor.variants(name, profile)
            if variants:
                spec["advisory_variants"] = variants
    return {**profile.view(), **_load_error(ctx)}, non_neutral


def _rules(ctx: Context) -> tuple[dict[str, str], dict[str, str]]:
    """Conditional advisory rules and their plain-language hints for the session's profile."""
    advisor, profile = ctx.registry.advisor, ctx.registry.profiles.resolve(ctx.session_id)
    return advisor.rules(profile), advisor.hints(profile)


def _rule_fields(rules: dict[str, str], hints: dict[str, str]) -> dict[str, Any]:
    return {"rules": rules, **({"hints": hints} if hints else {})} if rules else {}


def _prompt_hint_field(hints: dict[str, str]) -> dict[str, Any]:
    text = render_prompt_hint(hints)
    return {"prompt_hint": text} if text else {}


def _with_rules(
    out: dict[str, Any], rules: dict[str, str], hints: dict[str, str]
) -> dict[str, Any]:
    return {**out, **_rule_fields(rules, hints)}


BRIEF_DESC_CAP = 90


def _brief_domains(domains: dict[str, Any]) -> dict[str, Any]:
    """Compact brief view: first-sentence description, param names ('*' = required), example
    params, advisory level only (``why``/``source`` stay at summary/detail)."""
    out: dict[str, Any] = {}
    for domain, actions in domains.items():
        out[domain] = {}
        for action, spec in actions.items():
            desc = spec["description"].split("\n")[0].split(". ")[0]
            view: dict[str, Any] = {
                "description": desc[:BRIEF_DESC_CAP],
                "params": [
                    k + ("*" if v.get("required") else "") for k, v in spec["params"].items()
                ],
                "example": spec["example"].get("params", {}),
            }
            if "advisory" in spec:
                view["advisory"] = spec["advisory"]["level"]
            out[domain][action] = view
    return out


def _schema(params: Params, ctx: Context) -> Payload:
    schema = ctx.registry.schema()
    _, non_neutral = _annotate(schema["domains"], ctx)  # profile itself: discover/handshake
    rules, hints = _rules(ctx)
    return {
        "status": "OK",
        "summary": f"{len(schema['domains'])} domains registered",
        **schema,
        **_load_error(ctx),
        "verbosity": list(VERBOSITIES),
        **_rule_fields(rules, hints),
        "brief": _with_rules(
            {
                "domains": _brief_domains(schema["domains"]),
                "verbosity": list(VERBOSITIES),
                **_load_error(ctx),
            },
            rules,
            hints,
        ),
        "terse": _with_rules(
            {
                "domains": {d: _terse_actions(d, a) for d, a in schema["domains"].items()},
                "advisories": non_neutral,
                **_load_error(ctx),
            },
            rules,
            hints,
        ),
    }


def _advisory_views(domains: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """``domain.action -> advisory`` for the annotated actions (neutral ones are absent)."""
    return {
        f"{d}.{a}": spec["advisory"]
        for d, actions in domains.items()
        for a, spec in actions.items()
        if "advisory" in spec
    }


def _handshake(params: Params, ctx: Context) -> Payload:
    if not any(params.get(k) is not None for k in ("model", "family", "params_b")):
        return {"status": "ERROR", "error": "give at least one of model, family, params_b"}
    try:
        profile = from_fields(params, source="declared")
    except ValueError as exc:
        return {"status": "ERROR", "error": str(exc)}
    ctx.registry.profiles.declare(ctx.session_id, profile)
    view = ctx.registry.profiles.resolve(ctx.session_id).view()
    domains = {
        d: {a: {} for a in actions}
        for d, actions in ctx.registry.schema()["domains"].items()
        if d != "meta"
    }
    _, non_neutral = _annotate(domains, ctx)
    advisories = {
        name: {k: v for k, v in annotated.items() if k in ("level", "why")}
        for name, annotated in _advisory_views(domains).items()
    }
    rules, hints = _rules(ctx)
    return {
        "status": "OK",
        "summary": f"profile {view['source']}: class={view['class']}",
        "profile": view,
        "advisories": advisories,
        **_rule_fields(rules, hints),
        **_prompt_hint_field(hints),
        # prompt_hint stays out of brief/terse: those views clip strings (256/64 chars) and a
        # clipped hint cannot be pasted into a system prompt; they carry 'hints' instead.
        "brief": _with_rules({"profile": view, "advisories": non_neutral}, rules, hints),
        "terse": _with_rules({"class": view["class"], "advisories": non_neutral}, rules, hints),
        "next_action": "Optionally run meta.calibrate to self-assess competence per capability.",
    }


def _calibrate(params: Params, ctx: Context) -> Payload:
    answers = params.get("answers")
    if answers is None:
        probes = ctx.registry.profiles.issue(
            ctx.session_id, fresh=params.get("reissue") is True
        ).public
        return {
            "status": "OK",
            "summary": f"{len(probes)} probes (session-specific)",
            "probes": list(probes),
            "next_action": "Answer by probe id: meta.calibrate(answers={'<id>': <answer>, ...}).",
        }
    if not isinstance(answers, dict) or not answers:
        return {"status": "ERROR", "error": "answers must be a non-empty object {probe_id: answer}"}
    issued = ctx.registry.profiles.issued_for(ctx.session_id)
    if issued is None:
        return {
            "status": "ERROR",
            "error": "no open probes for this session (none issued, or already scored): "
            "call without answers (reissue=true for a fresh set)",
        }
    scores, unknown = score(answers, issued)
    if not scores:
        valid = sorted(issued.expected)
        return {
            "status": "ERROR",
            "error": "no known probe ids; use these keys: " + ", ".join(valid),
            "ignored": unknown,
            "valid": valid,
        }
    profile = ctx.registry.profiles.record_scores(ctx.session_id, scores).view()
    ctx.registry.profiles.consume(ctx.session_id)  # one scoring per issued set
    out: Payload = {
        "status": "OK",
        "summary": "scores " + ", ".join(f"{f}={v}" for f, v in scores.items()),
        "scores": scores,
        "profile": profile,
        **_prompt_hint_field(_rules(ctx)[1]),
    }
    if unknown:
        out["ignored"] = unknown
    return out


def _advise(params: Params, ctx: Context) -> Payload:
    domain, action = params.get("domain"), params.get("action")
    if not isinstance(domain, str) or not isinstance(action, str):
        return {"status": "ERROR", "error": "domain and action are required strings"}
    name = f"{domain}.{action}"
    known = {f"{d}.{a}" for d, acts in ctx.registry.schema()["domains"].items() for a in acts}
    if name not in known or domain == "meta":
        return {"status": "ERROR", "error": f"unknown or non-advisable action '{name}'"}
    task = params.get("task")
    if task is not None and not isinstance(task, dict):
        return {"status": "ERROR", "error": "task must be an object of task features"}
    unknown = sorted(set(task or {}) - set(task_features.FEATURES))
    if unknown:
        return {
            "status": "ERROR",
            "error": f"unknown task features {unknown}",
            "valid": list(task_features.FEATURES),
        }
    profile = ctx.registry.profiles.resolve(ctx.session_id)
    advisory = ctx.registry.advisor.lookup(name, profile, None, task or {})
    return {
        "status": "OK",
        "summary": f"{name}: {advisory['level']}",
        "action": name,
        "advisory": {
            "level": advisory["level"],
            "why": advisory["why"],
            **({"hint": advisory["hint"]} if advisory.get("hint") else {}),
        },
        "source": advisory["source"],
        "terse": {"level": advisory["level"]},
    }


def _discover(params: Params, ctx: Context) -> Payload:
    schema = ctx.registry.schema()["domains"]
    wanted = params.get("domain")
    if wanted is not None and wanted not in schema:
        return {"status": "ERROR", "error": f"unknown domain '{wanted}'", "valid": sorted(schema)}
    domains = {
        d: {
            a: {"description": spec["description"].split("\n")[0], "example": spec["example"]}
            for a, spec in actions.items()
        }
        for d, actions in schema.items()
        if wanted in (None, d)
    }
    profile, non_neutral = _annotate(domains, ctx)
    stamp = version.stamp({"domains": schema})
    stale = stamp["stale"]
    rules, hints = _rules(ctx)
    payload: Payload = {
        "status": "DEGRADED" if stale else "OK",
        "summary": f"{len(domains)} domain(s)" + (", STALE process" if stale else ""),
        "domains": domains,
        "profile": profile,
        **_rule_fields(rules, hints),
        **stamp,
        "terse": {
            "domains": {d: _terse_actions(d, a) for d, a in domains.items()},
            "advisories": non_neutral,
            **_rule_fields(rules, hints),
            **_load_error(ctx),
            "loaded_commit": stamp["loaded_commit"],
            "stale": stale,
            "schema_hash": stamp["schema_hash"],
        },
    }
    if stale:
        payload["next_action"] = f"restart chrysalis-mcp (pid {stamp['pid']}): code changed on disk"
    return payload


def _stats(params: Params, ctx: Context) -> Payload:
    scope = params.get("scope", "session")
    if scope not in ("session", "all"):
        return {"status": "ERROR", "error": "scope must be 'session' or 'all'"}
    group_by = params.get("group_by", "action")
    if group_by not in ("action", "agent"):
        return {"status": "ERROR", "error": "group_by must be 'action' or 'agent'"}
    agent_id = params.get("agent_id")
    if agent_id is not None and not (isinstance(agent_id, str) and AGENT_ID_RE.match(agent_id)):
        return {"status": "ERROR", "error": f"agent_id must match {AGENT_ID_RE.pattern}"}
    rows = ctx.registry.stats(
        ctx.session_id if scope == "session" else None, group_by=group_by, agent_id=agent_id
    )
    key = "actions" if group_by == "action" else "agents"
    return {
        "status": "OK",
        "summary": f"{len(rows)} {group_by}(s) with traffic",
        key: rows,
        "terse": {key: {k: v["calls"] for k, v in rows.items()}},
    }


def _batch(params: Params, ctx: Context) -> Payload:
    return run_batch(
        params["steps"],
        ctx.registry,
        ctx,
        return_only=params.get("return_only", "all"),
        select_fields=params.get("select_fields"),
        continue_on_error=bool(params.get("continue_on_error", False)),
    )


def register(registry: Registry) -> None:
    registry.register(
        "meta",
        "schema",
        _schema,
        description="Full contract: every domain, action, params and a canonical example.",
        example={},
    )
    registry.register(
        "meta",
        "discover",
        _discover,
        params_schema={"domain": {"type": "string", "required": False}},
        description="Compact orientation: domains, actions and one example each.",
        example={"domain": "meta"},
    )
    registry.register(
        "meta",
        "handshake",
        _handshake,
        params_schema={
            "model": {"type": "string", "required": False},
            "family": {"type": "string", "required": False},
            "params_b": {"type": "number", "required": False},
            "active_params_b": {"type": "number", "required": False},
            "context": {"type": "integer", "required": False},
            "thinking": {"type": "boolean", "required": False},
            "client": {"type": "string", "required": False},
        },
        description=(
            "Declare the calling model for this session_id (class derives from params_b: "
            "small <8B, medium 8-30B, large >30B; MoE: active_params_b "
            "-> sqrt(total*active)). 'client' (clientInfo) is audit-only. "
            "CHRYSALIS_MODEL_PROFILE overrides any declaration. The response carries "
            "'prompt_hint': plain-language tool advice for THIS session's profile, to paste "
            "into the system prompt."
        ),
        example={"model": "gemma-4-e4b", "family": "gemma", "params_b": 4, "thinking": False},
    )
    registry.register(
        "meta",
        "calibrate",
        _calibrate,
        params_schema={
            "answers": {"type": "object", "required": False},
            "reissue": {"type": "boolean", "required": False},
        },
        description=(
            "Without answers: issues session-specific probes (values differ per session; asking "
            "again returns the same probes unless reissue=true). With "
            "answers {probe_id: answer}: scores competence 0..1 per family; >=0.8 recommends and "
            "<=0.4 discourages the family's actions, overriding class defaults. Calibration is "
            "self-assessment and advisory-only."
        ),
        example={"answers": {"<probe_id>": "<answer>"}},
    )
    registry.register(
        "meta",
        "advise",
        _advise,
        params_schema={
            "domain": {"type": "string", "required": True},
            "action": {"type": "string", "required": True},
            "task": {"type": "object", "required": False},
        },
        description=(
            "Advisory level of domain.action for THIS task and the session's model: task is an "
            "object of features (n_vars, n_atoms, goal sat|opt, facts, ctx_tokens). Cheap; "
            "see 'rules' in handshake/discover for the conditions."
        ),
        example={"domain": "reason", "action": "solve", "task": {"n_atoms": 9, "goal": "sat"}},
    )
    registry.register(
        "meta",
        "stats",
        _stats,
        params_schema={
            "scope": {"type": "string", "required": False, "enum": ["session", "all"]},
            "group_by": {"type": "string", "required": False, "enum": ["action", "agent"]},
            "agent_id": {"type": "string", "required": False},
        },
        description=(
            "Calls, errors, bytes, est_tokens and total_ms since process start, grouped by "
            "domain.action or by agent_id; optional agent_id filter."
        ),
        example={"scope": "session"},
    )
    steps_example: list[dict[str, Any]] = [
        {"id": "a", "domain": "meta", "action": "discover", "store_as": "d"},
        {"id": "b", "domain": "meta", "action": "stats", "params": {"scope": "all"}},
    ]
    registry.register(
        "meta",
        "batch",
        _batch,
        params_schema={
            "steps": {"type": "array", "required": True, "max_items": MAX_STEPS},
            "return_only": {"type": "string", "required": False, "enum": list(RETURN_MODES)},
            "select_fields": {"type": "array", "required": False},
            "continue_on_error": {"type": "boolean", "required": False},
        },
        description=(
            "Chain steps server-side. Refs: '$cache.<key>.<field>' (key = step id or store_as). "
            "Stops on first error unless continue_on_error. Nested meta.batch is rejected."
        ),
        example={"steps": steps_example, "return_only": "last", "select_fields": ["status"]},
    )
