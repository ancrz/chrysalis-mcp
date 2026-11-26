"""domain/action registry: the single source of truth for dispatch, validation and schema.

Adapted from the AEP ``aep_ctl`` dispatcher (see NOTICE). Adding a domain is a pure
``register`` call; ``schema()`` and ``dispatch()`` read the same table, so they cannot drift.
"""

import difflib
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from . import task_features
from .advisory import Advisor, enforce_enabled
from .envelope import infer_severity
from .learning import AdviceStore
from .profile import ProfileStore
from .session import SessionCache, resolve_session_id
from .shape_hints import shape_hint

Params = dict[str, Any]
Payload = dict[str, Any]

ANONYMOUS = "anonymous"
AGENT_ID_RE = re.compile(r"^[A-Za-z0-9_.-]{1,64}\Z")
_BYTES_PER_TOKEN = 4  # == envelope._BYTES_PER_TOKEN (core.envelope does not import registry)


@dataclass(frozen=True)
class Context:
    session_id: str | None
    cache: SessionCache
    registry: Registry
    verbosity: str = "summary"
    agent_id: str = ANONYMOUS


Handler = Callable[[Params, Context], Payload]
Coercer = Callable[[Params], tuple[Params, list[str]]]


@dataclass(frozen=True)
class ActionSpec:
    domain: str
    action: str
    handler: Handler
    params_schema: dict[str, dict[str, Any]]
    description: str
    example: Params
    coerce: Coercer | None = None  # lenient pre-validation rewrite: (params, notes)
    alias_of: str | None = None


@dataclass
class _Counter:
    calls: int = 0
    errors: int = 0
    bytes: int = 0
    est_tokens: int = 0
    total_ms: float = 0.0


@dataclass
class Registry:
    _actions: dict[str, dict[str, ActionSpec]] = field(default_factory=dict)
    _stats: dict[tuple[str, str, str, str], _Counter] = field(default_factory=dict)
    profiles: ProfileStore = field(default_factory=ProfileStore)
    advisor: Advisor = field(default_factory=Advisor.default)
    advice: AdviceStore = field(default_factory=AdviceStore)

    def register(
        self,
        domain: str,
        action: str,
        handler: Handler,
        *,
        params_schema: dict[str, dict[str, Any]] | None = None,
        description: str = "",
        example: Params | None = None,
        coerce: Coercer | None = None,
        alias_of: str | None = None,
    ) -> None:
        if action in self._actions.get(domain, {}):
            raise ValueError(f"already registered: {domain}.{action}")
        spec = ActionSpec(
            domain, action, handler, params_schema or {}, description, example or {}, coerce,
            alias_of,
        )  # fmt: skip
        self._actions.setdefault(domain, {})[action] = spec

    def spec(self, domain: str, action: str) -> ActionSpec | None:
        return self._actions.get(domain, {}).get(action)

    def example_call(self, domain: str, action: str) -> Payload | None:
        """Full call form of an action's example (what a model should send), or ``None``."""
        spec = self.spec(domain, action)
        if spec is None:
            return None
        return {"domain": domain, "action": action, "params": spec.example}

    def sole_required_array(self, domain: str, action: str) -> str | None:
        """Name of the only required array param of an action; ``None`` if zero or several."""
        spec = self.spec(domain, action)
        if spec is None:
            return None
        names = [
            n
            for n, s in spec.params_schema.items()
            if s.get("required") and s.get("type") == "array"
        ]
        return names[0] if len(names) == 1 else None

    def domains(self) -> list[str]:
        return sorted(self._actions)

    def actions(self, domain: str) -> list[str]:
        return sorted(self._actions.get(domain, {}))

    def schema(self) -> Payload:
        return {
            "domains": {
                domain: {
                    action: {
                        "description": spec.description,
                        "params": spec.params_schema,
                        "example": {"domain": domain, "action": action, "params": spec.example},
                    }
                    for action, spec in sorted(actions.items())
                }
                for domain, actions in sorted(self._actions.items())
            }
        }

    def dispatch(self, domain: str, action: str, params: Params | None, ctx: Context) -> Payload:
        if domain not in self._actions:
            return _invalid("domain", domain, self.domains())
        spec = self._actions[domain].get(action)
        if spec is None:
            return _invalid("action", action, self.actions(domain), domain=domain)
        params = params or {}
        params, notes = _apply_renames(spec, params)
        if spec.coerce is not None:
            params, more = spec.coerce(params)
            notes += more
        override = False
        if "override" in params and "override" not in spec.params_schema:
            params = dict(params)  # generic advisory escape hatch, not an action param
            override = params.pop("override") is True
        problem = _check_params(self, spec, params)
        if problem is not None:
            return problem
        advisory = self._advise(spec, params, ctx)
        blocked = advisory is not None and advisory["level"] == "discouraged" and not override
        if blocked and advisory is not None and enforce_enabled():
            return {
                "status": "DISCOURAGED",
                "summary": f"{domain}.{action} discouraged for this model: {advisory['why']}",
                "advisory": advisory,
                "next_action": "Pick another path, or retry with params.override=true.",
            }
        counter = self._counter(ctx.session_id, ctx.agent_id, domain, action)
        counter.calls += 1
        started = time.perf_counter()
        try:
            out = spec.handler(params, ctx)
        except Exception as exc:
            out = {
                "status": "ERROR",
                "error": f"{type(exc).__name__}: {exc}",
                "summary": f"{domain}.{action} raised {type(exc).__name__}",
            }
        counter.total_ms += (time.perf_counter() - started) * 1000
        if advisory is not None and advisory["level"] in ("discouraged", "experimental"):
            out.setdefault("advisory", advisory)  # non-blocking annotation
        if infer_severity(str(out.get("status", "OK"))) == "ERROR":
            counter.errors += 1
        if notes:
            out["coerced"] = [*out.get("coerced", []), *notes]
        return out

    def _advise(self, spec: ActionSpec, params: Params, ctx: Context) -> Payload | None:
        if spec.domain == "meta":
            return None
        profile = self.profiles.resolve(ctx.session_id)
        return self.advisor.lookup(f"{spec.domain}.{spec.action}", profile, params)

    def add_bytes(
        self,
        session_id: str | None,
        domain: str,
        action: str,
        nbytes: int,
        agent_id: str = ANONYMOUS,
    ) -> None:
        if action in self._actions.get(domain, {}):
            counter = self._counter(session_id, agent_id, domain, action)
            counter.bytes += nbytes
            counter.est_tokens += max(1, round(nbytes / _BYTES_PER_TOKEN))

    def stats(
        self,
        session_id: str | None = None,
        *,
        group_by: str = "action",
        agent_id: str | None = None,
    ) -> dict[str, dict[str, float]]:
        """Totals per ``domain.action`` (or per agent); all sessions when ``session_id`` is None.

        Batch inner steps are attributed to the caller's ``agent_id``; ``total_ms`` of a batch
        includes its inner steps (nested time, not exclusive).
        """
        wanted = None if session_id is None else resolve_session_id(session_id)
        totals: dict[str, dict[str, float]] = {}
        for (sid, agent, domain, action), c in sorted(self._stats.items()):
            if (wanted is not None and sid != wanted) or (agent_id and agent != agent_id):
                continue
            key = f"{domain}.{action}" if group_by == "action" else agent
            row = totals.setdefault(key, dict.fromkeys(_ROW_FIELDS, 0))
            row["calls"] += c.calls
            row["errors"] += c.errors
            row["bytes"] += c.bytes
            row["est_tokens"] += c.est_tokens
            row["total_ms"] = round(row["total_ms"] + c.total_ms, 2)
        return totals

    def _counter(self, session_id: str | None, agent_id: str, domain: str, action: str) -> _Counter:
        key = (resolve_session_id(session_id), agent_id, domain, action)
        return self._stats.setdefault(key, _Counter())


_ROW_FIELDS = ("calls", "errors", "bytes", "est_tokens", "total_ms")


def _invalid(kind: str, got: str, valid: list[str], *, domain: str | None = None) -> Payload:
    where = f" in domain '{domain}'" if domain else ""
    return {
        "status": "ERROR",
        "error": f"unknown {kind} '{got}'{where}",
        "valid": valid,
        "next_action": "Retry with one of 'valid'; or call meta.schema for the full contract.",
    }


# Common small-model spellings of a param -> the real name (applied only if the real name exists).
SYNONYMS = {
    "used_tool": "used",
    "tool_used": "used",
    "used_tools": "used",
    "q": "query",
    "text": "query",
    "model_name": "model",
    "relation": "r",
    "predicate": "r",
    "subject": "s",
    "object": "o",
}
# Unambiguous misspellings applied automatically (difflib guesses stay suggestions). A rename is
# applied only if its target is a known param of the action, the target is absent, and the source
# is listed here; free-form names (task_description, ...) are never guessed.
AUTO_RENAMES = {
    "used_tool": "used",
    "tool_used": "used",
    "used_tools": "used",
    "n_variables": "n_vars",
    "num_variables": "n_vars",
    "num_vars": "n_vars",
    "number_of_variables": "n_vars",
    "model_name": "model",
}
_TRIPLE_WORDS = {"subject", "predicate", "relation", "object"}
CLOSE_CUTOFF = 0.6


def _rename_keys(
    params: Params, accepts: Callable[[str], bool], prefix: str = ""
) -> tuple[Params, list[str]]:
    out = dict(params)
    notes: list[str] = []
    for name in list(out):
        target = AUTO_RENAMES.get(name.lower()) if isinstance(name, str) else None
        if target and name != target and target not in out and accepts(target):
            out[target] = out.pop(name)
            notes.append(f"renamed: {prefix}{name}->{prefix}{target}")
    return out, notes


def _apply_renames(spec: ActionSpec, params: Params) -> tuple[Params, list[str]]:
    """Curated, unambiguous param renames (top level, plus the nested ``task`` features)."""
    known = spec.params_schema
    params, notes = _rename_keys(
        params, lambda t: t in known or (t in task_features.FEATURES and "task" in known)
    )
    if "task" in known and isinstance(params.get("task"), dict):
        task, more = _rename_keys(params["task"], lambda t: t in task_features.FEATURES, "task.")
        if more:
            params = {**params, "task": task}
            notes += more
    return params, notes


def _did_you_mean(unknown: list[str], known: dict[str, Any]) -> dict[str, str]:
    out: dict[str, str] = {}
    for name in unknown:
        target = SYNONYMS.get(name.lower())
        if target not in known:
            close = difflib.get_close_matches(name, list(known), n=1, cutoff=CLOSE_CUTOFF)
            target = close[0] if close else None
        if target is not None:
            out[name] = target
    return out


def _other_action(registry: Registry, spec: ActionSpec, params: Params) -> str | None:
    """The one other action of the domain whose schema fits ``params`` (aliases collapse)."""
    if not params:
        return None
    fits: dict[str, ActionSpec] = {}
    for name, cand in registry._actions[spec.domain].items():
        if name == spec.action or cand.alias_of == spec.action or spec.alias_of == name:
            continue
        schema = cand.params_schema
        if all(p in schema for p in params) and all(
            n in params for n, s in schema.items() if s.get("required")
        ):
            fits[name] = cand
    for name in list(fits):
        if fits[name].alias_of in fits:
            del fits[name]
    return next(iter(fits)) if len(fits) == 1 else None


MAX_ECHO_NAME_CHARS = 64


def _short(name: Any) -> str:
    """Bound a caller-supplied key name echoed in an error envelope."""
    text = str(name)
    return text if len(text) <= MAX_ECHO_NAME_CHARS else text[:MAX_ECHO_NAME_CHARS] + "..."


def _check_params(registry: Registry, spec: ActionSpec, params: Params) -> Payload | None:
    known = spec.params_schema
    missing = [n for n, s in known.items() if s.get("required") and n not in params]
    unknown = [n for n in params if n not in known]
    if not missing and not unknown:
        return None
    out: Payload = {
        "status": "ERROR",
        "error": f"invalid params for {spec.domain}.{spec.action}",
        "missing": missing,
        "unknown": [_short(n) for n in unknown],
        "valid": sorted(known),
        "example": registry.example_call(spec.domain, spec.action),
        "next_action": "Fix params per 'valid'/'missing' and copy the shape of 'example'.",
    }
    if unknown and (guess := _did_you_mean(unknown, known)):
        out["did_you_mean"] = {_short(k): v for k, v in guess.items()}
    if _TRIPLE_WORDS & set(unknown) and "triples" in known:
        out["hint"] = "triples items are {s, r, o}: params={'triples': [{'s':..,'r':..,'o':..}]}"
    other = _other_action(registry, spec, params)
    if other is not None:
        out["did_you_mean_action"] = f"{spec.domain}.{other}"
        out["did_you_mean_example"] = registry.example_call(spec.domain, other)
    if "did_you_mean_action" not in out and (
        redirect := shape_hint(spec.domain, spec.action, params)
    ):
        if "hint" in out:
            redirect["hint"] = f"{out['hint']} {redirect['hint']}"
        out.update(redirect)
    return out
