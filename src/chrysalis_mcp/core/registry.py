"""domain/action registry: the single source of truth for dispatch, validation and schema.

Adapted from the AEP ``aep_ctl`` dispatcher (see NOTICE). Adding a domain is a pure
``register`` call; ``schema()`` and ``dispatch()`` read the same table, so they cannot drift.
"""

import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from .advisory import Advisor, enforce_enabled
from .envelope import infer_severity
from .profile import ProfileStore
from .session import SessionCache, resolve_session_id

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


@dataclass(frozen=True)
class ActionSpec:
    domain: str
    action: str
    handler: Handler
    params_schema: dict[str, dict[str, Any]]
    description: str
    example: Params


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

    def register(
        self,
        domain: str,
        action: str,
        handler: Handler,
        *,
        params_schema: dict[str, dict[str, Any]] | None = None,
        description: str = "",
        example: Params | None = None,
    ) -> None:
        if action in self._actions.get(domain, {}):
            raise ValueError(f"already registered: {domain}.{action}")
        spec = ActionSpec(domain, action, handler, params_schema or {}, description, example or {})
        self._actions.setdefault(domain, {})[action] = spec

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
        override = False
        if "override" in params and "override" not in spec.params_schema:
            params = dict(params)  # generic advisory escape hatch, not an action param
            override = params.pop("override") is True
        problem = _check_params(spec, params)
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


def _check_params(spec: ActionSpec, params: Params) -> Payload | None:
    known = spec.params_schema
    missing = [n for n, s in known.items() if s.get("required") and n not in params]
    unknown = [n for n in params if n not in known]
    if not missing and not unknown:
        return None
    return {
        "status": "ERROR",
        "error": f"invalid params for {spec.domain}.{spec.action}",
        "missing": missing,
        "unknown": unknown,
        "valid": sorted(known),
        "next_action": "Fix params per 'valid'/'missing'; see meta.schema.",
    }
