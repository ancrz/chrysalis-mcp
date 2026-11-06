"""Model profile: who is calling, resolved by precedence (forced > declared > measured).

Adapted from the tier-detection idiom of agent-execution-plane ``engine/privilege.py``
(``detect_chmod_tier``: forced env -> measured probes -> tier -> degradation); see NOTICE. Here the
"tier" is a model class (small / medium / large / unknown) plus a thinking flag, and the "probes"
are tiny server-side checked tasks per capability family. MCP ``clientInfo`` is audit-only.
"""

import contextlib
import hashlib
import json
import math
import os
import random
import re
import secrets
from dataclasses import dataclass, field, replace
from typing import Any

from .session import resolve_session_id

ENV_PROFILE = "CHRYSALIS_MODEL_PROFILE"
CLASSES = ("small", "medium", "large", "unknown")
SMALL_MAX_B = 8.0
MEDIUM_MAX_B = 30.0
MAX_TEXT = 128
MAX_ANSWER_ITEMS = 16
MAX_STORED_SESSIONS = 256
_TRUE = frozenset({"thinking", "think", "1", "true", "on"})
_FALSE = frozenset({"nothinking", "nothink", "0", "false", "off"})


_SIZE_MOE = re.compile(r"^(\d+(?:\.\d+)?)[/a](\d+(?:\.\d+)?)$")  # compact MoE: 26/4 or 26a4


def effective_params_b(params_b: float | None, active_params_b: float | None) -> float | None:
    """Size used to derive the class: sqrt(total * active) for a MoE, else the total.

    The geometric mean is a community rule of thumb (unverified against a primary source);
    it is only the prior, calibration still overrides it.
    """
    if params_b is not None and active_params_b is not None and 0 < active_params_b < params_b:
        return math.sqrt(params_b * active_params_b)
    return params_b


def classify(params_b: float | None) -> str:
    if params_b is None or params_b <= 0:
        return "unknown"
    if params_b < SMALL_MAX_B:
        return "small"
    return "medium" if params_b <= MEDIUM_MAX_B else "large"


@dataclass
class Profile:
    source: str = "unknown"  # forced | declared | measured | unknown
    model: str | None = None
    family: str | None = None
    params_b: float | None = None
    active_params_b: float | None = None  # MoE only; None for dense models or an ignored value
    context: int | None = None
    thinking: bool | None = None
    scores: dict[str, float] = field(default_factory=dict)
    client: str | None = None  # clientInfo, audit only
    note: str | None = None
    warning: str | None = None  # e.g. an ignored active_params_b

    @property
    def effective_params_b(self) -> float | None:
        return effective_params_b(self.params_b, self.active_params_b)

    @property
    def model_class(self) -> str:
        return classify(self.effective_params_b)

    def view(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "source": self.source,
            "class": self.model_class,
            "thinking": self.thinking,
            "scores": dict(self.scores),
        }
        for key in (
            "model",
            "family",
            "params_b",
            "active_params_b",
            "context",
            "client",
            "note",
            "warning",
        ):
            if getattr(self, key) is not None:
                out[key] = getattr(self, key)
        if self.params_b is not None:
            out["effective_params_b"] = round(self.effective_params_b or 0.0, 2)
        return out


def parse_forced(raw: str) -> Profile:
    """``family:size_b[:thinking]`` or a JSON object; raises ValueError on anything else."""
    raw = raw.strip()
    if raw.startswith("{"):
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError("profile JSON must be an object")
        return from_fields(data, source="forced")
    parts = raw.split(":")
    if not 2 <= len(parts) <= 3:
        raise ValueError("expected family:size_b[:thinking]")
    fields: dict[str, Any] = {"family": parts[0], "params_b": parts[1]}
    moe = _SIZE_MOE.match(parts[1].strip())
    if moe:
        fields.update(params_b=moe.group(1), active_params_b=moe.group(2))
    if len(parts) == 3:
        low = parts[2].strip().lower()
        if low not in _TRUE | _FALSE:
            raise ValueError(f"unknown thinking flag '{parts[2]}'")
        fields["thinking"] = low in _TRUE
    return from_fields(fields, source="forced")


def _text(value: Any, name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip() or len(value) > MAX_TEXT:
        raise ValueError(f"{name} must be a non-empty string up to {MAX_TEXT} chars")
    return value.strip()


def _positive(value: Any, name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float | str):
        raise ValueError(f"{name} must be a positive number")
    try:
        num = float(value)
    except ValueError:
        raise ValueError(f"{name} must be a positive number") from None
    if not 0 < num < float("inf"):
        raise ValueError(f"{name} must be a positive number")
    return num


def _active(value: Any, params_b: float | None) -> tuple[float | None, str | None]:
    """Valid active size or (None, warning); an unusable active size never fails the profile."""
    if value is None:
        return None, None
    try:
        active = _positive(value, "active_params_b")
    except ValueError:
        return None, "active_params_b ignored: must be a positive number"
    if params_b is None or active is None or active > params_b:
        return None, "active_params_b ignored: needs params_b and active_params_b <= params_b"
    return active, None


def from_fields(data: dict[str, Any], *, source: str) -> Profile:
    thinking = data.get("thinking")
    if thinking is not None and not isinstance(thinking, bool):
        raise ValueError("thinking must be a boolean")
    context = _positive(data.get("context"), "context")
    params_b = _positive(data.get("params_b"), "params_b")
    active, warning = _active(data.get("active_params_b"), params_b)
    return Profile(
        source=source,
        model=_text(data.get("model"), "model"),
        family=_text(data.get("family"), "family"),
        params_b=params_b,
        active_params_b=active,
        warning=warning,
        context=None if context is None else int(context),
        thinking=thinking,
        client=_text(data.get("client"), "client"),
    )


# --- calibration probes ------------------------------------------------------------------------


_SINGLE_EQ = re.compile(r"(?<![=!<>])=(?!=)")
_PROBE_KEY = re.compile(r"[a-z]+-\d+")


def normalize(answer: Any) -> str | None:
    """Canonical text of an answer (scalar or list of scalars); None when malformed.

    A lone ``=`` is read as ``==`` (the DSL spells equality ``==``; the probes test translating
    the statement, not remembering that spelling).
    """
    if isinstance(answer, str):
        with contextlib.suppress(ValueError):
            answer = json.loads(answer)
    items = answer if isinstance(answer, list) else [answer]
    if not items or len(items) > MAX_ANSWER_ITEMS:
        return None
    parts: list[str] = []
    for item in items:
        if isinstance(item, bool) or not isinstance(item, str | int | float) or item == "":
            return None
        parts.append(_SINGLE_EQ.sub("==", re.sub(r"\s+", "", str(item)).lower()))
    return "|".join(sorted(parts))


def _digest(norm: str) -> str:
    return hashlib.sha256(norm.encode()).hexdigest()


@dataclass(frozen=True)
class ProbeSet:
    """Probes issued to one session. ``expected`` maps probe id -> (family, sha256 of the answer).

    The plain answers are never kept: only their digests, and only server-side.
    """

    public: tuple[dict[str, str], ...]
    expected: dict[str, tuple[str, str]]


_PLACES = ("Lima", "Oslo", "Quito", "Cairo", "Hanoi", "Dakar", "Kyiv", "Sofia", "Accra", "Bern")
_NAMES = ("Ana", "Bruno", "Carla", "Dario", "Elena", "Felix", "Greta", "Hugo")

_DSL_HINT = "Operators: + - * == != < <= > >= and or not; equality is written ==."
_Built = tuple[str, str, str, tuple[str, ...]]  # (id, family, prompt, plain answer items)


def _build(nonce: str) -> list[_Built]:
    """Probe instances derived from ``nonce``; every answer needs a transformation of the prompt."""
    rng = random.Random(nonce)
    a = rng.randint(2, 40)
    b = a + rng.randint(3, 60)
    total = rng.randint(10, 90)
    var = rng.choice("xyzwuv")
    verb, goal = rng.choice((("minimize", "min"), ("maximize", "max")))
    ids = [f"{rng.choice('nkmqr')}{rng.randint(10, 99)}" for _ in range(3)]
    tenths = rng.sample(range(1, 10), 3)
    best = max(range(3), key=lambda i: tenths[i])
    hits = ",".join(f'{{"id":"{i}","score":0.{t}}}' for i, t in zip(ids, tenths, strict=True))
    person, city = rng.choice(_NAMES), rng.choice(_PLACES)
    return [
        (
            "reason-1",
            "reason",
            f"Constraint DSL: give the `assert` list for 'x is an integer and x plus {a} equals "
            f"{b}' (vars: x int). {_DSL_HINT} Answer: JSON list of strings.",
            (f"x+{a}=={b}",),
        ),
        (
            "reason-2",
            "reason",
            f"Constraint DSL: give the `assert` list for 'x and y integers, x greater than y, "
            f"their sum is {total}'. {_DSL_HINT} Answer: JSON list of strings.",
            ("x>y", f"x+y=={total}"),
        ),
        (
            "reason-3",
            "reason",
            f"Constraint DSL: give the `goal` string to {verb} variable {var} "
            "(goal is 'sat', 'min:<var>' or 'max:<var>').",
            (f"{goal}:{var}",),
        ),
        (
            "memory-1",
            "memory",
            f"Given recall hits [{hits}], answer '<id>:<n>' for the best-scored hit, where n is "
            "its score times 10.",
            (f"{ids[best]}:{tenths[best]}",),
        ),
        (
            "memory-2",
            "memory",
            f'Given the recall result {{"hits":[{{"id":"a","text":"{person} lives in {city}"}}]}}'
            ", answer with the city name spelled backwards.",
            (city[::-1],),
        ),
    ]


def issue_probes(nonce: str | None = None) -> ProbeSet:
    """Fresh probes for one session; a random server-side nonce unless given (tests)."""
    built = _build(nonce or secrets.token_hex(8))
    return ProbeSet(
        public=tuple({"id": pid, "family": fam, "prompt": prompt} for pid, fam, prompt, _ in built),
        expected={pid: (fam, _digest(normalize(list(ans)) or "")) for pid, fam, _, ans in built},
    )


def _canonical_keys(answers: dict[str, Any], expected: dict[str, Any]) -> dict[str, Any]:
    """Map keys such as ``reason-1 (reason)`` or ``Reason-1:`` to the probe id they contain."""
    out: dict[str, Any] = {}
    for key, value in answers.items():
        found = _PROBE_KEY.search(str(key).lower())
        pid = found.group(0) if found and found.group(0) in expected else key
        out.setdefault(pid, value)  # first spelling wins; never silently overwrite
    return out


def score(answers: dict[str, Any], issued: ProbeSet) -> tuple[dict[str, float], list[str]]:
    """Per-family score (0..1) for families that received at least one answer.

    Missing or malformed answers count as wrong; unknown probe ids are reported and ignored.
    """
    expected = issued.expected
    answers = _canonical_keys(answers, expected)
    unknown = sorted(k for k in answers if k not in expected)
    families = sorted({expected[k][0] for k in answers if k in expected})
    scores: dict[str, float] = {}
    for family in families:
        ids = [pid for pid, (fam, _) in expected.items() if fam == family]
        right = 0
        for pid in ids:
            norm = normalize(answers[pid]) if pid in answers else None
            right += norm is not None and _digest(norm) == expected[pid][1]
        scores[family] = round(right / len(ids), 4)
    return scores, unknown


# --- store -------------------------------------------------------------------------------------


@dataclass
class ProfileStore:
    """Per-session declared profile and measured scores (process lifetime, bounded)."""

    declared: dict[str, Profile] = field(default_factory=dict)
    measured: dict[str, dict[str, float]] = field(default_factory=dict)
    by_model: dict[str, dict[str, float]] = field(default_factory=dict)
    issued: dict[str, ProbeSet] = field(default_factory=dict)

    @staticmethod
    def _put(table: dict[str, Any], key: str, value: Any) -> None:
        table.pop(key, None)
        while len(table) >= MAX_STORED_SESSIONS:
            del table[next(iter(table))]
        table[key] = value

    @staticmethod
    def _touch(table: dict[str, Any], key: str) -> Any:
        """Read with LRU refresh: a session in active use is the last to be evicted."""
        if key not in table:
            return None
        value = table.pop(key)
        table[key] = value
        return value

    def issue(self, session_id: str | None, *, fresh: bool = False) -> ProbeSet:
        """Probes of the session; idempotent (re-asking must not invalidate the probes the model
        already read), unless ``fresh`` asks for a new nonce."""
        sid = resolve_session_id(session_id)
        existing = None if fresh else self._touch(self.issued, sid)
        if existing is not None:
            return existing
        probes = issue_probes()
        self._put(self.issued, sid, probes)
        return probes

    def issued_for(self, session_id: str | None) -> ProbeSet | None:
        return self._touch(self.issued, resolve_session_id(session_id))

    def consume(self, session_id: str | None) -> None:
        """A scored probe set is spent: answers cannot be resubmitted to improve the score."""
        self.issued.pop(resolve_session_id(session_id), None)

    def declare(self, session_id: str | None, profile: Profile) -> None:
        self._put(self.declared, resolve_session_id(session_id), profile)

    def record_scores(self, session_id: str | None, scores: dict[str, float]) -> Profile:
        sid = resolve_session_id(session_id)
        merged = {**(self._touch(self.measured, sid) or {}), **scores}
        self._put(self.measured, sid, merged)
        declared = self._touch(self.declared, sid)
        model = declared.model if declared is not None else None
        if model:
            self._put(self.by_model, model, merged)
        return self.resolve(session_id)

    def resolve(self, session_id: str | None) -> Profile:
        sid = resolve_session_id(session_id)
        profile = Profile()
        forced = os.environ.get(ENV_PROFILE)
        if forced:
            try:
                profile = parse_forced(forced)
            except ValueError as exc:  # json.JSONDecodeError is a ValueError
                profile.note = f"{ENV_PROFILE} ignored: {exc}"
        declared = self._touch(self.declared, sid)
        if profile.source == "unknown" and declared is not None:
            profile = replace(declared, scores={}, note=profile.note)
        scores = self._touch(self.measured, sid) or self._touch(self.by_model, profile.model or "")
        scores = scores or {}
        if scores:
            profile.scores = dict(scores)
            if profile.source == "unknown":
                profile.source = "measured"
        return profile
