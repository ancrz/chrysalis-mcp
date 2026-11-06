"""Evidence-based capability advisories per model profile.

Same degradation-matrix idea as AEP ``engine/privilege.py`` (capability -> level by tier), but the
matrix is data (``data/evidence.json``: every entry cites a measured result) and measured
calibration scores override the class defaults. Advisories never block unless
``CHRYSALIS_ENFORCE_ADVISORY=1``; see NOTICE.
"""

import json
import os
from dataclasses import dataclass
from importlib import resources
from typing import Any

from . import task_features
from .profile import CLASSES, Profile

ENV_ENFORCE = "CHRYSALIS_ENFORCE_ADVISORY"
LEVELS = ("recommended", "neutral", "discouraged", "experimental")
RECOMMEND_AT = 0.8
DISCOURAGE_AT = 0.4
_REQUIRED = ("action", "model_class", "level", "metric", "value", "source", "date")
STATUSES = ("measured", "pending")  # pending: a hypothesis awaiting its run; never advises
_BOUNDS = ("min", "max")
_SEVERITY = {"neutral": 0, "recommended": 1, "experimental": 2, "discouraged": 3}
NEUTRAL: dict[str, Any] = {"level": "neutral", "why": "no evidence", "source": None}


def validate_entry(entry: Any) -> list[str]:
    """Problems with one evidence entry (empty list when valid)."""
    if not isinstance(entry, dict):
        return ["entry must be an object"]
    problems = [f"missing '{k}'" for k in _REQUIRED if k not in entry]
    if entry.get("level") not in LEVELS:
        problems.append(f"level must be one of {LEVELS}")
    if entry.get("model_class") not in (*CLASSES, "any"):
        problems.append("model_class must be a class or 'any'")
    if "thinking" in entry and not isinstance(entry["thinking"], bool):
        problems.append("thinking must be a boolean")
    if "when" in entry and not isinstance(entry["when"], dict):
        problems.append("when must be an object")
    if "task" in entry:
        problems += _task_problems(entry["task"])
    if entry.get("status", "measured") not in STATUSES:
        problems.append(f"status must be one of {STATUSES}")
    action = entry.get("action")
    if not isinstance(action, str) or action.count(".") != 1:
        problems.append("action must be 'domain.action'")
    return problems


def _task_problems(task: Any) -> list[str]:
    if not isinstance(task, dict) or not task:
        return ["task must be a non-empty object"]
    problems: list[str] = []
    for feature, cond in task.items():
        if feature not in task_features.FEATURES:
            problems.append(f"unknown task feature '{feature}'")
        elif isinstance(cond, dict):
            if not cond or set(cond) - set(_BOUNDS):
                problems.append(f"task.{feature}: bounds are {_BOUNDS}")
            elif not all(
                isinstance(v, int | float) and not isinstance(v, bool) for v in cond.values()
            ):
                problems.append(f"task.{feature}: bounds must be numbers")
        elif isinstance(cond, list):
            if not cond:
                problems.append(f"task.{feature}: empty list")
        elif not isinstance(cond, str | int | float | bool):
            problems.append(f"task.{feature}: unsupported condition")
    return problems


def _holds(cond: Any, value: Any) -> bool:
    if isinstance(cond, dict):
        if isinstance(value, bool) or not isinstance(value, int | float):
            return False
        return value >= cond.get("min", float("-inf")) and value <= cond.get("max", float("inf"))
    if isinstance(cond, list):
        return value in cond
    return value == cond and isinstance(value, bool) == isinstance(cond, bool)


def _cond_text(feature: str, cond: Any) -> str:
    if isinstance(cond, dict):
        parts = []
        if "min" in cond:
            parts.append(f"{feature}>={cond['min']:g}")
        if "max" in cond:
            parts.append(f"{feature}<={cond['max']:g}")
        return "&".join(parts)
    value = "|".join(map(str, cond)) if isinstance(cond, list) else str(cond).lower()
    return f"{feature}={value}"


def _task_text(task: dict[str, Any]) -> str:
    return "&".join(_cond_text(k, c) for k, c in task.items())


_TAG_WORDS = {"rec": "worth using when", "dis": "not worth using, do it yourself, when"}


def hint_for_task(task: dict[str, Any]) -> str | None:
    """Conditions of one task entry in plain words joined by ``and``; ``None`` if unphrasable."""
    parts = [t for k, c in task.items() if (t := task_features.hint_text(k, c))]
    return " and ".join(parts) if parts and len(parts) == len(task) else None


def hint_for_tags(tags: dict[str, list[dict[str, Any]]]) -> str:
    """``tag -> [task]`` as words: ``or`` between entries; unphrasable entries are skipped."""
    clauses = []
    for tag, tasks in sorted(tags.items()):
        phrases = [h for t in tasks if (h := hint_for_task(t))]
        if phrases:
            clauses.append(f"{_TAG_WORDS[tag]} {' or '.join(phrases)}")
    return "; ".join(clauses)


def render_advisory_line(rules: dict[str, str], hints: dict[str, str] | None = None) -> str:
    """`` Advisory <actions>: <rule>. In words: <hint>.`` per rule (single renderer for every
    surface that shows the rules in prose: the MCP tool description and the validation bridge)."""
    hints = hints or {}
    return "".join(
        f" Advisory {actions}: {rule}."
        + (f" In words: {hints[actions]}." if actions in hints else "")
        for actions, rule in rules.items()
    )


def render_prompt_hint(hints: dict[str, str]) -> str:
    """Plain-language tool advice for a SYSTEM PROMPT (``""`` when there are no hints).

    Single renderer for the two system-prompt channels: the server ``instructions`` (startup
    profile) and ``meta.handshake``'s ``prompt_hint`` (session profile). Evidence: a model that
    ignores the same rule inside a tool description follows it from the system prompt
    (measured in the T18 held-out runs).
    """
    if not hints:
        return ""
    return "Tool advice: " + " ".join(f"For {actions}: {hint}." for actions, hint in hints.items())


def _read_entries() -> list[Any]:
    raw = resources.files("chrysalis_mcp").joinpath("data/evidence.json").read_text("utf-8")
    entries = json.loads(raw)["entries"]
    if not isinstance(entries, list):
        raise ValueError("'entries' must be a list")
    return entries


def load_evidence() -> list[dict[str, Any]]:
    """Strict load: raises on the first problem (used by the evidence-file schema test)."""
    entries = _read_entries()
    bad = {i: p for i, e in enumerate(entries) if (p := validate_entry(e))}
    if bad:
        raise ValueError(f"invalid evidence entries: {bad}")
    return entries


def load_evidence_soft() -> tuple[list[dict[str, Any]], str | None]:
    """Fail-soft load: valid entries are kept, anything else becomes a ``load_error`` message."""
    try:
        entries = _read_entries()
    except (ValueError, OSError, KeyError, TypeError) as exc:  # JSONDecodeError is a ValueError
        return [], f"evidence unavailable: {type(exc).__name__}: {exc}"
    good = [e for e in entries if not validate_entry(e)]
    dropped = len(entries) - len(good)
    error = f"evidence: {dropped} invalid entr{'y' if dropped == 1 else 'ies'} dropped"
    return good, (error if dropped else None)


def enforce_enabled() -> bool:
    return os.environ.get(ENV_ENFORCE) == "1"


def _matches(
    entry: dict[str, Any],
    profile: Profile,
    params: dict[str, Any] | None,
    features: dict[str, Any] | None,
) -> bool:
    if entry.get("status", "measured") != "measured":
        return False
    task = entry.get("task")
    if (
        task is not None
        and features is not None
        and not all(k in features and _holds(c, features[k]) for k, c in task.items())
    ):
        return False  # a task condition needs every feature it names (unknown -> no match)
    if entry["model_class"] not in ("any", profile.model_class):
        return False
    wanted = entry.get("thinking")
    if wanted is not None and profile.thinking is not None and wanted != profile.thinking:
        return False
    when = entry.get("when")
    if when is None:
        return True
    return params is not None and all(params.get(k) == v for k, v in when.items())


def _specificity(entry: dict[str, Any]) -> tuple[int, int, int, int, str, int]:
    """Most specific wins (task > when > class > thinking); ties go to the most recent date."""
    return (
        len(entry.get("task", ())),
        1 if entry.get("when") else 0,
        1 if entry["model_class"] != "any" else 0,
        1 if "thinking" in entry else 0,
        entry["date"],
        _SEVERITY[entry["level"]],
    )


def _entry_view(entry: dict[str, Any]) -> dict[str, Any]:
    view = {
        "level": entry["level"],
        "why": f"{entry['metric']}: {entry['value']}",
        "source": entry["source"],
    }
    hint = hint_for_task(entry["task"]) if "task" in entry else None
    return {**view, "hint": hint} if hint else view


@dataclass
class Advisor:
    evidence: list[dict[str, Any]]
    load_error: str | None = None

    @classmethod
    def default(cls) -> Advisor:
        """Never raises: a broken evidence file degrades to neutral advisories plus load_error."""
        evidence, error = load_evidence_soft()
        return cls(evidence, error)

    def lookup(
        self,
        action: str,
        profile: Profile,
        params: dict[str, Any] | None = None,
        task: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """``{level, why, source}`` for ``domain.action`` under ``profile`` and the concrete call.

        Task features come from ``task`` (explicit, e.g. ``meta.advise``) or are measured from
        ``params``. A matched task-conditioned entry is evidence about this very kind of task, so
        it outranks the calibration score; calibration still overrides unconditioned entries.
        """
        try:  # features are advisory sugar: a hostile or odd param must never break a dispatch
            features = task_features.derive(
                task if task is not None else task_features.extract(action, params),
                profile.context,
            )
        except Exception:
            features = {}
        hits = [
            e
            for e in self.evidence
            if e["action"] == action and _matches(e, profile, params, features)
        ]
        best = max(hits, key=_specificity) if hits else None
        if best is not None and (best["level"] == "experimental" or "task" in best):
            return _entry_view(best)  # algorithm status or task-specific measurement
        score = profile.scores.get(action.split(".", 1)[0])
        if score is not None and score >= RECOMMEND_AT:
            return {
                "level": "recommended",
                "why": f"calibration {action.split('.', 1)[0]}={score}",
                "source": "meta.calibrate",
            }
        if score is not None and score <= DISCOURAGE_AT:
            return {
                "level": "discouraged",
                "why": f"calibration {action.split('.', 1)[0]}={score}",
                "source": "meta.calibrate",
            }
        return _entry_view(best) if best is not None else dict(NEUTRAL)

    def variants(self, action: str, profile: Profile) -> list[dict[str, Any]]:
        """Param-conditional entries for ``action`` (shown by discover/schema)."""
        return [
            {"when": e["when"], **_entry_view(e)}
            for e in self.evidence
            if e["action"] == action
            and e.get("when")
            and e["model_class"] in ("any", profile.model_class)
        ]

    def _task_rules(self, profile: Profile) -> dict[str, dict[str, list[dict[str, Any]]]]:
        """``action -> tag -> [task conditions]`` for the profile (recommended / discouraged)."""
        by_action: dict[str, dict[str, list[dict[str, Any]]]] = {}
        for e in self.evidence:
            if "task" not in e or e["level"] not in ("recommended", "discouraged"):
                continue
            if not _matches(e, profile, None, None):  # features=None: scope only
                continue
            tag = "rec" if e["level"] == "recommended" else "dis"
            slot = by_action.setdefault(e["action"], {}).setdefault(tag, [])
            if e["task"] not in slot:
                slot.append(e["task"])
        return by_action

    def _grouped(self, profile: Profile) -> dict[str, tuple[list[str], str]]:
        """``compact rule -> (actions sharing it, plain-language hint)``."""
        grouped: dict[str, tuple[list[str], str]] = {}
        for action, tags in sorted(self._task_rules(profile).items()):
            rule = " ".join(
                f"{tag}:{'|'.join(_task_text(t) for t in tasks)}"
                for tag, tasks in sorted(tags.items())
            )
            grouped.setdefault(rule, ([], hint_for_tags(tags)))[0].append(action)
        return grouped

    def rules(self, profile: Profile) -> dict[str, str]:
        """Compact conditional rules for the profile: ``"a1,a2" -> "rec:<cond>|<cond> dis:<cond>"``.

        Rendered from the same evidence the lookup uses (one source of truth); only recommended
        and discouraged task-conditioned entries appear. Actions with identical rules share a key.
        """
        return {",".join(a): rule for rule, (a, _) in self._grouped(profile).items()}

    def hints(self, profile: Profile) -> dict[str, str]:
        """Plain-language hint per rule, same keys as ``rules`` (unphrasable rules are absent)."""
        return {",".join(a): hint for _, (a, hint) in self._grouped(profile).items() if hint}
