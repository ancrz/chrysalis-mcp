"""Advisory learning: per-model Beta(alpha, beta) outcome counts, training-free.

Bio analogy (design rationale, not a claim of equivalence):

- dopamine reward-prediction error: an outcome (verified by ``reason.check``, a free reward
  signal) nudges the belief "does using the tool pay off for this model on this task?";
- bee scouts / foragers: mostly exploit the best known arm, sometimes explore the other one.

Each ``(model, thinking, action, bucket, arm)`` cell holds ``successes`` / ``failures``; the
posterior is ``Beta(1 + successes, 1 + failures)`` (uniform prior). Arms: ``used`` (the model
called the tool) and ``self`` (it answered without). Until a bucket has ``min_n`` outcomes
(both arms together) the rule-based advice is returned untouched (the rules are the prior).
After that the arm with the higher posterior is advised: Thompson sampling seeded per
(session, state) so repeated calls give the same answer (determinism), or the posterior mean
when exploration is disabled (``explore=false`` / ``CHRYSALIS_ADVICE_EXPLORE=0``).

Storage is a SEPARATE sqlite file (``CHRYSALIS_ADVICE_DB``); the memory substrate is untouched.
"""

import hashlib
import os
import random
import re
import sqlite3
import time
from collections.abc import Iterator
from contextlib import closing, contextmanager
from pathlib import Path
from typing import Any

from ..substrate.store import resolve_path as resolve_memory_path

ENV_ADVICE_DB = "CHRYSALIS_ADVICE_DB"
ENV_EXPLORE = "CHRYSALIS_ADVICE_EXPLORE"
ENV_MIN_N = "CHRYSALIS_ADVICE_MIN_N"
DEFAULT_MIN_N = 8
ARMS = ("used", "self")
OUTCOMES = ("success", "failure")
MAX_LEARNED_LINES = 4

# --- buckets: the ONLY place where task features become a bucket --------------------------------
# Bands follow the advisory rule threshold (n_atoms >= 7 is "hard", see docs/ADVISORY.md).
_ATOM_BANDS = ((3, "0-3"), (6, "4-6"), (12, "7-12"))
_ATOM_TOP = "13+"
_ATOM_UNKNOWN = "na"
_ATOM_WORDS = {
    "0-3": "up to 3 comparisons",
    "4-6": "about 4 to 6 comparisons",
    "7-12": "about 7 to 12 comparisons",
    "13+": "13 or more comparisons",
}
_GOAL_WORDS = {"opt": "asking for the smallest/largest value", "sat": "asking for any valid values"}


def atoms_band(features: dict[str, Any]) -> str:
    n = features.get("n_atoms")
    if isinstance(n, bool) or not isinstance(n, int | float) or n < 0:
        return _ATOM_UNKNOWN
    for upper, label in _ATOM_BANDS:
        if n <= upper:
            return label
    return _ATOM_TOP


def bucket_key(features: dict[str, Any]) -> str:
    """``atoms:<band>|goal:<sat|opt|na>`` from the same features the advisory rules use."""
    goal = features.get("goal")
    return f"atoms:{atoms_band(features)}|goal:{goal if goal in _GOAL_WORDS else 'na'}"


def bucket_phrase(bucket: str) -> str:
    """Plain-language rendering of a bucket key (for prompt hints)."""
    parts = dict(p.split(":", 1) for p in bucket.split("|") if ":" in p)
    found = (_ATOM_WORDS.get(parts.get("atoms", "")), _GOAL_WORDS.get(parts.get("goal", "")))
    return " and ".join(w for w in found if w) or "this kind of task"


def model_key(
    model: str | None, family: str | None = None, params_b: float | None = None
) -> str | None:
    """Normalized model name (lower case, runs of non-alphanumerics -> ``-``); ``None`` if unknown."""
    raw = model or (f"{family}-{params_b:g}b" if family and params_b else family)
    if not raw:
        return None
    key = re.sub(r"[^a-z0-9.]+", "-", raw.lower()).strip("-")
    return key or None


def thinking_key(thinking: bool | None) -> str:
    return "na" if thinking is None else ("on" if thinking else "off")


# --- configuration -----------------------------------------------------------------------------


def _is_special(path: str) -> bool:
    return path.startswith((":memory:", "file:"))


def advice_db_path() -> str:
    """``CHRYSALIS_ADVICE_DB`` > ``advice.db`` next to the memory db (or the XDG data dir)."""
    env = os.environ.get(ENV_ADVICE_DB)
    if env:
        return env
    memory = resolve_memory_path()
    if _is_special(memory):
        data_home = os.environ.get("XDG_DATA_HOME") or str(Path.home() / ".local" / "share")
        return str(Path(data_home) / "chrysalis" / "advice.db")
    return str(Path(memory).parent / "advice.db")


def explore_enabled(param: Any = None) -> bool:
    """Per-call ``explore`` wins over ``CHRYSALIS_ADVICE_EXPLORE`` (``0``/``false`` disables)."""
    if isinstance(param, bool):
        return param
    return os.environ.get(ENV_EXPLORE, "1").strip().lower() not in ("0", "false", "off", "no")


def min_n_value(param: Any = None) -> int:
    if isinstance(param, int) and not isinstance(param, bool) and param >= 1:
        return param
    try:
        return max(1, int(os.environ.get(ENV_MIN_N, DEFAULT_MIN_N)))
    except ValueError:
        return DEFAULT_MIN_N


# --- decision ----------------------------------------------------------------------------------

Cell = tuple[int, int]  # (successes, failures)


def rate(cell: Cell) -> float | None:
    s, f = cell
    return round(s / (s + f), 3) if s + f else None


def posterior_mean(cell: Cell) -> float:
    s, f = cell
    return (1 + s) / (2 + s + f)


def seed_for(session_id: str, key: tuple[str, ...], n: int) -> int:
    digest = hashlib.sha256("|".join((session_id, *key, str(n))).encode()).digest()
    return int.from_bytes(digest[:8], "big")


def decide(used: Cell, own: Cell, *, min_n: int, explore: bool, seed: int) -> str | None:
    """``"used"`` | ``"self"`` | ``None`` (not enough evidence, or an exact tie: keep the rule)."""
    if sum(used) + sum(own) < min_n:
        return None
    if explore:
        rng = random.Random(seed)
        draw_used = rng.betavariate(1 + used[0], 1 + used[1])
        draw_own = rng.betavariate(1 + own[0], 1 + own[1])
        if draw_used == draw_own:
            return None
        return "used" if draw_used > draw_own else "self"
    mean_used, mean_own = posterior_mean(used), posterior_mean(own)
    if mean_used == mean_own:
        return None
    return "used" if mean_used > mean_own else "self"


def evidence(
    source: str, used: Cell, own: Cell, *, bucket: str, min_n: int, explore: bool
) -> dict[str, Any]:
    return {
        "source": source,
        "bucket": bucket,
        "n_used": sum(used),
        "n_self": sum(own),
        "success_rate_used": rate(used),
        "success_rate_self": rate(own),
        "min_n": min_n,
        "explore": explore,
    }


# --- store -------------------------------------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS advice_stats (
    model TEXT NOT NULL, thinking TEXT NOT NULL, action TEXT NOT NULL, bucket TEXT NOT NULL,
    arm TEXT NOT NULL CHECK (arm IN ('used','self')),
    successes INTEGER NOT NULL DEFAULT 0, failures INTEGER NOT NULL DEFAULT 0,
    updated REAL NOT NULL,
    PRIMARY KEY (model, thinking, action, bucket, arm)
)"""


@contextmanager
def _connect(path: str) -> Iterator[sqlite3.Connection]:
    if not _is_special(path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(path, timeout=5.0)) as conn:
        conn.execute(_SCHEMA)
        yield conn
        conn.commit()


class AdviceStore:
    """Stateless handle: the path is resolved on every call and each call opens its own
    connection (cross-process safe; the volume is one row update per outcome)."""

    def __init__(self, path: str | None = None) -> None:
        self._path = path

    @property
    def path(self) -> str:
        return self._path or advice_db_path()

    def _exists(self) -> bool:
        return _is_special(self.path) or Path(self.path).exists()

    def record(
        self, model: str, thinking: str, action: str, bucket: str, arm: str, success: bool
    ) -> Cell:
        if arm not in ARMS:
            raise ValueError(f"unknown arm {arm!r}")
        column = "successes" if success else "failures"  # fixed identifiers, never user input
        with _connect(self.path) as conn:
            conn.execute(
                f"INSERT INTO advice_stats(model,thinking,action,bucket,arm,{column},updated) "
                f"VALUES (?,?,?,?,?,1,?) ON CONFLICT(model,thinking,action,bucket,arm) "
                f"DO UPDATE SET {column}={column}+1, updated=excluded.updated",
                (model, thinking, action, bucket, arm, time.time()),
            )
            row = conn.execute(
                "SELECT successes, failures FROM advice_stats WHERE model=? AND thinking=? "
                "AND action=? AND bucket=? AND arm=?",
                (model, thinking, action, bucket, arm),
            ).fetchone()
        return (row[0], row[1])

    def cells(self, model: str, thinking: str, action: str, bucket: str) -> tuple[Cell, Cell]:
        """``(used, self)`` cells; zeros when the db is absent or unreadable (advice must
        degrade to the rules, never fail)."""
        out: dict[str, Cell] = {"used": (0, 0), "self": (0, 0)}
        try:
            if self._exists():
                with _connect(self.path) as conn:
                    for arm, s, f in conn.execute(
                        "SELECT arm, successes, failures FROM advice_stats "
                        "WHERE model=? AND thinking=? AND action=? AND bucket=?",
                        (model, thinking, action, bucket),
                    ):
                        out[arm] = (s, f)
        except sqlite3.Error:
            return (0, 0), (0, 0)
        return out["used"], out["self"]

    def buckets(self, model: str, thinking: str) -> list[tuple[str, str]]:
        """``[(action, bucket)]`` with any outcome for the model (fail-soft)."""
        try:
            if not self._exists():
                return []
            with _connect(self.path) as conn:
                return [
                    (a, b)
                    for a, b in conn.execute(
                        "SELECT DISTINCT action, bucket FROM advice_stats "
                        "WHERE model=? AND thinking=? ORDER BY action, bucket",
                        (model, thinking),
                    )
                ]
        except sqlite3.Error:
            return []

    def rows(
        self, model: str, action: str | None, offset: int, limit: int
    ) -> tuple[list[dict[str, Any]], int]:
        """One deterministic page of rows plus the total row count."""
        if not self._exists():
            return [], 0
        clause, args = ("model=? AND action=?", [model, action]) if action else ("model=?", [model])
        with _connect(self.path) as conn:
            total = conn.execute(
                f"SELECT COUNT(*) FROM advice_stats WHERE {clause}", args
            ).fetchone()[0]
            found = conn.execute(
                "SELECT thinking, action, bucket, arm, successes, failures FROM advice_stats "
                f"WHERE {clause} ORDER BY thinking, action, bucket, arm LIMIT ? OFFSET ?",
                [*args, limit, offset],
            ).fetchall()
        return [
            {
                "thinking": t, "action": a, "bucket": b, "arm": arm,
                "successes": s, "failures": f, "n": s + f, "rate": rate((s, f)),
            }  # fmt: skip
            for t, a, b, arm, s, f in found
        ], total

    def reset(self, model: str, action: str | None) -> int:
        if not self._exists():
            return 0
        clause, args = ("model=? AND action=?", [model, action]) if action else ("model=?", [model])
        with _connect(self.path) as conn:
            return conn.execute(f"DELETE FROM advice_stats WHERE {clause}", args).rowcount


def learned_lines(store: AdviceStore, model: str, thinking: str, min_n: int) -> list[str]:
    """Plain-language lines for the handshake prompt hint: greedy (no sampling) decision of every
    bucket with enough evidence, so the hint is stable for a given database state."""
    lines: list[str] = []
    for action, bucket in store.buckets(model, thinking):
        used, own = store.cells(model, thinking, action, bucket)
        choice = decide(used, own, min_n=min_n, explore=False, seed=0)
        if choice is None:
            continue
        verb = "use the tool" if choice == "used" else "answer yourself, without the tool"
        lines.append(
            f"For {action} on tasks {bucket_phrase(bucket)} "
            f"(learned from {sum(used) + sum(own)} outcomes), {verb}."
        )
        if len(lines) >= MAX_LEARNED_LINES:
            break
    return lines
