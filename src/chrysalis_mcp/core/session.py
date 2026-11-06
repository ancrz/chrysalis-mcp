"""In-process session cache with TTL and a size cap.

Adapted from the session semantics of AEP ``engine/ipc_pool.py`` (TTL, ``store_as``); the UNIX
socket / shared-memory transport is deliberately not ported (see NOTICE).
"""

import copy
import time
from collections.abc import Callable
from typing import Any

DEFAULT_TTL_SEC = 900.0
DEFAULT_MAX_ENTRIES = 256
DEFAULT_SESSION = "default"


class _Missing:
    def __repr__(self) -> str:
        return "MISSING"


MISSING: Any = _Missing()


def resolve_session_id(session_id: str | None) -> str:
    return session_id.strip() if session_id and session_id.strip() else DEFAULT_SESSION


class SessionCache:
    """Values are isolated per ``(session_id, key)`` and expire ``ttl_sec`` after being stored."""

    def __init__(
        self,
        ttl_sec: float = DEFAULT_TTL_SEC,
        max_entries: int = DEFAULT_MAX_ENTRIES,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._ttl = ttl_sec
        self._max = max_entries
        self._clock = clock
        self._data: dict[tuple[str, str], tuple[float, Any]] = {}

    def put(self, session_id: str | None, key: str, value: Any) -> None:
        self.purge()
        slot = (resolve_session_id(session_id), key)
        self._data.pop(slot, None)
        self._data[slot] = (self._clock() + self._ttl, copy.deepcopy(value))
        while len(self._data) > self._max:
            oldest = min(self._data, key=lambda s: self._data[s][0])
            del self._data[oldest]

    def get(self, session_id: str | None, key: str, default: Any = MISSING) -> Any:
        slot = (resolve_session_id(session_id), key)
        entry = self._data.get(slot)
        if entry is None:
            return default
        if entry[0] <= self._clock():
            del self._data[slot]
            return default
        return copy.deepcopy(entry[1])

    def purge(self) -> None:
        now = self._clock()
        for slot in [s for s, (expires, _) in self._data.items() if expires <= now]:
            del self._data[slot]

    def __len__(self) -> int:
        self.purge()
        return len(self._data)
