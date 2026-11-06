"""Process version and integrity stamp: what code is this process actually running?

Adapted from agent-execution-plane ``server.py`` (``_get_git_commit`` / ``_get_schema_hash``,
see NOTICE). Fixes GAP-RT-04: the commit and the source hash are captured ONCE at import, so
``stale`` compares what the process loaded against what is on disk now, instead of reporting
the current checkout as if it were the running code. Stdlib only.
"""

import hashlib
import json
import os
import subprocess
import time
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

UNKNOWN = "unknown"
GIT_TIMEOUT_S = 2.0
HEAD_CACHE_TTL_S = 5.0
_PACKAGE_DIR = Path(__file__).resolve().parents[1]
_DIST_NAME = "chrysalis-mcp"


def _git_head() -> str:
    """Short HEAD of the repository that contains the package; ``unknown`` on any failure."""
    try:
        done = subprocess.run(  # fixed argv, no shell, no user input
            ["git", "-C", str(_PACKAGE_DIR), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=GIT_TIMEOUT_S,
            check=False,
        )
    except OSError, subprocess.SubprocessError:
        return UNKNOWN
    out = done.stdout.strip()
    return out if done.returncode == 0 and out else UNKNOWN


def source_hash_now() -> str:
    """sha256 over the package's ``*.py`` files (relative path + bytes), sorted by path."""
    digest = hashlib.sha256()
    for path in sorted(_PACKAGE_DIR.rglob("*.py")):
        digest.update(path.relative_to(_PACKAGE_DIR).as_posix().encode())
        digest.update(b"\0")
        try:
            digest.update(path.read_bytes())
        except OSError:
            digest.update(b"<unreadable>")
    return digest.hexdigest()


def _server_version() -> str:
    try:
        return metadata.version(_DIST_NAME)
    except metadata.PackageNotFoundError:
        return UNKNOWN


# Captured at import time (GAP-RT-04 fix): these describe the code this process loaded.
LOADED_COMMIT = _git_head()
LOADED_SOURCE_HASH = source_hash_now()
PROCESS_START = datetime.now(UTC)
_START_MONOTONIC = time.monotonic()
SERVER_VERSION = _server_version()

_head_cache: tuple[float, str] | None = None


def repo_head() -> str:
    """Current HEAD of the repository, cached for ``HEAD_CACHE_TTL_S`` seconds."""
    global _head_cache
    now = time.monotonic()
    if _head_cache is None or now - _head_cache[0] > HEAD_CACHE_TTL_S:
        _head_cache = (now, _git_head())
    return _head_cache[1]


def schema_hash(schema: dict[str, Any]) -> str:
    """sha256 of the canonical JSON of a registry schema."""
    canonical = json.dumps(schema, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()


def stamp(schema: dict[str, Any]) -> dict[str, Any]:
    """Version/integrity fields for ``meta.discover``."""
    head = repo_head()
    source_now = source_hash_now()
    commit_moved = LOADED_COMMIT != UNKNOWN and head != UNKNOWN and head != LOADED_COMMIT
    return {
        "server_version": SERVER_VERSION,
        "loaded_commit": LOADED_COMMIT,
        "repo_head": head,
        "stale": commit_moved or source_now != LOADED_SOURCE_HASH,
        "source_hash": LOADED_SOURCE_HASH,
        "schema_hash": schema_hash(schema),
        "pid": os.getpid(),
        "process_start_time": PROCESS_START.isoformat(timespec="seconds"),
        "uptime_s": round(time.monotonic() - _START_MONOTONIC, 1),
    }
