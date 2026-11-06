"""Rotating, integrity-checked backups of the substrate before lossy writes.

Adapted from agent-execution-plane ``engine/backup.py`` (snapshot + verify; see NOTICE):
``sqlite3`` online backup API, ``PRAGMA integrity_check`` on the copy, fixed rotating slots and a
sidecar metadata file. Depends on stdlib and ``substrate.store`` only.

Layout: ``<db_dir>/backups/substrate.<slot>.db`` + ``substrate.<slot>.json``.
"""

import contextlib
import json
import os
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .store import SUBSTRATE_CONTRACT, Database

SLOTS = 3
BACKUP_DIR = "backups"
SKIPPED_MEMORY = {"skipped": "memory_db"}


class BackupError(RuntimeError):
    """A backup could not be created, verified or restored; the live database is untouched."""


def _dir(db: Database) -> Path:
    return Path(db.path).parent / BACKUP_DIR


def _paths(db: Database, slot: int) -> tuple[Path, Path]:
    base = _dir(db)
    return base / f"substrate.{slot}.db", base / f"substrate.{slot}.json"


def _check_slot(slot: Any) -> int:
    if isinstance(slot, bool) or not isinstance(slot, int) or not 0 <= slot < SLOTS:
        raise BackupError(f"slot must be an integer in [0, {SLOTS - 1}]")
    return slot


def _inspect(path: Path) -> dict[str, Any]:
    """Open a copy read-only; verify integrity; return ``{contract, log_max_id}``."""
    try:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    except sqlite3.Error as exc:
        raise BackupError(f"cannot open backup: {exc}") from exc
    try:
        verdict = conn.execute("PRAGMA integrity_check").fetchall()
        if verdict != [("ok",)]:
            raise BackupError(f"integrity_check failed: {verdict[:3]}")
        row = conn.execute("SELECT value FROM meta WHERE key = 'contract'").fetchone()
        log_max = conn.execute("SELECT COALESCE(MAX(id), 0) FROM log").fetchone()[0]
    except sqlite3.Error as exc:
        raise BackupError(f"backup is not a valid substrate: {exc}") from exc
    finally:
        conn.close()
    return {"contract": row[0] if row else None, "log_max_id": log_max}


def _metadata(db: Database, slot: int) -> dict[str, Any] | None:
    db_file, meta_file = _paths(db, slot)
    if not db_file.exists():
        return None
    try:
        meta = json.loads(meta_file.read_text())
        return meta if isinstance(meta, dict) else {"slot": slot}
    except OSError, ValueError:
        return {"slot": slot, "created_at": "", "reason": "metadata unreadable"}


def list_backups(db: Database) -> list[dict[str, Any]]:
    """Metadata of the existing slots (no data), oldest first."""
    found = [m for slot in range(SLOTS) if (m := _metadata(db, slot)) is not None]
    return sorted(found, key=lambda m: str(m.get("created_at", "")))


def _next_slot(db: Database) -> int:
    existing = {m["slot"]: str(m.get("created_at", "")) for m in list_backups(db)}
    for slot in range(SLOTS):
        if slot not in existing:
            return slot
    return min(existing, key=lambda s: existing[s])


def snapshot(db: Database, reason: str) -> dict[str, Any]:
    """Copy the database into the next rotating slot after verifying the copy.

    In-memory databases return ``{"skipped": "memory_db"}``. On any failure the temporary copy is
    removed, the previous slot content stays intact and ``BackupError`` is raised.
    """
    if db.is_memory:
        return dict(SKIPPED_MEMORY)
    slot = _next_slot(db)
    db_file, meta_file = _paths(db, slot)
    db_file.parent.mkdir(parents=True, exist_ok=True)
    tmp = db_file.with_suffix(".tmp")
    try:
        db.copy_to(tmp)
        facts = _inspect(tmp)
        meta = {
            "slot": slot,
            "created_at": datetime.now(UTC).isoformat(timespec="microseconds"),
            "reason": reason,
            **facts,
        }
        meta_file.write_text(json.dumps(meta))
        os.replace(tmp, db_file)
    except (sqlite3.Error, OSError) as exc:
        raise BackupError(f"snapshot failed: {exc}") from exc
    finally:
        with contextlib.suppress(OSError):
            tmp.unlink()
    return meta


def restore(db: Database, slot: int) -> dict[str, Any]:
    """Replace the live data with backup ``slot``; reversible (current state is snapshotted first).

    Order: (1) verify the backup and pin a private copy of it (the pre-restore snapshot may
    rotate over this very slot), (2) snapshot the current state, (3) restore under the write
    lock, (4) append a ``restore`` log row to the restored database.
    """
    if db.is_memory:
        raise BackupError("backups are unavailable for in-memory databases")
    slot = _check_slot(slot)
    db_file, _ = _paths(db, slot)
    if not db_file.exists():
        raise BackupError(f"no backup in slot {slot}")
    facts = _inspect(db_file)
    if facts["contract"] != SUBSTRATE_CONTRACT:
        raise BackupError(
            f"backup contract {facts['contract']!r} != current {SUBSTRATE_CONTRACT!r}; refusing"
        )
    pinned = db_file.with_name(f"restore-{os.getpid()}.tmp")
    try:
        try:
            pinned.write_bytes(db_file.read_bytes())
            _inspect(pinned)  # the pinned copy is what gets restored
        except OSError as exc:
            raise BackupError(f"cannot pin backup: {exc}") from exc
        pre = snapshot(db, f"pre-restore from slot {slot}")
        try:
            db.replace_from(pinned)
            with db.transaction() as conn:
                conn.execute(
                    "INSERT INTO log (op, payload) VALUES ('restore', ?)",
                    (json.dumps({"from_slot": slot, "discarded_after": facts["log_max_id"]}),),
                )
        except sqlite3.Error as exc:
            raise BackupError(f"restore failed: {exc}") from exc
    finally:
        with contextlib.suppress(OSError):
            pinned.unlink()
    return {
        "restored_slot": slot,
        "pre_restore_slot": pre.get("slot"),
        "discarded_after_log_id": facts["log_max_id"],
    }
