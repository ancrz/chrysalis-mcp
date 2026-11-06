"""Shared memory substrate (SQLite) with read/write interface segregation."""

from pathlib import Path

from .backup import BackupError
from .read import SqliteRead, SubstrateRead
from .store import SUBSTRATE_CONTRACT, Database, FlyHashParams, SubstrateContractError
from .write import SqliteWrite, SubstrateWrite


def open_substrate(
    path: str | Path | None = None, flyhash: FlyHashParams | None = None
) -> tuple[SqliteRead, SqliteWrite]:
    """Open (and migrate) the substrate. The read side has its own ``query_only`` connection.

    ``flyhash`` fixes the FlyHash parameters of a new database; ``None`` keeps the stored ones
    (or the defaults for a new database). A different value for an existing database raises
    ``SubstrateContractError``.
    """
    db = Database(path, flyhash)
    return SqliteRead(db), SqliteWrite(db)


__all__ = [
    "BackupError",
    "SUBSTRATE_CONTRACT",
    "FlyHashParams",
    "SubstrateContractError",
    "SubstrateRead",
    "SubstrateWrite",
    "open_substrate",
]
