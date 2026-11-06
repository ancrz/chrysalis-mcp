"""SQLite connection, schema migrations and path resolution for the shared memory substrate.

Third-party dependency: ``sqlite-vec`` (optional, fail-soft; loading pattern adapted from the AEP
``engine/db.py`` import/load/fallback, see NOTICE). This package must not import
``chrysalis_mcp.core|domains|algorithms|server`` (Topos invariant, enforced by tests).

Contract history: v1 = base schema; v2 = FlyHash parameters in ``meta`` plus the ``kc_index``
sqlite-vec table (``bit[m]`` codes, rowid = node id, Hamming KNN); v3 = dense float vectors:
``dense_vecs`` (source of truth, one L2-normalised float32 vector of dimension ``flyhash.d`` per
node) plus the ``dense_index`` sqlite-vec table (``float[d]``, cosine KNN). The v2 -> v3 migration
only adds tables; nodes without a dense vector are backfilled by the memory domain on open.
v4 = ``examples`` (bounded per-node teach queries so learned associations can be explained to the
model). The v3 -> v4 migration only adds that table (backfill = none: associations taught before v4
simply have no stored example). Backups are whole-database copies, so they include ``examples``;
``restore`` only accepts snapshots written under the current contract.
"""

import os
import sqlite3
import struct
import threading
import uuid
from collections.abc import Iterator, Sequence
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path

try:  # fail-soft: the substrate works (slower) without the vector extension
    import sqlite_vec as _sqlite_vec
except ImportError, OSError:  # pragma: no cover - depends on host
    _sqlite_vec = None

SUBSTRATE_CONTRACT = "v4"
MAX_EXAMPLES_PER_NODE = 3  # most recent teach queries kept per node
ENV_DB_PATH = "CHRYSALIS_DB"
MEMORY_PATH = ":memory:"
BUSY_TIMEOUT_MS = 5000

_SCHEMA_V1 = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS nodes (
    id INTEGER PRIMARY KEY,
    text TEXT UNIQUE NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS edges (
    id INTEGER PRIMARY KEY,
    subj INTEGER NOT NULL REFERENCES nodes(id),
    rel TEXT NOT NULL,
    obj INTEGER NOT NULL REFERENCES nodes(id),
    weight REAL NOT NULL DEFAULT 1.0,
    source TEXT,
    version INTEGER NOT NULL DEFAULT 1,
    valid INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_edges_subj ON edges(subj);
CREATE INDEX IF NOT EXISTS idx_edges_obj ON edges(obj);
CREATE INDEX IF NOT EXISTS idx_edges_source_version ON edges(source, version);
CREATE TABLE IF NOT EXISTS kc_codes (
    node_id INTEGER PRIMARY KEY REFERENCES nodes(id),
    active BLOB NOT NULL,
    k INTEGER,
    m INTEGER
);
CREATE TABLE IF NOT EXISTS assoc (
    kc INTEGER NOT NULL,
    node_id INTEGER NOT NULL REFERENCES nodes(id),
    weight REAL NOT NULL,
    PRIMARY KEY (kc, node_id)
);
CREATE TABLE IF NOT EXISTS log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    op TEXT NOT NULL,
    payload TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
"""

_SCHEMA_V3 = """
CREATE TABLE IF NOT EXISTS dense_vecs (
    node_id INTEGER PRIMARY KEY REFERENCES nodes(id),
    vec BLOB NOT NULL
);
"""

_SCHEMA_V4 = """
CREATE TABLE IF NOT EXISTS examples (
    node_id INTEGER NOT NULL REFERENCES nodes(id),
    query TEXT NOT NULL,
    weight REAL NOT NULL,
    ts INTEGER NOT NULL,
    PRIMARY KEY (node_id, query)
);
"""


@dataclass(frozen=True)
class FlyHashParams:
    """FlyHash parameters, fixed at database creation (changing them invalidates every code)."""

    d: int = 1024  # input feature dimension
    m: int = 8192  # Kenyon cells (code width in bits)
    k: int = 64  # active cells per code (winner-take-all)
    c: int = 12  # input indices sampled per Kenyon cell
    seed: int = 1729

    def __post_init__(self) -> None:
        ints = (self.d, self.m, self.k, self.c, self.seed)
        if any(isinstance(v, bool) or not isinstance(v, int) for v in ints):
            raise ValueError(f"FlyHash parameters must be integers: {self}")
        if not (MIN_M <= self.m <= MAX_M and self.m % 8 == 0):
            raise ValueError(f"FlyHash m must be a multiple of 8 in [{MIN_M}, {MAX_M}]")
        if not 1 <= self.d <= MAX_D:
            raise ValueError(f"FlyHash d must be in [1, {MAX_D}]")
        if not (1 <= self.k <= self.m and 1 <= self.c <= self.d):
            raise ValueError(f"invalid FlyHash parameters: {self}")


MIN_M, MAX_M, MAX_D = 8, 65536, 65536
_FLYHASH_KEYS = ("d", "m", "k", "c", "seed")
MAX_VEC_K = 4096  # sqlite-vec KNN limit

TABLES = ("meta", "nodes", "edges", "kc_codes", "assoc", "log", "dense_vecs", "examples")


class SubstrateContractError(RuntimeError):
    """The database was written with a newer data contract than this code understands."""


def _contract_number(contract: str) -> int:
    if not contract.startswith("v") or not contract[1:].isdigit():
        raise SubstrateContractError(f"unparseable substrate contract: {contract!r}")
    return int(contract[1:])


def active_to_bits(active: Sequence[int], m: int) -> bytes:
    """Active KC indices -> ``m/8`` byte bit-vector (bit i of byte j is KC ``8*j + i``)."""
    bits = bytearray(m // 8)
    for kc in active:
        bits[kc >> 3] |= 1 << (kc & 7)
    return bytes(bits)


def _load_vec(conn: sqlite3.Connection) -> bool:
    """Load sqlite-vec into ``conn``; never raises. Extension loading is re-disabled afterwards."""
    if _sqlite_vec is None:
        return False
    try:
        conn.enable_load_extension(True)
        try:
            _sqlite_vec.load(conn)
        finally:
            conn.enable_load_extension(False)
        conn.execute("SELECT vec_version()").fetchone()
    except sqlite3.Error, OSError, AttributeError:
        return False
    return True


_READ_ACTIONS = frozenset(
    (sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION, sqlite3.SQLITE_RECURSIVE)
)


def _read_only_authorizer(action: int, *_args: object) -> int:
    """Read connection policy: SELECT/READ/FUNCTION only; ATTACH, PRAGMA, DDL and DML denied."""
    return sqlite3.SQLITE_OK if action in _READ_ACTIONS else sqlite3.SQLITE_DENY


def resolve_path(path: str | os.PathLike[str] | None = None) -> str:
    """Explicit path > ``CHRYSALIS_DB`` > ``$XDG_DATA_HOME/chrysalis/substrate.db``."""
    if path is not None:
        return str(path)
    env = os.environ.get(ENV_DB_PATH)
    if env:
        return env
    data_home = os.environ.get("XDG_DATA_HOME") or str(Path.home() / ".local" / "share")
    return str(Path(data_home) / "chrysalis" / "substrate.db")


class Database:
    """A write connection and a separate ``query_only`` read connection to the same database.

    Each connection is serialized by its own re-entrant lock. The read connection cannot mutate
    data at the SQLite level (``PRAGMA query_only``), so the read side has no route to a write.
    In-memory databases use a shared-cache URI so both connections see the same data.
    """

    def __init__(
        self,
        path: str | os.PathLike[str] | None = None,
        flyhash: FlyHashParams | None = None,
    ) -> None:
        self.path = resolve_path(path)
        self._is_memory = self.path == MEMORY_PATH
        if self._is_memory:
            target, uri = f"file:chrysalis-{uuid.uuid4().hex}?mode=memory&cache=shared", True
        else:
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
            target, uri = self.path, False
        self._write_lock = threading.RLock()
        self._read_lock = threading.RLock()
        self._wconn = self._connect(target, uri)
        self._rconn: sqlite3.Connection | None = None
        try:
            if not self._is_memory:
                self._wconn.execute("PRAGMA journal_mode = WAL")
                self._wconn.execute("PRAGMA synchronous = NORMAL")  # safe under WAL
            self.vec_available = _load_vec(self._wconn)
            self.flyhash = self._migrate(flyhash)
            self._rconn = self._connect(target, uri)
            self._rconn.execute("PRAGMA query_only = ON")
            if self._is_memory:
                self._rconn.execute("PRAGMA read_uncommitted = ON")  # shared-cache table locks
            self.vec_available = self.vec_available and _load_vec(self._rconn)
            self._rconn.set_authorizer(_read_only_authorizer)  # last: nothing else may run after
        except BaseException:
            self.close()
            raise

    @staticmethod
    def _connect(target: str, uri: bool) -> sqlite3.Connection:
        # isolation_level=None: autocommit, transactions are explicit (BEGIN IMMEDIATE).
        conn = sqlite3.connect(target, uri=uri, isolation_level=None, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout = 5000")  # == BUSY_TIMEOUT_MS
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def _migrate(self, requested: FlyHashParams | None) -> FlyHashParams:
        with self.transaction() as conn:
            for statement in (_SCHEMA_V1 + _SCHEMA_V3 + _SCHEMA_V4).split(";"):
                if statement.strip():
                    conn.execute(statement)
            row = conn.execute("SELECT value FROM meta WHERE key = 'contract'").fetchone()
            found = _contract_number(row["value"]) if row else 0
            if found > _contract_number(SUBSTRATE_CONTRACT):
                raise SubstrateContractError(
                    f"database contract {row['value']} is newer than supported "
                    f"{SUBSTRATE_CONTRACT}; refusing to open"
                )
            params = self._reconcile_params(conn, requested)
            conn.execute(
                "INSERT INTO meta (key, value) VALUES ('contract', ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (SUBSTRATE_CONTRACT,),
            )
            if self.vec_available:
                self._sync_kc_index(conn, params)
                self._sync_dense_index(conn, params)
            return params

    @staticmethod
    def _reconcile_params(
        conn: sqlite3.Connection, requested: FlyHashParams | None
    ) -> FlyHashParams:
        stored = {
            r["key"][len("flyhash_") :]: r["value"]
            for r in conn.execute("SELECT key, value FROM meta WHERE key GLOB 'flyhash_*'")
        }
        if stored:
            if set(stored) != set(_FLYHASH_KEYS):
                raise SubstrateContractError(
                    f"incomplete FlyHash parameters in meta: {sorted(stored)}"
                )
            try:
                current = FlyHashParams(**{k: int(v) for k, v in stored.items()})
            except (ValueError, TypeError) as exc:
                raise SubstrateContractError(f"invalid FlyHash parameters in meta: {exc}") from exc
            if requested is not None and requested != current:
                raise SubstrateContractError(
                    f"FlyHash parameters are fixed at creation: stored {current}, "
                    f"requested {requested}"
                )
            return current
        params = requested or FlyHashParams()
        for key in _FLYHASH_KEYS:
            conn.execute(
                "INSERT INTO meta (key, value) VALUES (?, ?)",
                (f"flyhash_{key}", str(getattr(params, key))),
            )
        return params

    def _sync_kc_index(self, conn: sqlite3.Connection, params: FlyHashParams) -> None:
        """Create ``kc_index`` and (re)build it from ``kc_codes`` when missing or marked stale."""
        exists = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE name = ?", ("kc_index",)
        ).fetchone()
        stale = conn.execute("SELECT 1 FROM meta WHERE key = 'kc_index_stale'").fetchone()
        if exists and stale:
            conn.execute("DROP TABLE kc_index")
            exists = None
        if not exists:
            # Only the integer bit width is interpolated: ``params.m`` is a FlyHashParams int
            # validated to 8 <= m <= 65536 and m % 8 == 0 (no user-controlled text reaches SQL).
            # nosemgrep: python.lang.security.audit.formatted-sql-query.formatted-sql-query
            conn.execute(f"CREATE VIRTUAL TABLE kc_index USING vec0(code bit[{int(params.m)}])")
            rows = conn.execute("SELECT node_id, active FROM kc_codes WHERE m = ?", (params.m,))
            for r in rows.fetchall():
                self.index_put(conn, r["node_id"], _unpack(r["active"]), params)
        conn.execute("DELETE FROM meta WHERE key = 'kc_index_stale'")

    def index_put(
        self,
        conn: sqlite3.Connection,
        node_id: int,
        active: Sequence[int],
        params: FlyHashParams | None = None,
    ) -> None:
        """Mirror one code into ``kc_index`` (caller holds the write transaction).

        Without the extension the index is never touched; if it exists on disk it is flagged
        stale so the next extension-enabled open rebuilds it.
        """
        p = params or self.flyhash
        if self.vec_available:
            conn.execute("DELETE FROM kc_index WHERE rowid = ?", (node_id,))
            conn.execute(
                "INSERT INTO kc_index (rowid, code) VALUES (?, vec_bit(?))",
                (node_id, active_to_bits(active, p.m)),
            )
        elif conn.execute("SELECT 1 FROM sqlite_master WHERE name = ?", ("kc_index",)).fetchone():
            conn.execute("INSERT OR REPLACE INTO meta (key, value) VALUES ('kc_index_stale', '1')")

    def _sync_dense_index(self, conn: sqlite3.Connection, params: FlyHashParams) -> None:
        """Create ``dense_index`` and (re)build it from ``dense_vecs`` when missing or stale."""
        exists = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE name = ?", ("dense_index",)
        ).fetchone()
        stale = conn.execute("SELECT 1 FROM meta WHERE key = 'dense_index_stale'").fetchone()
        if exists and stale:
            conn.execute("DROP TABLE dense_index")
            exists = None
        if not exists:
            # Only the integer dimension is interpolated: ``params.d`` is a FlyHashParams int
            # validated to 1 <= d <= 65536 (no user-controlled text reaches SQL).
            # nosemgrep: python.lang.security.audit.formatted-sql-query.formatted-sql-query
            conn.execute(
                "CREATE VIRTUAL TABLE dense_index USING vec0("
                f"embedding float[{int(params.d)}] distance_metric=cosine)"
            )
            rows = conn.execute("SELECT node_id, vec FROM dense_vecs").fetchall()
            for r in rows:
                conn.execute(
                    "INSERT INTO dense_index (rowid, embedding) VALUES (?, ?)",
                    (r["node_id"], r["vec"]),
                )
        conn.execute("DELETE FROM meta WHERE key = 'dense_index_stale'")

    def dense_put(self, conn: sqlite3.Connection, node_id: int, vec_blob: bytes) -> None:
        """Store one float32 vector in ``dense_vecs`` and mirror it into ``dense_index``.

        Caller holds the write transaction and guarantees ``len(vec_blob) == 4 * flyhash.d``.
        Without the extension an existing index is flagged stale for the next rebuild.
        """
        conn.execute(
            "INSERT OR REPLACE INTO dense_vecs (node_id, vec) VALUES (?, ?)", (node_id, vec_blob)
        )
        if self.vec_available:
            conn.execute("DELETE FROM dense_index WHERE rowid = ?", (node_id,))
            conn.execute(
                "INSERT INTO dense_index (rowid, embedding) VALUES (?, ?)", (node_id, vec_blob)
            )
        elif conn.execute(
            "SELECT 1 FROM sqlite_master WHERE name = ?", ("dense_index",)
        ).fetchone():
            conn.execute(
                "INSERT OR REPLACE INTO meta (key, value) VALUES ('dense_index_stale', '1')"
            )

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """Atomic write scope: commit on success, rollback on any exception."""
        with self._write_lock:
            self._wconn.execute("BEGIN IMMEDIATE")
            try:
                yield self._wconn
            except BaseException:
                self._wconn.execute("ROLLBACK")
                raise
            try:
                self._wconn.execute("COMMIT")
            except BaseException:
                with suppress(sqlite3.Error):  # a failed COMMIT must not leave the txn open
                    self._wconn.execute("ROLLBACK")
                raise

    @property
    def is_memory(self) -> bool:
        return self._is_memory

    def copy_to(self, dest: str | os.PathLike[str]) -> None:
        """Consistent online copy of the whole database into ``dest`` (SQLite backup API)."""
        with self._write_lock:
            target = sqlite3.connect(dest, isolation_level=None)
            try:
                self._wconn.backup(target)
                target.execute("PRAGMA journal_mode = DELETE")  # self-contained single file
            finally:
                target.close()

    def replace_from(self, src: str | os.PathLike[str]) -> None:
        """Overwrite the live database with ``src`` under the write lock (SQLite backup API)."""
        with self._write_lock:
            source = sqlite3.connect(f"file:{src}?mode=ro", uri=True, isolation_level=None)
            try:
                source.backup(self._wconn)
                if not self._is_memory:  # the copy's header carries the DELETE journal mode
                    self._wconn.execute("PRAGMA journal_mode = WAL")
            finally:
                source.close()

    @contextmanager
    def reader(self) -> Iterator[sqlite3.Connection]:
        """The ``query_only`` connection: any write attempt raises ``sqlite3.OperationalError``."""
        assert self._rconn is not None
        with self._read_lock:
            yield self._rconn

    def close(self) -> None:
        for conn in (self._rconn, self._wconn):
            if conn is not None:
                with suppress(sqlite3.Error):
                    conn.close()


def _unpack(blob: bytes) -> tuple[int, ...]:
    return struct.unpack(f"<{len(blob) // 4}I", blob)


def vec_to_blob(vec: Sequence[float], d: int) -> bytes:
    """L2-normalised little-endian float32 blob of dimension ``d`` (rejects zero/non-finite)."""
    if len(vec) != d:
        raise ValueError(f"dense vector must have dimension {d}, got {len(vec)}")
    norm = sum(float(x) * float(x) for x in vec) ** 0.5
    if not norm or norm != norm or norm == float("inf"):
        raise ValueError("dense vector must be finite and non-zero")
    return struct.pack(f"<{d}f", *(float(x) / norm for x in vec))
