"""Write side of the substrate. Every mutation appends to ``log`` in the same transaction."""

import json
import sqlite3
import struct
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from . import backup as _backup
from .store import MAX_EXAMPLES_PER_NODE, Database, vec_to_blob


@dataclass(frozen=True)
class IngestResult:
    """Outcome of ``ingest_batch``: texts in first-appearance order."""

    nodes: tuple[str, ...]
    preexisting: frozenset[str]  # nodes that existed before the call
    coded: tuple[str, ...]  # nodes that received a code in this call
    edges: int


class SubstrateWrite(Protocol):
    def ingest_batch(
        self,
        triples: Sequence[Mapping[str, Any]],
        code_for: Callable[[str], Sequence[int]],
        default_source: str | None = None,
        dense_for: Callable[[str], Sequence[float]] | None = None,
    ) -> IngestResult: ...
    def upsert_node(self, text: str) -> int: ...
    def add_triple(
        self,
        subj: str,
        rel: str,
        obj: str,
        weight: float = 1.0,
        source: str | None = None,
        version: int = 1,
    ) -> int: ...
    def invalidate(self, source: str, version: int | None = None) -> int: ...
    def invalidate_ids(self, source: str, version: int | None = None) -> list[int]: ...
    def put_kc_code(self, node_id: int, active: Sequence[int], k: int, m: int) -> None: ...
    def put_dense(self, node_id: int, vec: Sequence[float]) -> None: ...
    def backfill_dense(self, vectors: Mapping[int, Sequence[float]]) -> int: ...
    def bump_assoc(self, kc_ids: Iterable[int], node_id: int, delta: float) -> None: ...
    def reinforce(
        self, kc_ids: Iterable[int], node_id: int, delta: float, example: str | None = None
    ) -> int: ...
    def decay_assoc(self, decay: float, prune_below: float) -> tuple[int, int]: ...
    def backup(self, reason: str) -> dict[str, Any]: ...
    def list_backups(self) -> list[dict[str, Any]]: ...
    def restore(self, slot: int) -> dict[str, Any]: ...


def _log(conn: sqlite3.Connection, op: str, payload: dict[str, Any]) -> None:
    conn.execute("INSERT INTO log (op, payload) VALUES (?, ?)", (op, json.dumps(payload)))


def _insert_edge(
    conn: sqlite3.Connection,
    s_id: int,
    rel: str,
    o_id: int,
    weight: float,
    source: str | None,
    version: int,
) -> int:
    cur = conn.execute(
        "INSERT INTO edges (subj, rel, obj, weight, source, version) VALUES (?, ?, ?, ?, ?, ?)",
        (s_id, rel, o_id, weight, source, version),
    )
    assert cur.lastrowid is not None
    return cur.lastrowid


def _store_code(db: Database, conn: sqlite3.Connection, node_id: int, active: Sequence[int]) -> int:
    """kc_codes + kc_index for one node inside the caller's transaction; returns the cell count.

    The code must match the database's FlyHash parameters (exactly ``k`` distinct cells in
    ``[0, m)``): the Hamming/overlap equivalence depends on it.
    """
    p = db.flyhash
    ordered = sorted(set(active))
    if len(ordered) != p.k or len(active) != p.k:
        raise ValueError(f"code must have exactly k={p.k} distinct active cells")
    if ordered[0] < 0 or ordered[-1] >= p.m:
        raise ValueError(f"active indices must be in [0, {p.m})")
    blob = struct.pack(f"<{len(ordered)}I", *ordered)
    conn.execute(
        "INSERT OR REPLACE INTO kc_codes (node_id, active, k, m) VALUES (?, ?, ?, ?)",
        (node_id, blob, p.k, p.m),
    )
    db.index_put(conn, node_id, ordered)
    return len(ordered)


def _upsert_node(conn: sqlite3.Connection, text: str) -> int:
    conn.execute("INSERT OR IGNORE INTO nodes (text) VALUES (?)", (text,))
    return conn.execute("SELECT id FROM nodes WHERE text = ?", (text,)).fetchone()[0]


def _update_example(
    conn: sqlite3.Connection, node_id: int, kc_ids: Sequence[int], delta: float, query: str
) -> None:
    """Maintain the bounded ``examples`` rows of one node (caller holds the transaction)."""
    if delta > 0:
        # ts is a logical clock (strictly increasing), so "most recent" has no same-second ties.
        conn.execute(
            "INSERT INTO examples (node_id, query, weight, ts) "
            "VALUES (?, ?, ?, (SELECT COALESCE(MAX(ts), 0) + 1 FROM examples)) "
            "ON CONFLICT(node_id, query) DO UPDATE SET weight = weight + excluded.weight, "
            "ts = excluded.ts",
            (node_id, query, delta),
        )
        conn.execute(
            "DELETE FROM examples WHERE node_id = ? AND query NOT IN "
            "(SELECT query FROM examples WHERE node_id = ? ORDER BY ts DESC LIMIT ?)",
            (node_id, node_id, MAX_EXAMPLES_PER_NODE),
        )
    elif delta < 0:
        mass = conn.execute(
            "SELECT COALESCE(SUM(weight), 0.0) FROM assoc "
            "WHERE node_id = ? AND kc IN (SELECT value FROM json_each(?))",
            (node_id, json.dumps(list(kc_ids))),
        ).fetchone()[0]
        if mass <= 0.0:
            conn.execute("DELETE FROM examples WHERE node_id = ? AND query = ?", (node_id, query))


class SqliteWrite:
    def __init__(self, db: Database) -> None:
        self._db = db

    def upsert_node(self, text: str) -> int:
        with self._db.transaction() as conn:
            node_id = _upsert_node(conn, text)
            _log(conn, "upsert_node", {"id": node_id, "text": text})
            return node_id

    def add_triple(
        self,
        subj: str,
        rel: str,
        obj: str,
        weight: float = 1.0,
        source: str | None = None,
        version: int = 1,
    ) -> int:
        with self._db.transaction() as conn:
            s_id = _upsert_node(conn, subj)
            o_id = _upsert_node(conn, obj)
            cur = conn.execute(
                "INSERT INTO edges (subj, rel, obj, weight, source, version) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (s_id, rel, o_id, weight, source, version),
            )
            edge_id = cur.lastrowid
            assert edge_id is not None
            _log(
                conn,
                "add_triple",
                {"edge": edge_id, "subj": s_id, "rel": rel, "obj": o_id, "source": source},
            )
            return edge_id

    def invalidate(self, source: str, version: int | None = None) -> int:
        return len(self.invalidate_ids(source, version))

    def invalidate_ids(self, source: str, version: int | None = None) -> list[int]:
        """Soft-invalidate the valid edges of ``source`` (optionally one version); returns ids."""
        # version=NULL means "any version": a single fixed statement, no SQL composition.
        args = (source, version, version)
        with self._db.transaction() as conn:
            ids = [
                r[0]
                for r in conn.execute(
                    "SELECT id FROM edges WHERE source = ? AND valid = 1 "
                    "AND (? IS NULL OR version = ?) ORDER BY id",
                    args,
                )
            ]
            conn.execute(
                "UPDATE edges SET valid = 0 WHERE source = ? AND valid = 1 "
                "AND (? IS NULL OR version = ?)",
                args,
            )
            _log(
                conn,
                "invalidate",
                {"source": source, "version": version, "n": len(ids), "edge_ids": ids},
            )
            return ids

    def put_kc_code(self, node_id: int, active: Sequence[int], k: int, m: int) -> None:
        """Store a node's FlyHash code in ``kc_codes`` and ``kc_index`` in one transaction."""
        p = self._db.flyhash
        if m != p.m or k != p.k:
            raise ValueError(f"code must have m={p.m} and exactly k={p.k} distinct active cells")
        with self._db.transaction() as conn:
            n_active = _store_code(self._db, conn, node_id, active)
            _log(conn, "put_kc_code", {"node_id": node_id, "n_active": n_active, "k": k, "m": m})

    def put_dense(self, node_id: int, vec: Sequence[float]) -> None:
        """Store (or replace) a node's dense vector (normalised) in one logged transaction."""
        with self._db.transaction() as conn:
            if conn.execute("SELECT 1 FROM nodes WHERE id = ?", (node_id,)).fetchone() is None:
                raise ValueError(f"unknown node id {node_id}")
            self._db.dense_put(conn, node_id, vec_to_blob(vec, self._db.flyhash.d))
            _log(conn, "put_dense", {"node_id": node_id})

    def backfill_dense(self, vectors: Mapping[int, Sequence[float]]) -> int:
        """Store many dense vectors in one transaction (migration backfill); returns the count."""
        d = self._db.flyhash.d
        with self._db.transaction() as conn:
            for node_id, vec in vectors.items():
                self._db.dense_put(conn, node_id, vec_to_blob(vec, d))
            _log(conn, "backfill_dense", {"n": len(vectors)})
        return len(vectors)

    def ingest_batch(
        self,
        triples: Sequence[Mapping[str, Any]],
        code_for: Callable[[str], Sequence[int]],
        default_source: str | None = None,
        dense_for: Callable[[str], Sequence[float]] | None = None,
    ) -> IngestResult:
        """Nodes, edges, codes and one log row in a SINGLE transaction (all or nothing).

        ``triples``: mappings with ``s``, ``r``, ``o``, optional ``weight``/``source``/``version``.
        ``code_for(text)`` is called inside the transaction for every node that has no code yet and
        must be pure (no substrate access). ``dense_for(text)`` (optional, same contract) supplies
        the dense vector of every node that has none yet. Edges are never deduplicated.
        """
        order = list(dict.fromkeys(t[key] for t in triples for key in ("s", "o")))
        with self._db.transaction() as conn:
            ids: dict[str, int] = {}
            preexisting: set[str] = set()
            for text in order:
                row = conn.execute("SELECT id FROM nodes WHERE text = ?", (text,)).fetchone()
                if row is not None:
                    preexisting.add(text)
                ids[text] = row[0] if row is not None else _upsert_node(conn, text)
            edge_ids: list[int] = []
            sources: list[dict[str, Any]] = []
            for t in triples:
                source = t.get("source") or default_source
                version = int(t.get("version", 1))
                edge_ids.append(
                    _insert_edge(
                        conn,
                        ids[t["s"]],
                        t["r"],
                        ids[t["o"]],
                        float(t.get("weight", 1.0)),
                        source,
                        version,
                    )
                )
                if {"source": source, "version": version} not in sources:
                    sources.append({"source": source, "version": version})
            coded: list[str] = []
            for text in order:
                has_code = conn.execute(
                    "SELECT 1 FROM kc_codes WHERE node_id = ?", (ids[text],)
                ).fetchone()
                if has_code is None:
                    _store_code(self._db, conn, ids[text], code_for(text))
                    coded.append(text)
                if (
                    dense_for is not None
                    and conn.execute(
                        "SELECT 1 FROM dense_vecs WHERE node_id = ?", (ids[text],)
                    ).fetchone()
                    is None
                ):
                    self._db.dense_put(
                        conn, ids[text], vec_to_blob(dense_for(text), self._db.flyhash.d)
                    )
            _log(
                conn,
                "ingest_batch",
                {
                    "nodes": len(order) - len(preexisting),
                    "edges": len(triples),
                    "codes": len(coded),
                    "source": default_source,
                    # Full write set, so forget/consolidate/audit can reconstruct this call.
                    "node_ids": [ids[t] for t in order if t not in preexisting],
                    "edge_ids": edge_ids,
                    "coded_ids": [ids[t] for t in coded],
                    "sources": sources,
                },
            )
        return IngestResult(tuple(order), frozenset(preexisting), tuple(coded), len(triples))

    def bump_assoc(self, kc_ids: Iterable[int], node_id: int, delta: float) -> None:
        ids = list(kc_ids)
        with self._db.transaction() as conn:
            for kc in ids:
                conn.execute(
                    "INSERT INTO assoc (kc, node_id, weight) VALUES (?, ?, ?) "
                    "ON CONFLICT(kc, node_id) DO UPDATE SET weight = weight + excluded.weight",
                    (kc, node_id, delta),
                )
            _log(conn, "bump_assoc", {"node_id": node_id, "kcs": ids, "delta": delta})

    def reinforce(
        self, kc_ids: Iterable[int], node_id: int, delta: float, example: str | None = None
    ) -> int:
        """Add ``delta`` to ``assoc[kc, node_id]`` for each distinct kc, in one transaction.

        ``example`` (the teach query) is stored in ``examples`` on a positive ``delta`` (deduped by
        query: weight accumulates, ts refreshes; only the ``MAX_EXAMPLES_PER_NODE`` most recent
        survive). On a negative ``delta`` the matching example is removed once the node's summed
        assoc over these kcs is <= 0. Raises ``ValueError`` (rolled back) when the node does not
        exist. Returns the kc count.
        """
        ids = sorted(set(kc_ids))
        with self._db.transaction() as conn:
            if conn.execute("SELECT 1 FROM nodes WHERE id = ?", (node_id,)).fetchone() is None:
                raise ValueError(f"unknown node id {node_id}")
            for kc in ids:
                conn.execute(
                    "INSERT INTO assoc (kc, node_id, weight) VALUES (?, ?, ?) "
                    "ON CONFLICT(kc, node_id) DO UPDATE SET weight = weight + excluded.weight",
                    (kc, node_id, delta),
                )
            if example is not None:
                _update_example(conn, node_id, ids, delta, example)
            _log(conn, "reinforce", {"node_id": node_id, "n_kcs": len(ids), "delta": delta})
        return len(ids)

    def decay_assoc(self, decay: float, prune_below: float) -> tuple[int, int]:
        """``assoc.weight *= decay`` then delete ``|weight| < prune_below``: ``(decayed, pruned)``.

        The log is append-only and is never pruned.
        """
        with self._db.transaction() as conn:
            decayed = conn.execute("UPDATE assoc SET weight = weight * ?", (decay,)).rowcount
            pruned = conn.execute(
                "DELETE FROM assoc WHERE abs(weight) < ?", (prune_below,)
            ).rowcount
            # Examples follow assoc: a node whose assoc is fully gone keeps no examples.
            conn.execute("DELETE FROM examples WHERE node_id NOT IN (SELECT node_id FROM assoc)")
            _log(
                conn,
                "decay_assoc",
                {"decay": decay, "prune_below": prune_below, "decayed": decayed, "pruned": pruned},
            )
        return decayed, pruned

    def backup(self, reason: str) -> dict[str, Any]:
        """Verified rotating snapshot (see ``substrate.backup``); raises ``BackupError``."""
        return _backup.snapshot(self._db, reason)

    def list_backups(self) -> list[dict[str, Any]]:
        return _backup.list_backups(self._db)

    def restore(self, slot: int) -> dict[str, Any]:
        return _backup.restore(self._db, slot)
