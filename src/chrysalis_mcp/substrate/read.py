"""Read-only side of the substrate. Exposes no mutating method (interface segregation)."""

import json
import struct
from collections.abc import Iterable, Sequence
from typing import Any, Protocol

import numpy as np

from .write import protected_clause
from .store import (
    MAX_VEC_K,
    Database,
    FlyHashParams,
    active_to_bits,
    vec_to_blob,
)

Row = dict[str, Any]
COSINE_DECIMALS = 6  # float32 noise floor: both backends round to it, then order by (-sim, id)

_COUNT_SQL = {
    "meta": "SELECT COUNT(*) FROM meta",
    "nodes": "SELECT COUNT(*) FROM nodes",
    "edges": "SELECT COUNT(*) FROM edges",
    "kc_codes": "SELECT COUNT(*) FROM kc_codes",
    "assoc": "SELECT COUNT(*) FROM assoc",
    "log": "SELECT COUNT(*) FROM log",
    "dense_vecs": "SELECT COUNT(*) FROM dense_vecs",
    "examples": "SELECT COUNT(*) FROM examples",
    "teach_queries": "SELECT COUNT(*) FROM teach_queries",
    "node_kcs": "SELECT COUNT(*) FROM node_kcs",
    "edge_usage": "SELECT COUNT(*) FROM edge_usage",
    "assoc_tags": "SELECT COUNT(*) FROM assoc_tags",
    "flyvec_tokens": "SELECT COUNT(*) FROM flyvec_tokens",
    "flyvec_nodes": "SELECT COUNT(*) FROM flyvec_nodes",
}


class SubstrateRead(Protocol):
    def contract(self) -> str: ...
    def get_node(self, key: str | int) -> Row | None: ...
    def neighbors(self, node_id: int, valid_only: bool = True) -> list[Row]: ...
    def edges_by_source(self, source: str) -> list[Row]: ...
    def kc_code(self, node_id: int) -> Row | None: ...
    def assoc_for(self, kc_ids: Iterable[int]) -> list[Row]: ...
    def examples_for(self, node_ids: Iterable[int]) -> dict[int, list[Row]]: ...
    def recent_examples(self, exclude_ids: Iterable[int], limit: int) -> list[str]: ...
    def teach_for(self, node_ids: Iterable[int]) -> dict[int, list[Row]]: ...
    def kc_node_counts(self, kc_ids: Iterable[int]) -> dict[int, int]: ...
    def teach_node_count(self) -> int: ...
    def examples_without_teach(self, limit: int) -> list[Row]: ...
    def all_examples(self) -> list[Row]: ...
    def node_usage(self) -> dict[int, float]: ...
    def counts(self) -> dict[str, int]: ...
    def flyhash_params(self) -> FlyHashParams: ...
    def vec_available(self) -> bool: ...
    def nearest_codes(self, active: Iterable[int], k: int) -> list[tuple[int, int]]: ...
    def nodes_without_dense(self, limit: int) -> list[Row]: ...
    def nearest_dense(self, vec: Sequence[float], k: int) -> list[tuple[int, float]]: ...
    def subgraph(
        self, seed_ids: Iterable[int], hops: int, max_nodes: int
    ) -> tuple[list[int], list[tuple[int, int, float]]]: ...
    def live_nodes(self, node_ids: Iterable[int]) -> set[int]: ...
    def log_tail(self, n: int = 10) -> list[Row]: ...
    def capture_stats(self, window: int, nodes: Iterable[int]) -> dict[str, int]: ...
    def flyvec_info(self) -> Row | None: ...
    def flyvec_tokens(self, tokens: Iterable[str]) -> dict[str, tuple[int, ...]]: ...
    def flyvec_node_vectors(self) -> tuple[list[int], np.ndarray]: ...
    def training_sentences(self, limit: int) -> list[tuple[str, str, str]]: ...
    def node_texts(self, limit: int) -> list[tuple[int, str]]: ...


def unpack_active(blob: bytes) -> tuple[int, ...]:
    return struct.unpack(f"<{len(blob) // 4}I", blob)


class SqliteRead:
    def __init__(self, db: Database) -> None:
        self._db = db

    def contract(self) -> str:
        with self._db.reader() as conn:
            return conn.execute("SELECT value FROM meta WHERE key = 'contract'").fetchone()[0]

    def get_node(self, key: str | int) -> Row | None:
        """Node by text (``str``) or id (``int``); ``bool``/other types raise TypeError."""
        if isinstance(key, bool) or not isinstance(key, (str, int)):
            raise TypeError(f"node key must be str or int, got {type(key).__name__}")
        sql = (
            "SELECT * FROM nodes WHERE id = ?"
            if isinstance(key, int)
            else "SELECT * FROM nodes WHERE text = ?"
        )
        with self._db.reader() as conn:
            row = conn.execute(sql, (key,)).fetchone()
        return dict(row) if row else None

    def neighbors(self, node_id: int, valid_only: bool = True) -> list[Row]:
        """Edges touching ``node_id``. Edges are not deduplicated: the same triple ingested twice
        yields two rows (each is a separate piece of evidence with its own source/version)."""
        sql = "SELECT * FROM edges WHERE (subj = ? OR obj = ?)"
        if valid_only:
            sql += " AND valid = 1"
        with self._db.reader() as conn:
            return [dict(r) for r in conn.execute(sql + " ORDER BY id", (node_id, node_id))]

    def edges_by_source(self, source: str) -> list[Row]:
        with self._db.reader() as conn:
            rows = conn.execute("SELECT * FROM edges WHERE source = ? ORDER BY id", (source,))
            return [dict(r) for r in rows]

    def kc_code(self, node_id: int) -> Row | None:
        with self._db.reader() as conn:
            row = conn.execute("SELECT * FROM kc_codes WHERE node_id = ?", (node_id,)).fetchone()
        if row is None:
            return None
        return {
            "node_id": row["node_id"],
            "active": unpack_active(row["active"]),
            "k": row["k"],
            "m": row["m"],
        }

    def assoc_for(self, kc_ids: Iterable[int]) -> list[Row]:
        ids: Sequence[int] = list(kc_ids)
        if not ids:
            return []
        with self._db.reader() as conn:
            rows = conn.execute(
                "SELECT kc, node_id, weight FROM assoc "
                "WHERE kc IN (SELECT value FROM json_each(?)) ORDER BY kc, node_id",
                (json.dumps(ids),),
            )
            return [dict(r) for r in rows]

    def examples_for(self, node_ids: Iterable[int]) -> dict[int, list[Row]]:
        """Stored teach examples per node (``query``, ``weight``, ``ts``), most recent first."""
        ids = list(dict.fromkeys(node_ids))
        if not ids:
            return {}
        out: dict[int, list[Row]] = {}
        with self._db.reader() as conn:
            rows = conn.execute(
                "SELECT node_id, query, weight, ts FROM examples "
                "WHERE node_id IN (SELECT value FROM json_each(?)) ORDER BY node_id, ts DESC",
                (json.dumps(ids),),
            )
            for r in rows:
                out.setdefault(r["node_id"], []).append(
                    {"query": r["query"], "weight": r["weight"], "ts": r["ts"]}
                )
        return out

    def recent_examples(self, exclude_ids: Iterable[int], limit: int) -> list[str]:
        """Queries of the ``limit`` most recent examples of nodes NOT in ``exclude_ids``."""
        with self._db.reader() as conn:
            rows = conn.execute(
                "SELECT query FROM examples "
                "WHERE node_id NOT IN (SELECT value FROM json_each(?)) ORDER BY ts DESC LIMIT ?",
                (json.dumps(list(dict.fromkeys(exclude_ids))), limit),
            )
            return [r[0] for r in rows]

    def teach_for(self, node_ids: Iterable[int]) -> dict[int, list[Row]]:
        """Every stored teach query per node (``query``, ``active`` codes, ``ts``), recent first."""
        ids = list(dict.fromkeys(node_ids))
        if not ids:
            return {}
        out: dict[int, list[Row]] = {}
        with self._db.reader() as conn:
            rows = conn.execute(
                "SELECT node_id, query, active, ts FROM teach_queries "
                "WHERE node_id IN (SELECT value FROM json_each(?)) ORDER BY node_id, ts DESC",
                (json.dumps(ids),),
            )
            for r in rows:
                out.setdefault(r["node_id"], []).append(
                    {"query": r["query"], "active": unpack_active(r["active"]), "ts": r["ts"]}
                )
        return out

    def kc_node_counts(self, kc_ids: Iterable[int]) -> dict[int, int]:
        """Document frequency of each KC over DISTINCT nodes (kcs never taught are absent)."""
        ids = sorted(set(kc_ids))
        if not ids:
            return {}
        with self._db.reader() as conn:
            rows = conn.execute(
                "SELECT kc, COUNT(*) FROM node_kcs "
                "WHERE kc IN (SELECT value FROM json_each(?)) GROUP BY kc",
                (json.dumps(ids),),
            )
            return {r[0]: r[1] for r in rows}

    def teach_node_count(self) -> int:
        """Number of distinct nodes that have at least one stored teach query."""
        with self._db.reader() as conn:
            return conn.execute("SELECT COUNT(DISTINCT node_id) FROM teach_queries").fetchone()[0]

    def examples_without_teach(self, limit: int) -> list[Row]:
        """Examples (``node_id``, ``query``, ``ts``) without a ``teach_queries`` row (backfill)."""
        with self._db.reader() as conn:
            rows = conn.execute(
                "SELECT e.node_id, e.query, e.ts FROM examples e LEFT JOIN teach_queries t "
                "ON t.node_id = e.node_id AND t.query = e.query "
                "WHERE t.query IS NULL ORDER BY e.node_id, e.query LIMIT ?",
                (limit,),
            )
            return [dict(r) for r in rows]

    def all_examples(self) -> list[Row]:
        """Every stored example (``node_id``, ``query``, ``weight``), deterministic order."""
        with self._db.reader() as conn:
            rows = conn.execute(
                "SELECT node_id, query, weight FROM examples ORDER BY node_id, query"
            )
            return [dict(r) for r in rows]

    def node_usage(self) -> dict[int, float]:
        """Summed ``edge_usage`` of the valid edges touching each node (both endpoints)."""
        with self._db.reader() as conn:
            rows = conn.execute(
                "SELECT n, SUM(u) FROM ("
                "SELECT e.subj AS n, u.usage AS u FROM edge_usage u "
                "JOIN edges e ON e.id = u.edge_id WHERE e.valid = 1 "
                "UNION ALL SELECT e.obj, u.usage FROM edge_usage u "
                "JOIN edges e ON e.id = u.edge_id WHERE e.valid = 1) GROUP BY n"
            )
            return {r[0]: float(r[1]) for r in rows}

    def counts(self) -> dict[str, int]:
        with self._db.reader() as conn:
            return {name: conn.execute(sql).fetchone()[0] for name, sql in _COUNT_SQL.items()}

    def flyhash_params(self) -> FlyHashParams:
        return self._db.flyhash

    def vec_available(self) -> bool:
        return self._db.vec_available

    def nearest_codes(self, active: Iterable[int], k: int) -> list[tuple[int, int]]:
        """Top-``k`` nodes by code overlap with ``active``: ``[(node_id, overlap)]``.

        Ordered by overlap descending, ties by node id ascending. Uses sqlite-vec Hamming KNN when
        the extension is loaded (every code has exactly ``flyhash.k`` bits, so
        ``overlap = (n_query + k - hamming) / 2``), otherwise an exact Python scan of ``kc_codes``.
        Both paths return identical results.
        """
        query = sorted(set(active))
        if k < 1 or not query:
            return []
        p = self._db.flyhash
        if any(not 0 <= kc < p.m for kc in query):
            raise ValueError(f"active indices must be in [0, {p.m})")
        if self._db.vec_available:
            return self._nearest_vec(query, k, p)
        qset = set(query)
        with self._db.reader() as conn:
            rows = conn.execute("SELECT node_id, active FROM kc_codes WHERE m = ?", (p.m,))
            scored = [
                (r["node_id"], len(qset.intersection(unpack_active(r["active"]))))
                for r in rows.fetchall()
            ]
        scored.sort(key=lambda t: (-t[1], t[0]))
        return scored[:k]

    def nodes_without_dense(self, limit: int) -> list[Row]:
        """Up to ``limit`` nodes (``id``, ``text``) lacking a dense vector, by id."""
        with self._db.reader() as conn:
            rows = conn.execute(
                "SELECT n.id, n.text FROM nodes n LEFT JOIN dense_vecs v ON v.node_id = n.id "
                "WHERE v.node_id IS NULL ORDER BY n.id LIMIT ?",
                (limit,),
            )
            return [dict(r) for r in rows]

    def nearest_dense(self, vec: Sequence[float], k: int) -> list[tuple[int, float]]:
        """Top-``k`` nodes by exact cosine similarity: ``[(node_id, cosine)]``.

        Cosine descending, ties by node id ascending. sqlite-vec cosine KNN when the extension is
        loaded, otherwise an exact numpy scan of ``dense_vecs``; both rank identically.
        """
        if k < 1:
            return []
        d = self._db.flyhash.d
        query = np.frombuffer(vec_to_blob(vec, d), dtype="<f4")
        if self._db.vec_available:
            return self._nearest_dense_vec(query, k)
        with self._db.reader() as conn:
            rows = conn.execute("SELECT node_id, vec FROM dense_vecs ORDER BY node_id").fetchall()
        if not rows:
            return []
        ids = np.fromiter((r[0] for r in rows), dtype=np.int64, count=len(rows))
        matrix = np.vstack([np.frombuffer(r[1], dtype="<f4") for r in rows])
        sims = matrix @ query
        scored = [(int(ids[i]), round(float(sims[i]), COSINE_DECIMALS)) for i in range(len(ids))]
        scored.sort(key=lambda t: (-t[1], t[0]))
        return scored[:k]

    def _nearest_dense_vec(self, query: np.ndarray, k: int) -> list[tuple[int, float]]:
        blob = query.astype("<f4").tobytes()
        fetch = min(k, MAX_VEC_K)
        while True:
            with self._db.reader() as conn:
                rows = conn.execute(
                    "SELECT rowid, distance FROM dense_index WHERE embedding MATCH ? AND k = ?",
                    (blob, fetch),
                ).fetchall()
            # Grow the window while distance ties could straddle the cut (exact, deterministic).
            tied = len(rows) == fetch and len(rows) >= k and rows[-1][1] == rows[k - 1][1]
            if not tied or fetch >= MAX_VEC_K:
                break
            fetch = min(fetch * 2, MAX_VEC_K)
        scored = [(r[0], round(1.0 - float(r[1]), COSINE_DECIMALS)) for r in rows]
        scored.sort(key=lambda t: (-t[1], t[0]))  # same total order as the numpy fallback
        return scored[:k]

    def _nearest_vec(self, query: list[int], k: int, p: FlyHashParams) -> list[tuple[int, int]]:
        bits = active_to_bits(query, p.m)
        fetch = min(k, MAX_VEC_K)
        while True:
            with self._db.reader() as conn:
                rows = conn.execute(
                    "SELECT rowid, distance FROM kc_index WHERE code MATCH vec_bit(?) AND k = ?",
                    (bits, fetch),
                ).fetchall()
            # Grow the window while distance ties could straddle the cut (exact, deterministic).
            tied = len(rows) == fetch and len(rows) >= k and rows[-1][1] == rows[k - 1][1]
            if not tied or fetch >= MAX_VEC_K:
                break
            fetch = min(fetch * 2, MAX_VEC_K)
        ordered = sorted((r[1], r[0]) for r in rows)
        return [(node_id, (len(query) + p.k - int(dist)) // 2) for dist, node_id in ordered[:k]]

    def subgraph(
        self, seed_ids: Iterable[int], hops: int, max_nodes: int
    ) -> tuple[list[int], list[tuple[int, int, float]]]:
        """Valid edges within ``hops`` of the seeds, capped at ``max_nodes`` nodes.

        Returns ``(node_ids, [(subj, obj, weight)])``; ``node_ids`` keeps seeds first then
        discovered nodes by (hop, id), so the cap is deterministic. Edges are those valid ones with
        both endpoints kept (not deduplicated).
        """
        if hops < 0 or max_nodes < 1:
            raise ValueError("hops must be >= 0 and max_nodes >= 1")
        ordered = list(dict.fromkeys(seed_ids))[:max_nodes]
        known = set(ordered)
        frontier = list(ordered)
        with self._db.reader() as conn:
            for _ in range(hops):
                if not frontier or len(ordered) >= max_nodes:
                    break
                rows = conn.execute(
                    "SELECT subj, obj FROM edges WHERE valid = 1 AND "
                    "(subj IN (SELECT value FROM json_each(?)) "
                    "OR obj IN (SELECT value FROM json_each(?)))",
                    (json.dumps(frontier), json.dumps(frontier)),
                ).fetchall()
                found = sorted({n for r in rows for n in (r[0], r[1])} - known)
                found = found[: max_nodes - len(ordered)]
                ordered.extend(found)
                known.update(found)
                frontier = found
            if not ordered:
                return [], []
            blob = json.dumps(ordered)
            edge_rows = conn.execute(
                "SELECT subj, obj, weight FROM edges WHERE valid = 1 AND "
                "subj IN (SELECT value FROM json_each(?)) "
                "AND obj IN (SELECT value FROM json_each(?)) ORDER BY id",
                (blob, blob),
            ).fetchall()
        return ordered, [(r[0], r[1], r[2]) for r in edge_rows]

    def live_nodes(self, node_ids: Iterable[int]) -> set[int]:
        """Subset of ``node_ids`` with at least one valid incident edge (rows stay for audit)."""
        ids = list(dict.fromkeys(node_ids))
        if not ids:
            return set()
        with self._db.reader() as conn:
            rows = conn.execute(
                "SELECT value FROM json_each(?) WHERE EXISTS (SELECT 1 FROM edges "
                "WHERE valid = 1 AND (subj = value OR obj = value))",
                (json.dumps(ids),),
            )
            return {r[0] for r in rows}

    def capture_stats(self, window: int, nodes: Iterable[int]) -> dict[str, int]:
        """Associations a capture pass exempts: tagged in the window, confirmed, and either."""
        confirmed = json.dumps(sorted(set(nodes)))
        tagged_sql, tagged_args = protected_clause(window, [])
        any_sql, any_args = protected_clause(window, list(json.loads(confirmed)))
        sql = {
            "assoc": ("1", []),
            "tagged": (tagged_sql, tagged_args),
            "confirmed": ("assoc.node_id IN (SELECT value FROM json_each(?))", [confirmed]),
            "captured": (any_sql, any_args),
        }
        with self._db.reader() as conn:
            return {
                key: conn.execute(f"SELECT COUNT(*) FROM assoc WHERE {clause}", args).fetchone()[0]
                for key, (clause, args) in sql.items()
            }

    def flyvec_info(self) -> Row | None:
        """Parameters of the trained FlyVec model (``None`` when never trained)."""
        with self._db.reader() as conn:
            row = conn.execute("SELECT value FROM meta WHERE key = 'flyvec_info'").fetchone()
        return json.loads(row[0]) if row else None

    def flyvec_tokens(self, tokens: Iterable[str]) -> dict[str, tuple[int, ...]]:
        names = sorted(set(tokens))
        if not names:
            return {}
        with self._db.reader() as conn:
            rows = conn.execute(
                "SELECT token, kcs FROM flyvec_tokens "
                "WHERE token IN (SELECT value FROM json_each(?))",
                (json.dumps(names),),
            )
            return {r[0]: struct.unpack(f"<{len(r[1]) // 2}H", r[1]) for r in rows}

    def flyvec_node_vectors(self) -> tuple[list[int], np.ndarray]:
        with self._db.reader() as conn:
            rows = conn.execute("SELECT node_id, vec FROM flyvec_nodes ORDER BY node_id").fetchall()
        if not rows:
            return [], np.zeros((0, 0), dtype=np.float32)
        return [r[0] for r in rows], np.vstack([np.frombuffer(r[1], dtype="<f4") for r in rows])

    def training_sentences(self, limit: int) -> list[tuple[str, str, str]]:
        """Valid edges ``(subject text, relation, object text)``, oldest first, <= ``limit``."""
        with self._db.reader() as conn:
            rows = conn.execute(
                "SELECT s.text, e.rel, o.text FROM edges e JOIN nodes s ON s.id = e.subj "
                "JOIN nodes o ON o.id = e.obj WHERE e.valid = 1 ORDER BY e.id LIMIT ?",
                (limit,),
            )
            return [(r[0], r[1], r[2]) for r in rows]

    def node_texts(self, limit: int) -> list[tuple[int, str]]:
        with self._db.reader() as conn:
            rows = conn.execute("SELECT id, text FROM nodes ORDER BY id LIMIT ?", (limit,))
            return [(r[0], r[1]) for r in rows]

    def log_tail(self, n: int = 10) -> list[Row]:
        with self._db.reader() as conn:
            rows = conn.execute("SELECT * FROM log ORDER BY id DESC LIMIT ?", (n,)).fetchall()
        out = []
        for r in reversed(rows):
            d = dict(r)
            d["payload"] = json.loads(d["payload"])
            out.append(d)
        return out
