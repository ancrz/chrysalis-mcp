import random
import sqlite3

import pytest

from chrysalis_mcp.substrate import FlyHashParams, SubstrateContractError, open_substrate
from chrysalis_mcp.substrate import store as store_mod
from chrysalis_mcp.substrate import write as write_mod
from chrysalis_mcp.substrate.read import unpack_active

P = FlyHashParams(d=16, m=128, k=8, c=3, seed=11)


def _codes(n: int, seed: int = 0, m: int = P.m, k: int = P.k) -> list[list[int]]:
    rng = random.Random(seed)
    base = rng.sample(range(m), k)
    out = []
    for _ in range(n):  # clustered codes -> many distance ties, a harsh case for ordering
        c = set(base)
        for _ in range(rng.randint(0, k)):
            c.discard(rng.choice(sorted(c)))
        while len(c) < k:
            c.add(rng.randrange(m))
        out.append(sorted(c))
    return out


def _fill(w, codes):
    ids = [w.upsert_node(f"n{i}") for i in range(len(codes))]
    for nid, c in zip(ids, codes, strict=True):
        w.put_kc_code(nid, c, P.k, P.m)
    return ids


def _expected(codes, ids, q, k):
    scored = sorted(((-len(set(q) & set(c)), i) for i, c in zip(ids, codes, strict=True)))
    return [(i, -s) for s, i in scored[:k]]


def test_vec_available_in_this_env():
    r, _ = open_substrate(":memory:", P)
    assert r.vec_available()


@pytest.mark.parametrize("seed", range(5))
def test_hamming_order_equals_overlap_order(seed):
    r, w = open_substrate(":memory:", P)
    codes = _codes(60, seed)
    ids = _fill(w, codes)
    for q in _codes(5, seed + 100) + codes[:3]:
        for k in (1, 5, 17, 60):
            assert r.nearest_codes(q, k) == _expected(codes, ids, q, k)


def test_fallback_matches_vec_path(monkeypatch, tmp_path):
    codes = _codes(40, 7)
    db = tmp_path / "f.db"
    r, w = open_substrate(db, P)
    ids = _fill(w, codes)
    qs = _codes(4, 55)
    vec_results = [r.nearest_codes(q, 10) for q in qs]

    monkeypatch.setattr(store_mod, "_load_vec", lambda conn: False)
    r2, w2 = open_substrate(db, P)
    assert not r2.vec_available()
    assert [r2.nearest_codes(q, 10) for q in qs] == vec_results
    assert vec_results[0] == _expected(codes, ids, qs[0], 10)
    # write on a host without the extension must not touch kc_index, and flags it stale
    extra = w2.upsert_node("late")
    w2.put_kc_code(extra, qs[0], P.k, P.m)
    raw = sqlite3.connect(db)
    assert raw.execute("SELECT value FROM meta WHERE key='kc_index_stale'").fetchone()
    raw.close()
    monkeypatch.undo()
    r3, _ = open_substrate(db, P)  # extension back: index rebuilt, includes the late node
    assert r3.vec_available()
    assert extra in {n for n, _ in r3.nearest_codes(qs[0], 50)}
    raw = sqlite3.connect(db)
    assert raw.execute("SELECT value FROM meta WHERE key='kc_index_stale'").fetchone() is None
    raw.close()


def test_no_extension_at_all_ingests_and_queries(monkeypatch):
    monkeypatch.setattr(store_mod, "_load_vec", lambda conn: False)
    r, w = open_substrate(":memory:", P)
    codes = _codes(10, 2)
    ids = _fill(w, codes)
    assert not r.vec_available()
    assert r.nearest_codes(codes[0], 3) == _expected(codes, ids, codes[0], 3)
    with r._db.reader() as conn:
        assert not conn.execute("SELECT 1 FROM sqlite_master WHERE name='kc_index'").fetchone()


def _make_v1(path, n=3):
    conn = sqlite3.connect(path)
    conn.executescript(store_mod._SCHEMA_V1)
    conn.execute("INSERT INTO meta VALUES ('contract', 'v1')")
    for i in range(n):
        conn.execute("INSERT INTO nodes (text) VALUES (?)", (f"old{i}",))
    conn.execute("INSERT INTO edges (subj, rel, obj) VALUES (1, 'r', 2)")
    import struct

    for i, c in enumerate(_codes(n, 1), start=1):
        conn.execute(
            "INSERT INTO kc_codes (node_id, active, k, m) VALUES (?, ?, ?, ?)",
            (i, struct.pack(f"<{len(c)}I", *c), P.k, P.m),
        )
    conn.commit()
    conn.close()


def test_migration_v1_to_v2_preserves_data_and_is_idempotent(tmp_path):
    db = tmp_path / "v1.db"
    _make_v1(db)
    r, _ = open_substrate(db, P)
    assert r.contract() == "v4"
    assert r.counts()["nodes"] == 3 and r.counts()["edges"] == 1 and r.counts()["kc_codes"] == 3
    assert r.flyhash_params() == P
    assert r.get_node("old1")["id"] == 2
    assert unpack_active(
        sqlite3.connect(db).execute("SELECT active FROM kc_codes WHERE node_id=1").fetchone()[0]
    ) == tuple(_codes(3, 1)[0])
    assert len(r.nearest_codes(_codes(1, 1)[0], 3)) == 3  # index backfilled
    first = (
        sqlite3.connect(db).execute("SELECT type, name FROM sqlite_master ORDER BY 1,2").fetchall()
    )
    r2, _ = open_substrate(db)  # params=None -> stored ones
    assert r2.contract() == "v4" and r2.flyhash_params() == P
    assert (
        sqlite3.connect(db).execute("SELECT type, name FROM sqlite_master ORDER BY 1,2").fetchall()
        == first
    )
    assert r2.counts()["log"] == 0  # migration never writes log rows


def test_mismatched_flyhash_params_rejected(tmp_path):
    db = tmp_path / "p.db"
    open_substrate(db, P)
    for other in (
        FlyHashParams(d=16, m=128, k=8, c=3, seed=12),
        FlyHashParams(d=32, m=128, k=8, c=3, seed=11),
    ):
        with pytest.raises(SubstrateContractError):
            open_substrate(db, other)
    assert open_substrate(db)[0].flyhash_params() == P


def test_put_kc_code_validates_against_params():
    r, w = open_substrate(":memory:", P)
    n = w.upsert_node("a")
    good = list(range(P.k))
    for bad in (
        (good[:-1], P.k - 1, P.m),
        (good, P.k, P.m * 2),
        ([0, 0, *good[2:]], P.k, P.m),
        ([-1, *good[1:]], P.k, P.m),
        ([P.m, *good[1:]], P.k, P.m),
    ):
        with pytest.raises(ValueError):
            w.put_kc_code(n, *bad)
    assert r.counts()["kc_codes"] == 0


def test_put_kc_code_atomic_on_failure(monkeypatch):
    r, w = open_substrate(":memory:", P)
    n = w.upsert_node("a")
    base_log = r.counts()["log"]

    def boom(*_a, **_k):
        raise RuntimeError("forced")

    monkeypatch.setattr(write_mod, "_log", boom)
    with pytest.raises(RuntimeError):
        w.put_kc_code(n, list(range(P.k)), P.k, P.m)
    monkeypatch.undo()
    assert r.counts()["kc_codes"] == 0 and r.counts()["log"] == base_log
    assert r.nearest_codes(list(range(P.k)), 5) == []  # kc_index row rolled back too


def test_nearest_edge_cases():
    r, w = open_substrate(":memory:", P)
    assert r.nearest_codes([1, 2], 3) == []
    assert r.nearest_codes([], 3) == []
    with pytest.raises(ValueError):
        r.nearest_codes([P.m], 1)
    _fill(w, _codes(3, 4))
    assert r.nearest_codes([1, 2], 0) == []
    assert len(r.nearest_codes([1, 2], 100)) == 3


# ---------------------------------------------------------------- hardening / D4 --------------


def _craft(path, **overrides):
    """A v2 database whose meta carries hostile FlyHash parameters."""
    open_substrate(path, P)
    conn = sqlite3.connect(path)
    for key, value in overrides.items():
        conn.execute("UPDATE meta SET value = ? WHERE key = ?", (str(value), f"flyhash_{key}"))
    conn.commit()
    conn.close()


@pytest.mark.parametrize(
    "overrides",
    [
        {"m": 10**9},
        {"m": 65536 + 8},
        {"m": 12},
        {"m": 0},
        {"d": 8192 * 100},
        {"d": 0},
        {"d": 8192, "m": 10**9},
        {"k": 10**6},
        {"c": 10**6},
        {"m": "abc"},
    ],
)
def test_crafted_flyhash_params_rejected(tmp_path, overrides):
    db = tmp_path / "evil.db"
    _craft(db, **overrides)
    with pytest.raises(SubstrateContractError):
        open_substrate(db)


def test_flyhash_param_bounds_on_construction():
    for bad in (
        dict(m=65544),
        dict(m=4),
        dict(m=100),
        dict(d=65537),
        dict(d=0),
        dict(k=P.m + 8),
        dict(k=0),
        dict(c=P.d + 1),
        dict(c=0),
        dict(seed=1.5),
        dict(m=True),
    ):
        with pytest.raises(ValueError):
            FlyHashParams(**{**vars(P), **bad})
    FlyHashParams(d=65536, m=65536, k=65536, c=65536)  # upper edge is legal


def _tri(n=4):
    return [
        {"s": f"a{i}", "r": "r", "o": f"b{i}", "weight": 1.0 + i, "source": "src"} for i in range(n)
    ] + [{"s": "a0", "r": "r2", "o": "b1"}]


def _det_code(text):
    rng = random.Random(text)
    return sorted(rng.sample(range(P.m), P.k))


def _snapshot(r):
    with r._db.reader() as conn:
        index = [x[0] for x in conn.execute("SELECT rowid FROM kc_index ORDER BY rowid")]
        edges = [tuple(x) for x in conn.execute("SELECT * FROM edges ORDER BY id")]
    return r.counts(), index, edges


def test_ingest_batch_single_log_row_and_result():
    r, w = open_substrate(":memory:", P)
    w.upsert_node("a0")
    w.put_kc_code(1, _det_code("a0"), P.k, P.m)
    before = r.counts()["log"]
    res = w.ingest_batch(_tri(), _det_code, default_source="dflt")
    assert r.counts()["log"] == before + 1 and r.log_tail(1)[0]["op"] == "ingest_batch"
    assert res.preexisting == {"a0"} and res.edges == 5
    assert "a0" not in res.coded and set(res.coded) == set(res.nodes) - {"a0"}
    assert r.counts()["kc_codes"] == len(res.nodes) == 8
    assert [e["source"] for e in r.edges_by_source("dflt")] == ["dflt"]  # only the sourceless one
    assert [n for n, _ in r.nearest_codes(_det_code("b2"), 1)] == [r.get_node("b2")["id"]]


class _FailingConn:
    """Delegating proxy: ``execute`` call number ``fail_at`` raises (``None`` = only count)."""

    def __init__(self, real, fail_at=None):
        self._real, self.fail_at, self.calls = real, fail_at, 0

    def execute(self, sql, *args):
        self.calls += 1
        if self.calls == self.fail_at:
            raise sqlite3.OperationalError(
                f"injected failure at statement #{self.calls}: {sql[:40]}"
            )
        return self._real.execute(sql, *args)

    def __getattr__(self, name):
        return getattr(self._real, name)


def test_ingest_batch_atomic_at_every_write_and_retry_is_exact():
    r0, w0 = open_substrate(":memory:", P)
    w0.upsert_node("a0")  # same setup as the sweep below
    counter = _FailingConn(w0._db._wconn)
    w0._db._wconn = counter
    w0.ingest_batch(_tri(), _det_code)
    total = counter.calls
    assert total > 40  # BEGIN + nodes + edges + kc_codes + kc_index (DELETE/INSERT) + log + COMMIT

    for n in range(1, total + 1):
        r, w = open_substrate(":memory:", P)
        w.upsert_node("a0")  # a pre-existing, uncoded node
        base = _snapshot(r)
        real = w._db._wconn
        proxy = _FailingConn(real, fail_at=n)
        w._db._wconn = proxy
        try:
            with pytest.raises(sqlite3.OperationalError, match="injected"):
                w.ingest_batch(_tri(), _det_code)
        finally:
            w._db._wconn = real
        assert _snapshot(r) == base, f"state changed after failure at statement #{n}"
        w.ingest_batch(_tri(), _det_code)  # retry on the same database
        with r._db.reader() as conn:
            edges = [tuple(e) for e in conn.execute("SELECT subj, rel, obj FROM edges")]
        assert len(edges) == len(_tri()) and len(set(edges)) == len(edges)
        assert r.counts()["kc_codes"] == 8


def test_ingest_batch_failure_inside_code_validation_rolls_back():
    r, w = open_substrate(":memory:", P)
    base = _snapshot(r)
    with pytest.raises(ValueError):
        w.ingest_batch(_tri(), lambda text: [0, 1, 2] if text == "b3" else _det_code(text))
    assert _snapshot(r) == base


def test_read_authorizer_denies_escape_hatches(tmp_path):
    r, w = open_substrate(tmp_path / "auth.db", P)
    w.upsert_node("a")
    other = tmp_path / "other.db"
    sqlite3.connect(other).close()
    with r._db.reader() as conn:
        for stmt in (
            f"ATTACH DATABASE '{other}' AS x",
            "PRAGMA query_only = OFF",
            "PRAGMA writable_schema = ON",
            "PRAGMA journal_mode = DELETE",
            "CREATE TABLE t (a)",
            "CREATE TEMP TABLE t (a)",
            "INSERT INTO nodes (text) VALUES ('x')",
            "DROP TABLE kc_index",
        ):
            with pytest.raises(sqlite3.DatabaseError):
                conn.execute(stmt)
        # still read-only after the attempts, and reads (incl. vec MATCH) keep working
        assert conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0] == 1
    assert r.nearest_codes(list(range(P.k)), 1) == []


def test_ingest_batch_log_reconstructs_the_write_set():
    r, w = open_substrate(":memory:", P)
    pre = w.upsert_node("a0")  # pre-existing, uncoded: gets a code but is not "created"
    w.add_triple("old", "r", "b1", source="older")  # b1 pre-existing too
    edges_before = {e["id"] for e in r.neighbors(r.get_node("old")["id"])}
    triples = _tri() + [{"s": "a0", "r": "r3", "o": "b1", "source": "other", "version": 3}]
    res = w.ingest_batch(triples, _det_code, default_source="dflt")
    entry = r.log_tail(1)[0]
    assert entry["op"] == "ingest_batch"
    pl = entry["payload"]
    assert (pl["nodes"], pl["edges"], pl["codes"], pl["source"]) == (
        len(pl["node_ids"]),
        len(triples),
        len(pl["coded_ids"]),
        "dflt",
    )
    with r._db.reader() as conn:
        all_edges = {
            x[0]: tuple(x[1:]) for x in conn.execute("SELECT id, source, version FROM edges")
        }
        node_text = {x[0]: x[1] for x in conn.execute("SELECT id, text FROM nodes")}
        coded_ids = {x[0] for x in conn.execute("SELECT node_id FROM kc_codes")}
    # exactly the nodes created by this call (not a0, not b1, not "old")
    created = {n for n in res.nodes if n not in res.preexisting}
    assert {node_text[i] for i in pl["node_ids"]} == created
    assert pre not in pl["node_ids"] and r.get_node("b1")["id"] not in pl["node_ids"]
    # exactly the edges written here; edges of pre-existing nodes included, older edges excluded
    assert set(pl["edge_ids"]) == set(all_edges) - edges_before
    assert len(pl["edge_ids"]) == len(set(pl["edge_ids"])) == len(triples)
    # coded ids: exist in kc_codes; include pre-existing uncoded a0; exclude already-coded
    assert set(pl["coded_ids"]) <= coded_ids and pre in pl["coded_ids"]
    assert {node_text[i] for i in pl["coded_ids"]} == set(res.coded)
    # distinct effective source+version pairs
    assert (
        {(x["source"], x["version"]) for x in pl["sources"]}
        == {(all_edges[i][0], all_edges[i][1]) for i in pl["edge_ids"]}
        == {("src", 1), ("dflt", 1), ("other", 3)}
    )
    assert len(pl["sources"]) == 3
