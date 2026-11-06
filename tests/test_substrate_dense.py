import random
import sqlite3

import numpy as np
import pytest

from chrysalis_mcp.substrate import FlyHashParams, open_substrate
from chrysalis_mcp.substrate import store as store_mod
from chrysalis_mcp.substrate.read import COSINE_DECIMALS

P = FlyHashParams(d=16, m=128, k=8, c=3, seed=11)


def _vectors(n, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, P.d))


def _exact(vecs, ids, q, k):
    unit = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
    sims = unit @ (q / np.linalg.norm(q))
    scored = sorted(
        ((-round(float(s), COSINE_DECIMALS), i) for s, i in zip(sims, ids, strict=True))
    )
    return [(i, -s) for s, i in scored[:k]]


def _fill(w, vecs):
    ids = [w.upsert_node(f"n{i}") for i in range(len(vecs))]
    for nid, v in zip(ids, vecs, strict=True):
        w.put_dense(nid, v.tolist())
    return ids


@pytest.fixture(params=["vec", "numpy"])
def backend(request, monkeypatch):
    if request.param == "numpy":
        monkeypatch.setattr(store_mod, "_load_vec", lambda conn: False)
    return request.param


def test_cosine_topk_equals_numpy_exact(backend):
    r, w = open_substrate(":memory:", P)
    assert r.vec_available() == (backend == "vec")
    vecs = _vectors(60, 1)
    ids = _fill(w, vecs)
    for seed in range(5):
        q = _vectors(1, 100 + seed)[0]
        for k in (1, 5, 60, 100):
            got, want = r.nearest_dense(q.tolist(), k), _exact(vecs, ids, q, k)
            assert [n for n, _ in got] == [n for n, _ in want]
            assert [c for _, c in got] == pytest.approx([c for _, c in want], abs=1e-5)


def test_ties_break_by_node_id(backend):
    r, w = open_substrate(":memory:", P)
    v = [1.0] + [0.0] * (P.d - 1)
    ids = [w.upsert_node(f"t{i}") for i in range(6)]
    for nid in ids:
        w.put_dense(nid, v)
    assert [n for n, _ in r.nearest_dense(v, 4)] == ids[:4]


def test_put_dense_replaces_and_validates(backend):
    r, w = open_substrate(":memory:", P)
    a = w.upsert_node("a")
    w.put_dense(a, [1.0] + [0.0] * (P.d - 1))
    w.put_dense(a, [0.0, 1.0] + [0.0] * (P.d - 2))
    assert r.counts()["dense_vecs"] == 1
    assert r.nearest_dense([0.0, 1.0] + [0.0] * (P.d - 2), 1) == [(a, 1.0)]
    with pytest.raises(ValueError, match="dimension"):
        w.put_dense(a, [1.0])
    with pytest.raises(ValueError, match="non-zero"):
        w.put_dense(a, [0.0] * P.d)
    with pytest.raises(ValueError, match="unknown node"):
        w.put_dense(999, [1.0] * P.d)
    assert r.nearest_dense([1.0] * P.d, 0) == []


def test_ingest_batch_writes_dense_atomically(backend):
    r, w = open_substrate(":memory:", P)

    def code_for(text):
        return list(range(P.k))

    def dense_for(text):
        v = [0.0] * P.d
        v[len(text) % P.d] = 1.0
        return v

    w.ingest_batch([{"s": "ab", "r": "r", "o": "abc"}], code_for, dense_for=dense_for)
    assert r.counts()["dense_vecs"] == 2
    bad = [{"s": "ok", "r": "r", "o": "boom"}]

    def failing(text):
        if text == "boom":
            raise RuntimeError("x")
        return dense_for(text)

    before = r.counts()
    with pytest.raises(RuntimeError):
        w.ingest_batch(bad, code_for, dense_for=failing)
    assert r.counts() == before


def _make_v2(path, n=3):
    """A database as written by the v2 code: no dense tables."""
    conn = sqlite3.connect(path)
    conn.executescript(store_mod._SCHEMA_V1)
    conn.execute("INSERT INTO meta VALUES ('contract', 'v2')")
    for key in ("d", "m", "k", "c", "seed"):
        conn.execute("INSERT INTO meta VALUES (?, ?)", (f"flyhash_{key}", str(getattr(P, key))))
    rng = random.Random(3)
    for i in range(n):
        conn.execute("INSERT INTO nodes (text) VALUES (?)", (f"old{i}",))
    conn.execute("INSERT INTO edges (subj, rel, obj) VALUES (1, 'r', 2)")
    for i in range(1, n + 1):
        import struct

        active = sorted(rng.sample(range(P.m), P.k))
        conn.execute(
            "INSERT INTO kc_codes (node_id, active, k, m) VALUES (?, ?, ?, ?)",
            (i, struct.pack(f"<{P.k}I", *active), P.k, P.m),
        )
    conn.commit()
    conn.close()


def test_migration_v2_to_v3_preserves_data_and_is_idempotent(tmp_path):
    db = tmp_path / "v2.db"
    _make_v2(db)
    r, w = open_substrate(db)
    assert r.contract() == "v4"
    c = r.counts()
    assert (c["nodes"], c["edges"], c["kc_codes"], c["dense_vecs"]) == (3, 1, 3, 0)
    assert [x["text"] for x in r.nodes_without_dense(10)] == ["old0", "old1", "old2"]
    assert r.get_node("old1")["id"] == 2
    assert w.backfill_dense({1: [1.0] * P.d, 2: [2.0] * P.d, 3: [3.0] * P.d}) == 3
    assert r.nodes_without_dense(10) == []
    first = sqlite3.connect(db).execute("SELECT type, name FROM sqlite_master ORDER BY 1,2")
    first = first.fetchall()
    r2, _ = open_substrate(db)
    again = sqlite3.connect(db).execute("SELECT type, name FROM sqlite_master ORDER BY 1,2")
    assert again.fetchall() == first and r2.counts()["dense_vecs"] == 3
    assert len(r2.nearest_dense([1.0] * P.d, 5)) == 3  # index rebuilt/kept


def test_stale_dense_index_rebuilt_after_write_without_extension(tmp_path, monkeypatch):
    db = tmp_path / "s.db"
    r, w = open_substrate(db, P)
    a = w.upsert_node("a")
    w.put_dense(a, [1.0] + [0.0] * (P.d - 1))
    with monkeypatch.context() as m:
        m.setattr(store_mod, "_load_vec", lambda conn: False)
        r2, w2 = open_substrate(db)
        b = w2.upsert_node("b")
        w2.put_dense(b, [0.0, 1.0] + [0.0] * (P.d - 2))
        assert not r2.vec_available()
        assert r2.nearest_dense([0.0, 1.0] + [0.0] * (P.d - 2), 1)[0][0] == b
    r3, _ = open_substrate(db)
    assert r3.vec_available()
    assert r3.nearest_dense([0.0, 1.0] + [0.0] * (P.d - 2), 1)[0][0] == b
