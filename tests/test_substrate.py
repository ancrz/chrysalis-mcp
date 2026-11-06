import ast
import inspect
import sqlite3
from pathlib import Path

import pytest

import chrysalis_mcp.substrate as substrate_pkg
from chrysalis_mcp.substrate import (
    SUBSTRATE_CONTRACT,
    FlyHashParams,
    SubstrateContractError,
    open_substrate,
)
from chrysalis_mcp.substrate import write as write_mod
from chrysalis_mcp.substrate.read import SqliteRead
from chrysalis_mcp.substrate.write import SqliteWrite, SubstrateWrite

SMALL = FlyHashParams(d=8, m=16, k=3, c=2, seed=1)
WRITE_METHODS = {n for n in vars(SubstrateWrite) if not n.startswith("_")}


@pytest.fixture
def sub():
    return open_substrate(":memory:", SMALL)


def _schema(path: Path) -> list[tuple]:
    conn = sqlite3.connect(path)
    try:
        return conn.execute("SELECT type, name FROM sqlite_master ORDER BY 1, 2").fetchall()
    finally:
        conn.close()


def test_migration_idempotent(tmp_path):
    db = tmp_path / "s.db"
    r1, _ = open_substrate(db)
    first = _schema(db)
    r2, _ = open_substrate(db)
    assert _schema(db) == first
    assert r1.contract() == r2.contract() == SUBSTRATE_CONTRACT == "v4"


def test_newer_contract_rejected(tmp_path):
    db = tmp_path / "s.db"
    open_substrate(db)
    conn = sqlite3.connect(db)
    conn.execute("UPDATE meta SET value = 'v5' WHERE key = 'contract'")
    conn.commit()
    conn.close()
    with pytest.raises(SubstrateContractError):
        open_substrate(db)


def test_read_has_no_write_methods():
    assert WRITE_METHODS
    assert not {n for n in dir(SqliteRead) if not n.startswith("_")} & WRITE_METHODS
    assert not any(hasattr(SqliteRead, n) for n in WRITE_METHODS)


def test_each_write_logs_one_row(sub):
    r, w = sub
    n = w.upsert_node("a")
    assert r.counts()["log"] == 1
    w.add_triple("a", "likes", "b", source="s")
    w.put_kc_code(n, [1, 5, 9], k=3, m=16)
    w.bump_assoc([1, 5], n, 0.5)
    w.bump_assoc([1], n, 0.25)
    w.invalidate("s")
    tail = r.log_tail(10)
    assert [e["op"] for e in tail] == [
        "upsert_node",
        "add_triple",
        "put_kc_code",
        "bump_assoc",
        "bump_assoc",
        "invalidate",
    ]
    assert tail[-1]["payload"]["n"] == 1
    assert r.kc_code(n) == {"node_id": n, "active": (1, 5, 9), "k": 3, "m": 16}
    assert {(a["kc"], a["weight"]) for a in r.assoc_for([1, 5])} == {(1, 0.75), (5, 0.5)}
    assert r.assoc_for([]) == []


def test_failure_rolls_back_data_and_log(sub, monkeypatch):
    r, w = sub

    def boom(*_a, **_k):
        raise RuntimeError("forced")

    monkeypatch.setattr(write_mod, "_log", boom)
    with pytest.raises(RuntimeError):
        w.add_triple("a", "r", "b")
    monkeypatch.undo()
    assert r.counts()["nodes"] == 0
    assert r.counts()["edges"] == 0
    assert r.counts()["log"] == 0
    w.upsert_node("ok")  # connection still usable after rollback
    assert r.counts()["log"] == 1


def test_invalidate_hides_but_keeps(sub):
    r, w = sub
    w.add_triple("a", "r", "b", source="doc1", version=1)
    w.add_triple("a", "r", "c", source="doc1", version=2)
    a = r.get_node("a")["id"]
    assert len(r.neighbors(a)) == 2
    assert w.invalidate("doc1", version=1) == 1
    assert len(r.neighbors(a)) == 1
    assert len(r.neighbors(a, valid_only=False)) == 2
    assert len(r.edges_by_source("doc1")) == 2
    assert r.log_tail(1)[0]["op"] == "invalidate"


def test_node_dedupe_and_fk(sub):
    r, w = sub
    w.add_triple("a", "r", "b")
    w.add_triple("a", "r2", "c")
    assert r.counts()["nodes"] == 3
    assert r.get_node("a") == r.get_node(r.get_node("a")["id"])
    assert r.get_node("zzz") is None
    with pytest.raises(sqlite3.IntegrityError):
        w.bump_assoc([1], 9999, 1.0)
    with pytest.raises(sqlite3.IntegrityError):
        w.put_kc_code(9999, [1, 2, 3], 3, 16)
    assert r.counts()["assoc"] == 0
    assert r.counts()["log"] == 2  # failed writes left no log entry


def test_import_graph_isolated():
    forbidden = {"core", "domains", "algorithms", "server"}
    root = Path(inspect.getfile(substrate_pkg)).parent
    files = list(root.glob("*.py"))
    assert len(files) >= 4
    for f in files:
        for node in ast.walk(ast.parse(f.read_text())):
            if isinstance(node, ast.Import):
                mods = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                if node.level > 0:
                    assert node.level == 1, f"{f.name}: relative import escapes package"
                    continue
                mods = [node.module or ""]
            else:
                continue
            for m in mods:
                parts = m.split(".")
                assert not (
                    parts[0] == "chrysalis_mcp" and len(parts) > 1 and parts[1] in forbidden
                ), f"{f.name} imports {m}"
                assert not (parts[0] == "chrysalis_mcp" and len(parts) == 1), f.name


def test_memory_and_file_db(tmp_path):
    r, w = open_substrate(":memory:")
    w.upsert_node("x")
    assert r.counts()["nodes"] == 1
    db = tmp_path / "sub" / "f.db"
    r, w = open_substrate(db)
    w.upsert_node("x")
    conn = sqlite3.connect(db)
    assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    conn.close()
    assert isinstance(w, SqliteWrite)


def test_env_and_xdg_path(tmp_path, monkeypatch):
    monkeypatch.setenv("CHRYSALIS_DB", str(tmp_path / "env.db"))
    open_substrate()
    assert (tmp_path / "env.db").exists()
    monkeypatch.delenv("CHRYSALIS_DB")
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg"))
    open_substrate()
    assert (tmp_path / "xdg" / "chrysalis" / "substrate.db").exists()


def test_get_node_rejects_bool_and_other_types(sub):
    r, w = sub
    w.upsert_node("a")
    for bad in (True, False, 1.5, None, b"a"):
        with pytest.raises(TypeError):
            r.get_node(bad)


def test_edges_are_not_deduplicated(sub):
    r, w = sub
    w.add_triple("a", "r", "b", source="s")
    w.add_triple("a", "r", "b", source="s")
    assert len(r.neighbors(r.get_node("a")["id"])) == 2
    assert "not deduplicated" in (SqliteRead.neighbors.__doc__ or "")


@pytest.mark.parametrize("target", [":memory:", "file"])
def test_read_connection_cannot_write(tmp_path, target):
    r, w = open_substrate(":memory:" if target == ":memory:" else tmp_path / "q.db", SMALL)
    w.upsert_node("a")
    with r._db.reader() as conn:
        for stmt in (
            "INSERT INTO nodes (text) VALUES ('x')",
            "UPDATE nodes SET text = 'y'",
            "DELETE FROM nodes",
            "DROP TABLE log",
        ):
            with pytest.raises(sqlite3.DatabaseError):  # authorizer denial / query_only
                conn.execute(stmt)
    assert r.counts()["nodes"] == 1
    assert r._db._wconn is not r._db._rconn


def test_read_sees_committed_writes_on_second_connection(tmp_path):
    for target in (":memory:", tmp_path / "rw.db"):
        r, w = open_substrate(target, SMALL)
        for i in range(5):
            w.upsert_node(f"n{i}")
            assert r.counts()["nodes"] == i + 1


def test_connections_closed_when_migration_fails(tmp_path, monkeypatch):
    db = tmp_path / "bad.db"
    open_substrate(db)
    conn = sqlite3.connect(db)
    conn.execute("UPDATE meta SET value = 'v9' WHERE key = 'contract'")
    conn.commit()
    conn.close()
    closed = []
    real = sqlite3.Connection.close

    class Spy(sqlite3.Connection):
        def close(self):
            closed.append(1)
            real(self)

    real_connect = sqlite3.connect
    monkeypatch.setattr(
        sqlite3, "connect", lambda *a, **k: real_connect(*a, **{**k, "factory": Spy})
    )
    with pytest.raises(SubstrateContractError):
        open_substrate(db)
    assert closed, "write connection leaked after failed migration"
