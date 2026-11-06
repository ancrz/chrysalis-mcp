"""T7b L3: rotating verified backups, restore round-trip, fail-closed behavior."""

import json

import pytest

from chrysalis_mcp.domains import memory

CORPUS = [
    {"s": "París", "r": "capital_de", "o": "Francia", "source": "wiki"},
    {"s": "Berlín", "r": "capital_de", "o": "Alemania", "source": "wiki"},
    {"s": "mitocondria", "r": "produce", "o": "ATP", "source": "bio"},
]


@pytest.fixture(autouse=True)
def db(tmp_path):
    memory.configure(tmp_path / "mem.db")
    yield tmp_path / "mem.db"
    memory.configure(None)


def _live_dump(db):
    st = memory._get_state()
    with st.read._db.reader() as conn:
        return {
            t: sorted(tuple(r) for r in conn.execute(f"SELECT * FROM {t}"))  # noqa: S608
            for t in ("meta", "nodes", "edges", "kc_codes", "assoc")
        }


def test_forget_and_consolidate_create_slots(call, db):
    call("memory", "ingest", params={"triples": CORPUS})
    f = call("memory", "forget", params={"source": "wiki"})
    c = call("memory", "consolidate")
    assert f["backup_slot"] == 0 and c["backup_slot"] == 1
    meta = (db.parent / "backups" / "substrate.0.json").read_text()
    info = json.loads(meta)
    assert info["reason"] == "pre-forget" and info["contract"] == "v4" and info["log_max_id"] >= 1
    listing = call("memory", "restore")
    assert [b["slot"] for b in listing["backups"]] == [0, 1]


def test_rotation_keeps_three(call, db):
    call("memory", "ingest", params={"triples": CORPUS})
    slots = [call("memory", "consolidate")["backup_slot"] for _ in range(5)]
    assert slots == [0, 1, 2, 0, 1]
    assert len(list((db.parent / "backups").glob("substrate.*.db"))) == 3


def test_restore_round_trip_and_reversible(call, db):
    call("memory", "ingest", params={"triples": CORPUS})
    call("memory", "update", params={"query": "capital", "node": "París"})
    before = _live_dump(db)
    call("memory", "forget", params={"source": "wiki"})  # slot 0 holds the pre-forget state
    after_forget = _live_dump(db)
    assert after_forget != before
    out = call("memory", "restore", params={"slot": 0})
    assert out["status"] == "OK" and out["restored_slot"] == 0
    assert _live_dump(db) == before
    st = memory._get_state()
    with st.read._db.reader() as conn:
        last = conn.execute("SELECT op, payload FROM log ORDER BY id DESC LIMIT 1").fetchone()
    assert last["op"] == "restore" and json.loads(last["payload"])["from_slot"] == 0
    assert call("memory", "recall", params={"query": "paris", "mode": "seeds"})["status"] == "OK"
    # reversible: the state before the restore sits in pre_restore_slot
    again = call("memory", "restore", params={"slot": out["pre_restore_slot"]})
    assert again["status"] == "OK"
    assert _live_dump(db) == after_forget


def test_corrupted_backup_rejected_live_untouched(call, db):
    call("memory", "ingest", params={"triples": CORPUS})
    call("memory", "forget", params={"source": "wiki"})
    live = _live_dump(db)
    (db.parent / "backups" / "substrate.0.db").write_bytes(b"not a database" * 100)
    out = call("memory", "restore", params={"slot": 0})
    assert out["status"] == "ERROR"
    assert _live_dump(db) == live


def test_restore_errors(call):
    assert call("memory", "restore", params={"slot": 1})["status"] == "ERROR"  # empty slot
    assert call("memory", "restore", params={"slot": 7})["status"] == "ERROR"
    assert call("memory", "restore", params={"slot": True})["status"] == "ERROR"


def test_memory_db_skips_backup(call):
    memory.configure(":memory:")
    call("memory", "ingest", params={"triples": CORPUS})
    out = call("memory", "forget", params={"source": "wiki"})
    assert out["status"] == "OK" and out["backup_slot"] is None
    assert call("memory", "restore", params={"slot": 0})["status"] == "ERROR"


def test_forget_fails_closed_when_backup_fails(call, db, monkeypatch):
    call("memory", "ingest", params={"triples": CORPUS})
    st = memory._get_state()
    from chrysalis_mcp.substrate import BackupError

    def boom(reason):
        raise BackupError("disk full")

    monkeypatch.setattr(st.write, "backup", boom)
    before = _live_dump(db)
    out = call("memory", "forget", params={"source": "wiki"})
    assert out["status"] == "ERROR" and "nothing written" in out["error"]
    assert _live_dump(db) == before
