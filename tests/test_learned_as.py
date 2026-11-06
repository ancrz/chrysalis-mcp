"""T21: learned associations are explainable (``examples`` table, ``learned_as`` facts)."""

import sqlite3

import pytest

from chrysalis_mcp.domains import memory
from chrysalis_mcp.substrate import open_substrate
from chrysalis_mcp.substrate import store as store_mod
from chrysalis_mcp.substrate.store import MAX_EXAMPLES_PER_NODE

KB = [
    {"s": "Quortre Annex", "r": "has_elevation_m", "o": "4290"},
    {"s": "Quortre Annex", "r": "located_in", "o": "Brevan"},
    {"s": "Lumen Keep", "r": "has_elevation_m", "o": "310"},
    {"s": "Marsh Hall", "r": "has_elevation_m", "o": "55"},
]
ALIAS_Q = "How high above sea level, in meters, is the Sable Forge?"
UNRELATED_Q = "What is the elevation of Marsh Hall?"


@pytest.fixture(autouse=True)
def db(tmp_path):
    memory.configure(tmp_path / "mem.db")
    yield tmp_path / "mem.db"
    memory.configure(None)


def _examples(node: str) -> list[str]:
    st = memory._get_state()
    node_id = st.read.get_node(node)["id"]
    return [r["query"] for r in st.read.examples_for([node_id]).get(node_id, [])]


def _learned(res: dict) -> list[list[str]]:
    return [f for f in res["facts"] if f[1] == "learned_as"]


def _teach(call, query: str = ALIAS_Q.replace("How high", "Tell me how high"), n: int = 1):
    for _ in range(n):
        assert (
            call("memory", "update", params={"query": query, "node": "Quortre Annex"})["status"]
            == "OK"
        )


# --- substrate: migration ------------------------------------------------------------------------


def test_fresh_database_is_v4_with_an_empty_examples_table(tmp_path):
    r, _ = open_substrate(tmp_path / "fresh.db")
    assert r.contract() == "v4" == store_mod.SUBSTRATE_CONTRACT
    assert r.counts()["examples"] == 0 and r.examples_for([1, 2]) == {}


def test_v3_database_is_upgraded_idempotently_without_backfill(tmp_path):
    path = tmp_path / "v3.db"
    conn = sqlite3.connect(path)
    conn.executescript(store_mod._SCHEMA_V1 + store_mod._SCHEMA_V3)
    conn.execute("INSERT INTO meta VALUES ('contract', 'v3')")
    for key in ("d", "m", "k", "c", "seed"):
        value = getattr(store_mod.FlyHashParams(), key)
        conn.execute("INSERT INTO meta VALUES (?, ?)", (f"flyhash_{key}", str(value)))
    conn.execute("INSERT INTO nodes (text) VALUES ('old')")
    conn.execute("INSERT INTO assoc (kc, node_id, weight) VALUES (1, 1, 2.0)")
    conn.commit()
    conn.close()
    r, _ = open_substrate(path)
    assert r.contract() == "v4"
    assert r.counts()["nodes"] == 1 and r.counts()["assoc"] == 1 and r.counts()["examples"] == 0
    schema = sqlite3.connect(path).execute("SELECT type, name FROM sqlite_master ORDER BY 1,2")
    first = schema.fetchall()
    r2, _ = open_substrate(path)
    again = sqlite3.connect(path).execute("SELECT type, name FROM sqlite_master ORDER BY 1,2")
    assert again.fetchall() == first and r2.counts()["examples"] == 0
    assert r2.counts()["log"] == 0  # migration never writes log rows


# --- example storage -----------------------------------------------------------------------------


def test_examples_are_stored_deduped_and_capped_to_the_most_recent(call):
    call("memory", "ingest", params={"triples": KB})
    queries = [f"alias variant number {i}" for i in range(5)]
    for q in queries:
        _teach(call, q)
    assert MAX_EXAMPLES_PER_NODE == 3
    assert _examples("Quortre Annex") == queries[:1:-1]  # 4, 3, 2 (most recent first)
    _teach(call, queries[2])  # dedupe: refreshes ts and accumulates weight, no new row
    assert _examples("Quortre Annex") == [queries[2], queries[4], queries[3]]
    st = memory._get_state()
    node_id = st.read.get_node("Quortre Annex")["id"]
    assert st.read.examples_for([node_id])[node_id][0]["weight"] == pytest.approx(2.0)
    assert st.read.counts()["examples"] == 3


def test_negative_delta_removes_the_example_once_the_association_is_cancelled(call):
    call("memory", "ingest", params={"triples": KB})
    _teach(call, "alias q one")
    _teach(call, "alias q two")
    call(
        "memory", "update", params={"query": "alias q one", "node": "Quortre Annex", "delta": -0.5}
    )
    assert "alias q one" in _examples("Quortre Annex")  # mass still positive: kept
    call("memory", "update", params={"query": "alias q one", "node": "Quortre Annex", "delta": -2})
    assert _examples("Quortre Annex") == ["alias q two"]


# --- recall --------------------------------------------------------------------------------------


def test_recall_explains_boosted_nodes_only(call):
    call("memory", "ingest", params={"triples": KB})
    before = call("memory", "recall", params={"query": ALIAS_Q, "k": 5})
    assert _learned(before) == [] and not any("learned" in r for r in before["results"])
    _teach(call, n=3)
    res = call("memory", "recall", params={"query": ALIAS_Q, "k": 5})
    assert res["results"][0]["text"] == "Quortre Annex" and res["results"][0]["learned"] is True
    learned = _learned(res)
    assert learned and all(f[2] == "Quortre Annex" and "Sable Forge" in f[0] for f in learned)
    assert len(learned) <= memory.MAX_LEARNED_EXAMPLES and res["facts"][0] == learned[0]
    assert ["Quortre Annex", "has_elevation_m", "4290"] in res["facts"]
    assert len(res["facts"]) <= 2 * 5
    # unrelated query, lambda=0 and seeds mode: nothing learned is explained
    other = call("memory", "recall", params={"query": UNRELATED_Q, "k": 5})
    assert _learned(other) == [] and not any("learned" in r for r in other["results"])
    off = call("memory", "recall", params={"query": ALIAS_Q, "k": 5, "lambda": 0})
    assert _learned(off) == []
    seeds = call("memory", "recall", params={"query": ALIAS_Q, "mode": "seeds"})
    assert "facts" not in seeds and not any("learned" in r for r in seeds["results"])


def test_recall_picks_the_examples_most_similar_to_the_query(call):
    call("memory", "ingest", params={"triples": KB})
    for q in (
        "Sable Forge height",
        "Sable Forge in meters",
        "how high above sea level Sable Forge",
    ):
        _teach(call, q)
    res = call("memory", "recall", params={"query": "Sable Forge in meters", "k": 5})
    quoted = [f[0] for f in _learned(res)]
    assert len(quoted) == 2 and quoted[0] == "Sable Forge in meters"


def test_learned_facts_respect_the_fact_cap(call):
    call("memory", "ingest", params={"triples": KB})
    _teach(call, n=3)
    res = call("memory", "recall", params={"query": ALIAS_Q, "k": 1})
    assert len(res["facts"]) <= 2 and _learned(res)


# --- consolidate / forget / restore --------------------------------------------------------------


def test_consolidate_pruning_all_assoc_drops_the_examples(call):
    call("memory", "ingest", params={"triples": KB})
    _teach(call)
    assert _examples("Quortre Annex")
    call("memory", "consolidate", params={"decay": 0.9, "prune_below": 1e-3})
    assert _examples("Quortre Annex")  # assoc survives a mild decay: examples follow assoc
    call("memory", "consolidate", params={"decay": 0.5, "prune_below": 10})
    assert _examples("Quortre Annex") == []
    assert memory._get_state().read.counts()["assoc"] == 0


def test_forget_by_source_leaves_examples_alone(call):
    call("memory", "ingest", params={"triples": KB, "source": "kb"})
    _teach(call)
    call("memory", "forget", params={"source": "kb"})
    assert _examples("Quortre Annex")  # examples follow assoc, not sources


def test_restore_round_trip_includes_examples(call):
    call("memory", "ingest", params={"triples": KB})
    _teach(call, "alias q one")
    snap = call("memory", "consolidate", params={"decay": 1.0, "prune_below": 0})["backup_slot"]
    _teach(call, "alias q two")
    assert _examples("Quortre Annex") == ["alias q two", "alias q one"]
    call("memory", "restore", params={"slot": snap})
    assert _examples("Quortre Annex") == ["alias q one"]
