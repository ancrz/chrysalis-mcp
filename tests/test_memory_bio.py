"""Substrate v5 / bio layer: negative selection, quorum, replay, usage, v4 -> v5 migration."""

import json
import random
import sqlite3
import struct

import pytest

from chrysalis_mcp.domains import memory
from chrysalis_mcp.substrate import open_substrate
from chrysalis_mcp.substrate import store as store_mod
from chrysalis_mcp.substrate.store import MAX_TEACH_QUERIES
from validation import bench_hb_e2e
from validation.datasets import alias_dataset

KB = [
    {"s": "Quortre Annex", "r": "has_elevation_m", "o": "4290"},
    {"s": "Quortre Annex", "r": "located_in", "o": "Brevan"},
    {"s": "Lumen Keep", "r": "has_elevation_m", "o": "310"},
    {"s": "Marsh Hall", "r": "has_elevation_m", "o": "55"},
]
ALIAS_Q = "How high above sea level, in meters, is the Sable Forge?"
TEACH_1 = "Tell me how high above sea level, in meters, is the Sable Forge?"
TEACH_2 = "Sable Forge altitude"


@pytest.fixture(autouse=True)
def db(tmp_path):
    memory.configure(tmp_path / "mem.db")
    yield tmp_path / "mem.db"
    memory.configure(None)


def _teach(call, query: str, node: str = "Quortre Annex", delta: float = 1.0, n: int = 1):
    for _ in range(n):
        out = call("memory", "update", params={"query": query, "node": node, "delta": delta})
        assert out["status"] == "OK"


def _recall(call, query: str = ALIAS_Q, **extra):
    return call("memory", "recall", params={"query": query, "k": 5, **extra})


def _learned(res: dict) -> list[list[str]]:
    return [f for f in res["facts"] if f[1] == "learned_as"]


def _assoc_total(node: str) -> float:
    st = memory._get_state()
    node_id = st.read.get_node(node)["id"]
    return sum(
        r["weight"] for r in st.read.assoc_for(range(st.params.m)) if r["node_id"] == node_id
    )


# --- migration v4 -> v5 --------------------------------------------------------------------------


def _make_v4(path, with_example: bool = True) -> None:
    conn = sqlite3.connect(path)
    conn.executescript(store_mod._SCHEMA_V1 + store_mod._SCHEMA_V3 + store_mod._SCHEMA_V4)
    conn.execute("INSERT INTO meta VALUES ('contract', 'v4')")
    for key in ("d", "m", "k", "c", "seed"):
        value = getattr(store_mod.FlyHashParams(), key)
        conn.execute("INSERT INTO meta VALUES (?, ?)", (f"flyhash_{key}", str(value)))
    conn.execute("INSERT INTO nodes (text) VALUES ('Quortre Annex')")
    conn.execute("INSERT INTO nodes (text) VALUES ('Lumen Keep')")
    conn.execute("INSERT INTO edges (subj, rel, obj) VALUES (1, 'near', 2)")
    conn.execute("INSERT INTO assoc (kc, node_id, weight) VALUES (7, 1, 2.0)")
    if with_example:
        conn.execute("INSERT INTO examples VALUES (1, ?, 2.0, 5)", (TEACH_1,))
        conn.execute("INSERT INTO examples VALUES (1, ?, 1.0, 6)", (TEACH_2,))
    conn.commit()
    conn.close()


def test_fresh_database_is_v5_with_empty_bio_tables(tmp_path):
    r, _ = open_substrate(tmp_path / "fresh.db")
    assert r.contract() == "v6" == store_mod.SUBSTRATE_CONTRACT
    counts = r.counts()
    assert counts["teach_queries"] == counts["node_kcs"] == counts["edge_usage"] == 0
    assert {"teach_queries", "node_kcs", "edge_usage"} <= set(store_mod.TABLES)


def test_v4_database_is_upgraded_in_place_and_idempotently(tmp_path):
    path = tmp_path / "v4.db"
    _make_v4(path)
    r, _ = open_substrate(path)  # substrate layer: only adds tables, nothing is backfilled
    assert r.contract() == "v6"
    assert r.counts()["nodes"] == 2 and r.counts()["assoc"] == 1 and r.counts()["examples"] == 2
    assert r.counts()["teach_queries"] == 0 and r.counts()["log"] == 0
    first = sqlite3.connect(path).execute("SELECT type, name FROM sqlite_master ORDER BY 1,2")
    first = first.fetchall()
    open_substrate(path)
    again = sqlite3.connect(path).execute("SELECT type, name FROM sqlite_master ORDER BY 1,2")
    assert again.fetchall() == first


def test_v4_examples_are_backfilled_by_the_domain_exactly_once(tmp_path):
    path = tmp_path / "v4b.db"
    _make_v4(path)
    memory.configure(path)
    st = memory._get_state()
    counts = st.read.counts()
    assert counts["teach_queries"] == 2 and counts["node_kcs"] > 0
    teach = st.read.teach_for([1])[1]
    assert {t["query"] for t in teach} == {TEACH_1, TEACH_2}
    assert all(len(t["active"]) == st.params.k for t in teach)
    assert counts["assoc"] == 1 and counts["examples"] == 2  # v4 data untouched
    log_ops = [r["op"] for r in st.read.log_tail(20)]
    assert log_ops.count("backfill_teach") == 1
    memory.configure(path)  # reopen: nothing left to backfill
    st2 = memory._get_state()
    assert st2.read.counts()["teach_queries"] == 2
    assert [r["op"] for r in st2.read.log_tail(20)].count("backfill_teach") == 1


def test_backfilled_v4_associations_are_explained_and_learned_status_is_confirmed(tmp_path, call):
    path = tmp_path / "v4c.db"
    _make_v4(path, with_example=False)
    memory.configure(path)
    call("memory", "ingest", params={"triples": KB})
    _teach(call, TEACH_1, n=3)
    _teach(call, TEACH_2, n=3)
    res = _recall(call)
    assert (
        res["results"][0]["learned"] is True and res["results"][0]["learned_status"] == "confirmed"
    )


# --- teach_queries / node_kcs bookkeeping ----------------------


def test_teach_queries_are_deduped_bounded_and_follow_node_kcs(call):
    call("memory", "ingest", params={"triples": KB})
    st = memory._get_state()
    node_id = st.read.get_node("Quortre Annex")["id"]
    for i in range(MAX_TEACH_QUERIES + 4):
        _teach(call, f"alias variant number {i}")
    _teach(call, "alias variant number 19")  # dedupe: no new row
    rows = st.read.teach_for([node_id])[node_id]
    assert len(rows) == MAX_TEACH_QUERIES and rows[0]["query"] == "alias variant number 19"
    union = {kc for r in rows for kc in r["active"]}
    df = st.read.kc_node_counts(union)
    assert set(df) == union and set(df.values()) == {1}  # one node: df == 1 for every kc


def test_negative_delta_cancelling_a_teach_removes_its_kcs_and_row(call):
    call("memory", "ingest", params={"triples": KB})
    st = memory._get_state()
    node_id = st.read.get_node("Quortre Annex")["id"]
    _teach(call, "alias q one")
    _teach(call, "alias q two")
    _teach(call, "alias q one", delta=-2.0)
    assert [t["query"] for t in st.read.teach_for([node_id])[node_id]] == ["alias q two"]
    codes = set(st.read.teach_for([node_id])[node_id][0]["active"])
    assert set(st.read.kc_node_counts(range(st.params.m))) == codes


def test_consolidate_dropping_assoc_drops_teach_rows_and_node_kcs(call):
    call("memory", "ingest", params={"triples": KB})
    _teach(call, TEACH_1)
    assert memory._get_state().read.counts()["node_kcs"] > 0
    call("memory", "consolidate", params={"decay": 0.5, "prune_below": 10})
    c = memory._get_state().read.counts()
    assert c["teach_queries"] == c["node_kcs"] == c["examples"] == c["assoc"] == 0


# --- 1. negative selection ----------------------


def _decisions(data, order, path, k: int = 5):
    """learned_as facts per query (set-A tests + probes) after applying ``order`` on a fresh db."""
    run = bench_hb_e2e._BioRun(str(path))
    try:
        run.apply(bench_hb_e2e.kb_ops(data))
        run.apply(order)
        queries = [i.test for i in data.set_a] + [q for _, q in data.probes]
        return [frozenset((f[0], f[2]) for f in run.has_learned_as(q, k)) for q in queries]
    finally:
        memory.configure(None)


@pytest.mark.parametrize("common", [False, True])
def test_learned_as_decisions_are_invariant_to_update_order(common, tmp_path):
    data = alias_dataset(31, n_entities=8, n_interference=60, common_words=common)
    updates = [
        *bench_hb_e2e._teach_pairs("single", data.set_a),
        *bench_hb_e2e._teach_pairs("single", data.set_b),
        *[("update", {"query": q, "node": n}) for q, n in data.interference_pairs],
    ]
    baseline = _decisions(data, updates, tmp_path / "base.db")
    assert any(baseline)  # the property is not vacuous: some alias IS explained
    for seed in (1, 2, 3, 4):
        shuffled = list(updates)
        random.Random(seed).shuffle(shuffled)
        assert _decisions(data, shuffled, tmp_path / f"p{seed}.db") == baseline


def test_common_template_words_are_self_and_do_not_explain_an_untaught_alias(call):
    """Many nodes taught with one question template: the template cells become "self"."""
    triples = [{"s": f"Place{i}", "r": "elev", "o": str(i)} for i in range(12)]
    call("memory", "ingest", params={"triples": triples})
    for i in range(12):
        _teach(call, f"what is the elevation of the Zorb{i} Hold", node=f"Place{i}")
    st = memory._get_state()
    cells = st.read.kc_node_counts(range(st.params.m))
    assert max(cells.values()) >= 12  # template cells are shared by every taught node
    taught = _recall(call, "what is the elevation of the Zorb3 Hold")
    assert any("Zorb3" in f[0] for f in _learned(taught))
    template_only = _recall(call, "what is the elevation of the Quill Hold")
    assert _learned(template_only) == []


def test_specific_kcs_exclude_cells_firing_for_many_nodes(call):
    triples = [{"s": f"Place{i}", "r": "elev", "o": str(i)} for i in range(10)]
    call("memory", "ingest", params={"triples": triples})
    for i in range(10):
        _teach(call, f"what is the elevation of the Zorb{i} Hold", node=f"Place{i}")
    st = memory._get_state()
    q = memory._active(st, "what is the elevation of the Zorb3 Hold")
    specific = memory._specific_kcs(st, q)
    df = st.read.kc_node_counts(q)
    cut = max(memory.SELF_MIN_NODES, -(-int(memory.SELF_FRACTION * 100 * 10) // 100))
    assert specific == {kc for kc in q if df.get(kc, 0) < cut} and 0 < len(specific) < len(q)


# --- 2. quorum sensing ----------------------


def test_single_teach_is_provisional_and_two_distinct_phrasings_are_confirmed(call):
    call("memory", "ingest", params={"triples": KB})
    _teach(call, TEACH_1, n=3)
    one = _recall(call)
    assert one["results"][0]["learned_status"] == "provisional" and _learned(one)
    assert one["learned"] == [
        {"node": "Quortre Annex", "status": "provisional", "confirmations": 1}
    ]
    _teach(call, TEACH_2, n=3)
    two = _recall(call)
    assert two["results"][0]["learned_status"] == "confirmed"
    assert two["learned"][0]["confirmations"] == 2


def test_near_duplicate_teach_queries_count_as_one_confirmation(call):
    call("memory", "ingest", params={"triples": KB})
    _teach(call, TEACH_1, n=3)
    _teach(call, TEACH_1.upper(), n=3)  # same phrasing, different casing: same confirmation
    res = _recall(call)
    assert res["learned"][0]["confirmations"] == 1 and res["learned"][0]["status"] == "provisional"
    assert _recall(call, quorum=1)["learned"][0]["status"] == "confirmed"


def test_learned_min_status_confirmed_only_hides_provisional_explanations(call):
    call("memory", "ingest", params={"triples": KB})
    _teach(call, TEACH_1, n=3)
    default = _recall(call)
    strict = _recall(call, learned_min_status="confirmed")
    assert _learned(default) and _learned(strict) == [] and "learned" not in strict
    # ranking and scores are NOT gated by the status
    assert default["results"][0]["text"] == strict["results"][0]["text"] == "Quortre Annex"
    assert [r["score"] for r in default["results"]] == [r["score"] for r in strict["results"]]
    assert "learned" not in strict["results"][0]
    _teach(call, TEACH_2, n=3)
    assert _learned(_recall(call, learned_min_status="confirmed"))


def test_recall_rejects_bad_bio_parameters(call):
    call("memory", "ingest", params={"triples": KB})
    assert _recall(call, learned_min_status="nope")["status"] == "ERROR"
    assert _recall(call, quorum=0)["status"] == "ERROR"
    assert _recall(call, quorum=999)["status"] == "ERROR"
    assert _recall(call, track_usage="yes")["status"] == "ERROR"


# --- 3. replay ----------------------


def test_consolidate_defaults_are_the_previous_pure_decay(call):
    call("memory", "ingest", params={"triples": KB})
    _teach(call, TEACH_1)
    before = _assoc_total("Quortre Annex")
    out = call("memory", "consolidate", params={"decay": 0.5})
    assert out["status"] == "OK" and "replayed" not in out and "usage_applied" not in out
    assert _assoc_total("Quortre Annex") == pytest.approx(before * 0.5)


def test_replay_keeps_taught_associations_alive_under_strong_decay(call):
    call("memory", "ingest", params={"triples": KB})
    _teach(call, TEACH_1)
    _teach(call, "unrelated residue", node="Lumen Keep")
    st = memory._get_state()
    # residue without an example: bump_assoc has no example row, so it is not replayed
    lumen = st.read.get_node("Lumen Keep")["id"]
    st.write.bump_assoc(range(1000, 1010), lumen, 1.0)
    plain = {"decay": 0.5, "prune_below": 0.1}
    out = None
    for _ in range(4):
        out = call("memory", "consolidate", params={**plain, "replay": True})
    assert out["replayed"] == 2 and out["status"] == "OK"
    assert _assoc_total("Quortre Annex") > 0.1 * st.params.k  # taught association survived
    assert _learned(_recall(call))
    kcs = {r["kc"] for r in st.read.assoc_for(range(1000, 1010))}
    assert kcs == set()  # the un-replayed residue was pruned


def test_without_replay_the_same_cycles_prune_the_taught_association(call):
    call("memory", "ingest", params={"triples": KB})
    _teach(call, TEACH_1)
    for _ in range(4):
        call("memory", "consolidate", params={"decay": 0.5, "prune_below": 0.1})
    assert memory._get_state().read.counts()["assoc"] == 0


def test_replay_never_resurrects_a_cancelled_association_and_validates_rate(call):
    call("memory", "ingest", params={"triples": KB})
    _teach(call, TEACH_1)
    _teach(call, TEACH_1, delta=-5.0)  # cancelled: example removed, assoc mass <= 0
    out = call("memory", "consolidate", params={"replay": True})
    assert out["replayed"] == 0
    assert call("memory", "consolidate", params={"replay_rate": 0})["status"] == "ERROR"
    assert call("memory", "consolidate", params={"replay_rate": 2})["status"] == "ERROR"
    assert call("memory", "consolidate", params={"replay": "yes"})["status"] == "ERROR"


# --- 4. usage (pheromone) ----------------------


def test_track_usage_is_opt_in(call):
    call("memory", "ingest", params={"triples": KB})
    st = memory._get_state()
    _recall(call, "elevation of Quortre Annex")
    assert st.read.counts()["edge_usage"] == 0  # default: recall stays read-only
    _recall(call, "elevation of Quortre Annex", track_usage=True)
    assert st.read.counts()["edge_usage"] > 0
    assert any(r["op"] == "bump_usage" for r in st.read.log_tail(5))


def _edge_usage_values() -> list[float]:
    st = memory._get_state()
    with st.read._db.reader() as conn:
        return [r[0] for r in conn.execute("SELECT usage FROM edge_usage")]


def test_usage_counters_never_exceed_the_cap(call):
    call("memory", "ingest", params={"triples": KB})
    for _ in range(int(memory.USAGE_CAP) + 7):
        _recall(call, "elevation of Quortre Annex", track_usage=True)
    values = _edge_usage_values()
    assert values and max(values) == memory.USAGE_CAP


def test_usage_weighting_protects_used_nodes_and_evaporates_counters(call):
    call("memory", "ingest", params={"triples": KB})
    _teach(call, TEACH_1)  # Quortre Annex: will be used
    _teach(call, "marsh query words", node="Marsh Hall")  # never used
    for _ in range(3):
        _recall(call, "elevation of Quortre Annex", track_usage=True, k=1)
    used_before, unused_before = _assoc_total("Quortre Annex"), _assoc_total("Marsh Hall")
    usage_before = sum(_edge_usage_values())
    out = call("memory", "consolidate", params={"decay": 0.8, "usage_weighting": True})
    assert out["usage_applied"] is True
    used_ratio = _assoc_total("Quortre Annex") / used_before
    unused_ratio = _assoc_total("Marsh Hall") / unused_before
    assert unused_ratio == pytest.approx(0.8**memory.UNUSED_DECAY_EXPONENT)
    assert 0.8 < used_ratio <= 1.0 and used_ratio > unused_ratio  # protection, never growth
    assert sum(_edge_usage_values()) == pytest.approx(usage_before * memory.USAGE_EVAPORATION)


def test_usage_weighting_without_any_usage_falls_back_to_uniform_decay(call):
    call("memory", "ingest", params={"triples": KB})
    _teach(call, TEACH_1)
    before = _assoc_total("Quortre Annex")
    out = call("memory", "consolidate", params={"decay": 0.5, "usage_weighting": True})
    assert out["usage_applied"] is False
    assert _assoc_total("Quortre Annex") == pytest.approx(before * 0.5)


def test_usage_weighting_never_touches_source_facts(call):
    call("memory", "ingest", params={"triples": KB, "source": "kb"})
    _teach(call, TEACH_1)
    for _ in range(3):
        _recall(call, "elevation of Quortre Annex", track_usage=True)
    st = memory._get_state()
    with st.read._db.reader() as conn:
        edges_before = [tuple(r) for r in conn.execute("SELECT * FROM edges ORDER BY id")]
    for _ in range(6):
        call(
            "memory",
            "consolidate",
            params={"decay": 0.5, "prune_below": 0.2, "usage_weighting": True},
        )
    with st.read._db.reader() as conn:
        edges_after = [tuple(r) for r in conn.execute("SELECT * FROM edges ORDER BY id")]
    assert edges_after == edges_before
    assert call("memory", "consolidate", params={"usage_weighting": 1})["status"] == "ERROR"


def test_forgotten_edges_do_not_count_as_usage(call):
    call("memory", "ingest", params={"triples": KB, "source": "kb"})
    call("memory", "forget", params={"source": "kb"})
    assert memory._get_state().write.bump_usage([1, 2, 3], memory.USAGE_CAP) == 0


# --- schema, envelope, bench ----------------------


def test_schema_lists_the_new_parameters_with_defaults_and_examples(call):
    schema = call("meta", "schema")["domains"]["memory"]
    recall = schema["recall"]["params"]
    assert recall["quorum"]["default"] == memory.DEFAULT_QUORUM == 2
    assert recall["learned_min_status"]["enum"] == ["provisional", "confirmed"]
    assert recall["learned_min_status"]["default"] == "provisional"
    assert recall["track_usage"] == {"type": "boolean", "required": False, "default": False}
    cons = schema["consolidate"]["params"]
    assert cons["replay"]["default"] is False and cons["usage_weighting"]["default"] is False
    assert cons["replay_rate"]["default"] == memory.DEFAULT_REPLAY_RATE
    assert schema["consolidate"]["example"]["params"]["replay"] is True
    assert "learned_status" in schema["recall"]["description"]


def test_envelope_is_typed_for_the_new_paths(call):
    call("memory", "ingest", params={"triples": KB})
    _teach(call, TEACH_1, n=2)
    keys = {"status", "severity", "summary", "next_action", "truncated", "cursor"}
    res = _recall(call, track_usage=True, learned_min_status="confirmed")
    assert keys <= res.keys() and res["status"] == "OK"
    cons = call("memory", "consolidate", params={"replay": True, "usage_weighting": True})
    assert keys <= cons.keys() and cons["replayed"] == 1
    bad = call("memory", "consolidate", params={"replay_rate": "x"})
    assert bad["status"] == "ERROR" and keys <= bad.keys()
    terse = json.loads(json.dumps(res))
    assert "results" in terse


def test_restore_round_trip_includes_the_bio_tables(call):
    call("memory", "ingest", params={"triples": KB})
    _teach(call, TEACH_1)
    snap = call("memory", "consolidate", params={"decay": 1.0, "prune_below": 0})["backup_slot"]
    _teach(call, TEACH_2)
    st = memory._get_state()
    assert st.read.counts()["teach_queries"] == 2
    call("memory", "restore", params={"slot": snap})
    assert memory._get_state().read.counts()["teach_queries"] == 1


def test_bio_bench_runs_and_reports_the_new_metrics(tmp_path):
    out = bench_hb_e2e.run_bio(
        seeds=(19,),
        variants=("common",),
        arms=("single", "decay_replay", "usage_on"),
        n_entities=6,
        n_interference=10,
    )
    arms = out["data"]["common"]
    assert arms["order_agreement"] == 1.0
    for name in ("single", "decay_replay", "usage_on"):
        assert arms[name] is not None
    assert {"explained", "false_learned_as", "explained_confirmed", "false_confirmed"} <= set(
        arms["single"]
    )
    assert {"after_cycles_used", "after_cycles_unused"} <= set(arms["usage_on"])


def test_struct_roundtrip_of_stored_codes(call):
    call("memory", "ingest", params={"triples": KB})
    _teach(call, TEACH_1)
    st = memory._get_state()
    with st.read._db.reader() as conn:
        blob = conn.execute("SELECT active FROM teach_queries").fetchone()[0]
    assert list(struct.unpack(f"<{len(blob) // 4}I", blob)) == sorted(memory._active(st, TEACH_1))
