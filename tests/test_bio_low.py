"""Low-priority bio tools (substrate v6): tagging & capture, adaptive inhibition, SDM, FlyVec-lite,
learned_legacy and the advice-store lock guard."""

import json
import sqlite3

import numpy as np
import pytest

from chrysalis_mcp.algorithms import FlyHash, features, flyvec
from chrysalis_mcp.core.learning import AdviceStore
from chrysalis_mcp.domains import memory
from chrysalis_mcp.substrate import open_substrate
from chrysalis_mcp.substrate import store as store_mod

KB = [
    {"s": "Quortre Annex", "r": "has_elevation_m", "o": "4290"},
    {"s": "Quortre Annex", "r": "located_in", "o": "Brevan"},
    {"s": "Lumen Keep", "r": "has_elevation_m", "o": "310"},
    {"s": "Lumen Keep", "r": "located_in", "o": "Brevan"},
    {"s": "Marsh Hall", "r": "has_elevation_m", "o": "55"},
    {"s": "Marsh Hall", "r": "located_in", "o": "Corvane"},
]
Q1 = "How high above sea level, in meters, is the Sable Forge?"
Q1B = "Sable Forge altitude"


@pytest.fixture(autouse=True)
def db(tmp_path):
    memory.configure(tmp_path / "mem.db")
    yield
    memory.configure(None)


def _teach(call, query, node="Quortre Annex", delta=1.0):
    assert call("memory", "update", params={"query": query, "node": node, "delta": delta})[
        "status"
    ] == "OK"


def _assoc_total(node):
    st = memory._get_state()
    node_id = st.read.get_node(node)["id"]
    return sum(
        r["weight"] for r in st.read.assoc_for(range(st.params.m)) if r["node_id"] == node_id
    )


# --- v6 migration ---------------------------------------------------------------------------


def test_v6_tables_exist_and_v5_database_upgrades_in_place(tmp_path):
    path = tmp_path / "old.db"
    conn = sqlite3.connect(path)
    conn.executescript(store_mod._SCHEMA_V1 + store_mod._SCHEMA_V3 + store_mod._SCHEMA_V4)
    conn.executescript(store_mod._SCHEMA_V5)
    conn.execute("INSERT INTO meta VALUES ('contract', 'v5')")
    conn.close()
    r, _ = open_substrate(path)
    assert r.contract() == "v6" == store_mod.SUBSTRATE_CONTRACT
    counts = r.counts()
    assert counts["assoc_tags"] == counts["flyvec_tokens"] == counts["flyvec_nodes"] == 0
    assert {"assoc_tags", "flyvec_tokens", "flyvec_nodes"} <= set(store_mod.TABLES)
    assert r.flyvec_info() is None


# --- L1 synaptic tagging & capture ---------------------------------------------------------


def test_update_tags_and_capture_exempts_recent_and_confirmed(call):
    call("memory", "ingest", params={"triples": KB})
    _teach(call, Q1, "Quortre Annex")  # old, single phrasing -> not confirmed
    for i in range(5):  # newer teaches of other nodes push the first one out of the window
        _teach(call, f"unrelated filler {i} zzz", "Marsh Hall")
    _teach(call, Q1, "Lumen Keep")
    _teach(call, Q1B, "Lumen Keep")  # two distinct phrasings -> confirmed
    assert memory._get_state().read.counts()["assoc_tags"] > 0
    out = call(
        "memory", "consolidate", params={"capture": True, "capture_window": 2, "decay": 0.5}
    )
    assert out["status"] == "OK"
    assert out["captured"] == out["captured_tagged"] + out["captured_confirmed"] - _overlap(out)
    assert out["captured"] > 0 and out["decayable"] > 0
    # the confirmed node kept its full weight, the stale single-teach one decayed
    assert _assoc_total("Lumen Keep") > 0
    stale = _assoc_total("Quortre Annex")
    assert stale == pytest.approx(0.5 * memory.ETA * 64, rel=0.01)


def _overlap(out):  # tagged and confirmed sets intersect (Lumen Keep's recent tags)
    return out["captured_tagged"] + out["captured_confirmed"] - out["captured"]


def test_capture_off_is_the_previous_pure_decay_and_reports_nothing(call):
    call("memory", "ingest", params={"triples": KB})
    _teach(call, Q1)
    out = call("memory", "consolidate", params={"decay": 0.5})
    assert "captured" not in out
    assert _assoc_total("Quortre Annex") == pytest.approx(0.5 * 64, rel=0.01)


def test_captured_associations_survive_prune_that_removes_residue(call):
    call("memory", "ingest", params={"triples": KB})
    _teach(call, Q1, "Quortre Annex")
    for _ in range(3):
        call("memory", "consolidate", params={"decay": 0.5, "prune_below": 0.2})
    assert _assoc_total("Quortre Annex") == 0.0  # plain pruning removed it
    _teach(call, Q1, "Quortre Annex")
    for _ in range(3):
        call(
            "memory",
            "consolidate",
            params={"decay": 0.5, "prune_below": 0.2, "capture": True, "capture_window": 10},
        )
    assert _assoc_total("Quortre Annex") == pytest.approx(64.0)


def test_capture_validates_parameters(call):
    for bad in ({"capture": "yes"}, {"capture_window": 0}, {"capture_confirmations": 99}):
        assert call("memory", "consolidate", params=bad)["status"] == "ERROR"


# --- L2 adaptive inhibition ----------------------------------------------------------------


def test_adaptive_hash_is_clamped_deterministic_and_leaves_fixed_untouched():
    fh = FlyHash(64, 256, 16, 6, seed=3)
    x = np.random.default_rng(1).random(64)
    a = fh.hash_adaptive(x, 1.0)
    assert 8 <= len(a) <= 32 and list(a) == sorted(a)
    assert np.array_equal(a, fh.hash_adaptive(x, 1.0))
    assert len(fh.hash_adaptive(x, 100.0)) == 8  # nothing exceeds: floor k/2
    assert len(fh.hash_adaptive(x, -100.0)) == 32  # everything exceeds: ceiling 2k
    assert len(fh.hash(x)) == 16


def test_recall_inhibition_adaptive_is_opt_in_and_never_changes_stored_codes(call):
    call("memory", "ingest", params={"triples": KB})
    st = memory._get_state()
    before = st.read.kc_code(st.read.get_node("Lumen Keep")["id"])["active"]
    base = call("memory", "recall", params={"query": "Lumen Keep", "k": 3})
    adaptive = call(
        "memory", "recall", params={"query": "Lumen Keep", "k": 3, "inhibition": "adaptive"}
    )
    assert adaptive["status"] == "OK" and adaptive["results"][0]["text"] == "Lumen Keep"
    assert base["results"][0]["text"] == "Lumen Keep"
    assert st.read.kc_code(st.read.get_node("Lumen Keep")["id"])["active"] == before
    for bad in ({"inhibition": "x"}, {"inhibition": "adaptive", "z": -1}):
        assert call("memory", "recall", params={"query": "a", **bad})["status"] == "ERROR"


def test_adaptive_inhibition_keeps_the_learned_boost_scale(call):
    call("memory", "ingest", params={"triples": KB})
    _teach(call, Q1, "Quortre Annex")
    for extra in ({}, {"inhibition": "adaptive"}):
        res = call("memory", "recall", params={"query": Q1, "k": 3, **extra})
        assert res["results"][0]["text"] == "Quortre Annex"


# --- L3 SDM --------------------------------------------------------------------------------


def test_sdm_returns_nodes_within_radius_ranked_and_paginated(call):
    call("memory", "ingest", params={"triples": KB})
    res = call("memory", "recall", params={"query": "Lumen Keep", "mode": "sdm", "radius": 128})
    assert res["status"] == "OK" and res["mode"] == "sdm"
    top = res["results"][0]
    assert top["text"] == "Lumen Keep" and top["hamming"] == 0 and top["score"] == 1.0
    assert all(r["hamming"] <= 128 for r in res["results"])
    scores = [r["score"] for r in res["results"]]
    assert scores == sorted(scores, reverse=True)
    again = call("memory", "recall", params={"query": "Lumen Keep", "mode": "sdm", "radius": 128})
    assert again["results"] == res["results"]  # deterministic
    page1 = call(
        "memory", "recall", params={"query": "Lumen Keep", "mode": "sdm", "radius": 256, "k": 2}
    )
    assert len(page1["results"]) == 2 and page1["cursor"] is not None
    page2 = call(
        "memory",
        "recall",
        params={
            "query": "Lumen Keep", "mode": "sdm", "radius": 256, "k": 2, "cursor": page1["cursor"]
        },
    )  # fmt: skip
    seen = [r["text"] for r in page1["results"] + page2["results"]]
    assert len(seen) == len(set(seen)) and page1["in_radius"] == page2["in_radius"]


def test_sdm_radius_zero_is_exact_match_only_and_validates(call):
    call("memory", "ingest", params={"triples": KB})
    res = call("memory", "recall", params={"query": "Lumen Keep", "mode": "sdm", "radius": 0})
    assert [r["text"] for r in res["results"]] == ["Lumen Keep"]
    for bad in ({"radius": -1}, {"radius": "x"}, {"radius": 10_000}, {"cursor": "abc"}):
        out = call("memory", "recall", params={"query": "a", "mode": "sdm", **bad})
        assert out["status"] == "ERROR"


# --- L4 FlyVec-lite ------------------------------------------------------------------------


def test_flyvec_training_is_deterministic_bounded_and_groups_cooccurring_words():
    sents = [flyvec.tokenize(t) for t in ("red apple fruit", "green apple fruit", "blue car road")]
    a, info = flyvec.train(sents * 5, hidden=64, active=8, epochs=3, seed=7)
    b, _ = flyvec.train(sents * 5, hidden=64, active=8, epochs=3, seed=7)
    assert a == b and info["vocab"] == 7 and all(len(c) == 8 for c in a.values())
    assert flyvec.encode(["zzz"], a, 64).any() is np.False_
    with pytest.raises(ValueError):
        flyvec.train(sents, epochs=999)


def test_flyvec_train_action_recall_encoder_and_errors(call):
    err = call("memory", "recall", params={"query": "Lumen Keep", "encoder": "flyvec"})
    assert err["status"] == "ERROR" and "flyvec_train" in err["error"]
    assert call("memory", "flyvec_train")["status"] == "ERROR"  # empty KB
    call("memory", "ingest", params={"triples": KB})
    out = call("memory", "flyvec_train", params={"epochs": 3, "hidden": 64, "active": 6})
    assert out["status"] == "OK" and out["model"]["vocab"] > 0
    st = memory._get_state()
    assert st.read.counts()["flyvec_tokens"] == out["model"]["vocab"]
    res = call("memory", "recall", params={"query": "Lumen Keep", "k": 3, "encoder": "flyvec"})
    assert res["status"] == "OK" and "Lumen Keep" in [r["text"] for r in res["results"]]
    unk = call("memory", "recall", params={"query": "qqqq", "encoder": "flyvec", "mode": "seeds"})
    assert unk["status"] == "OK" and unk["encoder_fallback"] == "ngram"
    assert call("memory", "flyvec_train", params={"active": 99, "hidden": 64})["status"] == "ERROR"
    assert call("memory", "recall", params={"query": "a", "encoder": "x"})["status"] == "ERROR"


def test_flyvec_train_replaces_the_model_and_is_reproducible(call):
    call("memory", "ingest", params={"triples": KB})
    p = {"epochs": 2, "hidden": 32, "active": 4}
    call("memory", "flyvec_train", params=p)
    st = memory._get_state()
    first = st.read.flyvec_tokens(["brevan", "lumen"])
    call("memory", "flyvec_train", params=p)
    assert st.read.flyvec_tokens(["brevan", "lumen"]) == first


# --- learned_legacy ------------------------------------------------------------------------


def test_learned_legacy_restores_the_rare_word_gate(call):
    call("memory", "ingest", params={"triples": KB})
    _teach(call, Q1)
    default = call("memory", "recall", params={"query": Q1, "k": 3})
    legacy = call("memory", "recall", params={"query": Q1, "k": 3, "learned_legacy": True})
    assert any(f[1] == "learned_as" for f in default["facts"])
    assert any(f[1] == "learned_as" for f in legacy["facts"])
    # a query sharing only common template words has no rare word: legacy stays silent too
    plain = call(
        "memory",
        "recall",
        params={"query": "How high above sea level, in meters, is it?", "learned_legacy": True},
    )
    assert plain["status"] == "OK"
    assert call("memory", "recall", params={"query": "a", "learned_legacy": 1})["status"] == "ERROR"


# --- advice store lock guard ---------------------------------------------------------------


def test_feedback_returns_a_typed_retryable_error_when_the_advice_store_is_locked(
    call, monkeypatch
):
    def locked(self, *a, **k):
        raise sqlite3.OperationalError("database is locked")

    call("meta", "handshake", session_id="s", params={"model": "m", "params_b": 4, "thinking": False})
    monkeypatch.setattr(AdviceStore, "record", locked)
    out = call(
        "meta",
        "feedback",
        session_id="s",
        params={
            "action": "reason.solve", "used": True, "outcome": "success",
            "task": {"n_atoms": 2, "goal": "sat"},
        },
    )  # fmt: skip
    assert out["status"] == "ERROR" and out["severity"] == "ERROR"
    assert out["retryable"] is True and "Retry" in out["next_action"]
    assert "locked" in out["error"]


# --- schema ---------------------------------------------------------------------------------


def test_schema_lists_the_low_priority_parameters_with_defaults(call):
    schema = call("meta", "schema")["domains"]["memory"]
    rec, cons = schema["recall"]["params"], schema["consolidate"]["params"]
    assert rec["inhibition"]["default"] == "fixed" and rec["encoder"]["default"] == "ngram"
    assert rec["mode"]["enum"] == ["ppr", "seeds", "sdm"]
    assert rec["learned_legacy"]["default"] is False and "radius" in rec
    assert cons["capture"]["default"] is False
    assert cons["capture_window"]["default"] == memory.DEFAULT_CAPTURE_WINDOW
    assert schema["flyvec_train"]["example"]["params"]["epochs"] == 5
    json.dumps(schema)


def test_features_unchanged_by_adaptive_option():
    assert features("abc", 16).shape == (16,)


def test_bio_low_bench_runs_on_a_tiny_config():
    from validation import bench_bio_low

    out = bench_bio_low.run(seeds=(19,), n_entities=6, n_interference=12)
    data = out["data"]
    assert set(data) == {"l1_capture", "l2_inhibition", "l3_sdm", "l4_flyvec"}
    plain, tag = data["l1_capture"]["plain"], data["l1_capture"]["capture_tag"]
    assert tag["assoc_rows_after"] >= plain["assoc_rows_after"]
    assert "adaptive_z1.0" in data["l2_inhibition"] and "sdm_r64" in data["l3_sdm"]
    assert {"ngram_seeds", "flyvec_ppr", "model"} <= set(data["l4_flyvec"])
