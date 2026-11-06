import sqlite3

import pytest

from chrysalis_mcp.domains import memory
from chrysalis_mcp.substrate import store as store_mod

from .test_substrate_dense import _make_v2

CORPUS = [
    {"s": "París", "r": "capital_de", "o": "Francia"},
    {"s": "paris", "r": "es", "o": "ciudad"},
    {"s": "Berlín", "r": "capital_de", "o": "Alemania"},
    {"s": "mitocondria", "r": "produce", "o": "ATP"},
    {"s": "quantum chromodynamics", "r": "describe", "o": "quarks"},
]


@pytest.fixture(autouse=True)
def db(tmp_path):
    memory.configure(tmp_path / "mem.db")
    yield tmp_path / "mem.db"
    memory.configure(None)


def texts(res):
    return [r["text"] for r in res["results"]]


def test_default_is_cosine_and_flyhash_is_selectable(call):
    call("memory", "ingest", params={"triples": CORPUS})
    default = call("memory", "recall", params={"query": "parís", "k": 3, "mode": "seeds"})
    cos = call(
        "memory", "recall", params={"query": "parís", "k": 3, "mode": "seeds", "seeds": "cosine"}
    )
    fly = call(
        "memory", "recall", params={"query": "parís", "k": 3, "mode": "seeds", "seeds": "flyhash"}
    )
    assert default["results"] == cos["results"]
    assert default["seeds"] == "cosine" and fly["seeds"] == "flyhash"
    assert "similarity" in cos["results"][0] and "overlap" in fly["results"][0]
    assert cos["results"][0]["similarity"] == pytest.approx(1.0, abs=1e-5)


@pytest.mark.parametrize("mode", ["ppr", "seeds"])
def test_both_backends_recall_the_obvious_answer(call, mode):
    call("memory", "ingest", params={"triples": CORPUS})
    for backend in ("cosine", "flyhash"):
        res = call(
            "memory", "recall", params={"query": "paris", "k": 5, "mode": mode, "seeds": backend}
        )
        assert res["status"] == "OK" and res["seeds"] == backend
        assert {"París", "paris"} & set(texts(res))


def test_invalid_backend_is_structured_error(call):
    out = call("memory", "recall", params={"query": "x", "seeds": "annoy"})
    assert out["status"] == "ERROR" and out["valid"] == ["cosine", "flyhash"]


def test_cosine_beats_flyhash_precision_on_ngram_neighbours(call):
    words = ["capital", "capitol", "capita", "capitan", "orbital", "habitat", "marmot", "zephyr"]
    call("memory", "ingest", params={"triples": [{"s": w, "r": "is", "o": "word"} for w in words]})
    res = call("memory", "recall", params={"query": "capital", "k": 3, "mode": "seeds"})
    assert texts(res)[0] == "capital" and set(texts(res)[:3]) <= {
        "capital",
        "capitol",
        "capita",
        "capitan",
    }


def test_learning_still_uses_flyhash_codes_with_cosine_seeds(call):
    call("memory", "ingest", params={"triples": CORPUS})
    call("memory", "update", params={"query": "capital of france", "node": "París", "delta": 5.0})
    res = call("memory", "recall", params={"query": "capital of france", "k": 3})
    assert res["seeds"] == "cosine" and "París" in texts(res)


def test_cosine_fallback_without_extension_matches_vec(call, monkeypatch, db):
    call("memory", "ingest", params={"triples": CORPUS})
    memory.configure(db)
    with monkeypatch.context() as m:
        m.setattr(store_mod, "_load_vec", lambda conn: False)
        out = call(
            "memory",
            "ingest",
            params={"triples": [{"s": "Madrid", "r": "capital_de", "o": "España"}]},
        )
        assert out["status"] == "OK" and out["vec_index"] is False
        fallback = call("memory", "recall", params={"query": "madrid", "k": 8, "mode": "seeds"})[
            "results"
        ]
        ppr_fb = call("memory", "recall", params={"query": "madrid", "k": 4})["results"]
    memory.configure(db)
    assert memory._get_state().read.vec_available()
    vec = call("memory", "recall", params={"query": "madrid", "k": 8, "mode": "seeds"})["results"]
    assert len(vec) == 8 and texts({"results": vec}) == texts({"results": fallback})
    assert [r["similarity"] for r in vec] == pytest.approx(
        [r["similarity"] for r in fallback], abs=1e-5
    )
    ppr_vec = call("memory", "recall", params={"query": "madrid", "k": 4})["results"]
    assert texts({"results": ppr_vec}) == texts({"results": ppr_fb})


def test_v2_database_is_backfilled_on_open(call, db):
    _make_v2(db)
    memory.configure(db)
    st = memory._get_state()
    assert st.read.contract() == "v4" and st.read.counts()["dense_vecs"] == 3
    assert st.read.nodes_without_dense(10) == []
    res = call("memory", "recall", params={"query": "old1", "k": 1, "mode": "seeds"})
    assert texts(res) == ["old1"]
    conn = sqlite3.connect(db)
    assert conn.execute("SELECT COUNT(*) FROM kc_codes").fetchone()[0] == 3  # v2 data untouched
    assert conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0] == 1


def test_custom_embedding_is_the_stored_dense_vector(call):
    d = memory._get_state().params.d
    vec = [0.0] * d
    vec[5] = 3.0
    call(
        "memory",
        "ingest",
        params={"triples": [{"s": "a", "r": "r", "o": "b"}], "embeddings": {"a": vec}},
    )
    st = memory._get_state()
    node = st.read.get_node("a")
    top = st.read.nearest_dense(vec, 1)
    assert top[0][0] == node["id"] and top[0][1] == pytest.approx(1.0, abs=1e-5)


@pytest.mark.parametrize("backend", ["cosine", "flyhash"])
@pytest.mark.parametrize("mode", ["ppr", "seeds"])
@pytest.mark.parametrize("query", ["zorblax", "zorblax shared"])
def test_forget_leaks_nothing_through_any_backend(call, backend, mode, query):
    triples = [
        {"s": "zorblax", "r": "founded", "o": "shared", "source": "S1"},
        {"s": "shared", "r": "near", "o": "mitocondria", "source": "S2"},
    ]
    call("memory", "ingest", params={"triples": triples})
    call("memory", "forget", params={"source": "S1"})
    res = call("memory", "recall", params={"query": query, "k": 10, "mode": mode, "seeds": backend})
    assert "zorblax" not in texts(res)
    assert all("zorblax" not in fact for fact in res.get("facts", []))
    if query == "zorblax shared":  # the surviving knowledge is still reachable
        assert {"shared", "mitocondria"} <= set(texts(res))
