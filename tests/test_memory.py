import pytest

from chrysalis_mcp.domains import memory
from chrysalis_mcp.substrate import store as store_mod

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


def test_ingest_then_recall_seeds_semantic_sanity(call):
    out = call("memory", "ingest", params={"triples": CORPUS, "source": "t"})
    assert out["status"] == "OK"
    assert out["nodes_added"] == 10 and out["edges_added"] == 5 and out["codes_added"] == 10
    assert out["vec_index"] is True
    res = call(
        "memory", "recall", params={"query": "parís", "k": 3, "mode": "seeds", "seeds": "flyhash"}
    )
    top = [r["text"] for r in res["results"]]
    assert set(top[:2]) == {"París", "paris"}
    assert (
        res["results"][0]["overlap"] == res["results"][1]["overlap"] > res["results"][2]["overlap"]
    )
    assert res["mode"] == "seeds"
    default = call("memory", "recall", params={"query": "parís", "mode": "seeds"})
    assert default["seeds"] == "cosine" and len(default["results"]) == 5
    assert "similarity" in default["results"][0] and "overlap" not in default["results"][0]
    assert {r["text"] for r in default["results"][:2]} == {"París", "paris"}


def test_reingest_does_not_rehash_but_adds_edges(call):
    call("memory", "ingest", params={"triples": CORPUS})
    out = call("memory", "ingest", params={"triples": CORPUS[:2]})
    assert out["nodes_added"] == 0 and out["codes_added"] == 0 and out["edges_added"] == 2


def test_fallback_without_extension_same_results(call, monkeypatch, db):
    call("memory", "ingest", params={"triples": CORPUS})
    memory.configure(db)
    with monkeypatch.context() as m:
        m.setattr(store_mod, "_load_vec", lambda conn: False)
        new = [{"s": "Madrid", "r": "capital_de", "o": "España"}]
        out = call("memory", "ingest", params={"triples": new})
        assert out["status"] == "OK" and out["vec_index"] is False and out["codes_added"] == 2
        fallback = call("memory", "recall", params={"query": "paris", "k": 8, "mode": "seeds"})[
            "results"
        ]
    memory.configure(db)  # extension back: stale index is rebuilt, same data, vec path
    vec = call("memory", "recall", params={"query": "paris", "k": 8, "mode": "seeds"})["results"]
    assert memory._get_state().read.vec_available()
    assert len(vec) == 8 and vec == fallback


def test_custom_embeddings_and_dimension_check(call):
    d = memory._get_state().params.d
    vec = [0.0] * d
    vec[3] = 1.0
    ok = call(
        "memory",
        "ingest",
        params={"triples": [{"s": "a", "r": "r", "o": "b"}], "embeddings": {"a": vec}},
    )
    assert ok["status"] == "OK" and ok["codes_added"] == 2
    bad = call(
        "memory",
        "ingest",
        params={"triples": [{"s": "c", "r": "r", "o": "d"}], "embeddings": {"c": [1.0]}},
    )
    assert bad["status"] == "ERROR" and "dimension" in bad["error"]
    # validation precedes writes: nothing was stored for the rejected call
    assert memory._get_state().read.get_node("c") is None


@pytest.mark.parametrize(
    "triples, needle",
    [
        ([{"s": "a", "r": "r", "o": "b"}] * 501, "too many"),
        ([{"s": "x" * 513, "r": "r", "o": "b"}], "512"),
        ([{"s": "a", "r": "r" * 129, "o": "b"}], "128"),
        ([{"s": "a", "r": "r"}], "missing"),
        ([{"s": "a", "r": "r", "o": "b", "zzz": 1}], "unknown"),
        ([{"s": "a", "r": "r", "o": "b", "weight": float("nan")}], "weight"),
        ([{"s": "a", "r": "r", "o": "b", "weight": True}], "weight"),
        ([{"s": " ", "r": "r", "o": "b"}], "non-empty"),
        ([], "non-empty"),
        ("nope", "list"),
        ([42], "object"),
    ],
)
def test_ingest_validation_errors(call, triples, needle):
    out = call("memory", "ingest", params={"triples": triples})
    assert out["status"] == "ERROR" and needle in out["error"]
    assert memory._get_state().read.counts()["nodes"] == 0


def test_ingest_embedding_key_not_a_node(call):
    out = call(
        "memory",
        "ingest",
        params={"triples": [{"s": "a", "r": "r", "o": "b"}], "embeddings": {"zz": []}},
    )
    assert out["status"] == "ERROR" and "not a node" in out["error"]


def test_recall_errors(call):
    bad = call("memory", "recall", params={"query": "x", "mode": "bogus"})
    assert bad["status"] == "ERROR" and bad["valid"] == ["ppr", "seeds"]
    assert call("memory", "recall", params={"query": ""})["status"] == "ERROR"
    assert call("memory", "recall", params={"query": "x", "k": 0})["status"] == "ERROR"
    assert call("memory", "recall", params={"query": "x", "k": True})["status"] == "ERROR"
    assert call("memory", "recall", params={"query": "x", "k": 51})["status"] == "ERROR"
    assert call("memory", "recall", params={"query": "x"})["results"] == []  # empty DB


def test_terse_views(call):
    out = call("memory", "ingest", params={"triples": CORPUS}, verbosity="terse")
    assert out == {"status": "OK", "edges_added": 5} or out["edges_added"] == 5
    res = call("memory", "recall", params={"query": "paris", "mode": "seeds"}, verbosity="terse")
    assert res["results"][0] in ("París", "paris")


def test_batch_ingest_recall_with_cache(call):
    steps = [
        {
            "id": "i",
            "domain": "memory",
            "action": "ingest",
            "params": {"triples": CORPUS},
            "store_as": "ing",
        },
        {"id": "r", "domain": "memory", "action": "recall", "params": {"query": "paris", "k": 2}},
    ]
    out = call("meta", "batch", params={"steps": steps, "return_only": "last"})
    assert out["status"] == "OK" and out["steps_run"] == 2
    assert {x["text"] for x in out["results"]} == {"París", "paris"}


@pytest.mark.parametrize(
    "triple, needle",
    [
        ({"s": "a", "r": "r", "o": "b", "weight": 1e301}, "weight"),
        ({"s": "a", "r": "r", "o": "b", "weight": -1e301}, "weight"),
        ({"s": "a", "r": "r", "o": "b", "weight": 10**400}, "weight"),
        ({"s": "a", "r": "r", "o": "b", "weight": float("inf")}, "weight"),
        ({"s": "a\ud800", "r": "r", "o": "b"}, "surrogate"),
        ({"s": "a", "r": "r\udc00", "o": "b"}, "surrogate"),
        ({"s": "a", "r": "r", "o": "b", "source": "x\ud800"}, "surrogate"),
    ],
)
def test_ingest_rejects_hostile_scalars_with_structured_error(call, triple, needle):
    out = call("memory", "ingest", params={"triples": [triple]})
    assert out["status"] == "ERROR" and needle in out["error"]
    assert "Error:" not in out["error"]  # not the generic handler's "ExcType: msg"
    assert memory._get_state().read.counts()["nodes"] == 0


def test_boundary_weight_accepted(call):
    out = call(
        "memory", "ingest", params={"triples": [{"s": "a", "r": "r", "o": "b", "weight": 1e300}]}
    )
    assert out["status"] == "OK"


def test_recall_rejects_surrogate_query(call):
    out = call("memory", "recall", params={"query": "x\ud800"})
    assert out["status"] == "ERROR" and "surrogate" in out["error"]


def test_embedding_value_hardening(call):
    d = memory._get_state().params.d

    def ingest(vec):
        return call(
            "memory",
            "ingest",
            params={"triples": [{"s": "a", "r": "r", "o": "b"}], "embeddings": {"a": vec}},
        )

    for bad in (
        [float("nan")] + [0.0] * (d - 1),
        [float("inf")] + [0.0] * (d - 1),
        [10**400] + [0] * (d - 1),
    ):
        out = ingest(bad)
        assert out["status"] == "ERROR" and "finite" in out["error"], out
    assert ingest([0.0] * d)["error"].endswith("non-zero")
    assert memory._get_state().read.counts()["nodes"] == 0
    huge = [1e300] * d  # squared norm would overflow float64
    out = ingest(huge)
    assert out["status"] == "OK" and out["codes_added"] == 2


def test_embedding_scale_invariant_code(call):
    d = memory._get_state().params.d
    import random

    rng = random.Random(5)
    base = [rng.uniform(0.5, 2.0) for _ in range(d)]  # distinct values: no exact-tie rounding
    for name, scale in (("x", 1.0), ("y", 1e100)):
        call(
            "memory",
            "ingest",
            params={
                "triples": [{"s": name, "r": "r", "o": f"o{name}"}],
                "embeddings": {name: [v * scale for v in base]},
            },
        )
    read = memory._get_state().read
    cx = read.kc_code(read.get_node("x")["id"])["active"]
    cy = read.kc_code(read.get_node("y")["id"])["active"]
    assert cx == cy


def test_embeddings_ignored_reported(call):
    d = memory._get_state().params.d
    vec = [1.0] + [0.0] * (d - 1)
    call("memory", "ingest", params={"triples": [{"s": "a", "r": "r", "o": "b"}]})
    out = call(
        "memory",
        "ingest",
        params={
            "triples": [{"s": "a", "r": "r", "o": "c"}],
            "embeddings": {"a": vec, "c": vec},
        },
    )
    assert out["embeddings_ignored"] == ["a"] and out["codes_added"] == 1


def test_ingest_makes_exactly_one_write_call(call, monkeypatch):
    from chrysalis_mcp.substrate.write import SqliteWrite

    calls = []
    for name in (
        "upsert_node",
        "add_triple",
        "put_kc_code",
        "bump_assoc",
        "invalidate",
        "ingest_batch",
    ):
        orig = getattr(SqliteWrite, name)

        def wrap(self, *a, _o=orig, _n=name, **k):
            calls.append(_n)
            return _o(self, *a, **k)

        monkeypatch.setattr(SqliteWrite, name, wrap)
    out = call("memory", "ingest", params={"triples": CORPUS})
    assert out["status"] == "OK" and calls == ["ingest_batch"]


def test_ingest_failure_is_atomic_and_retry_has_single_edges(call, monkeypatch):
    from chrysalis_mcp.algorithms import FlyHash

    state = memory._get_state()
    real = FlyHash.hash
    seen = []

    def flaky(self, x):
        seen.append(1)
        if len(seen) == 4:
            raise RuntimeError("boom")
        return real(self, x)

    monkeypatch.setattr(FlyHash, "hash", flaky)
    out = call("memory", "ingest", params={"triples": CORPUS})
    assert out["status"] == "ERROR"
    counts = state.read.counts()
    assert counts["nodes"] == counts["edges"] == counts["kc_codes"] == counts["log"] == 0
    monkeypatch.undo()
    assert call("memory", "ingest", params={"triples": CORPUS})["status"] == "OK"
    assert state.read.counts()["edges"] == len(CORPUS)
