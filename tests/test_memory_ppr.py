import numpy as np
import pytest

from chrysalis_mcp.algorithms import personalized_pagerank
from chrysalis_mcp.domains import memory

CHAIN = [
    {"s": "zorblax", "r": "founded", "o": "quuxcorp", "source": "a"},
    {"s": "quuxcorp", "r": "located_in", "o": "wibbleton", "source": "a"},
    {"s": "mitocondria", "r": "produce", "o": "ATP", "source": "b"},
]


@pytest.fixture(autouse=True)
def db(tmp_path):
    memory.configure(tmp_path / "mem.db")
    yield
    memory.configure(None)


def _texts(res):
    return [r["text"] for r in res["results"]]


def test_ppr_matches_closed_form_with_dangling_node():
    edges = [(0, 1, 2.0), (1, 2, 1.0), (0, 2, 0.5)]  # node 3 is dangling
    s = np.array([1.0, 0.0, 0.0, 1.0])
    p = personalized_pagerank(4, edges, s, tol=1e-12, max_iter=500)
    a = np.zeros((4, 4))
    for u, v, w in edges:
        a[u, v] += w
        a[v, u] += w
    sn = s / s.sum()
    for i in range(4):
        a[i] = a[i] / a[i].sum() if a[i].sum() else sn  # dangling row teleports to s
    alpha = 0.5
    expected = alpha * np.linalg.solve(np.eye(4) - (1 - alpha) * a.T, sn)
    assert np.allclose(p, expected, atol=1e-6) and abs(p.sum() - 1) < 1e-6
    assert personalized_pagerank(0, [], np.zeros(0)).size == 0
    assert not personalized_pagerank(2, [], np.zeros(2)).any()


def test_multi_hop_ppr_beats_seeds(call):
    call("memory", "ingest", params={"triples": CHAIN})
    q = {"query": "zorblax", "k": 3}
    assert "wibbleton" not in _texts(call("memory", "recall", params={**q, "mode": "seeds"}))[:2]
    res = call("memory", "recall", params={**q, "hops": 2})
    assert res["mode"] == "ppr" and "wibbleton" in _texts(res)
    assert ["quuxcorp", "located_in", "wibbleton"] in res["facts"]
    assert call("memory", "recall", params=q, verbosity="terse")["results"][0] == "zorblax"


def test_empty_graph_ok(call):
    res = call("memory", "recall", params={"query": "anything"})
    assert res["status"] == "OK" and res["results"] == [] and res["facts"] == []


def test_update_boosts_node_and_touches_only_active_kcs(call):
    call("memory", "ingest", params={"triples": CHAIN})
    before = _texts(call("memory", "recall", params={"query": "zorblax", "k": 6}))
    out = call("memory", "update", params={"query": "zorblax", "node": "ATP", "delta": 5})
    assert out["status"] == "OK" and out["kcs_reinforced"] == memory._get_state().params.k
    st = memory._get_state()
    assert st.read.counts()["assoc"] == out["kcs_reinforced"]
    after = _texts(call("memory", "recall", params={"query": "zorblax", "k": 6}))
    assert "ATP" in after and (
        after.index("ATP") < before.index("ATP") if "ATP" in before else True
    )
    assert st.read.log_tail(1)[0]["op"] == "reinforce"
    assert call("memory", "update", params={"query": "x", "node": "nope"})["status"] == "ERROR"
    assert (
        call("memory", "update", params={"query": "x", "node": "ATP", "delta": 101})["status"]
        == "ERROR"
    )


def test_forget_scopes_to_source_and_logs_edge_ids(call):
    call("memory", "ingest", params={"triples": CHAIN})
    out = call("memory", "forget", params={"source": "a"})
    assert out["invalidated"] == 2 and len(out["edge_ids"]) == 2
    log = memory._get_state().read.log_tail(1)[0]
    assert log["payload"]["edge_ids"] == out["edge_ids"]
    res = call("memory", "recall", params={"query": "zorblax", "k": 6})
    assert ["zorblax", "founded", "quuxcorp"] not in res["facts"]
    assert not any(f[1] in ("founded", "located_in") for f in res["facts"])
    other = call("memory", "recall", params={"query": "mitocondria", "k": 3})
    assert ["mitocondria", "produce", "ATP"] in other["facts"]
    assert call("memory", "forget", params={"source": "ghost"})["invalidated"] == 0


def test_consolidate_decays_prunes_and_log_only_grows(call):
    call("memory", "ingest", params={"triples": CHAIN})
    call("memory", "update", params={"query": "zorblax", "node": "ATP", "delta": 1})
    st = memory._get_state()
    n_log = st.read.counts()["log"]
    out = call("memory", "consolidate", params={"decay": 0.5, "prune_below": 0.1})
    assert out["decayed"] == st.params.k and out["pruned"] == 0
    out = call("memory", "consolidate", params={"decay": 0.1, "prune_below": 0.1})
    assert out["pruned"] == st.params.k and st.read.counts()["assoc"] == 0
    assert st.read.counts()["log"] == n_log + 2


@pytest.mark.parametrize(
    ("action", "params"),
    [
        ("recall", {"query": "x", "hops": 4}),
        ("recall", {"query": "x", "max_nodes": 20001}),
        ("recall", {"query": "x", "k": 51}),
        ("recall", {"query": "x", "n_seeds": 51}),
        ("recall", {"query": "x", "lambda": 11}),
        ("recall", {"query": "x", "lambda": -1}),
        ("consolidate", {"decay": 0}),
        ("consolidate", {"decay": 1.5}),
        ("consolidate", {"prune_below": -1}),
    ],
)
def test_bounds_error(call, action, params):
    assert call("memory", action, params=params)["status"] == "ERROR"


def test_atomicity_of_reinforce_and_decay(call, monkeypatch):
    call("memory", "ingest", params={"triples": CHAIN})
    call("memory", "update", params={"query": "zorblax", "node": "ATP"})
    st = memory._get_state()
    snap = (st.read.assoc_for(range(st.params.m)), st.read.counts()["log"])

    with pytest.raises(ValueError):
        st.write.reinforce([1, 2], 10**9, 1.0)  # unknown node -> rolled back

    class Boom(Exception):
        pass

    orig = st.write._db.transaction

    def failing():
        ctx = orig()

        class _Ctx:
            def __enter__(self_inner):
                conn = ctx.__enter__()
                real_exec = conn

                class _Proxy:
                    def execute(self_p, sql, *a):
                        if sql.startswith("DELETE FROM assoc"):
                            raise Boom
                        return real_exec.execute(sql, *a)

                return _Proxy()

            def __exit__(self_inner, *exc):
                return ctx.__exit__(*exc)

        return _Ctx()

    monkeypatch.setattr(st.write._db, "transaction", failing)
    with pytest.raises(Boom):
        st.write.decay_assoc(0.5, 0.1)
    monkeypatch.undo()
    assert (st.read.assoc_for(range(st.params.m)), st.read.counts()["log"]) == snap


def test_forget_leaves_no_knowledge_leak(call):
    triples = [
        {"s": "zorblax", "r": "founded", "o": "shared", "source": "S1"},
        {"s": "shared", "r": "near", "o": "mitocondria", "source": "S2"},
    ]
    call("memory", "ingest", params={"triples": triples})
    call("memory", "forget", params={"source": "S1"})
    for mode in ("ppr", "seeds"):
        res = call("memory", "recall", params={"query": "zorblax shared", "k": 10, "mode": mode})
        assert "zorblax" not in _texts(res) and {"shared", "mitocondria"} <= set(_texts(res))
    assert memory._get_state().read.get_node("zorblax") is not None  # row kept for audit
