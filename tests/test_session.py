from chrysalis_mcp.core.session import MISSING, SessionCache


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def test_ttl_expiry():
    clock = Clock()
    cache = SessionCache(ttl_sec=10, clock=clock)
    cache.put("a", "k", {"v": 1})
    clock.now = 9.9
    assert cache.get("a", "k") == {"v": 1}
    clock.now = 10.0
    assert cache.get("a", "k") is MISSING
    assert len(cache) == 0


def test_session_isolation():
    cache = SessionCache()
    cache.put("A", "k", 1)
    assert cache.get("B", "k") is MISSING
    assert cache.get("A", "k") == 1


def test_max_entries_evicts_oldest():
    clock = Clock()
    cache = SessionCache(max_entries=2, clock=clock)
    for i, key in enumerate("abc"):
        clock.now = i
        cache.put("s", key, i)
    assert cache.get("s", "a") is MISSING
    assert cache.get("s", "c") == 2


def test_values_are_copied():
    cache = SessionCache()
    value = {"x": [1]}
    cache.put("s", "k", value)
    value["x"].append(2)
    assert cache.get("s", "k") == {"x": [1]}


def test_store_as_via_call_is_session_scoped(call):
    call("meta", "discover", session_id="A", store_as="d")
    out = call(
        "meta",
        "batch",
        session_id="B",
        params={
            "steps": [{"domain": "meta", "action": "stats", "params": {"scope": "$cache.d.x"}}]
        },
    )
    assert out["status"] == "ERROR"
    assert "unresolved" in out["failed_steps"][0]["error"]
    ok = call(
        "meta",
        "batch",
        session_id="A",
        params={
            "steps": [{"domain": "meta", "action": "stats", "params": {"scope": "$cache.d.status"}}]
        },
    )
    # status "OK" is not a valid scope: resolves, then the handler rejects it (not unresolved).
    assert "unresolved" not in str(ok["failed_steps"])
