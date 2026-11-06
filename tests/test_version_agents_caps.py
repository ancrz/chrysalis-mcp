"""T7b: version stamp (L1), per-agent accounting (L2), brief/terse caps (L4)."""

import json

import pytest

from chrysalis_mcp.core import version
from chrysalis_mcp.core.envelope import Telemetry, build_response
from chrysalis_mcp.domains import memory
from chrysalis_mcp.server import handle_call


def test_real_run_not_stale(call):
    out = call("meta", "discover")
    assert out["stale"] is False and out["status"] == "OK" and out["severity"] == "INFO"
    for key in ("server_version", "loaded_commit", "repo_head", "source_hash", "schema_hash"):
        assert out[key]
    assert out["pid"] > 0 and out["uptime_s"] >= 0 and out["process_start_time"]
    terse = call("meta", "discover", verbosity="terse")
    assert {"loaded_commit", "stale", "schema_hash"} <= set(terse)


def test_stale_when_head_moves(call, monkeypatch):
    monkeypatch.setattr(version, "LOADED_COMMIT", "aaaaaaa")
    monkeypatch.setattr(version, "repo_head", lambda: "bbbbbbb")
    out = call("meta", "discover")
    assert out["stale"] is True and out["severity"] == "WARN"
    assert "restart chrysalis-mcp" in out["next_action"]


def test_stale_when_source_changes(call, monkeypatch):
    monkeypatch.setattr(version, "source_hash_now", lambda: "different")
    assert call("meta", "discover")["stale"] is True


def test_unknown_head_is_not_stale_by_commit(call, monkeypatch):
    monkeypatch.setattr(version, "LOADED_COMMIT", version.UNKNOWN)
    monkeypatch.setattr(version, "repo_head", lambda: "bbbbbbb")
    assert call("meta", "discover")["stale"] is False


def test_loaded_commit_captured_at_import_not_per_call(call, monkeypatch):
    before = call("meta", "discover")["loaded_commit"]
    monkeypatch.setattr(version, "_git_head", lambda: "ffffff0")
    after = call("meta", "discover")
    assert after["loaded_commit"] == before and after["repo_head"] in ("ffffff0", before)


def test_schema_hash_changes_with_registration(registry, cache):
    def h():
        return json.loads(handle_call(registry, cache, "meta", "discover"))["schema_hash"]

    first = h()
    registry.register("dummy", "noop", lambda p, c: {"status": "OK"})
    assert h() != first


def test_agent_accounting_both_groupings(registry, cache):
    def run(action, agent, **kw):
        return json.loads(handle_call(registry, cache, "meta", action, agent_id=agent, **kw))

    run("schema", "alice")
    run("schema", "alice")
    run("discover", "bob")
    run("nope", "bob")
    by_agent = run("stats", "alice", params={"group_by": "agent", "scope": "all"})["agents"]
    assert by_agent["alice"]["calls"] == 3 and by_agent["bob"]["calls"] == 1
    assert by_agent["alice"]["est_tokens"] > 0 and by_agent["alice"]["total_ms"] >= 0
    by_action = run("stats", "alice", params={"scope": "all", "agent_id": "bob"})["actions"]
    assert set(by_action) == {"meta.discover"} and by_action["meta.discover"]["errors"] == 0


def test_errors_counted(registry, cache):
    handle_call(registry, cache, "meta", "stats", params={"scope": "bad"}, agent_id="e")
    rows = json.loads(
        handle_call(registry, cache, "meta", "stats", params={"group_by": "agent"}, agent_id="x")
    )["agents"]
    assert rows["e"]["errors"] == 1


@pytest.mark.parametrize("bad", ["", "a b", "x" * 65, "a/b", "ok\n"])
def test_invalid_agent_id_is_error(registry, cache, bad):
    out = json.loads(handle_call(registry, cache, "meta", "schema", agent_id=bad))
    assert out["status"] == "ERROR" and "agent_id" in out["error"]


def test_batch_inner_steps_attributed_to_caller(registry, cache):
    steps = [{"domain": "meta", "action": "schema"}, {"domain": "meta", "action": "discover"}]
    handle_call(registry, cache, "meta", "batch", params={"steps": steps}, agent_id="carol")
    rows = registry.stats(None, group_by="agent")
    assert rows["carol"]["calls"] == 3 and "anonymous" not in rows


def test_brief_terse_nested_caps():
    payload = {"status": "OK", "r": [{"t": "x" * 1000, "l": list(range(50))}], "error": "e" * 500}
    brief = json.loads(build_response(payload, Telemetry("d", "a"), verbosity="brief"))
    assert len(brief["r"][0]["t"]) <= 259 and len(brief["r"][0]["l"]) == 10
    assert brief["truncated"] is True and len(brief["error"]) == 500
    payload["terse"] = {"r": [{"t": "x" * 1000, "l": list(range(50))}] * 9}
    terse = json.loads(build_response(payload, Telemetry("d", "a"), verbosity="terse"))
    assert len(terse["r"]) == 5 and len(terse["r"][0]["t"]) <= 67 and len(terse["r"][0]["l"]) == 5
    assert terse["truncated"] is True


@pytest.fixture
def big_db(tmp_path):
    memory.configure(tmp_path / "big.db")
    yield
    memory.configure(None)


def test_every_action_brief_terse_within_budget(registry, cache, big_db):
    triples = [{"s": f"node {i} " + "w" * 200, "r": "rel", "o": f"obj {i}"} for i in range(300)]
    handle_call(registry, cache, "memory", "ingest", params={"triples": triples, "source": "big"})
    budgets = {"brief": 8_000, "terse": 1_500}
    for domain, actions in registry.schema()["domains"].items():
        for action, spec in actions.items():
            if (domain, action) == ("meta", "batch"):
                continue  # example batch runs meta steps; covered via its own projection test
            for verbosity, budget in budgets.items():
                text = handle_call(
                    registry,
                    cache,
                    domain,
                    action,
                    params=spec["example"]["params"],
                    verbosity=verbosity,
                )
                assert len(text.encode()) <= budget, (domain, action, verbosity, len(text))
