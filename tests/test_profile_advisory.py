import ast
import json
import pathlib

import pytest

from chrysalis_mcp.core import profile as prof
from chrysalis_mcp.core.advisory import Advisor, load_evidence, validate_entry
from chrysalis_mcp.core.profile import Profile, ProfileStore, classify, parse_forced
from chrysalis_mcp.core.registry import Registry
from chrysalis_mcp.core.session import SessionCache
from chrysalis_mcp.domains import meta
from chrysalis_mcp.server import handle_call

NONCE_A, NONCE_B = "nonce-a", "nonce-b"
SMALL_TASK = {"constraints": {"vars": {"x": "int", "y": "int"}, "assert": ["x + y == 10", "x > y"]}}
BIG_TASK = {
    "constraints": {
        "vars": {"x": "int", "y": "int"},
        "assert": [f"x >= {i}" for i in range(4)] + [f"y <= {i}" for i in range(4)],
    }
}
OPT_TASK = {"constraints": {**SMALL_TASK["constraints"], "goal": "max:x"}}


def plain(nonce):
    """Plain expected answers for a nonce (test-only: the server never exposes them)."""
    return {pid: list(ans) for pid, _, _, ans in prof._build(nonce)}


GOOD = plain(NONCE_A)
ISSUED_A = prof.issue_probes(NONCE_A)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv(prof.ENV_PROFILE, raising=False)
    monkeypatch.delenv("CHRYSALIS_ENFORCE_ADVISORY", raising=False)


@pytest.fixture
def world():
    registry = Registry()
    meta.register(registry)
    for domain, action in (("reason", "solve"), ("reason", "check"), ("memory", "recall")):
        registry.register(
            domain,
            action,
            lambda p, c: {"status": "OK", "ran": True},
            params_schema={
                "seeds": {"required": False},
                "constraints": {"required": False},
                "smt2": {"required": False},
            },
        )
    cache = SessionCache()

    def call(domain, action, params=None, session_id="s1", verbosity="detail"):
        text = handle_call(
            registry, cache, domain, action, params, verbosity=verbosity, session_id=session_id
        )
        return json.loads(text)

    return registry, call


@pytest.mark.parametrize(
    ("size", "expected"),
    [
        (None, "unknown"),
        (0, "unknown"),
        (4, "small"),
        (7.99, "small"),
        (8, "medium"),
        (30, "medium"),
        (30.1, "large"),
        (70, "large"),
    ],
)
def test_class_derivation(size, expected):
    assert classify(size) == expected


def test_parse_forced_forms_and_errors():
    p = parse_forced("gemma:4:thinking")
    assert (p.family, p.params_b, p.thinking, p.model_class, p.source) == (
        "gemma",
        4.0,
        True,
        "small",
        "forced",
    )
    assert parse_forced("x:70:off").thinking is False
    assert parse_forced('{"params_b": 12}').model_class == "medium"
    for bad in ("gemma", "a:b:c:d", "a:-1", "a:4:maybe", "[1]", '{"thinking": "yes"}', "{nope"):
        with pytest.raises(ValueError):
            parse_forced(bad)


def test_precedence_forced_over_declared_over_measured(monkeypatch):
    store = ProfileStore()
    assert store.resolve("s").source == "unknown"
    store.record_scores("s", {"reason": 0.5})
    assert store.resolve("s").source == "measured"
    assert store.resolve("s").model_class == "unknown"
    store.declare("s", Profile(source="declared", model="m", params_b=70))
    resolved = store.resolve("s")
    assert (resolved.source, resolved.model_class, resolved.scores) == (
        "declared",
        "large",
        {"reason": 0.5},
    )
    monkeypatch.setenv(prof.ENV_PROFILE, "gemma:4")
    forced = store.resolve("s")
    assert (forced.source, forced.model_class, forced.scores) == (
        "forced",
        "small",
        {"reason": 0.5},
    )
    monkeypatch.setenv(prof.ENV_PROFILE, "garbage")
    fallback = store.resolve("s")
    assert fallback.source == "declared" and "ignored" in (fallback.note or "")


def test_sessions_are_isolated_and_scores_follow_model_id():
    store = ProfileStore()
    store.declare("a", Profile(source="declared", model="m1", params_b=4))
    store.record_scores("a", {"memory": 1.0})
    assert store.resolve("b").scores == {}
    store.declare("c", Profile(source="declared", model="m1", params_b=4))
    assert store.resolve("c").scores == {"memory": 1.0}


def test_calibration_scoring_right_wrong_malformed():
    assert prof.score(GOOD, ISSUED_A)[0] == {"memory": 1.0, "reason": 1.0}
    scores, unknown = prof.score(
        {"reason-1": GOOD["reason-1"], "reason-2": ["x>y"], "reason-3": {"a": 1}, "zzz": "q"},
        ISSUED_A,
    )
    assert scores == {"reason": 0.3333}
    assert unknown == ["zzz"]
    assert prof.score({"memory-1": "n1", "memory-2": ""}, ISSUED_A)[0] == {"memory": 0.0}
    assert prof.score({"memory-1": [True]}, ISSUED_A)[0] == {"memory": 0.0}
    assert prof.score({"memory-1": list(range(40))}, ISSUED_A)[0] == {"memory": 0.0}
    upper = json.dumps([GOOD["memory-1"][0].upper()])
    assert prof.score({"memory-1": upper}, ISSUED_A)[0] == {"memory": 0.5}


def test_probes_never_expose_expected_answers():
    blob = json.dumps(prof.issue_probes(NONCE_A).public)
    assert all(d not in blob for _, d in ISSUED_A.expected.values())
    assert {p["family"] for p in ISSUED_A.public} == {"reason", "memory"}


def _squash(text):
    return "".join(text.split()).lower()


def _nonces(monkeypatch, *values):
    it = iter(values)
    monkeypatch.setattr(prof.secrets, "token_hex", lambda _n: next(it))


def test_no_plain_answer_in_schema_discover_or_issue(world, monkeypatch):
    _, call = world
    _nonces(monkeypatch, NONCE_A)
    blobs = []
    for verbosity in ("detail", "summary", "brief", "terse"):
        blobs.append(call("meta", "schema", verbosity=verbosity))
        blobs.append(call("meta", "discover", verbosity=verbosity))
    blobs.append(call("meta", "calibrate"))  # issues with NONCE_A
    text = _squash(json.dumps(blobs))
    items = {i for answers in GOOD.values() for i in answers}
    items |= {"|".join(a) for a in GOOD.values()}
    for item in items:
        assert _squash(item) not in text, item
    for fixed in ("x+3==10", "n7", "min:x"):
        assert fixed not in text


def test_replay_across_sessions_scores_below_one(world, monkeypatch):
    _, call = world
    _nonces(monkeypatch, NONCE_A, NONCE_B)
    prompts_a = call("meta", "calibrate", session_id="A")["probes"]
    prompts_b = call("meta", "calibrate", session_id="B")["probes"]
    assert prompts_a != prompts_b
    own = call("meta", "calibrate", {"answers": GOOD}, session_id="A")
    assert own["scores"] == {"memory": 1.0, "reason": 1.0}
    replay = call("meta", "calibrate", {"answers": GOOD}, session_id="B")
    assert min(replay["scores"].values()) < 1.0
    assert call("meta", "calibrate", {"answers": GOOD}, session_id="never")["status"] == "ERROR"


def test_evidence_json_schema():
    entries = load_evidence()
    assert entries and all(validate_entry(e) == [] for e in entries)
    assert validate_entry({"action": "x", "level": "wow"})
    assert validate_entry("nope") == ["entry must be an object"]


def test_advisory_lookup_from_evidence():
    adv = Advisor.default()
    small = Profile(params_b=4, thinking=False)
    assert adv.lookup("memory.recall", small)["level"] == "recommended"
    assert adv.lookup("reason.solve", small)["level"] == "neutral"  # no task -> no static verdict
    small_task = {"n_atoms": 2, "goal": "sat"}
    got = adv.lookup("reason.solve", small, task=small_task)
    assert got["level"] == "discouraged" and "0.8 -> 0.6" in got["why"]
    on = Profile(params_b=4, thinking=True)
    assert adv.lookup("reason.solve", on, task=small_task)["level"] == "neutral"  # within noise
    assert (
        adv.lookup("reason.solve", Profile(params_b=4), task=small_task)["level"] == "discouraged"
    )
    assert adv.lookup("reason.solve", Profile(params_b=70), task=small_task)["level"] == "neutral"
    assert adv.lookup("reason.solve", Profile(), task=small_task)["level"] == "neutral"
    flyhash = adv.lookup("memory.recall", Profile(), {"seeds": "flyhash"})
    assert flyhash["level"] == "experimental"
    assert adv.lookup("memory.recall", Profile(), {"seeds": "cosine"})["level"] == "recommended"


def test_measured_calibration_overrides_class_defaults():
    adv = Advisor.default()
    strong = Profile(params_b=4, thinking=False, scores={"reason": 0.85})
    assert adv.lookup("reason.solve", strong)["level"] == "recommended"
    weak = Profile(scores={"memory": 0.4})
    assert adv.lookup("memory.recall", weak)["level"] == "discouraged"
    middle = Profile(params_b=4, scores={"reason": 0.6})
    assert adv.lookup("reason.solve", middle)["level"] == "neutral"  # no evidence, no score verdict
    # task-matched evidence outranks calibration (it measures this very kind of task)
    task = {"n_atoms": 2, "goal": "sat"}
    assert adv.lookup("reason.solve", strong, task=task)["level"] == "discouraged"
    experimental = Profile(scores={"memory": 1.0})
    assert (
        adv.lookup("memory.recall", experimental, {"seeds": "flyhash"})["level"] == "experimental"
    )


def test_handshake_validation_and_discover_annotation(world):
    _, call = world
    assert call("meta", "handshake", {})["status"] == "ERROR"
    assert call("meta", "handshake", {"model": "m", "params_b": -3})["status"] == "ERROR"
    assert call("meta", "handshake", {"model": "m", "thinking": "x"})["status"] == "ERROR"
    out = call("meta", "handshake", {"model": "m", "params_b": 4, "client": "claude-code/1"})
    assert out["profile"]["class"] == "small" and out["profile"]["client"] == "claude-code/1"
    disc = call("meta", "discover")
    assert "advisory" not in disc["domains"]["reason"]["solve"]  # task-dependent: see rules
    assert "dis:n_atoms<=6&goal=sat" in disc["rules"]["reason.check,reason.solve"]
    assert disc["domains"]["memory"]["recall"]["advisory_variants"][0]["when"] == {
        "seeds": "flyhash"
    }
    assert "advisory" not in disc["domains"]["meta"]["stats"]
    assert "rec:n_atoms>=7|goal=opt" in call("meta", "schema")["rules"]["reason.check,reason.solve"]
    other = call("meta", "discover", session_id="other")
    assert "advisory" not in other["domains"]["reason"]["solve"]  # absent == neutral


def test_terse_lists_only_non_neutral(world):
    _, call = world
    call("meta", "handshake", {"params_b": 4})
    terse = call("meta", "discover", verbosity="terse")
    assert terse["advisories"]["memory.recall"] == "recommended"
    assert "reason.solve" not in terse["advisories"]  # conditional: carried by rules
    assert terse["rules"]["reason.check,reason.solve"].startswith("dis:")
    assert all(level != "neutral" for level in terse["advisories"].values())


def test_dispatch_annotates_but_does_not_block_by_default(world):
    _, call = world
    call("meta", "handshake", {"params_b": 4})
    out = call("reason", "solve", SMALL_TASK)
    assert out["ran"] is True and out["advisory"]["level"] == "discouraged"
    assert "advisory" not in call("reason", "solve", BIG_TASK)  # recommended is not noise
    assert "advisory" not in call("reason", "solve", OPT_TASK)
    ok = call("memory", "recall")
    assert "advisory" not in ok  # recommended is not noise on every response
    fly = call("memory", "recall", {"seeds": "flyhash"})
    assert fly["advisory"]["level"] == "experimental"


def test_enforce_blocks_discouraged_unless_override(world, monkeypatch):
    _, call = world
    call("meta", "handshake", {"params_b": 4})
    monkeypatch.setenv("CHRYSALIS_ENFORCE_ADVISORY", "1")
    blocked = call("reason", "solve", SMALL_TASK)
    assert blocked["status"] == "DISCOURAGED" and blocked["severity"] == "WARN"
    assert "ran" not in blocked and blocked["advisory"]["source"]
    assert call("reason", "solve", {**SMALL_TASK, "override": True})["ran"] is True
    assert call("reason", "solve", {**SMALL_TASK, "override": "yes"})["status"] == "DISCOURAGED"
    assert call("reason", "solve", BIG_TASK)["ran"] is True  # enforcement follows the task
    assert call("memory", "recall")["ran"] is True
    assert call("reason", "solve", SMALL_TASK, session_id="unprofiled")["ran"] is True
    monkeypatch.delenv("CHRYSALIS_ENFORCE_ADVISORY")
    assert call("reason", "solve", SMALL_TASK)["ran"] is True


def test_calibrate_flow_changes_advisory(world, monkeypatch):
    _, call = world
    _nonces(monkeypatch, NONCE_A)
    call("meta", "handshake", {"params_b": 4, "thinking": False})
    probes = call("meta", "calibrate")["probes"]
    assert len(probes) == 5 and "expected" not in probes[0]
    assert call("meta", "calibrate", {"answers": {}})["status"] == "ERROR"
    assert call("meta", "calibrate", {"answers": {"zzz": 1}})["status"] == "ERROR"
    answers = {k: v for k, v in GOOD.items() if k.startswith("reason")}
    out = call("meta", "calibrate", {"answers": answers})
    assert out["scores"] == {"reason": 1.0}
    solve = call("meta", "discover")["domains"]["reason"]["solve"]
    assert solve["advisory"]["level"] == "recommended"
    assert "advisory" not in call("reason", "solve")


def test_forced_env_wins_in_dispatch(world, monkeypatch):
    _, call = world
    call("meta", "handshake", {"params_b": 70})
    assert "advisory" not in call("reason", "solve", SMALL_TASK)
    monkeypatch.setenv(prof.ENV_PROFILE, '{"params_b": 3, "thinking": false}')
    assert call("reason", "solve", SMALL_TASK)["advisory"]["level"] == "discouraged"


def test_core_does_not_import_validation():
    root = pathlib.Path(prof.__file__).parent
    for name in ("profile.py", "advisory.py", "registry.py"):
        tree = ast.parse((root / name).read_text())
        for node in ast.walk(tree):
            mods = []
            if isinstance(node, ast.Import):
                mods = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                mods = [node.module or ""]
            assert not any(m.split(".")[0] == "validation" for m in mods), name


# --- evidence fail-soft ------------------------------------------------------------------------


def _evidence_file(monkeypatch, tmp_path, text):
    from chrysalis_mcp.core import advisory as adv_mod

    root = tmp_path / "chrysalis_mcp"
    (root / "data").mkdir(parents=True)
    (root / "data" / "evidence.json").write_text(text, "utf-8")
    monkeypatch.setattr(adv_mod.resources, "files", lambda _pkg: root)


def _registry():
    registry = Registry()
    meta.register(registry)
    registry.register("reason", "solve", lambda p, c: {"status": "OK"})
    return registry


def test_unparseable_evidence_is_neutral_with_visible_load_error(monkeypatch, tmp_path):
    _evidence_file(monkeypatch, tmp_path, "{not json")
    registry = _registry()
    assert registry.advisor.evidence == [] and registry.advisor.load_error
    call = _caller(registry)
    call("meta", "handshake", {"params_b": 4})
    disc = call("meta", "discover")
    assert "advisory" not in disc["domains"]["reason"]["solve"]
    assert "evidence unavailable" in disc["profile"]["load_error"]
    assert "load_error" in call("meta", "discover", verbosity="terse")
    assert "load_error" in call("meta", "schema")


def test_invalid_entries_dropped_valid_kept(monkeypatch, tmp_path):
    good = {
        "action": "reason.solve", "model_class": "small", "level": "discouraged",
        "metric": "m", "value": "v", "source": "s", "date": "2026-11-05",
    }  # fmt: skip
    _evidence_file(monkeypatch, tmp_path, json.dumps({"entries": [good, {"action": "x"}, 5]}))
    registry = _registry()
    assert registry.advisor.evidence == [good]
    assert "2 invalid entries dropped" in registry.advisor.load_error
    call = _caller(registry)
    call("meta", "handshake", {"params_b": 4})
    disc = call("meta", "discover")
    assert disc["domains"]["reason"]["solve"]["advisory"]["level"] == "discouraged"
    assert disc["profile"]["load_error"]


def test_missing_entries_key_and_clean_load(monkeypatch, tmp_path):
    _evidence_file(monkeypatch, tmp_path, "{}")
    assert _registry().advisor.load_error
    assert Advisor.default().load_error  # still patched: KeyError path, no raise
    monkeypatch.undo()
    assert Advisor.default().load_error is None


def _caller(registry):
    cache = SessionCache()

    def call(domain, action, params=None, session_id="s1", verbosity="detail"):
        return json.loads(
            handle_call(
                registry, cache, domain, action, params, verbosity=verbosity, session_id=session_id
            )
        )

    return call


# --- store LRU / batch -------------------------------------------------------------------------


def test_active_session_survives_flood():
    store = ProfileStore()
    store.declare("hot", Profile(source="declared", model="m", params_b=4))
    for i in range(prof.MAX_STORED_SESSIONS * 2):
        store.declare(f"cold-{i}", Profile(source="declared", params_b=4))
        assert store.resolve("hot").model == "m"  # each read refreshes recency
    assert len(store.declared) <= prof.MAX_STORED_SESSIONS
    assert "cold-0" not in store.declared


def test_batch_treats_discouraged_as_failure(world, monkeypatch):
    _, call = world
    call("meta", "handshake", {"params_b": 4})
    monkeypatch.setenv("CHRYSALIS_ENFORCE_ADVISORY", "1")
    steps = [
        {"id": "a", "domain": "reason", "action": "solve", "params": SMALL_TASK},
        {"id": "b", "domain": "meta", "action": "stats"},
    ]
    out = call("meta", "batch", {"steps": steps})
    assert out["status"] == "ERROR"
    assert [f["id"] for f in out["failed_steps"]] == ["a"] and out["steps_run"] == 1
    cont = call("meta", "batch", {"steps": steps, "continue_on_error": True})
    assert cont["status"] == "PARTIAL" and cont["steps_run"] == 2
    ok = call(
        "meta", "batch", {"steps": [{**steps[0], "params": {**SMALL_TASK, "override": True}}]}
    )
    assert ok["status"] == "OK" and ok["failed_steps"] == []


# --- MoE effective size (active_params_b) ---------------------------------------------------------


@pytest.mark.parametrize(
    ("fields", "cls", "effective"),
    [
        ({"params_b": 26, "active_params_b": 4}, "medium", 10.2),
        ({"params_b": 4}, "small", 4.0),
        ({"params_b": 8, "active_params_b": 1}, "small", 2.83),
        ({"params_b": 26, "active_params_b": -1}, "medium", 26.0),
        ({"params_b": 26, "active_params_b": "x"}, "medium", 26.0),
        ({"params_b": 4, "active_params_b": 26}, "small", 4.0),
        ({"params_b": 70, "active_params_b": 70}, "large", 70.0),
    ],
)
def test_effective_size_class_table(fields, cls, effective):
    p = prof.from_fields(fields, source="declared")
    assert p.model_class == cls
    assert p.effective_params_b == pytest.approx(effective, abs=0.01)


def test_invalid_or_oversized_active_is_ignored_with_warning():
    for bad in (-1, 0, "x", True, 99):
        p = prof.from_fields({"params_b": 26, "active_params_b": bad}, source="declared")
        assert p.active_params_b is None and "ignored" in p.view()["warning"]
    assert (
        "warning" not in prof.from_fields({"params_b": 26, "active_params_b": 4}, source="x").view()
    )


def test_view_exposes_effective_and_active():
    view = prof.from_fields({"params_b": 26, "active_params_b": 4}, source="declared").view()
    assert view["active_params_b"] == 4.0 and view["effective_params_b"] == 10.2
    assert view["class"] == "medium"


def test_forced_json_and_compact_moe(monkeypatch):
    assert (
        parse_forced('{"family": "gemma", "params_b": 26, "active_params_b": 4}').model_class
        == "medium"
    )
    for raw in ("gemma:26/4", "gemma:26a4:thinking"):
        assert parse_forced(raw).active_params_b == 4.0
        assert parse_forced(raw).model_class == "medium"
    assert parse_forced("gemma:26").model_class == "medium"
    monkeypatch.setenv(prof.ENV_PROFILE, '{"params_b": 26, "active_params_b": 4}')
    assert ProfileStore().resolve("s").model_class == "medium"


def test_handshake_declares_moe(world):
    _, call = world
    out = call(
        "meta", "handshake", {"model": "gemma-4-26b-a4b", "params_b": 26, "active_params_b": 4}
    )
    view = out["profile"]
    assert view["class"] == "medium" and view["effective_params_b"] == 10.2
    bad = call("meta", "handshake", {"model": "m", "params_b": 26, "active_params_b": 40})
    assert bad["profile"]["class"] == "medium" and "ignored" in bad["profile"]["warning"]
