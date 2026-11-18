"""Advisory learning: buckets, Beta update, prior until min_n, learned switch, reward, persistence."""

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from chrysalis_mcp.core import learning
from tests.conftest import ENVELOPE_KEYS

SID = "learn"
HARD = {"n_atoms": 9, "goal": "sat"}  # rule says recommended for the small class
EASY = {"n_atoms": 2, "goal": "sat"}


@pytest.fixture
def shake(call):
    def _shake(session: str = SID, model: str = "Gemma-4-E4B", thinking: bool = False) -> None:
        out = call(
            "meta", "handshake", session_id=session,
            params={"model": model, "params_b": 4, "thinking": thinking},
        )  # fmt: skip
        assert out["status"] == "OK"

    return _shake


def _fb(call, used: bool, outcome: str, task: dict[str, Any] = EASY, session: str = SID):
    return call(
        "meta", "feedback", session_id=session,
        params={"action": "reason.solve", "used": used, "task": task, "outcome": outcome},
    )  # fmt: skip


def _advise(call, task: dict[str, Any] = EASY, session: str = SID, **extra: Any):
    return call(
        "meta", "advise", session_id=session,
        params={"domain": "reason", "action": "solve", "task": task, **extra},
    )  # fmt: skip


# --- units -------------------------------------------------------------------------------------


def test_bucketing_bands_and_goal():
    key = learning.bucket_key
    assert key({"n_atoms": 0, "goal": "sat"}) == "atoms:0-3|goal:sat"
    assert key({"n_atoms": 3}) == "atoms:0-3|goal:na"
    assert key({"n_atoms": 4, "goal": "opt"}) == "atoms:4-6|goal:opt"
    assert key({"n_atoms": 6}) == "atoms:4-6|goal:na"
    assert key({"n_atoms": 7}) == "atoms:7-12|goal:na"  # same cut as the advisory rule (>= 7)
    assert key({"n_atoms": 13}) == "atoms:13+|goal:na"
    assert key({}) == "atoms:na|goal:na"
    assert key({"n_atoms": True, "goal": "x"}) == "atoms:na|goal:na"
    assert "comparisons" in learning.bucket_phrase("atoms:7-12|goal:opt")


def test_model_key_normalizes():
    assert learning.model_key("Gemma 4 E4B!") == "gemma-4-e4b"
    assert learning.model_key(None, "Gemma", 4) == "gemma-4b"
    assert learning.model_key(None) is None


def test_beta_posterior_and_decide_is_deterministic():
    assert learning.posterior_mean((0, 0)) == 0.5
    assert learning.posterior_mean((8, 0)) == pytest.approx(0.9)
    kw = {"min_n": 8, "explore": False, "seed": 1}
    assert learning.decide((1, 1), (1, 1), **kw) is None  # below min_n
    assert learning.decide((6, 0), (0, 4), **kw) == "used"
    assert learning.decide((0, 4), (6, 0), **kw) == "self"
    assert learning.decide((3, 3), (3, 3), **kw) is None  # exact tie keeps the rule
    ts = {"min_n": 8, "explore": True, "seed": 7}
    assert learning.decide((9, 1), (1, 9), **ts) == learning.decide((9, 1), (1, 9), **ts)
    assert learning.seed_for("a", ("m",), 3) == learning.seed_for("a", ("m",), 3)
    assert learning.seed_for("a", ("m",), 3) != learning.seed_for("a", ("m",), 4)


def test_db_path_defaults_next_to_memory_db(monkeypatch, tmp_path):
    monkeypatch.delenv("CHRYSALIS_ADVICE_DB")
    monkeypatch.setenv("CHRYSALIS_DB", str(tmp_path / "mem" / "substrate.db"))
    assert learning.advice_db_path() == str(tmp_path / "mem" / "advice.db")
    monkeypatch.setenv("CHRYSALIS_DB", ":memory:")
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    assert learning.advice_db_path() == str(tmp_path / "chrysalis" / "advice.db")


# --- feedback / advise -------------------------------------------------------------------------


def test_feedback_requires_handshake(call):
    out = _fb(call, True, "success", session="nobody")
    assert out["status"] == "ERROR" and "handshake" in out["next_action"]


def test_feedback_records_and_validates(call, shake):
    shake()
    out = _fb(call, True, "success")
    assert out["status"] == "OK" and ENVELOPE_KEYS <= set(out)
    assert out["bucket"] == "atoms:0-3|goal:sat" and out["arm"] == "used"
    assert out["cell"] == {"successes": 1, "failures": 0}
    assert _fb(call, True, "maybe")["status"] == "ERROR"
    bad = call("meta", "feedback", session_id=SID, params={"action": "meta.stats", "used": True, "outcome": "success"})
    assert bad["status"] == "ERROR"
    both = call(
        "meta", "feedback", session_id=SID,
        params={"action": "reason.solve", "used": True, "outcome": "success", "assignment": {}},
    )  # fmt: skip
    assert both["status"] == "ERROR"


def test_prior_until_min_n_then_learned_switch(call, shake):
    shake()
    rule = _advise(call, HARD)
    assert rule["advisory"]["level"] == "recommended"  # rule prior (n_atoms >= 7)
    assert rule["evidence"]["source"] == "rule" and rule["evidence"]["n_used"] == 0
    for _ in range(4):  # tool fails on hard tasks, answering yourself works: 8 outcomes in total
        _fb(call, True, "failure", HARD)
        _fb(call, False, "success", HARD)
    seven = _advise(call, HARD, explore=False, min_n=9)
    assert seven["evidence"]["source"] == "rule" and seven["advisory"]["level"] == "recommended"
    out = _advise(call, HARD, explore=False)
    assert out["source"] == "learned" and out["advisory"]["level"] == "discouraged"
    ev = out["evidence"]
    assert (ev["source"], ev["n_used"], ev["n_self"]) == ("learned", 4, 4)
    assert ev["success_rate_used"] == 0.0 and ev["success_rate_self"] == 1.0
    other = _advise(call, EASY, explore=False)  # other bucket: untouched, still rule
    assert other["evidence"]["source"] == "rule"
    assert "evidence" not in _advise(call, HARD, session="other")  # no handshake: rules only
    shake(session="other", model="Different")
    assert _advise(call, HARD, explore=False, session="other")["evidence"]["n_used"] == 0


def test_explore_is_deterministic_and_env_disables_it(call, shake, monkeypatch):
    shake()
    for _ in range(5):
        _fb(call, True, "success", HARD)
        _fb(call, False, "failure", HARD)
    first = _advise(call, HARD)
    assert first["evidence"]["explore"] is True and first["source"] == "learned"
    assert _advise(call, HARD)["advisory"] == first["advisory"]  # same state + session -> same pick
    monkeypatch.setenv("CHRYSALIS_ADVICE_EXPLORE", "0")
    greedy = _advise(call, HARD)
    assert greedy["evidence"]["explore"] is False and greedy["advisory"]["level"] == "recommended"
    assert _advise(call, HARD, explore=True)["evidence"]["explore"] is True  # param beats env


def test_handshake_prompt_hint_carries_learned_lines(call, shake):
    shake()
    assert "learned" not in call("meta", "handshake", session_id=SID, params={"model": "Gemma-4-E4B", "params_b": 4, "thinking": False}).get("prompt_hint", "")
    for _ in range(5):
        _fb(call, True, "failure", HARD)
        _fb(call, False, "success", HARD)
    out = call("meta", "handshake", session_id=SID, params={"model": "gemma-4-e4b", "params_b": 4, "thinking": False})
    assert "learned from 10 outcomes" in out["prompt_hint"] and "answer yourself" in out["prompt_hint"]


def test_thinking_flag_separates_models_arms(call, shake):
    shake(session="on", thinking=True)
    shake(session="off", thinking=False)
    _fb(call, True, "success", session="on")
    assert _advise(call, session="on")["evidence"]["n_used"] == 1
    assert _advise(call, session="off")["evidence"]["n_used"] == 0


# --- reward via reason.check -------------------------------------------------------------------

SAT = {"constraints": {"vars": {"x": "int", "y": "int"}, "assert": ["x + y == 10", "x > y", "y > 0"]}}
OPT = {
    "constraints": {
        "vars": {"x": "int", "y": "int"},
        "assert": ["x + y == 10", "x > y", "y > 0"],
        "goal": "max:x",
    }
}


def _reward(call, problem, assignment, used=True):
    return call(
        "meta", "feedback", session_id=SID,
        params={"action": "reason.solve", "used": used, "problem": problem, "assignment": assignment},
    )  # fmt: skip


def test_reward_from_check_sat(call, shake):
    shake()
    ok = _reward(call, SAT, {"x": 6, "y": 4})
    assert ok["status"] == "OK" and ok["outcome"] == "success"
    assert ok["bucket"].endswith("goal:sat")  # features derived from the problem
    assert _reward(call, SAT, {"x": 4, "y": 6})["outcome"] == "failure"


def test_reward_min_max_requires_optimality(call, shake):
    shake()
    assert _reward(call, OPT, {"x": 9, "y": 1})["outcome"] == "success"
    slack = _reward(call, OPT, {"x": 6, "y": 4})  # valid but not the maximum
    assert slack["outcome"] == "failure" and "not optimal" in slack["reward_reason"]
    assert slack["bucket"].endswith("goal:opt")
    assert _reward(call, OPT, {"y": 1})["outcome"] == "failure"


def test_reward_unjudgeable_records_nothing(call, shake):
    shake()
    unknown = _reward(call, SAT, {"zzz": 1})
    assert unknown["status"] == "ERROR"
    smt_opt = _reward(
        call,
        {"smt2": "(declare-const x Int)(assert (> x 0))(minimize x)"},
        {"x": 1},
    )
    assert smt_opt["status"] == "ERROR"
    stats = call("meta", "advice_stats", session_id=SID)
    assert stats["total"] == 0


# --- stats / reset -----------------------------------------------------------------------------


def test_stats_paginates_and_reset_needs_confirm(call, shake):
    shake()
    for task in (EASY, HARD, {"n_atoms": 5, "goal": "opt"}):
        _fb(call, True, "success", task)
    page = call("meta", "advice_stats", session_id=SID, params={"limit": 2})
    assert ENVELOPE_KEYS <= set(page) and len(page["rows"]) == 2 and page["total"] == 3
    assert page["truncated"] is True and page["cursor"] == "2"
    rest = call("meta", "advice_stats", session_id=SID, params={"limit": 2, "cursor": page["cursor"]})
    assert len(rest["rows"]) == 1 and rest["cursor"] is None
    assert call("meta", "advice_stats", session_id=SID, params={"limit": 0})["status"] == "ERROR"
    for params in ({}, {"confirm": False}):
        assert call("meta", "advice_reset", session_id=SID, params=params)["status"] == "ERROR"
    assert call("meta", "advice_stats", session_id=SID)["total"] == 3
    # model-scoped: another model's rows survive
    shake(session="b", model="Other-Model")
    _fb(call, True, "success", session="b")
    done = call("meta", "advice_reset", session_id=SID, params={"confirm": True})
    assert done["deleted"] == 3
    assert call("meta", "advice_stats", session_id=SID)["total"] == 0
    assert call("meta", "advice_stats", session_id="b")["total"] == 1


# --- persistence / schema ----------------------------------------------------------------------


def test_persistence_across_processes(call, shake, tmp_path):
    shake()
    for _ in range(3):
        _fb(call, True, "success")
    db = str(tmp_path / "advice.db")  # == CHRYSALIS_ADVICE_DB set by the autouse fixture
    code = (
        "import json;from chrysalis_mcp.core.learning import AdviceStore;"
        "print(json.dumps(AdviceStore().cells('gemma-4-e4b','off','reason.solve','atoms:0-3|goal:sat')))"
    )
    env = {**__import__("os").environ, "CHRYSALIS_ADVICE_DB": db}
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, check=True)
    assert json.loads(out.stdout) == [[3, 0], [0, 0]]
    assert Path(db).exists() and not (tmp_path / "substrate.db").exists()


def test_schema_lists_new_actions_with_examples(call):
    schema = call("meta", "schema")["domains"]["meta"]
    for name in ("feedback", "advice_stats", "advice_reset"):
        assert name in schema and schema[name]["example"]["action"] == name
    assert {"explore", "min_n"} <= set(schema["advise"]["params"])
    disc = call("meta", "discover", params={"domain": "meta"})["domains"]["meta"]
    assert {"feedback", "advice_stats", "advice_reset"} <= set(disc)


@pytest.mark.parametrize("verbosity", ["brief", "summary", "detail", "terse"])
def test_envelope_in_every_verbosity(call, shake, verbosity):
    shake()
    for params, action in (({"limit": 5}, "advice_stats"), ({"confirm": False}, "advice_reset")):
        out = call("meta", action, session_id=SID, params=params, verbosity=verbosity)
        assert ENVELOPE_KEYS <= set(out)
    fb = call("meta", "feedback", session_id=SID, verbosity=verbosity,
              params={"action": "reason.solve", "used": True, "outcome": "success"})  # fmt: skip
    assert ENVELOPE_KEYS <= set(fb) and fb["status"] == "OK"
