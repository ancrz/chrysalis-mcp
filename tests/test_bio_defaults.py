"""CHRYSALIS_BIO_DEFAULTS (server-wide bio profile) and the eval-side closed loop."""

import asyncio
import json
import tempfile
from typing import Any

import pytest

from chrysalis_mcp.core import bio_defaults
from chrysalis_mcp.domains import memory
from tests.fake_llm import FakeLlm, reply_text
from validation import eval_llm
from validation.bridge import ChrysalisSession
from validation.datasets import HARD_PUZZLES
from validation.llm_client import LlmClient

KB = [
    {"s": "Quortre Annex", "r": "has_elevation_m", "o": "4290"},
    {"s": "Quortre Annex", "r": "located_in", "o": "Brevan"},
    {"s": "Lumen Keep", "r": "has_elevation_m", "o": "310"},
]
WHO = {"model": "m-4b", "family": "m", "params_b": 4.0, "thinking": False}


@pytest.fixture(autouse=True)
def db(tmp_path, monkeypatch):
    monkeypatch.delenv(bio_defaults.ENV_VAR, raising=False)
    memory.configure(tmp_path / "mem.db")
    yield
    memory.configure(None)


def _ingest(call):
    return call("memory", "ingest", params={"triples": KB})


def _usage_rows() -> int:
    return memory._get_state().read.counts()["edge_usage"]


def _recall(call, **extra):
    return call("memory", "recall", params={"query": "elevation of Quortre Annex", **extra})


# --- T1: server side ------------------------------------------------------------------------


def test_default_is_off_and_garbage_means_off(monkeypatch):
    assert bio_defaults.profile() == "off"
    monkeypatch.setenv(bio_defaults.ENV_VAR, "banana")
    assert bio_defaults.profile() == "off"
    monkeypatch.setenv(bio_defaults.ENV_VAR, " ALL ")
    assert bio_defaults.profile() == "all"


def test_off_changes_nothing(call):
    out = _ingest(call)
    assert "auto_flyvec" not in out
    assert memory._get_state().read.flyvec_info() is None
    _recall(call)
    assert _usage_rows() == 0
    cons = call("memory", "consolidate", params={})
    assert not {"replayed", "usage_applied", "captured"} & set(cons)


def test_all_applies_recall_and_consolidate_defaults(call, monkeypatch):
    monkeypatch.setenv(bio_defaults.ENV_VAR, "all")
    _ingest(call)
    assert _recall(call)["status"] == "OK"
    assert _usage_rows() > 0  # track_usage defaulted on
    cons = call("memory", "consolidate", params={})
    assert cons["status"] == "OK"
    assert {"replayed", "usage_applied", "captured"} <= set(cons)


def test_explicit_params_win(call, monkeypatch):
    monkeypatch.setenv(bio_defaults.ENV_VAR, "all")
    _ingest(call)
    assert _recall(call, track_usage=False, inhibition="fixed", encoder="ngram")["status"] == "OK"
    assert _usage_rows() == 0
    cons = call("memory", "consolidate", params={"replay": False, "capture": False})
    assert "replayed" not in cons and "captured" not in cons and "usage_applied" in cons


def test_all_auto_trains_flyvec_after_ingest_and_recall_uses_it(call, monkeypatch):
    monkeypatch.setenv(bio_defaults.ENV_VAR, "all")
    out = _ingest(call)
    assert out["auto_flyvec"] == "trained"
    assert memory._get_state().read.flyvec_info() is not None
    res = _recall(call, seeds="cosine", mode="seeds")
    assert res["status"] == "OK" and "encoder_fallback" not in res


def test_flyvec_default_falls_back_silently_without_a_model(call, monkeypatch):
    monkeypatch.setenv(bio_defaults.ENV_VAR, "all")
    monkeypatch.setattr(bio_defaults, "AUTO_FLYVEC_MAX_NODES", 0)  # nothing is trained
    out = _ingest(call)
    assert out["auto_flyvec"].startswith("skipped")
    assert memory._get_state().read.flyvec_info() is None
    res = _recall(call, seeds="cosine", mode="seeds")
    assert res["status"] == "OK"  # the default never errors
    explicit = _recall(call, encoder="flyvec")
    assert explicit["status"] == "ERROR"  # an explicit request still does


def test_active_profile_is_exposed(call, monkeypatch):
    for name in ("schema", "discover"):
        assert call("meta", name)["bio_defaults"] == "off"
    monkeypatch.setenv(bio_defaults.ENV_VAR, "all")
    assert call("meta", "schema")["bio_defaults"] == "all"
    assert call("meta", "discover")["bio_defaults"] == "all"
    shake = call("meta", "handshake", params={"model": "m-4b"})
    assert shake["bio_defaults"] == "all"
    terse = call("meta", "schema", verbosity="terse")
    assert "all" in json.dumps(terse)


# --- T2: eval feedback loop -----------------------------------------------------------------


def _llm(fake: FakeLlm) -> LlmClient:
    return LlmClient(base_url=fake.url, model="fake", timeout_s=10, thinking=False)


def _adv_arm(fake: FakeLlm, n: int) -> dict[str, Any]:
    puzzles = list(HARD_PUZZLES[:n])
    fake.script = [reply_text("{}") for _ in puzzles]
    return asyncio.run(
        eval_llm._arm(
            _llm(fake),
            [(p.text, p.answer) for p in puzzles],
            "sys",
            False,
            lambda a, g: False,
            None,
            None,
            64,
            WHO,
            features=[eval_llm._features(p) for p in puzzles],
        )
    )


def test_features_match_what_advise_extracts():
    feats = eval_llm._features(HARD_PUZZLES[0])
    assert feats["n_vars"] == len(HARD_PUZZLES[0].answer) and feats["goal"] in ("sat", "opt")


def test_feedback_off_by_default(monkeypatch):
    monkeypatch.delenv(eval_llm.ENV_FEEDBACK, raising=False)
    with FakeLlm() as fake:
        out = _adv_arm(fake, 2)
    assert "feedback_sent" not in out and out["prompt_hint_used"] is not None


def test_feedback_loop_sends_one_feedback_per_item(monkeypatch):
    monkeypatch.setenv(eval_llm.ENV_FEEDBACK, "1")
    with FakeLlm() as fake:
        out = _adv_arm(fake, 3)
    assert out["feedback_sent"] == 3
    snap = out["advice_stats"]
    assert snap["total"] >= 1 and snap["rows"]
    assert sum(r["failures"] for r in snap["rows"]) == 3  # score always False, used always False


def test_advice_db_lives_under_results_dir_unless_set(tmp_path, monkeypatch):
    monkeypatch.delenv(eval_llm.ENV_ADVICE_DB, raising=False)
    monkeypatch.setattr(eval_llm, "RESULTS_DIR", tmp_path / "res")
    with eval_llm._advice_db_scope():
        path = eval_llm.os.environ[eval_llm.ENV_ADVICE_DB]
        assert path.startswith(str(tmp_path / "res"))
    assert eval_llm.ENV_ADVICE_DB not in eval_llm.os.environ
    monkeypatch.setenv(eval_llm.ENV_ADVICE_DB, "/custom/advice.db")
    with eval_llm._advice_db_scope():
        assert eval_llm.os.environ[eval_llm.ENV_ADVICE_DB] == "/custom/advice.db"
    assert eval_llm.os.environ[eval_llm.ENV_ADVICE_DB] == "/custom/advice.db"


# --- T3: consolidate between learning and test phase ----------------------------------------


def _hbe2e_actions(monkeypatch) -> list[str]:
    seen: list[str] = []
    real = ChrysalisSession.call

    async def spy(self, /, **args: Any) -> str:
        seen.append(f"{args.get('domain')}.{args.get('action')}")
        return await real(self, **args)

    monkeypatch.setattr(ChrysalisSession, "call", spy)
    with FakeLlm() as fake, tempfile.TemporaryDirectory() as tmp:
        asyncio.run(eval_llm._hbe2e_group(_llm(fake), 64, 1, tmp))
    return seen


def test_hbe2e_consolidates_once_per_learning_arm_only_under_all(monkeypatch):
    monkeypatch.delenv(eval_llm.ENV_BIO_DEFAULTS, raising=False)
    assert "memory.consolidate" not in _hbe2e_actions(monkeypatch)
    monkeypatch.setenv(eval_llm.ENV_BIO_DEFAULTS, "all")
    seen = _hbe2e_actions(monkeypatch)
    assert seen.count("memory.consolidate") == len(eval_llm.HBE2E_ARMS) - 1  # not no_learn
