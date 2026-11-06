"""Fix cycle: non-OK runs never erase OK data, parent env overrides do not reach the child, and
calibration probe answers never reach the model."""

import asyncio
import json
from typing import Any

from chrysalis_mcp.core import profile
from chrysalis_mcp.core.session import SessionCache
from chrysalis_mcp.domains import build_registry
from chrysalis_mcp.server import handle_call
from tests.test_validation_t12 import _WHO, _ProbeSolver
from validation import eval_llm
from validation.bridge import ChrysalisSession

OK_OLD = {
    "status": "OK",
    "config": {},
    "data": {"H-a": {"x": 1}},
    "modes": {"off": {"H-a": {"x": 1}}},
}
DEAD = {"status": "UNREACHABLE", "config": {}, "data": {}}
NONCE = "fixednonce0001"


def test_merge_keeps_old_ok_data_when_new_is_unreachable():
    assert eval_llm.merge(OK_OLD, DEAD) is OK_OLD


def test_merge_returns_new_when_no_ok_data_exists():
    assert eval_llm.merge(None, DEAD) is DEAD
    assert eval_llm.merge({"status": "UNREACHABLE", "data": {}}, DEAD) is DEAD


def test_main_leaves_llm_json_bytes_untouched_on_unreachable(tmp_path, monkeypatch, capsys):
    path = tmp_path / eval_llm.RESULT_FILE
    raw = json.dumps(OK_OLD, indent=7)  # non-canonical formatting: any rewrite would change bytes
    path.write_text(raw)
    monkeypatch.setattr(eval_llm, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(eval_llm, "run", lambda **_: DEAD)
    eval_llm.main([])
    assert path.read_text() == raw
    assert "NOTICE" in capsys.readouterr().out
    assert not list(tmp_path.glob("llm_thinking_*.json"))


def test_child_env_drops_advisory_and_profile_overrides(monkeypatch):
    monkeypatch.setenv("CHRYSALIS_ENFORCE_ADVISORY", "1")
    monkeypatch.setenv("CHRYSALIS_MODEL_PROFILE", "duck:1")
    env = ChrysalisSession(None)._params.env
    assert "CHRYSALIS_ENFORCE_ADVISORY" not in env
    assert "CHRYSALIS_MODEL_PROFILE" not in env


class _InProcSession:
    """Same surface as ChrysalisSession, dispatched in-process (server-side state reachable)."""

    def __init__(self, db_path: str | None = None) -> None:
        self._cache, self._reg = SessionCache(), build_registry()

    async def __aenter__(self) -> _InProcSession:
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None

    async def call(self, /, **args: Any) -> str:
        return handle_call(
            self._reg,
            self._cache,
            args["domain"],
            args["action"],
            args.get("params") or {},
            args.get("verbosity", "summary"),
        )

    async def openai_tool(self) -> dict[str, Any]:
        return {"type": "function", "function": {"name": "chrysalis", "parameters": {}}}


class _Recording(_ProbeSolver):
    def __init__(self, right: bool) -> None:
        super().__init__(right)
        self.seen: list[str] = []

    def chat(self, messages, tools=None, **kw):
        self.seen.extend(str(m.get("content") or "") for m in messages)
        return super().chat(messages, tools, **kw)


def test_calibration_never_sends_expected_answers_to_the_model(monkeypatch):
    monkeypatch.setattr(profile.secrets, "token_hex", lambda n=32: NONCE)
    monkeypatch.setattr(eval_llm, "ChrysalisSession", _InProcSession)
    solver = _Recording(True)
    out = asyncio.run(eval_llm.calibrate_once(solver, 64, _WHO))
    assert out["submitted"] and out["scores"] == {"reason": 1.0, "memory": 1.0}
    # Server-side expected answers for this session (plain text, computed from the same nonce).
    expected = {item for _, _, _, ans in profile._build(NONCE) for item in ans}
    assert len(expected) >= 5
    sent = "\n".join(solver.seen)
    assert "Probes:" in sent  # the model really saw the probe prompt
    leaked = sorted(a for a in expected if a in sent)
    assert leaked == [], f"expected answers reached the model: {leaked}"
