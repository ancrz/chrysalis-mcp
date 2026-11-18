"""Lab throughput: server slots, eval concurrency, lenient QA scoring, truncation metric."""

import asyncio
import threading
import time
from typing import Any

import pytest

from tests.fake_llm import FakeLlm, reply_text
from tests.test_validation_llm_server import make_cfg
from validation import eval_llm, report
from validation import llm_server as ls
from validation.datasets import score_qa, score_qa_lenient
from validation.llm_client import LlmClient
from validation.llm_server import Config, Resources

# ------------------------------------------------------------------------------ server slots


def test_parallel_defaults_to_one_and_leaves_argv_unchanged(tmp_path):
    cfg = make_cfg(tmp_path)
    argv = ls.server_argv(cfg)
    assert cfg.parallel == 1 and "-np" not in argv
    assert argv[argv.index("-c") + 1] == str(cfg.ctx)
    assert cfg.total_est_gpu_mib == cfg.est_gpu_mib


def test_parallel_scales_the_pool_so_each_slot_keeps_the_profile_ctx(tmp_path):
    argv = ls.server_argv(make_cfg(tmp_path, parallel=3, ctx=8192))
    assert argv[argv.index("-np") + 1] == "3"
    assert argv[argv.index("-c") + 1] == str(3 * 8192)


def test_parallel_adds_kv_per_extra_slot_to_the_admission_estimate(tmp_path):
    cfg = make_cfg(tmp_path, parallel=3, est_gpu_mib=4096, est_kv_mib_per_slot=512)
    assert cfg.total_est_gpu_mib == 4096 + 2 * 512
    # GTT needed = 5120 + margin 4096 = 9216
    ok = Resources(9216, 0, 30000)
    low = Resources(9215, 0, 30000)
    assert ls.admit(ok, cfg).admitted
    dec = ls.admit(low, cfg)
    assert not dec.admitted and "est 5120" in dec.reasons[0]
    assert ls.admit(
        low, make_cfg(tmp_path / "b", est_gpu_mib=4096)
    ).admitted  # parallel=1: 8192 enough


def test_parallel_env_profile_and_validation():
    env = {"HOME": "/h", "CHRYSALIS_LLM_PARALLEL": "4", "CHRYSALIS_LLM_EST_KV_MIB_PER_SLOT": "300"}
    cfg = Config.from_env(env, profile="qwen4b")
    assert (cfg.parallel, cfg.est_kv_mib_per_slot) == (4, 300)
    for name in ("qwen4b", "e4b"):
        prof = Config.from_env({"HOME": "/h"}, profile=name)
        assert (prof.parallel, prof.est_kv_mib_per_slot) == (1, 512)
    with pytest.raises(ValueError, match="PARALLEL"):
        Config.from_env({"HOME": "/h", "CHRYSALIS_LLM_PARALLEL": "0"})


# ---------------------------------------------------------------------------- lenient scoring


@pytest.mark.parametrize(
    "answer, gold, strict, lenient",
    [
        ("4290", "4290", True, True),
        ("4290 meters", "4290", False, True),
        ("4,290 meters", "4290", False, True),
        ("It is 4290.", "4290", False, True),
        ("4290 or 2166", "4290", False, False),
        ("4290 and 4290", "4290", False, False),  # two numbers: ambiguous by rule
        ("4290.5", "4290", False, False),
        ("1,2", "12", False, False),
        ("2020-2021", "2021", False, False),
        ("4291 meters", "4290", False, False),
        ("about four thousand", "4290", False, False),
        ("Paris", "Paris", True, True),
        ("Paris, France 75", "Paris", False, False),  # non-integer gold: strict only
        ("", "4290", False, False),
        (None, "4290", False, False),
        ("-5 degrees", "-5", False, True),
    ],
)
def test_score_qa_lenient_table(answer, gold, strict, lenient):
    assert score_qa(answer, gold) is strict
    assert score_qa_lenient(answer, gold) is lenient


# ------------------------------------------------------------------- truncation and budgets


def _llm(fake: FakeLlm, thinking: bool = False) -> LlmClient:
    return LlmClient(base_url=fake.url, model="fake", timeout_s=10, thinking=thinking)


def test_finish_reason_and_truncated_rate_with_lenient_accuracy():
    with FakeLlm() as fake:
        fake.script = [
            reply_text("4290 meters", finish_reason="stop"),
            reply_text("cut off mid", finish_reason="length"),
        ]
        out = eval_llm.run(_llm(fake), groups=("ctx",), modes=("off",), limit=2)
    arm = out["modes"]["off"]["H-a-ctx"]["context"]
    assert arm["truncated_rate"] == 0.5
    assert [i["finish_reason"] for i in arm["items"]] == ["stop", "length"]
    assert "accuracy_lenient" in arm and all("correct_lenient" in i for i in arm["items"])
    assert arm["accuracy_lenient"] >= arm["accuracy"]


def test_missing_finish_reason_is_not_truncation():
    with FakeLlm() as fake:
        fake.script = [reply_text("x")] * 3
        out = eval_llm.run(_llm(fake), groups=("hc",), modes=("off",), limit=1)
    alone = out["modes"]["off"]["H-c"]["alone"]
    assert alone["truncated_rate"] == 0.0 and alone["items"][0]["finish_reason"] is None
    assert "accuracy_lenient" not in alone  # puzzles are not QA


def test_max_tokens_env_overrides_per_mode(monkeypatch):
    assert eval_llm.mode_config("on")["max_tokens"] == 2048
    assert eval_llm.mode_config("off")["max_tokens"] == 512
    monkeypatch.setenv("CHRYSALIS_EVAL_MAX_TOKENS_ON", "4096")
    monkeypatch.setenv("CHRYSALIS_EVAL_MAX_TOKENS_OFF", "256")
    assert eval_llm.mode_config("on")["max_tokens"] == 4096
    assert eval_llm.mode_config("off")["max_tokens"] == 256
    assert eval_llm.MODES["on"]["max_tokens"] == 2048  # the table itself is untouched
    with FakeLlm() as fake:
        out = eval_llm.run(_llm(fake), groups=("ctx",), modes=("off",), limit=1)
        assert {r["max_tokens"] for r in fake.requests} == {256}
    assert out["modes"]["off"]["config"]["max_tokens"] == 256
    monkeypatch.setenv("CHRYSALIS_EVAL_MAX_TOKENS_OFF", "abc")
    with pytest.raises(ValueError, match="MAX_TOKENS_OFF"):
        eval_llm.mode_config("off")


def test_timeout_env_overrides_per_mode(monkeypatch):
    assert eval_llm.mode_config("on")["timeout_s"] == 300.0
    assert eval_llm.mode_config("off")["timeout_s"] == 120.0
    monkeypatch.setenv("CHRYSALIS_EVAL_TIMEOUT_ON", "900.5")
    monkeypatch.setenv("CHRYSALIS_EVAL_TIMEOUT_OFF", "45")
    assert eval_llm.mode_config("on")["timeout_s"] == 900.5
    assert eval_llm.mode_config("off")["timeout_s"] == 45.0
    assert eval_llm.MODES["on"]["timeout_s"] == 300.0  # the table itself is untouched
    assert eval_llm.MODES["off"]["timeout_s"] == 120.0
    base = LlmClient("http://x/v1", "m", 1.0, False)
    derived = base.derive(True, eval_llm.mode_config("on")["timeout_s"])
    assert derived.timeout_s == 900.5


@pytest.mark.parametrize("bad", ["abc", "0", "-5", "nan", "inf"])
def test_timeout_env_invalid_raises(monkeypatch, bad):
    monkeypatch.setenv("CHRYSALIS_EVAL_TIMEOUT_OFF", bad)
    with pytest.raises(ValueError, match="TIMEOUT_OFF"):
        eval_llm.mode_config("off")


# ------------------------------------------------------------------------------ concurrency


class SlowLlm:
    """Stand-in client: answers ``a<n>`` for question ``q<n>`` after a delay; tracks overlap."""

    thinking = False
    model = "slow"
    base_url = "stub"

    def __init__(self, delays: dict[int, float], unreachable_at: int | None = None) -> None:
        self.delays, self.unreachable_at = delays, unreachable_at
        self._lock = threading.Lock()
        self.now = self.peak = 0

    def chat(self, messages: list[dict[str, Any]], tools=None, **_: Any) -> dict[str, Any]:
        n = int(messages[-1]["content"][1:])
        with self._lock:
            self.now += 1
            self.peak = max(self.peak, self.now)
        try:
            time.sleep(self.delays.get(n, 0.0))
        finally:
            with self._lock:
                self.now -= 1
        if n == self.unreachable_at:
            return {"status": "UNREACHABLE", "error": "down"}
        return {
            "status": "OK",
            "message": {"role": "assistant", "content": f"a{n}"},
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            "finish_reason": "stop",
        }


def _arm(llm: SlowLlm, n: int, **kw: Any) -> dict[str, Any]:
    items = [(f"q{i}", f"a{i}") for i in range(n)]
    return asyncio.run(
        eval_llm._arm(llm, items, "sys", False, score_qa, None, None, 16, **kw)  # type: ignore[arg-type]
    )


def test_concurrent_items_overlap_but_results_keep_item_order():
    delays = {0: 0.3, 1: 0.2, 2: 0.1, 3: 0.0}  # later items finish first
    llm = SlowLlm(delays)
    arm = _arm(llm, 4, concurrency=4)
    assert llm.peak >= 2
    assert [i["answer"] for i in arm["items"]] == ["a0", "a1", "a2", "a3"]
    assert [i["i"] for i in arm["items"]] == [0, 1, 2, 3] and arm["accuracy"] == 1.0


def test_concurrency_one_is_sequential_and_matches_concurrent_results():
    seq_llm = SlowLlm({i: 0.02 for i in range(4)})
    seq = _arm(seq_llm, 4, concurrency=1)
    assert seq_llm.peak == 1
    par = _arm(SlowLlm({}), 4, concurrency=3)
    strip = lambda a: [(i["answer"], i["correct"], i["status"]) for i in a["items"]]  # noqa: E731
    assert strip(seq) == strip(par)


def test_concurrency_context_value_is_used_and_default_is_one():
    assert eval_llm.ARM_CONCURRENCY.get() == 1
    llm = SlowLlm({i: 0.1 for i in range(3)})
    arm = asyncio.run(_with_ctx(llm, 3))
    assert llm.peak >= 2 and arm["n"] == 3


async def _with_ctx(llm: SlowLlm, width: int) -> dict[str, Any]:
    eval_llm.ARM_CONCURRENCY.set(width)
    items = [(f"q{i}", f"a{i}") for i in range(3)]
    return await eval_llm._arm(llm, items, "s", False, score_qa, None, None, 16)  # type: ignore[arg-type]


def test_unreachable_cuts_the_arm_at_the_first_failing_item():
    llm = SlowLlm({}, unreachable_at=1)
    arm = _arm(llm, 6, concurrency=1)
    assert arm["n"] == 2 and arm["statuses"] == ["OK", "UNREACHABLE"]
    arm = _arm(SlowLlm({}, unreachable_at=1), 6, concurrency=3)
    assert arm["statuses"][-1] == "UNREACHABLE" and arm["n"] == 2


def test_concurrency_env_and_cli(monkeypatch, tmp_path):
    assert eval_llm.concurrency_from_env() == 1
    monkeypatch.setenv("CHRYSALIS_EVAL_CONCURRENCY", "3")
    assert eval_llm.concurrency_from_env() == 3
    monkeypatch.setenv("CHRYSALIS_EVAL_CONCURRENCY", "0")
    with pytest.raises(ValueError, match="CONCURRENCY"):
        eval_llm.concurrency_from_env()
    monkeypatch.delenv("CHRYSALIS_EVAL_CONCURRENCY")
    seen: dict[str, Any] = {}
    monkeypatch.setattr(
        eval_llm, "run", lambda **kw: seen.update(kw) or {"status": "UNREACHABLE", "data": {}}
    )
    monkeypatch.setattr(eval_llm, "RESULTS_DIR", tmp_path)
    eval_llm.main(["--hyp", "ha", "--concurrency", "5"])
    assert seen["concurrency"] == 5


def test_full_run_with_concurrency_records_it_in_the_config():
    with FakeLlm() as fake:
        fake.script = [reply_text("x")] * 10
        out = eval_llm.run(_llm(fake), groups=("ctx",), modes=("off",), limit=3, concurrency=3)
    assert out["modes"]["off"]["config"]["concurrency"] == 3
    assert out["modes"]["off"]["H-a-ctx"]["context"]["n"] == 3


# ----------------------------------------------------------------------------------- report


def test_report_rows_surface_lenient_and_truncated_when_present(tmp_path):
    arm = {
        "accuracy": 0.5,
        "accuracy_lenient": 0.75,
        "truncated_rate": 0.25,
        "mean_prompt_tokens": 1,
        "mean_completion_tokens": 1,
        "mean_tool_calls": 0,
    }
    old = {k: v for k, v in arm.items() if k not in ("accuracy_lenient", "truncated_rate")}
    llm = {"status": "OK", "data": {}, "modes": {"off": {"H-a": {"alone": arm, "memory": old}}}}
    (tmp_path / "llm.json").write_text(__import__("json").dumps(llm))
    text = report.render(tmp_path)
    assert "acc lenient" in text and "0.75" in text and "0.25" in text
