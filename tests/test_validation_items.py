"""Per-item records in arm summaries, EMPTY status, and old-format report compatibility."""

import asyncio
import json
from typing import Any

from tests.fake_llm import FakeLlm, reply_text
from tests.test_validation_t12 import _fixture
from validation import eval_llm, report
from validation.bridge import run_agent
from validation.llm_client import LlmClient


def _run(coro: Any) -> Any:
    return asyncio.run(coro)


def _llm(fake: FakeLlm) -> LlmClient:
    return LlmClient(base_url=fake.url, model="fake", timeout_s=10, thinking=False)


def test_arm_summary_keeps_one_record_per_item_with_truncated_answers():
    long_answer = "x" * 500
    items = [("q1", "yes"), ("q2", "no")]
    with FakeLlm() as fake:
        fake.script = [reply_text("yes", prompt=11, completion=3), reply_text(long_answer)]
        out = _run(
            eval_llm._arm(
                _llm(fake),
                items,
                "sys",
                False,
                lambda a, g: a == g,
                None,
                None,
                64,
                tags=["hard", "easy"],
            )
        )
    first, second = out["items"]
    assert first == {
        "i": 0,
        "tag": "hard",
        "correct": True,
        "status": "OK",
        "actions": [],
        "answer": "yes",
        "prompt_tokens": 11,
        "completion_tokens": 3,
    }
    assert second["tag"] == "easy" and second["correct"] is False
    assert len(second["answer"]) == eval_llm.ANSWER_MAX_CHARS
    assert out["accuracy"] == 0.5 and out["statuses"] == ["OK"]  # aggregates unchanged


def test_arm_without_tags_has_no_tag_key():
    with FakeLlm() as fake:
        fake.script = [reply_text("a")]
        out = _run(
            eval_llm._arm(_llm(fake), [("q", "a")], "s", False, lambda a, g: a == g, None, None, 64)
        )
    assert "tag" not in out["items"][0]


def test_empty_final_message_is_empty_status_not_correct_and_keeps_reasoning():
    with FakeLlm() as fake:
        fake.script = [reply_text("  \n", reasoning="r" * 40)]
        run = _run(run_agent(_llm(fake), "q", None))
        assert run["status"] == "EMPTY" and run["answer"] == ""
        assert run["metrics"]["reasoning_chars"] == 40
        fake.script = [reply_text("   ", reasoning="think")]
        out = _run(
            eval_llm._arm(_llm(fake), [("q", "")], "s", False, lambda a, g: a == g, None, None, 64)
        )
    assert out["accuracy"] == 0.0  # even a scorer that would accept "" cannot count EMPTY
    assert out["statuses"] == ["EMPTY"] and out["items"][0]["status"] == "EMPTY"
    assert out["n"] == 1


def test_empty_does_not_break_the_arm_loop_like_unreachable():
    with FakeLlm() as fake:
        fake.script = [reply_text(""), reply_text("b")]
        out = _run(
            eval_llm._arm(
                _llm(fake),
                [("q1", "a"), ("q2", "b")],
                "s",
                False,
                lambda a, g: a == g,
                None,
                None,
                64,
            )
        )
    assert out["n"] == 2 and out["statuses"] == ["EMPTY", "OK"] and out["accuracy"] == 0.5


def test_report_renders_old_json_without_items_and_new_json_with_items(tmp_path):
    old = _fixture()
    assert all(
        "items" not in arm
        for g in old["modes"]["off"].values()
        if isinstance(g, dict)
        for arm in g.values()
        if isinstance(arm, dict)
    )
    (tmp_path / "llm.json").write_text(json.dumps(old))
    before = report.render(tmp_path)
    for group in old["modes"]["off"].values():
        if isinstance(group, dict):
            for arm in group.values():
                if isinstance(arm, dict) and "n" in arm:
                    arm["items"] = [
                        {
                            "i": 0,
                            "correct": True,
                            "status": "OK",
                            "actions": [],
                            "answer": "a",
                            "prompt_tokens": 1,
                            "completion_tokens": 1,
                        }
                    ]
    (tmp_path / "llm.json").write_text(json.dumps(old))
    assert report.render(tmp_path) == before
