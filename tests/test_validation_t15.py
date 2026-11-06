"""T15: de-confounded advised prompt, format-lenient diagnostic metric, report rendering."""

import pytest

from validation import eval_llm, report
from validation.datasets import score_puzzle, score_puzzle_values

WHO = {"model": "m", "family": "gemma", "params_b": 4.0, "thinking": True}


def test_advised_prompt_is_tool_prompt_plus_advisory_with_format_last():
    s = eval_llm.advised_system(WHO)
    assert s.startswith(eval_llm.SYS_HARD_TOOL)
    assert s.endswith(eval_llm.SYS_FORMAT_REMINDER)
    assert "meta.advise" in s and "decide yourself" in s.lower()
    assert "your final reply is ONLY the" in s.splitlines()[-1]


GOLD = {"d": 5, "p": 7, "y": 6}


@pytest.mark.parametrize(
    "answer",
    [
        '{"d": 5, "p": 7, "y": 6}',
        "d = 5, p = 7, y = 6",
        "**d = 5**\n**p = 7**\n$y=6$",
        "The values: `d=5`; \\(p = 7\\); y = 6.",
    ],
)
def test_values_parser_accepts_json_and_markdown_assignments(answer):
    assert score_puzzle_values(answer, GOLD)


@pytest.mark.parametrize(
    "answer",
    [
        None,
        "",
        "d = 5, p = 7",
        "d = 5, p = 8, y = 6",
        "d == 5, p = 7, y = 6",
        "d = 9 then d = 5, p = 7, y = 6",
    ],
)
def test_values_parser_rejects_wrong_or_partial(answer):
    assert not score_puzzle_values(answer, GOLD)


def test_values_lenient_but_strict_score_unchanged():
    prose = "d = 5, p = 7, y = 6"
    assert score_puzzle_values(prose, GOLD) and not score_puzzle(prose, GOLD)


def _arm_dict(acc, values=None):
    d = {
        "n": 2,
        "accuracy": acc,
        "reason_use_rate": 1.0,
        "handshake_first_rate": 1.0,
        "mean_prompt_tokens": 1,
        "mean_completion_tokens": 1,
        "mean_tool_calls": 1,
    }
    if values is not None:
        d["values_accuracy"] = values
    return d


def _llm(with_values):
    v = (lambda x: x) if with_values else (lambda x: None)
    return {
        "status": "OK",
        "modes": {
            "on": {
                "H-c-hard": {"alone": _arm_dict(0.1, v(0.5)), "tool": _arm_dict(0.9, v(0.9))},
                "H-adv-heldout": {
                    "alone": _arm_dict(0.1, v(0.5)),
                    "tool": _arm_dict(0.9, v(0.9)),
                    "advised": _arm_dict(0.0, v(0.875)),
                },
            }
        },
    }


def test_report_shows_values_accuracy_as_diagnostic_and_verdict_is_unchanged():
    with_v, without = _llm(True), _llm(False)
    v1, t1 = report.verdict_adv(with_v)
    v2, t2 = report.verdict_adv(without)
    assert v1 == v2
    assert "format-lenient, diagnostic" in t1 and "advised 0.875" in t1
    assert "format-lenient" not in t2
    assert "format-lenient" in report.verdict_hard(with_v)[1]
    assert "format-lenient" not in report.verdict_hard(without)[1]


# ------------------------------------------------------------------------- served-model resolution
def test_client_resolves_served_model_lazily_and_caches(monkeypatch):
    from validation import llm_client as lc

    monkeypatch.delenv(lc.ENV_MODEL, raising=False)
    calls = []

    def fake_get(url, timeout):
        calls.append(url)
        return lc.httpx.Response(200, json={"data": [{"id": "gemma-4-e4b"}, {"id": "other"}]})

    monkeypatch.setattr(lc.httpx, "get", fake_get)
    c = lc.LlmClient("http://x/v1")
    assert calls == []  # lazy
    assert c.model == "gemma-4-e4b" and c.model == "gemma-4-e4b" and len(calls) == 1
    d = c.derive(True, 5.0)
    assert d.model == "gemma-4-e4b" and len(calls) == 1  # derive keeps the resolved name


@pytest.mark.parametrize("failure", ["error", "http500", "badjson", "empty"])
def test_client_falls_back_to_default_model(monkeypatch, failure):
    from validation import llm_client as lc

    monkeypatch.delenv(lc.ENV_MODEL, raising=False)

    def fake_get(url, timeout):
        if failure == "error":
            raise lc.httpx.ConnectTimeout("t")
        if failure == "http500":
            return lc.httpx.Response(500)
        if failure == "badjson":
            return lc.httpx.Response(200, text="nope")
        return lc.httpx.Response(200, json={"data": []})

    monkeypatch.setattr(lc.httpx, "get", fake_get)
    assert lc.LlmClient("http://x/v1").model == lc.DEFAULT_MODEL


def test_explicit_model_and_env_skip_the_lookup(monkeypatch):
    from validation import llm_client as lc

    monkeypatch.setattr(lc.httpx, "get", lambda *a, **k: pytest.fail("no lookup expected"))
    assert lc.LlmClient("http://x/v1", model="m").model == "m"
    monkeypatch.setenv(lc.ENV_MODEL, "envm")
    assert lc.LlmClient("http://x/v1").model == "envm"


def test_params_b_follows_obvious_model_names():
    assert eval_llm.params_b_for("gemma-4-e4b") == (4.0, None)
    assert eval_llm.params_b_for("gemma-4-26b-a4b") == (26.0, 4.0)
    assert eval_llm.params_b_for("mystery-70b") == (eval_llm.DEFAULT_PARAMS_B, None)


def test_moe_identity_declares_total_and_active_and_prompt_mentions_both():
    from tests.fake_llm import dead_url
    from validation.llm_client import LlmClient

    llm = LlmClient(dead_url(), model="gemma-4-26b-a4b")
    total, active = eval_llm.params_b_for(llm.model)
    who = eval_llm.identity(llm, total, active_params_b=active)
    assert who["params_b"] == 26.0 and who["active_params_b"] == 4.0
    assert "26.0B total, 4.0B active" in eval_llm.advised_system(who)
    dense = eval_llm.identity(llm, 4.0)
    assert "active_params_b" not in dense and "4.0B parameters" in eval_llm.advised_system(dense)


def test_run_records_both_param_counts():
    from tests.fake_llm import dead_url
    from validation.llm_client import LlmClient

    res = eval_llm.run(
        LlmClient(dead_url(), model="gemma-4-26b-a4b"), groups=("hard",), modes=("off",)
    )
    assert res["config"]["params_b"] == 26.0 and res["config"]["active_params_b"] == 4.0


def test_run_records_resolved_model_and_params_b():
    from tests.fake_llm import FakeLlm, dead_url
    from validation.llm_client import LlmClient

    res = eval_llm.run(LlmClient(dead_url(), model="gemma-4-e4b"), groups=("hard",), modes=("off",))
    assert res["config"]["model"] == "gemma-4-e4b" and res["config"]["params_b"] == 4.0
    with FakeLlm() as fake:  # the fake serves /models -> {"id": "fake"}
        assert LlmClient(fake.url).model == "fake"
