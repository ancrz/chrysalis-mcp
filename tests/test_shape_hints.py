"""Shape-based redirect hints on failed validation, and the tool description verbs."""

import asyncio

from chrysalis_mcp.core.shape_hints import SHAPE_RULES, shape_hint


def test_reason_task_descriptor_redirects_to_meta_advise(call):
    for action in ("solve", "check"):
        out = call("reason", action, params={"n_vars": 6, "goal": "sat"})
        assert out["status"] == "ERROR"
        assert out["did_you_mean_action"] == "meta.advise"
        assert out["did_you_mean_example"] == {
            "domain": "meta",
            "action": "advise",
            "params": {
                "domain": "reason",
                "action": action,
                "task": {"n_vars": 6, "goal": "sat"},
            },
        }
        assert "meta.feedback" in out["hint"]


def test_memory_ingest_feedback_record_redirects_to_meta_feedback(call):
    params = {"action": "reason.solve", "used": True, "outcome": "success"}
    out = call("memory", "ingest", params=params)
    assert out["status"] == "ERROR"
    assert out["did_you_mean_action"] == "meta.feedback"
    assert out["did_you_mean_example"]["params"] == params


def test_valid_and_unrelated_calls_are_untouched(call):
    ok = call("memory", "ingest", params={"triples": [{"s": "a", "r": "b", "o": "c"}]})
    assert ok["status"] == "OK" and "did_you_mean_action" not in ok
    real = {"constraints": {"vars": {"x": "int"}, "assert": ["x > 1"], "goal": "min:x"}}
    solved = call("reason", "solve", params=real)
    assert solved["status"] == "OK" and "hint" not in solved
    # constraints present -> a goal key is a real solver field, never a task descriptor
    assert shape_hint("reason", "solve", {"goal": "sat", "constraints": {}}) is None
    # real triples alongside stray feedback-ish keys: no redirect
    assert shape_hint("memory", "ingest", {"action": "a", "used": True, "triples": []}) is None
    assert shape_hint("memory", "recall", {"action": "a", "used": True}) is None
    typo = call("reason", "solve", params={"smt3": "x"})
    assert "did_you_mean_action" not in typo


def test_rules_table_targets_are_registered_actions(registry):
    for rule in SHAPE_RULES:
        domain, action = rule.target.split(".")
        assert action in registry.actions(domain)


def test_tool_description_names_the_three_self_report_verbs():
    from chrysalis_mcp.server import TOOL_DESCRIPTION, create_app

    for verb in ("meta.handshake", "meta.advise", "meta.feedback"):
        assert verb in TOOL_DESCRIPTION
    assert "Single dispatcher" in TOOL_DESCRIPTION and "meta.batch" in TOOL_DESCRIPTION
    listed = asyncio.run(create_app().list_tools())[0].description
    assert listed.startswith(TOOL_DESCRIPTION)


def _dispatch_example(call, out):
    ex = out["did_you_mean_example"]
    return call(ex["domain"], ex["action"], params=ex["params"])


def test_suggested_advise_example_is_a_valid_call(call):
    out = call("reason", "solve", params={"n_vars": 6, "goal": "sat"})
    assert _dispatch_example(call, out)["status"] in {"OK", "PARTIAL"}
    as_str = call("reason", "solve", params='{"n_vars": 6, "goal": "sat"}')
    assert as_str["did_you_mean_action"] == "meta.advise"


def test_suggested_feedback_example_is_a_valid_call(call):
    call("meta", "handshake", params={"model": "gemma-4-e4b", "family": "gemma", "params_b": 4})
    ok = call(
        "memory", "ingest", params={"action": "reason.solve", "used": True, "outcome": "success"}
    )
    assert _dispatch_example(call, ok)["status"] in {"OK", "PARTIAL"}
    # every shape the rule matches must yield an example meta.feedback accepts
    for params in (
        {"action": "reason.solve", "outcome": "success"},  # no 'used' (required by feedback)
        {"action": "reason.solve", "used": "yes", "outcome": "worked"},  # wrong types / enum
        {"action": "reason.solve", "used": True, "task": "not-an-object"},
    ):
        out = call("memory", "ingest", params=params)
        assert out["did_you_mean_action"] == "meta.feedback"
        assert _dispatch_example(call, out)["status"] in {"OK", "PARTIAL"}, params


def test_shape_hint_example_is_bounded_for_huge_values(call):
    huge = "A" * 100_000
    for domain, action, params in (
        ("reason", "solve", {"n_vars": huge, "goal": huge}),
        ("memory", "ingest", {"action": huge, "outcome": huge}),
    ):
        out = call(domain, action, params=params)
        assert len(str(out)) < 8_000, (domain, len(str(out)))
