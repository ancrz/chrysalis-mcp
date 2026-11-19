"""Leniency for small models: params as string/list, did-you-mean, aliases, implicit identity,
and the ``$prev`` / ``$steps`` batch references."""

import json

import pytest

from chrysalis_mcp.core.profile import ENV_PROFILE
from chrysalis_mcp.domains import memory

TRIPLE = {"s": "Paris", "r": "capital_de", "o": "Francia"}


@pytest.fixture(autouse=True)
def db(tmp_path):
    memory.configure(tmp_path / "mem.db")
    yield
    memory.configure(None)


# --- T1: params as JSON string / list ----------------------------------------------------------


def test_params_json_string_is_decoded_and_noted(call):
    out = call("memory", "ingest", params=json.dumps({"triples": [TRIPLE]}))
    assert out["status"] == "OK" and out["coerced"] == ["params_from_json_string"]


def test_params_blank_string_means_no_params(call):
    out = call("meta", "discover", params="  ")
    assert out["status"] == "OK" and "coerced" not in out


@pytest.mark.parametrize("bad", ["[1, 2]", "not json", '"text"', "3"])
def test_params_string_that_is_not_an_object_is_a_structured_error(call, bad):
    out = call("memory", "ingest", params=bad)
    assert out["status"] == "ERROR" and "JSON object" in out["error"]
    assert out["example"]["action"] == "ingest" and out["next_action"]


def test_params_list_wraps_into_sole_required_array(call):
    out = call("memory", "ingest", params=[TRIPLE])
    assert out["status"] == "OK" and "params_from_list" in out["coerced"]


def test_params_list_without_unique_array_param_is_an_error(call):
    out = call("memory", "recall", params=["x"])
    assert out["status"] == "ERROR" and out["example"]["action"] == "recall"
    out = call("meta", "schema", params=["x"])  # no required array param
    assert out["status"] == "ERROR"


def test_mcp_input_schema_accepts_object_string_list_null():
    import asyncio

    from chrysalis_mcp.server import create_app

    tool = asyncio.run(create_app().list_tools())[0]
    kinds = {b.get("type") for b in tool.inputSchema["properties"]["params"]["anyOf"]}
    assert kinds == {"object", "string", "array", "null"}


# --- T2: did_you_mean / example / cross-action / nesting / triples -----------------------------


def test_param_error_carries_example_and_synonym_and_close_match(call):
    out = call("meta", "feedback", params={"action": "reason.solve", "used_tool": True})
    assert out["status"] == "ERROR"
    assert out["did_you_mean"] == {"used_tool": "used"}
    assert out["example"]["action"] == "feedback" and "params" in out["example"]
    out = call("memory", "update", params={"query": "q", "node": "n", "delt": 1})
    assert out["did_you_mean"] == {"delt": "delta"}


def test_cross_action_hint_points_to_update_not_alias(call):
    out = call("memory", "ingest", params={"query": "q", "node": "n", "delta": 1})
    assert out["status"] == "ERROR"
    assert out["did_you_mean_action"] == "memory.update"
    assert out["did_you_mean_example"]["action"] == "update"


def test_no_cross_action_hint_when_nothing_fits(call):
    out = call("memory", "ingest", params={"zzz": 1})
    assert "did_you_mean_action" not in out and out["example"]["action"] == "ingest"


def test_triple_words_at_top_level_get_a_hint(call):
    out = call("memory", "ingest", params={"subject": "a", "predicate": "b", "object": "c"})
    assert out["status"] == "ERROR" and "{s, r, o}" in out["hint"]


def test_flat_task_features_are_nested_for_advise_and_feedback(call):
    out = call("meta", "advise", params={"domain": "reason", "action": "solve", "n_atoms": 9})
    assert out["status"] == "OK" and out["coerced"] == ["task_features_nested"]
    out = call(
        "meta",
        "feedback",
        params={
            "model": "tiny-4b",
            "action": "reason.solve",
            "used": True,
            "outcome": "success",
            "n_vars": 3,
            "goal": "sat",
        },
    )
    assert out["status"] == "OK" and "task_features_nested" in out["coerced"]
    assert out["bucket"]


def test_flat_features_merge_with_explicit_task(call):
    out = call(
        "meta",
        "advise",
        params={"domain": "reason", "action": "solve", "task": {"goal": "sat"}, "n_vars": 2},
    )
    assert out["status"] == "OK"


@pytest.mark.parametrize(
    "triples",
    [
        [{"subject": "A", "predicate": "rel", "object": "B"}],
        [{"subject": "A", "relation": "rel", "object": "B"}],
        [["A", "rel", "B"]],
        [{"s": "A", "r": "rel", "o": "B"}],
    ],
)
def test_ingest_normalizes_triple_shapes(call, triples):
    out = call("memory", "ingest", params={"triples": triples})
    assert out["status"] == "OK" and out["edges_added"] == 1
    assert ("coerced" in out) == (triples[0] != {"s": "A", "r": "rel", "o": "B"})


def test_ingest_still_rejects_bad_triples(call):
    assert call("memory", "ingest", params={"triples": [["a", "b"]]})["status"] == "ERROR"
    assert call("memory", "ingest", params={"triples": [{"subject": "a"}]})["status"] == "ERROR"


# --- T3: memory.teach alias ----------------------------------------------------------------------


def test_teach_is_in_schema_and_shares_handler_and_params(registry):
    update, teach = registry.spec("memory", "update"), registry.spec("memory", "teach")
    assert teach.handler is update.handler and teach.params_schema == update.params_schema
    assert "teach" in registry.actions("memory") and "alias" in teach.description.lower()


def test_update_and_teach_behave_identically(call):
    call("memory", "ingest", params={"triples": [TRIPLE]})
    a = call("memory", "update", params={"query": "capital of france", "node": "Paris"})
    b = call("memory", "teach", params={"query": "capital of france", "node": "Paris"})
    assert a["status"] == b["status"] == "OK"
    assert a["kcs_reinforced"] == b["kcs_reinforced"] and a["node"] == b["node"]
    bad = call("memory", "teach", params={"query": "q"})
    assert bad["missing"] == ["node"]


# --- T4: implicit identity ----------------------------------------------------------------------

FB = {"action": "reason.solve", "used": True, "outcome": "success"}


def test_feedback_with_model_declares_implicit_identity(call, registry):
    out = call("meta", "feedback", params={**FB, "model": "Gemma 4 E4B"}, session_id="s1")
    assert out["status"] == "OK" and out["implicit_handshake"] is True
    profile = registry.profiles.resolve("s1")
    assert profile.source == "declared" and profile.model == "Gemma 4 E4B"
    assert profile.params_b is None and profile.client is None
    again = call("meta", "feedback", params=FB, session_id="s1")  # identity now sticks
    assert again["status"] == "OK" and "implicit_handshake" not in again


def test_feedback_session_identity_wins_over_model_param(call):
    call("meta", "handshake", params={"model": "real-model"}, session_id="s2")
    out = call("meta", "feedback", params={**FB, "model": "other"}, session_id="s2")
    assert out["status"] == "OK" and out["model"] == "real-model"
    assert "implicit_handshake" not in out


def test_feedback_without_any_model_gives_handshake_example(call):
    out = call("meta", "feedback", params=FB, session_id="s3")
    assert out["status"] == "ERROR" and out["example"]["action"] == "handshake"
    assert "handshake" in out["next_action"]


def test_feedback_rejects_non_string_model(call):
    out = call("meta", "feedback", params={**FB, "model": 5}, session_id="s4")
    assert out["status"] == "ERROR" and "model" in out["error"]


def test_forced_profile_is_the_fallback_identity(call, monkeypatch):
    monkeypatch.setenv(ENV_PROFILE, "qwen:4")
    out = call("meta", "feedback", params=FB, session_id="s5")
    assert out["status"] == "OK" and "implicit_handshake" not in out and "qwen" in out["model"]


def test_advice_reset_without_model_gives_handshake_example(call):
    out = call("meta", "advice_reset", params={"confirm": True}, session_id="s6")
    assert out["status"] == "ERROR" and out["example"]["action"] == "handshake"


# --- T5: $prev / $steps ------------------------------------------------------------------------


def _batch(call, steps, **extra):
    return call("meta", "batch", params={"steps": steps, **extra})


def _echo_step(ref):
    # advise echoes "<domain>.<action>" in its error: the resolved reference is observable
    return {
        "domain": "meta",
        "action": "advise",
        "params": {"domain": "reason", "action": ref},
    }


def test_prev_and_steps_references_resolve_values(call):
    hs = {"id": "h", "domain": "meta", "action": "handshake", "params": {"model": "m1"}}
    out = _batch(
        call,
        [hs, _echo_step("$prev.profile.model"), _echo_step("$steps.h.profile.model")],
        continue_on_error=True,
    )
    assert "reason.m1" in out["results"][1]["output"]["error"]
    assert "reason.m1" in out["results"][2]["output"]["error"]


def test_steps_reference_with_missing_path_is_an_error(call):
    hs = {"id": "h", "domain": "meta", "action": "handshake", "params": {"model": "m1"}}
    out = _batch(call, [hs, _echo_step("$steps.h.profile.nope")])
    assert out["status"] == "ERROR" and "path not found" in json.dumps(out)


def test_prev_resolves_to_last_successful_step(call):
    out = _batch(
        call,
        [
            {"id": "ok", "domain": "meta", "action": "discover"},
            {"id": "bad", "domain": "meta", "action": "nope"},
            {
                "id": "use",
                "domain": "meta",
                "action": "stats",
                "params": {"scope": "$prev.status"},
            },
        ],
        continue_on_error=True,
    )
    # $prev skips the failed step: it resolves to the output of 'ok', never "no previous"
    assert "no previous" not in json.dumps(out["results"][2]["output"])


def test_unresolved_prev_at_step_zero_is_an_error(call):
    out = _batch(call, [{"domain": "meta", "action": "stats", "params": {"scope": "$prev.status"}}])
    assert out["status"] == "ERROR"
    assert "no previous successful step" in out["results"][0]["output"]["error"]


def test_unresolved_steps_reference_is_an_error(call):
    out = _batch(call, [{"domain": "meta", "action": "stats", "params": {"scope": "$steps.zz"}}])
    assert out["status"] == "ERROR" and "no successful step 'zz'" in json.dumps(out)


def test_steps_do_not_see_failed_steps_or_cache_keys(call):
    out = _batch(
        call,
        [
            {"id": "k", "domain": "meta", "action": "discover", "store_as": "cached"},
            {"domain": "meta", "action": "stats", "params": {"scope": "$steps.cached"}},
        ],
    )
    assert out["status"] == "ERROR"  # store_as keys live under $cache, not $steps


def test_cache_references_still_work(call):
    out = _batch(
        call,
        [
            {"id": "a", "domain": "meta", "action": "discover", "store_as": "d"},
            {
                "domain": "meta",
                "action": "advise",
                "params": {"domain": "reason", "action": "solve", "task": {"goal": "sat"}},
            },
            {"domain": "meta", "action": "stats", "params": {"agent_id": "$cache.d.status"}},
        ],
    )
    assert out["status"] == "OK"
