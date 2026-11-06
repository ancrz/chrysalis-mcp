from chrysalis_mcp.core.batch import MAX_STEPS, project_fields


def step(sid, action="discover", **extra):
    return {"id": sid, "domain": "meta", "action": action, **extra}


def test_cache_reference_from_store_as(call):
    out = call(
        "meta",
        "batch",
        params={
            "steps": [
                step("a", store_as="ctx"),
                step("b", "stats", params={"scope": "$cache.ctx.nothing"}),
            ]
        },
    )
    assert out["status"] == "ERROR"
    assert "unresolved reference" in out["failed_steps"][0]["error"]


def test_cache_reference_resolves_and_nested(call):
    # discover output has summary "1 domain(s)"; reference it through a step id.
    out = call(
        "meta",
        "batch",
        params={
            "steps": [
                step("a", store_as="ctx"),
                step(
                    "b",
                    "discover",
                    params={"domain": "$cache.ctx.domains.meta.schema.example.domain"},
                ),
            ],
            "return_only": "last",
            "select_fields": ["status", "summary", "domains.meta"],
        },
    )
    assert out["status"] == "OK"
    assert out["failed_steps"] == []
    assert "schema" in out["domains"]["meta"]


def test_return_only_last_and_select_fields(call):
    out = call(
        "meta",
        "batch",
        params={
            "steps": [step("a"), step("b", "stats", params={"scope": "all"})],
            "return_only": "last",
            "select_fields": ["status", "actions"],
        },
    )
    assert out["status"] == "OK"
    assert "actions" in out
    assert "results" not in out
    assert "domains" not in out


def test_return_only_summary_hides_outputs(call):
    out = call(
        "meta",
        "batch",
        params={"steps": [step("a"), step("b")], "return_only": "summary"},
    )
    assert out["results"] == [{"id": "a", "status": "OK"}, {"id": "b", "status": "OK"}]


def test_nested_batch_rejected(call):
    out = call("meta", "batch", params={"steps": [step("a", "batch", params={"steps": []})]})
    assert out["status"] == "ERROR"
    assert "nested" in out["error"]


def test_max_steps_enforced(call):
    out = call("meta", "batch", params={"steps": [step(f"s{i}") for i in range(MAX_STEPS + 1)]})
    assert out["status"] == "ERROR"
    assert "too many steps" in out["error"]


def test_error_stops_batch(call):
    out = call(
        "meta",
        "batch",
        params={"steps": [step("a"), step("bad", "nope"), step("c")], "return_only": "summary"},
    )
    assert out["status"] == "ERROR"
    assert [r["id"] for r in out["results"]] == ["a", "bad"]
    assert out["failed_steps"][0]["id"] == "bad"


def test_continue_on_error_runs_rest(call):
    out = call(
        "meta",
        "batch",
        params={
            "steps": [step("bad", "nope"), step("c")],
            "continue_on_error": True,
            "return_only": "summary",
        },
    )
    assert out["status"] == "PARTIAL"
    assert [r["id"] for r in out["results"]] == ["bad", "c"]


def test_failure_survives_projection(call):
    out = call(
        "meta",
        "batch",
        params={"steps": [step("bad", "nope")], "return_only": "last", "select_fields": ["valid"]},
    )
    assert out["status"] == "ERROR"
    assert out["failed_steps"]


def test_duplicate_ids_and_bad_return_only(call):
    assert call("meta", "batch", params={"steps": [step("a"), step("a")]})["status"] == "ERROR"
    bad = call("meta", "batch", params={"steps": [step("a")], "return_only": "x"})
    assert bad["status"] == "ERROR"


def test_project_fields_dot_paths_and_indexes():
    data = {"a": {"b": [10, {"c": 1}]}, "z": 0}
    assert project_fields(data, ["a.b.1.c", "missing"]) == {"a": {"b": {"1": {"c": 1}}}}
