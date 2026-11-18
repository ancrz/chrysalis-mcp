import json
from typing import Any

import pytest

from chrysalis_mcp.core.registry import Registry
from chrysalis_mcp.core.session import SessionCache
from chrysalis_mcp.domains import build_registry
from chrysalis_mcp.server import handle_call

ENVELOPE_KEYS = {"status", "severity", "summary", "next_action", "truncated", "cursor"}


@pytest.fixture
def registry() -> Registry:
    return build_registry()


@pytest.fixture
def cache() -> SessionCache:
    return SessionCache()


@pytest.fixture
def call(registry: Registry, cache: SessionCache):
    def _call(domain: str, action: str, **kwargs: Any) -> dict[str, Any]:
        return json.loads(handle_call(registry, cache, domain, action, **kwargs))

    return _call


@pytest.fixture
def raw_call(registry: Registry, cache: SessionCache):
    def _raw(domain: str, action: str, **kwargs: Any) -> str:
        return handle_call(registry, cache, domain, action, **kwargs)

    return _raw


@pytest.fixture(autouse=True)
def _isolated_advice_db(tmp_path, monkeypatch):
    """Never let a test touch the user's real advice.db."""
    monkeypatch.setenv("CHRYSALIS_ADVICE_DB", str(tmp_path / "advice.db"))
    monkeypatch.delenv("CHRYSALIS_ADVICE_EXPLORE", raising=False)
    monkeypatch.delenv("CHRYSALIS_ADVICE_MIN_N", raising=False)
