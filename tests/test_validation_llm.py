"""Live-model evaluation (``-m llm``): skips cleanly when CHRYSALIS_LLM_URL is unreachable."""

import pytest

from validation import RESULTS_DIR, eval_llm
from validation.llm_client import LlmClient

pytestmark = pytest.mark.llm


@pytest.fixture(scope="module")
def live() -> LlmClient:
    client = LlmClient()
    if not client.available():
        pytest.skip(f"no OpenAI-compatible model reachable at {client.base_url}")
    return client


def test_alone_vs_with_chrysalis(live):
    out = eval_llm.run(live)
    assert out["status"] == "OK"
    for arms in out["data"].values():
        assert all(0.0 <= a["accuracy"] <= 1.0 for a in arms.values())
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    eval_llm.write(out)
