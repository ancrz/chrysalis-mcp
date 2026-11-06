"""Minimal OpenAI-compatible chat client. Fail-soft: unreachable model = result, never a raise."""

import os
from typing import Any

import httpx

ENV_URL = "CHRYSALIS_LLM_URL"
ENV_MODEL = "CHRYSALIS_LLM_MODEL"
ENV_THINKING = "CHRYSALIS_LLM_THINKING"  # off (default) | on
DEFAULT_URL = "http://127.0.0.1:8080/v1"
DEFAULT_MODEL = "gemma-4-26B-A4B-it-Q4_0"
DEFAULT_TIMEOUT_S = 180.0
PROBE_TIMEOUT_S = 3.0


def thinking_from_env() -> bool:
    value = os.environ.get(ENV_THINKING, "off").strip().lower()
    if value not in ("off", "on"):
        raise ValueError(f"{ENV_THINKING} must be 'off' or 'on', got {value!r}")
    return value == "on"


class LlmClient:
    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        thinking: bool | None = None,
    ) -> None:
        self.base_url = (base_url or os.environ.get(ENV_URL) or DEFAULT_URL).rstrip("/")
        # explicit arg > env var > served id from GET /models (lazy, cached) > DEFAULT_MODEL
        self._model: str | None = model or os.environ.get(ENV_MODEL) or None
        self.timeout_s = timeout_s
        self.thinking = thinking_from_env() if thinking is None else thinking

    @property
    def model(self) -> str:
        if self._model is None:
            self._model = self._served_model() or DEFAULT_MODEL
        return self._model

    def _served_model(self) -> str | None:
        """First model id the server lists (llama-server returns its alias); None on any error."""
        try:
            resp = httpx.get(f"{self.base_url}/models", timeout=PROBE_TIMEOUT_S)
            if resp.is_success:
                first = resp.json()["data"][0]["id"]
                return first if isinstance(first, str) and first else None
        except httpx.HTTPError, ValueError, KeyError, IndexError, TypeError:
            pass
        return None

    def derive(self, thinking: bool, timeout_s: float) -> LlmClient:
        """Same server and model, different thinking mode / request timeout."""
        return LlmClient(self.base_url, self.model, timeout_s, thinking)

    def available(self) -> bool:
        try:
            return httpx.get(f"{self.base_url}/models", timeout=PROBE_TIMEOUT_S).is_success
        except httpx.HTTPError:
            return False

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> dict[str, Any]:
        """``{"status":"OK","message":{...},"usage":{prompt_tokens,completion_tokens}}`` or a
        structured ``UNREACHABLE`` / ``ERROR`` result (never an exception)."""
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if not self.thinking:
            body["chat_template_kwargs"] = {"enable_thinking": False}
        if tools:
            body["tools"] = tools
        try:
            resp = httpx.post(
                f"{self.base_url}/chat/completions", json=body, timeout=self.timeout_s
            )
        except httpx.HTTPError as exc:
            return {"status": "UNREACHABLE", "error": f"{type(exc).__name__}: {exc}"}
        if not resp.is_success:
            return {"status": "ERROR", "error": f"HTTP {resp.status_code}", "body": resp.text[:300]}
        try:
            data = resp.json()
            message = dict(data["choices"][0]["message"])
            # Reasoning is measured, then dropped: it must never re-enter the conversation.
            reasoning = message.pop("reasoning_content", None)
            usage = data.get("usage") or {}
            return {
                "status": "OK",
                "message": message,
                "usage": {
                    "prompt_tokens": int(usage.get("prompt_tokens", 0)),
                    "completion_tokens": int(usage.get("completion_tokens", 0)),
                },
                "reasoning_chars": len(reasoning) if isinstance(reasoning, str) else 0,
            }
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            return {"status": "ERROR", "error": f"malformed response: {type(exc).__name__}"}
