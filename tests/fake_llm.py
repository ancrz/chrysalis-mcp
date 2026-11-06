"""Fake OpenAI-compatible chat server (loopback, thread): scripted replies, recorded requests."""

import json
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


def reply_text(
    text: str, prompt: int = 10, completion: int = 5, reasoning: str | None = None
) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": text}
    if reasoning is not None:
        message["reasoning_content"] = reasoning
    return {
        "choices": [{"message": message}],
        "usage": {"prompt_tokens": prompt, "completion_tokens": completion},
    }


def reply_tool(
    arguments: Any,
    name: str = "chrysalis",
    call_id: str = "call_1",
    prompt: int = 20,
    completion: int = 8,
) -> dict[str, Any]:
    args = arguments if isinstance(arguments, str) else json.dumps(arguments)
    return {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {"name": name, "arguments": args},
                        }
                    ],
                }
            }
        ],
        "usage": {"prompt_tokens": prompt, "completion_tokens": completion},
    }


class FakeLlm:
    def __init__(self) -> None:
        self.script: list[Any] = []  # dict body, or int (HTTP status), or str (raw body)
        self.requests: list[dict[str, Any]] = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args: object) -> None:
                pass

            def do_GET(self) -> None:
                self._send(200, json.dumps({"data": [{"id": "fake"}]}))

            def do_POST(self) -> None:
                size = int(self.headers.get("Content-Length", 0))
                outer.requests.append(json.loads(self.rfile.read(size)))
                item = outer.script.pop(0) if outer.script else reply_text("(script exhausted)")
                if isinstance(item, int):
                    self._send(item, "{}")
                else:
                    self._send(200, item if isinstance(item, str) else json.dumps(item))

            def _send(self, status: int, body: str) -> None:
                data = body.encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self._server.server_address[1]}/v1"
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    def __enter__(self) -> FakeLlm:
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._server.shutdown()
        self._server.server_close()


def dead_url() -> str:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    return f"http://127.0.0.1:{port}/v1"
