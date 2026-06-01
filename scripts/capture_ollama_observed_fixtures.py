#!/usr/bin/env python3
"""Capture official Ollama Python/JS client HTTP envelopes for fixture refreshes.

Prerequisites:
  python3 -m pip install ollama==0.6.2
  npm install ollama@0.6.3

Run from the repository root. The script starts a localhost recorder, invokes
client-library calls, redacts localhost-only fields, and writes observed request
fixtures plus deterministic response/error transcripts.
"""
from __future__ import annotations

import http.server
import importlib.metadata
import json
import os
import pathlib
import queue
import socket
import subprocess
import sys
import threading
from datetime import datetime, timezone
from typing import Any, Callable, Iterable

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "fixtures" / "ollama" / "observed"
OUT.mkdir(parents=True, exist_ok=True)
STABLE_HEADER_KEYS = {"accept", "content-type", "user-agent", "content-length", "accept-encoding"}
MODEL = "qwen3:32b"
GENERATE_MODEL = "codellama:code"
CREATED_AT = "2026-05-31T12:00:00Z"


def now_rfc3339() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def json_bytes(value: Any) -> bytes:
    return json.dumps(value, separators=(",", ":")).encode("utf-8")


def ndjson_bytes(rows: Iterable[Any]) -> bytes:
    return "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows).encode("utf-8")


def stable_model_name(body: Any) -> str:
    if isinstance(body, dict):
        return body.get("model") or body.get("name") or MODEL
    return MODEL


def tags_response() -> dict[str, Any]:
    return {
        "models": [
            {
                "name": MODEL,
                "model": MODEL,
                "modified_at": CREATED_AT,
                "size": 123456789,
                "digest": "sha256:abc123",
                "details": {
                    "format": "gguf",
                    "family": "qwen3",
                    "parameter_size": "32.5B",
                    "quantization_level": "Q4_K_M",
                },
            }
        ]
    }


def show_response(model: str) -> dict[str, Any]:
    return {
        "modelfile": f"FROM {model}\n",
        "parameters": "temperature 0.2",
        "template": "{{ .Prompt }}",
        "license": "apache-2.0",
        "details": {
            "format": "gguf",
            "family": "qwen3",
            "parameter_size": "32.5B",
            "quantization_level": "Q4_K_M",
        },
        "model_info": {
            "general.architecture": "qwen3",
            "general.parameter_count": 32500000000,
        },
        "capabilities": ["completion", "tools"],
        "modified_at": CREATED_AT,
    }


def success_behavior(method: str, path: str, body: Any) -> tuple[int, str, bytes]:
    model = stable_model_name(body)
    if method == "GET" and path == "/api/tags":
        return 200, "application/json", json_bytes(tags_response())
    if method == "POST" and path == "/api/show":
        return 200, "application/json", json_bytes(show_response(model))
    if method == "POST" and path == "/api/pull":
        return 200, "application/json", json_bytes({"status": "success"})
    if method == "DELETE" and path == "/api/delete":
        return 200, "application/json", json_bytes({"status": "success"})
    if method == "POST" and path == "/api/chat":
        if isinstance(body, dict) and body.get("stream") is True:
            rows = [
                {"model": model, "created_at": CREATED_AT, "message": {"role": "assistant", "content": "hello"}, "done": False},
                {"model": model, "created_at": CREATED_AT, "message": {"role": "assistant", "content": ""}, "done": True, "done_reason": "stop", "total_duration": 1000000, "load_duration": 1000, "prompt_eval_count": 4, "prompt_eval_duration": 2000, "eval_count": 1, "eval_duration": 3000},
            ]
            return 200, "application/x-ndjson", ndjson_bytes(rows)
        return 200, "application/json", json_bytes({"model": model, "created_at": CREATED_AT, "message": {"role": "assistant", "content": "hello"}, "done": True, "done_reason": "stop", "total_duration": 1000000, "load_duration": 1000, "prompt_eval_count": 4, "prompt_eval_duration": 2000, "eval_count": 1, "eval_duration": 3000})
    if method == "POST" and path == "/api/generate":
        if isinstance(body, dict) and body.get("stream") is True:
            rows = [
                {"model": model, "created_at": CREATED_AT, "response": "hello", "done": False},
                {"model": model, "created_at": CREATED_AT, "response": "", "done": True, "done_reason": "stop", "total_duration": 1000000, "load_duration": 1000, "prompt_eval_count": 3, "prompt_eval_duration": 2000, "eval_count": 1, "eval_duration": 3000},
            ]
            return 200, "application/x-ndjson", ndjson_bytes(rows)
        return 200, "application/json", json_bytes({"model": model, "created_at": CREATED_AT, "response": "hello", "done": True, "done_reason": "stop", "total_duration": 1000000, "load_duration": 1000, "prompt_eval_count": 3, "prompt_eval_duration": 2000, "eval_count": 1, "eval_duration": 3000})
    return 404, "application/json", json_bytes({"error": f"unexpected {method} {path}"})


def backend_error_behavior(_method: str, _path: str, _body: Any) -> tuple[int, str, bytes]:
    return 502, "application/x-ndjson", b'{"error":"backend returned HTTP 502: model failed before stream start"}\n'


class Recorder(http.server.BaseHTTPRequestHandler):
    server_version = "OllamaFixtureRecorder/2.0"
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args: Any) -> None:
        return

    def do_GET(self) -> None:  # noqa: N802
        self._handle()

    def do_POST(self) -> None:  # noqa: N802
        self._handle()

    def do_DELETE(self) -> None:  # noqa: N802
        self._handle()

    def _handle(self) -> None:
        length = int(self.headers.get("content-length", "0"))
        raw_body = self.rfile.read(length) if length else b""
        body = json.loads(raw_body.decode("utf-8")) if raw_body else None
        headers = {
            key.lower(): value
            for key, value in self.headers.items()
            if key.lower() in STABLE_HEADER_KEYS
        }
        if "content-length" in headers:
            headers["content-length"] = "<redacted>"
        self.server.capture_queue.put(
            {"method": self.command, "path": self.path, "headers": headers, "body": body}
        )
        status, content_type, payload = self.server.behavior(self.command, self.path, body)
        self.send_response(status)
        self.send_header("content-type", content_type)
        self.send_header("content-length", str(len(payload)))
        self.send_header("connection", "close")
        self.end_headers()
        self.wfile.write(payload)
        self.close_connection = True


class ClientObservedError(Exception):
    def __init__(self, payload: dict[str, Any]) -> None:
        super().__init__(payload.get("message", "client error"))
        self.client_error_class = payload.get("error_class") or "Error"
        self.status_code = payload.get("status_code")


def with_server(fn: Callable[[str], Any], behavior: Callable[[str, str, Any], tuple[int, str, bytes]] = success_behavior) -> tuple[dict[str, Any], Any]:
    port = free_port()
    captures: queue.Queue[dict[str, Any]] = queue.Queue()
    server = http.server.ThreadingHTTPServer(("127.0.0.1", port), Recorder)
    server.capture_queue = captures
    server.behavior = behavior
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    result: Any = None
    try:
        try:
            result = fn(f"http://127.0.0.1:{port}")
        except Exception as exc:  # client-observed error fixture
            result = {
                "error_class": getattr(exc, "client_error_class", type(exc).__name__),
                "message": str(exc),
                "status_code": getattr(exc, "status_code", None) or getattr(exc, "status", None),
            }
        return captures.get(timeout=5), result
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def fixture_meta(client: str, client_version: str, note: str) -> dict[str, Any]:
    return {
        "kind": "observed-client-traffic",
        "client": client,
        "client_version": client_version,
        "captured_at": now_rfc3339(),
        "capture_method": "Captured from an official client library call against a local recorder.",
        "redactions": ["localhost port", "content-length"],
        "response_note": note,
    }


def write_json(path: pathlib.Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=False) + "\n")


def write_request(name: str, client: str, client_version: str, capture: dict[str, Any], response_note: str) -> None:
    write_json(OUT / f"{name}.request.json", {"_fixture": fixture_meta(client, client_version, response_note), **capture})


def write_expected_chat(name: str, expected: dict[str, Any]) -> None:
    write_json(OUT / f"{name}.expected-chat.json", expected)


def write_expected_backend(name: str, expected: dict[str, Any]) -> None:
    write_json(OUT / f"{name}.expected-backend.json", expected)


def write_response_json(name: str, value: Any) -> None:
    write_json(OUT / f"{name}.response.json", value)


def write_response_ndjson(name: str, rows: list[Any]) -> None:
    (OUT / f"{name}.response.ndjson").write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows)
    )


def write_client_error(name: str, error: Any) -> None:
    write_json(OUT / f"{name}.client-error.json", error)


def chat_expected(stream: bool) -> dict[str, Any]:
    return {
        "model": MODEL,
        "messages": [{"role": "user", "content": "Say hello."}],
        "stream": stream,
        "temperature": 0.2,
        "max_tokens": 16,
    }


def generate_expected(stream: bool) -> dict[str, Any]:
    return {
        "model": GENERATE_MODEL,
        "messages": [
            {"role": "system", "content": "Return code only."},
            {"role": "user", "content": "Write hello world in Rust."},
        ],
        "stream": stream,
        "response_format": {"type": "json_object"},
        "top_p": 0.9,
        "seed": 7,
    }


def chat_rows(model: str = MODEL) -> list[Any]:
    return [
        {"model": model, "created_at": CREATED_AT, "message": {"role": "assistant", "content": "hello"}, "done": False},
        {"model": model, "created_at": CREATED_AT, "message": {"role": "assistant", "content": ""}, "done": True, "done_reason": "stop", "total_duration": 1000000, "load_duration": 1000, "prompt_eval_count": 4, "prompt_eval_duration": 2000, "eval_count": 1, "eval_duration": 3000},
    ]


def generate_rows(model: str = GENERATE_MODEL) -> list[Any]:
    return [
        {"model": model, "created_at": CREATED_AT, "response": "hello", "done": False},
        {"model": model, "created_at": CREATED_AT, "response": "", "done": True, "done_reason": "stop", "total_duration": 1000000, "load_duration": 1000, "prompt_eval_count": 3, "prompt_eval_duration": 2000, "eval_count": 1, "eval_duration": 3000},
    ]


def run_python_chat(host: str, *, stream: bool) -> None:
    import ollama

    client = ollama.Client(host=host)
    response = client.chat(
        model=MODEL,
        messages=[{"role": "user", "content": "Say hello."}],
        stream=stream,
        options={"temperature": 0.2, "num_predict": 16},
    )
    if stream:
        for _chunk in response:
            pass


def run_python_generate(host: str, *, stream: bool) -> None:
    import ollama

    client = ollama.Client(host=host)
    response = client.generate(
        model=GENERATE_MODEL,
        system="Return code only.",
        prompt="Write hello world in Rust.",
        stream=stream,
        format="json",
        options={"top_p": 0.9, "seed": 7},
    )
    if stream:
        for _chunk in response:
            pass


def run_python_tags(host: str) -> None:
    import ollama

    ollama.Client(host=host).list()


def run_python_show(host: str) -> None:
    import ollama

    ollama.Client(host=host).show(MODEL)


def run_python_pull(host: str) -> None:
    import ollama

    ollama.Client(host=host).pull(MODEL, stream=False)


def run_python_delete(host: str) -> None:
    import ollama

    ollama.Client(host=host).delete(MODEL)


def js_version() -> str:
    script = "const p=require.resolve('ollama/package.json'); console.log(require(p).version);"
    return subprocess.check_output(["node", "-e", script], cwd=ROOT, text=True).strip()


def run_js(host: str, op: str) -> None:
    js = r'''
const { Ollama } = require('ollama');
const client = new Ollama({ host: process.env.OLLAMA_HOST });
const model = 'qwen3:32b';
async function main() {
  const op = process.env.OLLAMA_OP;
  if (op === 'tags') await client.list();
  if (op === 'show') await client.show({ model });
  if (op === 'pull') await client.pull({ model, stream: false });
  if (op === 'delete') await client.delete({ model });
  if (op === 'chat_nonstream') await client.chat({ model, messages: [{ role: 'user', content: 'Say hello.' }], stream: false, options: { temperature: 0.2, num_predict: 16 } });
  if (op === 'generate_nonstream') await client.generate({ model: 'codellama:code', system: 'Return code only.', prompt: 'Write hello world in Rust.', stream: false, format: 'json', options: { top_p: 0.9, seed: 7 } });
  if (op === 'chat_tools_stream') {
    const stream = await client.chat({
      model,
      messages: [{ role: 'user', content: 'Read Cargo.toml.' }],
      stream: true,
      tools: [{ type: 'function', function: { name: 'read_file', description: 'Read a file from the workspace', parameters: { type: 'object', properties: { path: { type: 'string' } }, required: ['path'] } } }],
      options: { temperature: 0, num_predict: 32 }
    });
    for await (const _chunk of stream) {}
  }
  if (op === 'chat_stream_error') {
    const stream = await client.chat({ model, messages: [{ role: 'user', content: 'Say hello.' }], stream: true });
    for await (const _chunk of stream) {}
  }
}
main().catch(err => { console.error(JSON.stringify({ error_class: err.name, message: err.message, status_code: err.status_code || err.status || null })); process.exit(1); });
'''
    env = os.environ.copy()
    env["OLLAMA_HOST"] = host
    env["OLLAMA_OP"] = op
    proc = subprocess.run(["node", "-e", js], cwd=ROOT, env=env, text=True, capture_output=True)
    if proc.returncode != 0:
        lines = [line for line in proc.stderr.strip().splitlines() if line.strip()]
        if lines:
            try:
                raise ClientObservedError(json.loads(lines[-1]))
            except json.JSONDecodeError:
                raise RuntimeError(lines[-1])
        raise RuntimeError(proc.stdout or f"node exited {proc.returncode}")


def run_js_capture(host: str, op: str) -> None:
    run_js(host, op)


def capture_success(name: str, client: str, version: str, fn: Callable[[str], Any], *, expected_chat: dict[str, Any] | None = None, response_json: Any | None = None, response_rows: list[Any] | None = None, expected_backend: dict[str, Any] | None = None) -> None:
    capture, _result = with_server(fn)
    response_note = "Recorder returned deterministic success data so the client call could finish."
    write_request(name, client, version, capture, response_note)
    if expected_chat is not None:
        write_expected_chat(name, expected_chat)
    if response_json is not None:
        write_response_json(name, response_json)
    if response_rows is not None:
        write_response_ndjson(name, response_rows)
    if expected_backend is not None:
        write_expected_backend(name, expected_backend)


def capture_error(name: str, client: str, version: str, fn: Callable[[str], Any]) -> None:
    capture, error = with_server(fn, backend_error_behavior)
    write_request(name, client, version, capture, "Recorder returned the proxy-shaped backend error response for client exception capture.")
    write_client_error(name, error)


def main() -> int:
    python_version = importlib.metadata.version("ollama")
    javascript_version = js_version()

    capture_success("python_0_6_2_tags", "ollama-python", python_version, run_python_tags, expected_backend={"method": "GET", "path": "/v1/models", "body": None, "stream": False, "response_kind": "tags"}, response_json=tags_response())
    capture_success("python_0_6_2_show", "ollama-python", python_version, run_python_show, expected_backend={"method": "GET", "path": "/v1/models", "body": None, "stream": False, "response_kind": "show", "requested_model": MODEL}, response_json=show_response(MODEL))
    capture_success("python_0_6_2_pull", "ollama-python", python_version, run_python_pull, response_json={"status": "success", "model": MODEL})
    capture_success("python_0_6_2_delete", "ollama-python", python_version, run_python_delete, response_json={"status": "success", "model": MODEL})
    capture_success("python_0_6_2_chat_nonstream", "ollama-python", python_version, lambda host: run_python_chat(host, stream=False), expected_chat=chat_expected(False), response_json={"model": MODEL, "created_at": CREATED_AT, "message": {"role": "assistant", "content": "hello"}, "done": True, "done_reason": "stop", "total_duration": 1000000, "load_duration": 1000, "prompt_eval_count": 4, "prompt_eval_duration": 2000, "eval_count": 1, "eval_duration": 3000})
    capture_success("python_0_6_2_generate_nonstream", "ollama-python", python_version, lambda host: run_python_generate(host, stream=False), expected_chat=generate_expected(False), response_json={"model": GENERATE_MODEL, "created_at": CREATED_AT, "response": "hello", "done": True, "done_reason": "stop", "total_duration": 1000000, "load_duration": 1000, "prompt_eval_count": 3, "prompt_eval_duration": 2000, "eval_count": 1, "eval_duration": 3000})
    capture_success("python_0_6_2_chat_stream", "ollama-python", python_version, lambda host: run_python_chat(host, stream=True), expected_chat=chat_expected(True), response_rows=chat_rows(MODEL))
    capture_success("python_0_6_2_generate_stream", "ollama-python", python_version, lambda host: run_python_generate(host, stream=True), expected_chat=generate_expected(True), response_rows=generate_rows(GENERATE_MODEL))
    capture_error("python_0_6_2_chat_stream_backend_error", "ollama-python", python_version, lambda host: run_python_chat(host, stream=True))
    capture_error("python_0_6_2_chat_nonstream_backend_error", "ollama-python", python_version, lambda host: run_python_chat(host, stream=False))

    capture_success("js_0_6_3_tags", "ollama-js", javascript_version, lambda host: run_js_capture(host, "tags"), expected_backend={"method": "GET", "path": "/v1/models", "body": None, "stream": False, "response_kind": "tags"}, response_json=tags_response())
    capture_success("js_0_6_3_show", "ollama-js", javascript_version, lambda host: run_js_capture(host, "show"), expected_backend={"method": "GET", "path": "/v1/models", "body": None, "stream": False, "response_kind": "show", "requested_model": MODEL}, response_json=show_response(MODEL))
    capture_success("js_0_6_3_pull", "ollama-js", javascript_version, lambda host: run_js_capture(host, "pull"), response_json={"status": "success", "model": MODEL})
    capture_success("js_0_6_3_delete", "ollama-js", javascript_version, lambda host: run_js_capture(host, "delete"), response_json={"status": "success", "model": MODEL})
    capture_success("js_0_6_3_chat_nonstream", "ollama-js", javascript_version, lambda host: run_js_capture(host, "chat_nonstream"), expected_chat=chat_expected(False), response_json={"model": MODEL, "created_at": CREATED_AT, "message": {"role": "assistant", "content": "hello"}, "done": True})
    capture_success("js_0_6_3_generate_nonstream", "ollama-js", javascript_version, lambda host: run_js_capture(host, "generate_nonstream"), expected_chat=generate_expected(False), response_json={"model": GENERATE_MODEL, "created_at": CREATED_AT, "response": "hello", "done": True})
    capture_success(
        "js_0_6_3_chat_tools_stream",
        "ollama-js",
        javascript_version,
        lambda host: run_js_capture(host, "chat_tools_stream"),
        expected_chat={
            "model": MODEL,
            "messages": [{"role": "user", "content": "Read Cargo.toml."}],
            "stream": True,
            "temperature": 0,
            "max_tokens": 32,
            "tools": [{"type": "function", "function": {"name": "read_file", "description": "Read a file from the workspace", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}}],
        },
        response_rows=chat_rows(MODEL),
    )
    capture_error("js_0_6_3_chat_stream_backend_error", "ollama-js", javascript_version, lambda host: run_js_capture(host, "chat_stream_error"))

    print(f"wrote observed fixtures to {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
