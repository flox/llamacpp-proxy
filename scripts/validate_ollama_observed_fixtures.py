#!/usr/bin/env python3
"""Validate native Ollama observed and integration fixtures without Rust."""
from __future__ import annotations

import json
import pathlib
import sys
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
OBSERVED = ROOT / "fixtures" / "ollama" / "observed"
INTEGRATION = ROOT / "fixtures" / "ollama" / "integration"

REQUIRED_BASENAMES = {
    "python_0_6_2_tags",
    "python_0_6_2_show",
    "python_0_6_2_pull",
    "python_0_6_2_delete",
    "python_0_6_2_chat_nonstream",
    "python_0_6_2_generate_nonstream",
    "python_0_6_2_chat_stream",
    "python_0_6_2_generate_stream",
    "python_0_6_2_chat_stream_backend_error",
    "python_0_6_2_chat_nonstream_backend_error",
    "js_0_6_3_tags",
    "js_0_6_3_show",
    "js_0_6_3_pull",
    "js_0_6_3_delete",
    "js_0_6_3_chat_nonstream",
    "js_0_6_3_generate_nonstream",
    "js_0_6_3_chat_tools_stream",
    "js_0_6_3_chat_stream_backend_error",
}

EXPECTED_METHODS = {
    "/api/tags": "GET",
    "/api/show": "POST",
    "/api/pull": "POST",
    "/api/delete": "DELETE",
    "/api/chat": "POST",
    "/api/generate": "POST",
}


def load_json(path: pathlib.Path) -> Any:
    try:
        return json.loads(path.read_text())
    except Exception as exc:
        raise SystemExit(f"{path}: invalid JSON: {exc}") from exc


def load_ndjson(path: pathlib.Path) -> list[Any]:
    rows: list[Any] = []
    for line_no, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except Exception as exc:
            raise SystemExit(f"{path}:{line_no}: invalid NDJSON JSON object: {exc}") from exc
    if not rows:
        raise SystemExit(f"{path}: expected at least one NDJSON row")
    return rows


def expect(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def basename(path: pathlib.Path) -> str:
    name = path.name
    for suffix in [
        ".expected-chat.json",
        ".expected-backend.json",
        ".client-error.json",
        ".request.json",
        ".response.ndjson",
        ".response.json",
    ]:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def validate_request_fixture(path: pathlib.Path) -> None:
    root = load_json(path)
    meta = root.get("_fixture", {})
    base = basename(path)
    expected_chat = OBSERVED / f"{base}.expected-chat.json"
    expected_backend = OBSERVED / f"{base}.expected-backend.json"
    response_json = OBSERVED / f"{base}.response.json"
    response_ndjson = OBSERVED / f"{base}.response.ndjson"
    client_error = OBSERVED / f"{base}.client-error.json"

    expect(meta.get("kind") == "observed-client-traffic", f"{path}: missing observed-client marker")
    expect(bool(meta.get("client")), f"{path}: missing client metadata")
    expect(bool(meta.get("client_version")), f"{path}: missing client_version metadata")
    expect(root.get("path") in EXPECTED_METHODS, f"{path}: unexpected Ollama path {root.get('path')!r}")
    expect(root.get("method") == EXPECTED_METHODS[root["path"]], f"{path}: unexpected method for {root['path']}")
    headers = root.get("headers")
    expect(isinstance(headers, dict), f"{path}: headers must be an object")
    expect("user-agent" in headers, f"{path}: observed fixture should include client user-agent")

    body = root.get("body")
    if root["path"] == "/api/tags":
        expect(body is None, f"{path}: tags should not send a body")
        expect(expected_backend.exists(), f"{path}: tags needs expected-backend fixture")
    else:
        expect(isinstance(body, dict), f"{path}: body must be an object")

    if root["path"] in {"/api/chat", "/api/generate"}:
        expect(isinstance(body.get("model"), str) and body["model"], f"{path}: body.model must be non-empty")
        if client_error.exists():
            error = load_json(client_error)
            expect(error.get("status_code") == 502, f"{client_error}: expected status_code 502")
            expect("backend returned HTTP 502" in error.get("message", ""), f"{client_error}: missing backend error text")
        else:
            expect(expected_chat.exists(), f"{path}: chat/generate fixtures need expected-chat")
            expected = load_json(expected_chat)
            expect(expected.get("model") == body.get("model"), f"{expected_chat}: model does not match request")
            expect(expected.get("stream") is body.get("stream"), f"{expected_chat}: stream does not match request")
            if body.get("stream") is True:
                rows = load_ndjson(response_ndjson)
                expect(rows[-1].get("done") is True, f"{response_ndjson}: final row must have done:true")
            else:
                response = load_json(response_json)
                expect(response.get("done") is True, f"{response_json}: expected done:true")

    if root["path"] in {"/api/show", "/api/pull", "/api/delete"}:
        expect(body.get("model") or body.get("name"), f"{path}: lifecycle body must name a model")
        if root["path"] == "/api/show":
            expect(expected_backend.exists(), f"{path}: show needs expected-backend fixture")
            backend = load_json(expected_backend)
            expect(backend.get("path") == "/v1/models", f"{expected_backend}: show should query /v1/models")
        else:
            response = load_json(response_json)
            expect(response.get("status") == "success", f"{response_json}: expected success status")
            expect(response.get("model") == "qwen3:32b", f"{response_json}: expected normalized model name")


def validate_matrix() -> None:
    basenames = {basename(path) for path in OBSERVED.glob("*.request.json")}
    missing = sorted(REQUIRED_BASENAMES - basenames)
    expect(not missing, f"{OBSERVED}: missing required observed fixtures: {', '.join(missing)}")


def validate_integration_fixtures() -> None:
    sse_path = INTEGRATION / "chat_tool_call_stream.backend.sse"
    ndjson_path = INTEGRATION / "chat_tool_call_stream.expected.ndjson"
    expect(sse_path.exists(), f"{sse_path}: missing backend SSE integration fixture")
    rows = load_ndjson(ndjson_path)
    expect(rows[-1].get("done") is True, f"{ndjson_path}: final row must have done:true")
    tool_calls = rows[-1].get("message", {}).get("tool_calls", [])
    expect(tool_calls, f"{ndjson_path}: expected tool call in final chat row")
    first = tool_calls[0].get("function", {})
    expect(first.get("name") == "read_file", f"{ndjson_path}: expected read_file tool call")
    expect(first.get("arguments", {}).get("path") == "Cargo.toml", f"{ndjson_path}: expected parsed tool arguments")
    expect("tool_calls" in sse_path.read_text(), f"{sse_path}: backend fixture should include OpenAI tool_calls deltas")

    non_sse_cases = {
        "streaming_non_sse_text_plain": "text/plain",
        "streaming_non_sse_text_html": "text/html",
    }
    for base, media_type in non_sse_cases.items():
        headers = load_json(INTEGRATION / f"{base}.headers.json")
        expect(headers.get("status") == 200, f"{base}: fixture must model a successful upstream response")
        expect(headers.get("content-type", "").startswith(media_type), f"{base}: expected {media_type} content type")
        rows = load_ndjson(INTEGRATION / f"{base}.expected.ndjson")
        expect(len(rows) == 1, f"{base}: expected exactly one NDJSON error row")
        expect("non-SSE response" in rows[0].get("error", ""), f"{base}: expected explicit non-SSE error")

    compressed_headers = load_json(INTEGRATION / "streaming_compressed_backend.headers.json")
    expect(compressed_headers.get("status") == 200, "streaming_compressed_backend: expected status 200")
    expect(
        compressed_headers.get("content-encoding") == "gzip",
        "streaming_compressed_backend: expected gzip content-encoding",
    )
    compressed_rows = load_ndjson(INTEGRATION / "streaming_compressed_backend.expected.ndjson")
    expect(len(compressed_rows) == 1, "streaming_compressed_backend: expected one NDJSON error row")
    expect(
        "Content-Encoding" in compressed_rows[0].get("error", ""),
        "streaming_compressed_backend: expected explicit Content-Encoding error",
    )


def main() -> int:
    validate_matrix()
    requests = sorted(OBSERVED.glob("*.request.json"))
    if not requests:
        raise SystemExit(f"{OBSERVED}: no observed request fixtures found")
    for request in requests:
        validate_request_fixture(request)
    validate_integration_fixtures()
    print(f"validated {len(requests)} observed Ollama request fixture set(s) plus integration fixtures")
    return 0


if __name__ == "__main__":
    sys.exit(main())
