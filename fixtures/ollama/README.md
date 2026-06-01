# Ollama fixtures

This directory is split by provenance:

- `observed/` contains captured client-originating HTTP envelopes from Ollama-compatible client libraries, expected OpenAI Chat Completions rewrites for chat/generate, expected backend routing for tags/show, deterministic success transcripts, and client-observed error fixtures.
- `integration/` contains backend-response translation fixtures that cannot come directly from client-originating request captures, such as backend OpenAI SSE tool-call streams.
- `synthetic/` contains focused edge-case fixtures that are hard to capture from normal clients but still need deterministic checks.

Observed fixtures currently cover:

- `ollama-python` 0.6.2: `/api/tags`, `/api/show`, `/api/pull`, `/api/delete`, streaming and non-streaming `/api/chat`, streaming and non-streaming `/api/generate`, and streaming/non-streaming backend error handling as an actual client exception.
- `ollama-js` 0.6.3: `/api/tags`, `/api/show`, `/api/pull`, `/api/delete`, non-streaming `/api/chat`, non-streaming `/api/generate`, streaming `/api/chat` with tools, and streaming backend error handling as an actual client exception.

Each observed `*.request.json` file stores the captured method, path, stable headers, and JSON body. The localhost port and `content-length` are redacted. Companion files use these suffixes:

- `*.expected-chat.json`: expected OpenAI Chat Completions body after proxy request rewrite.
- `*.expected-backend.json`: expected backend routing metadata for endpoints that do not rewrite into Chat Completions.
- `*.response.json`: deterministic JSON response accepted by the client during capture.
- `*.response.ndjson`: deterministic line-delimited response accepted by streaming client loops.
- `*.client-error.json`: exception shape reported by the actual client against a proxy-shaped backend error response.

Refresh observed fixtures with:

```bash
python3 -m pip install ollama==0.6.2
npm install ollama@0.6.3
python3 scripts/capture_ollama_observed_fixtures.py
python3 scripts/validate_ollama_observed_fixtures.py
```

When changing compatibility-sensitive Ollama behavior, add or refresh an observed fixture that captures the affected client request or client-observed error shape first. Use `integration/` for backend-response translations that clients cannot directly emit, such as OpenAI SSE tool-call deltas and finite non-SSE upstream responses.

The synthetic `streaming_backend_error.*` fixtures cover the case where an Ollama chat or generate request asked for streaming, but the backend returned a finite JSON error instead of OpenAI SSE. The proxy should return one Ollama-shaped NDJSON error line rather than an empty stream.

The `streaming_non_sse_text_plain.*` and `streaming_non_sse_text_html.*` integration fixtures cover successful upstream HTTP responses with non-SSE content types during an Ollama streaming request. They check that the proxy returns one explicit NDJSON error row instead of passing the body into the SSE transformer and producing an empty stream.

The `tags_missing_created.*` integration fixture covers backend model-list rows without creation timestamps. It locks the fallback `modified_at` value to `1970-01-01T00:00:00Z` rather than wall-clock time so `/api/tags` stays stable across repeated calls.
