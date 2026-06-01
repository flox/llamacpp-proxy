# Ollama integration fixtures

These fixtures exercise proxy-side transformations that cannot come directly from client-originating request captures.

- `chat_tool_call_stream.backend.sse` is a backend OpenAI Chat Completions SSE transcript with fragmented tool-call arguments.
- `chat_tool_call_stream.expected.ndjson` is the Ollama `/api/chat` NDJSON transcript expected from that backend stream.
- `streaming_non_sse_text_plain.*` models a successful `text/plain` upstream response during an Ollama streaming request and expects one NDJSON error row.
- `streaming_non_sse_text_html.*` models a successful `text/html` upstream response during an Ollama streaming request and expects one NDJSON error row.

Observed client fixtures prove request and client-error behavior. Integration fixtures prove backend response translation behavior.

- `streaming_compressed_backend.*` models an upstream that ignores `Accept-Encoding: identity` and sends `Content-Encoding: gzip`; translated streaming paths must return one explicit NDJSON error row instead of parsing compressed bytes.
- `tags_missing_created.*` models a backend `/v1/models` row without `created`; the expected Ollama `/api/tags` output uses the stable unknown `modified_at` value `1970-01-01T00:00:00Z` so repeated calls over the same backend state remain identical.
