# Implementation notes

The implementation follows these rules:

1. Prefer pass-through behavior unless a specific client incompatibility requires a rewrite.
2. Keep state local to a single request/response path.
3. Normalize only the schema and tool fields known to trigger `llama-server` parser failures. For Anthropic schemas, default to a conservative parser-compatible subset and move richer constraints into descriptions unless the operator explicitly selects semantic mode after backend verification.
4. Preserve unknown requests by forwarding them unchanged.
5. Emit protocol-shaped JSON errors instead of panicking.
6. Avoid TLS client dependencies because the intended backend is local HTTP.
7. Treat Codex namespace tools conservatively: flatten namespace tools for the backend, then default to flat unprefixing on responses because the exact Codex namespace wrapper schema is not fixture-proven. Keep any invented wrapper behind an explicit experimental flag.
8. Treat Gemini tool-call IDs as request-local state: generated OpenAI IDs must be reused by later translated Gemini `functionResponse` tool messages. Never use a bare function name as `tool_call_id`; if the prior function call is absent from the request history, synthesize a minimal matching assistant `tool_calls[]` entry immediately before the tool result.
9. Treat Gemini URL model names as informational. Translated Chat Completions requests must use the configured backend model name, not the client-supplied Gemini model segment.
10. Detect Gemini by path or by a narrowly validated Gemini body shape so nonstandard Gemini endpoints still translate while unrelated JSON requests pass through.
11. Treat malformed JSON as a protocol error, not a translation failure: `/v1/responses`, `/v1/messages`, and Gemini generation paths return protocol-shaped 400 responses before backend forwarding.
12. On well-formed request translation failure, log the raw request context and forward the original request bytes to the backend unchanged; do not synthesize a 400 for translator surprises.
13. Anthropic schema normalization must not silently collapse complex tool contracts: parser-risky standard constraints and unsupported branch details are carried into descriptions in default compat mode. The opt-in semantic mode preserves those standard constraints only for verified backends. Anthropic message-shaped responses and SSE content blocks receive safe defaults such as `cache_control` only when omitted; Anthropic error-shaped responses pass through unchanged.
14. Responses SSE rewriting must preserve original SSE metadata (`event:`, `id:`, `retry:`, comments) and rewrite only collected `data:` payloads; clients may dispatch on the event name.
15. Gemini content translation must preserve mixed-part order within each `contents[]` item. Gemini streaming remains a JSON `data:`-frame transformer, suppresses OpenAI `[DONE]`, and should be verified with captured Gemini CLI fixtures before claiming full wire compatibility for every client.
16. Gemini error conversion must preserve upstream HTTP error semantics: translate OpenAI-style error bodies into Gemini-shaped error bodies whose `error.code` matches the upstream HTTP error status when available, with canonical Gemini status strings derived from that code.
17. Do not forward Gemini `generationConfig.candidateCount` to OpenAI `n` unless the response translator supports every returned choice. The current brief-backed implementation maps only the first OpenAI choice, so it intentionally stays single-candidate.
18. Detect Ollama only by exact native API paths: `/api/chat`, `/api/generate`, `/api/tags`, `/api/show`, `/api/pull`, and `/api/delete`. Do not infer Ollama from arbitrary JSON bodies.
19. Treat Ollama `/api/chat` and `/api/generate` as streaming by default. Only set outgoing Chat Completions `stream` to false when the client explicitly sends `stream: false`.
20. Convert backend OpenAI SSE frames to Ollama newline-delimited JSON and return `application/x-ndjson`; suppress the OpenAI `[DONE]` sentinel.
21. Keep Ollama tool-call ID handling request-local. Generate deterministic IDs when inbound tool calls omit IDs, then reuse those IDs for later tool result messages by tool name and order.
22. Translate Ollama `/api/tags` through `/v1/models`; translate `/api/show` by querying backend `/v1/models`; synthesize `/api/pull` and `/api/delete` responses because model lifecycle remains the wrapper's responsibility.
23. Map only Ollama options with direct Chat Completions equivalents and pass model names through unchanged.
24. Return Ollama-shaped errors for malformed JSON on native Ollama paths without contacting the backend.
25. For Ollama streaming requests, do not assume every backend response is SSE. Buffer any finite upstream response whose content type is not `text/event-stream`: translate JSON success bodies, convert JSON errors, and convert non-JSON bodies into a single Ollama NDJSON error line so clients receive debuggable failures.
26. Parse Ollama streaming SSE boundaries over raw bytes, not decoded network chunks. Decode UTF-8 only after a complete SSE frame arrives so a multibyte character split across transport chunks cannot become replacement characters before JSON parsing.

The pure translation code lives in `src/lib.rs`; the HTTP proxy lives in `src/main.rs`. Unit tests focus on deterministic JSON transforms so protocol changes remain easy to review.
27. Keep native Ollama compatibility evidence split by provenance. `fixtures/ollama/observed/` must contain client-emitted HTTP envelopes for compatibility-sensitive request and client-error shapes; `fixtures/ollama/integration/` covers backend response translations that clients cannot emit directly; `fixtures/ollama/synthetic/` remains for focused edge cases such as backend error conversion and split-frame handling. Current observed coverage comes from `ollama-python` 0.6.2 and `ollama-js` 0.6.3 across tags, show, pull, delete, streaming/non-streaming chat and generate, tool-call streaming, and backend-error cases.
28. Accept both `model` and `name` for Ollama lifecycle model names because the official Python client sends `model` while the official JavaScript client sends `name` for pull/delete.
29. Do not pass client `Accept-Encoding` through to the backend on requests the proxy may translate. Backend requests set `Accept-Encoding: identity`; translated response paths reject non-identity `Content-Encoding` before JSON or SSE parsing, and Ollama streaming emits one NDJSON error row for that case.
30. Keep `/api/tags` idempotent when backend `/v1/models` rows omit `created`: prefer backend-reported timestamp fields (`created`, `created_at`, `modified_at`, including metadata variants) and otherwise emit the stable unknown timestamp `1970-01-01T00:00:00Z`; never use wall-clock time for model-list fallback metadata.
