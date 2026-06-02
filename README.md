# llamacpp-proxy

`llamacpp-proxy` is a small Rust HTTP proxy that makes `llama-server` easier to use with coding-agent harnesses that speak OpenAI Responses, Anthropic Messages, Gemini native APIs, and Ollama native APIs.

It listens locally, translates only the protocol pieces that `llama-server` cannot currently parse, forwards to a single `llama-server` backend, and rewrites responses when the client expects a different wire format.

## Status

This crate is designed as a rigorous starting implementation with unit-tested protocol transforms. It includes:

- OpenAI Responses tool normalization for Codex-style non-function and namespace tools, with conservative flat namespace-call unprefixing by default.
- Anthropic Messages JSON Schema normalization for Claude Code tool schemas, with a conservative parser-compatible default and an opt-in semantic mode for verified backends.
- Gemini `generateContent` and `streamGenerateContent` translation to OpenAI Chat Completions, including strict-safe tool-call ID mapping for multi-turn function responses, synthetic call IDs for orphan tool results, part-order-preserving mixed content translation, content-based Gemini request detection, backend-model forwarding that ignores Gemini path model names, single-candidate request semantics, and HTTP-status-preserving Gemini error conversion.
- Ollama native API translation for `/api/chat`, `/api/generate`, `/api/tags`, and `/api/show`, plus synthetic model-lifecycle responses for `/api/pull` and `/api/delete`. `/api/show` queries backend `/v1/models` and reshapes the selected backend model record into Ollama show-detail JSON. Ollama chat and generate requests preserve Ollama's default streaming behavior by requesting OpenAI Chat Completions streaming unless the client sends `stream: false`, converting backend SSE frames to Ollama NDJSON, parsing SSE frame boundaries over bytes so split UTF-8 survives intact, and converting backend JSON error bodies into Ollama-shaped NDJSON errors when a stream fails before SSE begins.
- Health check with backend reachability and Ollama fallback probing.
- Concurrent async request handling through Tokio + Hyper.
- Streaming pass-through for ordinary paths, SSE translation for Gemini text/tool-call chunks, OpenAI Responses SSE metadata-preserving data rewrites, and Anthropic SSE response defaulting for Claude Code compatibility.
- Conservative request/response header forwarding that drops hop-by-hop headers and replaces inbound `Accept-Encoding` with `identity` for backend requests whose bodies may be translated.
- Protocol-shaped error responses for malformed JSON on recognized protocol endpoints, backend failures, timeouts, and hard proxy limits; well-formed translation failures are logged with the raw request body and forwarded as-is best-effort.

## Build

```bash
cargo build --release
```

For a smaller binary, use:

```bash
cargo build --profile release-small
```

The release profiles enable LTO, one codegen unit, `panic = "abort"`, and symbol stripping.

## Validate

```bash
./scripts/validate.sh
```

The script runs formatting, Clippy, unit tests, release build, and a binary-size report.

## Run

```bash
llamacpp-proxy \
  --listen 127.0.0.1:8081 \
  --backend 127.0.0.1:8080 \
  --backend-api-key llamacpp-local \
  --backend-model local-model \
  --gemini-listen 127.0.0.1:8082
```

`--backend` accepts either `ADDR:PORT` or a full URL. If no scheme appears, the proxy assumes `http://`.

## Harness configuration

```bash
# Codex / OpenAI Responses API
export OPENAI_BASE_URL=http://127.0.0.1:8081/v1

# Claude Code / Anthropic Messages API
export ANTHROPIC_BASE_URL=http://127.0.0.1:8081

# Gemini CLI / Gemini native API
export GOOGLE_GEMINI_BASE_URL=http://127.0.0.1:8082

# Ollama native API clients
export OLLAMA_HOST=http://127.0.0.1:8081
```

Chat Completions clients such as aider and OpenCode can point at `http://127.0.0.1:8081/v1` and will pass through unchanged.

## CLI

```text
llamacpp-proxy

Usage:
  llamacpp-proxy [OPTIONS]

Options:
  --listen <ADDR:PORT>                 Proxy listen address [default: 127.0.0.1:8081]
  --backend <ADDR:PORT|URL>            llama-server backend [default: 127.0.0.1:8080]
  --backend-api-key <KEY>              Backend API key [default: llamacpp-local]
  --backend-model <MODEL>              Backend model used for Gemini rewrites and synthetic protocol metadata [default: local-model]
  --gemini-listen <ADDR:PORT>          Optional second listener for GOOGLE_GEMINI_BASE_URL
  --backend-timeout-secs <SECONDS>     Backend timeout [default: 120]
  --max-body-bytes <BYTES>             Max request body [default: 67108864]
  --no-gemini-hardcoded-classifier     Forward Gemini flash-lite classifier requests instead of short-circuiting
  --codex-namespace-response-mode <flat|experimental-wrapped>
                                      Flat unprefixes namespaced calls [default: flat]; experimental-wrapped is opt-in only
  --anthropic-schema-mode <compat|semantic>
                                      compat forwards a small parser-safe schema subset [default: compat]; semantic preserves more standard constraints after backend verification
  -h, --help                           Show help
```

## Protocol behavior

### `/v1/responses`

The proxy rewrites every tool to a `type: "function"` tool before forwarding to `llama-server`.

Namespace tools become flat function tools with a namespace prefix:

```json
{"type":"namespace","name":"multi_agent_v1","tools":[{"type":"function","name":"close_agent"}]}
```

becomes:

```json
{"type":"function","name":"multi_agent_v1__close_agent"}
```

For non-streaming responses and Responses SSE events, the default response rewrite is deliberately conservative: backend-returned calls such as `multi_agent_v1__close_agent` remain ordinary Responses `function_call` items, with only the synthetic namespace prefix stripped:

```json
{
  "type": "function_call",
  "name": "close_agent",
  "arguments": "{}"
}
```

The project brief says empirical testing may show that flat unprefixing is enough, and no captured Codex fixture in this bundle proves an exact wrapper schema. To avoid inventing an unverified client contract, `flat` is the default. An opt-in `--codex-namespace-response-mode experimental-wrapped` mode still emits the earlier experimental wrapper for fixture testing only:

```json
{
  "type": "namespace_call",
  "namespace": "multi_agent_v1",
  "name": "close_agent",
  "call": {
    "type": "function_call",
    "name": "close_agent",
    "arguments": "{}"
  }
}
```

Do not promote `experimental-wrapped` to the default unless a captured Codex Responses fixture proves this exact schema.

### `/v1/messages`

The proxy normalizes Anthropic `input_schema` definitions before forwarding. The default `--anthropic-schema-mode compat` is intentionally conservative because the exact `llama-server` Anthropic schema parser allowlist is empirical:

- Adds missing `type` fields instead of dropping untyped properties.
- Lowercases known JSON Schema type names.
- Forwards only the small parser-safe subset: `type`, `description`, object `properties`, filtered `required`, array `items`, and boolean/object `additionalProperties`.
- Moves parser-risky but semantically useful standard constraints such as `enum`, `const`, `default`, `pattern`, string lengths, numeric bounds, array lengths, uniqueness, and property-count bounds into compact `description` notes instead of putting those keys in the forwarded schema.
- Inlines nullable single-branch `anyOf`, `oneOf`, and `allOf` forms while adding a description note that null is accepted.
- Merges same-type multi-branch object/array schemas when possible.
- Converts unsupported multi-branch or parser-unfriendly schema details into compact `description` notes rather than silently discarding them.
- Still strips extension-style fields that are known to make `llama-server` reject schemas.

Use `--anthropic-schema-mode semantic` only after testing the target `llama-server` build with representative Claude Code schemas. Semantic mode preserves the larger set of standard JSON Schema constraints from earlier bundles.

For non-streaming Anthropic message-shaped responses and Anthropic SSE message/content-block events, the proxy adds safe Claude Code compatibility defaults if the backend omits them: top-level message `type`, `role`, `stop_reason`, `stop_sequence`, zero-valued `usage`, and per-content-block `cache_control: {"type":"ephemeral"}`. Existing fields are never overwritten. Anthropic error-shaped JSON bodies, including roots with `type: "error"` or an `error` object, pass through unchanged so backend and proxy errors keep their protocol shape.

### Gemini API

The proxy maps:

- `GET /v1beta/models` (and `/gemini/v1beta/models`) to `GET /v1/models`, translating the OpenAI model list into Gemini's model discovery format with `name`, `displayName`, `supportedGenerationMethods`, and available metadata. This enables gemini-cli's interactive model picker to show locally available models.
- `POST /v1beta/models/{model}:generateContent` to `POST /v1/chat/completions`.
- `POST /v1beta/models/{model}:streamGenerateContent` to streamed Chat Completions.
- Nonstandard paths carrying Gemini-shaped JSON bodies to `POST /v1/chat/completions` as well.

The `{model}` segment in Gemini URLs is treated as informational only. Translated Chat Completions requests always use `--backend-model` for the outgoing `model` field, so a client path such as `/v1beta/models/gemini-2.5-flash:generateContent` does not leak `gemini-2.5-flash` to `llama-server`. This matches a single-backend design where the backend server already owns model selection.

Content-based Gemini detection is intentionally narrow: the proxy requires a Gemini `contents[].parts[]` shape with Gemini indicators such as `systemInstruction.parts`, `tools[].functionDeclarations[]`, `generationConfig`, or a contents-only request. OpenAI-style bodies with `messages` or `input` remain pass-through.

It translates Gemini `contents`, `systemInstruction`, supported single-candidate `generationConfig` fields, and `tools[].functionDeclarations[]` into OpenAI Chat Completions fields, then maps OpenAI responses back to Gemini candidates. Because this proxy intentionally maps only the first OpenAI choice to `candidates[0]`, it does not forward Gemini `generationConfig.candidateCount` to OpenAI `n`; requesting multiple backend choices while dropping all but the first would be surprising and is outside the brief's required mapping. During request translation, every Gemini `functionCall` receives the deterministic OpenAI tool-call ID `call_<safe-name>_<ordinal>`, and later Gemini `functionResponse` parts resolve back to those generated IDs by function name and call order. If a request contains a Gemini `functionResponse` whose matching `functionCall` is absent from the transmitted history, the proxy now inserts a minimal synthetic assistant `tool_calls[]` message immediately before the translated tool result and reuses that generated ID; it never falls back to the bare function name as `tool_call_id`. Mixed text, `functionCall`, and `functionResponse` parts are emitted in the original part order instead of hoisting all tool results to the end of the content block. When the backend returns an OpenAI-style error for a Gemini request, the translated Gemini error body preserves the HTTP status code when it is an error status, and otherwise falls back to numeric OpenAI error codes or known OpenAI error types before defaulting to `500 INTERNAL`; the JSON body no longer claims `500 INTERNAL` for upstream `400`, `401`, `404`, `429`, or `504` responses.

By default, classifier-style `gemini-*-flash-lite:generateContent` requests without tools return a deterministic local classification response with route/category/type set to `general`. Use `--no-gemini-hardcoded-classifier` to forward those requests to the backend instead.



### Ollama API

The proxy responds to Ollama health probes locally without contacting the backend:

- `GET /` returns `200 OK` with `text/plain` body `Ollama is running`.
- `GET /api/version` returns `200 OK` with `{"version":"0.15.0"}`.

These are required by tools that verify Ollama is alive before sending requests (e.g., Codex's Ollama provider probes both endpoints during startup).

The proxy maps:

- `POST /api/chat` to `POST /v1/chat/completions` with Ollama messages, tools, options, and response format normalized into OpenAI Chat Completions fields.
- `POST /api/generate` to `POST /v1/chat/completions` by wrapping the prompt as a user message and the optional `system` field as a system message.
- `GET /api/tags` to `GET /v1/models`, then converts the OpenAI model list into Ollama `models[]` entries. When a backend model omits `created`, the fallback `modified_at` uses a stable sentinel (`1970-01-01T00:00:00Z`) rather than the current clock, so repeated calls over the same backend state return identical JSON.
- `POST /api/show` to backend `GET /v1/models`, then selects the requested model when present, falls back to the single loaded backend model otherwise, and reshapes backend metadata into Ollama `details`, `model_info`, `parameters`, `template`, `license`, `modified_at`, and `capabilities` fields.
- `POST /api/pull` and `DELETE /api/delete` to synthetic success responses because the local wrapper owns model lifecycle.

Ollama chat and generate endpoints stream by default. The proxy mirrors that default by setting outgoing Chat Completions `stream: true` unless the inbound request sends `stream: false`. Streamed backend SSE payloads are converted to Ollama newline-delimited JSON chunks with `application/x-ndjson`; non-streaming backend JSON is converted to the corresponding Ollama response object. The proxy does not forward a client's `Accept-Encoding` header to the backend; it sends `Accept-Encoding: identity` because response translators consume backend JSON and SSE bytes directly. If a translated response still arrives with a non-identity `Content-Encoding`, the proxy returns a protocol-shaped error instead of parsing compressed bytes as JSON or SSE. The Ollama stream transformer scans for SSE event boundaries as bytes and decodes UTF-8 only after a full event has arrived, so transport chunking cannot corrupt multibyte content. When the backend answers a streaming request with anything other than `text/event-stream`, the proxy buffers that finite response before choosing an output shape: JSON success bodies are translated into Ollama JSON and emitted as one NDJSON line, JSON error bodies become Ollama-shaped NDJSON errors, and non-JSON bodies such as `text/plain` or `text/html` become one explicit Ollama NDJSON error line instead of an empty stream.

Ollama compatibility coverage now includes observed client-library traffic under `fixtures/ollama/observed/` for `ollama-python` 0.6.2 and `ollama-js` 0.6.3 across `/api/tags`, `/api/show`, `/api/pull`, `/api/delete`, non-streaming chat/generate, streaming chat/generate, streaming chat with tools, and backend-error responses as actual client exceptions. Each observed request fixture records the client-emitted method, path, stable headers, JSON body, and the relevant companion artifact: expected Chat Completions rewrite, expected backend routing, deterministic JSON/NDJSON response, or client-observed error. `fixtures/ollama/integration/` covers backend response transforms that clients cannot directly emit, including OpenAI SSE tool-call deltas to Ollama NDJSON, successful non-SSE upstream bodies that must not disappear into the stream transformer, compressed upstream metadata that must trigger a protocol-shaped error on translated paths, and `/api/tags` model-list rows that omit backend creation timestamps. `scripts/validate_ollama_observed_fixtures.py` checks the full matrix without a Rust toolchain, `scripts/validate_proxy_compression_guard.py` checks the compression-request and response-guard invariants, `scripts/validate_ollama_tags_idempotency.py` checks the stable `/api/tags` fallback, and `scripts/capture_ollama_observed_fixtures.py` can refresh the observed fixtures from the official clients.

For tool use, Ollama assistant `tool_calls[]` become OpenAI `tool_calls[]`. If an Ollama tool result arrives without a `tool_call_id`, the translator reuses the most recent generated call ID for that tool name; if no matching assistant call appears in the current request history, it inserts a minimal assistant `tool_calls[]` message before the translated tool result. This keeps OpenAI Chat Completions history structurally valid without cross-request state.

Ollama-specific options are forwarded only when they have Chat Completions equivalents: `temperature`, `top_p`, `presence_penalty`, `frequency_penalty`, `seed`, `stop`, and `num_predict` as `max_tokens`. Model names pass through unchanged so the backend or wrapper can resolve the loaded model.


## Error handling and best-effort fallback

The proxy distinguishes hard proxy failures from translation failures:

- Hard proxy failures, such as a body exceeding `--max-body-bytes`, backend connection failure, and backend timeout, receive protocol-shaped JSON errors. Gemini error bodies use the same HTTP status code in `error.code` and map it to the closest Gemini canonical status string, such as `INVALID_ARGUMENT`, `UNAUTHENTICATED`, `NOT_FOUND`, `RESOURCE_EXHAUSTED`, or `DEADLINE_EXCEEDED`.
- Malformed JSON on recognized translation endpoints returns a protocol-shaped `400` without contacting the backend. This applies to `/v1/responses`, `/v1/messages`, Ollama JSON endpoints, and Gemini generation paths such as `/v1beta/models/{model}:generateContent` and `/v1beta/models/{model}:streamGenerateContent`.
- Well-formed request translation failures, such as unexpected request shapes that cannot be normalized safely, are logged to stderr with method, path, protocol, byte length, and the raw request body, then forwarded to `llama-server` at the original path after backend-incompatible field sanitization (see below).
- Responses from that fallback request are returned without response translation so `llama-server` errors or pass-through behavior propagate unchanged.

This matches the brief's split between malformed request bodies and translation failures: syntactically invalid JSON gets a protocol-shaped `400`, while surprising but well-formed client requests fall back to best-effort pass-through.

## Backend request sanitization

All requests forwarded to the backend are sanitized to remove fields that local inference servers do not support:

- `reasoning_effort` is removed from the top-level request body.
- `thinking` is removed from the top-level request body.
- `max_completion_tokens` is renamed to `max_tokens` when `max_tokens` is absent; if both are present, `max_completion_tokens` is dropped.
- `reasoning_content` and `reasoning` are removed from any message object in the `messages` array.

This sanitization applies to all protocols, including translation failure fallback and pass-through paths. Non-JSON request bodies pass through unchanged.

## Health check

```bash
curl -s http://127.0.0.1:8081/health | jq
```

A healthy response looks like:

```json
{"status":"ok","backend_ok":true,"backend":"http://127.0.0.1:8080","backend_probe_method":"GET","backend_probe_path":"/health"}
```

If the backend returns 404 on `/health`, the proxy falls back to `GET /` and treats a 200 response containing `"Ollama is running"` as healthy. The response includes `backend_probe_path` and `primary_backend_status` so operators can see which probe succeeded:

```json
{"status":"ok","backend_ok":true,"backend":"http://127.0.0.1:8080","backend_probe_method":"GET","backend_probe_path":"/","primary_backend_status":404}
```

If neither probe succeeds, the proxy returns `503` or `504` with `backend_ok: false`.

## Design notes

- The proxy keeps no cross-request state. It builds tool-name mappings per request and discards them after response forwarding.
- Unknown paths pass through without translation unless their JSON body is positively identified as Gemini-native `contents[].parts[]`.
- The proxy overwrites `Authorization` with `Bearer <backend-api-key>` when `--backend-api-key` is non-empty.
- Request bodies are bounded by `--max-body-bytes` before JSON parsing.
- `/api/tags` does not invent wall-clock `modified_at` values for backend model rows that lack `created`; it uses backend timestamps when available and otherwise emits the stable unknown timestamp `1970-01-01T00:00:00Z`.
- Streaming transformers buffer only incomplete SSE frames and, for Gemini/Ollama tool calls, partial tool-call argument deltas until the upstream finish event. Gemini streaming emits Gemini JSON `data:` frames and suppresses OpenAI's terminal `[DONE]` sentinel because it is not a Gemini JSON chunk. Ollama streaming emits newline-delimited JSON chunks, suppresses `[DONE]`, maps any finite non-SSE upstream response to either translated one-line NDJSON or an explicit Ollama NDJSON error, and handles split UTF-8 by buffering raw bytes until an SSE frame boundary arrives. Observed Ollama client fixtures cover official Python and JavaScript request/error shapes across the native endpoint matrix; integration fixtures cover backend SSE tool-call response translation and finite non-SSE fallback behavior. OpenAI Responses and Anthropic SSE rewriting preserves original `event:`, `id:`, `retry:`, comment, and blank-frame metadata and rewrites only `data:` payloads.

## Codex namespace response handling

Codex namespace tool calls use a conservative, evidence-gated response policy. Incoming namespace tools are flattened for `llama-server` with collision-resistant `namespace__tool` names. Responses are translated back to ordinary `function_call` objects by stripping the synthetic prefix, including in streaming SSE output-item payloads. Responses SSE frame metadata is preserved exactly at the field level; only the JSON `data:` payload is replaced when translation is needed.

The previous wrapper shape (`type: "namespace_call"`, `namespace`, `name`, nested `call`) remains available only via `--codex-namespace-response-mode experimental-wrapped`; it is intentionally documented as unproven because the bundle does not include captured Codex traffic that establishes this exact schema. Add real fixtures under `fixtures/codex/observed/` before treating a wrapper mode as rigorous.
