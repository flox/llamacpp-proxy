# Validation

This bundle was prepared in an execution environment without a Rust toolchain installed.

Attempted checks:

```text
$ ./scripts/validate.sh
validated 18 observed Ollama request fixture set(s) plus integration fixtures
validated proxy compression guard
./scripts/validate.sh: line 7: cargo: command not found
```

Checks completed in this environment:

- Captured observed Ollama client fixtures from `ollama-python` 0.6.2 and `ollama-js` 0.6.3 against a local recorder.
- Ran `python3 scripts/validate_ollama_observed_fixtures.py`; it validated 18 observed Ollama request fixture sets, the backend-SSE tool-call integration fixture, successful non-SSE upstream response fixtures for `text/plain` and `text/html`, and compressed upstream response metadata.
- Ran `python3 scripts/validate_proxy_compression_guard.py`; it validated that backend requests set `Accept-Encoding: identity`, client `Accept-Encoding` does not forward, and translated paths reject non-identity `Content-Encoding` before body parsing.
- Ran `python3 scripts/validate_ollama_tags_idempotency.py`; it validated that `/api/tags` does not call the wall-clock fallback when backend model rows omit `created`, and that the integration fixture uses the stable unknown timestamp.
- Verified all fixture JSON files parse with `python3 -m json.tool` semantics, including single-line NDJSON fixtures, non-SSE fallback metadata, and compressed-response fallback metadata.
- Verified ZIP integrity after packaging.
- Regenerated `MANIFEST.txt` from the actual bundle contents.
- Created the patch from the previous non-SSE streaming archive to this bundle.

Run this on a machine with Rust installed before merging:

```bash
cargo fmt --all --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all
cargo build --release
```

## Additional compression correction

This revision updates backend request forwarding so the proxy does not pass through client `Accept-Encoding`; it sends `Accept-Encoding: identity` to the backend. Translated response paths also check `Content-Encoding` before parsing JSON or SSE. Any non-identity encoded translated response returns a protocol-shaped error, with Ollama streaming returning one `application/x-ndjson` error row. Added integration fixtures cover a `200 text/event-stream` response with `Content-Encoding: gzip`.

## Additional `/api/tags` idempotency correction

This revision updates `openai_models_to_ollama_tags` so backend model rows without `created` no longer receive `current_rfc3339()`. The converter now prefers backend timestamp fields (`created`, `created_at`, `modified_at`, including metadata variants) and otherwise emits `1970-01-01T00:00:00Z` as a stable unknown timestamp. Added Rust regression tests and `fixtures/ollama/integration/tags_missing_created.*` cover repeated-call stability.
