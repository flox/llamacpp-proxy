#!/usr/bin/env bash
set -euo pipefail

python3 scripts/validate_ollama_observed_fixtures.py
python3 scripts/validate_proxy_compression_guard.py
python3 scripts/validate_ollama_tags_idempotency.py

cargo fmt --all --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all
cargo build --release

if [ -f target/release/llamacpp-proxy ]; then
  ls -lh target/release/llamacpp-proxy
fi
