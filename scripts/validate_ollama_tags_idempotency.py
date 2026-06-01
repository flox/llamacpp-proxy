#!/usr/bin/env python3
"""Validate that /api/tags fallback timestamps are stable without Rust."""
from __future__ import annotations

import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "lib.rs"
INTEGRATION = ROOT / "fixtures" / "ollama" / "integration"
UNKNOWN_MODIFIED_AT = "1970-01-01T00:00:00Z"


def expect(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def load_json(path: pathlib.Path):
    try:
        return json.loads(path.read_text())
    except Exception as exc:
        raise SystemExit(f"{path}: invalid JSON: {exc}") from exc


def function_body(source: str, name: str) -> str:
    marker = f"fn {name}"
    start = source.find(marker)
    expect(start >= 0, f"{SRC}: missing {name}")
    brace = source.find("{", start)
    expect(brace >= 0, f"{SRC}: missing body for {name}")
    depth = 0
    for idx in range(brace, len(source)):
        char = source[idx]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return source[brace + 1 : idx]
    raise SystemExit(f"{SRC}: unterminated body for {name}")


def main() -> int:
    source = SRC.read_text()
    expect(
        f'const OLLAMA_UNKNOWN_MODEL_MODIFIED_AT: &str = "{UNKNOWN_MODIFIED_AT}";' in source,
        f"{SRC}: missing stable unknown modified_at sentinel",
    )
    tags_body = function_body(source, "openai_models_to_ollama_tags")
    expect("current_rfc3339" not in tags_body, f"{SRC}: /api/tags must not use wall-clock fallback")
    expect("model_modified_at_from_record" in tags_body, f"{SRC}: /api/tags must use model timestamp helper")
    helper_body = function_body(source, "model_modified_at_from_record")
    expect(
        "OLLAMA_UNKNOWN_MODEL_MODIFIED_AT" in helper_body,
        f"{SRC}: timestamp helper must use stable sentinel when backend reports no timestamp",
    )
    expect(
        re.search(r"fn\s+ollama_tags_without_created_use_stable_modified_at\s*\(", source),
        f"{SRC}: missing regression test for timestamp idempotency",
    )

    backend = load_json(INTEGRATION / "tags_missing_created.backend.json")
    expected = load_json(INTEGRATION / "tags_missing_created.expected.json")
    expect(backend["data"][0].get("created") is None, "tags_missing_created backend fixture must omit created")
    expect(
        expected["models"][0]["modified_at"] == UNKNOWN_MODIFIED_AT,
        "tags_missing_created expected fixture must use stable unknown timestamp",
    )
    print("validated stable /api/tags fallback timestamp guard")
    return 0


if __name__ == "__main__":
    sys.exit(main())
