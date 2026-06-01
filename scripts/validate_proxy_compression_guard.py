#!/usr/bin/env python3
"""Static validation for upstream compression handling in translated response paths."""
from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
MAIN = ROOT / "src" / "main.rs"


def expect(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def main() -> int:
    src = MAIN.read_text()
    expect("ACCEPT_ENCODING" in src, "main.rs must import/use ACCEPT_ENCODING")
    expect("CONTENT_ENCODING" in src, "main.rs must import/use CONTENT_ENCODING")
    expect(
        'builder = builder.header(ACCEPT_ENCODING, "identity")' in src,
        "backend requests must set Accept-Encoding: identity",
    )
    expect(
        "&& *name != ACCEPT_ENCODING" in src,
        "client Accept-Encoding must not be forwarded to the backend",
    )
    expect(
        "fn unsupported_content_encoding" in src and "CONTENT_ENCODING" in src,
        "translated response paths must inspect Content-Encoding",
    )
    expect(
        "if response_body_is_rewritten(protocol, &state)" in src,
        "Content-Encoding guard must run before response translation",
    )
    expect(
        "compressed_backend_response_error" in src and "ndjson_response_from_parts" in src,
        "Ollama streaming compressed backend responses must become NDJSON errors",
    )
    print("validated proxy compression guard")
    return 0


if __name__ == "__main__":
    sys.exit(main())
