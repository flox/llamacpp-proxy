# Codex namespace fixtures

This directory exists to keep Codex namespace response handling evidence-gated.

- `synthetic/` contains small deterministic examples used to document transform intent.
- `observed/` is intentionally empty in this bundle. Put captured Codex Responses API request/response traffic here before promoting any namespace wrapper schema beyond `experimental-wrapped`.

The default runtime mode is `flat`: backend calls such as `multi_agent_v1__close_agent` are returned to Codex as ordinary `function_call` items named `close_agent`. That behavior follows the project brief's explicit compatibility possibility and avoids inventing a wrapper contract.

The `experimental-wrapped` mode is available only for controlled fixture testing. Treat it as unverified until an observed fixture proves Codex expects exactly that wire shape.
