# Gemini fixtures

Synthetic fixtures document edge cases covered by unit tests:

- `orphan_function_response.*`: a Gemini `functionResponse` without a prior transmitted `functionCall` receives a synthetic OpenAI assistant `tool_calls[]` message with a matching `tool_call_id`.
- `mixed_parts.request.json`: mixed text/function-call/function-response parts preserve part order during translation.
- `candidate_count_ignored.*`: `generationConfig.candidateCount` is intentionally not forwarded to OpenAI `n` because the response translator maps only the first OpenAI choice into Gemini `candidates[0]`.

Add captured Gemini CLI request/response or streaming wire fixtures under `observed/` before claiming full compatibility for a new Gemini client or SDK version.
