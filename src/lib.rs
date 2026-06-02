use serde_json::{json, Map, Value};
use std::collections::{HashMap, VecDeque};
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Protocol {
    OpenAiResponses,
    AnthropicMessages,
    OpenAiChat,
    Gemini,
    Ollama,
    PassThrough,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ResponseRewriteState {
    /// Backward-compatible lookup for prefixed backend tool names.
    pub tool_name_map: HashMap<String, String>,
    /// Full namespace metadata needed to rebuild Codex namespace tool-call items.
    pub namespace_tool_map: HashMap<String, NamespacedToolName>,
    /// Ollama response shape requested by the inbound endpoint, if applicable.
    pub ollama_response_kind: Option<OllamaResponseKind>,
    /// Original model requested by an Ollama endpoint when a later backend discovery
    /// response must be reshaped for that model.
    pub ollama_requested_model: Option<String>,
    /// Gemini response shape requested by inbound Gemini-compatible endpoints.
    pub gemini_response_kind: Option<GeminiResponseKind>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OllamaResponseKind {
    Chat,
    Generate,
    Tags,
    Show,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GeminiResponseKind {
    GenerateContent,
    ListModels,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NamespacedToolName {
    pub namespace: String,
    pub name: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[derive(Default)]
pub enum CodexNamespaceResponseMode {
    /// Conservative default: strip the synthetic namespace prefix and keep the
    /// normal OpenAI Responses `function_call` shape. This is the only mode
    /// supported by an explicit compatibility path in the project brief.
    #[default]
    Flat,
    /// Opt-in only: emits an invented namespace wrapper for experiments with
    /// captured Codex traffic. Do not make this the default unless an actual
    /// Codex fixture proves the client expects this exact schema.
    ExperimentalWrapped,
}


#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[derive(Default)]
pub enum AnthropicSchemaMode {
    /// Conservative default for llama-server's empirical Anthropic JSON Schema parser.
    /// It forwards only the small schema subset known to be necessary for tool
    /// shape (`type`, `description`, `properties`, `required`, `items`, and
    /// `additionalProperties`) and carries richer constraints as description notes.
    #[default]
    LlamaServerCompat,
    /// Opt-in mode for deployments that have verified their llama-server build
    /// accepts a larger set of standard JSON Schema keywords.
    Semantic,
}


impl std::str::FromStr for AnthropicSchemaMode {
    type Err = RewriteError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "compat" | "llama-server-compat" | "llama" => Ok(Self::LlamaServerCompat),
            "semantic" | "preserve-standard" => Ok(Self::Semantic),
            other => Err(RewriteError::new(format!(
                "invalid Anthropic schema mode {other:?}; expected compat or semantic"
            ))),
        }
    }
}

impl std::str::FromStr for CodexNamespaceResponseMode {
    type Err = RewriteError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "flat" => Ok(Self::Flat),
            "experimental-wrapped" => Ok(Self::ExperimentalWrapped),
            // Backward-compatible alias for earlier development bundles. It is
            // intentionally not documented because the wrapper is unproven.
            "wrapped" => Ok(Self::ExperimentalWrapped),
            other => Err(RewriteError::new(format!(
                "invalid Codex namespace response mode {other:?}; expected flat or experimental-wrapped"
            ))),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RewriteError {
    pub message: String,
}

impl RewriteError {
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl fmt::Display for RewriteError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl std::error::Error for RewriteError {}

pub fn protocol_for_path(path: &str) -> Protocol {
    if is_gemini_path(path) {
        return Protocol::Gemini;
    }
    if is_ollama_path(path) {
        return Protocol::Ollama;
    }
    match path {
        "/v1/responses" => Protocol::OpenAiResponses,
        "/v1/messages" => Protocol::AnthropicMessages,
        "/v1/chat/completions" => Protocol::OpenAiChat,
        _ => Protocol::PassThrough,
    }
}

/// Detect the protocol using the request path first, then inspect the JSON body
/// for Gemini-native request shapes on nonstandard paths. The body fallback is
/// deliberately narrow so ordinary JSON pass-through requests are not captured.
pub fn protocol_for_request(path: &str, body: &[u8]) -> Protocol {
    let path_protocol = protocol_for_path(path);
    if path_protocol != Protocol::PassThrough {
        return path_protocol;
    }

    serde_json::from_slice::<Value>(body)
        .ok()
        .filter(is_gemini_request_body)
        .map(|_| Protocol::Gemini)
        .unwrap_or(Protocol::PassThrough)
}

pub fn is_gemini_path(path: &str) -> bool {
    is_gemini_model_list_path(path) || path.starts_with("/v1beta/models/") || path.starts_with("/gemini/")
}

pub fn is_ollama_path(path: &str) -> bool {
    matches!(
        path,
        "/api/chat" | "/api/generate" | "/api/tags" | "/api/show" | "/api/pull" | "/api/delete"
    )
}

pub fn is_ollama_chat_path(path: &str) -> bool {
    path == "/api/chat"
}

pub fn is_ollama_generate_path(path: &str) -> bool {
    path == "/api/generate"
}

pub fn is_ollama_tags_path(path: &str) -> bool {
    path == "/api/tags"
}

pub fn is_ollama_show_path(path: &str) -> bool {
    path == "/api/show"
}

pub fn is_ollama_pull_path(path: &str) -> bool {
    path == "/api/pull"
}

pub fn is_ollama_delete_path(path: &str) -> bool {
    path == "/api/delete"
}

pub fn is_gemini_model_list_path(path: &str) -> bool {
    path == "/v1beta/models" || path == "/gemini/v1beta/models"
}

pub fn is_gemini_stream_path(path: &str) -> bool {
    path.contains(":streamGenerateContent")
}

pub fn is_gemini_generation_path(path: &str) -> bool {
    path.contains(":generateContent") || path.contains(":streamGenerateContent")
}

pub fn is_gemini_request_body(value: &Value) -> bool {
    let Some(root) = value.as_object() else {
        return false;
    };

    // Avoid hijacking already-normalized OpenAI-compatible requests.
    if root.contains_key("messages") || root.contains_key("input") {
        return false;
    }

    let has_gemini_contents = root
        .get("contents")
        .and_then(Value::as_array)
        .map(|contents| {
            contents.iter().any(|content| {
                content
                    .get("parts")
                    .and_then(Value::as_array)
                    .map(|parts| {
                        parts.iter().any(|part| {
                            part.get("text").is_some()
                                || part.get("functionCall").is_some()
                                || part.get("functionResponse").is_some()
                        })
                    })
                    .unwrap_or(false)
            })
        })
        .unwrap_or(false);

    if !has_gemini_contents {
        return false;
    }

    let has_gemini_system_instruction = root
        .get("systemInstruction")
        .and_then(|value| value.get("parts"))
        .and_then(Value::as_array)
        .is_some();
    let has_gemini_tools = root
        .get("tools")
        .and_then(Value::as_array)
        .map(|tools| {
            tools.iter().any(|tool| {
                tool.get("functionDeclarations")
                    .and_then(Value::as_array)
                    .is_some()
            })
        })
        .unwrap_or(false);
    let has_gemini_generation_config = root
        .get("generationConfig")
        .and_then(Value::as_object)
        .map(|config| {
            ["temperature", "topP", "maxOutputTokens", "candidateCount"]
                .iter()
                .any(|key| config.contains_key(*key))
        })
        .unwrap_or(false);

    has_gemini_system_instruction
        || has_gemini_tools
        || has_gemini_generation_config
        || root.keys().all(|key| key.as_str() == "contents")
}

pub fn gemini_model_from_path(path: &str) -> Option<String> {
    let start = path.find("/models/").map(|idx| idx + "/models/".len())?;
    let rest = &path[start..];
    let end = rest.find(':').unwrap_or(rest.len());
    let model = &rest[..end];
    if model.is_empty() {
        None
    } else {
        Some(model.to_owned())
    }
}

pub fn parse_json_body(body: &[u8]) -> Result<Value, RewriteError> {
    serde_json::from_slice(body).map_err(|err| RewriteError::new(format!("malformed JSON: {err}")))
}

pub fn body_declares_stream(value: &Value) -> bool {
    value.get("stream").and_then(Value::as_bool).unwrap_or(false)
}

pub fn ollama_request_declares_stream(value: &Value) -> bool {
    value.get("stream").and_then(Value::as_bool).unwrap_or(true)
}

pub fn rewrite_ollama_chat_request(root: Value) -> Result<Value, RewriteError> {
    let obj = root
        .as_object()
        .ok_or_else(|| RewriteError::new("well-formed JSON request root is not an object"))?;
    let model = required_string(obj, "model")?;
    let messages = ollama_messages_to_openai_messages(obj.get("messages"))?;

    let mut out = Map::new();
    out.insert("model".to_owned(), Value::String(model));
    out.insert("messages".to_owned(), Value::Array(messages));
    out.insert(
        "stream".to_owned(),
        Value::Bool(obj.get("stream").and_then(Value::as_bool).unwrap_or(true)),
    );
    copy_ollama_tools(obj.get("tools"), &mut out);
    copy_ollama_format(obj.get("format"), &mut out);
    copy_ollama_options(obj, &mut out);

    Ok(Value::Object(out))
}

pub fn rewrite_ollama_generate_request(root: Value) -> Result<Value, RewriteError> {
    let obj = root
        .as_object()
        .ok_or_else(|| RewriteError::new("well-formed JSON request root is not an object"))?;
    let model = required_string(obj, "model")?;
    let prompt = obj
        .get("prompt")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_owned();

    let mut messages = Vec::new();
    if let Some(system) = obj.get("system").and_then(Value::as_str).filter(|value| !value.is_empty()) {
        messages.push(json!({"role":"system","content":system}));
    }
    messages.push(openai_user_message_from_ollama_content(
        Value::String(prompt),
        obj.get("images"),
    ));

    let mut out = Map::new();
    out.insert("model".to_owned(), Value::String(model));
    out.insert("messages".to_owned(), Value::Array(messages));
    out.insert(
        "stream".to_owned(),
        Value::Bool(obj.get("stream").and_then(Value::as_bool).unwrap_or(true)),
    );
    copy_ollama_format(obj.get("format"), &mut out);
    copy_ollama_options(obj, &mut out);

    Ok(Value::Object(out))
}

pub fn is_ollama_chat_lifecycle_request(root: &Value) -> bool {
    root.get("messages")
        .and_then(Value::as_array)
        .map(Vec::is_empty)
        .unwrap_or(false)
}

pub fn is_ollama_generate_lifecycle_request(root: &Value) -> bool {
    let prompt_is_empty = root
        .get("prompt")
        .and_then(Value::as_str)
        .map(str::is_empty)
        .unwrap_or(true);
    let has_images = root
        .get("images")
        .and_then(Value::as_array)
        .map(|images| !images.is_empty())
        .unwrap_or(false);
    prompt_is_empty && !has_images
}

pub fn ollama_lifecycle_response(root: &Value, kind: OllamaResponseKind, backend_model: &str) -> Value {
    let model = ollama_requested_model(root, backend_model);
    let done_reason = if ollama_keep_alive_is_zero(root.get("keep_alive")) {
        "unload"
    } else {
        "load"
    };
    match kind {
        OllamaResponseKind::Chat => json!({
            "model": model,
            "created_at": current_rfc3339(),
            "message": {"role":"assistant","content":""},
            "done_reason": done_reason,
            "done": true
        }),
        OllamaResponseKind::Generate => json!({
            "model": model,
            "created_at": current_rfc3339(),
            "response": "",
            "done_reason": done_reason,
            "done": true
        }),
        OllamaResponseKind::Tags => json!({"models": []}),
        OllamaResponseKind::Show => ollama_show_response(root, backend_model),
    }
}

pub fn ollama_show_response(root: &Value, backend_model: &str) -> Value {
    let model = ollama_requested_model(root, backend_model);
    ollama_show_response_from_model_record(None, Some(&model), backend_model)
}

fn openai_models_to_ollama_show(
    root: Value,
    requested_model: Option<&str>,
    backend_model: &str,
    _http_status: u16,
) -> Value {
    let selected = select_openai_model_record(&root, requested_model);
    ollama_show_response_from_model_record(selected, requested_model, backend_model)
}

fn select_openai_model_record<'a>(root: &'a Value, requested_model: Option<&str>) -> Option<&'a Value> {
    let models = root.get("data").and_then(Value::as_array)?;
    let requested = requested_model.filter(|model| !model.is_empty());

    if let Some(requested) = requested {
        if let Some(exact) = models.iter().find(|item| item.get("id").and_then(Value::as_str) == Some(requested)) {
            return Some(exact);
        }
        if let Some(normalized) = models.iter().find(|item| {
            item.get("id")
                .and_then(Value::as_str)
                .map(|id| ollama_model_names_match(id, requested))
                .unwrap_or(false)
        }) {
            return Some(normalized);
        }
    }

    models.first()
}

fn ollama_model_names_match(candidate: &str, requested: &str) -> bool {
    fn without_latest(value: &str) -> &str {
        value.strip_suffix(":latest").unwrap_or(value)
    }
    without_latest(candidate) == without_latest(requested)
}

fn ollama_show_response_from_model_record(
    model_record: Option<&Value>,
    requested_model: Option<&str>,
    backend_model: &str,
) -> Value {
    let backend_id = model_record
        .and_then(|record| record.get("id"))
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
        .unwrap_or(backend_model);
    let display_model = requested_model
        .filter(|value| !value.is_empty())
        .unwrap_or(backend_id);

    let metadata = model_record.and_then(model_metadata_object);
    let model_info = build_ollama_model_info(model_record, metadata);
    let family = metadata_string(metadata, &["family", "general.architecture", "architecture"])
        .or_else(|| infer_family_from_model_name(backend_id))
        .unwrap_or_else(|| "unknown".to_owned());
    let families = metadata_array_of_strings(metadata, &["families"])
        .filter(|families| !families.is_empty())
        .unwrap_or_else(|| vec![family.clone()]);
    let parameter_size = metadata_string(metadata, &["parameter_size", "parameterSize", "parameters_size"])
        .or_else(|| metadata_u64(metadata, &["parameter_count", "general.parameter_count"]).map(format_parameter_count))
        .or_else(|| infer_parameter_size_from_model_name(backend_id))
        .unwrap_or_else(|| "unknown".to_owned());
    let quantization_level = metadata_string(metadata, &["quantization_level", "quantization", "quantizationLevel"])
        .or_else(|| infer_quantization_from_model_name(backend_id))
        .unwrap_or_else(|| "unknown".to_owned());
    let format = metadata_string(metadata, &["format", "general.file_type_format"])
        .unwrap_or_else(|| "gguf".to_owned());

    let mut out = Map::new();
    out.insert("license".to_owned(), Value::String(metadata_string(metadata, &["license"]).unwrap_or_default()));
    out.insert("modelfile".to_owned(), Value::String(metadata_string(metadata, &["modelfile"]).unwrap_or_else(|| format!("FROM {display_model}\n"))));
    out.insert("parameters".to_owned(), Value::String(metadata_string(metadata, &["parameters"]).unwrap_or_default()));
    out.insert("template".to_owned(), Value::String(metadata_string(metadata, &["template", "chat_template", "tokenizer.chat_template"]).unwrap_or_default()));
    if let Some(system) = metadata_string(metadata, &["system", "system_prompt"]) {
        out.insert("system".to_owned(), Value::String(system));
    }
    if let Some(created_at) = model_record
        .and_then(|record| record.get("created"))
        .and_then(Value::as_i64)
        .map(unix_seconds_to_rfc3339)
    {
        out.insert("modified_at".to_owned(), Value::String(created_at));
    }
    out.insert("details".to_owned(), json!({
        "parent_model": metadata_string(metadata, &["parent_model", "parentModel"]).unwrap_or_default(),
        "format": format,
        "family": family,
        "families": families,
        "parameter_size": parameter_size,
        "quantization_level": quantization_level
    }));
    out.insert("model_info".to_owned(), Value::Object(model_info));
    out.insert("capabilities".to_owned(), metadata_array_of_strings(metadata, &["capabilities"]).map(|values| {
        Value::Array(values.into_iter().map(Value::String).collect())
    }).unwrap_or_else(|| json!(["completion", "tools"])));
    Value::Object(out)
}

fn model_metadata_object(record: &Value) -> Option<&Map<String, Value>> {
    ["meta", "metadata", "model_info"]
        .iter()
        .find_map(|key| record.get(*key).and_then(Value::as_object))
}

fn build_ollama_model_info(
    record: Option<&Value>,
    metadata: Option<&Map<String, Value>>,
) -> Map<String, Value> {
    let mut out = Map::new();
    if let Some(metadata) = metadata {
        for (key, value) in metadata {
            if is_ollama_model_info_key(key) {
                out.insert(key.clone(), value.clone());
            }
        }
    }
    if let Some(record_info) = record
        .and_then(|record| record.get("model_info"))
        .and_then(Value::as_object)
    {
        for (key, value) in record_info {
            out.insert(key.clone(), value.clone());
        }
    }
    out
}

fn is_ollama_model_info_key(key: &str) -> bool {
    key.contains('.') || matches!(key, "parameter_count" | "context_length" | "embedding_length")
}

fn metadata_string(metadata: Option<&Map<String, Value>>, keys: &[&str]) -> Option<String> {
    let metadata = metadata?;
    keys.iter().find_map(|key| {
        metadata.get(*key).and_then(|value| match value {
            Value::String(text) if !text.is_empty() => Some(text.clone()),
            Value::Number(number) => Some(number.to_string()),
            _ => None,
        })
    })
}

fn metadata_u64(metadata: Option<&Map<String, Value>>, keys: &[&str]) -> Option<u64> {
    let metadata = metadata?;
    keys.iter().find_map(|key| metadata.get(*key).and_then(Value::as_u64))
}

fn metadata_array_of_strings(metadata: Option<&Map<String, Value>>, keys: &[&str]) -> Option<Vec<String>> {
    let metadata = metadata?;
    keys.iter().find_map(|key| {
        metadata.get(*key).and_then(|value| {
            value.as_array().map(|items| {
                items
                    .iter()
                    .filter_map(Value::as_str)
                    .filter(|item| !item.is_empty())
                    .map(str::to_owned)
                    .collect::<Vec<String>>()
            })
        })
    })
}

fn format_parameter_count(count: u64) -> String {
    if count >= 1_000_000_000 {
        format!("{:.1}B", count as f64 / 1_000_000_000.0)
    } else if count >= 1_000_000 {
        format!("{:.1}M", count as f64 / 1_000_000.0)
    } else {
        count.to_string()
    }
}

fn infer_family_from_model_name(model: &str) -> Option<String> {
    let base = model
        .rsplit('/')
        .next()
        .unwrap_or(model)
        .split(':')
        .next()
        .unwrap_or(model)
        .split('-')
        .next()
        .unwrap_or(model)
        .trim();
    if base.is_empty() {
        None
    } else {
        Some(base.to_ascii_lowercase())
    }
}

fn infer_parameter_size_from_model_name(model: &str) -> Option<String> {
    find_model_token(model, |token| {
        let lower = token.to_ascii_lowercase();
        if lower.ends_with('b') || lower.ends_with('m') {
            let number = &lower[..lower.len().saturating_sub(1)];
            if !number.is_empty() && number.chars().all(|ch| ch.is_ascii_digit() || ch == '.') {
                return Some(lower.to_ascii_uppercase());
            }
        }
        None
    })
}

fn infer_quantization_from_model_name(model: &str) -> Option<String> {
    find_model_token(model, |token| {
        let upper = token.to_ascii_uppercase();
        if upper.starts_with('Q') && upper.chars().any(|ch| ch.is_ascii_digit()) {
            Some(upper)
        } else {
            None
        }
    })
}

fn find_model_token<F>(model: &str, predicate: F) -> Option<String>
where
    F: FnMut(&str) -> Option<String>,
{
    model
        .split([':', '/', '-', '.'])
        .find_map(predicate)
}

pub fn ollama_pull_response(root: &Value) -> Value {
    let model = ollama_model_name(root).unwrap_or("model");
    json!({"status":"success","model":model})
}

pub fn ollama_delete_response(root: &Value) -> Value {
    let model = ollama_model_name(root).unwrap_or("model");
    json!({"status":"success","model":model})
}

pub fn openai_response_to_ollama(root: Value, kind: OllamaResponseKind, http_status: u16) -> Value {
    openai_response_to_ollama_with_context(root, kind, http_status, None, "local-model")
}

pub fn openai_response_to_ollama_with_context(
    root: Value,
    kind: OllamaResponseKind,
    http_status: u16,
    requested_model: Option<&str>,
    backend_model: &str,
) -> Value {
    if root.get("error").is_some() {
        return openai_error_to_ollama(root);
    }
    match kind {
        OllamaResponseKind::Chat => openai_chat_response_to_ollama_chat(root),
        OllamaResponseKind::Generate => openai_chat_response_to_ollama_generate(root),
        OllamaResponseKind::Tags => openai_models_to_ollama_tags(root, http_status),
        OllamaResponseKind::Show => openai_models_to_ollama_show(root, requested_model, backend_model, http_status),
    }
}

fn required_string(obj: &Map<String, Value>, key: &str) -> Result<String, RewriteError> {
    obj.get(key)
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
        .map(str::to_owned)
        .ok_or_else(|| RewriteError::new(format!("well-formed JSON does not contain required string field {key:?}")))
}

fn ollama_messages_to_openai_messages(value: Option<&Value>) -> Result<Vec<Value>, RewriteError> {
    let Some(messages) = value.and_then(Value::as_array) else {
        return Ok(Vec::new());
    };

    let mut out = Vec::with_capacity(messages.len());
    let mut tool_ids_by_name: HashMap<String, VecDeque<String>> = HashMap::new();
    let mut next_tool_ordinal = 0usize;

    for message in messages {
        let Some(obj) = message.as_object() else {
            return Err(RewriteError::new("Ollama messages must be objects"));
        };
        let role = obj.get("role").and_then(Value::as_str).unwrap_or("user");
        match role {
            "assistant" => {
                let mut rewritten = Map::new();
                rewritten.insert("role".to_owned(), Value::String("assistant".to_owned()));
                rewritten.insert(
                    "content".to_owned(),
                    obj.get("content")
                        .cloned()
                        .unwrap_or_else(|| Value::String(String::new())),
                );
                if let Some(tool_calls) = normalize_ollama_message_tool_calls(
                    obj.get("tool_calls"),
                    &mut tool_ids_by_name,
                    &mut next_tool_ordinal,
                ) {
                    rewritten.insert("tool_calls".to_owned(), tool_calls);
                }
                out.push(Value::Object(rewritten));
            }
            "tool" => {
                let tool_name = obj
                    .get("tool_name")
                    .or_else(|| obj.get("name"))
                    .and_then(Value::as_str)
                    .filter(|value| !value.is_empty())
                    .unwrap_or("tool")
                    .to_owned();
                let tool_call_id = obj
                    .get("tool_call_id")
                    .and_then(Value::as_str)
                    .filter(|value| !value.is_empty())
                    .map(str::to_owned)
                    .or_else(|| {
                        tool_ids_by_name
                            .get_mut(&tool_name)
                            .and_then(VecDeque::pop_front)
                    })
                    .unwrap_or_else(|| {
                        let id = generated_tool_call_id(&tool_name, next_tool_ordinal);
                        next_tool_ordinal += 1;
                        out.push(json!({
                            "role":"assistant",
                            "content":"",
                            "tool_calls":[{"id":id,"type":"function","function":{"name":tool_name,"arguments":"{}"}}]
                        }));
                        id
                    });
                out.push(json!({
                    "role":"tool",
                    "tool_call_id": tool_call_id,
                    "content": obj.get("content").cloned().unwrap_or_else(|| Value::String(String::new()))
                }));
            }
            "system" => out.push(json!({
                "role":"system",
                "content": obj.get("content").cloned().unwrap_or_else(|| Value::String(String::new()))
            })),
            _ => out.push(openai_user_message_from_ollama_content(
                obj.get("content")
                    .cloned()
                    .unwrap_or_else(|| Value::String(String::new())),
                obj.get("images"),
            )),
        }
    }

    Ok(out)
}

fn normalize_ollama_message_tool_calls(
    value: Option<&Value>,
    tool_ids_by_name: &mut HashMap<String, VecDeque<String>>,
    next_tool_ordinal: &mut usize,
) -> Option<Value> {
    let calls = value.and_then(Value::as_array)?;
    let mut out = Vec::with_capacity(calls.len());
    for call in calls {
        let function = call.get("function").and_then(Value::as_object);
        let name = function
            .and_then(|function| function.get("name"))
            .or_else(|| call.get("name"))
            .and_then(Value::as_str)
            .filter(|value| !value.is_empty())
            .unwrap_or("tool")
            .to_owned();
        let id = call
            .get("id")
            .and_then(Value::as_str)
            .filter(|value| !value.is_empty())
            .map(str::to_owned)
            .unwrap_or_else(|| {
                let id = generated_tool_call_id(&name, *next_tool_ordinal);
                *next_tool_ordinal += 1;
                id
            });
        tool_ids_by_name
            .entry(name.clone())
            .or_default()
            .push_back(id.clone());
        let arguments = function
            .and_then(|function| function.get("arguments"))
            .cloned()
            .or_else(|| call.get("arguments").cloned())
            .unwrap_or_else(|| json!({}));
        out.push(json!({
            "id": id,
            "type": "function",
            "function": {"name": name, "arguments": json_arguments_string(arguments)}
        }));
    }
    Some(Value::Array(out))
}

fn openai_user_message_from_ollama_content(content: Value, images: Option<&Value>) -> Value {
    let Some(images) = images.and_then(Value::as_array).filter(|images| !images.is_empty()) else {
        return json!({"role":"user","content":content});
    };

    let mut parts = Vec::new();
    match content {
        Value::String(text) if !text.is_empty() => parts.push(json!({"type":"text","text":text})),
        Value::String(_) => {}
        other => parts.push(json!({"type":"text","text":other.to_string()})),
    }
    for image in images.iter().filter_map(Value::as_str) {
        parts.push(json!({"type":"image_url","image_url":{"url":format!("data:image/jpeg;base64,{image}")}}));
    }
    json!({"role":"user","content":parts})
}

fn copy_ollama_tools(value: Option<&Value>, out: &mut Map<String, Value>) {
    let Some(tools) = value.and_then(Value::as_array) else {
        return;
    };
    let normalized = tools
        .iter()
        .filter_map(normalize_ollama_request_tool)
        .collect::<Vec<Value>>();
    if !normalized.is_empty() {
        out.insert("tools".to_owned(), Value::Array(normalized));
    }
}

fn normalize_ollama_request_tool(tool: &Value) -> Option<Value> {
    let obj = tool.as_object()?;
    if obj.get("type").and_then(Value::as_str) == Some("function") && obj.get("function").is_some() {
        return Some(tool.clone());
    }
    let function = obj.get("function").and_then(Value::as_object).unwrap_or(obj);
    let name = function.get("name").and_then(Value::as_str).unwrap_or("tool");
    let description = function
        .get("description")
        .cloned()
        .unwrap_or_else(|| Value::String(String::new()));
    let parameters = function
        .get("parameters")
        .cloned()
        .unwrap_or_else(|| json!({"type":"object","properties":{}}));
    Some(json!({"type":"function","function":{"name":name,"description":description,"parameters":parameters}}))
}

fn copy_ollama_options(src: &Map<String, Value>, out: &mut Map<String, Value>) {
    if let Some(options) = src.get("options").and_then(Value::as_object) {
        copy_ollama_option_number(options, out, "temperature", "temperature");
        copy_ollama_option_number(options, out, "top_p", "top_p");
        copy_ollama_option_number(options, out, "presence_penalty", "presence_penalty");
        copy_ollama_option_number(options, out, "frequency_penalty", "frequency_penalty");
        copy_ollama_option_number(options, out, "seed", "seed");
        copy_ollama_option_number(options, out, "num_predict", "max_tokens");
        if let Some(stop) = options.get("stop") {
            out.insert("stop".to_owned(), stop.clone());
        }
    }
    copy_ollama_option_number(src, out, "temperature", "temperature");
    copy_ollama_option_number(src, out, "top_p", "top_p");
    copy_ollama_option_number(src, out, "presence_penalty", "presence_penalty");
    copy_ollama_option_number(src, out, "frequency_penalty", "frequency_penalty");
    copy_ollama_option_number(src, out, "seed", "seed");
}

fn copy_ollama_option_number(src: &Map<String, Value>, out: &mut Map<String, Value>, from: &str, to: &str) {
    if let Some(value) = src.get(from).filter(|value| value.is_number()) {
        out.entry(to.to_owned()).or_insert_with(|| value.clone());
    }
}

fn copy_ollama_format(value: Option<&Value>, out: &mut Map<String, Value>) {
    match value {
        Some(Value::String(format)) if format == "json" => {
            out.insert("response_format".to_owned(), json!({"type":"json_object"}));
        }
        Some(Value::Object(_)) => {
            if let Some(schema) = value {
                out.insert(
                    "response_format".to_owned(),
                    json!({"type":"json_schema","json_schema":{"name":"ollama_response","schema":schema}}),
                );
            }
        }
        _ => {}
    }
}

fn openai_chat_response_to_ollama_chat(root: Value) -> Value {
    let choice = first_openai_choice(&root);
    let message = choice.get("message").cloned().unwrap_or_else(|| json!({}));
    let finish_reason = choice.get("finish_reason").and_then(Value::as_str);
    let mut ollama_message = Map::new();
    ollama_message.insert("role".to_owned(), Value::String("assistant".to_owned()));
    ollama_message.insert(
        "content".to_owned(),
        Value::String(message.get("content").and_then(Value::as_str).unwrap_or_default().to_owned()),
    );
    if let Some(tool_calls) = openai_tool_calls_to_ollama(message.get("tool_calls")) {
        ollama_message.insert("tool_calls".to_owned(), tool_calls);
    }

    let mut out = ollama_response_base(&root);
    out.insert("message".to_owned(), Value::Object(ollama_message));
    out.insert("done".to_owned(), Value::Bool(true));
    if let Some(reason) = finish_reason.and_then(ollama_done_reason_from_openai_finish) {
        out.insert("done_reason".to_owned(), Value::String(reason.to_owned()));
    }
    add_ollama_usage(&mut out, root.get("usage"));
    Value::Object(out)
}

fn openai_chat_response_to_ollama_generate(root: Value) -> Value {
    let choice = first_openai_choice(&root);
    let message = choice.get("message").cloned().unwrap_or_else(|| json!({}));
    let finish_reason = choice.get("finish_reason").and_then(Value::as_str);
    let mut out = ollama_response_base(&root);
    out.insert(
        "response".to_owned(),
        Value::String(message.get("content").and_then(Value::as_str).unwrap_or_default().to_owned()),
    );
    out.insert("done".to_owned(), Value::Bool(true));
    if let Some(reason) = finish_reason.and_then(ollama_done_reason_from_openai_finish) {
        out.insert("done_reason".to_owned(), Value::String(reason.to_owned()));
    }
    add_ollama_usage(&mut out, root.get("usage"));
    Value::Object(out)
}

const OLLAMA_UNKNOWN_MODEL_MODIFIED_AT: &str = "1970-01-01T00:00:00Z";

fn openai_models_to_ollama_tags(root: Value, _http_status: u16) -> Value {
    let models = root
        .get("data")
        .and_then(Value::as_array)
        .map(|items| {
            items
                .iter()
                .filter_map(|item| {
                    let id = item.get("id").and_then(Value::as_str)?;
                    let modified_at = model_modified_at_from_record(item);
                    Some(json!({
                        "name": id,
                        "model": id,
                        "modified_at": modified_at,
                        "size": 0,
                        "digest": "",
                        "details": {
                            "parent_model": "",
                            "format": "gguf",
                            "family": "unknown",
                            "families": ["unknown"],
                            "parameter_size": "unknown",
                            "quantization_level": "unknown"
                        }
                    }))
                })
                .collect::<Vec<Value>>()
        })
        .unwrap_or_default();
    json!({"models": models})
}

fn model_modified_at_from_record(record: &Value) -> String {
    model_timestamp_from_keys(record, &["created", "created_at", "createdAt", "modified_at", "modifiedAt"])
        .or_else(|| {
            let metadata = model_metadata_object(record)?;
            metadata_timestamp_from_keys(metadata, &["created", "created_at", "createdAt", "modified_at", "modifiedAt"])
        })
        .unwrap_or_else(|| OLLAMA_UNKNOWN_MODEL_MODIFIED_AT.to_owned())
}

fn model_timestamp_from_keys(record: &Value, keys: &[&str]) -> Option<String> {
    keys.iter().find_map(|key| timestamp_value_to_rfc3339(record.get(*key)?))
}

fn metadata_timestamp_from_keys(metadata: &Map<String, Value>, keys: &[&str]) -> Option<String> {
    keys.iter().find_map(|key| timestamp_value_to_rfc3339(metadata.get(*key)?))
}

fn timestamp_value_to_rfc3339(value: &Value) -> Option<String> {
    match value {
        Value::Number(number) => number
            .as_i64()
            .or_else(|| number.as_u64().and_then(|value| i64::try_from(value).ok()))
            .map(unix_seconds_to_rfc3339),
        Value::String(text) if !text.trim().is_empty() => Some(text.trim().to_owned()),
        _ => None,
    }
}

fn first_openai_choice(root: &Value) -> Value {
    root.get("choices")
        .and_then(Value::as_array)
        .and_then(|choices| choices.first())
        .cloned()
        .unwrap_or_else(|| json!({}))
}

fn ollama_response_base(root: &Value) -> Map<String, Value> {
    let mut out = Map::new();
    out.insert(
        "model".to_owned(),
        Value::String(root.get("model").and_then(Value::as_str).unwrap_or("local-model").to_owned()),
    );
    out.insert(
        "created_at".to_owned(),
        Value::String(
            root.get("created")
                .and_then(Value::as_i64)
                .map(unix_seconds_to_rfc3339)
                .unwrap_or_else(current_rfc3339),
        ),
    );
    out
}

fn openai_tool_calls_to_ollama(value: Option<&Value>) -> Option<Value> {
    let calls = value.and_then(Value::as_array)?;
    if calls.is_empty() {
        return None;
    }
    Some(Value::Array(
        calls
            .iter()
            .filter_map(|call| {
                let function = call.get("function")?;
                let name = function.get("name").and_then(Value::as_str).unwrap_or("tool");
                let arguments = parse_tool_arguments(function.get("arguments"));
                Some(json!({"function":{"name":name,"arguments":arguments}}))
            })
            .collect(),
    ))
}

fn add_ollama_usage(out: &mut Map<String, Value>, value: Option<&Value>) {
    let Some(usage) = value.and_then(Value::as_object) else {
        return;
    };
    let prompt = usage.get("prompt_tokens").and_then(Value::as_u64).unwrap_or(0);
    let completion = usage
        .get("completion_tokens")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    out.insert("prompt_eval_count".to_owned(), json!(prompt));
    out.insert("eval_count".to_owned(), json!(completion));
    out.insert("total_duration".to_owned(), json!(0));
    out.insert("load_duration".to_owned(), json!(0));
    out.insert("prompt_eval_duration".to_owned(), json!(0));
    out.insert("eval_duration".to_owned(), json!(0));
}

fn openai_error_to_ollama(root: Value) -> Value {
    let message = root
        .get("error")
        .and_then(|value| value.get("message"))
        .and_then(Value::as_str)
        .unwrap_or("upstream error");
    json!({"error": message})
}

fn ollama_done_reason_from_openai_finish(reason: &str) -> Option<&'static str> {
    match reason {
        "stop" => Some("stop"),
        "length" => Some("length"),
        "tool_calls" => Some("stop"),
        "content_filter" => Some("content_filter"),
        _ => None,
    }
}

fn ollama_model_name(root: &Value) -> Option<&str> {
    root.get("model")
        .or_else(|| root.get("name"))
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
}

fn ollama_requested_model(root: &Value, backend_model: &str) -> String {
    ollama_model_name(root).unwrap_or(backend_model).to_owned()
}

fn ollama_keep_alive_is_zero(value: Option<&Value>) -> bool {
    match value {
        Some(Value::Number(number)) => number.as_i64() == Some(0) || number.as_u64() == Some(0),
        Some(Value::String(text)) => matches!(text.trim(), "0" | "0s" | "0m" | "0h"),
        _ => false,
    }
}

fn generated_tool_call_id(name: &str, ordinal: usize) -> String {
    format!("call_{}_{}", safe_tool_id_component(name), ordinal)
}

fn current_rfc3339() -> String {
    let seconds = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_secs() as i64)
        .unwrap_or(0);
    unix_seconds_to_rfc3339(seconds)
}

fn unix_seconds_to_rfc3339(seconds: i64) -> String {
    let days = seconds.div_euclid(86_400);
    let second_of_day = seconds.rem_euclid(86_400);
    let (year, month, day) = civil_from_days(days);
    let hour = second_of_day / 3_600;
    let minute = (second_of_day % 3_600) / 60;
    let second = second_of_day % 60;
    format!("{year:04}-{month:02}-{day:02}T{hour:02}:{minute:02}:{second:02}Z")
}

fn civil_from_days(days_since_unix_epoch: i64) -> (i64, u32, u32) {
    let z = days_since_unix_epoch + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 }.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096).div_euclid(365);
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2).div_euclid(153);
    let day = doy - (153 * mp + 2).div_euclid(5) + 1;
    let month = mp + if mp < 10 { 3 } else { -9 };
    let year = y + if month <= 2 { 1 } else { 0 };
    (year, month as u32, day as u32)
}

pub fn rewrite_openai_responses_request(mut root: Value) -> (Value, ResponseRewriteState) {
    let mut state = ResponseRewriteState::default();

    let Some(tools) = root.get_mut("tools").and_then(Value::as_array_mut) else {
        return (root, state);
    };

    let old_tools = std::mem::take(tools);
    let mut new_tools = Vec::with_capacity(old_tools.len());

    for tool in old_tools {
        let tool_type = tool.get("type").and_then(Value::as_str).unwrap_or("function");
        if tool_type == "namespace" {
            flatten_namespace_tool(tool, &mut new_tools, &mut state);
            continue;
        }

        new_tools.push(normalize_responses_function_tool(tool, None));
    }

    *tools = new_tools;
    (root, state)
}

fn flatten_namespace_tool(
    tool: Value,
    out: &mut Vec<Value>,
    state: &mut ResponseRewriteState,
) {
    let namespace = tool
        .get("name")
        .and_then(Value::as_str)
        .filter(|name| !name.is_empty())
        .unwrap_or("namespace")
        .to_owned();

    let Some(subtools) = tool.get("tools").and_then(Value::as_array) else {
        out.push(normalize_responses_function_tool(tool, None));
        return;
    };

    for subtool in subtools {
        let sub_name = subtool
            .get("name")
            .and_then(Value::as_str)
            .or_else(|| {
                subtool
                    .get("function")
                    .and_then(|v| v.get("name"))
                    .and_then(Value::as_str)
            })
            .filter(|name| !name.is_empty())
            .unwrap_or("tool");
        let prefixed = format!("{namespace}__{sub_name}");
        state
            .tool_name_map
            .insert(prefixed.clone(), sub_name.to_owned());
        state.namespace_tool_map.insert(
            prefixed.clone(),
            NamespacedToolName {
                namespace: namespace.clone(),
                name: sub_name.to_owned(),
            },
        );
        out.push(normalize_responses_function_tool(
            subtool.clone(),
            Some(prefixed),
        ));
    }
}

fn normalize_responses_function_tool(tool: Value, override_name: Option<String>) -> Value {
    let mut obj = match tool {
        Value::Object(obj) => obj,
        _ => Map::new(),
    };

    let original_type = obj
        .get("type")
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
        .map(str::to_owned);

    if let Some(Value::Object(function)) = obj.remove("function") {
        copy_if_absent(&mut obj, &function, "name");
        copy_if_absent(&mut obj, &function, "description");
        copy_if_absent(&mut obj, &function, "parameters");
    }

    obj.insert("type".to_owned(), Value::String("function".to_owned()));

    let name = override_name
        .or_else(|| {
            obj.get("name")
                .and_then(Value::as_str)
                .filter(|value| !value.is_empty())
                .map(str::to_owned)
        })
        .or(original_type)
        .unwrap_or_else(|| "tool".to_owned());
    obj.insert("name".to_owned(), Value::String(name));

    if !obj.contains_key("description") {
        obj.insert("description".to_owned(), Value::String(String::new()));
    }
    if !obj.contains_key("parameters") {
        obj.insert(
            "parameters".to_owned(),
            json!({"type":"object","properties":{}}),
        );
    }

    Value::Object(obj)
}

fn copy_if_absent(dst: &mut Map<String, Value>, src: &Map<String, Value>, key: &str) {
    if !dst.contains_key(key) {
        if let Some(value) = src.get(key) {
            dst.insert(key.to_owned(), value.clone());
        }
    }
}

pub fn rewrite_tool_names_in_value(value: &mut Value, state: &ResponseRewriteState) {
    rewrite_responses_namespaced_tool_calls(value, state, CodexNamespaceResponseMode::Flat);
}

pub fn rewrite_responses_namespaced_tool_calls(
    value: &mut Value,
    state: &ResponseRewriteState,
    mode: CodexNamespaceResponseMode,
) {
    if state.tool_name_map.is_empty() {
        return;
    }

    match value {
        Value::Object(obj) => {
            if is_responses_function_call_object(obj) {
                rewrite_responses_function_call_object(obj, state, mode);
            }
            for value in obj.values_mut() {
                rewrite_responses_namespaced_tool_calls(value, state, mode);
            }
        }
        Value::Array(values) => {
            for value in values {
                rewrite_responses_namespaced_tool_calls(value, state, mode);
            }
        }
        _ => {}
    }
}

fn is_responses_function_call_object(obj: &Map<String, Value>) -> bool {
    matches!(obj.get("type").and_then(Value::as_str), Some("function_call"))
        && obj.get("name").and_then(Value::as_str).is_some()
}

fn rewrite_responses_function_call_object(
    obj: &mut Map<String, Value>,
    state: &ResponseRewriteState,
    mode: CodexNamespaceResponseMode,
) {
    let Some(prefixed_name) = obj.get("name").and_then(Value::as_str).map(str::to_owned) else {
        return;
    };
    let Some(tool_ref) = state.namespace_tool_map.get(&prefixed_name).cloned() else {
        if let Some(rewritten) = state.tool_name_map.get(&prefixed_name) {
            obj.insert("name".to_owned(), Value::String(rewritten.clone()));
        }
        return;
    };

    match mode {
        CodexNamespaceResponseMode::Flat => {
            obj.insert("name".to_owned(), Value::String(tool_ref.name));
        }
        CodexNamespaceResponseMode::ExperimentalWrapped => {
            wrap_responses_function_call_object(obj, &tool_ref);
        }
    }
}

fn wrap_responses_function_call_object(obj: &mut Map<String, Value>, tool_ref: &NamespacedToolName) {
    let mut inner_call = obj.clone();
    inner_call.insert("name".to_owned(), Value::String(tool_ref.name.clone()));

    let mut wrapper = inner_call.clone();
    wrapper.insert("type".to_owned(), Value::String("namespace_call".to_owned()));
    wrapper.insert(
        "namespace".to_owned(),
        Value::String(tool_ref.namespace.clone()),
    );
    wrapper.insert("name".to_owned(), Value::String(tool_ref.name.clone()));
    wrapper.insert("call".to_owned(), Value::Object(inner_call));

    *obj = wrapper;
}

pub fn rewrite_openai_responses_response(root: Value, state: &ResponseRewriteState) -> Value {
    rewrite_openai_responses_response_with_mode(root, state, CodexNamespaceResponseMode::Flat)
}

pub fn rewrite_openai_responses_response_with_mode(
    mut root: Value,
    state: &ResponseRewriteState,
    mode: CodexNamespaceResponseMode,
) -> Value {
    rewrite_responses_namespaced_tool_calls(&mut root, state, mode);
    root
}

pub fn rewrite_anthropic_messages_request(root: Value) -> Value {
    rewrite_anthropic_messages_request_with_mode(root, AnthropicSchemaMode::default())
}

pub fn rewrite_anthropic_messages_request_with_mode(
    mut root: Value,
    schema_mode: AnthropicSchemaMode,
) -> Value {
    let Some(tools) = root.get_mut("tools").and_then(Value::as_array_mut) else {
        return root;
    };

    for tool in tools {
        let Some(tool_obj) = tool.as_object_mut() else {
            continue;
        };
        let input_schema = tool_obj
            .remove("input_schema")
            .unwrap_or_else(|| json!({"type":"object","properties":{}}));
        tool_obj.insert(
            "input_schema".to_owned(),
            force_object_schema(sanitize_json_schema_with_mode(&input_schema, schema_mode)),
        );
    }

    root
}

pub fn rewrite_anthropic_messages_response(mut root: Value) -> Value {
    if is_anthropic_error_shape(&root) || !looks_like_anthropic_message_response(&root) {
        return root;
    }

    add_anthropic_message_defaults(&mut root);
    root
}

pub fn rewrite_anthropic_messages_sse_data(data: &str) -> String {
    if data.trim() == "[DONE]" {
        return data.to_owned();
    }
    let Ok(mut value) = serde_json::from_str::<Value>(data) else {
        return data.to_owned();
    };
    add_anthropic_sse_defaults(&mut value);
    serde_json::to_string(&value).unwrap_or_else(|_| data.to_owned())
}

fn add_anthropic_sse_defaults(value: &mut Value) {
    if is_anthropic_error_shape(value) {
        return;
    }

    if let Some(message) = value.get_mut("message") {
        if !is_anthropic_error_shape(message) && looks_like_anthropic_message_response(message) {
            add_anthropic_message_defaults(message);
        }
    }
    if let Some(content_block) = value.get_mut("content_block") {
        add_anthropic_content_block_defaults(content_block);
    }

    if looks_like_anthropic_message_response(value) {
        add_anthropic_message_defaults(value);
    }
}

fn is_anthropic_error_shape(value: &Value) -> bool {
    value.get("type").and_then(Value::as_str) == Some("error") || value.get("error").is_some()
}

fn looks_like_anthropic_message_response(value: &Value) -> bool {
    let Some(obj) = value.as_object() else {
        return false;
    };

    match obj.get("type").and_then(Value::as_str) {
        Some("message") => return true,
        Some("error") => return false,
        Some(_) => {}
        None => {}
    }

    obj.contains_key("content")
        || obj
            .get("id")
            .and_then(Value::as_str)
            .is_some_and(|id| id.starts_with("msg_"))
        || obj.get("role").and_then(Value::as_str) == Some("assistant")
}

fn add_anthropic_message_defaults(value: &mut Value) {
    let Some(obj) = value.as_object_mut() else {
        return;
    };

    insert_default_string(obj, "type", "message");
    insert_default_string(obj, "role", "assistant");
    obj.entry("content".to_owned()).or_insert_with(|| Value::Array(Vec::new()));
    obj.entry("stop_reason".to_owned()).or_insert(Value::Null);
    obj.entry("stop_sequence".to_owned()).or_insert(Value::Null);
    obj.entry("usage".to_owned()).or_insert_with(|| {
        json!({"input_tokens":0,"output_tokens":0})
    });

    if let Some(content) = obj.get_mut("content").and_then(Value::as_array_mut) {
        for block in content {
            add_anthropic_content_block_defaults(block);
        }
    }
}

fn add_anthropic_content_block_defaults(value: &mut Value) {
    let Some(obj) = value.as_object_mut() else {
        return;
    };

    obj.entry("cache_control".to_owned())
        .or_insert_with(|| json!({"type":"ephemeral"}));

    match obj.get("type").and_then(Value::as_str) {
        Some("text") => {
            obj.entry("text".to_owned())
                .or_insert_with(|| Value::String(String::new()));
        }
        Some("tool_use") => {
            obj.entry("input".to_owned()).or_insert_with(|| json!({}));
        }
        _ => {}
    }
}

fn insert_default_string(obj: &mut Map<String, Value>, key: &str, value: &str) {
    obj.entry(key.to_owned())
        .or_insert_with(|| Value::String(value.to_owned()));
}

fn force_object_schema(value: Value) -> Value {
    match value {
        Value::Object(mut obj) => {
            obj.insert("type".to_owned(), Value::String("object".to_owned()));
            if !obj.contains_key("properties") {
                obj.insert("properties".to_owned(), json!({}));
            }
            Value::Object(obj)
        }
        _ => json!({"type":"object","properties":{}}),
    }
}

pub fn sanitize_json_schema(schema: &Value) -> Value {
    sanitize_json_schema_with_mode(schema, AnthropicSchemaMode::default())
}

pub fn sanitize_json_schema_with_mode(schema: &Value, mode: AnthropicSchemaMode) -> Value {
    let Value::Object(map) = schema else {
        return json!({"type":"string"});
    };

    if let Some((key, values)) = schema_combinator(map) {
        return sanitize_combinator_schema(map, key, values, mode);
    }

    sanitize_plain_schema(map, mode)
}

fn sanitize_plain_schema(map: &Map<String, Value>, mode: AnthropicSchemaMode) -> Value {
    let schema_type = normalized_schema_type(map).unwrap_or_else(|| infer_schema_type(map));
    let mut out = Map::new();
    out.insert("type".to_owned(), Value::String(schema_type.to_owned()));

    copy_common_schema_keywords(map, &mut out, mode);

    match schema_type {
        "object" => {
            let properties = sanitize_properties(map.get("properties"), mode);
            let property_names: std::collections::HashSet<String> = properties.keys().cloned().collect();
            out.insert("properties".to_owned(), Value::Object(properties));
            if let Some(required) = sanitized_required(map.get("required"), &property_names) {
                out.insert("required".to_owned(), required);
            }
            if let Some(additional) = sanitize_additional_properties(map.get("additionalProperties"), mode) {
                out.insert("additionalProperties".to_owned(), additional);
            }
            if mode == AnthropicSchemaMode::Semantic {
                copy_usize_keyword(map, &mut out, "minProperties");
                copy_usize_keyword(map, &mut out, "maxProperties");
            }
        }
        "array" => {
            let items = map
                .get("items")
                .map(|value| sanitize_json_schema_with_mode(value, mode))
                .unwrap_or_else(|| json!({"type":"string"}));
            out.insert("items".to_owned(), items);
            if mode == AnthropicSchemaMode::Semantic {
                copy_usize_keyword(map, &mut out, "minItems");
                copy_usize_keyword(map, &mut out, "maxItems");
                copy_bool(map, &mut out, "uniqueItems");
            }
        }
        "string"
            if mode == AnthropicSchemaMode::Semantic => {
                copy_string(map, &mut out, "format");
                copy_string(map, &mut out, "pattern");
                copy_usize_keyword(map, &mut out, "minLength");
                copy_usize_keyword(map, &mut out, "maxLength");
            }
        "integer" | "number"
            if mode == AnthropicSchemaMode::Semantic => {
                copy_number(map, &mut out, "minimum");
                copy_number(map, &mut out, "maximum");
                copy_number(map, &mut out, "exclusiveMinimum");
                copy_number(map, &mut out, "exclusiveMaximum");
                copy_number(map, &mut out, "multipleOf");
            }
        _ => {}
    }

    append_parser_compat_notes(map, &mut out, mode);
    Value::Object(out)
}

fn schema_combinator(map: &Map<String, Value>) -> Option<(&'static str, &Vec<Value>)> {
    for key in ["anyOf", "oneOf", "allOf"] {
        if let Some(Value::Array(values)) = map.get(key) {
            return Some((key, values));
        }
    }
    None
}

fn sanitize_combinator_schema(
    parent: &Map<String, Value>,
    key: &'static str,
    values: &[Value],
    mode: AnthropicSchemaMode,
) -> Value {
    let non_null: Vec<&Value> = values
        .iter()
        .filter(|value| !schema_is_null_type(value))
        .collect();

    if non_null.len() == 1 {
        let mut out = sanitize_json_schema_with_mode(non_null[0], mode);
        merge_parent_schema_annotations(parent, &mut out, mode);
        if non_null.len() != values.len() {
            append_schema_note(&mut out, "Value may also be null.".to_owned());
        }
        return out;
    }

    let sanitized: Vec<Value> = non_null
        .iter()
        .map(|value| sanitize_json_schema_with_mode(value, mode))
        .collect();
    let schema_type = common_sanitized_type(&sanitized)
        .or_else(|| normalized_schema_type(parent))
        .unwrap_or("string");

    let mut out = Map::new();
    out.insert("type".to_owned(), Value::String(schema_type.to_owned()));
    copy_common_schema_keywords(parent, &mut out, mode);

    match schema_type {
        "object" => {
            let properties = merge_object_variant_properties(&sanitized);
            let property_names: std::collections::HashSet<String> = properties.keys().cloned().collect();
            out.insert("properties".to_owned(), Value::Object(properties));
            if key == "allOf" {
                if let Some(required) = merged_all_of_required(&sanitized, &property_names) {
                    out.insert("required".to_owned(), required);
                }
            }
        }
        "array" => {
            let item = sanitized
                .iter()
                .find_map(|value| value.get("items").cloned())
                .unwrap_or_else(|| json!({"type":"string"}));
            out.insert("items".to_owned(), item);
        }
        _ => {
            if mode == AnthropicSchemaMode::Semantic {
                if let Some(enum_values) = merged_enum_values(&sanitized, key) {
                    out.insert("enum".to_owned(), enum_values);
                }
            }
        }
    }

    let summaries = sanitized
        .iter()
        .map(schema_summary)
        .collect::<Vec<String>>()
        .join("; ");
    if !summaries.is_empty() {
        append_schema_note_obj(
            &mut out,
            format!("Original {key} alternatives were preserved as a parser-safe schema: {summaries}."),
        );
    }
    if non_null.len() != values.len() {
        append_schema_note_obj(&mut out, "Value may also be null.".to_owned());
    }
    append_parser_compat_notes(parent, &mut out, mode);

    Value::Object(out)
}

fn merge_parent_schema_annotations(
    parent: &Map<String, Value>,
    out: &mut Value,
    mode: AnthropicSchemaMode,
) {
    let parent_description = parent.get("description").and_then(Value::as_str).map(str::to_owned);
    let mut append_parent_description = false;

    if let Some(out_obj) = out.as_object_mut() {
        if mode == AnthropicSchemaMode::Semantic && !out_obj.contains_key("title") {
            copy_string(parent, out_obj, "title");
        }
        if !out_obj.contains_key("description") {
            copy_string(parent, out_obj, "description");
        } else if parent_description.is_some() {
            append_parent_description = true;
        }
        append_parser_compat_notes(parent, out_obj, mode);
    }

    if append_parent_description {
        if let Some(description) = parent_description {
            append_schema_note(out, description);
        }
    }
}

fn common_sanitized_type(values: &[Value]) -> Option<&'static str> {
    let mut iter = values.iter().filter_map(|value| {
        value
            .get("type")
            .and_then(Value::as_str)
            .and_then(normalize_schema_type_str)
    });
    let first = iter.next()?;
    if iter.all(|value| value == first) {
        Some(first)
    } else {
        None
    }
}

fn merge_object_variant_properties(values: &[Value]) -> Map<String, Value> {
    let mut merged = Map::new();
    for value in values {
        let Some(properties) = value.get("properties").and_then(Value::as_object) else {
            continue;
        };
        for (name, schema) in properties {
            merged.entry(name.clone()).or_insert_with(|| schema.clone());
        }
    }
    merged
}

fn merged_all_of_required(
    values: &[Value],
    property_names: &std::collections::HashSet<String>,
) -> Option<Value> {
    let mut required = Vec::new();
    for value in values {
        if let Some(items) = value.get("required").and_then(Value::as_array) {
            for item in items {
                let Some(name) = item.as_str() else {
                    continue;
                };
                if property_names.contains(name) && !required.iter().any(|existing: &Value| existing.as_str() == Some(name)) {
                    required.push(Value::String(name.to_owned()));
                }
            }
        }
    }
    if required.is_empty() {
        None
    } else {
        Some(Value::Array(required))
    }
}

fn merged_enum_values(values: &[Value], key: &str) -> Option<Value> {
    if key == "allOf" {
        let mut iter = values.iter();
        let first = iter.next()?.get("enum").and_then(Value::as_array)?.clone();
        let mut intersection = first;
        for value in iter {
            let items = value.get("enum").and_then(Value::as_array)?;
            intersection.retain(|item| items.contains(item));
        }
        if intersection.is_empty() {
            None
        } else {
            Some(Value::Array(intersection))
        }
    } else {
        let mut union: Vec<Value> = Vec::new();
        for value in values {
            let items = value.get("enum").and_then(Value::as_array)?;
            for item in items {
                if !union.contains(item) {
                    union.push(item.clone());
                }
            }
        }
        if union.is_empty() {
            None
        } else {
            Some(Value::Array(union))
        }
    }
}

fn schema_summary(value: &Value) -> String {
    let Some(obj) = value.as_object() else {
        return "schema".to_owned();
    };
    let type_name = obj.get("type").and_then(Value::as_str).unwrap_or("schema");
    let mut parts = vec![type_name.to_owned()];
    if let Some(enum_values) = obj.get("enum").and_then(Value::as_array) {
        let rendered = enum_values
            .iter()
            .map(|value| value.to_string())
            .collect::<Vec<String>>()
            .join("|");
        parts.push(format!("enum={rendered}"));
    }
    if let Some(properties) = obj.get("properties").and_then(Value::as_object) {
        let keys = properties.keys().cloned().collect::<Vec<String>>().join(",");
        parts.push(format!("properties={keys}"));
    }
    if let Some(required) = obj.get("required").and_then(Value::as_array) {
        let keys = required
            .iter()
            .filter_map(Value::as_str)
            .collect::<Vec<&str>>()
            .join(",");
        parts.push(format!("required={keys}"));
    }
    if let Some(description) = obj.get("description").and_then(Value::as_str) {
        let one_line = description.replace('\n', " ");
        let summary: String = one_line.chars().take(180).collect();
        if !summary.is_empty() {
            parts.push(format!("notes={summary}"));
        }
    }
    parts.join(" ")
}

fn copy_common_schema_keywords(
    src: &Map<String, Value>,
    dst: &mut Map<String, Value>,
    mode: AnthropicSchemaMode,
) {
    copy_string(src, dst, "description");
    if mode == AnthropicSchemaMode::Semantic {
        copy_string(src, dst, "title");
        copy_array(src, dst, "enum");
        copy_value(src, dst, "const");
        copy_value(src, dst, "default");
    }
}

fn append_parser_compat_notes(
    src: &Map<String, Value>,
    dst: &mut Map<String, Value>,
    mode: AnthropicSchemaMode,
) {
    if mode != AnthropicSchemaMode::LlamaServerCompat {
        return;
    }

    let mut notes = Vec::new();
    for key in [
        "title",
        "enum",
        "const",
        "default",
        "format",
        "pattern",
        "minLength",
        "maxLength",
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "multipleOf",
        "minItems",
        "maxItems",
        "uniqueItems",
        "minProperties",
        "maxProperties",
    ] {
        if dst.contains_key(key) {
            continue;
        }
        if let Some(value) = src.get(key) {
            let rendered = serde_json::to_string(value).unwrap_or_else(|_| "<unrenderable>".to_owned());
            notes.push(format!("{key}={rendered}"));
        }
    }

    if !notes.is_empty() {
        append_schema_note_obj(
            dst,
            format!(
                "Schema constraints retained for model guidance but omitted from forwarded JSON Schema for llama-server parser compatibility: {}.",
                notes.join(", ")
            ),
        );
    }
}

fn append_schema_note(value: &mut Value, note: String) {
    let Some(obj) = value.as_object_mut() else {
        return;
    };
    append_schema_note_obj(obj, note);
}

fn append_schema_note_obj(obj: &mut Map<String, Value>, note: String) {
    match obj.get_mut("description") {
        Some(Value::String(existing)) if !existing.is_empty() => {
            existing.push_str("\n\nSchema notes: ");
            existing.push_str(&note);
        }
        _ => {
            obj.insert("description".to_owned(), Value::String(note));
        }
    }
}

fn schema_is_null_type(value: &Value) -> bool {
    value
        .get("type")
        .and_then(Value::as_str)
        .map(|value| value.eq_ignore_ascii_case("null"))
        .unwrap_or(false)
}

fn normalized_schema_type(map: &Map<String, Value>) -> Option<&'static str> {
    match map.get("type") {
        Some(Value::String(value)) => normalize_schema_type_str(value),
        Some(Value::Array(values)) => values
            .iter()
            .filter_map(Value::as_str)
            .filter(|value| !value.eq_ignore_ascii_case("null"))
            .find_map(normalize_schema_type_str),
        _ => None,
    }
}

fn normalize_schema_type_str(value: &str) -> Option<&'static str> {
    match value.to_ascii_lowercase().as_str() {
        "object" => Some("object"),
        "array" => Some("array"),
        "string" => Some("string"),
        "integer" => Some("integer"),
        "number" => Some("number"),
        "boolean" => Some("boolean"),
        _ => None,
    }
}

fn infer_schema_type(map: &Map<String, Value>) -> &'static str {
    if map.contains_key("properties") {
        "object"
    } else if map.contains_key("items") {
        "array"
    } else if let Some(Value::Array(values)) = map.get("enum") {
        values
            .first()
            .and_then(|value| match value {
                Value::Bool(_) => Some("boolean"),
                Value::Number(number) if number.is_i64() || number.is_u64() => Some("integer"),
                Value::Number(_) => Some("number"),
                Value::Object(_) => Some("object"),
                Value::Array(_) => Some("array"),
                Value::String(_) => Some("string"),
                Value::Null => None,
            })
            .unwrap_or("string")
    } else {
        "string"
    }
}

fn sanitize_properties(value: Option<&Value>, mode: AnthropicSchemaMode) -> Map<String, Value> {
    let mut out = Map::new();
    let Some(Value::Object(properties)) = value else {
        return out;
    };
    for (name, schema) in properties {
        out.insert(name.clone(), sanitize_json_schema_with_mode(schema, mode));
    }
    out
}

fn sanitized_required(
    value: Option<&Value>,
    property_names: &std::collections::HashSet<String>,
) -> Option<Value> {
    let required = value?.as_array()?;
    let filtered: Vec<Value> = required
        .iter()
        .filter_map(Value::as_str)
        .filter(|name| property_names.contains(*name))
        .map(|name| Value::String(name.to_owned()))
        .collect();
    if filtered.is_empty() {
        None
    } else {
        Some(Value::Array(filtered))
    }
}

fn sanitize_additional_properties(value: Option<&Value>, mode: AnthropicSchemaMode) -> Option<Value> {
    let value = value?;
    match value {
        Value::Bool(value) => Some(Value::Bool(*value)),
        Value::Object(_) => Some(sanitize_json_schema_with_mode(value, mode)),
        _ => None,
    }
}

fn copy_string(src: &Map<String, Value>, dst: &mut Map<String, Value>, key: &str) {
    if let Some(Value::String(value)) = src.get(key) {
        dst.insert(key.to_owned(), Value::String(value.clone()));
    }
}

fn copy_array(src: &Map<String, Value>, dst: &mut Map<String, Value>, key: &str) {
    if let Some(Value::Array(value)) = src.get(key) {
        dst.insert(key.to_owned(), Value::Array(value.clone()));
    }
}

fn copy_bool(src: &Map<String, Value>, dst: &mut Map<String, Value>, key: &str) {
    if let Some(Value::Bool(value)) = src.get(key) {
        dst.insert(key.to_owned(), Value::Bool(*value));
    }
}

fn copy_number(src: &Map<String, Value>, dst: &mut Map<String, Value>, key: &str) {
    if let Some(Value::Number(value)) = src.get(key) {
        dst.insert(key.to_owned(), Value::Number(value.clone()));
    }
}

fn copy_usize_keyword(src: &Map<String, Value>, dst: &mut Map<String, Value>, key: &str) {
    if let Some(Value::Number(value)) = src.get(key) {
        if value.as_u64().is_some() {
            dst.insert(key.to_owned(), Value::Number(value.clone()));
        }
    }
}

fn copy_value(src: &Map<String, Value>, dst: &mut Map<String, Value>, key: &str) {
    if let Some(value) = src.get(key) {
        dst.insert(key.to_owned(), value.clone());
    }
}

pub fn sanitize_backend_request(mut root: Value) -> Value {
    let Some(obj) = root.as_object_mut() else {
        return root;
    };

    obj.remove("reasoning_effort");
    obj.remove("thinking");

    if let Some(max_completion_tokens) = obj.remove("max_completion_tokens") {
        if !obj.contains_key("max_tokens") {
            obj.insert("max_tokens".to_owned(), max_completion_tokens);
        }
    }

    if let Some(messages) = obj.get_mut("messages").and_then(Value::as_array_mut) {
        for message in messages {
            let Some(message_obj) = message.as_object_mut() else {
                continue;
            };
            message_obj.remove("reasoning_content");
            message_obj.remove("reasoning");
        }
    }

    root
}

pub fn openai_models_to_gemini(root: Value) -> Value {
    let models: Vec<Value> = root
        .get("data")
        .and_then(Value::as_array)
        .map(|records| records.iter().filter_map(openai_model_record_to_gemini).collect())
        .unwrap_or_default();

    let mut out = Map::new();
    out.insert("models".to_owned(), Value::Array(models));
    if let Some(next_page_token) = root.get("nextPageToken").or_else(|| root.get("next_page_token")) {
        out.insert("nextPageToken".to_owned(), next_page_token.clone());
    }
    Value::Object(out)
}

fn openai_model_record_to_gemini(record: &Value) -> Option<Value> {
    let id = record
        .get("id")
        .and_then(Value::as_str)
        .or_else(|| record.get("name").and_then(Value::as_str))?
        .trim();
    if id.is_empty() {
        return None;
    }

    let model_id = id.strip_prefix("models/").unwrap_or(id);
    if model_id.is_empty() {
        return None;
    }

    let mut model = Map::new();
    model.insert("name".to_owned(), Value::String(format!("models/{model_id}")));
    model.insert(
        "displayName".to_owned(),
        Value::String(gemini_display_name(record, model_id)),
    );
    model.insert(
        "supportedGenerationMethods".to_owned(),
        json!(["generateContent", "streamGenerateContent"]),
    );

    if let Some(value) = record.get("created").and_then(model_version_from_created) {
        model.insert("version".to_owned(), Value::String(value));
    }
    copy_gemini_model_metadata(record, &mut model);

    Some(Value::Object(model))
}

fn gemini_display_name(record: &Value, model_id: &str) -> String {
    record
        .get("displayName")
        .and_then(Value::as_str)
        .or_else(|| record.get("display_name").and_then(Value::as_str))
        .or_else(|| {
            record
                .get("owned_by")
                .and_then(Value::as_str)
                .filter(|value| !value.trim().is_empty())
        })
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_owned)
        .unwrap_or_else(|| model_id.to_owned())
}

fn model_version_from_created(value: &Value) -> Option<String> {
    match value {
        Value::Number(number) => Some(number.to_string()),
        Value::String(value) if !value.trim().is_empty() => Some(value.clone()),
        _ => None,
    }
}

fn copy_gemini_model_metadata(record: &Value, out: &mut Map<String, Value>) {
    let metadata = model_metadata_object(record);
    for (openai_key, gemini_key) in [
        ("base_model_id", "baseModelId"),
        ("baseModelId", "baseModelId"),
        ("description", "description"),
        ("input_token_limit", "inputTokenLimit"),
        ("inputTokenLimit", "inputTokenLimit"),
        ("output_token_limit", "outputTokenLimit"),
        ("outputTokenLimit", "outputTokenLimit"),
    ] {
        if let Some(value) = record.get(openai_key).or_else(|| metadata.and_then(|meta| meta.get(openai_key))) {
            out.entry(gemini_key.to_owned()).or_insert_with(|| value.clone());
        }
    }
}

pub fn rewrite_gemini_request(root: Value, stream: bool, backend_model: &str) -> Value {
    let model = if backend_model.is_empty() {
        "local-model"
    } else {
        backend_model
    };
    let mut out = Map::new();
    out.insert("model".to_owned(), Value::String(model.to_owned()));

    let mut messages = Vec::new();
    if let Some(system) = gemini_system_instruction_text(root.get("systemInstruction")) {
        messages.push(json!({"role":"system","content":system}));
    }
    messages.extend(gemini_contents_to_openai_messages(root.get("contents")));
    if messages.is_empty() {
        messages.push(json!({"role":"user","content":""}));
    }
    out.insert("messages".to_owned(), Value::Array(messages));

    if let Some(tools) = gemini_tools_to_openai_tools(root.get("tools")) {
        out.insert("tools".to_owned(), tools);
        out.insert("tool_choice".to_owned(), Value::String("auto".to_owned()));
    }

    if let Some(config) = root.get("generationConfig") {
        copy_generation_config(config, &mut out);
    }
    if stream {
        out.insert("stream".to_owned(), Value::Bool(true));
    }

    Value::Object(out)
}

fn gemini_system_instruction_text(value: Option<&Value>) -> Option<String> {
    let value = value?;
    let parts = value.get("parts").and_then(Value::as_array)?;
    let text = collect_gemini_text_parts(parts);
    if text.is_empty() {
        None
    } else {
        Some(text)
    }
}

fn gemini_contents_to_openai_messages(value: Option<&Value>) -> Vec<Value> {
    let mut messages = Vec::new();
    let Some(contents) = value.and_then(Value::as_array) else {
        return messages;
    };

    let mut call_ids = GeminiToolCallIdTracker::default();
    for content in contents {
        let role = content
            .get("role")
            .and_then(Value::as_str)
            .unwrap_or("user");
        let message_role = gemini_role_to_openai(role);
        let parts = content.get("parts").and_then(Value::as_array).cloned().unwrap_or_default();
        if parts.is_empty() {
            continue;
        }

        let mut text_parts: Vec<String> = Vec::new();
        let mut pending_function_calls: Vec<Value> = Vec::new();

        for part in &parts {
            if let Some(text) = part.get("text").and_then(Value::as_str) {
                if !pending_function_calls.is_empty() {
                    flush_gemini_function_calls(&mut messages, &mut pending_function_calls);
                }
                if !text.is_empty() {
                    text_parts.push(text.to_owned());
                }
                continue;
            }

            if let Some(call) = part.get("functionCall").and_then(Value::as_object) {
                flush_gemini_text_parts(&mut messages, message_role, &mut text_parts);
                pending_function_calls.push(gemini_function_call_to_openai(call, &mut call_ids));
                continue;
            }

            if let Some(response) = part.get("functionResponse").and_then(Value::as_object) {
                if !pending_function_calls.is_empty() {
                    flush_gemini_function_calls(&mut messages, &mut pending_function_calls);
                }
                flush_gemini_text_parts(&mut messages, message_role, &mut text_parts);
                let translated = gemini_function_response_to_openai(response, &mut call_ids);
                messages.extend(translated);
            }
        }

        if !pending_function_calls.is_empty() {
            flush_gemini_function_calls(&mut messages, &mut pending_function_calls);
        }
        flush_gemini_text_parts(&mut messages, message_role, &mut text_parts);
    }

    messages
}

fn gemini_role_to_openai(role: &str) -> &'static str {
    match role {
        "model" | "assistant" => "assistant",
        "system" => "system",
        _ => "user",
    }
}

fn flush_gemini_text_parts(messages: &mut Vec<Value>, role: &str, text_parts: &mut Vec<String>) {
    if text_parts.is_empty() {
        return;
    }
    let content = text_parts
        .iter()
        .map(String::as_str)
        .collect::<Vec<_>>()
        .join("\n")
        .trim()
        .to_owned();
    text_parts.clear();
    if !content.is_empty() {
        messages.push(json!({"role": role, "content": content}));
    }
}

fn flush_gemini_function_calls(messages: &mut Vec<Value>, calls: &mut Vec<Value>) {
    if calls.is_empty() {
        return;
    }
    let tool_calls = std::mem::take(calls);
    messages.push(json!({
        "role": "assistant",
        "content": Value::Null,
        "tool_calls": tool_calls,
    }));
}

fn collect_gemini_text_parts(parts: &[Value]) -> String {
    let joined = parts
        .iter()
        .filter_map(|part| part.get("text").and_then(Value::as_str))
        .collect::<Vec<_>>()
        .join("\n");
    joined.trim().to_owned()
}

#[derive(Debug, Clone, Default)]
struct GeminiToolCallIdTracker {
    generated_call_index: usize,
    pending_by_name: HashMap<String, VecDeque<String>>,
}

impl GeminiToolCallIdTracker {
    fn record_call(&mut self, name: &str) -> String {
        let id = self.next_id(name);
        self.pending_by_name
            .entry(name.to_owned())
            .or_default()
            .push_back(id.clone());
        id
    }

    fn resolve_response(&mut self, name: &str) -> Option<String> {
        self.pending_by_name
            .get_mut(name)
            .and_then(VecDeque::pop_front)
    }

    fn orphan_response_call_id(&mut self, name: &str) -> String {
        self.next_id(name)
    }

    fn next_id(&mut self, name: &str) -> String {
        let id = format!(
            "call_{}_{}",
            safe_tool_id_component(name), self.generated_call_index
        );
        self.generated_call_index += 1;
        id
    }
}

fn gemini_function_call_to_openai(
    call: &Map<String, Value>,
    call_ids: &mut GeminiToolCallIdTracker,
) -> Value {
    let name = call
        .get("name")
        .and_then(Value::as_str)
        .unwrap_or("tool")
        .to_owned();
    let args = call.get("args").cloned().unwrap_or_else(|| json!({}));
    let id = call_ids.record_call(&name);
    json!({
        "id": id,
        "type":"function",
        "function": {
            "name": name,
            "arguments": json_arguments_string(args),
        }
    })
}

fn gemini_function_response_to_openai(
    response: &Map<String, Value>,
    call_ids: &mut GeminiToolCallIdTracker,
) -> Vec<Value> {
    let name = response
        .get("name")
        .and_then(Value::as_str)
        .unwrap_or("tool")
        .to_owned();
    let (tool_call_id, synthetic_call) = match call_ids.resolve_response(&name) {
        Some(id) => (id, None),
        None => {
            let id = call_ids.orphan_response_call_id(&name);
            let call = json!({
                "role": "assistant",
                "content": Value::Null,
                "tool_calls": [{
                    "id": id.clone(),
                    "type": "function",
                    "function": {
                        "name": name.clone(),
                        "arguments": "{}",
                    }
                }]
            });
            (id, Some(call))
        }
    };
    let content = response
        .get("response")
        .cloned()
        .map(json_arguments_string)
        .unwrap_or_else(|| "{}".to_owned());
    let tool_message = json!({
        "role":"tool",
        "tool_call_id": tool_call_id,
        "name": name,
        "content": content,
    });
    match synthetic_call {
        Some(call) => vec![call, tool_message],
        None => vec![tool_message],
    }
}


fn safe_tool_id_component(value: &str) -> String {
    let mut out = String::with_capacity(value.len());
    for ch in value.chars() {
        if ch.is_ascii_alphanumeric() || ch == '_' || ch == '-' {
            out.push(ch);
        } else {
            out.push('_');
        }
    }
    if out.is_empty() {
        "tool".to_owned()
    } else {
        out
    }
}

fn json_arguments_string(value: Value) -> String {
    match value {
        Value::String(value) => value,
        value => serde_json::to_string(&value).unwrap_or_else(|_| "{}".to_owned()),
    }
}

fn gemini_tools_to_openai_tools(value: Option<&Value>) -> Option<Value> {
    let tool_sets = value?.as_array()?;
    let mut tools = Vec::new();
    for tool_set in tool_sets {
        let Some(declarations) = tool_set
            .get("functionDeclarations")
            .and_then(Value::as_array)
        else {
            continue;
        };
        for declaration in declarations {
            let Some(decl_obj) = declaration.as_object() else {
                continue;
            };
            let name = decl_obj
                .get("name")
                .and_then(Value::as_str)
                .unwrap_or("tool");
            let description = decl_obj
                .get("description")
                .and_then(Value::as_str)
                .unwrap_or("");
            let parameters = decl_obj
                .get("parameters")
                .map(|value| sanitize_json_schema_with_mode(value, AnthropicSchemaMode::Semantic))
                .unwrap_or_else(|| json!({"type":"object","properties":{}}));
            tools.push(json!({
                "type":"function",
                "function": {
                    "name": name,
                    "description": description,
                    "parameters": force_object_schema(parameters),
                }
            }));
        }
    }
    if tools.is_empty() {
        None
    } else {
        Some(Value::Array(tools))
    }
}

fn copy_generation_config(config: &Value, out: &mut Map<String, Value>) {
    let Some(config) = config.as_object() else {
        return;
    };
    // The brief only requires mapping the first OpenAI choice back to
    // Gemini candidates[0]. Do not forward Gemini candidateCount to
    // OpenAI `n` unless the response path also supports every returned
    // choice; otherwise the proxy would ask the backend for candidates it
    // later drops.
    for (gemini_key, openai_key) in [
        ("temperature", "temperature"),
        ("topP", "top_p"),
        ("maxOutputTokens", "max_tokens"),
    ] {
        if let Some(value) = config.get(gemini_key) {
            out.insert(openai_key.to_owned(), value.clone());
        }
    }
}

pub fn is_probable_gemini_classification_request(path: &str, body: &Value) -> bool {
    if !path.contains("flash-lite") || is_gemini_stream_path(path) {
        return false;
    }
    let has_tools = body
        .get("tools")
        .and_then(Value::as_array)
        .map(|tools| !tools.is_empty())
        .unwrap_or(false);
    !has_tools
}

pub fn hardcoded_gemini_classification_response() -> Value {
    let routing = json!({
        "route":"general",
        "category":"general",
        "type":"general",
        "confidence":1.0,
        "needs_tool":false,
        "requires_tool":false
    });
    json!({
        "candidates": [{
            "index": 0,
            "content": {
                "role":"model",
                "parts": [{"text": routing.to_string()}]
            },
            "finishReason":"STOP"
        }],
        "usageMetadata": {
            "promptTokenCount": 0,
            "candidatesTokenCount": 0,
            "totalTokenCount": 0
        }
    })
}

pub fn openai_chat_response_to_gemini(root: Value, http_status: u16) -> Value {
    if root.get("error").is_some() {
        return openai_error_to_gemini(root, http_status);
    }

    let choice = root
        .get("choices")
        .and_then(Value::as_array)
        .and_then(|choices| choices.first())
        .cloned()
        .unwrap_or_else(|| json!({}));
    let message = choice.get("message").cloned().unwrap_or_else(|| json!({}));
    let finish_reason = choice
        .get("finish_reason")
        .and_then(Value::as_str)
        .map(map_openai_finish_reason)
        .unwrap_or("STOP");

    let mut parts = Vec::new();
    if let Some(content) = message.get("content").and_then(Value::as_str) {
        if !content.is_empty() {
            parts.push(json!({"text": content}));
        }
    }
    parts.extend(openai_tool_calls_to_gemini_parts(message.get("tool_calls")));
    if let Some(function_call) = message.get("function_call") {
        parts.push(openai_legacy_function_call_to_gemini_part(function_call));
    }
    if parts.is_empty() {
        parts.push(json!({"text":""}));
    }

    let mut out = json!({
        "candidates": [{
            "index": 0,
            "content": {"role":"model","parts": parts},
            "finishReason": finish_reason
        }]
    });

    if let Some(usage) = openai_usage_to_gemini(root.get("usage")) {
        out.as_object_mut()
            .expect("object")
            .insert("usageMetadata".to_owned(), usage);
    }

    out
}

fn openai_error_to_gemini(root: Value, http_status: u16) -> Value {
    let message = root
        .get("error")
        .and_then(|value| value.get("message"))
        .and_then(Value::as_str)
        .unwrap_or("upstream error");
    let code = normalized_gemini_error_code(&root, http_status);
    json!({"error":{"code":code,"message":message,"status":status_to_gemini(code)}})
}

fn normalized_gemini_error_code(root: &Value, http_status: u16) -> u16 {
    if (400..=599).contains(&http_status) {
        return http_status;
    }

    if let Some(code) = root
        .get("error")
        .and_then(|error| error.get("code"))
        .and_then(error_code_value_to_http_status)
    {
        return code;
    }

    if let Some(code) = root
        .get("error")
        .and_then(|error| error.get("type"))
        .and_then(Value::as_str)
        .and_then(openai_error_type_to_http_status)
    {
        return code;
    }

    500
}

fn error_code_value_to_http_status(value: &Value) -> Option<u16> {
    match value {
        Value::Number(number) => number
            .as_u64()
            .and_then(|code| u16::try_from(code).ok())
            .filter(|code| (400..=599).contains(code)),
        Value::String(text) => text
            .parse::<u16>()
            .ok()
            .filter(|code| (400..=599).contains(code)),
        _ => None,
    }
}

fn openai_error_type_to_http_status(error_type: &str) -> Option<u16> {
    match error_type {
        "invalid_request_error" => Some(400),
        "authentication_error" => Some(401),
        "permission_error" => Some(403),
        "not_found_error" => Some(404),
        "rate_limit_error" => Some(429),
        "timeout_error" => Some(504),
        "server_error" | "internal_server_error" => Some(500),
        _ => None,
    }
}

fn map_openai_finish_reason(reason: &str) -> &'static str {
    match reason {
        "length" => "MAX_TOKENS",
        "content_filter" => "SAFETY",
        _ => "STOP",
    }
}

fn openai_tool_calls_to_gemini_parts(value: Option<&Value>) -> Vec<Value> {
    let Some(calls) = value.and_then(Value::as_array) else {
        return Vec::new();
    };
    calls
        .iter()
        .filter_map(|call| {
            let function = call.get("function")?;
            let name = function.get("name").and_then(Value::as_str).unwrap_or("tool");
            let args = parse_tool_arguments(function.get("arguments"));
            Some(json!({"functionCall":{"name":name,"args":args}}))
        })
        .collect()
}

fn openai_legacy_function_call_to_gemini_part(value: &Value) -> Value {
    let name = value.get("name").and_then(Value::as_str).unwrap_or("tool");
    let args = parse_tool_arguments(value.get("arguments"));
    json!({"functionCall":{"name":name,"args":args}})
}

fn parse_tool_arguments(value: Option<&Value>) -> Value {
    match value {
        Some(Value::String(arguments)) => {
            serde_json::from_str(arguments).unwrap_or_else(|_| json!({"_raw": arguments}))
        }
        Some(Value::Object(_)) => value.cloned().unwrap_or_else(|| json!({})),
        _ => json!({}),
    }
}

fn openai_usage_to_gemini(value: Option<&Value>) -> Option<Value> {
    let usage = value?.as_object()?;
    let prompt = usage.get("prompt_tokens").and_then(Value::as_u64).unwrap_or(0);
    let completion = usage
        .get("completion_tokens")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let total = usage
        .get("total_tokens")
        .and_then(Value::as_u64)
        .unwrap_or(prompt + completion);
    Some(json!({
        "promptTokenCount": prompt,
        "candidatesTokenCount": completion,
        "totalTokenCount": total
    }))
}

#[derive(Debug, Clone, Default)]
pub struct GeminiStreamAccumulator {
    tool_calls: HashMap<usize, PartialToolCall>,
}

#[derive(Debug, Clone, Default)]
struct PartialToolCall {
    id: String,
    name: String,
    arguments: String,
}

impl GeminiStreamAccumulator {
    pub fn translate_openai_sse_data(&mut self, data: &str) -> Vec<String> {
        let trimmed = data.trim();
        if trimmed == "[DONE]" {
            // Gemini streamGenerateContent clients expect the stream to end after
            // the final JSON candidate chunk; OpenAI's sentinel is not a Gemini
            // JSON event, so suppress it rather than forwarding an invalid frame.
            return Vec::new();
        }

        let Ok(root) = serde_json::from_str::<Value>(trimmed) else {
            return Vec::new();
        };
        let Some(choices) = root.get("choices").and_then(Value::as_array) else {
            return Vec::new();
        };

        let mut out = Vec::new();
        for choice in choices {
            if let Some(delta) = choice.get("delta") {
                if let Some(content) = delta.get("content").and_then(Value::as_str) {
                    if !content.is_empty() {
                        out.push(json!({
                            "candidates":[{"index":0,"content":{"role":"model","parts":[{"text":content}]}}]
                        })
                        .to_string());
                    }
                }
                self.accumulate_tool_call_delta(delta.get("tool_calls"));
            }

            let finish_reason = choice.get("finish_reason").and_then(Value::as_str);
            if finish_reason.is_some() {
                if !self.tool_calls.is_empty() {
                    let parts = self.drain_tool_call_parts();
                    out.push(json!({
                        "candidates":[{"index":0,"content":{"role":"model","parts":parts},"finishReason":"STOP"}]
                    })
                    .to_string());
                } else {
                    out.push(json!({
                        "candidates":[{"index":0,"content":{"role":"model","parts":[]},"finishReason":map_openai_finish_reason(finish_reason.unwrap_or("stop"))}]
                    })
                    .to_string());
                }
            }
        }

        out
    }

    fn accumulate_tool_call_delta(&mut self, value: Option<&Value>) {
        let Some(calls) = value.and_then(Value::as_array) else {
            return;
        };
        for call in calls {
            let index = call.get("index").and_then(Value::as_u64).unwrap_or(0) as usize;
            let partial = self.tool_calls.entry(index).or_default();
            if let Some(function) = call.get("function") {
                if let Some(name) = function.get("name").and_then(Value::as_str) {
                    partial.name.push_str(name);
                }
                if let Some(arguments) = function.get("arguments").and_then(Value::as_str) {
                    partial.arguments.push_str(arguments);
                }
            }
        }
    }

    fn drain_tool_call_parts(&mut self) -> Vec<Value> {
        let mut keys: Vec<usize> = self.tool_calls.keys().copied().collect();
        keys.sort_unstable();
        keys.into_iter()
            .filter_map(|key| self.tool_calls.remove(&key))
            .map(|partial| {
                let args = if partial.arguments.trim().is_empty() {
                    json!({})
                } else {
                    let raw_arguments = partial.arguments.clone();
                    serde_json::from_str(&raw_arguments)
                        .unwrap_or_else(|_| json!({"_raw": raw_arguments}))
                };
                let name = if partial.name.is_empty() {
                    "tool".to_owned()
                } else {
                    partial.name
                };
                json!({"functionCall":{"name": name, "args": args}})
            })
            .collect()
    }
}


#[derive(Debug, Clone)]
pub struct OllamaStreamAccumulator {
    kind: OllamaResponseKind,
    tool_calls: HashMap<usize, PartialToolCall>,
}

impl OllamaStreamAccumulator {
    pub fn new(kind: OllamaResponseKind) -> Self {
        Self {
            kind,
            tool_calls: HashMap::new(),
        }
    }

    pub fn translate_openai_sse_data(&mut self, data: &str) -> Vec<String> {
        let trimmed = data.trim();
        if trimmed == "[DONE]" {
            return Vec::new();
        }

        let Ok(root) = serde_json::from_str::<Value>(trimmed) else {
            return Vec::new();
        };
        let model = root.get("model").and_then(Value::as_str).unwrap_or("local-model");
        let created_at = root
            .get("created")
            .and_then(Value::as_i64)
            .map(unix_seconds_to_rfc3339)
            .unwrap_or_else(current_rfc3339);
        let Some(choices) = root.get("choices").and_then(Value::as_array) else {
            return Vec::new();
        };

        let mut out = Vec::new();
        for choice in choices {
            if let Some(delta) = choice.get("delta") {
                if let Some(content) = delta.get("content").and_then(Value::as_str) {
                    if !content.is_empty() {
                        out.push(self.streaming_content_chunk(model, &created_at, content));
                    }
                }
                self.accumulate_tool_call_delta(delta.get("tool_calls"));
            }

            if let Some(finish_reason) = choice.get("finish_reason").and_then(Value::as_str) {
                out.push(self.streaming_done_chunk(model, &created_at, finish_reason, root.get("usage")));
            }
        }
        out
    }

    fn streaming_content_chunk(&self, model: &str, created_at: &str, content: &str) -> String {
        match self.kind {
            OllamaResponseKind::Chat => json!({
                "model": model,
                "created_at": created_at,
                "message": {"role":"assistant","content":content},
                "done": false
            })
            .to_string(),
            OllamaResponseKind::Generate | OllamaResponseKind::Tags | OllamaResponseKind::Show => json!({
                "model": model,
                "created_at": created_at,
                "response": content,
                "done": false
            })
            .to_string(),
        }
    }

    fn streaming_done_chunk(
        &mut self,
        model: &str,
        created_at: &str,
        finish_reason: &str,
        usage: Option<&Value>,
    ) -> String {
        let done_reason = ollama_done_reason_from_openai_finish(finish_reason).unwrap_or("stop");
        let mut root = Map::new();
        root.insert("model".to_owned(), Value::String(model.to_owned()));
        root.insert("created_at".to_owned(), Value::String(created_at.to_owned()));
        root.insert("done".to_owned(), Value::Bool(true));
        root.insert("done_reason".to_owned(), Value::String(done_reason.to_owned()));
        add_ollama_usage(&mut root, usage);

        match self.kind {
            OllamaResponseKind::Chat => {
                let mut message = Map::new();
                message.insert("role".to_owned(), Value::String("assistant".to_owned()));
                message.insert("content".to_owned(), Value::String(String::new()));
                if !self.tool_calls.is_empty() {
                    message.insert("tool_calls".to_owned(), Value::Array(self.drain_tool_calls()));
                }
                root.insert("message".to_owned(), Value::Object(message));
            }
            OllamaResponseKind::Generate | OllamaResponseKind::Tags | OllamaResponseKind::Show => {
                root.insert("response".to_owned(), Value::String(String::new()));
            }
        }
        Value::Object(root).to_string()
    }

    fn accumulate_tool_call_delta(&mut self, value: Option<&Value>) {
        let Some(calls) = value.and_then(Value::as_array) else {
            return;
        };
        for call in calls {
            let index = call.get("index").and_then(Value::as_u64).unwrap_or(0) as usize;
            let partial = self.tool_calls.entry(index).or_default();
            if let Some(id) = call.get("id").and_then(Value::as_str) {
                partial.id.push_str(id);
            }
            if let Some(function) = call.get("function") {
                if let Some(name) = function.get("name").and_then(Value::as_str) {
                    partial.name.push_str(name);
                }
                if let Some(arguments) = function.get("arguments").and_then(Value::as_str) {
                    partial.arguments.push_str(arguments);
                }
            }
        }
    }

    fn drain_tool_calls(&mut self) -> Vec<Value> {
        let mut keys: Vec<usize> = self.tool_calls.keys().copied().collect();
        keys.sort_unstable();
        keys.into_iter()
            .filter_map(|key| self.tool_calls.remove(&key))
            .map(|partial| {
                let args = if partial.arguments.trim().is_empty() {
                    json!({})
                } else {
                    let raw_arguments = partial.arguments.clone();
                    serde_json::from_str(&raw_arguments)
                        .unwrap_or_else(|_| json!({"_raw": raw_arguments}))
                };
                let name = if partial.name.is_empty() {
                    "tool".to_owned()
                } else {
                    partial.name
                };
                json!({"function":{"name": name, "arguments": args}})
            })
            .collect()
    }
}

pub fn rewrite_openai_responses_sse_data(data: &str, state: &ResponseRewriteState) -> String {
    rewrite_openai_responses_sse_data_with_mode(data, state, CodexNamespaceResponseMode::Flat)
}

pub fn rewrite_openai_responses_sse_data_with_mode(
    data: &str,
    state: &ResponseRewriteState,
    mode: CodexNamespaceResponseMode,
) -> String {
    let trimmed = data.trim();
    if trimmed == "[DONE]" || state.tool_name_map.is_empty() {
        return data.to_owned();
    }
    let Ok(mut root) = serde_json::from_str::<Value>(trimmed) else {
        return data.to_owned();
    };
    rewrite_responses_namespaced_tool_calls(&mut root, state, mode);
    root.to_string()
}

pub fn protocol_error_body(protocol: Protocol, status_code: u16, message: &str) -> Value {
    match protocol {
        Protocol::AnthropicMessages => json!({
            "type":"error",
            "error":{"type":"invalid_request_error","message":message}
        }),
        Protocol::Gemini => json!({
            "error":{"code":status_code,"message":message,"status":status_to_gemini(status_code)}
        }),
        Protocol::Ollama => json!({"error":message}),
        _ => json!({
            "error":{"message":message,"type":"invalid_request_error","code":status_code}
        }),
    }
}

fn status_to_gemini(status_code: u16) -> &'static str {
    match status_code {
        400 => "INVALID_ARGUMENT",
        401 => "UNAUTHENTICATED",
        403 => "PERMISSION_DENIED",
        404 => "NOT_FOUND",
        408 => "DEADLINE_EXCEEDED",
        409 => "ABORTED",
        429 => "RESOURCE_EXHAUSTED",
        500 => "INTERNAL",
        502 | 503 => "UNAVAILABLE",
        504 => "DEADLINE_EXCEEDED",
        _ => "INTERNAL",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flattens_responses_namespace_tools() {
        let input = json!({
            "tools": [
                {"type":"function","name":"exec_command","parameters":{"type":"object"}},
                {"type":"namespace","name":"multi_agent_v1","tools":[
                    {"type":"function","name":"close_agent","parameters":{"type":"object"}},
                    {"type":"function","name":"spawn_agent","description":"spawn","parameters":{"type":"object"}}
                ]}
            ]
        });
        let (out, state) = rewrite_openai_responses_request(input);
        let tools = out.get("tools").and_then(Value::as_array).unwrap();
        assert_eq!(tools.len(), 3);
        assert_eq!(tools[1].get("type").and_then(Value::as_str), Some("function"));
        assert_eq!(
            tools[1].get("name").and_then(Value::as_str),
            Some("multi_agent_v1__close_agent")
        );
        assert_eq!(
            state.tool_name_map.get("multi_agent_v1__close_agent"),
            Some(&"close_agent".to_owned())
        );
        assert_eq!(
            state
                .namespace_tool_map
                .get("multi_agent_v1__close_agent")
                .map(|tool_ref| (tool_ref.namespace.as_str(), tool_ref.name.as_str())),
            Some(("multi_agent_v1", "close_agent"))
        );
    }

    #[test]
    fn converts_future_non_function_responses_tool_types() {
        let input = json!({"tools":[{"type":"shell","description":"run shell"}]});
        let (out, _) = rewrite_openai_responses_request(input);
        let tool = &out["tools"][0];
        assert_eq!(tool["type"], "function");
        assert_eq!(tool["name"], "shell");
        assert!(tool.get("parameters").is_some());
    }

    fn test_namespace_state() -> ResponseRewriteState {
        let mut state = ResponseRewriteState::default();
        state
            .tool_name_map
            .insert("multi_agent_v1__close_agent".to_owned(), "close_agent".to_owned());
        state.namespace_tool_map.insert(
            "multi_agent_v1__close_agent".to_owned(),
            NamespacedToolName {
                namespace: "multi_agent_v1".to_owned(),
                name: "close_agent".to_owned(),
            },
        );
        state
    }

    #[test]
    fn namespace_response_mode_defaults_to_flat_and_keeps_wrapper_experimental() {
        assert_eq!(CodexNamespaceResponseMode::default(), CodexNamespaceResponseMode::Flat);
        assert_eq!(
            "flat".parse::<CodexNamespaceResponseMode>().unwrap(),
            CodexNamespaceResponseMode::Flat
        );
        assert_eq!(
            "experimental-wrapped".parse::<CodexNamespaceResponseMode>().unwrap(),
            CodexNamespaceResponseMode::ExperimentalWrapped
        );
    }

    #[test]
    fn synthetic_codex_fixture_uses_flat_unprefixing_by_default() {
        let state = test_namespace_state();
        let input: Value = serde_json::from_str(include_str!(
            "../fixtures/codex/synthetic/backend_prefixed_function_call.json"
        ))
        .unwrap();
        let expected: Value = serde_json::from_str(include_str!(
            "../fixtures/codex/synthetic/flat_unprefixed_function_call.expected.json"
        ))
        .unwrap();
        assert_eq!(rewrite_openai_responses_response(input, &state), expected);
    }

    #[test]
    fn default_mode_flat_unprefixes_namespaced_response_tool_calls() {
        let state = test_namespace_state();
        let out = rewrite_openai_responses_response(
            json!({"output":[{
                "type":"function_call",
                "id":"fc_1",
                "call_id":"call_1",
                "name":"multi_agent_v1__close_agent",
                "arguments":"{\"agent_id\":\"a1\"}"
            }]}),
            &state,
        );
        let item = &out["output"][0];
        assert_eq!(item["type"], "function_call");
        assert_eq!(item["name"], "close_agent");
        assert_eq!(item["call_id"], "call_1");
        assert_eq!(item["arguments"], "{\"agent_id\":\"a1\"}");
        assert!(item.get("namespace").is_none());
        assert!(item.get("call").is_none());
    }

    #[test]
    fn experimental_wrapped_mode_wraps_namespaced_response_tool_calls() {
        let state = test_namespace_state();
        let out = rewrite_openai_responses_response_with_mode(
            json!({"output":[{
                "type":"function_call",
                "id":"fc_1",
                "call_id":"call_1",
                "name":"multi_agent_v1__close_agent",
                "arguments":"{\"agent_id\":\"a1\"}"
            }]}),
            &state,
            CodexNamespaceResponseMode::ExperimentalWrapped,
        );
        let item = &out["output"][0];
        assert_eq!(item["type"], "namespace_call");
        assert_eq!(item["namespace"], "multi_agent_v1");
        assert_eq!(item["name"], "close_agent");
        assert_eq!(item["call_id"], "call_1");
        assert_eq!(item["arguments"], "{\"agent_id\":\"a1\"}");
        assert_eq!(item["call"]["type"], "function_call");
        assert_eq!(item["call"]["name"], "close_agent");
        assert_eq!(item["call"]["call_id"], "call_1");
        assert_eq!(item["call"]["arguments"], "{\"agent_id\":\"a1\"}");
    }

    #[test]
    fn namespace_rewrite_targets_function_calls_only() {
        let state = test_namespace_state();
        let out = rewrite_openai_responses_response(
            json!({
                "metadata":{"name":"multi_agent_v1__close_agent"},
                "output":[{"type":"function_call","name":"multi_agent_v1__close_agent"}]
            }),
            &state,
        );
        assert_eq!(out["metadata"]["name"], "multi_agent_v1__close_agent");
        assert_eq!(out["output"][0]["type"], "function_call");
        assert_eq!(out["output"][0]["name"], "close_agent");
    }

    #[test]
    fn default_mode_flat_unprefixes_namespaced_response_sse_payloads() {
        let state = test_namespace_state();
        let rewritten = rewrite_openai_responses_sse_data(
            r#"{"type":"response.output_item.added","item":{"type":"function_call","name":"multi_agent_v1__close_agent"}}"#,
            &state,
        );
        let parsed: Value = serde_json::from_str(&rewritten).unwrap();
        assert_eq!(parsed["item"]["type"], "function_call");
        assert_eq!(parsed["item"]["name"], "close_agent");
        assert!(parsed["item"].get("namespace").is_none());
    }

    #[test]
    fn experimental_wrapped_mode_wraps_namespaced_response_sse_payloads() {
        let state = test_namespace_state();
        let rewritten = rewrite_openai_responses_sse_data_with_mode(
            r#"{"type":"response.output_item.added","item":{"type":"function_call","name":"multi_agent_v1__close_agent"}}"#,
            &state,
            CodexNamespaceResponseMode::ExperimentalWrapped,
        );
        let parsed: Value = serde_json::from_str(&rewritten).unwrap();
        assert_eq!(parsed["item"]["type"], "namespace_call");
        assert_eq!(parsed["item"]["namespace"], "multi_agent_v1");
        assert_eq!(parsed["item"]["call"]["name"], "close_agent");
    }

    #[test]
    fn anthropic_schema_sanitizer_adds_missing_property_type() {
        let input = json!({
            "tools":[{"name":"NotebookEdit","input_schema":{"type":"object","properties":{"args":{"description":"verbatim"}},"x-extra":true}}]
        });
        let out = rewrite_anthropic_messages_request(input);
        let schema = &out["tools"][0]["input_schema"];
        assert_eq!(schema["type"], "object");
        assert_eq!(schema["properties"]["args"]["type"], "string");
        assert!(schema.get("x-extra").is_none());
    }

    #[test]
    fn anthropic_schema_compat_mode_moves_const_default_to_description() {
        let out = sanitize_json_schema(&json!({
            "type":"string",
            "const":"readonly",
            "default":"readonly"
        }));
        assert_eq!(out["type"], "string");
        assert!(out.get("const").is_none());
        assert!(out.get("default").is_none());
        let description = out["description"].as_str().unwrap();
        assert!(description.contains("const"));
        assert!(description.contains("default"));
    }

    #[test]
    fn schema_sanitizer_inlines_single_any_of() {
        let out = sanitize_json_schema(&json!({"anyOf":[{"type":"null"},{"type":"array","items":{"description":"x"}}]}));
        assert_eq!(out["type"], "array");
        assert_eq!(out["items"]["type"], "string");
    }

    #[test]
    fn anthropic_schema_compat_mode_omits_parser_risky_standard_constraints() {
        let input = json!({
            "type":"object",
            "properties":{
                "path":{
                    "type":"string",
                    "description":"file path",
                    "pattern":"^/",
                    "minLength":1,
                    "maxLength":4096
                },
                "limit":{
                    "type":"integer",
                    "minimum":1,
                    "maximum":100
                },
                "tags":{
                    "type":"array",
                    "items":{"type":"string","enum":["a","b"]},
                    "minItems":1,
                    "uniqueItems":true
                }
            },
            "required":["path","limit"]
        });
        let out = sanitize_json_schema(&input);
        assert!(out["properties"]["path"].get("pattern").is_none());
        assert!(out["properties"]["path"].get("minLength").is_none());
        assert!(out["properties"]["limit"].get("maximum").is_none());
        assert!(out["properties"]["tags"].get("minItems").is_none());
        assert!(out["properties"]["tags"].get("uniqueItems").is_none());
        assert!(out["properties"]["tags"]["items"].get("enum").is_none());
        assert_eq!(out["required"], json!(["path","limit"]));
        let path_description = out["properties"]["path"]["description"].as_str().unwrap();
        assert!(path_description.contains("pattern"));
        assert!(path_description.contains("minLength"));
        let item_description = out["properties"]["tags"]["items"]["description"].as_str().unwrap();
        assert!(item_description.contains("enum"));
    }

    #[test]
    fn anthropic_schema_semantic_mode_preserves_standard_constraints() {
        let input = json!({
            "type":"object",
            "properties":{
                "path":{
                    "type":"string",
                    "description":"file path",
                    "pattern":"^/",
                    "minLength":1,
                    "maxLength":4096
                },
                "limit":{
                    "type":"integer",
                    "minimum":1,
                    "maximum":100
                },
                "tags":{
                    "type":"array",
                    "items":{"type":"string","enum":["a","b"]},
                    "minItems":1,
                    "uniqueItems":true
                }
            },
            "required":["path","limit"]
        });
        let out = sanitize_json_schema_with_mode(&input, AnthropicSchemaMode::Semantic);
        assert_eq!(out["properties"]["path"]["pattern"], "^/");
        assert_eq!(out["properties"]["path"]["minLength"], 1);
        assert_eq!(out["properties"]["limit"]["maximum"], 100);
        assert_eq!(out["properties"]["tags"]["minItems"], 1);
        assert_eq!(out["properties"]["tags"]["uniqueItems"], true);
        assert_eq!(out["properties"]["tags"]["items"]["enum"][0], "a");
    }

    #[test]
    fn anthropic_schema_sanitizer_preserves_multi_branch_notes() {
        let out = sanitize_json_schema(&json!({
            "description":"value to edit",
            "anyOf":[
                {"type":"string","enum":["all"]},
                {"type":"integer","minimum":1}
            ]
        }));
        assert_eq!(out["type"], "string");
        let description = out["description"].as_str().unwrap();
        assert!(description.contains("value to edit"));
        assert!(description.contains("Original anyOf alternatives"));
        assert!(description.contains("integer"));
    }

    #[test]
    fn anthropic_schema_sanitizer_merges_all_of_objects() {
        let out = sanitize_json_schema(&json!({
            "allOf":[
                {"type":"object","properties":{"path":{"type":"string"}},"required":["path"]},
                {"type":"object","properties":{"replace":{"type":"string"}},"required":["replace"]}
            ]
        }));
        assert_eq!(out["type"], "object");
        assert_eq!(out["properties"]["path"]["type"], "string");
        assert_eq!(out["properties"]["replace"]["type"], "string");
        assert_eq!(out["required"].as_array().unwrap().len(), 2);
    }

    #[test]
    fn anthropic_response_adds_safe_defaults() {
        let out = rewrite_anthropic_messages_response(json!({
            "id":"msg_1",
            "content":[
                {"type":"text","text":"hello"},
                {"type":"tool_use","id":"toolu_1","name":"Read"}
            ]
        }));
        assert_eq!(out["type"], "message");
        assert_eq!(out["role"], "assistant");
        assert_eq!(out["stop_reason"], Value::Null);
        assert_eq!(out["usage"]["input_tokens"], 0);
        assert_eq!(out["content"][0]["cache_control"]["type"], "ephemeral");
        assert_eq!(out["content"][1]["input"], json!({}));
        assert_eq!(out["content"][1]["cache_control"]["type"], "ephemeral");
    }

    #[test]
    fn anthropic_response_error_body_is_not_rewritten() {
        let input = json!({
            "type":"error",
            "error":{
                "type":"invalid_request_error",
                "message":"bad request"
            }
        });
        let out = rewrite_anthropic_messages_response(input.clone());
        assert_eq!(out, input);
        assert!(out.get("role").is_none());
        assert!(out.get("content").is_none());
        assert!(out.get("usage").is_none());
    }

    #[test]
    fn anthropic_response_error_field_is_not_rewritten() {
        let input = json!({
            "error":{
                "type":"overloaded_error",
                "message":"backend overloaded"
            }
        });
        let out = rewrite_anthropic_messages_response(input.clone());
        assert_eq!(out, input);
        assert!(out.get("role").is_none());
        assert!(out.get("content").is_none());
    }

    #[test]
    fn anthropic_response_unrelated_json_is_not_rewritten() {
        let input = json!({"ok":true});
        let out = rewrite_anthropic_messages_response(input.clone());
        assert_eq!(out, input);
    }

    #[test]
    fn anthropic_sse_response_adds_content_block_defaults() {
        let data = r#"{"type":"content_block_start","index":0,"content_block":{"type":"text","text":"hi"}}"#;
        let rewritten = rewrite_anthropic_messages_sse_data(data);
        let parsed: Value = serde_json::from_str(&rewritten).unwrap();
        assert_eq!(parsed["content_block"]["cache_control"]["type"], "ephemeral");
    }

    #[test]
    fn anthropic_sse_error_data_is_not_rewritten() {
        let data = r#"{"type":"error","error":{"type":"invalid_request_error","message":"bad"}}"#;
        let rewritten = rewrite_anthropic_messages_sse_data(data);
        let parsed: Value = serde_json::from_str(&rewritten).unwrap();
        assert_eq!(parsed["type"], "error");
        assert_eq!(parsed["error"]["type"], "invalid_request_error");
        assert!(parsed.get("role").is_none());
        assert!(parsed.get("content").is_none());
    }

    #[test]
    fn translates_gemini_request_to_chat_completions() {
        let input = json!({
            "systemInstruction":{"parts":[{"text":"be useful"}]},
            "contents":[{"role":"user","parts":[{"text":"hello"}]}],
            "generationConfig":{"temperature":0.2,"maxOutputTokens":64,"topP":0.9},
            "tools":[{"functionDeclarations":[{"name":"shell","description":"run","parameters":{"type":"OBJECT","properties":{"cmd":{"type":"STRING"}},"required":["cmd"]}}]}]
        });
        let out = rewrite_gemini_request(input, false, "llama-local");
        assert_eq!(out["model"], "llama-local");
        assert_eq!(out["messages"][0]["role"], "system");
        assert_eq!(out["messages"][1]["role"], "user");
        assert_eq!(out["tools"][0]["type"], "function");
        assert_eq!(out["tools"][0]["function"]["parameters"]["type"], "object");
        assert_eq!(out["max_tokens"], 64);
    }

    #[test]
    fn sanitizes_backend_incompatible_request_fields_idempotently() {
        let input = json!({
            "model":"local",
            "reasoning_effort":"high",
            "thinking":{"type":"enabled"},
            "max_completion_tokens":128,
            "messages":[
                {"role":"assistant","content":"hi","reasoning_content":"hidden","reasoning":{"text":"hidden"}},
                "not an object"
            ]
        });

        let once = sanitize_backend_request(input);
        let twice = sanitize_backend_request(once.clone());

        assert_eq!(once, twice);
        assert!(once.get("reasoning_effort").is_none());
        assert!(once.get("thinking").is_none());
        assert!(once.get("max_completion_tokens").is_none());
        assert_eq!(once["max_tokens"], 128);
        assert!(once["messages"][0].get("reasoning_content").is_none());
        assert!(once["messages"][0].get("reasoning").is_none());
        assert_eq!(once["messages"][1], "not an object");
    }

    #[test]
    fn sanitizer_drops_max_completion_tokens_when_max_tokens_exists() {
        let out = sanitize_backend_request(json!({
            "max_tokens":64,
            "max_completion_tokens":128,
            "messages":null
        }));

        assert_eq!(out["max_tokens"], 64);
        assert!(out.get("max_completion_tokens").is_none());
    }

    #[test]
    fn sanitizer_leaves_non_object_inputs_unchanged() {
        assert_eq!(sanitize_backend_request(json!(["reasoning_effort"])), json!(["reasoning_effort"]));
        assert_eq!(sanitize_backend_request(Value::Null), Value::Null);
    }

    #[test]
    fn translates_openai_models_to_gemini_model_list() {
        let input = json!({
            "object":"list",
            "next_page_token":"next-1",
            "data":[
                {
                    "id":"llama-3.1-8b-instruct",
                    "object":"model",
                    "display_name":"Llama 3.1 8B Instruct",
                    "created":1710000000,
                    "meta":{
                        "description":"local llama model",
                        "input_token_limit":8192,
                        "output_token_limit":2048
                    }
                },
                {"id":"models/qwen3-32b"},
                {"id":"registry.local/team/qwen3:8b"},
                {"id":"models/org/nested/model-name"},
                {"id":""},
                {"id":"models/"},
                {"object":"model"}
            ]
        });

        let out = openai_models_to_gemini(input);
        let models = out["models"].as_array().unwrap();
        assert_eq!(models.len(), 4);
        assert_eq!(models[0]["name"], "models/llama-3.1-8b-instruct");
        assert_eq!(models[0]["displayName"], "Llama 3.1 8B Instruct");
        assert_eq!(models[0]["supportedGenerationMethods"], json!(["generateContent", "streamGenerateContent"]));
        assert_eq!(models[0]["description"], "local llama model");
        assert_eq!(models[0]["inputTokenLimit"], 8192);
        assert_eq!(models[0]["outputTokenLimit"], 2048);
        assert_eq!(models[1]["name"], "models/qwen3-32b");
        assert_eq!(models[1]["displayName"], "qwen3-32b");
        assert_eq!(models[2]["name"], "models/registry.local/team/qwen3:8b");
        assert_eq!(models[2]["displayName"], "registry.local/team/qwen3:8b");
        assert_eq!(models[3]["name"], "models/org/nested/model-name");
        assert_eq!(models[3]["displayName"], "org/nested/model-name");
        assert_eq!(out["nextPageToken"], "next-1");
    }

    #[test]
    fn gemini_model_list_handles_malformed_openai_models_response() {
        assert_eq!(openai_models_to_gemini(json!({"data":"not an array"})), json!({"models":[]}));
        assert_eq!(openai_models_to_gemini(Value::Null), json!({"models":[]}));
    }

    #[test]
    fn gemini_candidate_count_is_not_forwarded_without_multi_candidate_support() {
        let input = json!({
            "contents":[{"role":"user","parts":[{"text":"hello"}]}],
            "generationConfig":{"candidateCount":3,"temperature":0.4,"maxOutputTokens":16}
        });
        let out = rewrite_gemini_request(input, false, "llama-local");

        assert_eq!(out["temperature"], 0.4);
        assert_eq!(out["max_tokens"], 16);
        assert!(out.get("n").is_none());
    }

    #[test]
    fn gemini_function_response_references_generated_tool_call_id() {
        let input = json!({
            "contents":[
                {"role":"model","parts":[{"functionCall":{"name":"shell","args":{"cmd":"pwd"}}}]},
                {"role":"user","parts":[{"functionResponse":{"name":"shell","response":{"output":"/tmp"}}}]}
            ]
        });
        let out = rewrite_gemini_request(input, false, "llama-local");

        assert_eq!(out["messages"][0]["role"], "assistant");
        assert_eq!(out["messages"][0]["tool_calls"][0]["id"], "call_shell_0");
        assert_eq!(out["messages"][1]["role"], "tool");
        assert_eq!(out["messages"][1]["tool_call_id"], "call_shell_0");
        assert_eq!(out["messages"][1]["name"], "shell");
        assert_eq!(out["messages"][1]["content"], "{\"output\":\"/tmp\"}");
    }

    #[test]
    fn gemini_function_responses_match_repeated_tool_names_in_order() {
        let input = json!({
            "contents":[
                {"role":"model","parts":[
                    {"functionCall":{"name":"read_file","args":{"path":"a.txt"}}},
                    {"functionCall":{"name":"read_file","args":{"path":"b.txt"}}}
                ]},
                {"role":"user","parts":[
                    {"functionResponse":{"name":"read_file","response":{"content":"A"}}},
                    {"functionResponse":{"name":"read_file","response":{"content":"B"}}}
                ]}
            ]
        });
        let out = rewrite_gemini_request(input, false, "llama-local");

        assert_eq!(out["messages"][0]["tool_calls"][0]["id"], "call_read_file_0");
        assert_eq!(out["messages"][0]["tool_calls"][1]["id"], "call_read_file_1");
        assert_eq!(out["messages"][1]["tool_call_id"], "call_read_file_0");
        assert_eq!(out["messages"][2]["tool_call_id"], "call_read_file_1");
    }

    #[test]
    fn gemini_orphan_function_response_gets_synthetic_tool_call_id() {
        let input = json!({
            "contents":[
                {"role":"user","parts":[{"functionResponse":{"name":"shell","response":{"output":"/tmp"}}}]}
            ]
        });
        let out = rewrite_gemini_request(input, false, "llama-local");

        assert_eq!(out["messages"][0]["role"], "assistant");
        assert_eq!(out["messages"][0]["tool_calls"][0]["id"], "call_shell_0");
        assert_eq!(out["messages"][0]["tool_calls"][0]["function"]["name"], "shell");
        assert_eq!(out["messages"][0]["tool_calls"][0]["function"]["arguments"], "{}");
        assert_eq!(out["messages"][1]["role"], "tool");
        assert_eq!(out["messages"][1]["tool_call_id"], "call_shell_0");
        assert_ne!(out["messages"][1]["tool_call_id"], "shell");
    }

    #[test]
    fn gemini_mixed_parts_are_translated_in_part_order() {
        let input = json!({
            "contents":[
                {"role":"model","parts":[
                    {"text":"I will inspect the file."},
                    {"functionCall":{"name":"read_file","args":{"path":"a.txt"}}}
                ]},
                {"role":"user","parts":[
                    {"functionResponse":{"name":"read_file","response":{"content":"A"}}},
                    {"text":"Continue."}
                ]}
            ]
        });
        let out = rewrite_gemini_request(input, false, "llama-local");
        let messages = out["messages"].as_array().unwrap();

        assert_eq!(messages[0]["role"], "assistant");
        assert_eq!(messages[0]["content"], "I will inspect the file.");
        assert_eq!(messages[1]["role"], "assistant");
        assert_eq!(messages[1]["tool_calls"][0]["id"], "call_read_file_0");
        assert_eq!(messages[2]["role"], "tool");
        assert_eq!(messages[2]["tool_call_id"], "call_read_file_0");
        assert_eq!(messages[3]["role"], "user");
        assert_eq!(messages[3]["content"], "Continue.");
    }

    #[test]
    fn gemini_stream_suppresses_openai_done_sentinel() {
        let mut acc = GeminiStreamAccumulator::default();
        let out = acc.translate_openai_sse_data("[DONE]");
        assert!(out.is_empty());
    }

    #[test]
    fn translates_openai_response_to_gemini() {
        let input = json!({
            "choices":[{"message":{"content":"hi","tool_calls":[{"function":{"name":"read_file","arguments":"{\"path\":\"Cargo.toml\"}"}}]},"finish_reason":"tool_calls"}],
            "usage":{"prompt_tokens":3,"completion_tokens":5,"total_tokens":8}
        });
        let out = openai_chat_response_to_gemini(input, 200);
        assert_eq!(out["candidates"][0]["content"]["parts"][0]["text"], "hi");
        assert_eq!(
            out["candidates"][0]["content"]["parts"][1]["functionCall"]["name"],
            "read_file"
        );
        assert_eq!(out["usageMetadata"]["totalTokenCount"], 8);
    }

    #[test]
    fn maps_openai_error_response_to_gemini_using_http_status() {
        let input = json!({
            "error": {
                "type":"invalid_request_error",
                "message":"bad request from backend",
                "code":"ignored_when_status_is_error"
            }
        });
        let out = openai_chat_response_to_gemini(input, 400);
        assert_eq!(out["error"]["code"], 400);
        assert_eq!(out["error"]["status"], "INVALID_ARGUMENT");
        assert_eq!(out["error"]["message"], "bad request from backend");
    }

    #[test]
    fn maps_openai_error_response_to_gemini_using_numeric_error_code_when_status_is_success() {
        let input = json!({"error":{"message":"missing model","code":404}});
        let out = openai_chat_response_to_gemini(input, 200);
        assert_eq!(out["error"]["code"], 404);
        assert_eq!(out["error"]["status"], "NOT_FOUND");
    }

    #[test]
    fn maps_openai_error_type_to_gemini_when_no_http_error_status_or_numeric_code() {
        let input = json!({"error":{"type":"rate_limit_error","message":"slow down"}});
        let out = openai_chat_response_to_gemini(input, 200);
        assert_eq!(out["error"]["code"], 429);
        assert_eq!(out["error"]["status"], "RESOURCE_EXHAUSTED");
    }

    #[test]
    fn accumulates_streaming_tool_call_chunks() {
        let mut acc = GeminiStreamAccumulator::default();
        assert!(acc
            .translate_openai_sse_data(
                r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"name":"shell","arguments":"{\"cmd\":"}}]}}]}"#,
            )
            .is_empty());
        assert!(acc
            .translate_openai_sse_data(
                r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"ls\"}"}}]}}]}"#,
            )
            .is_empty());
        let out = acc.translate_openai_sse_data(
            r#"{"choices":[{"delta":{},"finish_reason":"tool_calls"}]}"#,
        );
        assert_eq!(out.len(), 1);
        let parsed: Value = serde_json::from_str(&out[0]).unwrap();
        assert_eq!(
            parsed["candidates"][0]["content"]["parts"][0]["functionCall"]["name"],
            "shell"
        );
        assert_eq!(
            parsed["candidates"][0]["content"]["parts"][0]["functionCall"]["args"]["cmd"],
            "ls"
        );
    }

    #[test]
    fn detects_protocols() {
        assert_eq!(protocol_for_path("/v1/responses"), Protocol::OpenAiResponses);
        assert_eq!(protocol_for_path("/v1/messages"), Protocol::AnthropicMessages);
        assert_eq!(protocol_for_path("/v1/chat/completions"), Protocol::OpenAiChat);
        assert_eq!(
            protocol_for_path("/v1beta/models/gemini-2.5-flash:generateContent"),
            Protocol::Gemini
        );
        assert_eq!(protocol_for_path("/v1beta/models"), Protocol::Gemini);
        assert!(is_gemini_model_list_path("/v1beta/models"));
        assert!(is_gemini_model_list_path("/gemini/v1beta/models"));
    }

    #[test]
    fn detects_gemini_requests_by_body_on_nonstandard_paths() {
        let body = br#"{
            "contents":[{"role":"user","parts":[{"text":"hello"}]}],
            "generationConfig":{"maxOutputTokens":32}
        }"#;
        assert_eq!(protocol_for_request("/custom/proxy", body), Protocol::Gemini);
    }

    #[test]
    fn does_not_body_detect_openai_chat_as_gemini() {
        let body = br#"{
            "model":"local",
            "messages":[{"role":"user","content":"hello"}]
        }"#;
        assert_eq!(protocol_for_request("/custom/proxy", body), Protocol::PassThrough);
    }

    #[test]
    fn gemini_path_model_is_informational_only() {
        let input = json!({
            "contents":[{"role":"user","parts":[{"text":"hello"}]}]
        });
        let out = rewrite_gemini_request(input, false, "configured-backend-model");
        assert_eq!(out["model"], "configured-backend-model");
    }


    #[test]
    fn detects_ollama_paths() {
        assert_eq!(protocol_for_path("/api/chat"), Protocol::Ollama);
        assert_eq!(protocol_for_path("/api/generate"), Protocol::Ollama);
        assert_eq!(protocol_for_path("/api/tags"), Protocol::Ollama);
    }

    #[test]
    fn rewrites_ollama_chat_to_openai_chat_with_tool_ids() {
        let input = json!({
            "model":"qwen3:32b",
            "messages":[
                {"role":"assistant","content":"","tool_calls":[{"function":{"name":"read_file","arguments":{"path":"Cargo.toml"}}}]},
                {"role":"tool","tool_name":"read_file","content":"contents"}
            ],
            "tools":[{"type":"function","function":{"name":"read_file","description":"read","parameters":{"type":"object"}}}],
            "options":{"temperature":0.1,"num_predict":32}
        });

        let out = rewrite_ollama_chat_request(input).unwrap();
        assert_eq!(out["model"], "qwen3:32b");
        assert_eq!(out["stream"], true);
        assert_eq!(out["temperature"], 0.1);
        assert_eq!(out["max_tokens"], 32);
        assert_eq!(out["tools"][0]["type"], "function");
        assert_eq!(out["messages"][0]["tool_calls"][0]["id"], "call_read_file_0");
        assert_eq!(out["messages"][0]["tool_calls"][0]["function"]["arguments"], "{\"path\":\"Cargo.toml\"}");
        assert_eq!(out["messages"][1]["role"], "tool");
        assert_eq!(out["messages"][1]["tool_call_id"], "call_read_file_0");
    }

    #[test]
    fn rewrites_ollama_generate_prompt_and_options() {
        let input = json!({
            "model":"codellama:code",
            "system":"You write terse code.",
            "prompt":"fn main() {}",
            "stream":false,
            "format":"json",
            "options":{"top_p":0.9,"seed":7}
        });

        let out = rewrite_ollama_generate_request(input).unwrap();
        assert_eq!(out["model"], "codellama:code");
        assert_eq!(out["stream"], false);
        assert_eq!(out["messages"][0]["role"], "system");
        assert_eq!(out["messages"][1]["role"], "user");
        assert_eq!(out["messages"][1]["content"], "fn main() {}");
        assert_eq!(out["response_format"]["type"], "json_object");
        assert_eq!(out["top_p"], 0.9);
        assert_eq!(out["seed"], 7);
    }

    #[test]
    fn translates_openai_chat_response_to_ollama_chat() {
        let input = json!({
            "model":"qwen3:32b",
            "created":1710000000,
            "choices":[{"message":{"role":"assistant","content":"hi","tool_calls":[{"function":{"name":"read_file","arguments":"{\"path\":\"Cargo.toml\"}"}}]},"finish_reason":"tool_calls"}],
            "usage":{"prompt_tokens":3,"completion_tokens":5,"total_tokens":8}
        });

        let out = openai_response_to_ollama(input, OllamaResponseKind::Chat, 200);
        assert_eq!(out["model"], "qwen3:32b");
        assert_eq!(out["message"]["role"], "assistant");
        assert_eq!(out["message"]["content"], "hi");
        assert_eq!(out["message"]["tool_calls"][0]["function"]["name"], "read_file");
        assert_eq!(out["message"]["tool_calls"][0]["function"]["arguments"]["path"], "Cargo.toml");
        assert_eq!(out["done"], true);
        assert_eq!(out["done_reason"], "stop");
        assert_eq!(out["prompt_eval_count"], 3);
        assert_eq!(out["eval_count"], 5);
    }

    #[test]
    fn translates_openai_models_to_ollama_tags() {
        let input = json!({
            "object":"list",
            "data":[{"id":"qwen3:32b","created":1710000000,"object":"model"}]
        });

        let out = openai_response_to_ollama(input, OllamaResponseKind::Tags, 200);
        assert_eq!(out["models"][0]["name"], "qwen3:32b");
        assert_eq!(out["models"][0]["model"], "qwen3:32b");
        assert_eq!(out["models"][0]["modified_at"], "2024-03-09T16:00:00Z");
        assert_eq!(out["models"][0]["details"]["format"], "gguf");
    }

    #[test]
    fn ollama_tags_without_created_use_stable_modified_at() {
        let input = json!({
            "object":"list",
            "data":[{"id":"qwen3:32b","object":"model"}]
        });

        let first = openai_response_to_ollama(input.clone(), OllamaResponseKind::Tags, 200);
        let second = openai_response_to_ollama(input, OllamaResponseKind::Tags, 200);
        assert_eq!(first, second);
        assert_eq!(first["models"][0]["modified_at"], OLLAMA_UNKNOWN_MODEL_MODIFIED_AT);
    }

    #[test]
    fn ollama_tags_use_backend_metadata_modified_at_before_stable_fallback() {
        let input = json!({
            "object":"list",
            "data":[{
                "id":"qwen3:32b",
                "object":"model",
                "meta":{"modified_at":"2025-01-02T03:04:05Z"}
            }]
        });

        let out = openai_response_to_ollama(input, OllamaResponseKind::Tags, 200);
        assert_eq!(out["models"][0]["modified_at"], "2025-01-02T03:04:05Z");
    }

    #[test]
    fn translates_backend_model_record_to_ollama_show_details() {
        let input = json!({
            "object":"list",
            "data":[{
                "id":"qwen3:32b-q4_K_M",
                "created":1710000000,
                "object":"model",
                "meta":{
                    "general.architecture":"qwen3",
                    "general.parameter_count":32500000000u64,
                    "quantization_level":"Q4_K_M",
                    "tokenizer.chat_template":"{{ .Prompt }}",
                    "license":"apache-2.0",
                    "capabilities":["completion", "tools"]
                }
            }]
        });

        let out = openai_response_to_ollama_with_context(
            input,
            OllamaResponseKind::Show,
            200,
            Some("qwen3:32b"),
            "local-model",
        );
        assert_eq!(out["modelfile"], "FROM qwen3:32b\n");
        assert_eq!(out["template"], "{{ .Prompt }}");
        assert_eq!(out["license"], "apache-2.0");
        assert_eq!(out["modified_at"], "2024-03-09T16:00:00Z");
        assert_eq!(out["details"]["family"], "qwen3");
        assert_eq!(out["details"]["parameter_size"], "32.5B");
        assert_eq!(out["details"]["quantization_level"], "Q4_K_M");
        assert_eq!(out["model_info"]["general.parameter_count"], 32500000000u64);
        assert_eq!(out["capabilities"][1], "tools");
    }


    fn observed_ollama_fixture_body(raw: &str) -> Value {
        let root: Value = serde_json::from_str(raw).unwrap();
        root.get("body").cloned().expect("observed fixture body")
    }

    #[test]
    fn observed_python_client_chat_stream_fixture_rewrites_to_openai_chat() {
        let input = observed_ollama_fixture_body(include_str!(
            "../fixtures/ollama/observed/python_0_6_2_chat_stream.request.json"
        ));
        let expected: Value = serde_json::from_str(include_str!(
            "../fixtures/ollama/observed/python_0_6_2_chat_stream.expected-chat.json"
        ))
        .unwrap();

        assert_eq!(rewrite_ollama_chat_request(input).unwrap(), expected);
    }

    #[test]
    fn observed_python_client_generate_stream_fixture_rewrites_to_openai_chat() {
        let input = observed_ollama_fixture_body(include_str!(
            "../fixtures/ollama/observed/python_0_6_2_generate_stream.request.json"
        ));
        let expected: Value = serde_json::from_str(include_str!(
            "../fixtures/ollama/observed/python_0_6_2_generate_stream.expected-chat.json"
        ))
        .unwrap();

        assert_eq!(rewrite_ollama_generate_request(input).unwrap(), expected);
    }

    #[test]
    fn observed_python_client_chat_nonstream_fixture_rewrites_to_openai_chat() {
        let input = observed_ollama_fixture_body(include_str!(
            "../fixtures/ollama/observed/python_0_6_2_chat_nonstream.request.json"
        ));
        let expected: Value = serde_json::from_str(include_str!(
            "../fixtures/ollama/observed/python_0_6_2_chat_nonstream.expected-chat.json"
        ))
        .unwrap();

        assert_eq!(rewrite_ollama_chat_request(input).unwrap(), expected);
    }

    #[test]
    fn observed_python_client_generate_nonstream_fixture_rewrites_to_openai_chat() {
        let input = observed_ollama_fixture_body(include_str!(
            "../fixtures/ollama/observed/python_0_6_2_generate_nonstream.request.json"
        ));
        let expected: Value = serde_json::from_str(include_str!(
            "../fixtures/ollama/observed/python_0_6_2_generate_nonstream.expected-chat.json"
        ))
        .unwrap();

        assert_eq!(rewrite_ollama_generate_request(input).unwrap(), expected);
    }

    #[test]
    fn observed_js_client_chat_nonstream_fixture_rewrites_to_openai_chat() {
        let input = observed_ollama_fixture_body(include_str!(
            "../fixtures/ollama/observed/js_0_6_3_chat_nonstream.request.json"
        ));
        let expected: Value = serde_json::from_str(include_str!(
            "../fixtures/ollama/observed/js_0_6_3_chat_nonstream.expected-chat.json"
        ))
        .unwrap();

        assert_eq!(rewrite_ollama_chat_request(input).unwrap(), expected);
    }

    #[test]
    fn observed_js_client_generate_nonstream_fixture_rewrites_to_openai_chat() {
        let input = observed_ollama_fixture_body(include_str!(
            "../fixtures/ollama/observed/js_0_6_3_generate_nonstream.request.json"
        ));
        let expected: Value = serde_json::from_str(include_str!(
            "../fixtures/ollama/observed/js_0_6_3_generate_nonstream.expected-chat.json"
        ))
        .unwrap();

        assert_eq!(rewrite_ollama_generate_request(input).unwrap(), expected);
    }

    #[test]
    fn observed_js_lifecycle_fixtures_accept_name_field() {
        let pull = observed_ollama_fixture_body(include_str!(
            "../fixtures/ollama/observed/js_0_6_3_pull.request.json"
        ));
        let delete = observed_ollama_fixture_body(include_str!(
            "../fixtures/ollama/observed/js_0_6_3_delete.request.json"
        ));

        assert_eq!(ollama_pull_response(&pull)["model"], "qwen3:32b");
        assert_eq!(ollama_delete_response(&delete)["model"], "qwen3:32b");
    }

    #[test]
    fn observed_js_client_chat_tools_stream_fixture_rewrites_to_openai_chat() {
        let input = observed_ollama_fixture_body(include_str!(
            "../fixtures/ollama/observed/js_0_6_3_chat_tools_stream.request.json"
        ));
        let expected: Value = serde_json::from_str(include_str!(
            "../fixtures/ollama/observed/js_0_6_3_chat_tools_stream.expected-chat.json"
        ))
        .unwrap();

        assert_eq!(rewrite_ollama_chat_request(input).unwrap(), expected);
    }

    #[test]
    fn observed_ollama_streaming_response_fixtures_are_ndjson() {
        for fixture in [
            include_str!("../fixtures/ollama/observed/python_0_6_2_chat_stream.response.ndjson"),
            include_str!("../fixtures/ollama/observed/python_0_6_2_generate_stream.response.ndjson"),
            include_str!("../fixtures/ollama/observed/js_0_6_3_chat_tools_stream.response.ndjson"),
        ] {
            let rows: Vec<Value> = fixture
                .lines()
                .filter(|line| !line.trim().is_empty())
                .map(|line| serde_json::from_str(line).unwrap())
                .collect();
            assert!(!rows.is_empty());
            assert_eq!(rows.last().unwrap()["done"], true);
        }
    }

    #[test]
    fn integration_backend_sse_tool_call_stream_matches_expected_ollama_ndjson() {
        let fixture = include_str!("../fixtures/ollama/integration/chat_tool_call_stream.backend.sse");
        let expected: Vec<Value> = include_str!(
            "../fixtures/ollama/integration/chat_tool_call_stream.expected.ndjson"
        )
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();

        let mut accumulator = OllamaStreamAccumulator::new(OllamaResponseKind::Chat);
        let mut actual = Vec::new();
        for event in fixture.split("\n\n") {
            for line in event.lines() {
                if let Some(data) = line.strip_prefix("data: ") {
                    actual.extend(
                        accumulator
                            .translate_openai_sse_data(data)
                            .into_iter()
                            .map(|line| serde_json::from_str::<Value>(&line).unwrap()),
                    );
                }
            }
        }

        assert_eq!(actual, expected);
    }

    #[test]
    fn ollama_stream_accumulator_converts_sse_payloads_to_ndjson_payloads() {
        let mut acc = OllamaStreamAccumulator::new(OllamaResponseKind::Chat);
        let chunks = acc.translate_openai_sse_data(
            r#"{"model":"qwen3:32b","created":1710000000,"choices":[{"delta":{"content":"hel"}}]}"#,
        );
        assert_eq!(chunks.len(), 1);
        let first: Value = serde_json::from_str(&chunks[0]).unwrap();
        assert_eq!(first["message"]["content"], "hel");
        assert_eq!(first["done"], false);

        let done = acc.translate_openai_sse_data(
            r#"{"model":"qwen3:32b","created":1710000000,"choices":[{"delta":{},"finish_reason":"stop"}]}"#,
        );
        assert_eq!(done.len(), 1);
        let parsed: Value = serde_json::from_str(&done[0]).unwrap();
        assert_eq!(parsed["done"], true);
        assert_eq!(parsed["done_reason"], "stop");
    }

}
