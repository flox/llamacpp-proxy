use async_stream::stream;
use axum::body::{to_bytes, Body};
use axum::extract::State;
use axum::http::header::{
    ACCEPT_ENCODING, AUTHORIZATION, CONTENT_ENCODING, CONTENT_LENGTH, CONTENT_TYPE, HOST,
};
use axum::http::{HeaderMap, HeaderName, HeaderValue, Method, Request, Response, StatusCode, Uri};
use axum::Router;
use bytes::Bytes;
use futures_util::StreamExt;
use http_body_util::{BodyExt, Full};
use hyper::body::Incoming;
use hyper_util::client::legacy::connect::HttpConnector;
use hyper_util::client::legacy::Client;
use hyper_util::rt::TokioExecutor;
use llamacpp_proxy::{
    body_declares_stream, hardcoded_gemini_classification_response,
    is_gemini_generation_path, is_gemini_model_list_path, is_gemini_request_body,
    is_gemini_stream_path, is_ollama_chat_lifecycle_request, is_ollama_chat_path, is_ollama_delete_path,
    is_ollama_generate_lifecycle_request, is_ollama_generate_path, is_ollama_pull_path,
    is_ollama_show_path, is_ollama_tags_path, ollama_delete_response,
    ollama_lifecycle_response, ollama_pull_response, ollama_request_declares_stream,
    openai_chat_response_to_gemini, openai_models_to_gemini,
    openai_response_to_ollama_with_context, parse_json_body, protocol_error_body,
    protocol_for_path, protocol_for_request, sanitize_backend_request,
    rewrite_anthropic_messages_request_with_mode, rewrite_anthropic_messages_response,
    rewrite_anthropic_messages_sse_data, rewrite_gemini_request, rewrite_ollama_chat_request,
    rewrite_ollama_generate_request, rewrite_openai_responses_request,
    rewrite_openai_responses_response_with_mode, rewrite_openai_responses_sse_data_with_mode,
    AnthropicSchemaMode, CodexNamespaceResponseMode, GeminiResponseKind, GeminiStreamAccumulator,
    OllamaResponseKind, OllamaStreamAccumulator, Protocol, ResponseRewriteState,
};
use serde_json::{json, Value};
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;
use tokio::net::TcpListener;
use tokio::time::timeout;
use tracing::{debug, info, warn};

const DEFAULT_LISTEN: &str = "127.0.0.1:8081";
const DEFAULT_BACKEND: &str = "127.0.0.1:8080";
const DEFAULT_BACKEND_API_KEY: &str = "llamacpp-local";
const DEFAULT_BACKEND_MODEL: &str = "local-model";
const DEFAULT_TIMEOUT_SECS: u64 = 120;
const DEFAULT_MAX_BODY_BYTES: usize = 64 * 1024 * 1024;

#[derive(Debug, Clone)]
struct Config {
    listen: SocketAddr,
    gemini_listen: Option<SocketAddr>,
    backend_base: String,
    backend_api_key: String,
    backend_model: String,
    backend_timeout: Duration,
    max_body_bytes: usize,
    hardcoded_gemini_classifier: bool,
    codex_namespace_response_mode: CodexNamespaceResponseMode,
    anthropic_schema_mode: AnthropicSchemaMode,
}

#[derive(Clone)]
struct AppState {
    config: Arc<Config>,
    client: Client<HttpConnector, Full<Bytes>>,
}

#[derive(Debug)]
enum ProxyError {
    BadGateway(String),
    Timeout,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let config = Arc::new(Config::parse()?);
    init_tracing();

    let mut connector = HttpConnector::new();
    connector.enforce_http(false);
    let client = Client::builder(TokioExecutor::new()).build(connector);
    let state = AppState { config, client };

    info!(listen=%state.config.listen, backend=%state.config.backend_base, "starting llamacpp-proxy");

    if let Some(gemini_listen) = state.config.gemini_listen {
        let main_state = state.clone();
        let gemini_state = state.clone();
        tokio::select! {
            result = serve(main_state.config.listen, main_state) => result?,
            result = serve(gemini_listen, gemini_state) => result?,
            _ = shutdown_signal() => info!("shutdown requested"),
        }
    } else {
        tokio::select! {
            result = serve(state.config.listen, state) => result?,
            _ = shutdown_signal() => info!("shutdown requested"),
        }
    }

    Ok(())
}

async fn serve(
    addr: SocketAddr,
    state: AppState,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let listener = TcpListener::bind(addr).await?;
    info!(%addr, "listening");
    let app = Router::new().fallback(proxy_handler).with_state(state);
    axum::serve(listener, app).await?;
    Ok(())
}

async fn shutdown_signal() {
    if let Err(err) = tokio::signal::ctrl_c().await {
        warn!(%err, "failed to listen for ctrl-c");
    }
}

async fn proxy_handler(State(state): State<AppState>, req: Request<Body>) -> Response<Body> {
    let method = req.method().clone();
    let uri = req.uri().clone();
    let path = uri.path().to_owned();
    let path_and_query = uri
        .path_and_query()
        .map(|value| value.as_str().to_owned())
        .unwrap_or_else(|| path.clone());
    let inbound_headers = req.headers().clone();

    if method == Method::GET && path == "/health" {
        return health_response(&state).await;
    }

    // Ollama compatibility: health probe and version check
    if method == Method::GET && path == "/" {
        return text_response(StatusCode::OK, "Ollama is running");
    }
    if method == Method::GET && path == "/api/version" {
        return json_response(StatusCode::OK, json!({"version": "0.15.0"}));
    }

    let path_protocol = protocol_for_path(&path);
    let body = match to_bytes(req.into_body(), state.config.max_body_bytes).await {
        Ok(body) => body,
        Err(err) => {
            return json_response(
                StatusCode::PAYLOAD_TOO_LARGE,
                protocol_error_body(path_protocol, 413, &format!("request body too large: {err}")),
            );
        }
    };

    let protocol = protocol_for_request(&path, &body);

    let original_path_and_query = path_and_query.clone();
    let mut target_path = path_and_query;
    let target_method;
    let stream_response;
    let response_rewrite_state;
    let response_protocol;
    let outgoing_body = match translate_request(&state, &method, protocol, &path, &target_path, &body) {
        Ok(TranslatedRequest::Forward {
            target,
            body,
            stream,
            state,
            response_protocol: translated_response_protocol,
        }) => {
            target_path = target;
            stream_response = stream;
            response_rewrite_state = state;
            response_protocol = translated_response_protocol;
            target_method = translated_request_method(&method, response_protocol, &response_rewrite_state);
            body
        }
        Ok(TranslatedRequest::Immediate { status, body }) => return json_response(status, body),
        Ok(TranslatedRequest::ImmediateRaw {
            status,
            body,
            content_type,
        }) => return bytes_response(status, body, content_type),
        Err(err) => {
            log_translation_failure(&method, &path, protocol, &err, &body);
            let fallback = translation_failure_passthrough(&original_path_and_query, &body);
            target_path = fallback.target;
            stream_response = fallback.stream;
            response_rewrite_state = fallback.state;
            response_protocol = fallback.response_protocol;
            target_method = method.clone();
            fallback.body
        }
    };

    match forward_request(&state, target_method, &target_path, &inbound_headers, outgoing_body).await {
        Ok(upstream) => {
            translate_response(
                response_protocol,
                stream_response,
                response_rewrite_state,
                state.config.codex_namespace_response_mode,
                state.config.backend_model.clone(),
                upstream,
            )
            .await
        }
        Err(ProxyError::Timeout) => json_response(
            StatusCode::GATEWAY_TIMEOUT,
            protocol_error_body(protocol, 504, "backend request timed out"),
        ),
        Err(ProxyError::BadGateway(message)) => json_response(
            StatusCode::BAD_GATEWAY,
            protocol_error_body(protocol, 502, &message),
        ),
    }
}

fn translated_request_method(
    original: &Method,
    response_protocol: Protocol,
    state: &ResponseRewriteState,
) -> Method {
    if (response_protocol == Protocol::Ollama
        && state.ollama_response_kind == Some(OllamaResponseKind::Show))
        || (response_protocol == Protocol::Gemini
            && *original == Method::GET
            && state.gemini_response_kind == Some(GeminiResponseKind::ListModels))
    {
        Method::GET
    } else {
        original.clone()
    }
}

#[derive(Debug)]
enum TranslatedRequest {
    Forward {
        target: String,
        body: Bytes,
        stream: bool,
        state: ResponseRewriteState,
        response_protocol: Protocol,
    },
    Immediate {
        status: StatusCode,
        body: Value,
    },
    ImmediateRaw {
        status: StatusCode,
        body: Bytes,
        content_type: HeaderValue,
    },
}

fn log_translation_failure(
    method: &Method,
    path: &str,
    protocol: Protocol,
    err: &llamacpp_proxy::RewriteError,
    body: &Bytes,
) {
    let raw_request_body = String::from_utf8_lossy(body);
    warn!(
        %method,
        path = %path,
        ?protocol,
        error = %err,
        body_bytes = body.len(),
        raw_request_body = %raw_request_body,
        "request translation failed; forwarding original request with JSON backend sanitization when possible"
    );
}

#[derive(Debug)]
struct PassthroughFallback {
    target: String,
    body: Bytes,
    stream: bool,
    state: ResponseRewriteState,
    response_protocol: Protocol,
}

fn translation_failure_passthrough(
    original_path_and_query: &str,
    body: &Bytes,
) -> PassthroughFallback {
    PassthroughFallback {
        target: original_path_and_query.to_owned(),
        body: backend_compatible_body_bytes(body),
        stream: false,
        state: ResponseRewriteState::default(),
        response_protocol: Protocol::PassThrough,
    }
}

fn translate_request(
    state: &AppState,
    method: &Method,
    protocol: Protocol,
    path: &str,
    path_and_query: &str,
    body: &Bytes,
) -> Result<TranslatedRequest, llamacpp_proxy::RewriteError> {
    match protocol {
        Protocol::OpenAiResponses => {
            let parsed = match parse_json_body(body) {
                Ok(value) => value,
                Err(err) => return malformed_json_request(Protocol::OpenAiResponses, err),
            };
            ensure_openai_responses_request_shape(&parsed)?;
            let stream = body_declares_stream(&parsed);
            let (rewritten, rewrite_state) = rewrite_openai_responses_request(parsed);
            Ok(TranslatedRequest::Forward {
                target: path_and_query.to_owned(),
                body: backend_json_bytes(rewritten),
                stream,
                state: rewrite_state,
                response_protocol: Protocol::OpenAiResponses,
            })
        }
        Protocol::AnthropicMessages => {
            let parsed = match parse_json_body(body) {
                Ok(value) => value,
                Err(err) => return malformed_json_request(Protocol::AnthropicMessages, err),
            };
            ensure_anthropic_messages_request_shape(&parsed)?;
            let stream = body_declares_stream(&parsed);
            let rewritten = rewrite_anthropic_messages_request_with_mode(
                parsed,
                state.config.anthropic_schema_mode,
            );
            Ok(TranslatedRequest::Forward {
                target: path_and_query.to_owned(),
                body: backend_json_bytes(rewritten),
                stream,
                state: ResponseRewriteState::default(),
                response_protocol: Protocol::AnthropicMessages,
            })
        }
        Protocol::Gemini => {
            if is_gemini_model_list_path(path) {
                if *method == Method::GET {
                    return Ok(TranslatedRequest::Forward {
                        target: "/v1/models".to_owned(),
                        body: Bytes::new(),
                        stream: false,
                        state: gemini_response_state(GeminiResponseKind::ListModels),
                        response_protocol: Protocol::Gemini,
                    });
                }

                return Ok(TranslatedRequest::Forward {
                    target: path_and_query.to_owned(),
                    body: backend_compatible_body_bytes(body),
                    stream: false,
                    state: ResponseRewriteState::default(),
                    response_protocol: Protocol::PassThrough,
                });
            }

            let path_is_generation = is_gemini_generation_path(path);
            let parsed = match parse_json_body(body) {
                Ok(value) => value,
                Err(_) if !path_is_generation => {
                    return Ok(TranslatedRequest::Forward {
                        target: path_and_query.to_owned(),
                        body: body.clone(),
                        stream: false,
                        state: ResponseRewriteState::default(),
                        response_protocol: Protocol::PassThrough,
                    });
                }
                Err(err) => return malformed_json_request(Protocol::Gemini, err),
            };

            if !is_gemini_request_body(&parsed) {
                if path_is_generation {
                    return Err(llamacpp_proxy::RewriteError::new(
                        "well-formed JSON does not match Gemini generateContent request shape",
                    ));
                }
                return Ok(TranslatedRequest::Forward {
                    target: path_and_query.to_owned(),
                    body: backend_compatible_json_bytes(body, parsed),
                    stream: false,
                    state: ResponseRewriteState::default(),
                    response_protocol: Protocol::PassThrough,
                });
            }

            if state.config.hardcoded_gemini_classifier
                && path_is_generation
                && llamacpp_proxy::is_probable_gemini_classification_request(path, &parsed)
            {
                return Ok(TranslatedRequest::Immediate {
                    status: StatusCode::OK,
                    body: hardcoded_gemini_classification_response(),
                });
            }
            let stream = is_gemini_stream_path(path) || body_declares_stream(&parsed);
            let rewritten = rewrite_gemini_request(parsed, stream, &state.config.backend_model);
            Ok(TranslatedRequest::Forward {
                target: "/v1/chat/completions".to_owned(),
                body: backend_json_bytes(rewritten),
                stream,
                state: gemini_response_state(GeminiResponseKind::GenerateContent),
                response_protocol: Protocol::Gemini,
            })
        }
        Protocol::Ollama => translate_ollama_request(state, path, body),
        Protocol::OpenAiChat => {
            let parsed = parse_json_body(body).ok();
            let stream = parsed.as_ref().is_some_and(body_declares_stream);
            let body = parsed.map(backend_json_bytes).unwrap_or_else(|| body.clone());
            Ok(TranslatedRequest::Forward {
                target: path_and_query.to_owned(),
                body,
                stream,
                state: ResponseRewriteState::default(),
                response_protocol: Protocol::OpenAiChat,
            })
        }
        Protocol::PassThrough => Ok(TranslatedRequest::Forward {
            target: path_and_query.to_owned(),
            body: backend_compatible_body_bytes(body),
            stream: false,
            state: ResponseRewriteState::default(),
            response_protocol: Protocol::PassThrough,
        }),
    }
}


fn translate_ollama_request(
    state: &AppState,
    path: &str,
    body: &Bytes,
) -> Result<TranslatedRequest, llamacpp_proxy::RewriteError> {
    if is_ollama_tags_path(path) {
        return Ok(TranslatedRequest::Forward {
            target: "/v1/models".to_owned(),
            body: Bytes::new(),
            stream: false,
            state: ollama_response_state(OllamaResponseKind::Tags),
            response_protocol: Protocol::Ollama,
        });
    }

    let parsed = match parse_json_body(body) {
        Ok(value) => value,
        Err(err) => return malformed_json_request(Protocol::Ollama, err),
    };

    if is_ollama_show_path(path) {
        return Ok(TranslatedRequest::Forward {
            target: "/v1/models".to_owned(),
            body: Bytes::new(),
            stream: false,
            state: ollama_show_response_state(&parsed, &state.config.backend_model),
            response_protocol: Protocol::Ollama,
        });
    }

    if is_ollama_pull_path(path) {
        let response = ollama_pull_response(&parsed);
        return Ok(synthetic_ollama_status_response(&parsed, response));
    }

    if is_ollama_delete_path(path) {
        return Ok(TranslatedRequest::Immediate {
            status: StatusCode::OK,
            body: ollama_delete_response(&parsed),
        });
    }

    if is_ollama_chat_path(path) {
        if is_ollama_chat_lifecycle_request(&parsed) {
            return Ok(TranslatedRequest::Immediate {
                status: StatusCode::OK,
                body: ollama_lifecycle_response(
                    &parsed,
                    OllamaResponseKind::Chat,
                    &state.config.backend_model,
                ),
            });
        }
        let stream = ollama_request_declares_stream(&parsed);
        let rewritten = rewrite_ollama_chat_request(parsed)?;
        return Ok(TranslatedRequest::Forward {
            target: "/v1/chat/completions".to_owned(),
            body: backend_json_bytes(rewritten),
            stream,
            state: ollama_response_state(OllamaResponseKind::Chat),
            response_protocol: Protocol::Ollama,
        });
    }

    if is_ollama_generate_path(path) {
        if is_ollama_generate_lifecycle_request(&parsed) {
            return Ok(TranslatedRequest::Immediate {
                status: StatusCode::OK,
                body: ollama_lifecycle_response(
                    &parsed,
                    OllamaResponseKind::Generate,
                    &state.config.backend_model,
                ),
            });
        }
        let stream = ollama_request_declares_stream(&parsed);
        let rewritten = rewrite_ollama_generate_request(parsed)?;
        return Ok(TranslatedRequest::Forward {
            target: "/v1/chat/completions".to_owned(),
            body: backend_json_bytes(rewritten),
            stream,
            state: ollama_response_state(OllamaResponseKind::Generate),
            response_protocol: Protocol::Ollama,
        });
    }

    Ok(TranslatedRequest::Forward {
        target: path.to_owned(),
        body: backend_compatible_json_bytes(body, parsed),
        stream: false,
        state: ResponseRewriteState::default(),
        response_protocol: Protocol::PassThrough,
    })
}

fn ollama_response_state(kind: OllamaResponseKind) -> ResponseRewriteState {
    ResponseRewriteState {
        ollama_response_kind: Some(kind),
        ..ResponseRewriteState::default()
    }
}

fn gemini_response_state(kind: GeminiResponseKind) -> ResponseRewriteState {
    ResponseRewriteState {
        gemini_response_kind: Some(kind),
        ..ResponseRewriteState::default()
    }
}

fn ollama_show_response_state(request: &Value, backend_model: &str) -> ResponseRewriteState {
    let mut state = ollama_response_state(OllamaResponseKind::Show);
    state.ollama_requested_model = Some(
        request
            .get("model")
            .and_then(Value::as_str)
            .filter(|value| !value.is_empty())
            .unwrap_or(backend_model)
            .to_owned(),
    );
    state
}

fn synthetic_ollama_status_response(request: &Value, response: Value) -> TranslatedRequest {
    if ollama_request_declares_stream(request) {
        let mut body = response.to_string();
        body.push('\n');
        TranslatedRequest::ImmediateRaw {
            status: StatusCode::OK,
            body: Bytes::from(body),
            content_type: HeaderValue::from_static("application/x-ndjson"),
        }
    } else {
        TranslatedRequest::Immediate {
            status: StatusCode::OK,
            body: response,
        }
    }
}

fn malformed_json_request(
    protocol: Protocol,
    err: llamacpp_proxy::RewriteError,
) -> Result<TranslatedRequest, llamacpp_proxy::RewriteError> {
    Ok(TranslatedRequest::Immediate {
        status: StatusCode::BAD_REQUEST,
        body: protocol_error_body(protocol, 400, &err.to_string()),
    })
}

fn ensure_openai_responses_request_shape(value: &Value) -> Result<(), llamacpp_proxy::RewriteError> {
    ensure_object_with_any_key(
        value,
        &[
            "input",
            "model",
            "tools",
            "instructions",
            "stream",
            "temperature",
            "max_output_tokens",
            "previous_response_id",
            "metadata",
        ],
        "well-formed JSON does not match OpenAI Responses request shape",
    )
}

fn ensure_anthropic_messages_request_shape(value: &Value) -> Result<(), llamacpp_proxy::RewriteError> {
    ensure_object_with_any_key(
        value,
        &[
            "messages",
            "model",
            "tools",
            "system",
            "max_tokens",
            "stream",
            "temperature",
            "top_p",
            "metadata",
            "thinking",
        ],
        "well-formed JSON does not match Anthropic Messages request shape",
    )
}

fn ensure_object_with_any_key(
    value: &Value,
    keys: &[&str],
    message: &str,
) -> Result<(), llamacpp_proxy::RewriteError> {
    let Some(obj) = value.as_object() else {
        return Err(llamacpp_proxy::RewriteError::new(
            "well-formed JSON request root is not an object",
        ));
    };
    if keys.iter().any(|key| obj.contains_key(*key)) {
        Ok(())
    } else {
        Err(llamacpp_proxy::RewriteError::new(message))
    }
}

fn json_bytes(value: &Value) -> Bytes {
    Bytes::from(serde_json::to_vec(value).expect("JSON serialization is infallible for Value"))
}

fn backend_json_bytes(value: Value) -> Bytes {
    json_bytes(&sanitize_backend_request(value))
}

fn backend_compatible_json_bytes(original_body: &Bytes, value: Value) -> Bytes {
    let sanitized = sanitize_backend_request(value.clone());
    if sanitized == value {
        original_body.clone()
    } else {
        json_bytes(&sanitized)
    }
}

fn backend_compatible_body_bytes(body: &Bytes) -> Bytes {
    parse_json_body(body)
        .map(|value| backend_compatible_json_bytes(body, value))
        .unwrap_or_else(|_| body.clone())
}

async fn forward_request(
    state: &AppState,
    method: Method,
    target_path: &str,
    inbound_headers: &HeaderMap,
    body: Bytes,
) -> Result<Response<Incoming>, ProxyError> {
    let uri = backend_uri(&state.config.backend_base, target_path)?;
    let mut builder = Request::builder().method(method).uri(uri);

    for (name, value) in inbound_headers.iter() {
        if should_forward_request_header(name) {
            builder = builder.header(name.clone(), value.clone());
        }
    }

    if !state.config.backend_api_key.is_empty() {
        let auth = HeaderValue::from_str(&format!("Bearer {}", state.config.backend_api_key))
            .map_err(|err| ProxyError::BadGateway(format!("invalid backend api key header: {err}")))?;
        builder = builder.header(AUTHORIZATION, auth);
    }
    // Response translators consume upstream bodies as JSON or SSE bytes. Ask the
    // backend for identity encoding so compressed payloads do not reach those
    // translators undecoded.
    builder = builder.header(ACCEPT_ENCODING, "identity");
    if !body.is_empty() && !inbound_headers.contains_key(CONTENT_TYPE) {
        builder = builder.header(CONTENT_TYPE, "application/json");
    }

    let request = builder
        .body(Full::new(body))
        .map_err(|err| ProxyError::BadGateway(format!("failed to build backend request: {err}")))?;

    match timeout(state.config.backend_timeout, state.client.request(request)).await {
        Ok(Ok(response)) => Ok(response),
        Ok(Err(err)) => Err(ProxyError::BadGateway(format!("backend request failed: {err}"))),
        Err(_) => Err(ProxyError::Timeout),
    }
}

fn backend_uri(backend_base: &str, target_path: &str) -> Result<Uri, ProxyError> {
    let separator = if target_path.starts_with('/') { "" } else { "/" };
    format!("{backend_base}{separator}{target_path}")
        .parse::<Uri>()
        .map_err(|err| ProxyError::BadGateway(format!("invalid backend URI: {err}")))
}

fn should_forward_request_header(name: &HeaderName) -> bool {
    !is_hop_by_hop(name)
        && *name != HOST
        && *name != CONTENT_LENGTH
        && *name != AUTHORIZATION
        && *name != ACCEPT_ENCODING
}

fn should_forward_response_header(name: &HeaderName) -> bool {
    !is_hop_by_hop(name) && *name != CONTENT_LENGTH
}

fn is_hop_by_hop(name: &HeaderName) -> bool {
    matches!(
        name.as_str(),
        "connection"
            | "keep-alive"
            | "proxy-authenticate"
            | "proxy-authorization"
            | "te"
            | "trailer"
            | "transfer-encoding"
            | "upgrade"
    )
}

async fn translate_response(
    protocol: Protocol,
    stream_response: bool,
    state: ResponseRewriteState,
    codex_namespace_response_mode: CodexNamespaceResponseMode,
    backend_model: String,
    upstream: Response<Incoming>,
) -> Response<Body> {
    if response_body_is_rewritten(protocol, &state) {
        if let Some(encoding) = unsupported_content_encoding(upstream.headers()) {
            return compressed_backend_response_error(
                protocol,
                stream_response,
                upstream.status(),
                encoding,
            );
        }
    }

    if stream_response {
        if protocol == Protocol::Ollama
            && should_buffer_ollama_stream_response(upstream.status(), upstream.headers())
        {
            return buffered_ollama_stream_response(state, backend_model, upstream).await;
        }
        return streaming_response(protocol, state, codex_namespace_response_mode, upstream);
    }

    let status = upstream.status();
    let headers = upstream.headers().clone();
    let body = match upstream.into_body().collect().await {
        Ok(collected) => collected.to_bytes(),
        Err(err) => {
            return json_response(
                StatusCode::BAD_GATEWAY,
                protocol_error_body(protocol, 502, &format!("failed to read backend response: {err}")),
            );
        }
    };

    let maybe_translated = match protocol {
        Protocol::OpenAiResponses if !state.tool_name_map.is_empty() => {
            serde_json::from_slice::<Value>(&body)
                .map(|value| {
                    rewrite_openai_responses_response_with_mode(
                        value,
                        &state,
                        codex_namespace_response_mode,
                    )
                })
        }
        Protocol::Gemini if state.gemini_response_kind == Some(GeminiResponseKind::ListModels) => {
            serde_json::from_slice::<Value>(&body).map(|value| {
                if status.is_success() {
                    openai_models_to_gemini(value)
                } else {
                    openai_chat_response_to_gemini(value, status.as_u16())
                }
            })
        }
        Protocol::Gemini => serde_json::from_slice::<Value>(&body)
            .map(|value| openai_chat_response_to_gemini(value, status.as_u16())),
        Protocol::Ollama => serde_json::from_slice::<Value>(&body).map(|value| {
            openai_response_to_ollama_with_context(
                value,
                state.ollama_response_kind.unwrap_or(OllamaResponseKind::Chat),
                status.as_u16(),
                state.ollama_requested_model.as_deref(),
                &backend_model,
            )
        }),
        Protocol::AnthropicMessages => serde_json::from_slice::<Value>(&body).map(rewrite_anthropic_messages_response),
        _ => return response_from_parts(status, &headers, Body::from(body)),
    };

    match maybe_translated {
        Ok(value) => response_from_parts(status, &headers, Body::from(json_bytes(&value))),
        Err(err) => {
            warn!(%err, ?protocol, "failed to translate response; returning original backend body");
            response_from_parts(status, &headers, Body::from(body))
        }
    }
}


fn response_body_is_rewritten(protocol: Protocol, state: &ResponseRewriteState) -> bool {
    match protocol {
        Protocol::Gemini | Protocol::Ollama | Protocol::AnthropicMessages => true,
        Protocol::OpenAiResponses => !state.tool_name_map.is_empty(),
        Protocol::OpenAiChat | Protocol::PassThrough => false,
    }
}

fn unsupported_content_encoding(headers: &HeaderMap) -> Option<String> {
    headers
        .get(CONTENT_ENCODING)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| {
            let trimmed = value.trim();
            let unsupported = !trimmed.is_empty()
                && !trimmed.eq_ignore_ascii_case("identity")
                && !trimmed.eq_ignore_ascii_case("none");
            unsupported.then(|| trimmed.to_owned())
        })
}

fn compressed_backend_response_error(
    protocol: Protocol,
    stream_response: bool,
    status: StatusCode,
    encoding: String,
) -> Response<Body> {
    let message = format!(
        "backend returned Content-Encoding {encoding:?} for a response that must be translated; \
         the proxy requests identity encoding and cannot safely translate compressed upstream bodies"
    );
    let out_status = if status.is_success() {
        StatusCode::BAD_GATEWAY
    } else {
        status
    };
    if protocol == Protocol::Ollama && stream_response {
        return ndjson_response_from_parts(
            out_status,
            &HeaderMap::new(),
            protocol_error_body(Protocol::Ollama, out_status.as_u16(), &message),
        );
    }
    json_response(out_status, protocol_error_body(protocol, out_status.as_u16(), &message))
}


fn should_buffer_ollama_stream_response(status: StatusCode, headers: &HeaderMap) -> bool {
    if !status.is_success() {
        return true;
    }

    !content_type_is_event_stream(headers)
}

fn content_type_is_event_stream(headers: &HeaderMap) -> bool {
    headers
        .get(CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .map(|value| {
            value
                .split(';')
                .next()
                .map(str::trim)
                .is_some_and(|media_type| media_type.eq_ignore_ascii_case("text/event-stream"))
        })
        .unwrap_or(false)
}

async fn buffered_ollama_stream_response(
    state: ResponseRewriteState,
    backend_model: String,
    upstream: Response<Incoming>,
) -> Response<Body> {
    let status = upstream.status();
    let headers = upstream.headers().clone();
    let body = match upstream.into_body().collect().await {
        Ok(collected) => collected.to_bytes(),
        Err(err) => {
            let empty_headers = HeaderMap::new();
            return ndjson_response_from_parts(
                StatusCode::BAD_GATEWAY,
                &empty_headers,
                protocol_error_body(
                    Protocol::Ollama,
                    502,
                    &format!("failed to read backend response: {err}"),
                ),
            );
        }
    };

    let (out_status, value) = buffered_ollama_stream_value(status, &body, &state, &backend_model);

    ndjson_response_from_parts(out_status, &headers, value)
}

fn buffered_ollama_stream_value(
    status: StatusCode,
    body: &Bytes,
    state: &ResponseRewriteState,
    backend_model: &str,
) -> (StatusCode, Value) {
    if status.is_success() {
        match serde_json::from_slice::<Value>(body) {
            Ok(root) => {
                if let Some(message) = backend_json_error_message(&root) {
                    (
                        StatusCode::BAD_GATEWAY,
                        protocol_error_body(
                            Protocol::Ollama,
                            502,
                            &format!("backend returned error: {message}"),
                        ),
                    )
                } else {
                    (
                        status,
                        openai_response_to_ollama_with_context(
                            root,
                            state.ollama_response_kind.unwrap_or(OllamaResponseKind::Chat),
                            status.as_u16(),
                            state.ollama_requested_model.as_deref(),
                            backend_model,
                        ),
                    )
                }
            }
            Err(_) => (
                StatusCode::BAD_GATEWAY,
                protocol_error_body(
                    Protocol::Ollama,
                    502,
                    &format!(
                        "backend returned a non-SSE response for a streaming request: {}",
                        body_preview(body)
                    ),
                ),
            ),
        }
    } else {
        (status, backend_error_body_to_ollama(status, body))
    }
}

fn backend_error_body_to_ollama(status: StatusCode, body: &Bytes) -> Value {
    let message = serde_json::from_slice::<Value>(body)
        .ok()
        .and_then(|value| backend_json_error_message(&value))
        .unwrap_or_else(|| body_preview(body));

    let message = if message.is_empty() {
        format!("backend returned HTTP {} without a response body", status.as_u16())
    } else {
        format!("backend returned HTTP {}: {message}", status.as_u16())
    };

    protocol_error_body(Protocol::Ollama, status.as_u16(), &message)
}

fn backend_json_error_message(root: &Value) -> Option<String> {
    root.get("error")
        .and_then(|error| match error {
            Value::String(message) => Some(message.clone()),
            Value::Object(object) => object
                .get("message")
                .or_else(|| object.get("detail"))
                .or_else(|| object.get("error"))
                .and_then(Value::as_str)
                .map(str::to_owned),
            _ => None,
        })
        .or_else(|| root.get("message").and_then(Value::as_str).map(str::to_owned))
}

fn body_preview(body: &Bytes) -> String {
    const MAX_PREVIEW_CHARS: usize = 2048;
    let text = String::from_utf8_lossy(body);
    let trimmed = text.trim();
    if trimmed.chars().count() <= MAX_PREVIEW_CHARS {
        return trimmed.to_owned();
    }

    let mut preview: String = trimmed.chars().take(MAX_PREVIEW_CHARS).collect();
    preview.push_str("...");
    preview
}

fn ndjson_response_from_parts(
    status: StatusCode,
    headers: &HeaderMap,
    value: Value,
) -> Response<Body> {
    let mut line = value.to_string();
    line.push('\n');
    let mut response = response_from_parts(status, headers, Body::from(line));
    response.headers_mut().insert(
        CONTENT_TYPE,
        HeaderValue::from_static("application/x-ndjson"),
    );
    response
}

fn ollama_backend_error_ndjson_line_from_str(value: &str) -> Option<String> {
    let root = serde_json::from_str::<Value>(value.trim()).ok()?;
    let message = backend_json_error_message(&root)?;
    Some(ollama_stream_protocol_error_ndjson_line(&message))
}

fn ollama_stream_protocol_error_ndjson_line(message: &str) -> String {
    let mut line = protocol_error_body(Protocol::Ollama, 502, message).to_string();
    line.push('\n');
    line
}

fn streaming_response(
    protocol: Protocol,
    state: ResponseRewriteState,
    codex_namespace_response_mode: CodexNamespaceResponseMode,
    upstream: Response<Incoming>,
) -> Response<Body> {
    let status = upstream.status();
    let headers = upstream.headers().clone();
    let incoming = upstream.into_body();

    let body = match protocol {
        Protocol::Gemini => Body::from_stream(transform_gemini_sse(incoming)),
        Protocol::Ollama => Body::from_stream(transform_ollama_sse(
            incoming,
            state.ollama_response_kind.unwrap_or(OllamaResponseKind::Chat),
        )),
        Protocol::AnthropicMessages => Body::from_stream(transform_anthropic_sse(incoming)),
        Protocol::OpenAiResponses if !state.tool_name_map.is_empty() => {
            Body::from_stream(transform_responses_sse(
                incoming,
                state,
                codex_namespace_response_mode,
            ))
        }
        _ => Body::from_stream(incoming.into_data_stream()),
    };

    let mut response = response_from_parts(status, &headers, body);
    if protocol == Protocol::Ollama {
        response.headers_mut().insert(
            CONTENT_TYPE,
            HeaderValue::from_static("application/x-ndjson"),
        );
    }
    response
}

fn transform_gemini_sse(
    incoming: Incoming,
) -> impl futures_util::Stream<Item = Result<Bytes, std::io::Error>> {
    stream! {
        let mut data_stream = incoming.into_data_stream();
        let mut buffer = String::new();
        let mut accumulator = GeminiStreamAccumulator::default();

        while let Some(next) = data_stream.next().await {
            let bytes = match next {
                Ok(bytes) => bytes,
                Err(err) => {
                    yield Err(std::io::Error::other(err));
                    continue;
                }
            };
            buffer.push_str(&String::from_utf8_lossy(&bytes));
            while let Some((event, boundary_len)) = take_next_sse_event(&mut buffer) {
                for payload in translate_gemini_sse_event(&event, &mut accumulator) {
                    yield Ok(Bytes::from(payload));
                }
                debug!(boundary_len, "translated gemini SSE event");
            }
        }

        if !buffer.is_empty() {
            for payload in translate_gemini_sse_event(&buffer, &mut accumulator) {
                yield Ok(Bytes::from(payload));
            }
        }
    }
}


fn transform_ollama_sse(
    incoming: Incoming,
    kind: OllamaResponseKind,
) -> impl futures_util::Stream<Item = Result<Bytes, std::io::Error>> {
    stream! {
        let mut data_stream = incoming.into_data_stream();
        let mut buffer = Vec::new();
        let mut accumulator = OllamaStreamAccumulator::new(kind);

        while let Some(next) = data_stream.next().await {
            let bytes = match next {
                Ok(bytes) => bytes,
                Err(err) => {
                    yield Err(std::io::Error::other(err));
                    continue;
                }
            };
            buffer.extend_from_slice(&bytes);
            while let Some(event) = take_next_sse_event_bytes(&mut buffer) {
                let event = match event {
                    Ok(event) => event,
                    Err(err) => {
                        yield Ok(Bytes::from(ollama_stream_protocol_error_ndjson_line(&err)));
                        continue;
                    }
                };
                for payload in translate_ollama_sse_event(&event, &mut accumulator) {
                    yield Ok(Bytes::from(payload));
                }
            }
        }

        if !buffer.is_empty() {
            match String::from_utf8(std::mem::take(&mut buffer)) {
                Ok(event) => {
                    for payload in translate_ollama_sse_event(&event, &mut accumulator) {
                        yield Ok(Bytes::from(payload));
                    }
                }
                Err(err) => {
                    yield Ok(Bytes::from(ollama_stream_protocol_error_ndjson_line(&format!(
                        "backend SSE tail was not valid UTF-8: {err}"
                    ))));
                }
            }
        }
    }
}

fn transform_anthropic_sse(
    incoming: Incoming,
) -> impl futures_util::Stream<Item = Result<Bytes, std::io::Error>> {
    stream! {
        let mut data_stream = incoming.into_data_stream();
        let mut buffer = String::new();

        while let Some(next) = data_stream.next().await {
            let bytes = match next {
                Ok(bytes) => bytes,
                Err(err) => {
                    yield Err(std::io::Error::other(err));
                    continue;
                }
            };
            buffer.push_str(&String::from_utf8_lossy(&bytes));
            while let Some((event, _boundary_len)) = take_next_sse_event(&mut buffer) {
                yield Ok(Bytes::from(translate_anthropic_sse_event(&event)));
            }
        }

        if !buffer.is_empty() {
            yield Ok(Bytes::from(translate_anthropic_sse_event(&buffer)));
        }
    }
}

fn transform_responses_sse(
    incoming: Incoming,
    state: ResponseRewriteState,
    codex_namespace_response_mode: CodexNamespaceResponseMode,
) -> impl futures_util::Stream<Item = Result<Bytes, std::io::Error>> {
    stream! {
        let mut data_stream = incoming.into_data_stream();
        let mut buffer = String::new();

        while let Some(next) = data_stream.next().await {
            let bytes = match next {
                Ok(bytes) => bytes,
                Err(err) => {
                    yield Err(std::io::Error::other(err));
                    continue;
                }
            };
            buffer.push_str(&String::from_utf8_lossy(&bytes));
            while let Some((event, _boundary_len)) = take_next_sse_event(&mut buffer) {
                yield Ok(Bytes::from(translate_responses_sse_event(
                    &event,
                    &state,
                    codex_namespace_response_mode,
                )));
            }
        }

        if !buffer.is_empty() {
            yield Ok(Bytes::from(translate_responses_sse_event(
                &buffer,
                &state,
                codex_namespace_response_mode,
            )));
        }
    }
}

fn take_next_sse_event(buffer: &mut String) -> Option<(String, usize)> {
    let crlf = buffer.find("\r\n\r\n").map(|idx| (idx, 4));
    let lf = buffer.find("\n\n").map(|idx| (idx, 2));
    let (idx, boundary_len) = match (crlf, lf) {
        (Some(crlf), Some(lf)) => {
            if crlf.0 <= lf.0 { crlf } else { lf }
        },
        (Some(value), None) | (None, Some(value)) => value,
        (None, None) => return None,
    };
    let event = buffer[..idx].to_owned();
    buffer.drain(..idx + boundary_len);
    Some((event, boundary_len))
}

fn take_next_sse_event_bytes(buffer: &mut Vec<u8>) -> Option<Result<String, String>> {
    let (idx, boundary_len) = find_next_sse_boundary(buffer)?;
    let event_bytes = buffer[..idx].to_vec();
    buffer.drain(..idx + boundary_len);
    Some(
        String::from_utf8(event_bytes)
            .map_err(|err| format!("backend SSE event was not valid UTF-8: {err}")),
    )
}

fn find_next_sse_boundary(buffer: &[u8]) -> Option<(usize, usize)> {
    let crlf = find_bytes(buffer, b"\r\n\r\n").map(|idx| (idx, 4));
    let lf = find_bytes(buffer, b"\n\n").map(|idx| (idx, 2));
    match (crlf, lf) {
        (Some(crlf), Some(lf)) => Some(if crlf.0 <= lf.0 { crlf } else { lf }),
        (Some(value), None) | (None, Some(value)) => Some(value),
        (None, None) => None,
    }
}

fn find_bytes(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() || haystack.len() < needle.len() {
        return None;
    }

    haystack
        .windows(needle.len())
        .position(|window| window == needle)
}

fn translate_gemini_sse_event(
    event: &str,
    accumulator: &mut GeminiStreamAccumulator,
) -> Vec<String> {
    let data = collect_sse_data(event);
    if data.is_none() {
        return vec![format!("{event}\n\n")];
    }
    let data = data.unwrap();
    accumulator
        .translate_openai_sse_data(&data)
        .into_iter()
        .map(|payload| format!("data: {payload}\n\n"))
        .collect()
}


fn translate_ollama_sse_event(
    event: &str,
    accumulator: &mut OllamaStreamAccumulator,
) -> Vec<String> {
    let Some(data) = collect_sse_data(event) else {
        return ollama_backend_error_ndjson_line_from_str(event)
            .into_iter()
            .collect();
    };

    if let Some(line) = ollama_backend_error_ndjson_line_from_str(&data) {
        return vec![line];
    }

    accumulator
        .translate_openai_sse_data(&data)
        .into_iter()
        .map(|payload| format!("{payload}\n"))
        .collect()
}

fn translate_anthropic_sse_event(event: &str) -> String {
    let Some(data) = collect_sse_data(event) else {
        return format!("{event}\n\n");
    };
    let rewritten = rewrite_anthropic_messages_sse_data(&data);
    replace_sse_data(event, &rewritten)
}

fn replace_sse_data(event: &str, rewritten_data: &str) -> String {
    let mut lines = Vec::new();
    let mut replaced_data = false;

    for line in event.lines() {
        if line.starts_with("data:") {
            if !replaced_data {
                for rewritten_line in rewritten_data.lines() {
                    lines.push(format!("data: {rewritten_line}"));
                }
                replaced_data = true;
            }
        } else {
            lines.push(line.to_owned());
        }
    }

    if !replaced_data {
        lines.push(format!("data: {rewritten_data}"));
    }

    let mut out = lines.join("\n");
    out.push_str("\n\n");
    out
}

fn translate_responses_sse_event(
    event: &str,
    state: &ResponseRewriteState,
    codex_namespace_response_mode: CodexNamespaceResponseMode,
) -> String {
    let Some(data) = collect_sse_data(event) else {
        return format!("{event}\n\n");
    };
    let rewritten = rewrite_openai_responses_sse_data_with_mode(
        &data,
        state,
        codex_namespace_response_mode,
    );
    replace_sse_data(event, &rewritten)
}

fn collect_sse_data(event: &str) -> Option<String> {
    let data_lines: Vec<&str> = event
        .lines()
        .filter_map(|line| line.strip_prefix("data:"))
        .map(str::trim_start)
        .collect();
    if data_lines.is_empty() {
        None
    } else {
        Some(data_lines.join("\n"))
    }
}

fn response_from_parts(status: StatusCode, headers: &HeaderMap, body: Body) -> Response<Body> {
    let mut response = Response::new(body);
    *response.status_mut() = status;
    for (name, value) in headers.iter() {
        if should_forward_response_header(name) {
            response.headers_mut().insert(name, value.clone());
        }
    }
    response
}

fn text_response(status: StatusCode, body: &str) -> Response<Body> {
    let mut response = Response::new(Body::from(body.to_owned()));
    *response.status_mut() = status;
    response.headers_mut().insert(
        CONTENT_TYPE,
        HeaderValue::from_static("text/plain; charset=utf-8"),
    );
    response
}

fn json_response(status: StatusCode, body: Value) -> Response<Body> {
    let mut response = Response::new(Body::from(json_bytes(&body)));
    *response.status_mut() = status;
    response.headers_mut().insert(
        CONTENT_TYPE,
        HeaderValue::from_static("application/json; charset=utf-8"),
    );
    response
}

fn bytes_response(status: StatusCode, body: Bytes, content_type: HeaderValue) -> Response<Body> {
    let mut response = Response::new(Body::from(body));
    *response.status_mut() = status;
    response.headers_mut().insert(CONTENT_TYPE, content_type);
    response
}

async fn health_response(state: &AppState) -> Response<Body> {
    let primary = match probe_backend(state, "/health").await {
        Ok(probe) => probe,
        Err(err) => return health_probe_error_response(err, "/health"),
    };

    if primary.status.is_success() {
        return json_response(
            StatusCode::OK,
            json!({
                "status":"ok",
                "backend_ok":true,
                "backend":state.config.backend_base,
                "backend_probe_method":"GET",
                "backend_probe_path":"/health"
            }),
        );
    }

    if primary.status == StatusCode::NOT_FOUND {
        let fallback = match probe_backend(state, "/").await {
            Ok(probe) => probe,
            Err(err) => {
                return health_probe_error_response_with_primary(err, "/", primary.status);
            }
        };

        if fallback.status == StatusCode::OK
            && String::from_utf8_lossy(&fallback.body).contains("Ollama is running")
        {
            return json_response(
                StatusCode::OK,
                json!({
                    "status":"ok",
                    "backend_ok":true,
                    "backend":state.config.backend_base,
                    "backend_probe_method":"GET",
                    "backend_probe_path":"/",
                    "primary_backend_status":primary.status.as_u16()
                }),
            );
        }

        return json_response(
            StatusCode::SERVICE_UNAVAILABLE,
            json!({
                "status":"degraded",
                "backend_ok":false,
                "backend_status":fallback.status.as_u16(),
                "backend_probe_method":"GET",
                "backend_probe_path":"/",
                "primary_backend_status":primary.status.as_u16()
            }),
        );
    }

    json_response(
        StatusCode::SERVICE_UNAVAILABLE,
        json!({
            "status":"degraded",
            "backend_ok":false,
            "backend_status":primary.status.as_u16(),
            "backend_probe_method":"GET",
            "backend_probe_path":"/health"
        }),
    )
}

struct HealthProbe {
    status: StatusCode,
    body: Bytes,
}

async fn probe_backend(state: &AppState, path: &str) -> Result<HealthProbe, ProxyError> {
    let uri = backend_uri(&state.config.backend_base, path)?;
    let request = Request::builder()
        .method(Method::GET)
        .uri(uri)
        .body(Full::new(Bytes::new()))
        .map_err(|err| ProxyError::BadGateway(format!("failed to build backend health request: {err}")))?;

    let response = match timeout(state.config.backend_timeout, state.client.request(request)).await {
        Ok(Ok(response)) => response,
        Ok(Err(err)) => return Err(ProxyError::BadGateway(format!("backend health check failed: {err}"))),
        Err(_) => return Err(ProxyError::Timeout),
    };
    let status = response.status();
    let body = match timeout(state.config.backend_timeout, response.into_body().collect()).await {
        Ok(Ok(collected)) => collected.to_bytes(),
        Ok(Err(err)) => {
            return Err(ProxyError::BadGateway(format!(
                "failed to read backend health response: {err}"
            )));
        }
        Err(_) => return Err(ProxyError::Timeout),
    };

    Ok(HealthProbe { status, body })
}

fn health_probe_error_response(err: ProxyError, path: &str) -> Response<Body> {
    match err {
        ProxyError::BadGateway(message) => json_response(
            StatusCode::SERVICE_UNAVAILABLE,
            json!({
                "status":"degraded",
                "backend_ok":false,
                "error":message,
                "backend_probe_method":"GET",
                "backend_probe_path":path
            }),
        ),
        ProxyError::Timeout => json_response(
            StatusCode::GATEWAY_TIMEOUT,
            json!({
                "status":"degraded",
                "backend_ok":false,
                "error":"backend health check timed out",
                "backend_probe_method":"GET",
                "backend_probe_path":path
            }),
        ),
    }
}

fn health_probe_error_response_with_primary(
    err: ProxyError,
    path: &str,
    primary_status: StatusCode,
) -> Response<Body> {
    match err {
        ProxyError::BadGateway(message) => json_response(
            StatusCode::SERVICE_UNAVAILABLE,
            json!({
                "status":"degraded",
                "backend_ok":false,
                "error":message,
                "backend_probe_method":"GET",
                "backend_probe_path":path,
                "primary_backend_status":primary_status.as_u16()
            }),
        ),
        ProxyError::Timeout => json_response(
            StatusCode::GATEWAY_TIMEOUT,
            json!({
                "status":"degraded",
                "backend_ok":false,
                "error":"backend health check timed out",
                "backend_probe_method":"GET",
                "backend_probe_path":path,
                "primary_backend_status":primary_status.as_u16()
            }),
        ),
    }
}


impl Config {
    fn parse() -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let mut listen = DEFAULT_LISTEN.parse::<SocketAddr>()?;
        let mut gemini_listen = None;
        let mut backend = DEFAULT_BACKEND.to_owned();
        let mut backend_api_key = DEFAULT_BACKEND_API_KEY.to_owned();
        let mut backend_model = DEFAULT_BACKEND_MODEL.to_owned();
        let mut backend_timeout_secs = DEFAULT_TIMEOUT_SECS;
        let mut max_body_bytes = DEFAULT_MAX_BODY_BYTES;
        let mut hardcoded_gemini_classifier = true;
        let mut codex_namespace_response_mode = CodexNamespaceResponseMode::Flat;
        let mut anthropic_schema_mode = AnthropicSchemaMode::default();

        let mut args = std::env::args().skip(1).peekable();
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--listen" => listen = parse_next(&mut args, "--listen")?.parse()?,
                "--gemini-listen" => {
                    gemini_listen = Some(parse_next(&mut args, "--gemini-listen")?.parse()?);
                }
                "--backend" => backend = parse_next(&mut args, "--backend")?,
                "--backend-api-key" => backend_api_key = parse_next(&mut args, "--backend-api-key")?,
                "--backend-model" => backend_model = parse_next(&mut args, "--backend-model")?,
                "--backend-timeout-secs" => {
                    backend_timeout_secs = parse_next(&mut args, "--backend-timeout-secs")?.parse()?;
                }
                "--max-body-bytes" => max_body_bytes = parse_next(&mut args, "--max-body-bytes")?.parse()?,
                "--no-gemini-hardcoded-classifier" => hardcoded_gemini_classifier = false,
                "--codex-namespace-response-mode" => {
                    codex_namespace_response_mode = parse_next(
                        &mut args,
                        "--codex-namespace-response-mode",
                    )?
                    .parse()?;
                }
                "--anthropic-schema-mode" => {
                    anthropic_schema_mode = parse_next(&mut args, "--anthropic-schema-mode")?.parse()?;
                }
                "--help" | "-h" => {
                    print_help();
                    std::process::exit(0);
                }
                unknown => return Err(format!("unknown argument: {unknown}").into()),
            }
        }

        Ok(Self {
            listen,
            gemini_listen,
            backend_base: normalize_backend_base(&backend),
            backend_api_key,
            backend_model,
            backend_timeout: Duration::from_secs(backend_timeout_secs),
            max_body_bytes,
            hardcoded_gemini_classifier,
            codex_namespace_response_mode,
            anthropic_schema_mode,
        })
    }
}

fn parse_next<I>(
    args: &mut std::iter::Peekable<I>,
    flag: &str,
) -> Result<String, Box<dyn std::error::Error + Send + Sync>>
where
    I: Iterator<Item = String>,
{
    args.next()
        .ok_or_else(|| format!("missing value for {flag}").into())
}

fn normalize_backend_base(value: &str) -> String {
    let with_scheme = if value.starts_with("http://") || value.starts_with("https://") {
        value.to_owned()
    } else {
        format!("http://{value}")
    };
    with_scheme.trim_end_matches('/').to_owned()
}

fn init_tracing() {
    let env_filter = std::env::var("RUST_LOG").unwrap_or_else(|_| "llamacpp_proxy=info".to_owned());
    tracing_subscriber::fmt()
        .with_env_filter(env_filter)
        .with_writer(std::io::stderr)
        .init();
}

fn print_help() {
    println!(
        "llamacpp-proxy\n\n\
Usage:\n  llamacpp-proxy [OPTIONS]\n\n\
Options:\n  --listen <ADDR:PORT>                 Proxy listen address [default: 127.0.0.1:8081]\n  --backend <ADDR:PORT|URL>            llama-server backend [default: 127.0.0.1:8080]\n  --backend-api-key <KEY>              Backend API key [default: llamacpp-local]\n  --backend-model <MODEL>              Model name sent to llama-server for translated Gemini requests [default: local-model]\n  --gemini-listen <ADDR:PORT>          Optional second listener for GOOGLE_GEMINI_BASE_URL\n  --backend-timeout-secs <SECONDS>     Backend timeout [default: 120]\n  --max-body-bytes <BYTES>             Max request body [default: 67108864]\n  --no-gemini-hardcoded-classifier     Forward Gemini flash-lite classifier requests instead of short-circuiting\n  --codex-namespace-response-mode <flat|experimental-wrapped>\n                                      Flat unprefixes namespaced calls [default: flat]; experimental-wrapped is opt-in only\n  --anthropic-schema-mode <compat|semantic>\n                                      compat forwards a small parser-safe schema subset [default: compat]; semantic preserves more standard constraints after backend verification\n  -h, --help                           Show this help\n"
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};


    fn test_state() -> AppState {
        let mut connector = HttpConnector::new();
        connector.enforce_http(false);
        AppState {
            config: Arc::new(Config {
                listen: "127.0.0.1:0".parse().unwrap(),
                gemini_listen: None,
                backend_base: "http://127.0.0.1:8080".to_owned(),
                backend_api_key: "llamacpp-local".to_owned(),
                backend_model: "local-model".to_owned(),
                backend_timeout: Duration::from_secs(1),
                max_body_bytes: DEFAULT_MAX_BODY_BYTES,
                hardcoded_gemini_classifier: true,
                codex_namespace_response_mode: CodexNamespaceResponseMode::Flat,
                anthropic_schema_mode: AnthropicSchemaMode::default(),
            }),
            client: Client::builder(TokioExecutor::new()).build(connector),
        }
    }

    fn test_state_with_backend_base(backend_base: String) -> AppState {
        let mut state = test_state();
        let mut config = (*state.config).clone();
        config.backend_base = backend_base;
        state.config = Arc::new(config);
        state
    }

    async fn response_json(response: Response<Body>) -> Value {
        let body = to_bytes(response.into_body(), DEFAULT_MAX_BODY_BYTES).await.unwrap();
        serde_json::from_slice(&body).unwrap()
    }

    async fn spawn_http_backend(
        responses: Vec<(StatusCode, &'static str)>,
    ) -> (String, tokio::task::JoinHandle<Vec<String>>) {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let handle = tokio::spawn(async move {
            let mut paths = Vec::new();
            for (status, body) in responses {
                let (mut socket, _) = listener.accept().await.unwrap();
                let mut request = Vec::new();
                loop {
                    let mut buf = [0_u8; 1024];
                    let read = socket.read(&mut buf).await.unwrap();
                    if read == 0 {
                        break;
                    }
                    request.extend_from_slice(&buf[..read]);
                    if request.windows(4).any(|window| window == b"\r\n\r\n") {
                        break;
                    }
                }
                let request_text = String::from_utf8_lossy(&request);
                let path = request_text
                    .lines()
                    .next()
                    .and_then(|line| line.split_whitespace().nth(1))
                    .unwrap_or("")
                    .to_owned();
                paths.push(path);
                let response = format!(
                    "HTTP/1.1 {} OK\r\ncontent-length: {}\r\nconnection: close\r\ncontent-type: text/plain\r\n\r\n{}",
                    status.as_u16(),
                    body.len(),
                    body
                );
                socket.write_all(response.as_bytes()).await.unwrap();
            }
            paths
        });
        (format!("http://{addr}"), handle)
    }
    #[derive(Debug)]
    struct CapturedRequest {
        method: String,
        path: String,
        body: Bytes,
    }

    async fn read_captured_request(socket: &mut tokio::net::TcpStream) -> CapturedRequest {
        let mut request = Vec::new();
        let header_end = loop {
            let mut buf = [0_u8; 1024];
            let read = socket.read(&mut buf).await.unwrap();
            if read == 0 {
                break request.len();
            }
            request.extend_from_slice(&buf[..read]);
            if let Some(pos) = request.windows(4).position(|window| window == b"\r\n\r\n") {
                break pos + 4;
            }
        };

        let headers = String::from_utf8_lossy(&request[..header_end]);
        let mut lines = headers.lines();
        let request_line = lines.next().unwrap_or_default();
        let mut request_parts = request_line.split_whitespace();
        let method = request_parts.next().unwrap_or_default().to_owned();
        let path = request_parts.next().unwrap_or_default().to_owned();
        let mut content_length = 0_usize;
        for line in lines {
            if let Some((name, value)) = line.split_once(':') {
                if name.eq_ignore_ascii_case("content-length") {
                    content_length = value.trim().parse::<usize>().unwrap_or_default();
                    break;
                }
            }
        }

        let mut body = request[header_end..].to_vec();
        while body.len() < content_length {
            let mut buf = vec![0_u8; content_length - body.len()];
            let read = socket.read(&mut buf).await.unwrap();
            if read == 0 {
                break;
            }
            body.extend_from_slice(&buf[..read]);
        }
        body.truncate(content_length);

        CapturedRequest {
            method,
            path,
            body: Bytes::from(body),
        }
    }

    async fn spawn_capturing_json_backend(
        responses: Vec<(StatusCode, &'static str)>,
    ) -> (String, tokio::task::JoinHandle<Vec<CapturedRequest>>) {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let handle = tokio::spawn(async move {
            let mut requests = Vec::new();
            for (status, body) in responses {
                let (mut socket, _) = listener.accept().await.unwrap();
                requests.push(read_captured_request(&mut socket).await);
                let response = format!(
                    "HTTP/1.1 {} OK\r\ncontent-length: {}\r\nconnection: close\r\ncontent-type: application/json\r\n\r\n{}",
                    status.as_u16(),
                    body.len(),
                    body
                );
                socket.write_all(response.as_bytes()).await.unwrap();
            }
            requests
        });
        (format!("http://{addr}"), handle)
    }

    async fn spawn_ollama_fallback_timeout_backend() -> (String, tokio::task::JoinHandle<Vec<String>>) {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let handle = tokio::spawn(async move {
            let mut paths = Vec::new();

            let (mut health_socket, _) = listener.accept().await.unwrap();
            let health_request = read_captured_request(&mut health_socket).await;
            paths.push(health_request.path);
            let health_response = "HTTP/1.1 404 Not Found\r\ncontent-length: 9\r\nconnection: close\r\ncontent-type: text/plain\r\n\r\nnot found";
            health_socket.write_all(health_response.as_bytes()).await.unwrap();

            let (mut fallback_socket, _) = listener.accept().await.unwrap();
            let fallback_request = read_captured_request(&mut fallback_socket).await;
            paths.push(fallback_request.path);
            tokio::time::sleep(Duration::from_secs(2)).await;

            paths
        });
        (format!("http://{addr}"), handle)
    }

    fn test_state_with_backend_base_and_timeout(backend_base: String, backend_timeout: Duration) -> AppState {
        let mut state = test_state_with_backend_base(backend_base);
        let mut config = (*state.config).clone();
        config.backend_timeout = backend_timeout;
        state.config = Arc::new(config);
        state
    }

    fn json_proxy_request(method: Method, uri: &str, body: Value) -> Request<Body> {
        Request::builder()
            .method(method)
            .uri(uri)
            .header(CONTENT_TYPE, "application/json")
            .body(Body::from(json_bytes(&body)))
            .unwrap()
    }

    fn openai_chat_completion_success() -> &'static str {
        r#"{
            "id":"chatcmpl-test",
            "object":"chat.completion",
            "created":0,
            "model":"local-model",
            "choices":[{
                "index":0,
                "message":{"role":"assistant","content":"ok"},
                "finish_reason":"stop"
            }],
            "usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}
        }"#
    }

    fn assert_backend_body_sanitized(protocol_name: &str, body: &Bytes) -> Value {
        let parsed: Value = serde_json::from_slice(body)
            .unwrap_or_else(|err| panic!("{protocol_name} backend body should be JSON: {err}"));
        assert!(
            parsed.get("reasoning_effort").is_none(),
            "{protocol_name} leaked reasoning_effort: {parsed}"
        );
        assert!(
            parsed.get("thinking").is_none(),
            "{protocol_name} leaked thinking: {parsed}"
        );
        assert!(
            parsed.get("max_completion_tokens").is_none(),
            "{protocol_name} leaked max_completion_tokens: {parsed}"
        );
        if let Some(messages) = parsed.get("messages").and_then(Value::as_array) {
            for message in messages {
                assert!(
                    message.get("reasoning_content").is_none(),
                    "{protocol_name} leaked message reasoning_content: {parsed}"
                );
                assert!(
                    message.get("reasoning").is_none(),
                    "{protocol_name} leaked message reasoning: {parsed}"
                );
            }
        }
        parsed
    }


    fn observed_ollama_request(raw: &str) -> (Method, String, Bytes) {
        let root: Value = serde_json::from_str(raw).unwrap();
        let method = Method::from_bytes(root["method"].as_str().unwrap().as_bytes()).unwrap();
        let path = root["path"].as_str().unwrap().to_owned();
        let body = if root["body"].is_null() {
            Bytes::new()
        } else {
            Bytes::from(serde_json::to_vec(&root["body"]).unwrap())
        };
        (method, path, body)
    }

    #[test]
    fn gemini_path_passthrough_disables_response_translation() {
        let state = test_state();
        let body = Bytes::from_static(br#"{"ordinary":true}"#);
        let translated = translate_request(
            &state,
            &Method::POST,
            Protocol::Gemini,
            "/gemini/unknown",
            "/gemini/unknown",
            &body,
        )
        .unwrap();

        match translated {
            TranslatedRequest::Forward {
                target,
                body: forwarded_body,
                stream,
                response_protocol,
                ..
            } => {
                assert_eq!(target, "/gemini/unknown");
                assert_eq!(forwarded_body, body);
                assert!(!stream);
                assert_eq!(response_protocol, Protocol::PassThrough);
            }
            TranslatedRequest::Immediate { .. } | TranslatedRequest::ImmediateRaw { .. } => {
                panic!("unexpected immediate response")
            }
        }
    }

    #[test]
    fn gemini_model_list_routes_to_openai_models_endpoint() {
        let state = test_state();
        let translated = translate_request(
            &state,
            &Method::GET,
            Protocol::Gemini,
            "/v1beta/models",
            "/v1beta/models?pageSize=10",
            &Bytes::new(),
        )
        .unwrap();

        match translated {
            TranslatedRequest::Forward {
                target,
                body,
                stream,
                state,
                response_protocol,
            } => {
                assert_eq!(target, "/v1/models");
                assert!(body.is_empty());
                assert!(!stream);
                assert_eq!(state.gemini_response_kind, Some(GeminiResponseKind::ListModels));
                assert_eq!(response_protocol, Protocol::Gemini);
                assert_eq!(
                    translated_request_method(&Method::GET, response_protocol, &state),
                    Method::GET
                );
            }
            TranslatedRequest::Immediate { .. } | TranslatedRequest::ImmediateRaw { .. } => {
                panic!("Gemini model listing should query backend models")
            }
        }
    }

    #[tokio::test]
    async fn proxy_translates_gemini_model_list_response_end_to_end() {
        let backend_response = r#"{
            "object":"list",
            "next_page_token":"cursor-2",
            "data":[
                {
                    "id":"registry.local/team/qwen3:8b",
                    "display_name":"Qwen 3 8B Local",
                    "meta":{"description":"local registry model","input_token_limit":8192}
                },
                {"id":"models/org/nested/model-name"}
            ]
        }"#;
        let (backend_base, handle) =
            spawn_capturing_json_backend(vec![(StatusCode::OK, backend_response)]).await;
        let state = test_state_with_backend_base(backend_base);
        let request = Request::builder()
            .method(Method::GET)
            .uri("/v1beta/models?pageSize=10")
            .body(Body::empty())
            .unwrap();

        let response = proxy_handler(State(state), request).await;
        assert_eq!(response.status(), StatusCode::OK);
        let body = response_json(response).await;
        assert_eq!(body["nextPageToken"], "cursor-2");
        assert_eq!(body["models"][0]["name"], "models/registry.local/team/qwen3:8b");
        assert_eq!(body["models"][0]["displayName"], "Qwen 3 8B Local");
        assert_eq!(body["models"][0]["description"], "local registry model");
        assert_eq!(body["models"][0]["inputTokenLimit"], 8192);
        assert_eq!(
            body["models"][0]["supportedGenerationMethods"],
            json!(["generateContent", "streamGenerateContent"])
        );
        assert_eq!(body["models"][1]["name"], "models/org/nested/model-name");

        let captured = handle.await.unwrap();
        assert_eq!(captured.len(), 1);
        assert_eq!(captured[0].method, "GET");
        assert_eq!(captured[0].path, "/v1/models");
        assert!(captured[0].body.is_empty());
    }

    #[test]
    fn gemini_model_list_non_get_does_not_route_to_backend_models() {
        let state = test_state();
        let body = Bytes::from_static(br#"{
            "reasoning_effort":"low",
            "max_completion_tokens":7,
            "messages":[{
                "role":"assistant",
                "content":"hi",
                "reasoning_content":"hidden"
            }]
        }"#);
        let translated = translate_request(
            &state,
            &Method::POST,
            Protocol::Gemini,
            "/v1beta/models",
            "/v1beta/models?pageSize=10",
            &body,
        )
        .unwrap();

        match translated {
            TranslatedRequest::Forward {
                target,
                body,
                stream,
                state,
                response_protocol,
            } => {
                assert_eq!(target, "/v1beta/models?pageSize=10");
                assert!(!stream);
                assert_eq!(state.gemini_response_kind, None);
                assert_eq!(response_protocol, Protocol::PassThrough);
                assert_eq!(
                    translated_request_method(
                        &Method::POST,
                        Protocol::Gemini,
                        &gemini_response_state(GeminiResponseKind::ListModels),
                    ),
                    Method::POST
                );

                let parsed: Value = serde_json::from_slice(&body).unwrap();
                assert_eq!(parsed["max_tokens"], 7);
                assert!(parsed.get("max_completion_tokens").is_none());
                assert!(parsed.get("reasoning_effort").is_none());
                assert!(parsed["messages"][0].get("reasoning_content").is_none());
            }
            TranslatedRequest::Immediate { .. } | TranslatedRequest::ImmediateRaw { .. } => {
                panic!("non-GET Gemini model-list path should pass through")
            }
        }
    }

    #[test]
    fn openai_chat_requests_are_sanitized_before_forwarding() {
        let state = test_state();
        let body = Bytes::from_static(br#"{
            "model":"local",
            "stream":true,
            "reasoning_effort":"high",
            "thinking":{"type":"enabled"},
            "max_completion_tokens":32,
            "messages":[{
                "role":"assistant",
                "content":"hello",
                "reasoning_content":"hidden",
                "reasoning":{"text":"hidden"}
            }]
        }"#);

        let TranslatedRequest::Forward { body, stream, .. } = translate_request(
            &state,
            &Method::POST,
            Protocol::OpenAiChat,
            "/v1/chat/completions",
            "/v1/chat/completions",
            &body,
        )
        .unwrap() else {
            panic!("OpenAI Chat request should forward");
        };

        assert!(stream);
        let parsed: Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(parsed["max_tokens"], 32);
        assert!(parsed.get("max_completion_tokens").is_none());
        assert!(parsed.get("reasoning_effort").is_none());
        assert!(parsed.get("thinking").is_none());
        assert!(parsed["messages"][0].get("reasoning_content").is_none());
        assert!(parsed["messages"][0].get("reasoning").is_none());
    }

    #[tokio::test]
    async fn proxy_sanitizes_backend_requests_for_all_translated_protocols() {
        let cases = [
            (
                "openai chat",
                "/v1/chat/completions",
                json!({
                    "model":"local",
                    "stream":false,
                    "reasoning_effort":"high",
                    "thinking":{"type":"enabled"},
                    "max_completion_tokens":21,
                    "messages":[{
                        "role":"assistant",
                        "content":"hello",
                        "reasoning_content":"hidden",
                        "reasoning":{"text":"hidden"}
                    }]
                }),
                "/v1/chat/completions",
                21,
            ),
            (
                "openai responses",
                "/v1/responses",
                json!({
                    "model":"local",
                    "input":"hello",
                    "reasoning_effort":"high",
                    "thinking":{"type":"enabled"},
                    "max_completion_tokens":22,
                    "messages":[{
                        "role":"assistant",
                        "content":"hello",
                        "reasoning_content":"hidden",
                        "reasoning":{"text":"hidden"}
                    }]
                }),
                "/v1/responses",
                22,
            ),
            (
                "anthropic",
                "/v1/messages",
                json!({
                    "model":"local",
                    "max_tokens":64,
                    "reasoning_effort":"high",
                    "thinking":{"type":"enabled"},
                    "max_completion_tokens":23,
                    "messages":[{
                        "role":"assistant",
                        "content":"hello",
                        "reasoning_content":"hidden",
                        "reasoning":{"text":"hidden"}
                    }]
                }),
                "/v1/messages",
                64,
            ),
            (
                "gemini",
                "/v1beta/models/gemini-2.5-flash:generateContent",
                json!({
                    "reasoning_effort":"high",
                    "thinking":{"type":"enabled"},
                    "max_completion_tokens":24,
                    "contents":[{"role":"user","parts":[{"text":"hello"}]}],
                    "generationConfig":{"maxOutputTokens":24}
                }),
                "/v1/chat/completions",
                24,
            ),
            (
                "ollama",
                "/api/chat",
                json!({
                    "model":"local",
                    "stream":false,
                    "reasoning_effort":"high",
                    "thinking":{"type":"enabled"},
                    "max_completion_tokens":25,
                    "messages":[{
                        "role":"assistant",
                        "content":"hello",
                        "reasoning_content":"hidden",
                        "reasoning":{"text":"hidden"}
                    }],
                    "options":{"num_predict":25}
                }),
                "/v1/chat/completions",
                25,
            ),
        ];

        for (protocol_name, uri, request_body, expected_backend_path, expected_max_tokens) in cases {
            let (backend_base, handle) = spawn_capturing_json_backend(vec![(
                StatusCode::OK,
                openai_chat_completion_success(),
            )])
            .await;
            let state = test_state_with_backend_base(backend_base);
            let request = json_proxy_request(Method::POST, uri, request_body);

            let response = proxy_handler(State(state), request).await;
            assert!(
                response.status().is_success(),
                "{protocol_name} proxy response was {}",
                response.status()
            );

            let captured = handle.await.unwrap();
            assert_eq!(captured.len(), 1, "{protocol_name} should send one backend request");
            assert_eq!(captured[0].method, "POST", "{protocol_name} backend method");
            assert_eq!(
                captured[0].path, expected_backend_path,
                "{protocol_name} backend path"
            );
            let parsed = assert_backend_body_sanitized(protocol_name, &captured[0].body);
            assert_eq!(
                parsed["max_tokens"], expected_max_tokens,
                "{protocol_name} max_tokens"
            );
        }
    }

    #[test]
    fn pass_through_protocol_sanitizes_json_before_forwarding() {
        let state = test_state();
        let body = Bytes::from_static(br#"{
            "model":"local",
            "reasoning_effort":"medium",
            "thinking":{"budget_tokens":128},
            "max_completion_tokens":24,
            "messages":[{
                "role":"assistant",
                "content":"hi",
                "reasoning_content":"hidden",
                "reasoning":{"trace":"hidden"}
            }]
        }"#);

        let TranslatedRequest::Forward {
            body,
            response_protocol,
            ..
        } = translate_request(
            &state,
            &Method::POST,
            Protocol::PassThrough,
            "/custom/proxy",
            "/custom/proxy",
            &body,
        )
        .unwrap() else {
            panic!("pass-through request should forward");
        };

        assert_eq!(response_protocol, Protocol::PassThrough);
        let parsed: Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(parsed["max_tokens"], 24);
        assert!(parsed.get("max_completion_tokens").is_none());
        assert!(parsed.get("reasoning_effort").is_none());
        assert!(parsed.get("thinking").is_none());
        assert!(parsed["messages"][0].get("reasoning_content").is_none());
        assert!(parsed["messages"][0].get("reasoning").is_none());
    }

    #[test]
    fn pass_through_protocol_preserves_malformed_non_json_body() {
        let state = test_state();
        let body = Bytes::from_static(b"not json but still pass through");

        let TranslatedRequest::Forward {
            body: forwarded_body,
            response_protocol,
            ..
        } = translate_request(
            &state,
            &Method::POST,
            Protocol::PassThrough,
            "/custom/proxy",
            "/custom/proxy",
            &body,
        )
        .unwrap() else {
            panic!("pass-through request should forward");
        };

        assert_eq!(response_protocol, Protocol::PassThrough);
        assert_eq!(forwarded_body, body);
    }

    #[tokio::test]
    async fn proxy_sanitizes_pass_through_and_translation_failure_fallback_requests() {
        let cases = [
            (
                "pass-through",
                "/custom/proxy?x=1",
                json!({
                    "ordinary":true,
                    "reasoning_effort":"medium",
                    "thinking":{"budget_tokens":128},
                    "max_completion_tokens":31,
                    "messages":[{
                        "role":"assistant",
                        "content":"hi",
                        "reasoning_content":"hidden",
                        "reasoning":{"trace":"hidden"}
                    }]
                }),
                "/custom/proxy?x=1",
                31,
            ),
            (
                "translation failure fallback",
                "/v1/responses?x=1",
                json!({
                    "unexpected":true,
                    "reasoning_effort":"medium",
                    "thinking":{"budget_tokens":128},
                    "max_completion_tokens":32,
                    "messages":[{
                        "role":"assistant",
                        "content":"hi",
                        "reasoning_content":"hidden",
                        "reasoning":{"trace":"hidden"}
                    }]
                }),
                "/v1/responses?x=1",
                32,
            ),
        ];

        for (case_name, uri, request_body, expected_backend_path, expected_max_tokens) in cases {
            let (backend_base, handle) =
                spawn_capturing_json_backend(vec![(StatusCode::OK, r#"{"ok":true}"#)]).await;
            let state = test_state_with_backend_base(backend_base);
            let response = proxy_handler(
                State(state),
                json_proxy_request(Method::POST, uri, request_body),
            )
            .await;
            assert_eq!(response.status(), StatusCode::OK, "{case_name} status");

            let captured = handle.await.unwrap();
            assert_eq!(captured.len(), 1, "{case_name} backend request count");
            assert_eq!(captured[0].path, expected_backend_path, "{case_name} backend path");
            let parsed = assert_backend_body_sanitized(case_name, &captured[0].body);
            assert_eq!(parsed["max_tokens"], expected_max_tokens, "{case_name} max_tokens");
        }
    }

    #[tokio::test]
    async fn health_response_reports_primary_probe_path() {
        let (backend_base, handle) = spawn_http_backend(vec![(StatusCode::OK, "healthy")]).await;
        let state = test_state_with_backend_base(backend_base);

        let response = health_response(&state).await;
        assert_eq!(response.status(), StatusCode::OK);
        let body = response_json(response).await;
        assert_eq!(body["backend_ok"], true);
        assert_eq!(body["backend_probe_method"], "GET");
        assert_eq!(body["backend_probe_path"], "/health");
        assert_eq!(handle.await.unwrap(), vec!["/health".to_owned()]);
    }

    #[tokio::test]
    async fn health_response_falls_back_to_ollama_root_after_health_404() {
        let (backend_base, handle) = spawn_http_backend(vec![
            (StatusCode::NOT_FOUND, "not found"),
            (StatusCode::OK, "Ollama is running"),
        ])
        .await;
        let state = test_state_with_backend_base(backend_base);

        let response = health_response(&state).await;
        assert_eq!(response.status(), StatusCode::OK);
        let body = response_json(response).await;
        assert_eq!(body["backend_ok"], true);
        assert_eq!(body["backend_probe_method"], "GET");
        assert_eq!(body["backend_probe_path"], "/");
        assert_eq!(body["primary_backend_status"], 404);
        assert_eq!(handle.await.unwrap(), vec!["/health".to_owned(), "/".to_owned()]);
    }

    #[tokio::test]
    async fn health_response_does_not_accept_non_ollama_root_body() {
        let (backend_base, handle) = spawn_http_backend(vec![
            (StatusCode::NOT_FOUND, "not found"),
            (StatusCode::OK, "not ollama"),
        ])
        .await;
        let state = test_state_with_backend_base(backend_base);

        let response = health_response(&state).await;
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        let body = response_json(response).await;
        assert_eq!(body["backend_ok"], false);
        assert_eq!(body["backend_status"], 200);
        assert_eq!(body["backend_probe_path"], "/");
        assert_eq!(body["primary_backend_status"], 404);
        assert_eq!(handle.await.unwrap(), vec!["/health".to_owned(), "/".to_owned()]);
    }

    #[tokio::test]
    async fn health_response_reports_primary_probe_connection_error() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        drop(listener);
        let state = test_state_with_backend_base(format!("http://{addr}"));

        let response = health_response(&state).await;
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        let body = response_json(response).await;
        assert_eq!(body["backend_ok"], false);
        assert_eq!(body["backend_probe_method"], "GET");
        assert_eq!(body["backend_probe_path"], "/health");
        assert!(body["error"]
            .as_str()
            .unwrap()
            .contains("backend health check failed"));
    }

    #[tokio::test]
    async fn health_response_reports_fallback_probe_timeout_after_primary_404() {
        let (backend_base, handle) = spawn_ollama_fallback_timeout_backend().await;
        let state = test_state_with_backend_base_and_timeout(
            backend_base,
            Duration::from_millis(500),
        );

        let response = health_response(&state).await;
        assert_eq!(response.status(), StatusCode::GATEWAY_TIMEOUT);
        let body = response_json(response).await;
        assert_eq!(body["backend_ok"], false);
        assert_eq!(body["backend_probe_method"], "GET");
        assert_eq!(body["backend_probe_path"], "/");
        assert_eq!(body["primary_backend_status"], 404);
        assert_eq!(body["error"], "backend health check timed out");
        assert_eq!(handle.await.unwrap(), vec!["/health".to_owned(), "/".to_owned()]);
    }

    #[test]
    fn translation_failure_fallback_preserves_json_without_backend_incompatible_fields() {
        let body = Bytes::from_static(br#"{"bad":"shape"}"#);
        let fallback = translation_failure_passthrough("/v1/messages?x=1", &body);
        assert_eq!(fallback.target, "/v1/messages?x=1");
        assert_eq!(fallback.body, body);
        assert!(!fallback.stream);
        assert_eq!(fallback.response_protocol, Protocol::PassThrough);
        assert!(fallback.state.tool_name_map.is_empty());
        assert!(fallback.state.namespace_tool_map.is_empty());
    }

    #[test]
    fn translation_failure_fallback_sanitizes_json_before_forwarding() {
        let body = Bytes::from_static(br#"{
            "unexpected":true,
            "reasoning_effort":"high",
            "thinking":{"type":"enabled"},
            "max_completion_tokens":9,
            "messages":[{
                "role":"assistant",
                "content":"hi",
                "reasoning_content":"hidden",
                "reasoning":"hidden"
            }]
        }"#);

        let fallback = translation_failure_passthrough("/v1/messages?x=1", &body);
        assert_eq!(fallback.target, "/v1/messages?x=1");
        assert_eq!(fallback.response_protocol, Protocol::PassThrough);

        let parsed: Value = serde_json::from_slice(&fallback.body).unwrap();
        assert_eq!(parsed["max_tokens"], 9);
        assert!(parsed.get("max_completion_tokens").is_none());
        assert!(parsed.get("reasoning_effort").is_none());
        assert!(parsed.get("thinking").is_none());
        assert!(parsed["messages"][0].get("reasoning_content").is_none());
        assert!(parsed["messages"][0].get("reasoning").is_none());
    }

    #[test]
    fn gemini_non_generation_passthrough_sanitizes_json_before_forwarding() {
        let state = test_state();
        let body = Bytes::from_static(br#"{
            "ordinary":true,
            "reasoning_effort":"low",
            "max_completion_tokens":11
        }"#);
        let translated = translate_request(
            &state,
            &Method::POST,
            Protocol::Gemini,
            "/gemini/unknown",
            "/gemini/unknown",
            &body,
        )
        .unwrap();

        let TranslatedRequest::Forward {
            body,
            response_protocol,
            ..
        } = translated else {
            panic!("non-generation Gemini path should pass through");
        };

        assert_eq!(response_protocol, Protocol::PassThrough);
        let parsed: Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(parsed["ordinary"], true);
        assert_eq!(parsed["max_tokens"], 11);
        assert!(parsed.get("max_completion_tokens").is_none());
        assert!(parsed.get("reasoning_effort").is_none());
    }

    #[test]
    fn well_formed_unexpected_responses_json_uses_fallback_path() {
        let state = test_state();
        let body = Bytes::from_static(br#"{"unexpected":true}"#);
        let translated = translate_request(
            &state,
            &Method::POST,
            Protocol::OpenAiResponses,
            "/v1/responses",
            "/v1/responses",
            &body,
        );
        let err = translated.expect_err("unexpected but well-formed JSON should use fallback path");
        assert!(err
            .to_string()
            .contains("does not match OpenAI Responses request shape"));
    }

    #[test]
    fn well_formed_unexpected_anthropic_json_uses_fallback_path() {
        let state = test_state();
        let body = Bytes::from_static(br#"{"unexpected":true}"#);
        let translated = translate_request(
            &state,
            &Method::POST,
            Protocol::AnthropicMessages,
            "/v1/messages",
            "/v1/messages",
            &body,
        );
        let err = translated.expect_err("unexpected but well-formed JSON should use fallback path");
        assert!(err
            .to_string()
            .contains("does not match Anthropic Messages request shape"));
    }

    #[test]
    fn anthropic_translation_uses_compat_schema_mode_by_default() {
        let state = test_state();
        let body = Bytes::from_static(br#"{
            "model":"local",
            "messages":[{"role":"user","content":"hello"}],
            "max_tokens":16,
            "tools":[{
                "name":"Edit",
                "input_schema":{
                    "type":"object",
                    "properties":{
                        "path":{"type":"string","pattern":"^/"},
                        "mode":{"type":"string","enum":["safe","force"]}
                    },
                    "required":["path","mode"]
                }
            }]
        }"#);
        let translated = translate_request(
            &state,
            &Method::POST,
            Protocol::AnthropicMessages,
            "/v1/messages",
            "/v1/messages",
            &body,
        )
        .unwrap();

        let TranslatedRequest::Forward { body, .. } = translated else {
            panic!("valid Anthropic request should forward");
        };
        let parsed: Value = serde_json::from_slice(&body).unwrap();
        let schema = &parsed["tools"][0]["input_schema"];
        assert!(schema["properties"]["path"].get("pattern").is_none());
        assert!(schema["properties"]["mode"].get("enum").is_none());
        assert_eq!(schema["required"], json!(["path","mode"]));
        assert!(schema["properties"]["path"]["description"]
            .as_str()
            .unwrap()
            .contains("pattern"));
        assert!(schema["properties"]["mode"]["description"]
            .as_str()
            .unwrap()
            .contains("enum"));
    }

    #[test]
    fn valid_anthropic_without_tools_still_forwards_as_anthropic() {
        let state = test_state();
        let body = Bytes::from_static(br#"{"model":"local","messages":[{"role":"user","content":"hello"}],"max_tokens":16}"#);
        let translated = translate_request(
            &state,
            &Method::POST,
            Protocol::AnthropicMessages,
            "/v1/messages",
            "/v1/messages",
            &body,
        )
        .unwrap();

        match translated {
            TranslatedRequest::Forward {
                target,
                response_protocol,
                ..
            } => {
                assert_eq!(target, "/v1/messages");
                assert_eq!(response_protocol, Protocol::AnthropicMessages);
            }
            TranslatedRequest::Immediate { .. } | TranslatedRequest::ImmediateRaw { .. } => {
                panic!("valid Anthropic request should forward")
            }
        }
    }

    #[test]
    fn malformed_json_on_responses_returns_protocol_400() {
        let state = test_state();
        let body = Bytes::from_static(b"{not json");
        let translated = translate_request(
            &state,
            &Method::POST,
            Protocol::OpenAiResponses,
            "/v1/responses",
            "/v1/responses",
            &body,
        )
        .unwrap();

        match translated {
            TranslatedRequest::Immediate { status, body } => {
                assert_eq!(status, StatusCode::BAD_REQUEST);
                assert_eq!(body["error"]["code"], 400);
                assert!(body["error"]["message"].as_str().unwrap().contains("malformed JSON"));
            }
            TranslatedRequest::Forward { .. } | TranslatedRequest::ImmediateRaw { .. } => {
                panic!("malformed JSON must not be forwarded")
            }
        }
    }

    #[test]
    fn malformed_json_on_anthropic_messages_returns_protocol_400() {
        let state = test_state();
        let body = Bytes::from_static(b"{not json");
        let translated = translate_request(
            &state,
            &Method::POST,
            Protocol::AnthropicMessages,
            "/v1/messages",
            "/v1/messages",
            &body,
        )
        .unwrap();

        match translated {
            TranslatedRequest::Immediate { status, body } => {
                assert_eq!(status, StatusCode::BAD_REQUEST);
                assert_eq!(body["type"], "error");
                assert_eq!(body["error"]["type"], "invalid_request_error");
                assert!(body["error"]["message"].as_str().unwrap().contains("malformed JSON"));
            }
            TranslatedRequest::Forward { .. } | TranslatedRequest::ImmediateRaw { .. } => {
                panic!("malformed JSON must not be forwarded")
            }
        }
    }

    #[test]
    fn malformed_json_on_gemini_generation_returns_protocol_400() {
        let state = test_state();
        let body = Bytes::from_static(b"{not json");
        let translated = translate_request(
            &state,
            &Method::POST,
            Protocol::Gemini,
            "/v1beta/models/gemini-2.5-flash:generateContent",
            "/v1beta/models/gemini-2.5-flash:generateContent",
            &body,
        )
        .unwrap();

        match translated {
            TranslatedRequest::Immediate { status, body } => {
                assert_eq!(status, StatusCode::BAD_REQUEST);
                assert_eq!(body["error"]["code"], 400);
                assert_eq!(body["error"]["status"], "INVALID_ARGUMENT");
                assert!(body["error"]["message"].as_str().unwrap().contains("malformed JSON"));
            }
            TranslatedRequest::Forward { .. } | TranslatedRequest::ImmediateRaw { .. } => {
                panic!("malformed JSON must not be forwarded")
            }
        }
    }

    #[test]
    fn well_formed_unexpected_gemini_generation_falls_back_to_passthrough() {
        let state = test_state();
        let body = Bytes::from_static(br#"{"unexpected":true}"#);
        let translated = translate_request(
            &state,
            &Method::POST,
            Protocol::Gemini,
            "/v1beta/models/gemini-2.5-flash:generateContent",
            "/v1beta/models/gemini-2.5-flash:generateContent?alt=sse",
            &body,
        );
        let err = translated.expect_err("unexpected but well-formed JSON should use fallback path");
        assert!(err
            .to_string()
            .contains("does not match Gemini generateContent request shape"));

        let fallback = translation_failure_passthrough(
            "/v1beta/models/gemini-2.5-flash:generateContent?alt=sse",
            &body,
        );
        assert_eq!(fallback.target, "/v1beta/models/gemini-2.5-flash:generateContent?alt=sse");
        assert_eq!(fallback.body, body);
        assert_eq!(fallback.response_protocol, Protocol::PassThrough);
    }

    #[test]
    fn malformed_json_on_non_generation_gemini_path_still_passes_through() {
        let state = test_state();
        let body = Bytes::from_static(b"{not json");
        let translated = translate_request(
            &state,
            &Method::POST,
            Protocol::Gemini,
            "/gemini/unknown",
            "/gemini/unknown",
            &body,
        )
        .unwrap();

        match translated {
            TranslatedRequest::Forward {
                target,
                body: forwarded_body,
                response_protocol,
                ..
            } => {
                assert_eq!(target, "/gemini/unknown");
                assert_eq!(forwarded_body, body);
                assert_eq!(response_protocol, Protocol::PassThrough);
            }
            TranslatedRequest::Immediate { .. } | TranslatedRequest::ImmediateRaw { .. } => {
                panic!("non-generation Gemini path should pass through")
            }
        }
    }

    #[test]
    fn ollama_unknown_path_passthrough_sanitizes_json_before_forwarding() {
        let state = test_state();
        let body = Bytes::from_static(br#"{
            "model":"local",
            "reasoning_effort":"medium",
            "thinking":true,
            "max_completion_tokens":17
        }"#);

        let translated = translate_request(
            &state,
            &Method::POST,
            Protocol::Ollama,
            "/api/unknown",
            "/api/unknown",
            &body,
        )
        .unwrap();

        let TranslatedRequest::Forward {
            body,
            response_protocol,
            ..
        } = translated else {
            panic!("unknown Ollama path should pass through");
        };

        assert_eq!(response_protocol, Protocol::PassThrough);
        let parsed: Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(parsed["max_tokens"], 17);
        assert!(parsed.get("max_completion_tokens").is_none());
        assert!(parsed.get("reasoning_effort").is_none());
        assert!(parsed.get("thinking").is_none());
    }

    #[test]
    fn anthropic_sse_rewrite_preserves_event_name() {
        let event = "event: content_block_start\ndata: {\"type\":\"content_block_start\",\"content_block\":{\"type\":\"text\",\"text\":\"hi\"}}";
        let out = translate_anthropic_sse_event(event);
        assert!(out.starts_with("event: content_block_start\n"));
        assert!(out.contains("cache_control"));
    }

    #[test]
    fn responses_sse_rewrite_preserves_event_metadata() {
        let request = Bytes::from_static(br#"{
            "model": "local",
            "stream": true,
            "tools": [{
                "type": "namespace",
                "name": "multi_agent_v1",
                "tools": [{"type": "function", "name": "close_agent", "parameters": {"type": "object"}}]
            }]
        }"#);
        let translated = translate_request(
            &test_state(),
            &Method::POST,
            Protocol::OpenAiResponses,
            "/v1/responses",
            "/v1/responses",
            &request,
        )
        .unwrap();
        let state = match translated {
            TranslatedRequest::Forward { state, .. } => state,
            TranslatedRequest::Immediate { .. } | TranslatedRequest::ImmediateRaw { .. } => {
                panic!("request should forward")
            }
        };

        let event = concat!(
            ": upstream keepalive\n",
            "id: evt_123\n",
            "event: response.output_item.done\n",
            "retry: 2500\n",
            "data: {\"type\":\"response.output_item.done\",\"item\":{\"type\":\"function_call\",\"name\":\"multi_agent_v1__close_agent\"}}"
        );
        let out = translate_responses_sse_event(
            event,
            &state,
            CodexNamespaceResponseMode::Flat,
        );

        assert!(out.starts_with(": upstream keepalive\n"));
        assert!(out.contains("id: evt_123\n"));
        assert!(out.contains("event: response.output_item.done\n"));
        assert!(out.contains("retry: 2500\n"));
        assert!(out.ends_with("\n\n"));
        assert!(out.contains("data: "));
        assert!(out.contains("\"name\":\"close_agent\""));
        assert!(!out.contains("multi_agent_v1__close_agent"));
    }

    #[test]
    fn responses_sse_rewrite_preserves_multiline_data_shape() {
        let state = ResponseRewriteState::default();
        let event = "event: response.created\ndata: {\"type\":\"response.created\",\ndata: \"response\":{}}";
        let out = translate_responses_sse_event(
            event,
            &state,
            CodexNamespaceResponseMode::Flat,
        );
        assert!(out.starts_with("event: response.created\n"));
        assert!(out.contains("data: {\"type\":\"response.created\","));
        assert!(out.contains("data: \"response\":{}"));
    }


    #[test]
    fn ollama_chat_translates_to_openai_chat_and_defaults_streaming() {
        let state = test_state();
        let body = Bytes::from_static(br#"{"model":"qwen3:32b","messages":[{"role":"user","content":"hello"}]}"#);
        let translated = translate_request(&state, &Method::POST, Protocol::Ollama, "/api/chat", "/api/chat", &body).unwrap();

        match translated {
            TranslatedRequest::Forward {
                target,
                body,
                stream,
                response_protocol,
                state,
            } => {
                assert_eq!(target, "/v1/chat/completions");
                assert!(stream);
                assert_eq!(response_protocol, Protocol::Ollama);
                assert_eq!(state.ollama_response_kind, Some(OllamaResponseKind::Chat));
                let parsed: Value = serde_json::from_slice(&body).unwrap();
                assert_eq!(parsed["model"], "qwen3:32b");
                assert_eq!(parsed["stream"], true);
                assert_eq!(parsed["messages"][0]["role"], "user");
            }
            TranslatedRequest::Immediate { .. } | TranslatedRequest::ImmediateRaw { .. } => {
                panic!("Ollama chat should forward")
            }
        }
    }

    #[test]
    fn ollama_tags_forwards_to_openai_models() {
        let state = test_state();
        let translated = translate_request(
            &state,
            &Method::POST,
            Protocol::Ollama,
            "/api/tags",
            "/api/tags",
            &Bytes::new(),
        )
        .unwrap();

        match translated {
            TranslatedRequest::Forward {
                target,
                body,
                stream,
                response_protocol,
                state,
            } => {
                assert_eq!(target, "/v1/models");
                assert!(body.is_empty());
                assert!(!stream);
                assert_eq!(response_protocol, Protocol::Ollama);
                assert_eq!(state.ollama_response_kind, Some(OllamaResponseKind::Tags));
            }
            TranslatedRequest::Immediate { .. } | TranslatedRequest::ImmediateRaw { .. } => {
                panic!("Ollama tags should forward")
            }
        }
    }

    #[test]
    fn ollama_show_queries_backend_models_with_get() {
        let state = test_state();
        let body = Bytes::from_static(br#"{"model":"qwen3:32b"}"#);
        let translated = translate_request(&state, &Method::POST, Protocol::Ollama, "/api/show", "/api/show", &body).unwrap();

        match translated {
            TranslatedRequest::Forward {
                target,
                body,
                stream,
                response_protocol,
                state,
            } => {
                assert_eq!(target, "/v1/models");
                assert!(body.is_empty());
                assert!(!stream);
                assert_eq!(response_protocol, Protocol::Ollama);
                assert_eq!(state.ollama_response_kind, Some(OllamaResponseKind::Show));
                assert_eq!(state.ollama_requested_model.as_deref(), Some("qwen3:32b"));
                assert_eq!(
                    translated_request_method(&Method::POST, response_protocol, &state),
                    Method::GET
                );
            }
            TranslatedRequest::Immediate { .. } | TranslatedRequest::ImmediateRaw { .. } => {
                panic!("Ollama show should query backend model metadata")
            }
        }
    }

    #[test]
    fn ollama_pull_defaults_to_ndjson_synthetic_success() {
        let state = test_state();
        let body = Bytes::from_static(br#"{"model":"qwen3:32b"}"#);
        let translated = translate_request(&state, &Method::POST, Protocol::Ollama, "/api/pull", "/api/pull", &body).unwrap();

        match translated {
            TranslatedRequest::ImmediateRaw {
                status,
                body,
                content_type,
            } => {
                assert_eq!(status, StatusCode::OK);
                assert_eq!(content_type, HeaderValue::from_static("application/x-ndjson"));
                let line = std::str::from_utf8(&body).unwrap();
                assert!(line.ends_with('\n'));
                let parsed: Value = serde_json::from_str(line.trim()).unwrap();
                assert_eq!(parsed["status"], "success");
                assert_eq!(parsed["model"], "qwen3:32b");
            }
            TranslatedRequest::Forward { .. } | TranslatedRequest::Immediate { .. } => {
                panic!("Ollama pull should return synthetic NDJSON by default")
            }
        }
    }

    #[test]
    fn observed_tags_and_show_fixtures_route_to_backend_models() {
        let state = test_state();
        for raw in [
            include_str!("../fixtures/ollama/observed/python_0_6_2_tags.request.json"),
            include_str!("../fixtures/ollama/observed/js_0_6_3_tags.request.json"),
            include_str!("../fixtures/ollama/observed/python_0_6_2_show.request.json"),
            include_str!("../fixtures/ollama/observed/js_0_6_3_show.request.json"),
        ] {
            let (method, path, body) = observed_ollama_request(raw);
            let translated = translate_request(&state, &method, Protocol::Ollama, &path, &path, &body).unwrap();
            match translated {
                TranslatedRequest::Forward {
                    target,
                    body,
                    stream,
                    response_protocol,
                    state,
                } => {
                    assert_eq!(target, "/v1/models");
                    assert!(body.is_empty());
                    assert!(!stream);
                    assert_eq!(response_protocol, Protocol::Ollama);
                    assert_eq!(translated_request_method(&method, response_protocol, &state), Method::GET);
                    assert!(matches!(
                        state.ollama_response_kind,
                        Some(OllamaResponseKind::Tags | OllamaResponseKind::Show)
                    ));
                }
                TranslatedRequest::Immediate { .. } | TranslatedRequest::ImmediateRaw { .. } => {
                    panic!("tags/show observed fixtures should query backend model metadata")
                }
            }
        }
    }

    #[test]
    fn observed_pull_and_delete_fixtures_return_client_compatible_synthetic_success() {
        let state = test_state();
        for raw in [
            include_str!("../fixtures/ollama/observed/python_0_6_2_pull.request.json"),
            include_str!("../fixtures/ollama/observed/js_0_6_3_pull.request.json"),
            include_str!("../fixtures/ollama/observed/python_0_6_2_delete.request.json"),
            include_str!("../fixtures/ollama/observed/js_0_6_3_delete.request.json"),
        ] {
            let (_method, path, body) = observed_ollama_request(raw);
            let translated = translate_request(&state, &Method::POST, Protocol::Ollama, &path, &path, &body).unwrap();
            match translated {
                TranslatedRequest::Immediate { status, body } => {
                    assert_eq!(status, StatusCode::OK);
                    assert_eq!(body["status"], "success");
                    assert_eq!(body["model"], "qwen3:32b");
                }
                TranslatedRequest::ImmediateRaw { status, body, .. } => {
                    assert_eq!(status, StatusCode::OK);
                    let parsed: Value = serde_json::from_slice(&body).unwrap();
                    assert_eq!(parsed["status"], "success");
                    assert_eq!(parsed["model"], "qwen3:32b");
                }
                TranslatedRequest::Forward { .. } => {
                    panic!("pull/delete observed fixtures should not forward lifecycle operations")
                }
            }
        }
    }

    #[test]
    fn observed_nonstream_chat_and_generate_fixtures_route_without_streaming() {
        let state = test_state();
        for (raw, expected_raw) in [
            (
                include_str!("../fixtures/ollama/observed/python_0_6_2_chat_nonstream.request.json"),
                include_str!("../fixtures/ollama/observed/python_0_6_2_chat_nonstream.expected-chat.json"),
            ),
            (
                include_str!("../fixtures/ollama/observed/js_0_6_3_chat_nonstream.request.json"),
                include_str!("../fixtures/ollama/observed/js_0_6_3_chat_nonstream.expected-chat.json"),
            ),
            (
                include_str!("../fixtures/ollama/observed/python_0_6_2_generate_nonstream.request.json"),
                include_str!("../fixtures/ollama/observed/python_0_6_2_generate_nonstream.expected-chat.json"),
            ),
            (
                include_str!("../fixtures/ollama/observed/js_0_6_3_generate_nonstream.request.json"),
                include_str!("../fixtures/ollama/observed/js_0_6_3_generate_nonstream.expected-chat.json"),
            ),
        ] {
            let (_method, path, body) = observed_ollama_request(raw);
            let expected: Value = serde_json::from_str(expected_raw).unwrap();
            let translated = translate_request(&state, &Method::POST, Protocol::Ollama, &path, &path, &body).unwrap();
            match translated {
                TranslatedRequest::Forward { target, body, stream, .. } => {
                    assert_eq!(target, "/v1/chat/completions");
                    assert!(!stream);
                    let parsed: Value = serde_json::from_slice(&body).unwrap();
                    assert_eq!(parsed, expected);
                }
                TranslatedRequest::Immediate { .. } | TranslatedRequest::ImmediateRaw { .. } => {
                    panic!("non-streaming chat/generate observed fixtures should forward")
                }
            }
        }
    }

    #[test]
    fn malformed_json_on_ollama_returns_protocol_400() {
        let state = test_state();
        let translated = translate_request(
            &state,
            &Method::POST,
            Protocol::Ollama,
            "/api/chat",
            "/api/chat",
            &Bytes::from_static(b"{not json"),
        )
        .unwrap();

        match translated {
            TranslatedRequest::Immediate { status, body } => {
                assert_eq!(status, StatusCode::BAD_REQUEST);
                assert!(body["error"].as_str().unwrap().contains("malformed JSON"));
            }
            TranslatedRequest::Forward { .. } | TranslatedRequest::ImmediateRaw { .. } => {
                panic!("malformed Ollama JSON must not be forwarded")
            }
        }
    }

    #[test]
    fn proxy_replaces_client_accept_encoding_with_identity_for_backend() {
        assert!(!should_forward_request_header(&ACCEPT_ENCODING));
        assert!(should_forward_request_header(&CONTENT_TYPE));
    }

    #[test]
    fn translated_paths_reject_compressed_backend_responses() {
        let mut headers = HeaderMap::new();
        assert_eq!(unsupported_content_encoding(&headers), None);

        headers.insert(CONTENT_ENCODING, HeaderValue::from_static("identity"));
        assert_eq!(unsupported_content_encoding(&headers), None);

        headers.insert(CONTENT_ENCODING, HeaderValue::from_static("gzip"));
        assert_eq!(unsupported_content_encoding(&headers), Some("gzip".to_owned()));

        let mut state = ResponseRewriteState::default();
        assert!(response_body_is_rewritten(Protocol::Ollama, &state));
        assert!(!response_body_is_rewritten(Protocol::OpenAiChat, &state));
        assert!(!response_body_is_rewritten(Protocol::PassThrough, &state));

        assert!(!response_body_is_rewritten(Protocol::OpenAiResponses, &state));
        state
            .tool_name_map
            .insert("namespace__tool".to_owned(), "tool".to_owned());
        assert!(response_body_is_rewritten(Protocol::OpenAiResponses, &state));
    }

    #[test]
    fn ollama_stream_guard_buffers_every_non_sse_upstream_response() {
        let mut headers = HeaderMap::new();
        assert!(should_buffer_ollama_stream_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            &headers
        ));

        assert!(should_buffer_ollama_stream_response(StatusCode::OK, &headers));

        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        assert!(should_buffer_ollama_stream_response(StatusCode::OK, &headers));

        headers.insert(CONTENT_TYPE, HeaderValue::from_static("text/plain; charset=utf-8"));
        assert!(should_buffer_ollama_stream_response(StatusCode::OK, &headers));

        headers.insert(CONTENT_TYPE, HeaderValue::from_static("text/html"));
        assert!(should_buffer_ollama_stream_response(StatusCode::OK, &headers));

        headers.insert(CONTENT_TYPE, HeaderValue::from_static("text/event-stream"));
        assert!(!should_buffer_ollama_stream_response(StatusCode::OK, &headers));

        headers.insert(CONTENT_TYPE, HeaderValue::from_static("text/event-stream; charset=utf-8"));
        assert!(!should_buffer_ollama_stream_response(StatusCode::OK, &headers));
    }

    #[test]
    fn ollama_backend_json_error_body_becomes_protocol_error() {
        let body = Bytes::from_static(
            br#"{"error":{"message":"model failed before stream start","type":"server_error"}}"#,
        );
        let out = backend_error_body_to_ollama(StatusCode::BAD_GATEWAY, &body);
        assert_eq!(
            out["error"],
            "backend returned HTTP 502: model failed before stream start"
        );
    }

    #[test]
    fn ollama_buffered_successful_non_sse_plaintext_becomes_ndjson_error_value() {
        let mut state = ResponseRewriteState::default();
        state.ollama_response_kind = Some(OllamaResponseKind::Chat);
        let body = Bytes::from_static(b"<html>llama-server fallback page</html>");

        let (status, value) =
            buffered_ollama_stream_value(StatusCode::OK, &body, &state, "local-model");

        assert_eq!(status, StatusCode::BAD_GATEWAY);
        assert_eq!(
            value["error"],
            "backend returned a non-SSE response for a streaming request: <html>llama-server fallback page</html>"
        );
    }

    #[test]
    fn ollama_sse_byte_parser_preserves_split_utf8() {
        let data = json!({
            "model": "local-model",
            "created": 0,
            "choices": [{"delta": {"content": "héllo 🚀"}}]
        })
        .to_string();
        let event = format!("data: {data}\n\n");
        let split = event.find('é').unwrap() + 1;
        let mut buffer = Vec::new();

        buffer.extend_from_slice(&event.as_bytes()[..split]);
        assert!(take_next_sse_event_bytes(&mut buffer).is_none());

        buffer.extend_from_slice(&event.as_bytes()[split..]);
        let parsed_event = take_next_sse_event_bytes(&mut buffer)
            .expect("complete event")
            .expect("valid UTF-8 once the frame is complete");

        let mut accumulator = OllamaStreamAccumulator::new(OllamaResponseKind::Chat);
        let out = translate_ollama_sse_event(&parsed_event, &mut accumulator);
        assert_eq!(out.len(), 1);
        let parsed: Value = serde_json::from_str(out[0].trim()).unwrap();
        assert_eq!(parsed["message"]["content"], "héllo 🚀");
    }

    #[test]
    fn ollama_sse_byte_parser_reports_invalid_utf8_frame() {
        let mut buffer = b"data: {\"bad\":".to_vec();
        buffer.push(0xff);
        buffer.extend_from_slice(b"}\n\n");

        let err = take_next_sse_event_bytes(&mut buffer)
            .expect("complete event")
            .expect_err("invalid UTF-8 should become an explicit proxy error");

        assert!(err.contains("not valid UTF-8"));
        assert!(buffer.is_empty());
    }

    #[test]
    fn ollama_transform_fallback_turns_raw_json_error_into_ndjson() {
        let mut accumulator = OllamaStreamAccumulator::new(OllamaResponseKind::Chat);
        let out = translate_ollama_sse_event(
            r#"{"error":{"message":"model failed before stream start"}}"#,
            &mut accumulator,
        );

        assert_eq!(out.len(), 1);
        assert!(out[0].ends_with('\n'));
        let parsed: Value = serde_json::from_str(out[0].trim()).unwrap();
        assert_eq!(parsed["error"], "model failed before stream start");
    }

    #[test]
    fn ollama_transform_fallback_turns_sse_error_data_into_ndjson() {
        let mut accumulator = OllamaStreamAccumulator::new(OllamaResponseKind::Generate);
        let out = translate_ollama_sse_event(
            r#"event: error
data: {"error":{"message":"bad model"}}"#,
            &mut accumulator,
        );

        assert_eq!(out.len(), 1);
        let parsed: Value = serde_json::from_str(out[0].trim()).unwrap();
        assert_eq!(parsed["error"], "bad model");
    }

}
