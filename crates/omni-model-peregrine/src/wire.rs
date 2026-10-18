//! The wire: building a request, and reading a reply — with no I/O.
//!
//! # Why this file exists separately from the transport
//!
//! The awkward parts of peregrine's dialect are all things that can be checked
//! without a server running:
//!
//! - `temperature` defaults to **0.0**, not OpenAI's 1.0;
//! - `top_k` is applied **before** `top_p`;
//! - only the exact string `"none"` disables tools, and `"required"` is accepted
//!   and ignored;
//! - `model`, `n`, `stop`, `seed`, `logprobs` and the penalties are **dropped
//!   on the floor**, so a client that depends on them is depending on nothing.
//!
//! If that logic lives inside the HTTP client, every one of those is a comment
//! asserting a behaviour nobody can test. So the construction is a pure
//! function over `(&Config, &ChatRequest)` returning a [`WireRequest`], the
//! [`omni-verify`] `peregrine` group checks it, and the transport is a thin
//! thing that sends bytes.
//!
//! # The dropped-fields rule
//!
//! peregrine accepts a large OpenAI superset and ignores part of it. omni sends
//! only what peregrine acts on. That is not minimalism for its own sake: a field
//! that is silently discarded is a field whose value can drift out of sync with
//! what the server does without anything failing, and a harness that believes it
//! pinned `seed` when the server dropped it has a reproducibility claim it does
//! not have.

use std::collections::BTreeMap;

use omni_core::{Context, Error};
use omni_model::{ChatReply, ChatRequest, FinishReason, Message, ToolCall, ToolSpec};
use serde_json::{Value, json};

/// A fully-formed HTTP request, ready to send and inspectable before sending.
#[derive(Debug, Clone, PartialEq)]
pub struct WireRequest {
    pub method: &'static str,
    pub url: String,
    pub headers: Vec<(String, String)>,
    /// The body as it will go on the wire.
    pub body: String,
}

/// How to reach a peregrine server.
#[derive(Debug, Clone, PartialEq)]
pub struct Config {
    /// Base URL, e.g. `http://127.0.0.1:8080`.
    pub base_url: String,
    /// Bearer token, if the server was started with one. Sent on `/v1/*` only;
    /// `/health` and `/metrics` are unauthenticated on the server side.
    pub api_key: Option<String>,
    /// The server's prompt-token ceiling. Exceeding it earns a 413, which is a
    /// hard budget error and never a retry.
    pub max_prompt_tokens: usize,
    /// Whole-request timeout, in seconds.
    pub timeout_secs: u64,
    /// The queue to enter. `Some(High)` sets `X-Peregrine-Priority: high`.
    pub priority: Option<&'static str>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            // peregrine's own default bind address.
            base_url: "http://127.0.0.1:8080".to_string(),
            api_key: None,
            // peregrine's `--max-prompt-tokens` has no default in its own
            // config, so this is a *client-side* pre-check rather than a claim
            // about the server. Being wrong here costs one wasted request.
            max_prompt_tokens: 32_768,
            timeout_secs: 900,
            priority: None,
        }
    }
}

/// Normalise a base URL: trim surrounding whitespace, then any trailing slash.
///
/// Done once, here, rather than at each join: `format!("{base}/v1/...")` against
/// a base ending in `/` silently produces `//v1`, which some servers 404 and
/// others happily accept — the kind of difference that only shows up on the
/// machine whose base URL someone typed with a trailing slash.
///
/// Whitespace first, and that order is load-bearing. A config value read from a
/// `.env` file very often arrives as `"http://h:8080/  "`, and trimming only the
/// slashes leaves both the spaces *and* — because the string no longer ends in
/// `/` — the slash as well. Trimming the space first is what makes the slash
/// visible to the second step.
#[must_use]
pub fn normalize_base(raw: &str) -> String {
    raw.trim().trim_end_matches('/').to_string()
}

/// The completion path for a base URL.
#[must_use]
pub fn completions_url(base: &str) -> String {
    format!("{}/v1/chat/completions", normalize_base(base))
}

/// Build the request for one chat turn.
///
/// # Errors
///
/// [`Error::Format`] when a tool's wire form cannot be produced, or when the
/// assembled body is not valid JSON. Neither should happen from the types, so
/// if one does the message is the finding.
pub fn build_request(cfg: &Config, req: &ChatRequest<'_>) -> Result<WireRequest, Error> {
    let messages: Vec<Value> = req.messages.iter().map(message_to_wire).collect();

    let mut body = json!({
        "messages": messages,
        // 0.0 explicitly, every time. peregrine defaults to 0.0 but a frozen
        // target that produces the same tokens twice is a property of the
        // *request*, not of a server default nobody here controls.
        "temperature": req.temperature,
        "top_p": req.top_p,
        "max_tokens": req.max_tokens,
    });

    // `top_k: 0` means off. Sending 0 unconditionally would be noise; sending
    // it only when set keeps the request minimal and matches the server's own
    // representation of "off".
    if req.top_k > 0 {
        body["top_k"] = json!(req.top_k);
    }

    // Tools, and the choice about them.
    //
    // peregrine honours only the exact string `"none"`. `"required"` is
    // accepted and ignored, which is a trap: sending `"required"` looks like
    // forcing a tool call and is not.
    if !req.tools.is_empty() {
        let schemas: Vec<Value> = req.tools.iter().map(ToolSpec::to_wire).collect();
        body["tools"] = Value::Array(schemas);
        // Advertised tools, not forced ones. omni never forces a call: a forced
        // call on a turn where the model wanted to answer is a harness lying
        // about what it observed.
        body["tool_choice"] = json!("auto");
    } else {
        body["tool_choice"] = json!("none");
    }

    if let Some(model) = req.model {
        body["model"] = json!(model);
    }

    let body_text = serde_json::to_string(&body)
        .ctx(|| "serialising the chat completion request body".to_string())?;

    let mut headers = vec![("content-type".to_string(), "application/json".to_string())];
    if let Some(key) = &cfg.api_key {
        headers.push(("authorization".to_string(), format!("Bearer {key}")));
    }
    if let Some(p) = cfg.priority {
        headers.push(("x-peregrine-priority".to_string(), p.to_ascii_lowercase()));
    }

    Ok(WireRequest {
        method: "POST",
        url: completions_url(&cfg.base_url),
        headers,
        body: body_text,
    })
}

/// One message in the OpenAI dialect.
fn message_to_wire(m: &Message) -> Value {
    match m {
        Message::System { content } => json!({"role": "system", "content": content}),
        Message::User { content } => json!({"role": "user", "content": content}),
        Message::Tool { tool_call_id, name, content } => json!({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "name": name,
            "content": content,
        }),
        Message::Assistant { content, tool_calls } => {
            let mut o = json!({"role": "assistant"});
            // `content: null` on a tool-calling turn is the documented shape and
            // is what OpenAI sends. Sending `""` instead is accepted by some
            // servers and rejected by others.
            o["content"] = match content {
                Some(c) => json!(c),
                None => Value::Null,
            };
            if !tool_calls.is_empty() {
                o["tool_calls"] = Value::Array(tool_calls.iter().map(tool_call_to_wire).collect());
            }
            o
        }
    }
}

fn tool_call_to_wire(c: &ToolCall) -> Value {
    json!({
        "id": c.id,
        "type": "function",
        // The OpenAI wire form carries arguments as a *string* containing JSON.
        // omni keeps them parsed at its own boundary, so this is where they go
        // back to being a string.
        "function": {
            "name": c.name,
            "arguments": serde_json::to_string(&c.arguments).unwrap_or_else(|_| "{}".to_string()),
        }
    })
}

/// Read one reply.
///
/// # Errors
///
/// [`Error::Format`] naming what was missing or malformed. Every failure here
/// is a reply the loop must not treat as "the model said nothing" — an empty
/// string plus `Ok` is how a harness loops forever on a server that stopped
/// answering.
pub fn parse_reply(body: &str) -> Result<ChatReply, Error> {
    let v: Value = serde_json::from_str(body)
        .ctx(|| format!("chat completion reply was not JSON ({} bytes)", body.len()))?;

    if let Some(err) = v.get("error") {
        return Err(Error::Format(format!(
            "server returned an error object: {}",
            error_message(err)
        )));
    }

    let choice = v
        .get("choices")
        .and_then(Value::as_array)
        .and_then(|c| c.first())
        .ok_or_else(|| Error::Format("chat completion reply has no choices[0]".into()))?;

    let finish = FinishReason::parse(
        choice.get("finish_reason").and_then(Value::as_str).unwrap_or("unknown"),
    );

    let raw = choice.get("message").ok_or_else(|| {
        Error::Format("chat completion choices[0] has no message".into())
    })?;

    let content = raw.get("content").and_then(Value::as_str).map(str::to_string);

    let mut tool_calls = Vec::new();
    if let Some(arr) = raw.get("tool_calls").and_then(Value::as_array) {
        for c in arr {
            let id = c.get("id").and_then(Value::as_str).unwrap_or("");
            let function = c
                .get("function")
                .ok_or_else(|| Error::Format(format!("tool call {id:?} has no function")))?;
            let name = function.get("name").and_then(Value::as_str).unwrap_or("");
            let arguments = function
                .get("arguments")
                .and_then(Value::as_str)
                .unwrap_or("{}");
            tool_calls.push(ToolCall::parse(id, name, arguments)?);
        }
    }

    let usage = v.get("usage").map(|u| omni_model::Usage {
        prompt_tokens: u.get("prompt_tokens").and_then(Value::as_u64).unwrap_or(0) as u32,
        completion_tokens: u.get("completion_tokens").and_then(Value::as_u64).unwrap_or(0) as u32,
    });

    Ok(ChatReply {
        message: Message::Assistant { content, tool_calls },
        finish_reason: finish,
        usage,
    })
}

fn error_message(err: &Value) -> String {
    let msg = err
        .get("message")
        .and_then(Value::as_str)
        .unwrap_or("(no message field)");
    let kind = err.get("type").and_then(Value::as_str).unwrap_or("unspecified");
    format!("{kind}: {msg}")
}

/// Extract the OpenAI-shaped error message from a non-2xx body, for a status
/// the caller has already classified.
///
/// Falls back to the first 200 characters of a body that is not in the
/// expected shape — a proxy in front of the server may answer with HTML, and
/// losing that would leave "HTTP 502: " with nothing to act on.
#[must_use]
pub fn extract_error(body: &str) -> String {
    let fallback = || body.chars().take(200).collect::<String>();
    serde_json::from_str::<Value>(body)
        .ok()
        // Unwrap to the inner `error` object, not the envelope: `error_message`
        // reads `message`/`type` from what it is given, and the envelope has
        // neither.
        .and_then(|v| v.get("error").cloned())
        .map_or_else(fallback, |inner| error_message(&inner))
}

// ── Streaming ───────────────────────────────────────────────────────────────
//
// # The dialect, read out of `peregrine-serve`
//
// Everything below was taken from `peregrine-serve`'s own source rather than
// from the OpenAI spec, because the two differ in ways that are invisible until
// a reader mis-handles one of them. `peregrine-serve` builds its events with
// `axum` 0.7.9, whose `Event::field` writes the name, `:`, one space, the value
// and `\n`, and whose `finalize` appends one more `\n`. So **one event is
// exactly `data: <payload>\n\n` and nothing else** — `Event::default()` sets no
// `event:`, `id:` or `retry:` field, and the only other thing on the wire is
// `KeepAlive::default()`, which is an empty *comment*, `b":\n\n"`, once every
// 15 seconds.
//
// | frame | source | bytes |
// |---|---|---|
// | opening chunk | `main.rs:1009` | `data: {"…","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}\n\n` |
// | a text delta | `main.rs:1045` | same shape, `delta.content` set |
// | a reasoning delta | `main.rs:1446` | `delta.reasoning_content` |
// | a tool call | `main.rs:1468` | `delta.tool_calls`, **one whole call per delta** |
// | a mid-stream error | `main.rs:1483` | `data: {"error":{"message":"…"}}\n\n` |
// | the closing chunk | `main.rs:1093` | `delta: {}`, `finish_reason: "stop"` or `"tool_calls"` |
// | the tail | `main.rs:1094` | `data: [DONE]\n\n` |
// | a keep-alive | `main.rs:1100` | `:\n\n` |
//
// Three properties of that table drive the design of this module, and each is a
// failure mode with a certificate:
//
// 1. **The error path returns.** `main.rs:1054` returns from the pump task
//    immediately after an error event, so a failed generation has a `finish`
//    chunk **and** a `[DONE]` neither. Accepting the end of the body as the end
//    of the turn would report a truncated answer as a finished one.
// 2. **No chunk carries `usage`.** `usage` exists in exactly one place on the
//    server, the non-streaming body at `main.rs:1252`. A stream that reported
//    zeros would be worse than one that reports nothing.
// 3. **`finish_reason` is `null` on every chunk but the last.** Reading `null`
//    as "stop" ends the turn at the first token.
//
// One caveat stated plainly rather than papered over: `peregrine-serve` has no
// test asserting the raw bytes of a stream — it asserts against `axum`'s `Sse`
// type — so the framing above is a property of `axum` 0.7.9 as much as of
// peregrine. The parser below therefore implements the WHATWG event-stream
// rules rather than only what `axum` happens to emit, and the certificates pin
// the spec behaviour using `axum`'s output as the fixture.

/// The payload a completed stream ends with.
///
/// `Event::default().data("[DONE]")` at `main.rs:1094` (live) and `main.rs:1379`
/// (memo replay). It is an ordinary `data:` payload, which is precisely why a
/// reader compares the payload instead of assuming the last frame is special:
/// anything beginning with those five characters and continuing is data.
pub const DONE_SENTINEL: &str = "[DONE]";

/// The media type `axum`'s `Sse` response carries, and the only thing that
/// distinguishes a stream from a whole JSON body.
#[must_use]
pub fn is_event_stream(content_type: &str) -> bool {
    // Parameters (`; charset=utf-8`) are allowed on the type, and the token is
    // case-insensitive per RFC 9110.
    let media = content_type.split(';').next().unwrap_or("").trim();
    media.eq_ignore_ascii_case("text/event-stream")
}

/// One decoded event from a stream, before it is understood.
///
/// Framing only. Whether a `Data` payload is a chunk, an error object or
/// nonsense is [`chunk_from_frame`]'s job, and whether the *sequence* of frames
/// is a well-formed turn is [`StreamAccumulator`]'s. Splitting the three means
/// each can fail on its own, and a certificate for one cannot be satisfied by a
/// bug in another.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SseFrame {
    /// A `data:` payload, with the multiple `data:` lines of one event joined
    /// by `\n` as the event-stream rules require.
    Data(String),
    /// The sentinel, recognised by its payload rather than by its position.
    Done,
}

/// An incremental reader for the event-stream framing.
///
/// Fed arbitrary byte slices — a socket hands over whatever arrived, which may
/// be one byte or a quarter of a character. Lines are therefore buffered as
/// **bytes** and only decoded once terminated, because decoding each slice on
/// its own would corrupt the first multi-byte character after a boundary, and a
/// corrupted character inside a JSON payload is a parse failure that reads as
/// "the server sent garbage".
#[derive(Debug, Default)]
pub struct SseParser {
    /// Bytes of a line not yet terminated by `\n`.
    buf: Vec<u8>,
    /// The current event's joined `data:` payload, each field terminated by the
    /// `\n` the rules append.
    data: String,
}

impl SseParser {
    /// A parser with nothing buffered.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Consume the next slice of the stream and return every event it completed.
    ///
    /// Returns an empty vector when the slice ended mid-line, which is the
    /// normal case and not a failure.
    ///
    /// # Errors
    ///
    /// [`Error::Format`] when a *complete* line is not valid UTF-8. A partial
    /// line is never decoded, so a split character cannot reach this path.
    pub fn feed(&mut self, bytes: &[u8]) -> Result<Vec<SseFrame>, Error> {
        self.buf.extend_from_slice(bytes);
        let mut frames = Vec::new();
        // Hand-rolled rather than a dependency: this is a scan for one byte over
        // a buffer that is usually a single line long.
        while let Some(nl) = self.buf.iter().position(|b| *b == b'\n') {
            let mut line: Vec<u8> = self.buf.drain(..=nl).collect();
            line.pop();
            // CRLF is legal and a proxy between here and there is free to
            // rewrite the line endings; the `\r` is not part of the value.
            if line.last() == Some(&b'\r') {
                line.pop();
            }
            let line = match std::str::from_utf8(&line) {
                Ok(l) => l,
                Err(e) => {
                    return Err(Error::Format(format!("a completed stream line was not valid UTF-8: {e}"))
                        .annotated(|| "decoding one event-stream line".to_string()));
                }
            };
            if let Some(frame) = self.line(line) {
                frames.push(frame);
            }
        }
        Ok(frames)
    }

    /// The end of the body. Anything still buffered was cut off mid-frame.
    ///
    /// # Errors
    ///
    /// [`Error::Format`] naming what was left over. The event-stream rules say
    /// to discard an incomplete event; this client refuses instead, because
    /// discarding a half-frame here would drop a token without saying so, and a
    /// turn that silently loses its tail is a turn nobody can debug.
    pub fn finish(&mut self) -> Result<(), Error> {
        let held_bytes = self.buf.len();
        let held_data = self.data.len();
        self.buf.clear();
        self.data.clear();
        if held_bytes > 0 || held_data > 0 {
            return Err(Error::Format(format!(
                "the stream ended mid-frame: {held_bytes} bytes of an unterminated line \
                 and {held_data} bytes of an undelivered event"
            ))
            .annotated(|| "closing the event stream".to_string()));
        }
        Ok(())
    }

    /// One line, and the event it may have completed.
    fn line(&mut self, line: &str) -> Option<SseFrame> {
        if !line.is_empty() {
            // `data` is the only field that carries payload. Comments (`:`),
            // `event`, `id`, `retry` and anything unknown are ignored — a
            // keep-alive is a comment, and treating one as data would inject
            // ":" into a JSON payload.
            let (field, value) = match line.find(':') {
                Some(i) => {
                    let rest = &line[i + 1..];
                    // Exactly one leading space is part of the framing.
                    (&line[..i], rest.strip_prefix(' ').unwrap_or(rest))
                }
                None => (line, ""),
            };
            if field == "data" {
                self.data.push_str(value);
                self.data.push('\n');
            }
            return None;
        }

        // A blank line dispatches, and dispatches nothing when this event
        // carried no data — which is what a comment line on its own is.
        if self.data.is_empty() {
            return None;
        }
        let mut payload = std::mem::take(&mut self.data);
        // Every data field appends a LF, so the final character is that LF and
        // removing it restores the joined payload. Guarded because the claim is
        // about a string that was built one field at a time, not about `\n`.
        if payload.ends_with('\n') {
            payload.pop();
        }
        Some(if payload == DONE_SENTINEL { SseFrame::Done } else { SseFrame::Data(payload) })
    }
}

/// One tool call as it arrived, before its fragments have been joined.
///
/// peregrine sends a call **whole** in a single delta (`main.rs:1461`: "this
/// emits it complete in a single delta instead"), so a fragment carrying an
/// index, an id, a name and the complete argument string is the common case. The
/// OpenAI dialect allows the same call spread over many deltas with the
/// arguments split mid-token, and any OpenAI-compatible server omni is pointed
/// at may do that, so `id` and `name` are [`Option`]s that only the first
/// fragment is required to carry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolCallFragment {
    /// The call's position, which is what correlates fragments.
    pub index: usize,
    /// The call id, when this fragment carried one.
    pub id: Option<String>,
    /// The function name, when this fragment carried one.
    pub name: Option<String>,
    /// The argument **string**, to be appended to whatever came before.
    ///
    /// A string and not a parsed object because at this point it is usually a
    /// prefix: parsing a fragment is how a client turns a split token into a
    /// JSON error halfway through a call that is merely not finished yet.
    pub arguments: String,
}

/// One thing a stream said.
///
/// A single `data:` frame can say several at once — the OpenAI dialect puts
/// `role` and the first `content` in one chunk — so [`chunk_from_frame`]
/// returns a list rather than one event.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StreamChunk {
    /// `data: [DONE]`.
    Done,
    /// The opening `delta.role`.
    Role(String),
    /// A piece of `delta.content`.
    Text(String),
    /// A piece of `delta.reasoning_content`.
    ///
    /// Delivered because the caller may want it, and deliberately **not** folded
    /// into [`ChatReply`]: `omni_model::ChatReply` has nowhere to put it, and the
    /// one place it must not go is `content` — a harness that recorded the
    /// model's scratchpad as its answer would score its own notes. The
    /// non-streaming parser drops it for the same reason.
    Reasoning(String),
    /// One tool-call fragment.
    ToolCall(ToolCallFragment),
    /// `finish_reason`, on the closing chunk only.
    Finish(FinishReason),
    /// The server's own mid-stream error: `{"error":{"message":"…"}}`.
    ///
    /// A variant rather than a bare [`Error`] because reading a frame and
    /// judging the stream are different acts; [`StreamAccumulator::push`] is
    /// where it becomes an error.
    ServerError(String),
}

/// Understand one event.
///
/// # Errors
///
/// [`Error::Format`] for a payload that is not JSON, and for a chunk that
/// carries neither `choices[0]` nor a `delta`. Both are refused rather than
/// skipped: a skipped chunk can be the one carrying `finish_reason`, and a turn
/// whose finish reason went missing looks exactly like a turn that finished
/// cleanly.
pub fn chunk_from_frame(frame: &SseFrame) -> Result<Vec<StreamChunk>, Error> {
    let SseFrame::Data(payload) = frame else {
        return Ok(vec![StreamChunk::Done]);
    };
    let v: Value = serde_json::from_str(payload)
        .ctx(|| format!("a streamed `data:` payload was not JSON ({} bytes)", payload.len()))?;

    // `main.rs:1483`: the error event is an ordinary data frame whose only key
    // is `error`, so it is recognised before anything else is expected of it.
    if let Some(err) = v.get("error") {
        return Ok(vec![StreamChunk::ServerError(error_message(err))]);
    }

    let choice = v
        .get("choices")
        .and_then(Value::as_array)
        .and_then(|c| c.first())
        .ok_or_else(|| {
            Error::Format(format!("a streamed chunk carried no choices[0]: {payload}"))
                .annotated(|| "reading one streamed chunk".to_string())
        })?;
    let delta = choice.get("delta").ok_or_else(|| {
        Error::Format(format!("a streamed chunk carried no delta: {payload}"))
            .annotated(|| "reading one streamed chunk".to_string())
    })?;

    let mut out = Vec::new();
    if let Some(role) = delta.get("role").and_then(Value::as_str) {
        out.push(StreamChunk::Role(role.to_string()));
    }
    if let Some(r) = delta.get("reasoning_content").and_then(Value::as_str) {
        out.push(StreamChunk::Reasoning(r.to_string()));
    }
    if let Some(calls) = delta.get("tool_calls").and_then(Value::as_array) {
        for c in calls {
            out.push(StreamChunk::ToolCall(tool_call_fragment(c)?));
        }
    }
    if let Some(content) = delta.get("content").and_then(Value::as_str) {
        out.push(StreamChunk::Text(content.to_string()));
    }
    // `main.rs:1441` writes `"finish_reason": null` on every chunk but the
    // last. Null is not a finish reason and must not become one — see
    // `StreamChunk::Finish`.
    match choice.get("finish_reason") {
        None | Some(Value::Null) => {}
        Some(Value::String(s)) => out.push(StreamChunk::Finish(FinishReason::parse(s))),
        Some(other) => {
            return Err(Error::Format(format!("finish_reason was {other}, expected a string or null"))
                .annotated(|| "reading one streamed chunk".to_string()));
        }
    }
    Ok(out)
}

/// Read one tool-call fragment, or refuse it.
///
/// An empty or absent `id`/`name` becomes `None` rather than `Some("")`,
/// because OpenAI's later fragments carry neither and a client that wrote the
/// empty string would erase what the first fragment established.
fn tool_call_fragment(c: &Value) -> Result<ToolCallFragment, Error> {
    let index = c
        .get("index")
        .and_then(Value::as_u64)
        .and_then(|i| usize::try_from(i).ok())
        .ok_or_else(|| {
            Error::Format(format!("a streamed tool call carried no usable index: {c}"))
                .annotated(|| "correlating one streamed tool call".to_string())
        })?;
    let function = c.get("function");
    Ok(ToolCallFragment {
        index,
        id: present_text(c.get("id")),
        name: function.and_then(|f| present_text(f.get("name"))),
        arguments: function
            .and_then(|f| f.get("arguments"))
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string(),
    })
}

fn present_text(v: Option<&Value>) -> Option<String> {
    v.and_then(Value::as_str).filter(|s| !s.is_empty()).map(str::to_string)
}

/// A tool call being assembled from its fragments.
#[derive(Debug, Default)]
struct PartialCall {
    id: String,
    name: String,
    arguments: String,
}

/// Folds a stream into one [`ChatReply`].
///
/// The accumulating half, and the place every "a stream that stops early is not
/// a finished turn" rule lives. It holds no socket, so every property of it is
/// checkable from a literal event sequence.
#[derive(Debug, Default)]
pub struct StreamAccumulator {
    /// `None` until a non-empty text delta arrives, and deliberately distinct
    /// from `Some("")`: "said nothing" and "said the empty string" are different
    /// answers, exactly as they are in [`omni_model::Message`].
    content: Option<String>,
    /// Keyed by the call's own `index`, so fragments may arrive in any order and
    /// calls come out in call order rather than arrival order.
    calls: BTreeMap<usize, PartialCall>,
    finish: Option<FinishReason>,
    done: bool,
    poisoned: bool,
    /// Events seen, so the error messages can say how far the stream got.
    events: usize,
}

impl StreamAccumulator {
    /// An accumulator with nothing seen.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Absorb one chunk.
    ///
    /// # Errors
    ///
    /// [`Error::Format`] for an event after the `[DONE]` sentinel, a second
    /// finish reason, a fragment that contradicts an earlier one for the same
    /// call, or the server's own mid-stream error. Once any of those happens
    /// the accumulator refuses everything afterwards, so a caller that keeps
    /// pushing after ignoring an error cannot assemble a turn out of the debris.
    pub fn push(&mut self, chunk: &StreamChunk) -> Result<(), Error> {
        match self.absorb(chunk) {
            Ok(()) => {
                self.events += 1;
                Ok(())
            }
            Err(e) => {
                self.poisoned = true;
                Err(e)
            }
        }
    }

    fn absorb(&mut self, chunk: &StreamChunk) -> Result<(), Error> {
        if self.poisoned {
            return Err(Error::Format(
                "this stream already failed; its events cannot be assembled into a reply".into(),
            ));
        }
        if self.done {
            return Err(Error::Format(
                "the stream carried an event after its [DONE] sentinel".into(),
            ));
        }
        match chunk {
            StreamChunk::Done => {
                self.done = true;
                return Ok(());
            }
            StreamChunk::ServerError(msg) => {
                return Err(Error::Format(format!(
                    "the stream reported an error mid-flight: {msg}"
                ))
                .annotated(|| "assembling a streamed completion".to_string()));
            }
            StreamChunk::Role(_) => {}
            StreamChunk::Reasoning(_) => {
                // See `StreamChunk::Reasoning`: it reaches the caller through
                // the stream and is not folded into the reply.
            }
            StreamChunk::Text(t) => match &mut self.content {
                Some(c) => c.push_str(t),
                None if t.is_empty() => {}
                None => self.content = Some(t.to_string()),
            },
            StreamChunk::ToolCall(f) => self.merge_call(f)?,
            StreamChunk::Finish(f) => match self.finish {
                Some(previous) => {
                    return Err(Error::Format(format!(
                        "the stream reported two finish reasons, {previous:?} then {f:?}"
                    )));
                }
                None => self.finish = Some(*f),
            },
        }
        Ok(())
    }

    /// Join one fragment into its call, refusing anything that contradicts what
    /// the call already established.
    fn merge_call(&mut self, f: &ToolCallFragment) -> Result<(), Error> {
        let slot = self.calls.entry(f.index).or_default();
        if let Some(id) = &f.id {
            if !slot.id.is_empty() && &slot.id != id {
                return Err(Error::Format(format!(
                    "tool call {} carried two ids, {:?} then {id:?}",
                    f.index, slot.id
                )));
            }
            slot.id.clone_from(id);
        }
        if let Some(name) = &f.name {
            if !slot.name.is_empty() && &slot.name != name {
                return Err(Error::Format(format!(
                    "tool call {} carried two names, {:?} then {name:?}",
                    f.index, slot.name
                )));
            }
            slot.name.clone_from(name);
        }
        slot.arguments.push_str(&f.arguments);
        Ok(())
    }

    /// The reply, once the stream has ended.
    ///
    /// # Errors
    ///
    /// [`Error::Format`] when the stream ended without its `[DONE]`, when it
    /// reached `[DONE]` without a finish reason, when the finish reason and the
    /// calls it announced disagree, or when a call's joined arguments are not a
    /// JSON object.
    pub fn finish(&self) -> Result<ChatReply, Error> {
        if self.poisoned {
            return Err(Error::Format("this stream already failed".into()));
        }
        if !self.done {
            return Err(Error::Format(format!(
                "the stream ended after {} events without its [DONE] sentinel; a stream \
                 that stops early is a truncated turn, not a finished one",
                self.events
            ))
            .annotated(|| "assembling a streamed completion".to_string()));
        }
        let finish = self.finish.ok_or_else(|| {
            Error::Format(format!(
                "the stream reached [DONE] after {} events but never reported a finish \
                 reason, so a chunk was lost",
                self.events
            ))
            .annotated(|| "assembling a streamed completion".to_string())
        })?;

        // `main.rs:1091`: the finish reason is derived from whether any call was
        // emitted, so the two cannot disagree. When they do, one of them is
        // about to be believed and it is the wrong one — a `tool_calls` turn
        // with nothing to run, or a `stop` turn whose calls were dropped.
        if self.calls.is_empty() && finish.wants_tools() {
            return Err(Error::Format(
                "the stream finished with `tool_calls` but carried no tool call to run".into(),
            ));
        }
        if !self.calls.is_empty() && !finish.wants_tools() {
            return Err(Error::Format(format!(
                "the stream finished with `{}` but carried {} tool call(s)",
                finish.as_str(),
                self.calls.len()
            )));
        }

        let mut tool_calls = Vec::with_capacity(self.calls.len());
        for (index, slot) in &self.calls {
            if slot.name.is_empty() {
                return Err(Error::Format(format!(
                    "tool call {index} closed with no function name, so nothing could be run"
                )));
            }
            // Parsed once, here, after every fragment has arrived. A call whose
            // id is empty is left alone: peregrine replays tool results
            // positionally, so the id is decoration rather than the correlation
            // key (`omni_model::Message::Tool`).
            tool_calls.push(ToolCall::parse(&slot.id, &slot.name, &slot.arguments)?);
        }

        Ok(ChatReply {
            message: Message::Assistant { content: self.content.clone(), tool_calls },
            finish_reason: finish,
            // Never fabricated. `usage` appears once in all of `peregrine-serve`
            // — the non-streaming body at `main.rs:1252` — so a stream has none
            // to report. `None` says "this path does not count tokens", which a
            // caller can act on; zeroes would read as "this was free".
            usage: None,
        })
    }
}

/// Build the request for one chat turn, asking for a stream.
///
/// The same request [`build_request`] builds, plus `"stream": true` and an
/// `Accept`. The field is inserted into the parsed body rather than by a second
/// builder, because two body builders are two things to keep in sync with the
/// dialect and only one of them would be updated when the dialect moves.
///
/// `stream` is absent rather than `false` on the non-streaming path, matching
/// the server's `#[serde(default)]`.
///
/// # Errors
///
/// Whatever [`build_request`] raises, plus [`Error::Format`] if the assembled
/// body cannot be read back or re-serialised — which would mean the body this
/// function just produced is not JSON, and the message is the finding.
pub fn chat_stream_request(cfg: &Config, req: &ChatRequest<'_>) -> Result<WireRequest, Error> {
    let mut wire = build_request(cfg, req)?;
    let mut body: Value = serde_json::from_str(&wire.body)
        .ctx(|| "reading the assembled chat completion body back to add `stream`".to_string())?;
    body["stream"] = Value::Bool(true);
    wire.body = serde_json::to_string(&body)
        .ctx(|| "serialising the streaming chat completion request body".to_string())?;
    wire.headers.push(("accept".to_string(), "text/event-stream".to_string()));
    Ok(wire)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn cfg() -> Config {
        Config { base_url: "http://127.0.0.1:8132".into(), ..Config::default() }
    }

    fn body_of(w: &WireRequest) -> Value {
        serde_json::from_str(&w.body).unwrap_or(Value::Null)
    }

    #[test]
    fn a_trailing_slash_in_the_base_url_does_not_double_it() {
        // `//v1` is accepted by some servers and 404s on others, which is the
        // kind of difference that only appears on the machine whose config
        // someone typed with a trailing slash.
        assert_eq!(completions_url("http://h:8080/"), "http://h:8080/v1/chat/completions");
        assert_eq!(completions_url("http://h:8080"), "http://h:8080/v1/chat/completions");
        assert_eq!(normalize_base("http://h:8080///"), "http://h:8080");
    }

    #[test]
    fn temperature_is_sent_explicitly_rather_than_left_to_the_server() -> Result<(), Error> {
        let msgs = [Message::user("hi")];
        let req = ChatRequest::new(&msgs);
        let wire = build_request(&cfg(), &req)?;
        let body = body_of(&wire);
        // 0.0, not OpenAI's 1.0. Left implicit, a server default change would
        // silently produce a different agent.
        assert_eq!(body["temperature"], json!(0.0));
        assert_eq!(body["top_p"], json!(0.95));
        Ok(())
    }

    #[test]
    fn top_k_is_omitted_when_off_and_sent_when_set() -> Result<(), Error> {
        let msgs = [Message::user("hi")];
        let off = ChatRequest::new(&msgs);
        assert!(body_of(&build_request(&cfg(), &off)?).get("top_k").is_none());

        let on = ChatRequest { top_k: 40, ..ChatRequest::new(&msgs) };
        assert_eq!(body_of(&build_request(&cfg(), &on)?)["top_k"], json!(40));
        Ok(())
    }

    #[test]
    fn no_tools_means_tool_choice_none() -> Result<(), Error> {
        let msgs = [Message::user("hi")];
        let req = ChatRequest::new(&msgs);
        let body = body_of(&build_request(&cfg(), &req)?);
        assert_eq!(body["tool_choice"], json!("none"));
        assert!(body.get("tools").is_none());
        Ok(())
    }

    #[test]
    fn tools_are_advertised_but_never_forced() -> Result<(), Error> {
        let msgs = [Message::user("hi")];
        let tools = [ToolSpec::new("echo", "echo", json!({"type":"object"}))?];
        let req = ChatRequest::new(&msgs).with_tools(&tools);
        let body = body_of(&build_request(&cfg(), &req)?);
        assert_eq!(body["tool_choice"], json!("auto"));
        assert_eq!(body["tools"][0]["type"], "function");
        assert_eq!(body["tools"][0]["function"]["name"], "echo");
        // "required" would look like forcing a call; peregrine ignores it, and
        // a forced call on a turn where the model wanted to answer is a harness
        // lying about what it observed.
        assert_ne!(body["tool_choice"], json!("required"));
        Ok(())
    }

    #[test]
    fn a_tool_calling_assistant_turn_sends_null_content() -> Result<(), Error> {
        let call = ToolCall::new("c1", "shell", json!({"command":"ls"}))?;
        let msgs = [Message::assistant_tool_calls(vec![call])];
        let body = body_of(&build_request(&cfg(), &ChatRequest::new(&msgs))?);
        assert!(body["messages"][0]["content"].is_null(), "documented shape is null, not empty string");
        assert_eq!(body["messages"][0]["tool_calls"][0]["type"], "function");
        // Arguments go back to being a string on the wire.
        assert!(body["messages"][0]["tool_calls"][0]["function"]["arguments"].is_string());
        Ok(())
    }

    #[test]
    fn only_fields_the_server_acts_on_are_sent() -> Result<(), Error> {
        let msgs = [Message::user("hi")];
        let req = ChatRequest::new(&msgs);
        let body = body_of(&build_request(&cfg(), &req)?);
        // peregrine accepts and discards these. Sending them would create the
        // appearance of pinning behaviour it does not have.
        for dropped in ["n", "stop", "seed", "logprobs", "presence_penalty", "frequency_penalty", "response_format"] {
            assert!(body.get(dropped).is_none(), "{dropped} must not be sent: the server discards it");
        }
        Ok(())
    }

    #[test]
    fn auth_and_priority_become_headers() -> Result<(), Error> {
        let c = Config {
            api_key: Some("secret-token".into()),
            priority: Some("high"),
            ..cfg()
        };
        let msgs = [Message::user("hi")];
        let wire = build_request(&c, &ChatRequest::new(&msgs))?;
        let auth = wire.headers.iter().find(|(k, _)| k == "authorization").map(|(_, v)| v.clone());
        assert_eq!(auth.as_deref(), Some("Bearer secret-token"));
        let prio = wire.headers.iter().find(|(k, _)| k == "x-peregrine-priority").map(|(_, v)| v.clone());
        assert_eq!(prio.as_deref(), Some("high"));
        Ok(())
    }

    #[test]
    fn a_reply_with_a_tool_call_parses() -> Result<(), Error> {
        let body = json!({
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "shell", "arguments": "{\"command\":\"ls\"}"}
                    }]
                },
                "finish_reason": "tool_calls"
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 4}
        })
        .to_string();
        let reply = parse_reply(&body)?;
        assert_eq!(reply.finish_reason, FinishReason::ToolCalls);
        assert_eq!(reply.text(), None, "content null must not become empty string");
        assert_eq!(reply.message.tool_calls().len(), 1);
        assert_eq!(reply.message.tool_calls()[0].arg_str("command")?, "ls");
        assert_eq!(reply.usage.map(|u| u.total()), Some(14));
        Ok(())
    }

    #[test]
    fn a_truncated_tool_call_is_never_reported_as_a_finished_answer() -> Result<(), Error> {
        // The failure this prevents: a turn that ran out of budget mid-JSON
        // recorded as "the model said stop", which reads as "no tool needed".
        let body = json!({
            "choices": [{
                "message": {"role": "assistant", "content": "{\"comm"},
                "finish_reason": "length"
            }]
        })
        .to_string();
        let reply = parse_reply(&body)?;
        assert_eq!(reply.finish_reason, FinishReason::Length);
        assert!(!reply.finish_reason.wants_tools());
        Ok(())
    }

    #[test]
    fn a_server_error_object_is_an_error_not_an_empty_reply() {
        let body = json!({"error": {"message": "queue full", "type": "overloaded_error"}}).to_string();
        let Err(err) = parse_reply(&body) else {
            return;
        };
        assert!(err.to_string().contains("overloaded_error"));
        assert!(err.to_string().contains("queue full"));
    }

    #[test]
    fn malformed_replies_are_refused_rather_than_answered_as_empty() {
        // An empty string plus Ok is how a harness loops forever on a server
        // that has stopped answering.
        assert!(parse_reply("not json").is_err());
        assert!(parse_reply("{}").is_err());
        assert!(parse_reply(r#"{"choices":[]}"#).is_err());
        assert!(parse_reply(r#"{"choices":[{"finish_reason":"stop"}]}"#).is_err());
    }

    #[test]
    fn a_non_object_argument_bag_is_refused() {
        let body = json!({
            "choices": [{
                "message": {"role":"assistant","tool_calls":[{
                    "id":"c","type":"function",
                    "function":{"name":"t","arguments":"[1,2]"}
                }]},
                "finish_reason":"tool_calls"
            }]
        })
        .to_string();
        assert!(parse_reply(&body).is_err());
    }

    // ── Streaming fixtures ──────────────────────────────────────────────────
    //
    // Transcribed from what `axum` 0.7.9 writes, not generated by this module:
    // `Event::field` emits `name`, `:`, one space, the value and `\n`, and
    // `finalize` appends one more `\n`, so an event is `data: <payload>\n\n`.
    // A fixture produced by our own framing would prove the parser agrees with
    // the framing, which is the thing in question.

    /// `axum`'s framing for one event, spelled out as literal bytes.
    fn frame(payload: &str) -> String {
        format!("data: {payload}\n\n")
    }

    fn chunk(delta: Value, finish: Value) -> String {
        frame(&json!({
            "id": "chatcmpl-7",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "m",
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]
        })
        .to_string())
    }

    /// The exact stream a greedy text turn produces, as `main.rs` writes it.
    fn text_stream() -> String {
        let mut s = chunk(json!({"role": "assistant"}), Value::Null);
        s.push_str(&chunk(json!({"content": "2 + 2 "}), Value::Null));
        s.push_str(&chunk(json!({"content": "= 4"}), Value::Null));
        s.push_str(&chunk(json!({}), json!("stop")));
        s.push_str(&frame(DONE_SENTINEL));
        s
    }

    fn collect_stream(body: &str) -> Result<ChatReply, Error> {
        let mut p = SseParser::new();
        let mut acc = StreamAccumulator::new();
        for f in p.feed(body.as_bytes())? {
            for c in chunk_from_frame(&f)? {
                acc.push(&c)?;
            }
        }
        p.finish()?;
        acc.finish()
    }

    fn chunks_of(body: &str) -> Result<Vec<StreamChunk>, Error> {
        let mut p = SseParser::new();
        let mut out = Vec::new();
        for f in p.feed(body.as_bytes())? {
            out.extend(chunk_from_frame(&f)?);
        }
        p.finish()?;
        Ok(out)
    }

    #[test]
    fn a_stream_read_one_byte_at_a_time_yields_the_same_frames() -> Result<(), Error> {
        // A socket hands over whatever arrived. The `→` is three bytes and the
        // JSON payload is not escaped, so a boundary can land inside it; decoding
        // each slice separately would corrupt that character and turn a working
        // stream into a JSON error.
        let body = chunk(json!({"role": "assistant"}), Value::Null)
            + &chunk(json!({"content": "x → y"}), Value::Null)
            + &frame(DONE_SENTINEL);
        let mut whole = SseParser::new();
        let expected = whole.feed(body.as_bytes())?;

        let mut dribbled = SseParser::new();
        let mut got = Vec::new();
        for b in body.as_bytes() {
            got.extend(dribbled.feed(std::slice::from_ref(b))?);
        }
        assert_eq!(got, expected);
        assert_eq!(expected.len(), 3, "two chunks and the sentinel");
        dribbled.finish()?;
        Ok(())
    }

    #[test]
    fn a_keep_alive_is_a_comment_and_not_an_event() -> Result<(), Error> {
        // `KeepAlive::default()` is `b":\n\n"`, one every 15 seconds. A reader
        // that treated it as a payload would try to parse ":" as JSON.
        let with = format!(":\n\n{}\n:\n\n{}\n:\n\n", frame("[DONE]"), frame("[DONE]"));
        let without = format!("{}\n{}", frame("[DONE]"), frame("[DONE]"));
        let mut a = SseParser::new();
        let mut b = SseParser::new();
        assert_eq!(a.feed(with.as_bytes())?, b.feed(without.as_bytes())?);
        a.finish()?;
        b.finish()?;
        Ok(())
    }

    #[test]
    fn the_data_lines_of_one_event_join_with_a_newline() -> Result<(), Error> {
        // `axum`'s `Event::data` splits a multi-line payload across `data:`
        // fields, and the rules say a reader rejoins them with `\n`.
        let mut p = SseParser::new();
        let frames = p.feed(b"data: {\"a\":\ndata: 1}\n\n")?;
        assert_eq!(frames, vec![SseFrame::Data("{\"a\":\n1}".to_string())]);
        p.finish()?;

        let mut one = SseParser::new();
        assert_eq!(one.feed(b"data: {\"a\":1}\n\n")?, vec![SseFrame::Data("{\"a\":1}".to_string())]);
        one.finish()?;
        Ok(())
    }

    #[test]
    fn crlf_line_endings_parse_like_lf_ones() -> Result<(), Error> {
        let mut p = SseParser::new();
        let frames = p.feed(b"data: [DONE]\r\n\r\n")?;
        assert_eq!(frames, vec![SseFrame::Done]);
        p.finish()?;
        Ok(())
    }

    #[test]
    fn the_sentinel_is_recognised_by_its_payload_not_its_position() -> Result<(), Error> {
        let mut p = SseParser::new();
        assert_eq!(p.feed(b"data: [DONE]\n\n")?, vec![SseFrame::Done]);
        // A payload that merely starts with the sentinel is data. Matching on a
        // prefix would delete a chunk.
        assert_eq!(
            p.feed(b"data: [DONE] and then some\n\n")?,
            vec![SseFrame::Data("[DONE] and then some".to_string())]
        );
        p.finish()?;
        Ok(())
    }

    #[test]
    fn a_stream_cut_off_mid_frame_is_refused_rather_than_halved() -> Result<(), Error> {
        let mut p = SseParser::new();
        p.feed(b"data: {\"a\":")?;
        let Err(err) = p.finish() else {
            return Err(Error::Format("a half-frame must not close cleanly".into()));
        };
        assert!(err.to_string().contains("mid-frame"), "{err}");

        // A body that ended on an event boundary is not cut off, even with no
        // sentinel: that is the accumulator's judgement, not the framing's.
        let mut clean = SseParser::new();
        clean.feed(frame(DONE_SENTINEL).as_bytes())?;
        clean.finish()?;
        Ok(())
    }

    #[test]
    fn an_undecodable_line_is_refused_rather_than_decoded_lossily() -> Result<(), Error> {
        // The tempting alternative is `String::from_utf8_lossy`, which keeps
        // reading and substitutes U+FFFD. A replacement character inside a JSON
        // payload still parses, so the turn would assemble and the harness would
        // record a mangled byte as something the model said.
        let mut p = SseParser::new();
        let Err(err) = p.feed(b"data: {\"content\":\"a\xffb\"}\n\n") else {
            return Err(Error::Format("an undecodable line must not decode lossy".into()));
        };
        assert!(err.to_string().contains("UTF-8"), "{err}");

        // The control, going the other way: the same characters, complete, are
        // delivered exactly — so the refusal is about bytes that cannot be
        // decoded and not about refusing everything above ASCII.
        let mut ok = SseParser::new();
        assert_eq!(
            ok.feed("data: → 🦀\n\n".as_bytes())?,
            vec![SseFrame::Data("→ 🦀".to_string())]
        );
        ok.finish()?;
        Ok(())
    }

    #[test]
    fn the_opening_chunk_is_a_role_and_the_closing_chunk_is_a_finish() -> Result<(), Error> {
        let chunks = chunks_of(&text_stream())?;
        assert_eq!(chunks[0], StreamChunk::Role("assistant".to_string()));
        // `finish_reason: null` on the content chunks must produce no Finish.
        assert_eq!(
            chunks.iter().filter(|c| matches!(c, StreamChunk::Finish(_))).count(),
            1,
            "exactly one finish, not one per chunk"
        );
        assert_eq!(chunks.last(), Some(&StreamChunk::Done));
        Ok(())
    }

    #[test]
    fn a_null_finish_reason_is_not_the_end_of_the_stream() -> Result<(), Error> {
        // The failure this prevents: reading `null` as "stop" ends the turn at
        // the first token and reports a two-character answer as complete.
        let reply = collect_stream(&text_stream())?;
        assert_eq!(reply.text(), Some("2 + 2 = 4"));
        assert_eq!(reply.finish_reason, FinishReason::Stop);
        Ok(())
    }

    #[test]
    fn a_finish_reason_the_stream_did_name_is_kept() -> Result<(), Error> {
        let body = chunk(json!({"content": "x"}), Value::Null)
            + &chunk(json!({}), json!("length"))
            + &frame(DONE_SENTINEL);
        assert_eq!(collect_stream(&body)?.finish_reason, FinishReason::Length);
        Ok(())
    }

    #[test]
    fn a_stream_ending_without_its_sentinel_is_not_a_finished_turn() -> Result<(), Error> {
        // Exactly what `main.rs:1054` leaves on the wire: an error, then the
        // socket closes. Treating that as a completed turn is how a truncated
        // answer gets recorded as the model's answer.
        let truncated = chunk(json!({"role": "assistant"}), Value::Null)
            + &chunk(json!({"content": "half a sen"}), Value::Null);
        let Err(err) = collect_stream(&truncated) else {
            return Err(Error::Format("a stream that stops early must not assemble".into()));
        };
        assert!(err.to_string().contains("[DONE]"), "{err}");
        Ok(())
    }

    #[test]
    fn a_stream_that_reaches_done_without_a_finish_reason_is_refused() -> Result<(), Error> {
        let body = chunk(json!({"role": "assistant"}), Value::Null) + &frame(DONE_SENTINEL);
        let Err(err) = collect_stream(&body) else {
            return Err(Error::Format("a lost finish chunk must not pass as finished".into()));
        };
        assert!(err.to_string().contains("finish reason"), "{err}");
        Ok(())
    }

    #[test]
    fn an_event_after_the_sentinel_is_refused() -> Result<(), Error> {
        let body = frame(DONE_SENTINEL) + &chunk(json!({"content": "late"}), Value::Null);
        let Err(err) = collect_stream(&body) else {
            return Err(Error::Format("content after [DONE] must not be assembled".into()));
        };
        assert!(err.to_string().contains("after its [DONE]"), "{err}");
        Ok(())
    }

    #[test]
    fn a_server_error_mid_stream_is_an_error_and_not_a_turn() -> Result<(), Error> {
        // `sse_error` at `main.rs:1483`.
        let body = chunk(json!({"role": "assistant"}), Value::Null)
            + &frame(&json!({"error": {"message": "kv page fault"}}).to_string());
        let Err(err) = collect_stream(&body) else {
            return Err(Error::Format("a mid-stream error must not become a turn".into()));
        };
        assert!(err.to_string().contains("kv page fault"), "the server's own words: {err}");
        Ok(())
    }

    #[test]
    fn a_chunk_with_nothing_to_say_is_refused_rather_than_skipped() -> Result<(), Error> {
        // A silently skipped chunk can be the one carrying the finish reason,
        // and a turn that lost its finish reason looks exactly like a clean one.
        for body in [
            frame(&json!({"id": "c", "object": "chat.completion.chunk"}).to_string()),
            frame(&json!({"choices": []}).to_string()),
            frame(&json!({"choices": [{"index": 0, "finish_reason": null}]}).to_string()),
        ] {
            assert!(collect_stream(&body).is_err(), "{body}");
        }
        Ok(())
    }

    #[test]
    fn reasoning_arrives_in_its_own_field_and_is_not_the_answer() -> Result<(), Error> {
        let body = chunk(json!({"reasoning_content": "2 + 2 is 4"}), Value::Null)
            + &chunk(json!({"content": "4"}), Value::Null)
            + &chunk(json!({}), json!("stop"))
            + &frame(DONE_SENTINEL);
        let chunks = chunks_of(&body)?;
        assert!(chunks.contains(&StreamChunk::Reasoning("2 + 2 is 4".to_string())));
        let reply = collect_stream(&body)?;
        assert_eq!(reply.text(), Some("4"), "the scratchpad is not the answer");
        Ok(())
    }

    #[test]
    fn a_whole_tool_call_and_the_same_call_in_fragments_agree() -> Result<(), Error> {
        // peregrine emits the whole call at once (`main.rs:1468`); OpenAI
        // splits the arguments across deltas. Both must reach the same reply.
        let whole = chunk(json!({"role": "assistant"}), Value::Null)
            + &chunk(
                json!({"tool_calls": [{
                    "index": 0, "id": "call_7_0", "type": "function",
                    "function": {"name": "shell", "arguments": "{\"command\":\"ls -la\"}"}
                }]}),
                Value::Null,
            )
            + &chunk(json!({}), json!("tool_calls"))
            + &frame(DONE_SENTINEL);

        let fragmented = chunk(json!({"role": "assistant"}), Value::Null)
            + &chunk(
                json!({"tool_calls": [{
                    "index": 0, "id": "call_7_0", "type": "function",
                    "function": {"name": "shell", "arguments": ""}
                }]}),
                Value::Null,
            )
            + &chunk(
                json!({"tool_calls": [{"index": 0, "function": {"arguments": "{\"comm"}}]}),
                Value::Null,
            )
            + &chunk(
                json!({"tool_calls": [{"index": 0, "function": {"arguments": "and\":\"ls -la\"}"}}]}),
                Value::Null,
            )
            + &chunk(json!({}), json!("tool_calls"))
            + &frame(DONE_SENTINEL);

        let a = collect_stream(&whole)?;
        let b = collect_stream(&fragmented)?;
        assert_eq!(a, b, "a split call must reassemble to the whole one");
        assert_eq!(a.message.tool_calls()[0].arg_str("command")?, "ls -la");
        // Nothing was said in words, so nothing is reported as said.
        assert_eq!(a.text(), None);
        Ok(())
    }

    #[test]
    fn a_fragment_without_an_id_does_not_erase_the_one_before_it() -> Result<(), Error> {
        let body = chunk(
            json!({"tool_calls": [{
                "index": 0, "id": "call_7_0", "type": "function",
                "function": {"name": "shell", "arguments": "{\"a\":1}"}
            }]}),
            Value::Null,
        )
            + &chunk(json!({"tool_calls": [{"index": 0, "function": {"arguments": ""}}]}), Value::Null)
            + &chunk(json!({}), json!("tool_calls"))
            + &frame(DONE_SENTINEL);
        assert_eq!(collect_stream(&body)?.message.tool_calls()[0].id, "call_7_0");
        Ok(())
    }

    #[test]
    fn fragments_that_contradict_their_own_call_are_refused() -> Result<(), Error> {
        // The other direction: agreement is not the check. Two different names
        // for one call means the fragments cannot be joined, and picking one
        // silently would run the wrong tool.
        let body = chunk(
            json!({"tool_calls": [{
                "index": 0, "id": "c0", "type": "function",
                "function": {"name": "shell", "arguments": "{}"}
            }]}),
            Value::Null,
        )
            + &chunk(
                json!({"tool_calls": [{"index": 0, "id": "c1", "function": {"arguments": ""}}]}),
                Value::Null,
            )
            + &chunk(json!({}), json!("tool_calls"))
            + &frame(DONE_SENTINEL);
        let Err(err) = collect_stream(&body) else {
            return Err(Error::Format("two ids for one call must be refused".into()));
        };
        assert!(err.to_string().contains("two ids"), "{err}");
        Ok(())
    }

    #[test]
    fn a_fragment_with_no_index_cannot_be_correlated_and_is_refused() -> Result<(), Error> {
        let body = chunk(
            json!({"tool_calls": [{
                "id": "c0", "type": "function",
                "function": {"name": "shell", "arguments": "{}"}
            }]}),
            Value::Null,
        );
        assert!(collect_stream(&body).is_err());
        Ok(())
    }

    #[test]
    fn truncated_arguments_are_refused_rather_than_defaulted_to_empty() -> Result<(), Error> {
        // `{}` here would be a tool call that runs nothing and reports success.
        let body = chunk(
            json!({"tool_calls": [{
                "index": 0, "id": "c0", "type": "function",
                "function": {"name": "shell", "arguments": "{\"comm"}
            }]}),
            Value::Null,
        )
            + &chunk(json!({}), json!("tool_calls"))
            + &frame(DONE_SENTINEL);
        let Err(err) = collect_stream(&body) else {
            return Err(Error::Format("truncated arguments must not parse".into()));
        };
        assert!(err.to_string().contains("arguments"), "{err}");
        Ok(())
    }

    #[test]
    fn a_non_object_argument_bag_is_refused_on_the_streaming_path_too() -> Result<(), Error> {
        let body = chunk(
            json!({"tool_calls": [{
                "index": 0, "id": "c0", "type": "function",
                "function": {"name": "shell", "arguments": "[1,2]"}
            }]}),
            Value::Null,
        )
            + &chunk(json!({}), json!("tool_calls"))
            + &frame(DONE_SENTINEL);
        assert!(collect_stream(&body).is_err());
        Ok(())
    }

    #[test]
    fn a_finish_reason_that_contradicts_the_calls_is_refused() -> Result<(), Error> {
        // `main.rs:1091` derives the finish reason from whether any call was
        // emitted, so a `tool_calls` turn with nothing to run is a lost chunk,
        // not a turn the loop can act on.
        let promised_none = chunk(json!({}), json!("tool_calls")) + &frame(DONE_SENTINEL);
        assert!(collect_stream(&promised_none).is_err());

        let promised_some = chunk(
            json!({"tool_calls": [{
                "index": 0, "id": "c0", "type": "function",
                "function": {"name": "shell", "arguments": "{}"}
            }]}),
            Value::Null,
        )
            + &chunk(json!({}), json!("stop"))
            + &frame(DONE_SENTINEL);
        assert!(collect_stream(&promised_some).is_err());

        // And the agreeing pair is still accepted.
        let ok = chunk(
            json!({"tool_calls": [{
                "index": 0, "id": "c0", "type": "function",
                "function": {"name": "shell", "arguments": "{}"}
            }]}),
            Value::Null,
        )
            + &chunk(json!({}), json!("tool_calls"))
            + &frame(DONE_SENTINEL);
        assert!(collect_stream(&ok)?.finish_reason.wants_tools());
        Ok(())
    }

    #[test]
    fn a_stream_reports_no_usage_rather_than_fabricating_one() -> Result<(), Error> {
        let reply = collect_stream(&text_stream())?;
        assert_eq!(reply.usage, None, "zeros would read as a free turn");

        // The control, going the other way: the same generation on the JSON path
        // does report its tokens, so the absence above is a property of the
        // streaming dialect rather than of this parser.
        let json_reply = parse_reply(
            &json!({
                "choices": [{"message": {"role":"assistant","content":"2 + 2 = 4"}, "finish_reason":"stop"}],
                "usage": {"prompt_tokens": 9, "completion_tokens": 4}
            })
            .to_string(),
        )?;
        assert_eq!(json_reply.usage.map(|u| u.total()), Some(13));
        Ok(())
    }

    #[test]
    fn a_stream_says_nothing_rather_than_saying_the_empty_string() -> Result<(), Error> {
        let body = chunk(json!({"role": "assistant"}), Value::Null)
            + &chunk(json!({}), json!("stop"))
            + &frame(DONE_SENTINEL);
        let reply = collect_stream(&body)?;
        assert_eq!(reply.text(), None, "said-nothing must differ from said-empty");
        assert_eq!(reply.finish_reason, FinishReason::Stop);
        Ok(())
    }

    #[test]
    fn two_finish_reasons_are_refused() -> Result<(), Error> {
        let body = chunk(json!({}), json!("stop"))
            + &chunk(json!({}), json!("length"))
            + &frame(DONE_SENTINEL);
        assert!(collect_stream(&body).is_err());
        Ok(())
    }

    #[test]
    fn an_accumulator_that_failed_stays_failed() -> Result<(), Error> {
        let mut acc = StreamAccumulator::new();
        assert!(acc.push(&StreamChunk::ServerError("boom".into())).is_err());
        assert!(acc.push(&StreamChunk::Done).is_err());
        assert!(acc.finish().is_err());
        Ok(())
    }

    #[test]
    fn a_streaming_request_is_the_blocking_request_plus_stream() -> Result<(), Error> {
        let msgs = [Message::user("hi")];
        let req = ChatRequest::new(&msgs);
        let blocking = body_of(&build_request(&cfg(), &req)?);
        let streaming = body_of(&chat_stream_request(&cfg(), &req)?);
        assert_eq!(streaming["stream"], json!(true));
        assert!(blocking.get("stream").is_none(), "absent, not false, matching serde(default)");
        let mut a = blocking.clone();
        let b = streaming.clone();
        a["stream"] = json!(true);
        assert_eq!(a, b, "streaming must change nothing else about the request");

        let wire = chat_stream_request(&cfg(), &req)?;
        assert!(wire.headers.iter().any(|(k, v)| k == "accept" && v == "text/event-stream"));
        Ok(())
    }

    #[test]
    fn only_an_event_stream_content_type_is_one() {
        assert!(is_event_stream("text/event-stream"));
        assert!(is_event_stream("text/event-stream; charset=utf-8"));
        assert!(is_event_stream("  TEXT/EVENT-STREAM  "));
        // The 2xx a JSON-path server answers with, and the HTML a proxy does.
        assert!(!is_event_stream("application/json"));
        assert!(!is_event_stream("text/html"));
        assert!(!is_event_stream(""));
    }
}