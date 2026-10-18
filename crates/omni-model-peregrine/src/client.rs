//! The transport: send bytes, classify what comes back.
//!
//! Deliberately thin. Every dialect decision lives in [`crate::wire`] and is
//! checked by a certificate; what remains here is the part that genuinely needs
//! a socket — and, importantly, the part that knows what each *failure* means,
//! because a status code is only meaningful with a policy attached.
//!
//! # Status classification
//!
//! | Status | Meaning | Retry? |
//! |---|---|---|
//! | 2xx | the reply, parsed by [`crate::wire::parse_reply`] | — |
//! | 401 | the key is wrong; every retry will also be wrong | no |
//! | 413 | the prompt is over the server's ceiling | **no** |
//! | 429 | standard rate-limit shape | yes |
//! | 503 `overloaded_error` | the server saying it is full | yes |
//! | 5xx | the server's problem | yes |
//! | other 4xx | our request is wrong | no |
//!
//! 413 is the row that matters. It looks like a server-side limit, and the
//! obvious "try again" produces a loop that can never succeed, because a bigger
//! prompt is not a shorter one.
//!
//! # Streaming
//!
//! [`Peregrine::stream`] opens an SSE stream and yields [`StreamChunk`]s as the
//! server produces them. **It is not a [`Model`] implementation**, and that is a
//! deliberate choice rather than an omission. [`Model::chat`] returns one
//! [`ChatReply`], which is a *completed* turn: it has a finish reason and, in the
//! non-streaming path, a token count. Streaming has neither yet, and a trait
//! method that returned before the turn was over could not report the one thing
//! streaming exists for. So `Model` stays the seam for "give me the answer" and
//! [`ChatStream`] is a second, clearly-named entry point for "give me the answer
//! as it arrives" — the same relationship a blocking call has to a callback, and
//! not something to bolt onto the trait as a defaulted method that every backend
//! would then have to implement.
//!
//! Three things the stream can do that the blocking path cannot, and each is a
//! separate failure with its own certificate in `omni-verify`'s `peregrine` group:
//!
//! - it can **stop without finishing** — `peregrine-serve`'s error path returns
//!   from its pump task without sending a finish chunk or `[DONE]`, so end of
//!   body is not end of turn;
//! - it can carry a **tool call split across chunks**, so arguments must be
//!   joined before they are parsed, never parsed fragment by fragment;
//! - it can carry **nothing about tokens at all**, because `usage` exists in the
//!   non-streaming body only, and a stream reports `None` rather than zeros.

use std::collections::VecDeque;
use std::io::Read;
use std::time::Duration;

use omni_core::{Context, Error};
use omni_model::{ChatReply, ChatRequest, Model};

use crate::wire::{
    Config, SseParser, StreamAccumulator, StreamChunk, WireRequest, build_request,
    chat_stream_request, extract_error, is_event_stream, parse_reply,
};

/// A [`Model`] backed by a peregrine server.
pub struct Peregrine {
    agent: ureq::Agent,
    config: Config,
    /// Name used in logs and the reward fingerprint. Defaults to the base URL,
    /// which identifies the *server*; a harness serving two models from one
    /// endpoint needs to override it.
    name: String,
}

impl Peregrine {
    /// Build a client.
    ///
    /// # Errors
    ///
    /// [`Error::Io`] if the agent cannot be configured. There is no network
    /// call here — a client that fails at construction because the server is
    /// down cannot report "the server is down" anywhere else, and the two are
    /// different problems with different fixes.
    pub fn new(config: Config) -> Result<Self, Error> {
        let agent = ureq::Agent::new_with_config(
            ureq::Agent::config_builder()
                // Non-2xx must arrive as a Response, not as a transport error:
                // ureq's `Error::StatusCode` carries the code and *discards the
                // body*, and peregrine's body is where its error message lives.
                // Setting this false is what lets `classify` read both.
                .http_status_as_error(false)
                .timeout_global(Some(Duration::from_secs(config.timeout_secs)))
                .timeout_recv_body(Some(Duration::from_secs(config.timeout_secs)))
                .build(),
        );
        let name = config.base_url.clone();
        Ok(Self { agent, config, name })
    }

    /// Override the name used in logs and fingerprints.
    ///
    /// # Errors
    ///
    /// [`Error::Format`] on an empty name; an unnamed backend produces an
    /// anonymous row in the reward fingerprint, which cannot be attributed to
    /// anything afterwards.
    pub fn named(mut self, name: &str) -> Result<Self, Error> {
        if name.trim().is_empty() {
            return Err(Error::Format("a backend needs a non-empty name".into()));
        }
        self.name = name.to_string();
        Ok(self)
    }

    /// The base URL this client talks to.
    ///
    /// Public because `omni model peregrine` prints it and a config pointing at
    /// the wrong machine is worth seeing before a request is sent.
    #[must_use]
    pub fn base_url(&self) -> &str {
        &self.config.base_url
    }

    /// Fetch a path, bypassing the chat dialect. Used for `/health`, which is
    /// unauthenticated and returns plain text rather than a completion.
    ///
    /// # Errors
    ///
    /// [`Error::Format`] on a non-2xx, [`Error::Io`] on a transport failure.
    pub fn health(&self, url: &str) -> Result<String, Error> {
        let mut response = self
            .agent
            .get(url)
            .call()
            .map_err(transport)
            .map_err(|e| e.annotated(|| format!("GET {url}")))?;
        let status = response.status().as_u16();
        let text = response
            .body_mut()
            .read_to_string()
            .map_err(transport)
            .map_err(|e| e.annotated(|| format!("reading the HTTP {status} body from {url}")))?;
        if !(200..300).contains(&status) {
            return Err(classify(status, &text, url));
        }
        Ok(text)
    }

    /// Send one prepared request. Separated from [`Model::chat`] so the
    /// classification path can be exercised without a `ChatRequest`.
    ///
    /// # Errors
    ///
    /// [`Error::Overloaded`] for 503/429/5xx, [`Error::PromptTooLong`] for 413,
    /// [`Error::Format`] for 401 and every other 4xx, [`Error::Io`] for
    /// transport, and whatever [`parse_reply`] raises for a 2xx.
    pub fn send(&self, wire: &crate::wire::WireRequest) -> Result<ChatReply, Error> {
        let mut builder = self.agent.post(&wire.url);
        for (k, v) in &wire.headers {
            builder = builder.header(k, v);
        }
        let mut response = builder.send(&wire.body).map_err(transport).map_err(|e| {
            e.annotated(|| format!("POST {}", wire.url))
        })?;

        let status = response.status().as_u16();
        let text = response
            .body_mut()
            .read_to_string()
            .map_err(transport)
            .map_err(|e| e.annotated(|| format!("reading the HTTP {status} body from {}", wire.url)))?;

        if !(200..300).contains(&status) {
            return Err(classify(status, &text, &wire.url));
        }
        parse_reply(&text)
    }

    /// Open a streaming chat turn.
    ///
    /// Returns as soon as the server has sent its headers, so the first chunk
    /// arrives while the rest of the answer is still being generated. The
    /// dialect decision — `"stream": true`, and nothing else different — is
    /// [`crate::wire::chat_stream_request`]'s, and every reading decision is
    /// [`crate::wire`]'s; this opens the socket.
    ///
    /// Not a [`Model`] implementation, for the reason in this module's docs.
    ///
    /// # Errors
    ///
    /// [`Error::Overloaded`] for 503/429/5xx, [`Error::PromptTooLong`] for 413,
    /// [`Error::Format`] for 401, every other 4xx, and a 2xx whose content type
    /// is not an event stream, [`Error::Io`] for transport. The status
    /// classification is the same function [`Peregrine::send`] uses, because
    /// asking for a stream must not change what a 503 means.
    pub fn stream(&self, req: &ChatRequest<'_>) -> Result<ChatStream, Error> {
        let wire = chat_stream_request(&self.config, req)?;
        self.open(&wire)
    }

    fn open(&self, wire: &WireRequest) -> Result<ChatStream, Error> {
        let mut builder = self.agent.post(&wire.url);
        for (k, v) in &wire.headers {
            builder = builder.header(k, v);
        }
        let response = builder.send(&wire.body).map_err(transport).map_err(|e| {
            e.annotated(|| format!("POST {}", wire.url))
        })?;

        let status = response.status().as_u16();
        if !(200..300).contains(&status) {
            let text = response
                .into_body()
                .read_to_string()
                .map_err(transport)
                .map_err(|e| e.annotated(|| format!("reading the HTTP {status} body from {}", wire.url)))?;
            return Err(classify(status, &text, &wire.url));
        }

        // The content type is the only thing that separates the two dialects on
        // a 200. Checking it here is what turns "the body was JSON when I asked
        // for events" from a JSON parse error two hundred lines into a stream
        // into one sentence at the point it went wrong.
        let content_type = response
            .headers()
            .get("content-type")
            .map_or("", |v| v.to_str().unwrap_or(""));
        if !is_event_stream(content_type) {
            return Err(Error::Format(format!(
                "{} answered 200 with content-type {content_type:?}, which is not an \
                 event stream; a streaming request was answered on the JSON path",
                wire.url
            ))
            .annotated(|| "opening a peregrine event stream".to_string()));
        }

        Ok(ChatStream::new(response.into_body().into_reader()))
    }
}

/// A live peregrine event stream.
///
/// Yields one [`StreamChunk`] per thing the stream said, in order. Ending is
/// checked rather than assumed: the iterator's last word on a stream that
/// stopped early is the error explaining that it did, not `None`. That is the
/// difference `peregrine-serve` makes between a finished generation and a failed
/// one — both close the socket, and only one sends `[DONE]`.
pub struct ChatStream {
    reader: ureq::BodyReader<'static>,
    parser: SseParser,
    acc: StreamAccumulator,
    pending: VecDeque<StreamChunk>,
    buf: [u8; 4096],
    finished: bool,
}

impl ChatStream {
    /// Wrap an open event-stream body.
    ///
    /// Public so a caller who has already made the HTTP call can hand the body
    /// over; [`Peregrine::stream`] is the usual way in.
    #[must_use]
    pub fn new(reader: ureq::BodyReader<'static>) -> Self {
        Self {
            reader,
            parser: SseParser::new(),
            acc: StreamAccumulator::new(),
            pending: VecDeque::new(),
            buf: [0u8; 4096],
            finished: false,
        }
    }

    /// Read once. `Ok(false)` means the body ended.
    fn pump(&mut self) -> Result<bool, Error> {
        let n = self
            .reader
            .read(&mut self.buf)
            .ctx(|| "reading the peregrine event stream".to_string())?;
        if n == 0 {
            return Ok(false);
        }
        for frame in self.parser.feed(&self.buf[..n])? {
            for chunk in crate::wire::chunk_from_frame(&frame)? {
                // Judged before it is queued, so a stream that goes wrong
                // halfway cannot hand the caller a chunk it has already refused.
                self.acc.push(&chunk)?;
                self.pending.push_back(chunk);
            }
        }
        Ok(true)
    }

    /// Check that the stream ended as a *finished* turn.
    ///
    /// Both halves can fail and both are about the body having ended, which is
    /// exactly when a reader is most tempted to call it finished: the framing
    /// may have been cut off mid-frame, and the dialect may never have said so.
    fn end(&mut self) -> Result<(), Error> {
        self.parser.finish()?;
        self.acc.finish().map(|_| ())
    }

    /// Drain the stream into one [`ChatReply`].
    ///
    /// Exactly equivalent to iterating to the end and calling
    /// [`StreamAccumulator::finish`], for a caller that does not need the
    /// deltas. **The two paths are not interchangeable for a long answer**: this
    /// one buffers the whole turn, which is what the blocking path costs too.
    ///
    /// # Errors
    ///
    /// Whatever the iterator reports, then whatever [`StreamAccumulator::finish`]
    /// reports about the shape of the stream as a whole.
    pub fn collect(mut self) -> Result<ChatReply, Error> {
        for chunk in self.by_ref() {
            chunk?;
        }
        self.acc.finish()
    }
}

impl Iterator for ChatStream {
    type Item = Result<StreamChunk, Error>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if let Some(chunk) = self.pending.pop_front() {
                return Some(Ok(chunk));
            }
            if self.finished {
                return None;
            }
            match self.pump() {
                // Bytes but no event yet: the frame is still arriving.
                Ok(true) => {}
                Ok(false) => {
                    self.finished = true;
                    return match self.end() {
                        Ok(()) => None,
                        Err(e) => Some(Err(e)),
                    };
                }
                Err(e) => {
                    self.finished = true;
                    return Some(Err(e));
                }
            }
        }
    }
}

/// The one place a `ureq::Error` becomes an `omni_core::Error`.
///
/// `ureq::Error` is a foreign type and `omni_core::Error` belongs to another
/// crate, so `From` cannot be implemented for it — the orphan rule applies.
/// That makes this function the workspace's second and last error conversion,
/// and it lives here rather than at each call site for the same reason
/// `Context::ctx` does.
///
/// Each variant is mapped by what it *means*, not by how it prints:
///
/// - a timeout becomes an `io::Error` of kind `TimedOut`, because that is
///   genuinely what happened and `Error::retryable` already knows a timed-out
///   operation is worth retrying;
/// - `StatusCode` cannot occur here, because the agent is configured with
///   `http_status_as_error(false)` and non-2xx arrives as a `Response`. If it
///   ever does, it means the configuration was bypassed, so the message says so
///   rather than pretending to classify it.
fn transport(e: ureq::Error) -> Error {
    match e {
        ureq::Error::Io(io) => Error::from(io),
        ureq::Error::Timeout(t) => Error::from(std::io::Error::new(
            std::io::ErrorKind::TimedOut,
            format!("{t:?}"),
        )),
        ureq::Error::StatusCode(code) => Error::Format(format!(
            "transport returned HTTP {code} directly, which means the agent's \
             http_status_as_error(false) setting was bypassed; the status \
             classifier did not run"
        )),
        other => Error::Format(format!("transport: {other}")),
    }
}

/// Turn a non-2xx into the error that says what to do about it.
///
/// `Error` is a closed enum precisely so this function can exist: a
/// stringly-typed error would have to say "503 overloaded" in prose and hope the
/// caller read it.
fn classify(status: u16, body: &str, url: &str) -> Error {
    let detail = extract_error(body);
    match status {
        // The server's own load signal. Retryable, because it is the server
        // saying "not now" rather than "not ever".
        503 | 429 => Error::Overloaded { status, detail },
        // Over the ceiling. Retrying is the tempting mistake and always wrong.
        413 => Error::PromptTooLong { status, detail },
        // Server-side faults are usually transient, and unlike 413 a retry can
        // plausibly succeed.
        s if s >= 500 => Error::Overloaded { status, detail },
        _ => Error::Format(format!("{url} returned HTTP {status}: {detail}")),
    }
}

impl Model for Peregrine {
    fn chat(&mut self, req: &ChatRequest<'_>) -> Result<ChatReply, Error> {
        let wire = build_request(&self.config, req)?;
        self.send(&wire)
    }

    fn name(&self) -> &str {
        &self.name
    }

    /// `true`, on the strength of `temperature: 0.0` being sent explicitly and
    /// peregrine defaulting to 0.0.
    ///
    /// Stated rather than assumed because a harness claiming a matched baseline
    /// needs it: a backend that samples anyway produces a `Δ` that is noise,
    /// and a noise `Δ` is worse than none because it looks like evidence.
    fn deterministic_at_zero(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use std::io::Write;
    use std::net::TcpListener;

    use super::*;

    fn detail() -> String {
        r#"{"error":{"message":"queue full","type":"overloaded_error"}}"#.to_string()
    }

    #[test]
    fn a_503_is_the_servers_own_overload_signal_and_is_retryable() {
        let e = classify(503, &detail(), "http://h/v1");
        assert!(e.retryable(), "a server saying 'full' is worth retrying");
        let msg = e.to_string();
        assert!(msg.contains("overloaded"), "{msg}");
        assert!(msg.contains("queue full"), "the server's message must survive: {msg}");
    }

    #[test]
    fn a_413_is_never_retryable() {
        // The row that matters: it looks like a server limit, and "try again"
        // produces a loop that can never succeed.
        let e = classify(413, "prompt too long", "http://h/v1");
        assert!(!e.retryable(), "a bigger prompt is not a shorter one");
        assert!(matches!(e, Error::PromptTooLong { status: 413, .. }));
    }

    #[test]
    fn a_401_is_a_configuration_problem_not_a_transient_one() {
        let e = classify(401, r#"{"error":{"message":"bad key","type":"auth"}}"#, "http://h/v1");
        assert!(!e.retryable(), "every retry will also be rejected");
        assert!(matches!(e, Error::Format(_)));
    }

    #[test]
    fn a_400_is_our_request_being_wrong() {
        let e = classify(400, r#"{"error":{"message":"empty messages","type":"bad_request"}}"#, "http://h/v1");
        assert!(!e.retryable());
        assert!(e.to_string().contains("empty messages"));
    }

    #[test]
    fn a_5xx_is_worth_retrying_even_when_it_is_not_an_overload() {
        assert!(classify(500, "boom", "http://h/v1").retryable());
        assert!(classify(502, "bad gateway", "http://h/v1").retryable());
    }

    #[test]
    fn an_unparseable_error_body_still_produces_a_useful_message() {
        // A proxy in front of the server may answer with HTML. Losing the body
        // would leave "HTTP 502: " and nothing to act on.
        let e = classify(502, "<html>bad gateway</html>", "http://h/v1");
        assert!(e.to_string().contains("bad gateway"), "{e}");
    }

    #[test]
    fn classification_looks_through_annotations() -> Result<(), Error> {
        let annotated = Error::Overloaded { status: 503, detail: "full".into() }
            .annotated(|| "talking to the model".to_string());
        assert!(annotated.retryable(), "the annotation must not hide the signal");
        assert!(annotated.root().to_string().contains("full"));
        Ok(())
    }

    #[test]
    fn a_client_can_be_named_and_an_empty_name_is_refused() -> Result<(), Error> {
        let p = Peregrine::new(Config::default())?.named("qwen3.8-27b")?;
        assert_eq!(p.name(), "qwen3.8-27b");
        assert!(Peregrine::new(Config::default())?.named("  ").is_err());
        Ok(())
    }

    #[test]
    fn a_client_constructs_without_touching_the_network() -> Result<(), Error> {
        // Construction must not depend on the server: a client that cannot be
        // built when the server is down cannot report that the server is down.
        // Port 1 is reserved and nothing listens on it, so if construction did
        // dial, this test would be slow rather than instant.
        Peregrine::new(Config { base_url: "http://127.0.0.1:1".into(), ..Config::default() })?;
        Ok(())
    }

    #[test]
    fn the_peregrine_backend_declares_itself_deterministic_at_zero() -> Result<(), Error> {
        assert!(Peregrine::new(Config::default())?.deterministic_at_zero());
        Ok(())
    }

    // ── A real socket, for the three branches `open` has ────────────────────
    //
    // The dialect is checked against literals, but the *transport* decisions —
    // that a 200 is refused when it is not an event stream, that a 503 on the
    // streaming path means the same thing it means on the blocking one — can
    // only be checked against something that answers.

    /// A socket that answers exactly one request with `response` and closes.
    struct OneShot {
        url: String,
        handle: std::thread::JoinHandle<std::io::Result<()>>,
    }

    impl OneShot {
        fn start(status: &str, content_type: &str, body: &str) -> Result<Self, Error> {
            let listener = TcpListener::bind("127.0.0.1:0")?;
            let addr = listener.local_addr()?;
            let response = format!(
                "HTTP/1.1 {status}\r\ncontent-type: {content_type}\r\n\
                 content-length: {}\r\nconnection: close\r\n\r\n{body}",
                body.len()
            );
            let handle = std::thread::spawn(move || -> std::io::Result<()> {
                let (mut sock, _) = listener.accept()?;
                // Read the request before answering it. Writing first and
                // closing races the client's own body write, and a broken pipe
                // in the *client* would be a broken test rather than a finding.
                sock.set_read_timeout(Some(std::time::Duration::from_secs(10)))?;
                let mut seen = Vec::new();
                let mut chunk = [0u8; 1024];
                let header_end = loop {
                    if let Some(i) = find(&seen, b"\r\n\r\n") {
                        break i + 4;
                    }
                    let n = sock.read(&mut chunk)?;
                    if n == 0 {
                        return Err(std::io::Error::new(
                            std::io::ErrorKind::UnexpectedEof,
                            "the client hung up before sending its headers",
                        ));
                    }
                    seen.extend_from_slice(&chunk[..n]);
                };
                let head = String::from_utf8_lossy(&seen[..header_end]).to_string();
                let want: usize = head
                    .lines()
                    .find_map(|l| {
                        let (k, v) = l.split_once(':')?;
                        k.trim()
                            .eq_ignore_ascii_case("content-length")
                            .then(|| v.trim().parse().unwrap_or(0))
                    })
                    .unwrap_or(0);
                while seen.len() < header_end + want {
                    let n = sock.read(&mut chunk)?;
                    if n == 0 {
                        break;
                    }
                    seen.extend_from_slice(&chunk[..n]);
                }
                sock.write_all(response.as_bytes())?;
                sock.flush()
            });
            Ok(Self { url: format!("http://{addr}"), handle })
        }

        fn finish(self) -> Result<(), Error> {
            match self.handle.join() {
                Ok(Ok(())) => Ok(()),
                Ok(Err(e)) => {
                    Err(Error::from(e).annotated(|| "writing the one-shot test response".to_string()))
                }
                Err(_) => Err(Error::Format("the one-shot test server panicked".into())),
            }
        }
    }

    fn find(haystack: &[u8], needle: &[u8]) -> Option<usize> {
        haystack.windows(needle.len()).position(|w| w == needle)
    }

    fn chunk(delta: serde_json::Value, finish: serde_json::Value) -> String {
        format!(
            "data: {}\n\n",
            serde_json::json!({
                "id": "chatcmpl-7", "object": "chat.completion.chunk", "created": 1, "model": "m",
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]
            })
        )
    }

    #[test]
    fn a_stream_is_read_end_to_end_over_a_real_socket() -> Result<(), Error> {
        let body = chunk(serde_json::json!({"role": "assistant"}), serde_json::Value::Null)
            + &chunk(serde_json::json!({"content": "2 + 2 = 4"}), serde_json::Value::Null)
            + &chunk(serde_json::json!({}), serde_json::json!("stop"))
            + "data: [DONE]\n\n";
        let server = OneShot::start("200 OK", "text/event-stream", &body)?;

        let msgs = [omni_model::Message::user("What is 2+2?")];
        let req = ChatRequest::new(&msgs);
        let client = Peregrine::new(Config { base_url: server.url.clone(), ..Config::default() })?;
        let reply = client.stream(&req)?.collect()?;

        server.finish()?;
        assert_eq!(reply.text(), Some("2 + 2 = 4"));
        assert_eq!(reply.finish_reason, omni_model::FinishReason::Stop);
        assert_eq!(reply.usage, None, "peregrine sends usage on the JSON path only");
        Ok(())
    }

    #[test]
    fn deltas_arrive_before_the_turn_is_over() -> Result<(), Error> {
        let body = chunk(serde_json::json!({"role": "assistant"}), serde_json::Value::Null)
            + &chunk(serde_json::json!({"content": "par"}), serde_json::Value::Null)
            + &chunk(serde_json::json!({"content": "tial"}), serde_json::Value::Null)
            + &chunk(serde_json::json!({}), serde_json::json!("stop"))
            + "data: [DONE]\n\n";
        let server = OneShot::start("200 OK", "text/event-stream", &body)?;

        let msgs = [omni_model::Message::user("hi")];
        let req = ChatRequest::new(&msgs);
        let client = Peregrine::new(Config { base_url: server.url.clone(), ..Config::default() })?;
        let mut stream = client.stream(&req)?;
        let mut text = String::new();
        let mut chunks = 0usize;
        for c in &mut stream {
            if let crate::wire::StreamChunk::Text(t) = c? {
                // Concatenating rather than collecting is the whole point: the
                // text exists before the finish reason does.
                assert!(chunks == 0 || !text.is_empty());
                text.push_str(&t);
                chunks += 1;
            }
        }
        server.finish()?;
        assert_eq!(text, "partial");
        assert_eq!(chunks, 2);
        Ok(())
    }

    #[test]
    fn a_200_that_is_not_an_event_stream_is_refused_by_its_content_type() -> Result<(), Error> {
        let server = OneShot::start("200 OK", "application/json", r#"{"choices":[]}"#)?;
        let msgs = [omni_model::Message::user("hi")];
        let client = Peregrine::new(Config { base_url: server.url.clone(), ..Config::default() })?;
        let Err(err) = client.stream(&ChatRequest::new(&msgs)) else {
            return Err(Error::Format("a JSON body must not pass as an event stream".into()));
        };
        server.finish()?;
        let msg = err.to_string();
        assert!(msg.contains("application/json"), "must name what arrived: {msg}");
        Ok(())
    }

    #[test]
    fn a_503_on_the_streaming_path_is_still_the_servers_own_overload_signal() -> Result<(), Error> {
        let body = r#"{"error":{"message":"queue full","type":"overloaded_error"}}"#;
        let server = OneShot::start("503 Service Unavailable", "application/json", body)?;
        let msgs = [omni_model::Message::user("hi")];
        let client = Peregrine::new(Config { base_url: server.url.clone(), ..Config::default() })?;
        let err = match client.stream(&ChatRequest::new(&msgs)) {
            Ok(_) => return Err(Error::Format("a 503 must not open a stream".into())),
            Err(e) => e,
        };
        server.finish()?;
        // Asking for a stream must not change what a status code means.
        assert!(err.retryable(), "the server saying 'full' is worth retrying: {err}");
        assert!(err.to_string().contains("queue full"), "{err}");
        Ok(())
    }
}