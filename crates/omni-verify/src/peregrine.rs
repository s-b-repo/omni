//! `peregrine` — the dialect, as certificates rather than as comments.
//!
//! peregrine implements *nearly* the OpenAI chat-completions dialect. The
//! differences are the part that matters, because an agent that assumes the
//! OpenAI behaviour produces different tokens on peregrine than it did on the
//! API it was developed against — and **nothing fails**. The difference surfaces
//! weeks later as "the model got worse".
//!
//! So each difference is a claim, a residual, and a control that goes the other
//! way.
//!
//! # The streaming half
//!
//! The last third of this group certifies the event stream, and its fixtures are
//! transcribed from `peregrine-serve`'s own source rather than written from the
//! OpenAI spec. `peregrine-serve` builds events with `axum` 0.7.9, whose
//! `Event::field` writes `name`, `:`, one space, the value and `\n` and whose
//! `finalize` appends one more `\n` — so an event is exactly `data: <payload>
//! \n\n` — and whose `KeepAlive::default()` is an empty comment, `b":\n\n"`.
//! The shapes certified are the ones `main.rs` writes:
//!
//! | frame | source |
//! |---|---|
//! | `delta: {"role":"assistant"}`, `finish_reason: null` | `main.rs:1009` |
//! | `delta: {"content": …}`, `finish_reason: null` | `main.rs:1045` |
//! | `delta: {"reasoning_content": …}` | `main.rs:1446` |
//! | `delta: {"tool_calls": [whole call]}` | `main.rs:1468` |
//! | `data: {"error":{"message":…}}` | `main.rs:1483` |
//! | `delta: {}` with `finish_reason` | `main.rs:1093` |
//! | `data: [DONE]` | `main.rs:1094` |
//!
//! One caveat, stated rather than papered over: `peregrine-serve` has no test
//! asserting a stream's raw bytes — it asserts against `axum`'s `Sse` type — so
//! the framing is a property of `axum` 0.7.9 as much as of peregrine. These
//! certificates therefore pin the event-stream rules using `axum`'s output as
//! the fixture, not `axum`'s output as the specification.

use omni_core::canonical::{obj, value_id};
use omni_core::Error;
use omni_model::{ChatReply, ChatRequest, FinishReason, Message, ToolSpec};
use omni_model_peregrine::wire::{
    Config, DONE_SENTINEL, SseFrame, SseParser, StreamAccumulator, StreamChunk, build_request,
    chat_stream_request, chunk_from_frame, completions_url, is_event_stream, normalize_base,
    parse_reply,
};
use serde_json::{Value, json};

use crate::{Certificate, cert, exact, failed, with_cases};

/// Run the `peregrine` group.
#[must_use]
pub fn run() -> Vec<Certificate> {
    vec![
        temperature_default_is_zero(),
        top_k_omitted_when_off(),
        tools_advertised_never_forced(),
        tool_choice_none_is_the_only_off_switch(),
        ignored_fields_are_not_sent(),
        trailing_slash_does_not_double(),
        message_content_null_is_null_not_empty(),
        identical_knobs_hash_identically(),
        stream_flag_is_the_only_difference(),
        frames_survive_read_boundaries(),
        keep_alive_comments_are_not_events(),
        multi_line_data_fields_are_joined(),
        done_sentinel_is_recognised_by_payload(),
        a_stream_without_its_sentinel_is_refused(),
        a_mid_stream_error_is_an_error_not_a_turn(),
        an_event_after_the_sentinel_is_refused(),
        a_chunk_with_nothing_to_say_is_refused(),
        tool_call_fragments_reassemble_by_index(),
        a_stream_reports_no_usage_not_a_fabricated_one(),
        a_tool_calling_stream_carries_no_text(),
        truncated_tool_arguments_are_refused(),
        finish_reason_agrees_with_the_calls_announced(),
        a_null_finish_reason_is_not_the_end(),
        only_an_event_stream_content_type_is_one(),
        reasoning_content_is_not_answered_text(),
        a_body_cut_off_mid_frame_is_refused(),
        an_undecodable_line_is_refused_not_decoded_lossily(),
    ]
}

fn cfg() -> Config {
    Config { base_url: "http://127.0.0.1:8132".into(), ..Config::default() }
}

fn body_of(req: &ChatRequest<'_>) -> Option<Value> {
    let wire = build_request(&cfg(), req).ok()?;
    serde_json::from_str(&wire.body).ok()
}

fn msgs() -> Vec<Message> {
    vec![Message::user("hi")]
}

// ── Stream fixtures ─────────────────────────────────────────────────────────
//
// Transcribed, not generated: a fixture produced by this client's own framing
// would prove the reader agrees with the framing, which is the thing in
// question. `axum` 0.7.9 writes `data: `, the payload, and a blank line; one
// more character anywhere and this file stops describing the server.

/// One event, in `axum`'s framing.
fn frame(payload: &str) -> String {
    format!("data: {payload}\n\n")
}

/// One `chat.completion.chunk` event.
fn chunk(delta: Value, finish: Value) -> String {
    frame(
        &json!({
            "id": "chatcmpl-7",
            "object": "chat.completion.chunk",
            "created": 1700000000,
            "model": "peregrine-test",
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]
        })
        .to_string(),
    )
}

/// One tool call, whole, as `main.rs:1468` writes it.
fn whole_call(index: usize, id: &str, name: &str, arguments: &str) -> Value {
    json!({"tool_calls": [{
        "index": index, "id": id, "type": "function",
        "function": {"name": name, "arguments": arguments}
    }]})
}

/// Drive the production reader over a literal stream, byte for byte.
///
/// The same calls the transport makes, in the same order, so a certificate here
/// is a statement about shipped code rather than about a re-reading of it.
fn read_stream(body: &str) -> Result<ChatReply, Error> {
    let mut parser = SseParser::new();
    let mut acc = StreamAccumulator::new();
    for frame in parser.feed(body.as_bytes())? {
        for chunk in chunk_from_frame(&frame)? {
            acc.push(&chunk)?;
        }
    }
    parser.finish()?;
    acc.finish()
}

/// The events a stream produces, without assembling them.
fn read_chunks(body: &str) -> Result<Vec<StreamChunk>, Error> {
    let mut parser = SseParser::new();
    let mut out = Vec::new();
    for frame in parser.feed(body.as_bytes())? {
        out.extend(chunk_from_frame(&frame)?);
    }
    parser.finish()?;
    Ok(out)
}

/// The frames a stream produces.
fn read_frames(body: &str) -> Result<Vec<SseFrame>, Error> {
    let mut parser = SseParser::new();
    let frames = parser.feed(body.as_bytes())?;
    parser.finish()?;
    Ok(frames)
}

/// The same frames, from a stream delivered one byte at a time.
fn read_frames_dribbled(body: &str) -> Result<Vec<SseFrame>, Error> {
    let mut parser = SseParser::new();
    let mut frames = Vec::new();
    for b in body.as_bytes() {
        frames.extend(parser.feed(std::slice::from_ref(b))?);
    }
    parser.finish()?;
    Ok(frames)
}

/// A refusal in the *framing* that must name `needle`.
///
/// Separate from [`refused_with`] because it never reaches the accumulator: a
/// corrupt line fails while the bytes are being read and a half-frame fails
/// while the body is being closed, and in both cases there is nothing yet to
/// assemble. Running it here rather than through `read_stream` is what keeps
/// "the framing refused" distinguishable from "the turn was refused".
///
/// Takes bytes rather than a `&str` because one of the failures it exists for is
/// a line that is *not* text: `\u{80}` in a Rust literal is the character U+0080
/// in valid UTF-8, which is the opposite of the fixture, and a reader handed
/// only `&str` could not be shown the case at all.
fn framing_refused_with(body: &[u8], needle: &str) -> bool {
    let mut parser = SseParser::new();
    // Either point can refuse and which one fired is the finding, not a detail.
    let outcome = match parser.feed(body) {
        Err(e) => Err(e),
        Ok(_) => parser.finish(),
    };
    // Rendered lossy for the message only. The reader is what has to refuse an
    // undecodable byte; a diagnostic that printed `104, 105` instead of the text
    // would just move the problem to whoever reads the failure.
    let shown = format!("{:?}", String::from_utf8_lossy(body));
    match outcome {
        Ok(()) => {
            eprintln!("omni-verify: peregrine: {shown} closed cleanly when it should have been refused");
            false
        }
        Err(e) => {
            let msg = e.to_string();
            if msg.contains(needle) {
                true
            } else {
                eprintln!("omni-verify: peregrine: the refusal of {shown} did not name {needle:?}: {msg}");
                false
            }
        }
    }
}

/// A refusal that must name `needle`, so a failure names the rule it broke
/// rather than just "something went wrong".
fn refused_with(result: Result<ChatReply, Error>, needle: &str) -> bool {
    match result {
        Ok(_) => {
            eprintln!("omni-verify: peregrine: the stream assembled when it should not have");
            false
        }
        Err(e) => {
            let msg = e.to_string();
            if msg.contains(needle) {
                true
            } else {
                eprintln!("omni-verify: peregrine: refusal did not name {needle:?}: {msg}");
                false
            }
        }
    }
}

fn temperature_default_is_zero() -> Certificate {
    // The load-bearing one. OpenAI's default is 1.0; peregrine's is 0.0; and a
    // client that leaves it implicit inherits whichever server it reached. Send
    // it explicitly so the property belongs to the request.
    let probes = [0.0f64, 0.7, 1.0];
    let history = msgs();
    let mut agreed = 0usize;
    for t in probes {
        let req = ChatRequest { temperature: t, ..ChatRequest::new(&history) };
        match body_of(&req) {
            Some(b) if b.get("temperature").and_then(Value::as_f64) == Some(t) => agreed += 1,
            _ => eprintln!("omni-verify: peregrine: temperature {t} did not survive into the body"),
        }
    }

    // The control, in the other direction: the default must be 0.0, not 1.0.
    let default_temp = ChatRequest::default().temperature;
    if default_temp != 0.0 {
        eprintln!("omni-verify: peregrine: ChatRequest defaults temperature to {default_temp}, expected 0.0");
    } else {
        agreed += 1;
    }

    with_cases(
        cert(
            "peregrine",
            "temperature_is_sent_explicitly",
            "the sampling temperature reaches the wire unchanged and the client \
             default is 0.0 rather than OpenAI's 1.0, so a server default \
             change cannot silently produce a different agent",
            exact(agreed == probes.len() + 1),
            0.0,
        ),
        probes.len() + 1,
    )
}

fn top_k_omitted_when_off() -> Certificate {
    // `top_k: 0` means off. Omitting it when off keeps the request minimal and
    // matches how the server represents "off"; sending it when set is the part
    // that changes behaviour.
    let history = msgs();
    let off = body_of(&ChatRequest::new(&history));
    let on = body_of(&ChatRequest { top_k: 40, ..ChatRequest::new(&history) });

    let mut agreed = 0usize;
    if let Some(b) = &off {
        if b.get("top_k").is_none() {
            agreed += 1;
        } else {
            eprintln!("omni-verify: peregrine: top_k sent while disabled");
        }
    }
    if let Some(b) = &on {
        if b.get("top_k").and_then(Value::as_u64) == Some(40) {
            agreed += 1;
        } else {
            eprintln!("omni-verify: peregrine: top_k 40 did not survive");
        }
    }

    with_cases(
        cert(
            "peregrine",
            "top_k_present_only_when_enabled",
            "top_k is omitted while disabled and sent unchanged when enabled, so \
             the request says what it means rather than always carrying a zero",
            exact(agreed == 2),
            0.0,
        ),
        2,
    )
}

fn tools_advertised_never_forced() -> Certificate {
    // A forced tool call on a turn where the model wanted to answer is a
    // harness lying about what it observed.
    let tools = match ToolSpec::new("echo", "echo back", serde_json::json!({"type":"object"})) {
        Ok(t) => vec![t],
        Err(e) => {
            return cert(
                "peregrine",
                "tools_are_advertised_never_forced",
                "tools are advertised and never forced",
                failed("peregrine", &e.to_string()),
                0.0,
            )
        }
    };
    let history = msgs();
    let req = ChatRequest::new(&history).with_tools(&tools);
    let Some(b) = body_of(&req) else {
        return cert(
            "peregrine",
            "tools_are_advertised_never_forced",
            "tools are advertised and never forced",
            failed("peregrine", "request did not build"),
            0.0,
        );
    };

    let mut agreed = 0usize;
    if b.get("tools").and_then(Value::as_array).map(|a| a.len() == 1).unwrap_or(false) {
        agreed += 1;
    }
    if b.get("tool_choice").and_then(Value::as_str) == Some("auto") {
        agreed += 1;
    }
    if b.get("tool_choice").and_then(Value::as_str) != Some("required") {
        agreed += 1;
    }

    with_cases(
        cert(
            "peregrine",
            "tools_are_advertised_never_forced",
            "a request carrying tools advertises them with `tool_choice: auto` \
             and never `required`, because peregrine ignores `required` and a \
             forced call would misreport what the model chose",
            exact(agreed == 3),
            0.0,
        ),
        3,
    )
}

fn tool_choice_none_is_the_only_off_switch() -> Certificate {
    // peregrine honours only the exact string `"none"`. A request with no tools
    // must therefore send `"none"` explicitly rather than omitting the field —
    // omitting it is "whatever the default is", which is not the same statement.
    let history = msgs();
    let no_tools = body_of(&ChatRequest::new(&history));
    let mut agreed = 0usize;
    match &no_tools {
        Some(b) if b.get("tool_choice").and_then(Value::as_str) == Some("none") => agreed += 1,
        Some(b) => eprintln!(
            "omni-verify: peregrine: a toolless request sent tool_choice {:?}",
            b.get("tool_choice")
        ),
        None => eprintln!("omni-verify: peregrine: toolless request did not build"),
    }
    with_cases(
        cert(
            "peregrine",
            "tool_choice_none_is_explicit",
            "a request with no tools states `tool_choice: \"none\"` explicitly, \
             since peregrine acts on that exact string and omitting the field is \
             not the same statement",
            exact(agreed == 1),
            0.0,
        ),
        1,
    )
}

fn ignored_fields_are_not_sent() -> Certificate {
    // The rule that keeps a claim honest: anything peregrine discards must not
    // be sent, or the harness looks like it is pinning behaviour it does not
    // have. A `seed` that is accepted and dropped is worse than no seed, because
    // the author believes they made the run reproducible.
    const DROPPED: &[&str] = &[
        "n", "stop", "seed", "logprobs", "presence_penalty", "frequency_penalty",
        "response_format", "stream_options",
    ];
    let history = msgs();
    let Some(b) = body_of(&ChatRequest::new(&history)) else {
        return cert(
            "peregrine",
            "ignored_fields_are_not_sent",
            "fields the server discards are not sent",
            failed("peregrine", "request did not build"),
            0.0,
        );
    };
    let absent = DROPPED.iter().filter(|k| b.get(**k).is_none()).count();
    with_cases(
        cert(
            "peregrine",
            "ignored_fields_are_not_sent",
            "no field peregrine accepts-and-discards is sent, so the harness \
             never appears to pin a behaviour the server does not implement",
            exact(absent == DROPPED.len()),
            0.0,
        ),
        DROPPED.len(),
    )
}

fn trailing_slash_does_not_double() -> Certificate {
    // `//v1` is accepted by some servers and 404s on others — a difference that
    // only appears on the machine whose config someone typed with a slash.
    const BASES: &[(&str, &str)] = &[
        ("http://h:8080", "http://h:8080/v1/chat/completions"),
        ("http://h:8080/", "http://h:8080/v1/chat/completions"),
        ("http://h:8080///", "http://h:8080/v1/chat/completions"),
        ("  http://h:8080/  ", "http://h:8080/v1/chat/completions"),
    ];
    let mut agreed = 0usize;
    for (base, want) in BASES {
        if &completions_url(base) == want {
            agreed += 1;
        } else {
            eprintln!(
                "omni-verify: peregrine: base {base:?} produced {:?}, expected {want:?}",
                completions_url(base)
            );
        }
    }
    if normalize_base("http://h:8080/") == "http://h:8080" {
        agreed += 1;
    }

    with_cases(
        cert(
            "peregrine",
            "base_url_normalizes",
            "a base URL with trailing slashes or whitespace produces exactly one \
             separator before /v1, so the request URL does not depend on how the \
             operator typed the config",
            exact(agreed == BASES.len() + 1),
            0.0,
        ),
        BASES.len() + 1,
    )
}

fn message_content_null_is_null_not_empty() -> Certificate {
    // A tool-calling assistant turn sends `content: null`. `""` is accepted by
    // some servers and rejected by others, and the difference is invisible until
    // a multi-tool turn starts failing.
    let call = match omni_model::ToolCall::new("c1", "shell", serde_json::json!({"command":"ls"})) {
        Ok(c) => c,
        Err(e) => {
            return cert(
                "peregrine",
                "tool_call_content_is_null",
                "a tool-calling turn sends content null",
                failed("peregrine", &e.to_string()),
                0.0,
            )
        }
    };
    let msgs = vec![Message::assistant_tool_calls(vec![call])];
    let Some(b) = body_of(&ChatRequest::new(&msgs)) else {
        return cert(
            "peregrine",
            "tool_call_content_is_null",
            "a tool-calling turn sends content null",
            failed("peregrine", "request did not build"),
            0.0,
        );
    };

    let content = &b["messages"][0]["content"];
    let args = &b["messages"][0]["tool_calls"][0]["function"]["arguments"];

    let mut agreed = 0usize;
    if content.is_null() {
        agreed += 1;
    } else {
        eprintln!("omni-verify: peregrine: a tool-calling turn sent content {content}, expected null");
    }
    // And the arguments go back to being a *string* on the wire, which is what
    // the OpenAI schema says and what the server parses.
    if args.is_string() {
        agreed += 1;
    } else {
        eprintln!("omni-verify: peregrine: tool arguments were not sent as a string");
    }

    with_cases(
        cert(
            "peregrine",
            "tool_call_content_is_null",
            "an assistant turn that only calls tools sends `content: null` and \
             string-encoded arguments, matching the shape the server parses \
             rather than the one that merely looks similar",
            exact(agreed == 2),
            0.0,
        ),
        2,
    )
}

fn identical_knobs_hash_identically() -> Certificate {
    // The reward-fingerprint property. Two requests that would produce the same
    // tokens must hash the same, or the cache misses on every call; two that
    // would not must not, or a matched baseline silently becomes an unmatched
    // one.
    //
    // This is the property Harness-R1's identity pinning needs and does not have
    // a direct analogue for, because the knobs it pins are task-side while these
    // are server-side.
    let built = (
        obj([
            ("temperature", serde_json::Value::from(0.0)),
            ("top_p", serde_json::Value::from(0.95)),
        ]),
        obj([
            ("top_p", serde_json::Value::from(0.95)),
            ("temperature", serde_json::Value::from(0.0)),
        ]),
        obj([
            ("temperature", serde_json::Value::from(0.7)),
            ("top_p", serde_json::Value::from(0.95)),
        ]),
    );
    let (Ok(a), Ok(b), Ok(c)) = built else {
        return cert(
            "peregrine",
            "sampling_knobs_are_identity",
            "sampling knobs hash by value",
            failed("peregrine", "knob fixture failed to build"),
            0.0,
        );
    };

    let (Ok(ia), Ok(ib), Ok(ic)) = (value_id(&a), value_id(&b), value_id(&c)) else {
        return cert(
            "peregrine",
            "sampling_knobs_are_identity",
            "sampling knobs hash by value",
            failed("peregrine", "knob fingerprints did not hash"),
            0.0,
        );
    };

    // Two of the three cases must hold: same knobs agree, different knobs differ.
    let agreed = usize::from(ia == ib) + usize::from(ia != ic);
    with_cases(
        cert(
            "peregrine",
            "sampling_knobs_are_identity",
            "two requests with the same sampling knobs share an identity and two \
             with different knobs do not, which is what lets a reward \
             fingerprint separate a matched baseline from an unmatched one",
            exact(agreed == 2),
            0.0,
        ),
        2,
    )
}

// ── The event stream ────────────────────────────────────────────────────────

fn stream_flag_is_the_only_difference() -> Certificate {
    // `main.rs:977` gates the SSE path on one field. A streamed request that
    // differed in anything else would be a different experiment wearing the same
    // name, and nothing would fail.
    let history = msgs();
    let req = ChatRequest::new(&history);
    let mut agreed = 0usize;

    let (Ok(blocking), Ok(streaming)) = (build_request(&cfg(), &req), chat_stream_request(&cfg(), &req))
    else {
        return cert(
            "peregrine",
            "stream_flag_is_the_only_difference",
            "a streaming request is the blocking request plus `stream: true`",
            failed("peregrine", "the request bodies did not build"),
            0.0,
        );
    };
    let (Ok(a), Ok(b)): (Result<Value, _>, Result<Value, _>) = (
        serde_json::from_str(&blocking.body),
        serde_json::from_str(&streaming.body),
    ) else {
        return cert(
            "peregrine",
            "stream_flag_is_the_only_difference",
            "a streaming request is the blocking request plus `stream: true`",
            failed("peregrine", "the request bodies were not JSON"),
            0.0,
        );
    };

    if a.get("stream").is_none() {
        // Absent, not `false`: the server's field is `#[serde(default)]`, and a
        // request that stated `false` would be stating something the blocking
        // path never had to.
        agreed += 1;
    } else {
        eprintln!("omni-verify: peregrine: the blocking request sent a `stream` field");
    }
    if b.get("stream") == Some(&Value::Bool(true)) {
        agreed += 1;
    } else {
        eprintln!("omni-verify: peregrine: the streaming request did not send `stream: true`");
    }
    // Nothing else may differ. Adding the flag to the blocking body must land
    // exactly on the streaming one.
    let mut expected = a.clone();
    expected["stream"] = Value::Bool(true);
    if expected == b {
        agreed += 1;
    } else {
        eprintln!("omni-verify: peregrine: streaming changed more than `stream`");
    }
    if streaming.headers.iter().any(|(k, v)| k == "accept" && v == "text/event-stream") {
        agreed += 1;
    } else {
        eprintln!("omni-verify: peregrine: the streaming request did not ask for an event stream");
    }

    // The control, the other way: a body that differs in a knob the server acts
    // on must be seen as different, or the comparison above is vacuous.
    let hotter = ChatRequest { temperature: 0.7, ..ChatRequest::new(&history) };
    let colder = ChatRequest::new(&history);
    let streamed = |req: &ChatRequest<'_>| -> Option<Value> {
        let wire = chat_stream_request(&cfg(), req).ok()?;
        serde_json::from_str(&wire.body).ok()
    };
    match (streamed(&hotter), streamed(&colder)) {
        (Some(x), Some(y)) if x != y => agreed += 1,
        (Some(_), Some(_)) => {
            eprintln!("omni-verify: peregrine: two streamed requests at different temperatures built the same body");
        }
        _ => {
            eprintln!("omni-verify: peregrine: the temperature control did not build");
        }
    }

    with_cases(
        cert(
            "peregrine",
            "stream_flag_is_the_only_difference",
            "a streaming request is the blocking request with `stream: true` added and \
             nothing else changed, so a streamed turn is the same experiment as a \
             blocking one and two different requests are never mistaken for each other",
            exact(agreed == 5),
            0.0,
        ),
        5,
    )
}

fn frames_survive_read_boundaries() -> Certificate {
    // A socket hands over whatever arrived. The payloads are raw JSON, so a
    // boundary can land inside a multi-byte character, and a reader that decoded
    // each slice on its own would corrupt it — turning a working stream into a
    // JSON error that reads as "the server sent garbage".
    const TEXT: &str = "x → ünïcode";
    let streams = [
        chunk(json!({"role": "assistant"}), Value::Null)
            + &chunk(json!({"content": TEXT}), Value::Null)
            + &frame(DONE_SENTINEL),
        chunk(json!({"role": "assistant"}), Value::Null) + &frame(DONE_SENTINEL),
        chunk(whole_call(0, "call_7_0", "shell", r#"{"command":"ls"}"#), Value::Null) + &frame(DONE_SENTINEL),
        // A payload whose multi-byte characters sit at the very start and the
        // very end of the byte stream, where a reader that decoded a slice on
        // its own would have nothing to fall back on.
        format!("data: {{\"a\":\"{TEXT}\"}}\n\ndata: {DONE_SENTINEL}\n\n"),
        // Several events, so a boundary lands in the middle of one line and not
        // only at an event edge.
        chunk(json!({"reasoning_content": TEXT}), Value::Null)
            + &chunk(json!({"content": TEXT}), Value::Null)
            + &chunk(json!({}), json!("stop"))
            + &frame(DONE_SENTINEL),
    ];
    let mut agreed = 0usize;
    for (i, body) in streams.iter().enumerate() {
        match (read_frames(body), read_frames_dribbled(body)) {
            (Ok(whole), Ok(dribbled)) if whole == dribbled && !whole.is_empty() => agreed += 1,
            (Ok(whole), Ok(dribbled)) => {
                eprintln!(
                    "omni-verify: peregrine: stream {i} read whole as {} frames but one byte at a time as {}",
                    whole.len(),
                    dribbled.len()
                );
            }
            _ => eprintln!("omni-verify: peregrine: stream {i} did not parse"),
        }
    }

    // The control, the other way: two streams differing in one byte must not
    // produce the same frames, or "identical either way" would be true of
    // everything.
    let one = chunk(json!({"content": "a"}), Value::Null);
    let other = chunk(json!({"content": "b"}), Value::Null);
    let near_done = format!("data: {DONE_SENTINEL} and more\n\n");
    let (Ok(f1), Ok(f2), Ok(f3)) = (
        read_frames_dribbled(&one),
        read_frames_dribbled(&other),
        read_frames_dribbled(&near_done),
    ) else {
        return cert(
            "peregrine",
            "stream_frames_survive_read_boundaries",
            "a stream parses the same however its bytes are divided",
            failed("peregrine", "the one-byte control streams did not parse"),
            0.0,
        );
    };
    if f1 != f2 {
        agreed += 1;
    } else {
        eprintln!("omni-verify: peregrine: two different streams produced the same frames");
    }
    if f3 == vec![SseFrame::Data(format!("{DONE_SENTINEL} and more"))] {
        agreed += 1;
    } else {
        eprintln!("omni-verify: peregrine: a near-miss sentinel dribbled as {f3:?}");
    }

    with_cases(
        cert(
            "peregrine",
            "stream_frames_survive_read_boundaries",
            "an event stream parses to the same frames however its bytes are divided, \
             including a division inside a multi-byte character, so a socket boundary \
             can never corrupt a payload the server sent intact",
            exact(agreed == 7),
            0.0,
        ),
        7,
    )
}

fn keep_alive_comments_are_not_events() -> Certificate {
    // `KeepAlive::default()` is `b":\n\n"`, once every 15 seconds, and it lands
    // in the middle of a generation. A reader that took it for a payload would
    // try to parse `:` as JSON on every quiet turn.
    const KEEPALIVE: &str = ":\n\n";
    let events = [
        chunk(json!({"role": "assistant"}), Value::Null),
        chunk(json!({"content": "hi"}), Value::Null),
        chunk(json!({}), json!("stop")),
        format!("data: {DONE_SENTINEL}\n\n"),
    ];
    let plain = events.concat();
    let with_heartbeat = events
        .iter()
        .map(|e| format!("{KEEPALIVE}{e}"))
        .collect::<String>()
        + KEEPALIVE;

    let mut agreed = 0usize;
    match (read_frames(&plain), read_frames(&with_heartbeat)) {
        (Ok(quiet), Ok(beat)) if quiet == beat && !quiet.is_empty() => agreed += 1,
        (Ok(quiet), Ok(beat)) => {
            eprintln!("omni-verify: peregrine: keep-alives changed the frames: {quiet:?} vs {beat:?}");
        }
        _ => eprintln!("omni-verify: peregrine: the keep-alive fixture did not parse"),
    }
    // Only heartbeats is silence, not an event and not an error.
    match read_frames(&format!("{KEEPALIVE}{KEEPALIVE}")) {
        Ok(f) if f.is_empty() => agreed += 1,
        Ok(f) => eprintln!("omni-verify: peregrine: heartbeats alone produced {f:?}"),
        Err(e) => eprintln!("omni-verify: peregrine: heartbeats alone failed to parse: {e}"),
    }

    // The control, the other way: the check is inert *and* the parser is not
    // inert. A `data:` line still becomes a frame.
    let Ok(comment_shaped) = read_frames(KEEPALIVE) else {
        return cert(
            "peregrine",
            "keep_alive_comments_are_not_events",
            "a keep-alive comment produces no event",
            failed("peregrine", "the keep-alive fixture did not parse"),
            0.0,
        );
    };
    if comment_shaped.is_empty()
        && read_frames(&format!("data: {DONE_SENTINEL}\n\n")).unwrap_or_default()
            == vec![SseFrame::Done]
    {
        agreed += 1;
    } else {
        eprintln!("omni-verify: peregrine: a keep-alive and a data line parsed the same way");
    }

    with_cases(
        cert(
            "peregrine",
            "keep_alive_comments_are_not_events",
            "peregrine's 15-second keep-alive is an empty comment and produces no \
             event, so a quiet turn is not mistaken for a finished one and a \
             heartbeat is never handed to the JSON reader",
            exact(agreed == 3),
            0.0,
        ),
        3,
    )
}

fn multi_line_data_fields_are_joined() -> Certificate {
    // `axum`'s `Event::data` splits a payload containing a newline across
    // `data:` fields, and the rules say a reader rejoins them with `\n`. Never
    // produced by peregrine — its payloads are `serde_json::to_string`, which
    // escapes newlines — but every other server on this dialect can.
    let mut agreed = 0usize;
    let cases: [(&str, &str); 3] = [
        ("data: {\"a\":\ndata: 1}\n\n", "{\"a\":\n1}"),
        ("data: a\ndata: b\ndata: c\n\n", "a\nb\nc"),
        // The control, the other way: one field joins to itself with nothing
        // between the parts, or "joined" would be true of anything.
        ("data: {\"a\":1}\n\n", "{\"a\":1}"),
    ];
    for (body, want) in cases {
        match read_frames(body) {
            Ok(f) if f == vec![SseFrame::Data(want.to_string())] => agreed += 1,
            Ok(f) => eprintln!("omni-verify: peregrine: {body:?} parsed as {f:?}, expected {want:?}"),
            Err(e) => eprintln!("omni-verify: peregrine: {body:?} did not parse: {e}"),
        }
    }

    with_cases(
        cert(
            "peregrine",
            "multi_line_data_fields_are_joined",
            "the several `data:` lines of one event are rejoined with newlines, and a \
             single line is rejoined to itself unchanged, so a payload split across \
             fields arrives whole and a one-field event is not given a spurious newline",
            exact(agreed == 3),
            0.0,
        ),
        3,
    )
}

fn done_sentinel_is_recognised_by_payload() -> Certificate {
    // `Event::default().data("[DONE]")`. It is an ordinary `data:` payload, so
    // matching on anything but the exact string either loses a chunk or
    // terminates a stream early.
    let mut agreed = 0usize;
    let cases: [(&str, bool); 4] = [
        ("data: [DONE]\n\n", true),
        ("data: [DONE]\r\n\r\n", true),
        // The controls, the other way: payloads that merely start with the
        // sentinel are data, and the sentinel is recognised wherever it appears.
        ("data: [DONE] and then some\n\n", false),
        ("data: [DONE]x\n\n", false),
    ];
    for (body, is_done) in cases {
        match read_frames(body) {
            Ok(f) => {
                let saw_done = f == vec![SseFrame::Done];
                if saw_done == is_done {
                    agreed += 1;
                } else {
                    eprintln!("omni-verify: peregrine: {body:?} produced {f:?}");
                }
            }
            Err(e) => eprintln!("omni-verify: peregrine: {body:?} did not parse: {e}"),
        }
    }

    with_cases(
        cert(
            "peregrine",
            "done_sentinel_is_recognised_by_its_payload",
            "only the exact `data: [DONE]` payload ends a stream, and a payload that \
             merely begins with those five characters is delivered as data, so \
             finishing early and losing a chunk are both ruled out",
            exact(agreed == 4),
            0.0,
        ),
        4,
    )
}

fn a_stream_without_its_sentinel_is_refused() -> Certificate {
    // The load-bearing one. `main.rs:1054` returns from the pump task the moment
    // the engine errors, so a failed generation sends no finish chunk and no
    // `[DONE]` — and closes the socket, exactly like a finished one. Accepting
    // the end of the body records a truncated answer as the model's answer.
    let truncated = chunk(json!({"role": "assistant"}), Value::Null)
        + &chunk(json!({"content": "half a sen"}), Value::Null);
    let after_finish = chunk(json!({"content": "all of it"}), Value::Null)
        + &chunk(json!({}), json!("stop"));

    let mut agreed = 0usize;
    if refused_with(read_stream(&truncated), DONE_SENTINEL) {
        agreed += 1;
    }
    if refused_with(read_stream(&after_finish), DONE_SENTINEL) {
        agreed += 1;
    }

    // The control, the other way: the *same* stream with the sentinel appended
    // does assemble, so the refusal above is about the sentinel and not about
    // these fixtures being malformed.
    match read_stream(&(after_finish.clone() + &format!("data: {DONE_SENTINEL}\n\n"))) {
        Ok(r) if r.text() == Some("all of it") && r.finish_reason == FinishReason::Stop => agreed += 1,
        Ok(r) => eprintln!("omni-verify: peregrine: the control assembled as {r:?}"),
        Err(e) => eprintln!("omni-verify: peregrine: the control should have assembled: {e}"),
    }

    with_cases(
        cert(
            "peregrine",
            "a_stream_without_its_sentinel_is_refused",
            "a stream that ends without `data: [DONE]` is a truncated turn and is \
             refused, because peregrine's error path closes the socket exactly the \
             way a finished one does and only the sentinel tells them apart",
            exact(agreed == 3),
            0.0,
        ),
        3,
    )
}

fn a_mid_stream_error_is_an_error_not_a_turn() -> Certificate {
    // `sse_error` at `main.rs:1483`: an ordinary data frame whose only key is
    // `error`. Read as a turn it is a completion with no text and no finish
    // reason, which is indistinguishable from a model that chose to say nothing.
    let failed_midway = chunk(json!({"role": "assistant"}), Value::Null)
        + &chunk(json!({"content": "thinking"}), Value::Null)
        + &frame(&json!({"error": {"message": "kv page fault at shard 3"}}).to_string());

    let mut agreed = 0usize;
    if refused_with(read_stream(&failed_midway), "kv page fault at shard 3") {
        agreed += 1;
    }
    // The very first frame being an error is the same failure.
    if refused_with(read_stream(&frame(&json!({"error": {"message": "queue closed"}}).to_string())), "queue closed") {
        agreed += 1;
    }

    // The control, the other way: a chunk that merely has no content is not an
    // error, or this check would be passing by refusing every quiet turn.
    let clean = chunk(json!({"role": "assistant"}), Value::Null)
        + &chunk(json!({}), json!("stop"))
        + &frame(DONE_SENTINEL);
    match read_stream(&clean) {
        Ok(r) if r.finish_reason == FinishReason::Stop => agreed += 1,
        Ok(r) => eprintln!("omni-verify: peregrine: the clean control assembled as {r:?}"),
        Err(e) => eprintln!("omni-verify: peregrine: the clean control should have assembled: {e}"),
    }

    with_cases(
        cert(
            "peregrine",
            "a_mid_stream_error_is_an_error_not_a_turn",
            "peregrine's mid-stream `{\"error\":{…}}` frame becomes an error carrying \
             the server's own words rather than an empty turn, because a generation \
             that failed and one that chose to say nothing are different events and \
             only one of them may be recorded as an answer",
            exact(agreed == 3),
            0.0,
        ),
        3,
    )
}

fn an_event_after_the_sentinel_is_refused() -> Certificate {
    // peregrine sends `[DONE]` last on both the live and the memo-replay path
    // (`main.rs:1094`, `main.rs:1379`). Anything after it means the stream is not
    // what it claimed to be, and quietly accepting it would let a server append
    // a second answer to a turn that was already closed.
    let mut agreed = 0usize;
    if refused_with(
        read_stream(&(format!("data: {DONE_SENTINEL}\n\n") + &chunk(json!({"content": "late"}), Value::Null))),
        "after its [DONE]",
    ) {
        agreed += 1;
    }
    if refused_with(read_stream(&format!("data: {DONE_SENTINEL}\n\ndata: {DONE_SENTINEL}\n\n")), "after its [DONE]") {
        agreed += 1;
    }

    // The control, the other way: the sentinel last is the ordinary case.
    let good = chunk(json!({"content": "done"}), Value::Null)
        + &chunk(json!({}), json!("stop"))
        + &frame(DONE_SENTINEL);
    match read_stream(&good) {
        Ok(r) if r.text() == Some("done") => agreed += 1,
        Ok(r) => eprintln!("omni-verify: peregrine: the control assembled as {r:?}"),
        Err(e) => eprintln!("omni-verify: peregrine: the control should have assembled: {e}"),
    }

    with_cases(
        cert(
            "peregrine",
            "an_event_after_the_sentinel_is_refused",
            "nothing after `data: [DONE]` is accepted, because the sentinel closes the \
             response and a second answer appended behind it would be attributed to a \
             turn that had already finished",
            exact(agreed == 3),
            0.0,
        ),
        3,
    )
}

fn a_chunk_with_nothing_to_say_is_refused() -> Certificate {
    // Every chunk `peregrine-serve` writes carries `choices[0].delta` — there is
    // no path that writes one without it. A chunk that does not is either a
    // different dialect or a truncated frame, and skipping it is the worst
    // option available: a dropped chunk can be the one carrying the finish
    // reason, and a turn that lost its finish reason looks exactly like a clean
    // one.
    let silent = [
        frame(&json!({"id": "c", "object": "chat.completion.chunk"}).to_string()),
        frame(&json!({"choices": []}).to_string()),
        frame(&json!({"choices": [{"index": 0, "finish_reason": null}]}).to_string()),
    ];
    let mut agreed = 0usize;
    for body in silent {
        match read_stream(&(body.clone() + &format!("data: {DONE_SENTINEL}\n\n"))) {
            Err(_) => agreed += 1,
            Ok(r) => eprintln!("omni-verify: peregrine: the empty chunk {body:?} assembled as {r:?}"),
        }
    }

    // The control, the other way: a chunk that does say something is read, and
    // its finish reason is the one the reply reports.
    let talkative = chunk(json!({"role": "assistant"}), Value::Null)
        + &chunk(json!({"content": "x"}), Value::Null)
        + &chunk(json!({}), json!("length"))
        + &frame(DONE_SENTINEL);
    match read_stream(&talkative) {
        Ok(r) if r.finish_reason == FinishReason::Length => agreed += 1,
        Ok(r) => eprintln!("omni-verify: peregrine: the control assembled as {r:?}"),
        Err(e) => eprintln!("omni-verify: peregrine: the control should have assembled: {e}"),
    }

    with_cases(
        cert(
            "peregrine",
            "a_chunk_with_nothing_to_say_is_refused",
            "a chunk carrying neither `choices[0]` nor a `delta` is refused rather \
             than skipped, because the one chunk a reader must not lose silently is \
             the one carrying the finish reason",
            exact(agreed == 4),
            0.0,
        ),
        4,
    )
}

fn tool_call_fragments_reassemble_by_index() -> Certificate {
    // peregrine emits one whole call per delta (`main.rs:1461`: "this emits it
    // complete in a single delta instead"), because the markup is not a valid
    // call until its closing tag arrives. The OpenAI dialect splits the same
    // call across deltas with the arguments cut mid-token, and any other server
    // on this dialect may do it. Parsing a fragment is how a client turns a split
    // token into a JSON error halfway through a call that is merely not finished.
    let whole = chunk(whole_call(0, "call_7_0", "shell", r#"{"command":"ls -la"}"#), Value::Null)
        + &chunk(json!({}), json!("tool_calls"))
        + &frame(DONE_SENTINEL);
    let split = chunk(whole_call(0, "call_7_0", "shell", ""), Value::Null)
        + &chunk(json!({"tool_calls": [{"index": 0, "function": {"arguments": "{\"comm"}}]}), Value::Null)
        + &chunk(json!({"tool_calls": [{"index": 0, "function": {"arguments": "and\":\"ls -la\"}"}}]}), Value::Null)
        + &chunk(json!({}), json!("tool_calls"))
        + &frame(DONE_SENTINEL);
    // Two calls, out of order, correlated by index rather than by arrival.
    let interleaved = chunk(whole_call(1, "call_7_1", "echo", r#"{"text":"b"}"#), Value::Null)
        + &chunk(whole_call(0, "call_7_0", "shell", r#"{"command":"ls"}"#), Value::Null)
        + &chunk(json!({}), json!("tool_calls"))
        + &frame(DONE_SENTINEL);

    let mut agreed = 0usize;
    match (read_stream(&whole), read_stream(&split)) {
        (Ok(a), Ok(b)) if a == b && a.message.tool_calls().len() == 1 => agreed += 1,
        (Ok(a), Ok(b)) => eprintln!("omni-verify: peregrine: a split call is not the whole call: {a:?} vs {b:?}"),
        _ => eprintln!("omni-verify: peregrine: the tool-call fixtures did not assemble"),
    }
    match read_stream(&interleaved) {
        Ok(r) => {
            let names: Vec<&str> = r.message.tool_calls().iter().map(|c| c.name.as_str()).collect();
            if names == vec!["shell", "echo"] {
                agreed += 1;
            } else {
                eprintln!("omni-verify: peregrine: calls came back as {names:?}, expected index order");
            }
        }
        Err(e) => eprintln!("omni-verify: peregrine: the interleaved fixture did not assemble: {e}"),
    }
    // A fragment with no id must not erase the id the first one established.
    let no_id = chunk(whole_call(0, "call_7_0", "shell", ""), Value::Null)
        + &chunk(json!({"tool_calls": [{"index": 0, "function": {"arguments": "{\"a\":1}"}}]}), Value::Null)
        + &chunk(json!({}), json!("tool_calls"))
        + &frame(DONE_SENTINEL);
    match read_stream(&no_id) {
        Ok(r) if r.message.tool_calls()[0].id == "call_7_0" => agreed += 1,
        Ok(r) => eprintln!("omni-verify: peregrine: a bare fragment erased the call id: {r:?}"),
        Err(e) => eprintln!("omni-verify: peregrine: the bare-fragment fixture did not assemble: {e}"),
    }

    // The controls, the other way: fragments that *disagree* are not joined by
    // picking one. A fragment with no index cannot be correlated at all.
    let two_ids = chunk(whole_call(0, "call_7_0", "shell", "{}"), Value::Null)
        + &chunk(json!({"tool_calls": [{"index": 0, "id": "call_9_9", "function": {"arguments": ""}}]}), Value::Null)
        + &chunk(json!({}), json!("tool_calls"))
        + &frame(DONE_SENTINEL);
    let two_names = chunk(whole_call(0, "call_7_0", "shell", "{}"), Value::Null)
        + &chunk(json!({"tool_calls": [{"index": 0, "function": {"name": "rm", "arguments": ""}}]}), Value::Null)
        + &chunk(json!({}), json!("tool_calls"))
        + &frame(DONE_SENTINEL);
    let no_index = chunk(
        json!({"tool_calls": [{"id": "call_7_0", "type": "function",
            "function": {"name": "shell", "arguments": "{}"}}]}),
        Value::Null,
    ) + &frame(DONE_SENTINEL);
    for (label, body) in [("two ids", &two_ids), ("two names", &two_names), ("no index", &no_index)] {
        match read_stream(body) {
            Err(_) => agreed += 1,
            Ok(r) => eprintln!("omni-verify: peregrine: {label} assembled as {r:?}"),
        }
    }

    with_cases(
        cert(
            "peregrine",
            "tool_call_fragments_reassemble_by_index",
            "a tool call split across chunks rejoins by index into the same call \
             peregrine sends whole, calls come back in index order rather than \
             arrival order, and fragments that contradict each other are refused \
             rather than joined by picking one",
            exact(agreed == 6),
            0.0,
        ),
        6,
    )
}

fn a_stream_reports_no_usage_not_a_fabricated_one() -> Certificate {
    // `usage` appears exactly once in all of `peregrine-serve`: the
    // non-streaming body at `main.rs:1252`. No streaming chunk carries it, and
    // `stream_options` — the OpenAI field that asks for it — is not in the
    // server's request struct at all. Zeros would read as "this turn was
    // free", which is a claim about cost that nobody measured.
    let text_stream = chunk(json!({"role": "assistant"}), Value::Null)
        + &chunk(json!({"content": "2 + 2 = 4"}), Value::Null)
        + &chunk(json!({}), json!("stop"))
        + &frame(DONE_SENTINEL);
    let call_stream = chunk(whole_call(0, "call_7_0", "shell", r#"{"command":"ls"}"#), Value::Null)
        + &chunk(json!({}), json!("tool_calls"))
        + &frame(DONE_SENTINEL);

    let mut agreed = 0usize;
    for body in [&text_stream, &call_stream] {
        match read_stream(body) {
            Ok(r) if r.usage.is_none() => agreed += 1,
            Ok(r) => eprintln!("omni-verify: peregrine: a stream reported usage it was never sent: {r:?}"),
            Err(e) => eprintln!("omni-verify: peregrine: the usage fixtures did not assemble: {e}"),
        }
    }

    // The control, the other way: the same generation on the JSON path *does*
    // report its tokens, so the absence above is a property of the streaming
    // dialect rather than of this parser losing the field.
    let json_reply = json!({
        "choices": [{"message": {"role": "assistant", "content": "2 + 2 = 4"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 9, "completion_tokens": 4, "total_tokens": 13}
    })
    .to_string();
    match parse_reply(&json_reply) {
        Ok(r) if r.usage.map(|u| u.total()) == Some(13) => agreed += 1,
        Ok(r) => eprintln!("omni-verify: peregrine: the JSON control reported {r:?}"),
        Err(e) => eprintln!("omni-verify: peregrine: the JSON control should have parsed: {e}"),
    }

    with_cases(
        cert(
            "peregrine",
            "a_stream_reports_no_usage_not_a_fabricated_one",
            "a streamed reply reports no token usage, because peregrine sends usage \
             only on the JSON path, and reports none rather than zeros so a caller \
             can tell an uncounted turn from a free one",
            exact(agreed == 3),
            0.0,
        ),
        3,
    )
}

fn a_tool_calling_stream_carries_no_text() -> Certificate {
    // `chunk_event` with `delta: None` writes `"delta": {}`, so a tool-calling
    // turn streams no `content` field at all. Collapsing that to `""` would make
    // "the model called a tool" indistinguishable from "the model said the empty
    // string", which is the distinction the whole `Option<String>` on
    // `Message::Assistant` exists to keep.
    let only_a_call = chunk(json!({"role": "assistant"}), Value::Null)
        + &chunk(whole_call(0, "call_7_0", "shell", r#"{"command":"ls"}"#), Value::Null)
        + &chunk(json!({}), json!("tool_calls"))
        + &frame(DONE_SENTINEL);
    let said_nothing = chunk(json!({"role": "assistant"}), Value::Null)
        + &chunk(json!({}), json!("stop"))
        + &frame(DONE_SENTINEL);

    let mut agreed = 0usize;
    for body in [&only_a_call, &said_nothing] {
        match read_stream(body) {
            Ok(r) if r.text().is_none() => agreed += 1,
            Ok(r) => eprintln!("omni-verify: peregrine: a silent turn reported text: {r:?}"),
            Err(e) => eprintln!("omni-verify: peregrine: the silent fixtures did not assemble: {e}"),
        }
    }

    // The control, the other way: a stream that does speak produces exactly
    // what it said, so `None` is a statement about this turn and not a parser
    // that drops text.
    let spoke = chunk(json!({"content": "  "}), Value::Null)
        + &chunk(json!({"content": "4"}), Value::Null)
        + &chunk(json!({}), json!("stop"))
        + &frame(DONE_SENTINEL);
    match read_stream(&spoke) {
        Ok(r) if r.text() == Some("  4") => agreed += 1,
        Ok(r) => eprintln!("omni-verify: peregrine: the control reported text {:?}", r.text()),
        Err(e) => eprintln!("omni-verify: peregrine: the control should have assembled: {e}"),
    }

    with_cases(
        cert(
            "peregrine",
            "a_tool_calling_stream_carries_no_text",
            "a streamed turn that says nothing reports no text rather than an empty \
             string, so a tool-calling turn stays distinguishable from one that \
             spoke, while a turn that does speak reports exactly what it said",
            exact(agreed == 3),
            0.0,
        ),
        3,
    )
}

fn truncated_tool_arguments_are_refused() -> Certificate {
    // `arguments` is a JSON *string* holding an object (`tools.rs:79`), and it is
    // only parseable once every fragment has arrived. Defaulting a broken one to
    // `{}` produces a tool call that runs nothing and reports success — the
    // quietest wrong answer in this file.
    let broken = [
        r#"{"comm"#,
        r#"[1,2]"#,
        "",
        "null",
    ];
    let mut agreed = 0usize;
    for arguments in broken {
        let body = chunk(whole_call(0, "call_7_0", "shell", arguments), Value::Null)
            + &chunk(json!({}), json!("tool_calls"))
            + &frame(DONE_SENTINEL);
        match read_stream(&body) {
            Err(_) => agreed += 1,
            Ok(r) => eprintln!(
                "omni-verify: peregrine: arguments {arguments:?} assembled as {}",
                r.message.tool_calls()[0].arguments
            ),
        }
    }

    // The control, the other way: well-formed arguments still parse.
    let good = chunk(whole_call(0, "call_7_0", "shell", r#"{"command":"ls -la"}"#), Value::Null)
        + &chunk(json!({}), json!("tool_calls"))
        + &frame(DONE_SENTINEL);
    match read_stream(&good) {
        Ok(r) => {
            if r.message.tool_calls()[0].arguments.get("command") == Some(&json!("ls -la")) {
                agreed += 1;
            } else {
                eprintln!("omni-verify: peregrine: the control parsed as {}", r.message.tool_calls()[0].arguments);
            }
        }
        Err(e) => eprintln!("omni-verify: peregrine: the control should have assembled: {e}"),
    }

    with_cases(
        cert(
            "peregrine",
            "truncated_tool_arguments_are_refused",
            "a streamed tool call whose joined arguments are not a JSON object is \
             refused rather than defaulted to `{}`, which would be a call that runs \
             nothing and reports success",
            exact(agreed == 5),
            0.0,
        ),
        5,
    )
}

fn finish_reason_agrees_with_the_calls_announced() -> Certificate {
    // `main.rs:1091`: `let finish = if emitted_calls > 0 { "tool_calls" } else { "stop" };`
    // — and identically at `main.rs:1377` on the replay path. The finish reason
    // is *derived* from the calls, so the two cannot disagree. When they do, one
    // of them is about to be believed and it is the wrong one: a `tool_calls`
    // turn with nothing to run, or a `stop` turn whose calls were dropped.
    let mut agreed = 0usize;
    let contradicted = [
        chunk(json!({}), json!("tool_calls")) + &frame(DONE_SENTINEL),
        chunk(whole_call(0, "call_7_0", "shell", "{}"), Value::Null)
            + &chunk(json!({}), json!("stop"))
            + &frame(DONE_SENTINEL),
    ];
    for body in contradicted {
        match read_stream(&body) {
            Err(_) => agreed += 1,
            Ok(r) => eprintln!("omni-verify: peregrine: a contradicted finish reason assembled as {r:?}"),
        }
    }

    // The controls, the other way: the two agreeing pairs are the ordinary ones.
    let with_call = chunk(whole_call(0, "call_7_0", "shell", "{}"), Value::Null)
        + &chunk(json!({}), json!("tool_calls"))
        + &frame(DONE_SENTINEL);
    let without_call = chunk(json!({"content": "x"}), Value::Null)
        + &chunk(json!({}), json!("stop"))
        + &frame(DONE_SENTINEL);
    match read_stream(&with_call) {
        Ok(r) if r.finish_reason.wants_tools() && r.message.tool_calls().len() == 1 => agreed += 1,
        Ok(r) => eprintln!("omni-verify: peregrine: the tool_calls control assembled as {r:?}"),
        Err(e) => eprintln!("omni-verify: peregrine: the tool_calls control should have assembled: {e}"),
    }
    match read_stream(&without_call) {
        Ok(r) if !r.finish_reason.wants_tools() && r.message.tool_calls().is_empty() => agreed += 1,
        Ok(r) => eprintln!("omni-verify: peregrine: the stop control assembled as {r:?}"),
        Err(e) => eprintln!("omni-verify: peregrine: the stop control should have assembled: {e}"),
    }

    with_cases(
        cert(
            "peregrine",
            "finish_reason_agrees_with_the_calls_announced",
            "a streamed `finish_reason` and the tool calls it announces are accepted \
             only when they agree, because peregrine derives the one from the other \
             and a mismatch means a chunk was lost rather than that the model chose",
            exact(agreed == 4),
            0.0,
        ),
        4,
    )
}

fn a_null_finish_reason_is_not_the_end() -> Certificate {
    // `main.rs:1441` writes `"finish_reason": null` on every chunk but the last.
    // Reading that as `stop` — or as `unknown` — ends the turn at the first
    // token, and an agent streaming a long answer would show four words and call
    // it finished.
    let body = chunk(json!({"role": "assistant"}), Value::Null)
        + &chunk(json!({"content": "The "}), Value::Null)
        + &chunk(json!({"content": "answer "}), Value::Null)
        + &chunk(json!({"content": "is 4."}), Value::Null)
        + &chunk(json!({}), json!("stop"))
        + &frame(DONE_SENTINEL);

    let mut agreed = 0usize;
    let (Ok(reply), Ok(chunks)) = (read_stream(&body), read_chunks(&body)) else {
        return cert(
            "peregrine",
            "a_null_finish_reason_is_not_the_end",
            "`finish_reason: null` does not end a stream",
            failed("peregrine", "the fixture did not parse"),
            0.0,
        );
    };
    if reply.text() == Some("The answer is 4.") {
        agreed += 1;
    } else {
        eprintln!("omni-verify: peregrine: the reply stopped early at {:?}", reply.text());
    }
    let finishes = chunks.iter().filter(|c| matches!(c, StreamChunk::Finish(_))).count();
    let deltas = chunks.iter().filter(|c| matches!(c, StreamChunk::Text(_))).count();
    if finishes == 1 && deltas == 3 {
        agreed += 1;
    } else {
        eprintln!("omni-verify: peregrine: {deltas} deltas produced {finishes} finish events, expected 3 and 1");
    }

    // The controls, the other way: a finish reason the stream *does* name is
    // kept, not dropped for being the last chunk.
    for (wire, want) in [(json!("length"), FinishReason::Length), (json!("content_filter"), FinishReason::ContentFilter)] {
        let named = chunk(json!({"content": "x"}), Value::Null) + &chunk(json!({}), wire) + &frame(DONE_SENTINEL);
        match read_stream(&named) {
            Ok(r) if r.finish_reason == want => agreed += 1,
            Ok(r) => eprintln!("omni-verify: peregrine: {want:?} came back as {:?}", r.finish_reason),
            Err(e) => eprintln!("omni-verify: peregrine: the {want:?} control did not assemble: {e}"),
        }
    }

    with_cases(
        cert(
            "peregrine",
            "a_null_finish_reason_is_not_the_end",
            "`finish_reason: null`, which every chunk but the last carries, produces no \
             finish event at all, so a streamed answer runs to the sentinel rather \
             than stopping at its first token",
            exact(agreed == 4),
            0.0,
        ),
        4,
    )
}

fn only_an_event_stream_content_type_is_one() -> Certificate {
    // The content type is the only thing separating the two dialects on a 200.
    // Checking it turns "the body was JSON when I asked for events" from a parse
    // error two hundred lines into a stream into one sentence where it happened.
    let mut agreed = 0usize;
    for wrong in ["application/json", "text/html", "", "application/json; charset=utf-8"] {
        if is_event_stream(wrong) {
            eprintln!("omni-verify: peregrine: {wrong:?} was accepted as an event stream");
        } else {
            agreed += 1;
        }
    }

    // The controls, the other way: the real type is accepted, with or without
    // parameters and in either case.
    for right in ["text/event-stream", "text/event-stream; charset=utf-8", "TEXT/EVENT-STREAM"] {
        if is_event_stream(right) {
            agreed += 1;
        } else {
            eprintln!("omni-verify: peregrine: {right:?} was refused as an event stream");
        }
    }

    with_cases(
        cert(
            "peregrine",
            "only_an_event_stream_content_type_is_one",
            "a streaming request accepts a 200 only when its content type is an event \
             stream, parameters and case aside, so an answer that arrived on the JSON \
             path is refused where it happened instead of as a parse error later",
            exact(agreed == 7),
            0.0,
        ),
        7,
    )
}

fn reasoning_content_is_not_answered_text() -> Certificate {
    // `reasoning_chunk_event` at `main.rs:1446` puts the model's thinking in
    // `delta.reasoning_content`, a field OpenAI does not have. Folding it into
    // `content` would record the model's scratchpad as its answer — and a
    // harness that scores the difference would be scoring its own notes.
    let with_thinking = chunk(json!({"reasoning_content": "2 + 2 is 4, obviously"}), Value::Null)
        + &chunk(json!({"content": "4"}), Value::Null)
        + &chunk(json!({}), json!("stop"))
        + &frame(DONE_SENTINEL);
    let thinking_only = chunk(json!({"reasoning_content": "still working on it"}), Value::Null)
        + &chunk(json!({}), json!("stop"))
        + &frame(DONE_SENTINEL);

    let mut agreed = 0usize;
    match read_stream(&thinking_only) {
        Ok(r) if r.text().is_none() => agreed += 1,
        Ok(r) => eprintln!("omni-verify: peregrine: reasoning became the answer: {:?}", r.text()),
        Err(e) => eprintln!("omni-verify: peregrine: the reasoning-only fixture did not assemble: {e}"),
    }
    match read_stream(&with_thinking) {
        Ok(r) if r.text() == Some("4") => agreed += 1,
        Ok(r) => eprintln!("omni-verify: peregrine: the mixed turn reported {:?}", r.text()),
        Err(e) => eprintln!("omni-verify: peregrine: the mixed fixture did not assemble: {e}"),
    }

    // The control, the other way: a stream with no reasoning reports exactly its
    // content, so `None` above is about the turn and not about a reader that
    // drops text.
    let plain = chunk(json!({"content": "4"}), Value::Null)
        + &chunk(json!({}), json!("stop"))
        + &frame(DONE_SENTINEL);
    match read_stream(&plain) {
        Ok(r) if r.text() == Some("4") => agreed += 1,
        Ok(r) => eprintln!("omni-verify: peregrine: the plain control reported {:?}", r.text()),
        Err(e) => eprintln!("omni-verify: peregrine: the plain control did not assemble: {e}"),
    }

    with_cases(
        cert(
            "peregrine",
            "reasoning_content_is_not_answered_text",
            "`reasoning_content` is delivered as its own event and never folded into \
             the answer, so a harness scoring `content` never scores the model's \
             scratchpad while still seeing every byte the stream sent",
            exact(agreed == 3),
            0.0,
        ),
        3,
    )
}

fn a_body_cut_off_mid_frame_is_refused() -> Certificate {
    // The framing half of ending, and the half `a_stream_without_its_sentinel_is_refused`
    // cannot see. There, the body ended *between* events and the dialect never
    // said `[DONE]`; here the body ends *inside* one, so the bytes the server
    // wrote are not bytes anyone received. `SseParser::finish` refuses rather
    // than discarding the half-frame the way the event-stream rules allow,
    // because discarding it drops part of an answer without saying so — and a
    // socket closing mid-write is what a proxy in front of the server does to a
    // long generation.
    const CUT: &[&str] = &[
        // No newline at all: the payload line never terminated.
        "data: {\"id\":\"chatcmpl-7\",\"object\":\"chat.completion.chunk\"",
        // Cut inside the first and only event.
        "data: {\"choices\":[{\"index\":0,\"de",
        // A whole event delivered, then the next one cut: the frames that did
        // arrive are kept and the tail is still refused.
        "data: {\"a\":1}\n\ndata: {\"b\":",
        // A proxy rewriting line endings does not make the tail any less cut.
        "data: [DONE]\r\n\r\ndata: {\"c\":",
    ];
    let mut agreed = 0usize;
    for body in CUT {
        if framing_refused_with(body.as_bytes(), "mid-frame") {
            agreed += 1;
        }
    }

    // The two ways a stream can end early are told apart, so a failure can be
    // attributed: "mid-frame" is the framing, the sentinel is the dialect. The
    // fixture is a well-formed chunk, because a body the *chunk reader* refuses
    // would prove nothing about which of the two ended it.
    let between = chunk(json!({"content": "all of it"}), Value::Null);
    match read_stream(&between) {
        Err(e) => {
            let msg = e.to_string();
            if msg.contains(DONE_SENTINEL) && !msg.contains("mid-frame") {
                agreed += 1;
            } else {
                eprintln!("omni-verify: peregrine: a stream that ended between events reported {msg}");
            }
        }
        Ok(r) => eprintln!("omni-verify: peregrine: a stream with no sentinel assembled as {r:?}"),
    }

    // The controls, the other way: a body that ends *on* an event boundary is
    // not a cut body, and an empty body has nothing for the framing to say —
    // both close cleanly, or the refusal above would be "the body ended".
    let clean = chunk(json!({"content": "all of it"}), Value::Null)
        + &chunk(json!({}), json!("stop"))
        + &frame(DONE_SENTINEL);
    match read_stream(&clean) {
        Ok(r) if r.text() == Some("all of it") => agreed += 1,
        Ok(r) => eprintln!("omni-verify: peregrine: the clean control assembled as {r:?}"),
        Err(e) => eprintln!("omni-verify: peregrine: the clean control should have assembled: {e}"),
    }
    let mut empty = SseParser::new();
    match empty.feed(b"").and_then(|_| empty.finish()) {
        Ok(()) => agreed += 1,
        Err(e) => eprintln!("omni-verify: peregrine: an empty body failed to close cleanly: {e}"),
    }

    with_cases(
        cert(
            "peregrine",
            "a_body_cut_off_mid_frame_is_refused",
            "a body that ends inside an event is refused rather than half-delivered, and \
             told apart from a body that ended cleanly without its sentinel, while a body \
             that does end on an event boundary still assembles",
            exact(agreed == 7),
            0.0,
        ),
        7,
    )
}

fn an_undecodable_line_is_refused_not_decoded_lossily() -> Certificate {
    // Lines are buffered as bytes and decoded once terminated, so a multi-byte
    // character split across two reads is held rather than corrupted
    // (`stream_frames_survive_read_boundaries`). This is the other side of that
    // guarantee: a line that *did* terminate and is still not UTF-8 is refused.
    // The tempting alternative is `String::from_utf8_lossy`, which substitutes
    // U+FFFD and keeps reading — and a U+FFFD inside a JSON string still parses,
    // so the harness would record a mangled byte as something the model said.
    let mut agreed = 0usize;
    for body in [
        // An invalid byte in the middle of the answer. Written as bytes, because
        // no `&str` can hold one.
        &b"data: {\"content\":\"a\xffb\"}\n\n"[..],
        // A lone continuation byte: the tail of a character whose head is gone.
        &b"data: [DONE]\n\ndata: {\"content\":\"\xc3\"}\n\n"[..],
        // An invalid byte before any payload, so nothing was ever assembled.
        &b"data: \x80\n\n"[..],
    ] {
        if framing_refused_with(body, "UTF-8") {
            agreed += 1;
        }
    }

    // The controls, the other way: the same characters, complete, arrive
    // exactly — so the refusal is about bytes that cannot be decoded and not
    // about refusing everything above ASCII.
    for payload in ["\u{2192} two bytes", "\u{1f980} four bytes"] {
        let want = json!({ "content": payload }).to_string();
        match read_frames(&frame(&want)) {
            Ok(f) if f == vec![SseFrame::Data(want)] => agreed += 1,
            Ok(f) => eprintln!("omni-verify: peregrine: {payload:?} arrived as {f:?}"),
            Err(e) => eprintln!("omni-verify: peregrine: {payload:?} should have parsed: {e}"),
        }
    }

    with_cases(
        cert(
            "peregrine",
            "an_undecodable_line_is_refused_not_decoded_lossily",
            "a completed stream line that is not valid UTF-8 is refused rather than decoded \
             with replacement characters, because a substituted byte inside a JSON payload \
             still parses and would be recorded as something the model said",
            exact(agreed == 5),
            0.0,
        ),
        5,
    )
}

#[cfg(test)]
mod tests {
    use crate::run_group;

    #[test]
    fn the_group_runs_and_passes() -> Result<(), String> {
        let report = run_group("peregrine").map_err(|e| e.to_string())?;
        assert!(report.passed(), "{}", report.render());
        Ok(())
    }
}