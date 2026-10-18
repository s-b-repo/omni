//! `omni-model-peregrine` — the OpenAI-compatible client for `peregrine-serve`.
//!
//! peregrine is designed to sit behind an agent harness: it speaks the OpenAI
//! chat-completions dialect, accepts tool schemas, and returns tool calls in the
//! OpenAI shape. This crate is omni's side of that.
//!
//! # Why the dialect is split out from the transport
//!
//! peregrine's dialect is *nearly* OpenAI's, and the differences are the part
//! worth being careful about:
//!
//! | | OpenAI | peregrine |
//! |---|---|---|
//! | `temperature` default | 1.0 | **0.0** |
//! | `top_p` default | 1.0 | **0.95** |
//! | `top_k` | not part of the dialect | sent, **applied before `top_p`** |
//! | `tool_choice` | `none`/`auto`/`required`/named | only **`"none"`** does anything; `"required"` is ignored |
//! | `model`, `n`, `stop`, `seed`, `logprobs`, penalties, `response_format` | meaningful | **accepted and discarded** |
//! | auth | bearer | bearer on `/v1/*`; `/health` and `/metrics` open |
//! | overload | varies | `503` `overloaded_error` |
//!
//! An agent that assumes the left column produces different behaviour on
//! peregrine than it did on the API it was developed against — and nothing
//! fails, so the difference shows up weeks later as "the model got worse".
//!
//! So all of it is in [`wire`], a pure function of the request, checked by the
//! `peregrine` certificate group. [`client`] only sends bytes and classifies
//! what comes back.
//!
//! # Streaming
//!
//! [`Peregrine::stream`] opens an SSE stream and yields [`wire::StreamChunk`]s
//! as they arrive. It is a second entry point rather than a [`omni_model::Model`]
//! implementation, because [`omni_model::Model::chat`] returns a *completed* turn
//! and streaming has no completed turn to return yet.
//!
//! The dialect it reads was taken from `peregrine-serve`'s own source, and it is
//! worth knowing that `peregrine-serve` never sends `usage` on a stream — token
//! counts are a property of the JSON path only — and that its error path ends the
//! stream without a finish chunk or a `[DONE]`, so the end of the body is not the
//! end of the turn.
//!
//! ```
//! use omni_model::prelude::*;
//! use omni_model_peregrine::{Config, chat_stream_request};
//!
//! let messages = [Message::user("What is 2+2?")];
//! let req = ChatRequest::new(&messages);
//! let wire = chat_stream_request(&Config::default(), &req)?;
//!
//! let body: serde_json::Value = serde_json::from_str(&wire.body)?;
//! assert_eq!(body["stream"], true, "the only difference from the blocking path");
//! assert_eq!(body["temperature"], 0.0, "still sent explicitly");
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! ```
//! use omni_model::prelude::*;
//! use omni_model_peregrine::{Config, build_request};
//!
//! let messages = [Message::user("What is 2+2?")];
//! let req = ChatRequest::new(&messages);
//! let wire = build_request(&Config::default(), &req)?;
//!
//! let body: serde_json::Value = serde_json::from_str(&wire.body)?;
//! assert_eq!(body["temperature"], 0.0, "peregrine's default, sent explicitly");
//! assert_eq!(wire.url, "http://127.0.0.1:8080/v1/chat/completions");
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

pub mod client;
pub mod wire;

pub use client::{ChatStream, Peregrine};
pub use wire::{
    Config, DONE_SENTINEL, SseFrame, SseParser, StreamAccumulator, StreamChunk, ToolCallFragment,
    WireRequest, build_request, chat_stream_request, chunk_from_frame, completions_url,
    extract_error, is_event_stream, parse_reply,
};