//! `omni-core` — the vocabulary every other `omni` crate speaks.
//!
//! An agent harness whose runtime is itself editable is a harness with more
//! seams than a normal one: hooks cross into a sandbox, evidence crosses to a
//! context runtime, model traffic crosses to a server, and every one of those
//! boundaries needs to answer the same three questions — *what shape is this?*
//! *which version?* *who caused the failure?* This crate holds those answers so
//! that they are stated once.
//!
//! # The pieces
//!
//! | Module | What it settles |
//! |---|---|
//! | [`error`] | one closed [`Error`] enum and the [`Context`](error::Context) trait, so a failure is classifiable and a call site names the operation without discarding the cause |
//! | [`contract`] | the version gate every cross-process message passes through, and the context message shapes already deployed at version 1 |
//! | [`canonical`] | one byte string per value, so a hash means something |
//! | [`ids`] | content-derived ids and a logical clock, so "the same work" is nameable |
//! | [`atomic`] | write-then-rename, so a polling reader never sees a partial envelope |
//!
//! # What is deliberately not here
//!
//! No async runtime, no HTTP client, no model, no interpreter. Those belong to
//! `omni-model`, `omni-hook` and their neighbours. This crate's job is to be
//! depended on by all of them without dragging any of them along — a core that
//! pulls in a tokio runtime stops being core the moment two consumers disagree
//! about the runtime.
//!
//! ```
//! use omni_core::{Clock, ContextQuery, content_id_str};
//!
//! let query = ContextQuery::new("what is the server ip?", 6, "pentest");
//! query.validate()?;
//!
//! // Content ids are stable across processes, so the same question asked in two
//! // runs produces the same identity — which is what makes a baseline and a
//! // patched run comparable afterwards.
//! assert_eq!(content_id_str("what is the server ip?"), content_id_str("what is the server ip?"));
//! assert_eq!(Clock::new().tick(), 1);
//! # Ok::<(), omni_core::Error>(())
//! ```

#![forbid(unsafe_code)]
#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

pub mod atomic;
pub mod canonical;
pub mod contract;
pub mod error;
pub mod ids;

pub use atomic::{write_atomic, write_atomic_if_absent};
pub use canonical::{canonical_bytes, canonicalize, value_id};
pub use contract::{
    CHANNEL_ENVELOPE_VERSION, CONTRACT_FILE, CONTEXT_HIT_VERSION, CONTEXT_PUT_VERSION,
    CONTEXT_QUERY_VERSION, ContextHit, ContextPut, ContextQuery, HookEpisode, HookPacket,
    INTEROP_DIR, MEMORY_APPEND_VERSION, MemoryAppend, PROTOCOL, Provenance, VERSIONED_MESSAGES,
    check_version, validate_id,
};
pub use error::{Context, Error, note_advisory};
pub use ids::{Clock, content_id, content_id_str};