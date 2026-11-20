//! The `dblocks` backend for the `Model` trait.
//!
//! # The honest version first
//!
//! unifur ships no agent-capable checkpoint of its own. What exists is a
//! 4.6 MB set of Burn `.mpk` checkpoints for 2-D geometric scenes (6 points,
//! 5 question kinds), a 22 GB `/srv` registry of *routing classifiers* whose
//! `router_top1` is near 1.0 and whose job is to classify a prompt rather
//! than answer it, and `quality_coder/`, whose own module header says it
//! "does not load any model, run any network calls, or spawn any subprocess".
//! Pointing any of those at `[`Dblocks::new`]` would compile; its answers
//! would be a `Model`-shaped wrapper around noise.
//!
//! So the `dblocks` cargo feature is off by default, and the default build
//! is the honest one. See [`CONTRACT`](../../../docs/CONTRACT.md) "What is
//! not true yet".
//!
//! # What is true, when the feature is on
//!
//! With `--features dblocks`, this crate exposes [`Dblocks`], a `Model`
//! implementation that shells out to a compiled `dblocks lm generate`
//! binary per `chat` call:
//!
//! ```ignore
//! use omni_model_unifur::Dblocks;
//! use omni_model::prelude::*;
//!
//! let mut backend = Dblocks::new(
//!     "/path/to/dblocks",
//!     "/path/to/checkpoint.mpk",
//! )?;
//! let msgs = [Message::user("hi")];
//! let reply = backend.chat(&ChatRequest::new(&msgs))?;
//! # Ok::<(), omni_core::Error>(())
//! ```
//!
//! Design choices, stated once here so they do not need to be guessed from
//! the code:
//!
//! * **Shelling out, not linking.** unifur is a large workspace with its own
//!   toolchain pin; building it into omni would double the compile cost and
//!   couple two trees whose joined-at-arm's-length arrangement is the point.
//!   The same coupling arrangement the sibling `mesh-shim` uses.
//! * **Per-request subprocess.** Each `chat` spawns its own `dblocks`.
//!   Model-loading cost is real (a 200 KiB toy checkpoint cold-loads in ~0.6s
//!   on this laptop); a future backend that cares about latency would reuse a
//!   long-lived dblocks via a line-oriented protocol, but that is a change to
//!   dblocks, not to this crate.
//! * **Argv, not shell.** Prompts go through `Command::args`, never through a
//!   shell string. Nothing is interpolated; a prompt containing shell
//!   metacharacters is not an injection surface.
//! * **Prompt rendering.** Multi-turn requests are concatenated with `[role]`
//!   tags, same convention `mesh-shim` uses on the HTTP side. `dblocks` has
//!   no multi-turn state of its own, so the concatenation is the entire
//!   history by design.
//! * **Fence parser.** `dblocks lm generate` prints architecture + decode
//!   lines, then the generated continuation between two `---` fences, with
//!   the prompt echoed at the head of the fenced block. [`Dblocks::extract`]
//!   is the smallest robust parser: the fenced block, with the prompt
//!   stripped once as a prefix if it matches. A missing fence is
//!   [`omni_core::Error::Format`] rather than a silent empty reply.
//! * **Usage and finish_reason.** `dblocks` reports neither over stdout, so
//!   `usage` is `None` (a backend that invents counts is misreporting) and
//!   `finish_reason` is a 4-chars-per-token heuristic: if the returned text
//!   is at least `max_tokens * 4` bytes, the turn is tagged `Length`,
//!   otherwise `Stop`. Documented, so a caller that needs real counts knows
//!   not to trust them.
//!
//! What is still not true even with the feature on:
//!
//! * **No checkpoint is shipped.** The user must provide one. On this
//!   laptop a toy lives at `nina-data/tiny-model/lm-*.mpk` (minted by
//!   `dblocks lm train --tiny --steps 5`); its answers are noise, which is
//!   exactly what a toy training run produces.
//! * **No `nina chat` path.** This crate wraps `lm generate` only. The
//!   routing-in-the-loop path lives behind `dblocks nina chat`, which needs
//!   a registry (`registry.json` + `models/` + `logs/`) not present on this
//!   laptop. When the registry lands, a second backend variant can sit
//!   beside this one — the fence parser does not apply to `nina chat` output.
//! * **No tool calls.** `lm generate` is a prompt-to-text path. A request
//!   with `tools` is accepted, the specs are ignored, and `finish_reason`
//!   is never `ToolCalls`.

#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
#![forbid(unsafe_code)]

#[cfg(feature = "dblocks")]
mod backend;

#[cfg(feature = "dblocks")]
pub use backend::{Dblocks, Sampling};
