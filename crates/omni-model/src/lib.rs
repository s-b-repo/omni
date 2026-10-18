//! `omni-model` — the one seam every backend implements.
//!
//! An agent harness that can talk to a model in five different ways is a harness
//! whose guarantees apply to none of them. So there is exactly one way here: the
//! [`Model`] trait, plus the vocabulary that surrounds it — [`Message`],
//! [`ToolSpec`], [`ChatRequest`], [`ChatReply`].
//!
//! Backends live in their own crates (`omni-model-peregrine`,
//! `omni-model-unifur`) and depend on this one, never the reverse.
//!
//! # The two ideas that matter most
//!
//! **Enforcement is pure.** [`ToolRegistry::evaluate`] answers "may this call
//! run?" without dispatching anything, because the graph frontier has to ask
//! that question about actions that do not exist yet. The dispatcher and the
//! frontier must get the same answer or a dry-run predicts nothing.
//!
//! **Defaults are the backend's, not the spec's.** `temperature` defaults to
//! `0.0` and `top_p` to `0.95` because peregrine does. Assuming OpenAI's 1.0
//! silently produces a different agent the moment the backend changes, and the
//! difference is invisible for weeks.
//!
//! ```
//! use omni_model::{GateKind, ToolRegistry, ToolCall};
//! use serde_json::json;
//!
//! let registry = ToolRegistry::new();
//! let call = ToolCall::new("c1", "shell", json!({"command": "rm -rf /"}))?;
//!
//! // An unregistered tool is refused as `Unknown` — before any policy runs.
//! // "install the tool" and "change the ROE" are different fixes, so they must
//! // not be reported as the same refusal.
//! let decision = registry.evaluate(&call);
//! assert!(!decision.is_allowed());
//! assert_eq!(decision.kind, GateKind::Unknown);
//!
//! // No gate installed is an explicit, reportable state rather than an implicit
//! // "everything is allowed" — a harness with no ROE should say so.
//! assert!(registry.policy_description().contains("no gate installed"));
//! # Ok::<(), omni_model::Error>(())
//! ```

#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

pub mod gate;
pub mod limits;
pub mod message;
pub mod request;
pub mod tool;

pub use gate::{AllowAll, Decision, Gate, GateChain, GateKind};
pub use limits::{CharHeuristic, TokenCounter};
pub use message::{Message, ToolCall};
pub use request::{ChatReply, ChatRequest, FinishReason, Model, Priority, Usage, complete};
pub use tool::{Tool, ToolOutput, ToolRegistry, ToolSpec};

/// Everything a backend needs, in one place.
///
/// Re-exported so a backend crate writes `use omni_model::prelude::*` and gets
/// the vocabulary without naming seven modules — and, more usefully, so that a
/// rename in here is a rename in one file rather than a sweep through every
/// backend.
pub mod prelude {
    pub use crate::{Context, Error};
    pub use crate::gate::{Decision, Gate, GateChain, GateKind};
    pub use crate::message::{Message, ToolCall};
    pub use crate::request::{
        ChatReply, ChatRequest, FinishReason, Model, Priority, Usage, complete,
    };
    pub use crate::tool::{Tool, ToolOutput, ToolSpec};
}

/// The error type a backend returns.
///
/// Re-exported from `omni-core` rather than redefined, so a backend's failure
/// and the harness's failure are the same type and no caller has to convert
/// between them at the seam.
pub use omni_core::error::{Context, Error};