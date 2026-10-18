//! The `Model` trait and its request/reply vocabulary.
//!
//! # This is the seam every backend implements
//!
//! `omni` joins five systems written by different people. Everything that talks
//! to an actual language model goes through this one trait. That is deliberate:
//! a second way to ask a model a question is a second way for the harness's
//! guarantees to stop applying.
//!
//! The trait is deliberately small — one method, object-safe, no lifetime
//! parameters — so that a backend is a struct plus a match, and so the graph
//! frontier can hold a `&mut dyn Model` while running candidates speculatively.
//!
//! Defaults are peregrine's, not OpenAI's, and the difference is load-bearing:
//!
//! | | OpenAI | peregrine | omni |
//! |---|---|---|---|
//! | `temperature` | 1.0 | **0.0** | **0.0** |
//! | `top_p` | 1.0 | **0.95** | **0.95** |
//! | `top_k` | not sent | 0 = off | 0 = off |
//!
//! A harness that assumes 1.0 gets a different agent every time the backend
//! changes, which is the kind of difference that shows up as "the new model is
//! worse" six weeks later.

use serde_json::Value;

use crate::Error;
use crate::message::Message;
use crate::tool::ToolSpec;

/// Why the model stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FinishReason {
    /// The model finished on its own.
    Stop,
    /// The model wants tools run.
    ToolCalls,
    /// The output budget ran out mid-turn.
    ///
    /// Distinct from [`FinishReason::Stop`] because an openhack-style loop that
    /// treats a truncated tool-call turn as a finished one will record "no tool
    /// needed" for a turn that ran out of room mid-JSON — which is the exact
    /// failure Harness-R1's parse cascade exists to catch.
    Length,
    /// The backend refused on content grounds.
    ContentFilter,
    /// The server ended the stream for a reason the client cannot name.
    Unknown,
}

impl FinishReason {
    /// The OpenAI wire spelling.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            FinishReason::Stop => "stop",
            FinishReason::ToolCalls => "tool_calls",
            FinishReason::Length => "length",
            FinishReason::ContentFilter => "content_filter",
            FinishReason::Unknown => "unknown",
        }
    }

    /// Parse the OpenAI wire spelling.
    #[must_use]
    pub fn parse(s: &str) -> Self {
        match s {
            "stop" => FinishReason::Stop,
            "tool_calls" | "function_call" => FinishReason::ToolCalls,
            "length" => FinishReason::Length,
            "content_filter" => FinishReason::ContentFilter,
            _ => FinishReason::Unknown,
        }
    }

    /// Whether this turn wants tools run before the next one.
    #[must_use]
    pub fn wants_tools(self) -> bool {
        matches!(self, FinishReason::ToolCalls)
    }
}

/// Token accounting, when the backend reports it.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
}

impl Usage {
    #[must_use]
    pub fn total(&self) -> u32 {
        self.prompt_tokens.saturating_add(self.completion_tokens)
    }
}

/// Which queue to enter, if the backend has more than one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Priority {
    High,
    Normal,
}

/// One turn's request.
///
/// Borrowed rather than owned: a loop sends the same growing message list many
/// times, and cloning a transcript per turn is how an agent runs out of memory
/// long before it runs out of budget.
#[derive(Debug, Clone, PartialEq)]
pub struct ChatRequest<'a> {
    pub messages: &'a [Message],
    pub tools: &'a [ToolSpec],
    /// 0.0 deterministic, 2.0 unconstrained. Default **0.0**, matching
    /// peregrine: a frozen target must produce the same tokens twice for a
    /// baseline/patched comparison to mean anything.
    pub temperature: f64,
    /// Nucleus cut. Default 0.95, matching peregrine.
    pub top_p: f64,
    /// Top-k cut. Default 0 = off. Applied **before** `top_p`, matching
    /// peregrine's documented precedence.
    pub top_k: usize,
    /// Output budget in tokens.
    pub max_tokens: usize,
    /// Sent, but peregrine ignores it. Held so a backend that *does* honour it
    /// can, and so the harness can record which name it asked for.
    pub model: Option<&'a str>,
    pub priority: Option<Priority>,
}

impl Default for ChatRequest<'_> {
    fn default() -> Self {
        Self {
            messages: &[],
            tools: &[],
            temperature: 0.0,
            top_p: 0.95,
            top_k: 0,
            max_tokens: 1024,
            model: None,
            priority: None,
        }
    }
}

impl<'a> ChatRequest<'a> {
    /// A request carrying only messages, with peregrine's defaults.
    #[must_use]
    pub fn new(messages: &'a [Message]) -> Self {
        Self { messages, ..Self::default() }
    }

    /// Attach tools.
    #[must_use]
    pub fn with_tools(mut self, tools: &'a [ToolSpec]) -> Self {
        self.tools = tools;
        self
    }

    /// Set the output budget.
    #[must_use]
    pub fn with_max_tokens(mut self, n: usize) -> Self {
        self.max_tokens = n;
        self
    }

    /// The knob values that can change which tokens come back.
    ///
    /// The reward fingerprint hashes this. Every field here is a value a
    /// different run could set differently and get different output, which is
    /// precisely what a "matched baseline" must not be.
    #[must_use]
    pub fn sampling_knobs(&self) -> Value {
        serde_json::json!({
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "max_tokens": self.max_tokens,
        })
    }
}

/// One turn's reply.
#[derive(Debug, Clone, PartialEq)]
pub struct ChatReply {
    /// The assistant turn. Carries `tool_calls` when the model asked for tools.
    pub message: Message,
    pub finish_reason: FinishReason,
    pub usage: Option<Usage>,
}

impl ChatReply {
    /// A plain text reply.
    ///
    /// Named `saying` rather than `text` because the reply also *has* text, and
    /// a type with a constructor and an accessor of the same name is one where
    /// `ChatReply::text(x)` and `reply.text()` read as the same operation.
    #[must_use]
    pub fn saying(content: impl Into<String>) -> Self {
        Self {
            message: Message::assistant(content),
            finish_reason: FinishReason::Stop,
            usage: None,
        }
    }

    /// The reply's text, if it has any.
    #[must_use]
    pub fn text(&self) -> Option<&str> {
        self.message.text()
    }
}

/// A backend that answers chat turns.
///
/// Object-safe on purpose: the loop holds `&mut dyn Model`, the frontier holds
/// `&dyn Model` while speculating, and `omni-eval` holds a `&mut dyn Model` to
/// rerun a frozen target. A generic trait could only be one of those at a time.
pub trait Model: Send {
    /// Answer one turn.
    ///
    /// # Errors
    ///
    /// [`Error::Format`] for a reply that cannot be parsed — an unterminated
    /// tool call, a non-object argument bag. [`Error::Io`] for transport. A
    /// backend must **not** return an empty string and `Ok(())` to mean "the
    /// server is sad"; that is the failure mode that makes a harness loop on a
    /// model that has stopped answering.
    fn chat(&mut self, req: &ChatRequest<'_>) -> Result<ChatReply, Error>;

    /// A stable name for logs and for the reward fingerprint.
    fn name(&self) -> &str;

    /// Whether this backend can be relied on to be deterministic at
    /// `temperature: 0.0`.
    ///
    /// A harness that claims a matched baseline needs this to be `true`, and
    /// needs it stated rather than assumed: a backend that samples anyway
    /// produces a `Δ` that is pure noise, and a noise `Δ` is worse than none
    /// because it looks like evidence.
    fn deterministic_at_zero(&self) -> bool {
        false
    }
}

/// Convenience: one round trip through `model`, discarding the reply text.
pub fn complete(model: &mut dyn Model, system: &str, prompt: &str, max_tokens: usize) -> Result<String, Error> {
    let msgs = [Message::system(system), Message::user(prompt)];
    let req = ChatRequest::new(&msgs).with_max_tokens(max_tokens);
    let reply = model.chat(&req)?;
    Ok(reply.text().unwrap_or("").to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn the_defaults_are_peregrines_not_openais() {
        let r = ChatRequest::default();
        // 1.0 here would silently change the agent whenever the backend does.
        assert_eq!(r.temperature, 0.0, "peregrine defaults temperature to 0.0");
        assert_eq!(r.top_p, 0.95, "peregrine defaults top_p to 0.95");
        assert_eq!(r.top_k, 0, "0 means top_k is off");
    }

    #[test]
    fn finish_reasons_round_trip() {
        for f in [
            FinishReason::Stop,
            FinishReason::ToolCalls,
            FinishReason::Length,
            FinishReason::ContentFilter,
        ] {
            assert_eq!(FinishReason::parse(f.as_str()), f);
        }
        assert_eq!(FinishReason::parse("function_call"), FinishReason::ToolCalls);
        assert_eq!(FinishReason::parse("nonsense"), FinishReason::Unknown);
    }

    #[test]
    fn only_tool_calls_asks_for_tools() {
        assert!(FinishReason::ToolCalls.wants_tools());
        // A truncated turn is not a finished one: treating it as finished
        // records "no tool needed" for a turn that ran out of room mid-call.
        assert!(!FinishReason::Length.wants_tools());
        assert!(!FinishReason::Stop.wants_tools());
    }

    #[test]
    fn sampling_knobs_capture_everything_that_changes_the_tokens() -> Result<(), Error> {
        let a = ChatRequest::default();
        let b = ChatRequest { temperature: 0.7, ..ChatRequest::default() };
        let knobs_a = a.sampling_knobs();
        let knobs_b = b.sampling_knobs();
        assert_ne!(
            omni_core::value_id(&knobs_a)?,
            omni_core::value_id(&knobs_b)?,
            "two requests with different sampling must not share an identity"
        );
        Ok(())
    }

    struct Fixed(&'static str);

    impl Model for Fixed {
        fn chat(&mut self, _req: &ChatRequest<'_>) -> Result<ChatReply, Error> {
            Ok(ChatReply::saying(self.0))
        }
        fn name(&self) -> &str {
            "fixed"
        }
        fn deterministic_at_zero(&self) -> bool {
            true
        }
    }

    #[test]
    fn the_trait_is_object_safe() -> Result<(), Error> {
        // The loop, the frontier and omni-eval each hold a different borrow of
        // a dyn Model. A generic trait could only serve one of them.
        let mut m: Box<dyn Model> = Box::new(Fixed("hello"));
        let r: &mut dyn Model = m.as_mut();
        let out = complete(r, "sys", "prompt", 32)?;
        assert_eq!(out, "hello");
        assert_eq!(m.name(), "fixed");
        Ok(())
    }

    #[test]
    fn a_backend_must_declare_whether_it_is_deterministic() {
        // The default is false, so a backend that forgets to say is treated as
        // noisy. The other default — assuming determinism — would let a noisy
        // backend quietly produce a meaningless reward delta.
        struct Forgetful;
        impl Model for Forgetful {
            fn chat(&mut self, _req: &ChatRequest<'_>) -> Result<ChatReply, Error> {
                Ok(ChatReply::saying("x"))
            }
            fn name(&self) -> &str {
                "forgetful"
            }
        }
        assert!(!Forgetful.deterministic_at_zero());
        assert!(Fixed("x").deterministic_at_zero());
    }

    #[test]
    fn usage_totals_saturate() {
        let u = Usage { prompt_tokens: u32::MAX, completion_tokens: 10 };
        assert_eq!(u.total(), u32::MAX, "token accounting must not wrap");
    }

    #[test]
    fn a_reply_can_carry_a_tool_call_and_text_together() -> Result<(), Error> {
        let call = crate::message::ToolCall::new("c1", "echo", json!({"text":"hi"}))?;
        let reply = ChatReply {
            message: Message::assistant_tool_calls(vec![call]),
            finish_reason: FinishReason::ToolCalls,
            usage: None,
        };
        assert!(reply.message.tool_calls().len() == 1);
        assert_eq!(reply.text(), None);
        assert!(reply.finish_reason.wants_tools());
        Ok(())
    }
}