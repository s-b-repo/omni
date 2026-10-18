//! Tool enforcement as a **pure, dry-runnable** function.
//!
//! # Why purity is the whole point
//!
//! A policy that can only be consulted after a tool has run is not a gate; it is
//! a report. The only way a model-facing system gets to *avoid* a destructive
//! action is to know about the block before dispatching it — which means the
//! decision has to be reachable from places that have no side effects and no
//! transaction to roll back:
//!
//! - the dispatcher, right before the call;
//! - the graph frontier, over a candidate action that does not exist yet;
//! - a dry-run mode that answers "would this be blocked?" with no execution.
//!
//! All three need the *same* answer, and two implementations of a policy that
//! agree today and drift tomorrow are worse than one implementation. So
//! [`Gate::evaluate`] takes `&self`, returns a [`Decision`], and does no I/O. If
//! a gate needs to read a file, that read happened when the gate was built.

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Which rule produced a decision.
///
/// Ordered from most to least general, because a caller that wants "was this
/// blocked, and why" needs the `reason` to be the *specific* one — reporting
/// "blocked: scope" when the answer was "blocked: the ROE window closed" throws
/// away the only part the operator can act on.
///
/// Serializes in snake_case so a persisted decision reads the same in a
/// graph store as it does in a log line.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GateKind {
    /// A destructive-command rule.
    Safety,
    /// Outside the engagement scope.
    Scope,
    /// The Rules of Engagement refused it.
    Roe,
    /// The tool is not registered.
    Unknown,
    /// A backend-specific rule with no general name.
    Custom,
}

impl GateKind {
    /// A short, stable label for logs and audit rows.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            GateKind::Safety => "safety",
            GateKind::Scope => "scope",
            GateKind::Roe => "roe",
            GateKind::Unknown => "unknown",
            GateKind::Custom => "custom",
        }
    }
}

/// The verdict on one candidate tool call.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Decision {
    pub allowed: bool,
    /// Why. Empty when `allowed`, and **never empty when refused** — a refusal
    /// with no reason cannot be acted on, and the audit trail of "blocked"
    /// without a cause is how a policy gets disabled after it blocks something
    /// the operator did not understand.
    pub reason: String,
    pub kind: GateKind,
}

impl Decision {
    /// Allow, with no note.
    #[must_use]
    pub fn allow() -> Self {
        Self { allowed: true, reason: String::new(), kind: GateKind::Custom }
    }

    /// Refuse, naming the rule that refused.
    #[must_use]
    pub fn block(kind: GateKind, reason: impl Into<String>) -> Self {
        Self { allowed: false, reason: reason.into(), kind }
    }

    /// Whether the call may proceed.
    #[must_use]
    pub fn is_allowed(&self) -> bool {
        self.allowed
    }
}

/// A policy that decides whether a tool call may proceed.
///
/// # Implementors
///
/// `evaluate` must be a pure function of `(&self, tool, args)`. No I/O, no
/// clock, no network, no mutation of `self` beyond interior-mutable caches that
/// do not affect the verdict. This is not a style preference: the frontier
/// calls it concurrently over speculative actions while the dispatcher calls it
/// on real ones, and two answers to the same question is the bug this exists to
/// prevent.
pub trait Gate: Send + Sync {
    /// Decide one candidate call. Called from the dispatcher, the frontier and
    /// dry-run alike, so all three get the same answer.
    fn evaluate(&self, tool: &str, args: &Value) -> Decision;

    /// A name for audit rows. Stable across runs; it appears in the log, so a
    /// value that changes per call would make the log useless.
    fn name(&self) -> &str {
        "gate"
    }
}

/// A gate that allows everything.
///
/// Useful as the explicit "no policy installed" — which is different from
/// having no gate at all, because `None` has to be handled at every call site
/// and an allow-all gate does not. A harness with no ROE should say so rather
/// than silently behave as if it had one.
#[derive(Debug, Clone, Copy, Default)]
pub struct AllowAll;

impl Gate for AllowAll {
    fn evaluate(&self, _tool: &str, _args: &Value) -> Decision {
        Decision::allow()
    }

    fn name(&self) -> &str {
        "allow-all"
    }
}

/// Gates applied in order; the first refusal wins.
///
/// Ordering is explicit and stable rather than derived, so that the reason a
/// call was refused does not change when an unrelated gate is added: adding a
/// gate in the middle of a list that has been producing audit rows reinterprets
/// history.
#[derive(Default)]
pub struct GateChain {
    gates: Vec<Box<dyn Gate>>,
}

impl GateChain {
    #[must_use]
    pub fn new() -> Self {
        Self { gates: Vec::new() }
    }

    /// Append a gate. Runs after every gate already in the chain.
    pub fn push(&mut self, gate: Box<dyn Gate>) -> &mut Self {
        self.gates.push(gate);
        self
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.gates.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.gates.is_empty()
    }

    /// A stable description of the installed order, for the audit trail.
    #[must_use]
    pub fn describe(&self) -> String {
        self.gates
            .iter()
            .map(|g| g.name())
            .collect::<Vec<_>>()
            .join(" -> ")
    }
}

impl Gate for GateChain {
    fn evaluate(&self, tool: &str, args: &Value) -> Decision {
        for g in &self.gates {
            let d = g.evaluate(tool, args);
            if !d.allowed {
                return d;
            }
        }
        Decision::allow()
    }

    fn name(&self) -> &str {
        "chain"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    struct DenyEverything;
    impl Gate for DenyEverything {
        fn evaluate(&self, tool: &str, _args: &Value) -> Decision {
            Decision::block(GateKind::Safety, format!("no {tool} today"))
        }
        fn name(&self) -> &str {
            "deny-everything"
        }
    }

    struct DenyReadOnly;
    impl Gate for DenyReadOnly {
        fn evaluate(&self, tool: &str, _args: &Value) -> Decision {
            if tool == "write" {
                Decision::block(GateKind::Scope, "write is out of scope")
            } else {
                Decision::allow()
            }
        }
        fn name(&self) -> &str {
            "deny-read-only"
        }
    }

    #[test]
    fn an_empty_chain_allows() {
        let d = GateChain::new().evaluate("shell", &json!({}));
        assert!(d.is_allowed());
    }

    #[test]
    fn the_first_refusal_wins_and_keeps_its_reason() {
        let mut chain = GateChain::new();
        chain.push(Box::new(DenyEverything)).push(Box::new(DenyReadOnly));
        let d = chain.evaluate("write", &json!({}));
        assert!(!d.is_allowed());
        assert_eq!(d.kind, GateKind::Safety, "the first gate's verdict must survive");
        assert!(d.reason.contains("no write today"), "reason must be the specific one: {}", d.reason);
    }

    #[test]
    fn a_refusal_always_carries_a_reason() {
        let d = DenyEverything.evaluate("shell", &json!({}));
        assert!(!d.reason.is_empty(), "a block with no reason cannot be acted on");
        assert!(!d.kind.as_str().is_empty());
    }

    #[test]
    fn a_later_gate_still_refuses_what_the_first_allowed() {
        let mut chain = GateChain::new();
        chain.push(Box::new(DenyReadOnly));
        assert!(chain.evaluate("read", &json!({})).is_allowed());
        assert!(!chain.evaluate("write", &json!({})).is_allowed());
    }

    #[test]
    fn the_chain_describes_its_order_stably() {
        let mut chain = GateChain::new();
        chain.push(Box::new(DenyReadOnly)).push(Box::new(DenyEverything));
        assert_eq!(chain.describe(), "deny-read-only -> deny-everything");
        assert_eq!(chain.len(), 2);
    }

    #[test]
    fn evaluate_is_repeatable_because_it_is_pure() {
        // The property the frontier depends on: asking twice must give the same
        // answer, or a dry-run is not a prediction of the real call.
        let gate = DenyReadOnly;
        let args = json!({"path": "/etc/passwd"});
        let first = gate.evaluate("write", &args);
        let second = gate.evaluate("write", &args);
        assert_eq!(first, second);
    }
}