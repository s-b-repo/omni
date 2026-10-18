//! The orchestration loop: a deterministic council tally, an explicit round
//! budget, and the per-round plan it hands the harness.
//!
//! Ported from **openhack**'s `council.ts` and `round-budget.ts`. Two
//! properties carried across deliberately, per AGENTS.md:
//!
//! - **The tally is deterministic.** The same verdicts produce the same
//!   decision — always — and never with a tie-break by arrival or by hash.
//!   omni goes one step further than the source: verdicts are aggregated in a
//!   canonical order (by verdict kind, then by advisor name), so the same
//!   *set* of verdicts aggregates identically in any arrival order. The source
//!   only promised determinism for the same order.
//! - **The round budget is explicit.** A loop that terminates for an unstated
//!   reason is a loop whose failures are uninterpretable, so every stop is a
//!   named stop: converged, stalled, budget spent — with the numbers that
//!   caused it.
//!
//! What is *not* here, deliberately: the loop does not dispatch anything. It
//! plans a round — enforcement across the frontier, in priority order — and
//! returns what may be dispatched and what the gate refused. Dispatch and
//! observation belong to the harness that owns the tools; keeping them out is
//! what lets this crate be tested without a model or a network.

#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
#![forbid(unsafe_code)]

use std::collections::BTreeMap;

use omni_core::Error;
use omni_graph::{ActionNode, Graph};
use omni_model::{Gate, GateKind};
use serde::{Deserialize, Serialize};

pub mod rollout;

// ── The council ─────────────────────────────────────────────────────────────

/// One reviewer's verdict on one finding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Verdict {
    /// Real, correctly classified.
    Confirmed,
    /// Cannot be judged from the evidence given.
    NeedsEvidence,
    /// Not real; a misclassification.
    FalsePositive,
}

impl Verdict {
    /// A stable label, in enum order — which is also the tally's tie-break
    /// order, stated once here.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Confirmed => "confirmed",
            Self::NeedsEvidence => "needs-evidence",
            Self::FalsePositive => "false-positive",
        }
    }

    /// Parse the label. The inverse of [`Verdict::as_str`], so a wire value
    /// and a printed one are the same vocabulary.
    ///
    /// # Errors
    ///
    /// [`Error::Format`] for anything else, naming the input.
    pub fn parse(s: &str) -> Result<Self, Error> {
        match s {
            "confirmed" => Ok(Self::Confirmed),
            "needs-evidence" | "needs_evidence" => Ok(Self::NeedsEvidence),
            "false-positive" | "false_positive" => Ok(Self::FalsePositive),
            other => Err(Error::Format(format!(
                "unknown council verdict {other:?}; expected confirmed | needs-evidence | false-positive"
            ))),
        }
    }

    /// The enum's own index — the tie-break order.
    fn rank(self) -> u8 {
        match self {
            Self::Confirmed => 0,
            Self::NeedsEvidence => 1,
            Self::FalsePositive => 2,
        }
    }
}

/// One advisor's say.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AdvisorVerdict {
    /// Which advisor said it. Named so an audit row can distinguish two
    /// advisors agreeing from one voice repeated.
    pub advisor: String,
    pub verdict: Verdict,
    /// 0..1; clamped. `NaN` is treated as no verdict at all, because a NaN
    /// weight would poison the weighted share it is summed into.
    pub confidence: f64,
}

impl AdvisorVerdict {
    /// A say with its weight clamped into `[0, 1]`.
    #[must_use]
    pub fn new(advisor: &str, verdict: Verdict, confidence: f64) -> Self {
        let c = if confidence.is_nan() { 0.0 } else { confidence.clamp(0.0, 1.0) };
        Self { advisor: advisor.to_string(), verdict, confidence: c }
    }
}

/// How the council decided.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Outcome {
    Confirmed,
    FalsePositive,
    Escalate,
}

impl Outcome {
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Confirmed => "confirmed",
            Self::FalsePositive => "false-positive",
            Self::Escalate => "escalate",
        }
    }
}

/// The aggregation knobs, with openhack's defaults.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CouncilConfig {
    /// Minimum weighted agreement for a non-conflict outcome.
    pub threshold: f64,
    /// Any advisor below this confidence forces escalation.
    pub low_confidence: f64,
}

impl Default for CouncilConfig {
    fn default() -> Self {
        Self { threshold: 0.66, low_confidence: 0.4 }
    }
}

/// The decision the council reached on one finding.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Tally {
    pub outcome: Outcome,
    /// Weighted share of the winning verdict.
    pub agreement: f64,
    pub confirmed_weight: f64,
    pub false_positive_weight: f64,
    pub needs_evidence_weight: f64,
    /// True when the council refused to decide: a split, an uncertain voice,
    /// or no usable verdicts. An escalation is a decision that names why.
    pub conflict: bool,
    pub reason: String,
}

/// Confidence-weighted majority, in code, never in a model.
///
/// A split or an uncertain finding **escalates** rather than being silently
/// averaged away — that is the "blind consensus" failure the port exists to
/// prevent, and the reason the outcome set has an `escalate` at all.
///
/// # Determinism, stated precisely
///
/// Verdicts are summed in canonical order — verdict kind first, then advisor
/// name — so the same *set* of verdicts produces the same weights and the same
/// decision in any arrival order. (The openhack source summed in arrival order
/// and so only promised determinism for the same order; floating-point sums
/// are not associative, so the canonical pass is doing real work, not style.)
///
/// A tie between two verdict kinds' weights resolves by verdict kind order —
/// `confirmed`, then `needs-evidence`, then `false-positive` — and never by
/// who answered first.
#[must_use]
pub fn tally(verdicts: &[AdvisorVerdict], cfg: &CouncilConfig) -> Tally {
    // Canonical aggregation order: verdict kind, then advisor name, then
    // confidence — a total order over the inputs that arrival cannot disturb.
    let mut ordered: Vec<&AdvisorVerdict> = verdicts.iter().collect();
    ordered.sort_by(|a, b| {
        a.verdict
            .rank()
            .cmp(&b.verdict.rank())
            .then_with(|| a.advisor.cmp(&b.advisor))
            .then_with(|| a.confidence.total_cmp(&b.confidence))
    });

    let mut weights = [0.0f64; 3];
    let mut total = 0.0f64;
    let mut any_low = false;
    let mut usable = 0usize;
    for v in &ordered {
        // A clamped-zero confidence is a voice with no weight, not a refusal:
        // the verdict's *name* still cannot decide anything, but it is honest
        // to say it was counted as nothing rather than ignored as a shape.
        weights[v.verdict.rank() as usize] += v.confidence;
        total += v.confidence;
        if v.confidence > 0.0 && v.confidence < cfg.low_confidence {
            any_low = true;
        }
        usable += 1;
    }

    if usable == 0 || total <= 0.0 {
        return Tally {
            outcome: Outcome::Escalate,
            agreement: 0.0,
            confirmed_weight: 0.0,
            false_positive_weight: 0.0,
            needs_evidence_weight: 0.0,
            conflict: true,
            reason: "escalate — no usable verdicts".to_string(),
        };
    }

    // Winner: highest weight, ties resolved by verdict rank.
    let mut winner = Verdict::Confirmed;
    let mut winner_weight = weights[0];
    for (rank, w) in weights.iter().enumerate() {
        if *w > winner_weight || (*w == winner_weight && rank < winner.rank() as usize) {
            winner_weight = *w;
            winner = rank_to_verdict(rank);
        }
    }
    // Agreement is rounded to three decimals so the *reported* number is the
    // same at any arrival order, matching the sum that produced it.
    let agreement = (winner_weight / total * 1000.0).round() / 1000.0;

    let decisive = matches!(winner, Verdict::Confirmed | Verdict::FalsePositive);
    let conflict = any_low || agreement < cfg.threshold || !decisive;

    let outcome = if conflict {
        Outcome::Escalate
    } else if winner == Verdict::Confirmed {
        Outcome::Confirmed
    } else {
        Outcome::FalsePositive
    };

    let reason = if conflict {
        if any_low {
            "escalate — an advisor's confidence is below the floor".to_string()
        } else if !decisive {
            "escalate — the winning verdict is 'needs-evidence'".to_string()
        } else {
            format!("escalate — agreement {agreement} below threshold {}", cfg.threshold)
        }
    } else {
        format!("{} — weighted agreement {agreement}", outcome.as_str())
    };

    Tally {
        outcome,
        agreement,
        confirmed_weight: weights[0],
        false_positive_weight: weights[2],
        needs_evidence_weight: weights[1],
        conflict,
        reason,
    }
}

fn rank_to_verdict(rank: usize) -> Verdict {
    match rank {
        0 => Verdict::Confirmed,
        1 => Verdict::NeedsEvidence,
        _ => Verdict::FalsePositive,
    }
}

/// Count outcomes across a set of tallies, for the review write-up.
#[must_use]
pub fn summarize(tallies: &[Tally]) -> BTreeMap<&'static str, usize> {
    let mut out = BTreeMap::new();
    out.insert("confirmed", 0usize);
    out.insert("false-positive", 0);
    out.insert("escalate", 0);
    for t in tallies {
        let key = match t.outcome {
            Outcome::Confirmed => "confirmed",
            Outcome::FalsePositive => "false-positive",
            Outcome::Escalate => "escalate",
        };
        *out.entry(key).or_insert(0) += 1;
    }
    out
}

// ── The round budget ────────────────────────────────────────────────────────

/// One round's record, as the bench harness reads it. Structural fields only:
/// omni has no cost-in-dollars seam yet, so the fields that would carry it are
/// absent rather than carried as zeros that look like measurements.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct RoundRecord {
    pub round: u64,
    /// Findings on the graph after this round.
    pub findings_total: u64,
    /// Findings first seen this round.
    pub new_findings: u64,
    /// Combos still open, when the profile counts combos.
    pub combos_open: Option<u64>,
    /// How many actions were queued when the round started.
    pub frontier_size: u64,
}

/// The knobs of the adaptive budget, with openhack's defaults.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BudgetConfig {
    /// Consecutive dry rounds before early termination.
    pub dry_streak_limit: u64,
    /// A round is productive at or above this delta (new findings + closed combos).
    pub productive_min_delta: u64,
    /// Never extend past this cap regardless of pace.
    pub hard_ceiling: u64,
}

impl Default for BudgetConfig {
    fn default() -> Self {
        Self { dry_streak_limit: 2, productive_min_delta: 1, hard_ceiling: 50 }
    }
}

/// The budget's verdict on the loop so far.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BudgetVerdict {
    /// The loop must terminate this iteration, whatever `max_rounds` said.
    pub terminate: bool,
    /// The last round was productive — advisory, and what an extension asks.
    pub productive: bool,
    /// Consecutive unproductive rounds, counting the latest.
    pub dry_streak: u64,
    /// Combos closed by the latest round, when the profile counts combos.
    pub combos_closed: Option<u64>,
    /// Set exactly when `terminate` is true. A loop that stops must be able
    /// to say why in one line an operator can act on.
    pub reason: Option<String>,
}

/// The adaptive verdict, pure: same records, same config, same answer.
///
/// Ported from openhack's `RoundBudget::verdict`. A dry streak over the limit
/// terminates with the reason naming what was left open, so an operator can
/// tell "we finished" from "we stalled with work outstanding".
#[must_use]
pub fn budget_verdict(records: &[RoundRecord], cfg: &BudgetConfig) -> BudgetVerdict {
    if records.is_empty() {
        return BudgetVerdict { terminate: false, productive: false, dry_streak: 0, combos_closed: None, reason: None };
    }
    let last = &records[records.len() - 1];
    let prev = records.len().checked_sub(2).map(|i| &records[i]);
    let combos_closed = match (prev.and_then(|p| p.combos_open), last.combos_open) {
        (Some(p_open), Some(l_open)) => Some(p_open.saturating_sub(l_open)),
        _ => None,
    };

    let total_delta = last.new_findings + combos_closed.unwrap_or(0);
    let productive = total_delta >= cfg.productive_min_delta;

    // Dry streak: the tail of records that produced nothing.
    let mut dry = 0u64;
    for (i, r) in records.iter().enumerate().rev() {
        let prev_combos = if i > 0 { records[i - 1].combos_open } else { None };
        let close = match (prev_combos, r.combos_open) {
            (Some(p), Some(l)) => p.saturating_sub(l),
            _ => 0,
        };
        if r.new_findings + close < cfg.productive_min_delta {
            dry += 1;
        } else {
            break;
        }
    }

    if dry >= cfg.dry_streak_limit {
        let open_left = last.combos_open.unwrap_or(0);
        let front_left = last.frontier_size;
        let reason = if open_left == 0 && front_left == 0 {
            format!("converged: {dry} dry rounds, 0 combos + 0 frontier")
        } else {
            format!("stalled: {dry} dry rounds, {open_left} combos + {front_left} frontier still open")
        };
        return BudgetVerdict { terminate: true, productive, dry_streak: dry, combos_closed, reason: Some(reason) };
    }

    BudgetVerdict { terminate: false, productive, dry_streak: dry, combos_closed, reason: None }
}

/// The driver's own cap, separate from the adaptive verdict: `max_rounds` is
/// what the operator asked for, and spending is explicit.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Rounds {
    max_rounds: u64,
    used: u64,
}

impl Rounds {
    /// An explicit budget. Zero is a valid statement — "plan nothing" — and
    /// the loop's first question reports [`RoundBudget`]-style exhaustion for
    /// it rather than treating it as "unbounded".
    #[must_use]
    pub fn new(max_rounds: u64) -> Self {
        Self { max_rounds, used: 0 }
    }

    /// Consume one round if there is one. `None` is the stated stop: budget.
    pub fn spend(&mut self) -> Option<u64> {
        if self.used >= self.max_rounds {
            return None;
        }
        self.used += 1;
        Some(self.used)
    }

    /// Rounds consumed so far.
    #[must_use]
    pub fn used(&self) -> u64 {
        self.used
    }

    /// Rounds remaining.
    #[must_use]
    pub fn remaining(&self) -> u64 {
        self.max_rounds.saturating_sub(self.used)
    }
}

// ── The per-round plan ──────────────────────────────────────────────────────

/// What one round would do, before doing any of it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RoundPlan {
    pub round: u64,
    /// Allowed candidates in frontier order, capped at `k`.
    pub kept: Vec<ActionNode>,
    /// Refusals, each with the gate's own kind and reason.
    pub blocked: Vec<ActionNode>,
}

impl RoundPlan {
    /// A one-line summary for a log.
    #[must_use]
    pub fn describe(&self) -> String {
        format!(
            "round {}: {} kept, {} blocked by the gate",
            self.round,
            self.kept.len(),
            self.blocked.len()
        )
    }
}

/// Plan one round against the graph: budget first, then the frontier with the
/// real gate.
///
/// Returns `None` exactly when the budget is spent — the loop's stated stop.
/// The `gate` is the same `dyn Gate` the dispatcher will consult, per rule 7.
pub fn plan_round(
    graph: &mut Graph,
    gate: &dyn Gate,
    rounds: &mut Rounds,
    k: usize,
) -> Option<RoundPlan> {
    let round = rounds.spend()?;
    let (kept, blocked) = graph.frontier_gated(k, gate);
    Some(RoundPlan { round, kept, blocked })
}

/// A round's record from a plan, to hand to [`budget_verdict`].
///
/// `new_findings` is what the *caller* observed this round — planning cannot
/// know it — so this records what was dispatched, and the caller merges its own
/// outcome delta into the record before asking the budget.
#[must_use]
pub fn record_of(plan: &RoundPlan, findings_total: u64, new_findings: u64) -> RoundRecord {
    RoundRecord {
        round: plan.round,
        findings_total,
        new_findings,
        combos_open: None,
        frontier_size: plan.kept.len() as u64 + plan.blocked.len() as u64,
    }
}

// ── A real escalation gate the harness can install ──────────────────────────

/// Refuse everything: the gate a harness installs when it wants a dry run of
/// the whole enforcement path. Distinct from [`omni_graph::ScopeAllowlist`]
/// with an empty list only in name and in the reason it records.
#[derive(Debug, Clone, Copy, Default)]
pub struct DryRun;

impl Gate for DryRun {
    fn evaluate(&self, _tool: &str, _args: &serde_json::Value) -> omni_model::Decision {
        omni_model::Decision::block(GateKind::Safety, "dry run: nothing dispatches")
    }

    fn name(&self) -> &str {
        "dry-run"
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use omni_core::content_id_str;

    fn say(advisor: &str, verdict: Verdict, confidence: f64) -> AdvisorVerdict {
        AdvisorVerdict::new(advisor, verdict, confidence)
    }

    #[test]
    fn the_tally_is_identical_under_any_arrival_order() -> Result<(), Error> {
        let base = vec![
            say("a", Verdict::Confirmed, 0.9),
            say("b", Verdict::FalsePositive, 0.8),
            say("c", Verdict::Confirmed, 0.7),
        ];
        let first = tally(&base, &CouncilConfig::default());
        let reversed = {
            let mut v = base.clone();
            v.reverse();
            v
        };
        let second = tally(&reversed, &CouncilConfig::default());
        assert_eq!(first, second, "the same verdicts in any order must tally identically");
        assert_eq!(first.outcome, Outcome::Confirmed, "3 × confirmed vs 1 weighted fp: {:?}", first.reason);
        Ok(())
    }

    #[test]
    fn a_split_council_escalates_instead_of_averaging_away() -> Result<(), Error> {
        let base = vec![
            say("a", Verdict::Confirmed, 0.9),
            say("b", Verdict::FalsePositive, 0.9),
        ];
        let t = tally(&base, &CouncilConfig::default());
        assert_eq!(t.outcome, Outcome::Escalate, "a 50/50 split is a conflict: {}", t.reason);
        assert!(t.conflict);
        // And the same split in the opposite order escalates identically.
        let mut flipped = base.clone();
        flipped.reverse();
        assert_eq!(t, tally(&flipped, &CouncilConfig::default()));
        Ok(())
    }

    #[test]
    fn a_voice_below_the_confidence_floor_forces_escalation() -> Result<(), Error> {
        let base = vec![
            say("a", Verdict::Confirmed, 0.9),
            say("b", Verdict::Confirmed, 0.35),
        ];
        let t = tally(&base, &CouncilConfig::default());
        assert_eq!(t.outcome, Outcome::Escalate, "one uncertain voice is a conflict: {}", t.reason);
        assert!(t.reason.contains("below the floor"));
        Ok(())
    }

    #[test]
    fn needs_evidence_never_confirms_on_its_own() -> Result<(), Error> {
        let base = vec![
            say("a", Verdict::NeedsEvidence, 0.9),
            say("b", Verdict::NeedsEvidence, 0.9),
        ];
        let t = tally(&base, &CouncilConfig::default());
        assert_eq!(t.outcome, Outcome::Escalate, "unanimous 'cannot judge' is not a confirmation");
        assert!(t.reason.contains("needs-evidence"));
        Ok(())
    }

    #[test]
    fn no_usable_verdicts_is_an_escalation_with_a_reason() -> Result<(), Error> {
        let t = tally(&[], &CouncilConfig::default());
        assert_eq!(t.outcome, Outcome::Escalate);
        assert!(t.conflict);
        assert!(t.reason.contains("no usable verdicts"), "{}", t.reason);
        let nan = tally(&[say("a", Verdict::Confirmed, f64::NAN)], &CouncilConfig::default());
        assert_eq!(nan.outcome, Outcome::Escalate, "a NaN voice carries no weight");
        Ok(())
    }

    #[test]
    fn a_tie_between_kinds_resolves_by_kind_order_never_arrival() -> Result<(), Error> {
        // confirmed and false-positive at exactly equal weight: the tie-break
        // is the kind order, so which advisor handed which in last is moot.
        let a = vec![say("x", Verdict::Confirmed, 0.5), say("y", Verdict::FalsePositive, 0.5)];
        let t1 = tally(&a, &CouncilConfig::default());
        let b = vec![say("y", Verdict::FalsePositive, 0.5), say("x", Verdict::Confirmed, 0.5)];
        let t2 = tally(&b, &CouncilConfig::default());
        assert_eq!(t1, t2, "the tie-break may not depend on arrival");
        assert!(t1.conflict, "equal weights are below the threshold, so it escalates anyway");
        Ok(())
    }

    #[test]
    fn the_winner_is_reported_with_its_share_when_decisive() -> Result<(), Error> {
        let t = tally(
            &[say("a", Verdict::Confirmed, 1.0), say("b", Verdict::Confirmed, 1.0), say("c", Verdict::FalsePositive, 1.0)],
            &CouncilConfig::default(),
        );
        assert_eq!(t.outcome, Outcome::Confirmed);
        assert_eq!(t.agreement, 0.667, "weighted share, rounded: {:?}", t.reason);
        assert_eq!(t.confirmed_weight, 2.0);
        assert!(!t.conflict);
        Ok(())
    }

    #[test]
    fn summarize_counts_the_three_outcomes() -> Result<(), Error> {
        let s = summarize(&[
            tally(&[say("a", Verdict::Confirmed, 1.0)], &CouncilConfig::default()),
            tally(&[], &CouncilConfig::default()),
        ]);
        assert_eq!(s["confirmed"], 1);
        assert_eq!(s["escalate"], 1);
        assert_eq!(s["false-positive"], 0);
        Ok(())
    }

    #[test]
    fn verdicts_parse_from_their_own_labels() -> Result<(), Error> {
        for v in [Verdict::Confirmed, Verdict::NeedsEvidence, Verdict::FalsePositive] {
            assert_eq!(Verdict::parse(v.as_str())?, v);
        }
        assert!(Verdict::parse("probably").is_err());
        Ok(())
    }

    #[test]
    fn a_dry_streak_terminates_and_says_what_was_left_open() -> Result<(), Error> {
        let cfg = BudgetConfig::default();
        // Round 3 and round 2 produced nothing (no new findings, no combos
        // closed); round 3 still leaves four combos and one frontier action
        // open, so the reason names the leftovers.
        let records = vec![
            RoundRecord { round: 1, findings_total: 5, new_findings: 3, combos_open: Some(4), frontier_size: 2 },
            RoundRecord { round: 2, findings_total: 5, new_findings: 0, combos_open: Some(4), frontier_size: 2 },
            RoundRecord { round: 3, findings_total: 5, new_findings: 0, combos_open: Some(4), frontier_size: 1 },
        ];
        let v = budget_verdict(&records, &cfg);
        assert!(v.terminate, "{:?}", v.reason);
        assert!(
            v.reason.as_ref().is_some_and(|r| r.contains("stalled") && r.contains("4 combos + 1 frontier")),
            "{:?}",
            v.reason
        );
        assert_eq!(v.dry_streak, 2);
        assert_eq!(v.combos_closed, Some(0));
        Ok(())
    }

    #[test]
    fn convergence_terminates_with_a_converged_reason() -> Result<(), Error> {
        let cfg = BudgetConfig::default();
        // A productive round closes everything, then two dry rounds confirm
        // nothing is left open: the loop is done, and says so.
        let records = vec![
            RoundRecord { round: 1, findings_total: 5, new_findings: 5, combos_open: Some(3), frontier_size: 3 },
            RoundRecord { round: 2, findings_total: 7, new_findings: 2, combos_open: Some(0), frontier_size: 0 },
            RoundRecord { round: 3, findings_total: 7, new_findings: 0, combos_open: Some(0), frontier_size: 0 },
            RoundRecord { round: 4, findings_total: 7, new_findings: 0, combos_open: Some(0), frontier_size: 0 },
        ];
        let v = budget_verdict(&records, &cfg);
        assert!(v.terminate, "a dry streak with nothing left open is done: {:?}", v.reason);
        assert!(v.reason.as_ref().is_some_and(|r| r.contains("converged")));
        Ok(())
    }

    #[test]
    fn a_productive_tail_does_not_terminate() -> Result<(), Error> {
        let cfg = BudgetConfig::default();
        let records = vec![
            RoundRecord { round: 1, new_findings: 2, ..Default::default() },
            RoundRecord { round: 2, new_findings: 1, ..Default::default() },
        ];
        let v = budget_verdict(&records, &cfg);
        assert!(!v.terminate);
        assert!(v.productive);
        assert_eq!(v.dry_streak, 0);
        assert_eq!(v.reason, None);
        Ok(())
    }

    #[test]
    fn an_empty_log_is_not_a_termination() {
        let v = budget_verdict(&[], &BudgetConfig::default());
        assert!(!v.terminate);
        assert_eq!(v.dry_streak, 0);
        assert_eq!(v.reason, None);
    }

    #[test]
    fn the_round_budget_is_explicit_and_spends_in_order() {
        let mut r = Rounds::new(2);
        assert_eq!(r.spend(), Some(1));
        assert_eq!(r.spend(), Some(2));
        assert_eq!(r.spend(), None, "the stop is the budget, stated by returning None");
        assert_eq!(r.remaining(), 0);
        assert_eq!(r.used(), 2);

        let mut zero = Rounds::new(0);
        assert_eq!(zero.spend(), None, "a zero budget plans nothing, and says so immediately");
    }

    #[test]
    fn plan_round_spends_a_round_and_applies_the_gate() -> Result<(), Error> {
        let action = omni_graph::Action::new("curl", serde_json::json!({"url": "example.com"}))?;
        let mut g = Graph::new();
        g.apply(
            &omni_graph::GraphUpdate {
                add_nodes: vec![omni_graph::Node::Action(omni_graph::ActionNode::new(action, 1, 1))],
                add_edges: Vec::new(),
            },
            1,
        );
        let mut rounds = Rounds::new(3);

        let plan = plan_round(&mut g, &omni_graph::ScopeAllowlist::new(["curl"]), &mut rounds, 4)
            .ok_or_else(|| Error::Format("the budget has rounds left".into()))?;
        assert_eq!(plan.round, 1);
        assert_eq!(plan.kept.len(), 1);
        assert!(plan.blocked.is_empty());
        assert_eq!(plan.describe(), "round 1: 1 kept, 0 blocked by the gate");

        let rec = record_of(&plan, 2, 1);
        assert_eq!(rec.new_findings, 1);
        assert_eq!(rec.frontier_size, 1);
        assert_eq!(rounds.remaining(), 2);
        Ok(())
    }

    #[test]
    fn plan_round_reports_budget_exhaustion_as_none() -> Result<(), Error> {
        let mut g = Graph::new();
        let mut rounds = Rounds::new(1);
        assert!(plan_round(&mut g, &omni_graph::ScopeAllowlist::new(Vec::<String>::new()), &mut rounds, 4).is_some());
        assert!(plan_round(&mut g, &DryRun, &mut rounds, 4).is_none());
        // The graph was not touched by the refused round.
        assert_eq!(g.node_count(), 0);
        Ok(())
    }

    #[test]
    fn the_dry_run_gate_refuses_everything_with_the_same_words_every_time() {
        let a = DryRun.evaluate("curl", &serde_json::json!({}));
        let b = DryRun.evaluate("curl", &serde_json::json!({}));
        assert_eq!(a, b, "a gate that changes its answer between calls is not a gate");
        assert!(!a.is_allowed());
        assert!(a.reason.contains("dry run"));
    }

    #[test]
    fn action_ids_from_the_graph_and_the_loop_agree() -> Result<(), Error> {
        // The plan names actions by the same content id the graph derived, so
        // a harness dispatching from a plan cannot dispatch a different call
        // than the one the frontier gated.
        let action = omni_graph::Action::new("curl", serde_json::json!({"url": "x"}))?;
    let direct = content_id_str(r#"{"args":{"url":"x"},"tool":"curl"}"#);
        assert_eq!(
            action.id,
            format!("action:{direct}"),
            "the id is the canonical hash of the action body"
        );
        Ok(())
    }
}
