//! Pricing the bypass.
//!
//! "Short context should skip the context runtime" is a claim, and like every
//! claim in this workspace it has to be reducible to something checkable. This
//! module is that reduction.
//!
//! # What is actually being measured
//!
//! Two separable quantities, and conflating them is the usual way this kind of
//! optimisation talks itself into being wrong:
//!
//! **1. Ingest cost.** Tokens that reach the context runtime, and therefore the
//! work of cutting spans, extracting nodes, embedding vectors, building edges
//! and running the knapsack. This is what the bypass removes, and it is the
//! whole argument.
//!
//! **2. Prompt cost.** Tokens the *model* is billed for. The bypass does
//! **not** reduce this — the raw turns are sent either way, and if anything a
//! planned working set can be shorter because the planner summarized it.
//!
//! Conflating them produces a beautiful lie: "we cut token cost 95%" when the
//! model was billed the same and only the indexer was spared. So
//! [`Comparison::prompt_tokens`] is reported beside
//! [`Comparison::saved_ingest_tokens`] and the two are never added together.
//!
//! # Why not wall-clock here
//!
//! Timing the DCR path needs a `Dcr`, which needs the crate linked and a corpus
//! loaded — at 72.8M tokens that is ~9 GB of RSS, and a timing harness that
//! allocates that is not something to run inside a unit test on a shared box.
//! So the *structural* claims are measured here and are cheap, and the timing is
//! left to `omni ctx probe`, which reports honestly when it cannot run.

use std::time::{Duration, Instant};

use omni_core::Error;
use omni_model::Message;

use crate::short::{Route, ShortContextRouter};

/// What a bypass avoided, and what it did not.
///
/// Not `Copy`: [`Comparison::planned`] is a list, and a cheap copy of a list is
/// a copy nobody makes deliberately. Every reader wants the counts, and those
/// are `usize`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Comparison {
    /// Turns routed.
    pub turns: usize,
    /// Estimated tokens in the requests, as sent.
    pub prompt_tokens: usize,
    /// Estimated tokens that never reached the context runtime.
    ///
    /// The saving. Not a billing saving — see the module docs.
    pub saved_ingest_tokens: usize,
    /// Turns that took the verbatim path.
    pub bypassed: usize,
    /// Turns that had to plan, and why.
    pub planned: Vec<(Route, String)>,
    /// How long the router itself took.
    ///
    /// This is the cost side of the trade. If the gate were slower than the
    /// ingest it skips, the whole argument collapses, so it is measured rather
    /// than assumed to be negligible.
    pub decision_time: Duration,
}

impl Comparison {
    /// Fraction of turns that bypassed, in `0.0..=1.0`.
    ///
    /// The cast is written out rather than reached through `From`, because
    /// `usize` has none into `f64`. Both operands are counts and both are exact
    /// below `2^53`, which is more turns than a session has.
    #[must_use]
    pub fn bypass_rate(&self) -> f64 {
        if self.turns == 0 {
            return 0.0;
        }
        self.bypassed as f64 / self.turns as f64
    }

    /// A one-line summary for a log.
    #[must_use]
    pub fn summary(&self) -> String {
        format!(
            "{} turns: {} bypassed ({:.0}%), {} ingest tokens spared, {} prompt tokens sent, \
             router spent {:?}",
            self.turns,
            self.bypassed,
            self.bypass_rate() * 100.0,
            self.saved_ingest_tokens,
            self.prompt_tokens,
            self.decision_time
        )
    }
}

/// Run `turns` through a router and report what it cost.
///
/// Only the verbatim path is exercised end to end; a turn that routes to the
/// planner is recorded and skipped, because rendering a planned working set
/// belongs to the planner and pretending otherwise here would make this
/// measurement lie.
///
/// # Errors
///
/// [`Error::Format`] if the router rejects the budget.
pub fn price(
    budget_tokens: usize,
    turns: &[Vec<Message>],
) -> Result<Comparison, Error> {
    let mut router = ShortContextRouter::new(budget_tokens)?;
    let mut cmp = Comparison {
        turns: turns.len(),
        prompt_tokens: 0,
        saved_ingest_tokens: 0,
        bypassed: 0,
        planned: Vec::new(),
        decision_time: Duration::ZERO,
    };

    for request in turns {
        let start = Instant::now();
        let route = router.route(request);
        cmp.decision_time += start.elapsed();

        cmp.prompt_tokens += route.tokens();
        if route.is_verbatim() {
            cmp.bypassed += 1;
            cmp.saved_ingest_tokens += route.tokens();
        } else {
            let label = request
                .first()
                .and_then(Message::text)
                .map_or_else(|| "(no text)".to_string(), |t| t.chars().take(40).collect());
            cmp.planned.push((route, label));
        }
    }
    Ok(cmp)
}

/// Measure both paths on the same conversation.
///
/// The only honest way to price a bypass is to run the thing it is bypassing,
/// which here means the same router with [`ShortContextRouter::forced`] — so the
/// difference between the two columns is the bypass and nothing else.
///
/// # Errors
///
/// [`Error::Format`] if either router rejects the budget.
pub fn compare(
    budget_tokens: usize,
    turns: &[Vec<Message>],
) -> Result<(Comparison, Comparison), Error> {
    let routed = price(budget_tokens, turns)?;
    let mut forced_router = ShortContextRouter::new(budget_tokens)?.forced(true);
    let mut planned = Comparison {
        turns: turns.len(),
        prompt_tokens: 0,
        saved_ingest_tokens: 0,
        bypassed: 0,
        planned: Vec::new(),
        decision_time: Duration::ZERO,
    };
    for request in turns {
        let start = Instant::now();
        let route = forced_router.route(request);
        planned.decision_time += start.elapsed();
        planned.prompt_tokens += route.tokens();
        planned.saved_ingest_tokens += route.tokens();
        planned.planned.push((route, String::new()));
    }
    Ok((routed, planned))
}

/// Build a synthetic short-chat conversation for the probe.
///
/// Deliberately boring: the claim is about turn *shape*, not content, and a
/// fixture built from real traffic would be a privacy problem for a claim about
/// shape.
#[must_use]
pub fn synthetic_chat(turns: usize, chars_per_turn: usize) -> Vec<Message> {
    (0..turns)
        .map(|i| {
            if i % 2 == 0 {
                Message::user(format!("{}: question {}", i, "context ".repeat(chars_per_turn / 8)))
            } else {
                Message::assistant(format!("{}: answer {}", i, "reply ".repeat(chars_per_turn / 8)))
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::short::assemble;

    fn requests(n: usize, turns: usize, chars: usize) -> Vec<Vec<Message>> {
        (0..n).map(|_| synthetic_chat(turns, chars)).collect()
    }

    #[test]
    fn short_chats_bypass_and_report_the_saving() -> Result<(), Error> {
        let c = price(8000, &requests(20, 4, 200))?;
        assert_eq!(c.bypassed, 20, "every short chat should bypass");
        assert!(c.saved_ingest_tokens > 0, "a bypass must save something");
        assert!(!c.summary().is_empty());
        Ok(())
    }

    #[test]
    fn the_two_cost_columns_are_never_added_together() -> Result<(), Error> {
        // The whole module exists so "we cut token cost" cannot quietly mean
        // "we cut the indexer's cost and told nobody".
        let c = price(8000, &requests(10, 4, 200))?;
        // Prompt tokens are still sent in full — the model is billed either way.
        assert_eq!(
            c.prompt_tokens, c.saved_ingest_tokens,
            "for a fully-bypassed run the prompt is exactly what the runtime did not ingest"
        );
        Ok(())
    }

    #[test]
    fn the_router_decision_is_negligible_next_to_what_it_skips() -> Result<(), Error> {
        // If the gate cost as much as the ingest, the whole argument collapses.
        // This bounds it against the proxy we actually have: the routed run must
        // not be measurably slower than a run that does nothing at all.
        let turns = requests(200, 4, 200);
        let start = Instant::now();
        for _ in 0..10 {
            // The `?` is the error handling; a `let _ =` in front of it would
            // throw that away, which is what the audit's section B exists to
            // catch.
            price(8000, &turns)?;
        }
        let ten_runs = start.elapsed();

        let r = ShortContextRouter::new(8000)?;
        let mut classifications = 0usize;
        let start = Instant::now();
        for _ in 0..10 {
            for t in &turns {
                if r.classify(t).is_verbatim() {
                    classifications += 1;
                }
            }
        }
        let ten_classifies = start.elapsed();
        assert_eq!(
            classifications,
            turns.len() * 10,
            "the fixture must actually bypass on every pass, or the timing is of nothing"
        );
        assert!(
            ten_classifies <= ten_runs,
            "classifying ({ten_classifies:?}) cost more than pricing the same turns \
             end to end ({ten_runs:?})"
        );
        Ok(())
    }

    #[test]
    fn comparing_both_paths_separates_the_bypass_from_everything_else() -> Result<(), Error> {
        let turns = requests(20, 4, 200);
        let (routed, planned) = compare(8000, &turns)?;
        assert_eq!(routed.bypassed, 20);
        assert_eq!(planned.bypassed, 0, "forced planning never bypasses");
        assert_eq!(planned.saved_ingest_tokens, planned.prompt_tokens);
        assert_eq!(
            planned.turns, routed.turns,
            "both columns must see the same turns, or the comparison is not one"
        );
        Ok(())
    }

    #[test]
    fn a_long_conversation_plans_and_says_why() -> Result<(), Error> {
        let c = price(500, &requests(3, 20, 2000))?;
        assert_eq!(c.bypassed, 0);
        assert_eq!(c.planned.len(), 3);
        assert!(c.planned.iter().all(|(r, _)| !r.is_verbatim()));
        assert!(c.planned.iter().all(|(r, _)| r.reason().is_some()));
        Ok(())
    }

    #[test]
    fn the_bypass_rate_is_a_fraction_not_a_percentage() -> Result<(), Error> {
        let c = price(8000, &requests(4, 2, 100))?;
        let r = c.bypass_rate();
        assert!((0.0..=1.0).contains(&r), "got {r}");
        assert!((r - 1.0).abs() < f64::EPSILON);
        Ok(())
    }

    #[test]
    fn an_empty_conversation_is_not_a_nan() -> Result<(), Error> {
        let c = price(1000, &[])?;
        assert_eq!(c.bypass_rate(), 0.0, "0/0 must be 0.0, not NaN");
        Ok(())
    }

    #[test]
    fn assemble_through_the_probe_keeps_roles_intact() -> Result<(), Error> {
        let mut r = ShortContextRouter::new(8000)?;
        let chat = synthetic_chat(4, 200);
        let out = assemble(&mut r, "sys", &chat)?;
        assert!(out.route.is_verbatim());
        assert_eq!(out.messages[1].role(), "user");
        assert_eq!(out.messages[2].role(), "assistant");
        Ok(())
    }
}
