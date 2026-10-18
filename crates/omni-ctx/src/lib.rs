//! A working set that outlives the attention window.
//!
//! # What a `ContextPlane` is for
//!
//! The model sees a few thousand tokens. The work does not fit in a few thousand
//! tokens. Everything here is about making that gap survivable *without* putting
//! it in the weights — the discipline unifur's `interop.rs` states as "context
//! lives outside the model; the corpus, memory and long documents are owned and
//! served elsewhere".
//!
//! Three implementations, one trait:
//!
//! | Impl | Holds the corpus? | Speaks |
//! |---|---|---|
//! | [`LocalIndex`] | yes, in-process | nothing |
//! | [`Channel`] | no | the deployed v1 file channel |
//! | [`DcrPlane`] | yes, a full `dcr` runtime | the same channel |
//!
//! The local one is not a toy fallback. It is what makes the seam testable
//! without a responder process, and on the 1M-token fixture it is *faster* than
//! recomputing document frequencies per query — see [`index`].
//!
//! # The honest direction of retrieval
//!
//! Over-retrieving is recoverable; under-retrieving is not. A planner that
//! returns three useless blocks has wasted some budget; one that omits the
//! answer has produced a confident wrong answer, and the model has no way to
//! tell the difference. Every ranking decision below is therefore biased
//! toward returning the marginal block rather than none.
//!
//! The DCR side states the same thing about its `field` hint: the taxonomy is
//! recorded and *not* filtered on, because the store does not own it.
//!
//! ```
//! use omni_ctx::{LocalIndex, augment};
//! use serde_json::json;
//!
//! // Shaped like the real fixture: a fact buried in high-frequency filler.
//! let idx = LocalIndex::build(vec![(
//!     "doc-a".into(),
//!     "The Heliodrome Array keeps its primary mirror at 77 kelvin. qx1 qx2".into(),
//!     "heliodrome.txt".into(),
//!     "heliodrome.txt".into(),
//! )]);
//!
//! let hits = idx.retrieve("At what temperature is the Heliodrome mirror?", 3);
//! assert_eq!(hits[0].id, "doc-a");
//!
//! // The designed property: with no hits the prompt is byte-identical, so a
//! // retrieval failure cannot silently change the agent's behaviour.
//! assert_eq!(augment("summarise", &[], 400), "summarise");
//! # let _ = json!({});
//! # Ok::<(), omni_core::Error>(())
//! ```

#![forbid(unsafe_code)]
#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

pub mod channel;
pub mod cost;
pub mod index;
pub mod reasoner;
pub mod short;

use omni_core::contract::{ContextHit, ContextQuery};
use omni_core::Error;
use serde_json::Value;

pub use channel::{Channel, ChannelOptions, PollReport};
pub use cost::{Comparison, compare, price};
pub use index::{IndexStats, LocalIndex, tokenize};
pub use reasoner::{DEFAULT_MAX_HITS, DcrPlane, ModelReasoner};
pub use short::{Assembled, PlanReason, Route, ShortContextRouter, assemble};

/// A source of planned context.
pub trait ContextPlane {
    /// Retrieve the `k` most relevant blocks for `query`.
    ///
    /// `field` is a taxonomy hint the peer may use to route; a plane that does
    /// not own that taxonomy records it and does not filter on it. See the
    /// module docs.
    fn retrieve(&mut self, query: &str, k: usize, field: &str) -> Result<Vec<ContextHit>, Error>;

    /// A stable name for logs and for the reward fingerprint.
    fn name(&self) -> &str;
}

/// Turn retrieved hits into a prompt block.
///
/// # The designed property
///
/// With no hits the prompt comes back **byte-identical**. That is the whole
/// point: a context plane that is down, empty, or slow must degrade to "no
/// context", never to "the prompt is now subtly different". An agent that
/// behaves differently when retrieval fails is an agent whose behaviour depends
/// on a subsystem it was not told about.
///
/// Budgeted in characters, not tokens. omni's token counter is an estimate
/// ([`omni_model::CharHeuristic`]) and a budget expressed in estimated tokens is
/// a budget that is wrong at the limit. Characters are exact and the unit is
/// stated rather than implied.
pub fn augment(prompt: &str, hits: &[ContextHit], max_chars: usize) -> String {
    if hits.is_empty() {
        return prompt.to_string();
    }
    let mut block = String::new();
    let mut used = 0usize;
    for hit in hits {
        let header = format!("[context id={} source={}]", hit.id, hit.provenance.source);
        let cost = header.chars().count() + 1;
        if cost >= max_chars.saturating_sub(used) {
            break;
        }
        let budget = max_chars - used - cost;
        let body: String = hit.text.chars().take(budget).collect();
        block.push_str(&header);
        block.push('\n');
        block.push_str(&body);
        block.push_str("\n[/context]\n");
        used += cost + body.chars().count();
    }
    if block.is_empty() {
        prompt.to_string()
    } else {
        format!("{block}\n{prompt}")
    }
}

/// How a plane should be described in a log line.
#[must_use]
pub fn describe(name: &str, hits: usize) -> Value {
    serde_json::json!({ "plane": name, "hits": hits })
}

/// The default `field` omni sends.
///
/// Empty, because omni does not own a taxonomy and an invented one would filter
/// nothing while looking like it filtered something.
pub const DEFAULT_FIELD: &str = "";

/// A convenience constructor for a query at the current contract version.
#[must_use]
pub fn query(text: &str, k: usize) -> ContextQuery {
    ContextQuery::new(text, k, DEFAULT_FIELD)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_hits_leaves_the_prompt_byte_identical() -> Result<(), Error> {
        // The property that keeps a retrieval failure from changing behaviour.
        let prompt = "summarise the findings";
        assert_eq!(augment(prompt, &[], 400), prompt);
        assert_eq!(augment(prompt, &[], 0), prompt);
        Ok(())
    }

    #[test]
    fn hits_are_budgeted_and_bounded() -> Result<(), Error> {
        let hits = LocalIndex::build(vec![
            ("doc-a".into(), "alpha ".repeat(100), "a".into(), "a".into()),
            ("doc-b".into(), "beta ".repeat(100), "b".into(), "b".into()),
        ])
        .retrieve("alpha beta", 5);
        assert!(!hits.is_empty(), "the fixture must actually retrieve, or the budget is of nothing");
        let out = augment("go", &hits, 300);
        assert!(out.len() < 700, "budget must be respected, got {}", out.len());
        assert!(out.starts_with("[context id="), "blocks must be clearly marked");
        assert!(out.ends_with("go"), "the prompt must survive at the end");
        Ok(())
    }

    #[test]
    fn a_block_that_cannot_fit_is_dropped_rather_than_mangled() -> Result<(), Error> {
        let hits = LocalIndex::build(vec![("doc-a".into(), "x".into(), "s".into(), "s".into())])
            .retrieve("x", 1);
        // A budget smaller than the header alone admits no block at all, and the
        // prompt comes back untouched — not with a truncated header.
        assert_eq!(augment("go", &hits, 4), "go");
        Ok(())
    }

    #[test]
    fn the_default_field_is_empty_because_omni_owns_no_taxonomy() {
        // An invented taxonomy would filter nothing while appearing to filter
        // something.
        assert_eq!(DEFAULT_FIELD, "");
        assert_eq!(query("q", 3).field, "");
        assert_eq!(query("q", 3).k, 3);
    }

    #[test]
    fn tokenizing_drops_single_characters_and_lowercases() {
        // In a corpus of noisy tokens a one-character term has a document
        // frequency near the corpus size, so its idf is ~0 and it costs a lookup
        // to contribute nothing.
        let t = tokenize("The Heliodrome, Array: 77K! (x)");
        assert!(!t.contains(&"x".to_string()), "single characters are noise");
        assert!(t.contains(&"heliodrome".to_string()));
        assert!(t.contains(&"array".to_string()));
    }
}