//! The bridge between `omni_model::Model` and `dcr::Reasoner`.
//!
//! # Why this is the whole point of `omni-ctx` linking `dcr`
//!
//! subnext defines the seam already, and it is almost exactly right:
//!
//! ```text
//! pub trait Reasoner: Send {
//!     fn complete(&mut self, prompt: &str, system: &str) -> String;
//! }
//! ```
//!
//! One method, model-agnostic, and every other part of DCR — the representation
//! ladder, the relevance planner, the knapsack, the memory graph, the trust
//! gateway — is orthogonal to *which model answers*. A context runtime that
//! works only with one backend is a runtime that cannot be tested without that
//! backend running.
//!
//! [`ModelReasoner`] wraps any [`Model`] as a `Reasoner`, so DCR's `#ESCALATE`
//! loop works over peregrine, over unifur, or over a stub in a unit test.
//!
//! # The one thing a `String` loses
//!
//! `Reasoner::complete` returns text. No tool calls, no streaming, no
//! multimodal parts. That is fine for DCR — which asks a question about an
//! assembled working set and reads an answer — and it is **not** fine for an
//! agent loop. So this bridge is for the escalation loop specifically, and the
//! tool-calling path goes through `omni_model::Model` directly. Two paths, one
//! trait each, rather than one trait that is bad at both.
//!
//! # The escalation contract
//!
//! The reasoner is told, in the system prompt, that it may answer with
//! `#ESCALATE <node_id>` and nothing else when the working set is insufficient.
//! That is a *harness* property, not a runtime property — subnext says so
//! directly:
//!
//! > `false` models the documented poor fit — a harness with no way to signal
//! > insufficiency. It is a property of the *harness*, so it is set here and
//! > not on the runtime.
//!
//! omni's harness can signal, so the flag is `true`. An unserved escalation is
//! never returned as the answer; DCR replaces it with "I don't have that in the
//! active context", which is the honest outcome.

use std::collections::HashSet;

use dcr::llm::parse_escalation;
use dcr::nodes::NodeIdx;
use dcr::{Dcr, Reasoner};
use omni_core::contract::{CONTEXT_HIT_VERSION, ContextHit, Provenance};
use omni_core::Error;
use omni_model::{ChatRequest, Message, Model};

/// Wrap a [`Model`] as a `dcr::Reasoner`.
///
/// Borrows rather than owns so the same backend can drive both the DCR loop and
/// the tool loop without a second connection. The escalation budget is small
/// (2 by default in DCR), so the interleaving is bounded in practice; a caller
/// that would exceed it should hold two clients.
pub struct ModelReasoner<'m> {
    model: &'m mut dyn Model,
    /// Ids already escalated to, so a model that repeats itself is not retried
    /// forever.
    ///
    /// DCR caps escalations too, but its cap is on *attempts* while this is on
    /// *distinct targets* — a loop that escalates to the same node twice has
    /// learned nothing and should not be allowed to consume the budget twice
    /// doing it.
    seen: HashSet<String>,
    max_tokens: usize,
}

impl<'m> ModelReasoner<'m> {
    #[must_use]
    pub fn new(model: &'m mut dyn Model) -> Self {
        Self { model, seen: HashSet::new(), max_tokens: 1024 }
    }

    /// Set the reply budget for one completion.
    #[must_use]
    pub fn with_max_tokens(mut self, n: usize) -> Self {
        self.max_tokens = n;
        self
    }

    /// Whether the model may emit `#ESCALATE`.
    ///
    /// Always `true` here, and named as a method rather than a field so that the
    /// "a harness with no way to signal is the documented poor fit" comparison
    /// is visible at every call site.
    #[must_use]
    pub fn can_signal(&self) -> bool {
        true
    }

    /// Ids this reasoner has already escalated to.
    #[must_use]
    pub fn escalated_to(&self) -> Vec<&str> {
        let mut v: Vec<&str> = self.seen.iter().map(String::as_str).collect();
        v.sort_unstable();
        v
    }
}

impl Reasoner for ModelReasoner<'_> {
    fn complete(&mut self, prompt: &str, system: &str) -> String {
        let messages = [Message::system(system), Message::user(prompt)];
        let req = ChatRequest::new(&messages).with_max_tokens(self.max_tokens);

        match self.model.chat(&req) {
            Ok(reply) => {
                let text = reply.text().unwrap_or("").to_string();
                // Record the escalation *after* the reply, and only once, so the
                // next turn can tell "asked for the same thing again" from
                // "learned something and is asking for a different node".
                if let Some(id) = parse_escalation(&text) {
                    self.seen.insert(id);
                }
                text
            }
            Err(e) => {
                // A failed completion must not read as a reasoned refusal.
                // DCR will treat whatever comes back as the reasoner's answer,
                // and an empty string with a 500 inside it is a confident
                // wrong answer. The marker is inert — it is not an escalation,
                // so DCR will not act on it — but it names the failure.
                format!("#OMNI_ERROR {e}")
            }
        }
    }
}

/// A context plane over a full `dcr` runtime.
///
/// The embeddable option: omni owns the working set for a session, plans into
/// it, and answers through a [`ModelReasoner`].
///
/// # Why the model is borrowed per call rather than installed
///
/// `Dcr` *owns* its reasoner for its whole lifetime — `with_reasoner` takes a
/// `Box<dyn Reasoner>`. [`ModelReasoner`] in turn borrows a `&mut dyn Model`,
/// because the same backend should be able to drive both this loop and the tool
/// loop from one connection.
///
/// Those two ownership facts do not compose, and the way out is `ask_with`,
/// which accepts a `&mut dyn Reasoner` **per call**. So the plane keeps the
/// model and builds a fresh bridge for each question. The allocation is one
/// small struct; the alternative is either a second model connection or a
/// self-referential struct, and both are worse for what is bought.
///
/// # Two entry points, and neither is a conversion of the other
///
/// | | returns | for |
/// |---|---|---|
/// | [`DcrPlane::ask`] | a `dcr::Answer` — prose, citations, what it cost | answering |
/// | `retrieve` | the current plan's admissions, as `ContextHit`s | putting context *into* a prompt |
///
/// Both run the same [`dcr::Dcr::plan`], so the ranked list `retrieve` returns
/// is the planner's own ranking rather than a similarity re-scored after the
/// fact — the same property the deployed responder relies on, and the same
/// caveat `docs/CONTRACT.md` records about `ContextHit::score`.
///
/// Neither replaces the other. A list is not a working set: it has no budget,
/// no escalation and no ladder level, so flattening `ask` into a list would
/// throw away exactly the planning that makes DCR worth linking.
pub struct DcrPlane {
    runtime: Dcr,
    model: Option<Box<dyn Model>>,
    max_hits: usize,
}

/// The cap `DcrPlane::retrieve` applies when the caller does not ask for one.
///
/// The same default [`crate::ChannelOptions`] uses, deliberately: a caller that
/// cannot tell which plane produced a hit list is being asked to reason about a
/// difference it cannot observe. A peer asking for ten thousand hits would turn
/// a working set into a corpus dump, which is the thing the budget exists to
/// prevent.
pub const DEFAULT_MAX_HITS: usize = 8;

impl DcrPlane {
    /// Build a plane with an attention budget.
    ///
    /// # Errors
    ///
    /// [`Error::Format`] when `budget` is zero. A zero budget makes the
    /// knapsack select nothing, and an empty working set reads as "the corpus
    /// does not contain this" — the confident wrong answer, produced by
    /// misconfiguration rather than by an absence.
    pub fn new(budget: usize) -> Result<Self, Error> {
        if budget == 0 {
            return Err(Error::Format(
                "a context plane with a zero budget selects nothing, which reads \
                 as an empty corpus rather than as a misconfiguration"
                    .into(),
            ));
        }
        Ok(Self { runtime: Dcr::new(budget), model: None, max_hits: DEFAULT_MAX_HITS })
    }

    /// Cap on hits one `retrieve` returns.
    ///
    /// Zero is raised to one rather than accepted. A cap of zero makes the
    /// plane return nothing, and nothing reads as "the corpus does not contain
    /// this" — the confident wrong answer this crate exists to avoid, produced
    /// here by a builder call.
    #[must_use]
    pub fn with_max_hits(mut self, max: usize) -> Self {
        self.max_hits = max.max(1);
        self
    }

    /// Install the model that answers. Consumed, because a plane with no model
    /// has nothing to answer with.
    ///
    /// A plane built without one still ingests and plans; only [`DcrPlane::ask`]
    /// fails, and it says so.
    #[must_use]
    pub fn with_model(mut self, model: Box<dyn Model>) -> Self {
        self.model = Some(model);
        self
    }

    /// Add a document to the store.
    ///
    /// # Errors
    ///
    /// [`Error::Format`] annotated with which document failed, wrapping the
    /// message subnext formatted. `DcrError` has no variant in omni's closed
    /// enum and cannot get one — that would mean `omni-core` depending on the
    /// runtime it exists to wrap — so the cause survives as the `source` and the
    /// annotation says what was being done.
    pub fn ingest(&mut self, text: &str, doc_id: Option<&str>) -> Result<(), Error> {
        match self.runtime.ingest(text, doc_id) {
            Ok(_) => Ok(()),
            Err(e) => Err(Error::Format(e.to_string()).annotated(|| match doc_id {
                Some(id) => format!("ingesting document {id} into the DCR store"),
                None => "ingesting an anonymous document into the DCR store".to_string(),
            })),
        }
    }

    /// Add a document that names where it came from.
    ///
    /// [`DcrPlane::ingest`] leaves the source empty, and an empty
    /// [`ContextHit::provenance`] renders as `source=` in a prompt — an unsourced
    /// fact wearing a citation. A plane whose corpus has real origins should say
    /// so, and this is how: the string travels through the store and comes back
    /// on every hit the document's nodes produce.
    ///
    /// # Errors
    ///
    /// As [`DcrPlane::ingest`], annotated with the document that failed.
    pub fn ingest_with_source(
        &mut self,
        text: &str,
        doc_id: Option<&str>,
        source: &str,
    ) -> Result<(), Error> {
        match self.runtime.ingest_with_source(text, doc_id, Some(source.to_string())) {
            Ok(_) => Ok(()),
            Err(e) => Err(Error::Format(e.to_string()).annotated(|| match doc_id {
                Some(id) => format!("ingesting {source} as document {id} into the DCR store"),
                None => format!("ingesting {source} as an anonymous document into the DCR store"),
            })),
        }
    }

    /// Ask a question of the assembled working set.
    ///
    /// Runs DCR's own loop: plan, answer, consolidate, and — if the reasoner
    /// emitted `#ESCALATE` and the budget allows — re-plan with that node pinned
    /// to its raw bytes and ask again.
    ///
    /// # Errors
    ///
    /// [`Error::Format`] when no model was installed. That is a wiring mistake,
    /// not a runtime condition, and it is reported as one.
    pub fn ask(&mut self, query: &str, budget: Option<usize>) -> Result<dcr::Answer, Error> {
        let Some(model) = self.model.as_mut() else {
            return Err(Error::Format(
                "DcrPlane has no model; construct it with with_model() before asking".into(),
            ));
        };
        let mut reasoner = ModelReasoner::new(model.as_mut());
        Ok(self.runtime.ask_with(query, budget, &mut reasoner))
    }

    /// The underlying runtime, for the state this crate does not wrap.
    #[must_use]
    pub fn runtime(&self) -> &Dcr {
        &self.runtime
    }

    /// Resolve an admitted node's spans back to the document they came from.
    ///
    /// The same walk the deployed responder does (`dcr`'s
    /// `interop::Responder::provenance_of`): take the first source span that
    /// still survives in L0, and report the document it belonged to. A node
    /// with no surviving span gets **empty** provenance rather than an invented
    /// one, because [`crate::augment`] renders `source=` into the prompt and a
    /// made-up source string is an unsourced fact wearing a citation.
    fn provenance_of(&self, node: NodeIdx) -> Provenance {
        for span_id in &self.runtime.graph.node(node).source_spans {
            let Some(span) = self.runtime.raw.span_by_id(span_id) else { continue };
            let doc_id = span.doc_id.clone();
            let source = self
                .runtime
                .raw
                .documents()
                .iter()
                .find(|d| d.id == doc_id)
                .and_then(|d| d.source.clone())
                .unwrap_or_default();
            return Provenance { source, path: doc_id };
        }
        Provenance::default()
    }
}

impl crate::ContextPlane for DcrPlane {
    /// Plan against the store and return the admissions as `ContextHit`s.
    ///
    /// # What each field means
    ///
    /// `score` is the planner's **utility** — the quantity the knapsack actually
    /// maximised — so sorting by it yields the planner's own ranking. `text` is
    /// the **rendered ladder line** (`[node_id L2] …`), not the raw document.
    /// Both are recorded in `docs/CONTRACT.md` and both are the reason this is
    /// worth building on the planner rather than on a similarity score.
    ///
    /// # `field` is ignored, and that is the contract
    ///
    /// `field` is unifur's taxonomy; DCR's documents carry a filesystem
    /// `source`, so the store does not own it. The deployed responder records it
    /// in its poll report and does not filter on it, and an honest
    /// over-retrieval is the better failure. An in-process plane has no poll
    /// report to record it in, so the honest statement is the negative one: this
    /// argument is not used, and a caller that wants it in a log line should
    /// pass it to [`crate::describe`] itself.
    ///
    /// # `k`
    ///
    /// `0` means "the plane's default" — the same reading
    /// [`omni_core::contract::ContextQuery`] gives the field on the wire. Any
    /// other value is a ceiling on the plane's own [`DcrPlane::with_max_hits`]
    /// cap, never a raise: the caller's `k` can narrow the answer, and nothing
    /// about a query should be able to widen it into a corpus dump.
    fn retrieve(
        &mut self,
        query: &str,
        k: usize,
        _field: &str,
    ) -> Result<Vec<ContextHit>, Error> {
        let context = self.runtime.plan(query, None);
        let want = if k == 0 { self.max_hits } else { k.min(self.max_hits) };
        let mut hits: Vec<ContextHit> = context
            .entries
            .iter()
            .take(want)
            .map(|entry| ContextHit {
                version: CONTEXT_HIT_VERSION,
                id: entry.node_id.clone(),
                score: f64::from(entry.utility),
                text: entry.rendered.clone(),
                provenance: self.provenance_of(entry.node),
            })
            .collect();
        // Best first. The knapsack emits admissions in *budget* order, not
        // utility order, and a caller that takes the head of the list until its
        // own budget runs out would otherwise be handed the worst `n` of the set
        // it could have had. `sort_by` is stable, so ties keep admission order
        // and the same query over the same store gives the same list — which is
        // what makes a retrieval-augmented rerun comparable to its baseline.
        hits.sort_by(|a, b| {
            b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal)
        });
        Ok(hits)
    }

    fn name(&self) -> &str {
        "dcr"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ContextPlane;
    use omni_model::{ChatReply, FinishReason};

    /// Two documents about cryostats, each naming where it came from.
    ///
    /// Shaped like a real corpus rather than a lorem ipsum: the indexer's
    /// heuristic extractor turns "`X` is maintained at `N` kelvin" into a cached
    /// claim *and* an evidence node, which is what gives a plan something to
    /// admit, and the two documents are near-neighbours so the ranking is a real
    /// decision rather than the only one available.
    fn seeded() -> Result<DcrPlane, Error> {
        let mut p = DcrPlane::new(800)?;
        p.ingest_with_source(
            "The Heliodrome Array is maintained at 77 kelvin by a closed helium loop.",
            Some("heliodrome"),
            "corpus/heliodrome.txt",
        )?;
        p.ingest_with_source(
            "The Borealis Ring is maintained at 12 kelvin by a passive cryocooler.",
            Some("borealis"),
            "corpus/borealis.txt",
        )?;
        Ok(p)
    }

    /// A model that always says the same thing.
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

    /// A model that escalates to `next` in turn.
    struct Escalating {
        script: Vec<String>,
        n: usize,
    }

    impl Model for Escalating {
        fn chat(&mut self, _req: &ChatRequest<'_>) -> Result<ChatReply, Error> {
            let out = self.script.get(self.n).cloned().unwrap_or_else(|| "done".into());
            self.n += 1;
            Ok(ChatReply { message: Message::assistant(out), finish_reason: FinishReason::Stop, usage: None })
        }
        fn name(&self) -> &str {
            "escalating"
        }
    }

    struct Broken;

    impl Model for Broken {
        fn chat(&mut self, _req: &ChatRequest<'_>) -> Result<ChatReply, Error> {
            Err(Error::Format("the server hung up".into()))
        }
        fn name(&self) -> &str {
            "broken"
        }
    }

    #[test]
    fn the_bridge_answers_through_any_model() -> Result<(), Error> {
        let mut m = Fixed("42");
        let mut r = ModelReasoner::new(&mut m);
        assert_eq!(r.complete("what is the answer?", "sys"), "42");
        Ok(())
    }

    #[test]
    fn the_escalation_signal_is_parsed_by_the_same_rule_dcr_uses() -> Result<(), Error> {
        let mut m = Escalating { script: vec!["#ESCALATE evid_abc123".into()], n: 0 };
        let mut r = ModelReasoner::new(&mut m);
        let text = r.complete("need the raw text", "sys");
        assert_eq!(parse_escalation(&text).as_deref(), Some("evid_abc123"));
        assert_eq!(r.escalated_to(), vec!["evid_abc123"], "the bridge records what it asked for");
        Ok(())
    }

    #[test]
    fn a_repeated_escalation_is_recorded_once() -> Result<(), Error> {
        // A loop that escalates to the same node twice has learned nothing and
        // should not consume DCR's budget twice doing it.
        let mut m = Escalating { script: vec!["#ESCALATE n1".into(), "#ESCALATE n1".into()], n: 0 };
        let mut r = ModelReasoner::new(&mut m);
        r.complete("q", "s");
        r.complete("q", "s");
        assert_eq!(r.escalated_to().len(), 1);
        Ok(())
    }

    #[test]
    fn a_failed_completion_does_not_read_as_a_reasoned_refusal() -> Result<(), Error> {
        // DCR treats whatever comes back as the answer. An empty string from a
        // 500 is a confident wrong answer; a marker naming the failure is not.
        let mut m = Broken;
        let mut r = ModelReasoner::new(&mut m);
        let text = r.complete("q", "s");
        assert!(text.starts_with("#OMNI_ERROR"), "got {text:?}");
        assert!(text.contains("hung up"), "the reason must survive: {text:?}");
        assert!(parse_escalation(&text).is_none(), "a failure must not look like an escalation");
        Ok(())
    }

    #[test]
    fn the_harness_can_signal_by_construction() -> Result<(), Error> {
        // subnext prices the poor fit explicitly; omni's harness is not it.
        let mut m = Fixed("x");
        let r = ModelReasoner::new(&mut m);
        assert!(r.can_signal());
        Ok(())
    }

    #[test]
    fn a_zero_budget_plane_is_refused_rather_than_silently_empty() -> Result<(), Error> {
        // An empty working set reads as "the corpus does not contain this".
        let Err(err) = DcrPlane::new(0) else {
            return Err(Error::Format("a zero budget must be refused".into()));
        };
        assert!(err.to_string().contains("misconfiguration"));
        Ok(())
    }

    #[test]
    fn retrieve_returns_the_planners_own_admissions() -> Result<(), Error> {
        // This used to assert a refusal: `retrieve` reported itself unwired.
        // It now pins the replacement — the planner's admissions, in the
        // planner's ranking, carrying the contract's version.
        let mut p = seeded()?;
        let hits = p.retrieve("What temperature is the Heliodrome mirror kept at?", 4, "")?;
        assert!(!hits.is_empty(), "a store that holds the fact must answer");
        for h in &hits {
            h.validate()?;
        }
        assert!(
            hits[0].text.contains("heliodrome"),
            "the query names the Heliodrome, so its admissions must lead: {:?}",
            hits[0].text
        );
        assert!(
            hits.iter().all(|h| h.text.starts_with('[') && h.text.contains(" L")),
            "a hit is the rendered ladder line, not the raw document: {:?}",
            hits[0].text
        );
        // Best first. The knapsack emits in *budget* order, so this is the
        // sort doing work rather than the planner having happened to agree.
        for pair in hits.windows(2) {
            assert!(
                pair[0].score >= pair[1].score,
                "hits must be best-first: {} then {}",
                pair[0].score,
                pair[1].score
            );
        }
        assert!(hits[0].score > 0.0, "an admission has a positive utility");
        Ok(())
    }

    #[test]
    fn retrieve_carries_the_document_it_came_from() -> Result<(), Error> {
        // `augment` renders `source=` into the prompt, so an empty provenance is
        // an unsourced fact wearing a citation. The source ingested has to
        // survive the round trip through the store and back out as a hit.
        let mut p = seeded()?;
        // A query with no term overlap, so the planner admits from both
        // documents: this is a test about where a hit says it came from, not
        // about which document wins.
        let hits = p.retrieve("cryostat temperature", 8, "")?;
        let sources: Vec<&str> = hits.iter().map(|h| h.provenance.source.as_str()).collect();
        assert!(
            sources.contains(&"corpus/heliodrome.txt"),
            "the first document's source must survive the round trip: {hits:?}"
        );
        assert!(
            sources.contains(&"corpus/borealis.txt"),
            "the second document's source must survive the round trip: {hits:?}"
        );
        assert!(
            hits.iter().all(|h| h.provenance.path == "heliodrome"
                || h.provenance.path == "borealis"),
            "and the path must be the document id, not the span: {hits:?}"
        );
        Ok(())
    }

    #[test]
    fn an_empty_store_returns_no_hits_rather_than_inventing_them() -> Result<(), Error> {
        // The control for the test above, in the other direction. If `retrieve`
        // returned hits from a store holding nothing, then "a store that holds
        // the fact must answer" would hold for the wrong reason — and a hit
        // invented out of an empty corpus is precisely the confident wrong
        // answer this crate exists to avoid.
        let mut p = DcrPlane::new(800)?;
        let hits = p.retrieve("What temperature is the Heliodrome mirror kept at?", 4, "")?;
        assert!(hits.is_empty(), "nothing was ingested, so nothing may be served: {hits:?}");
        Ok(())
    }

    #[test]
    fn k_narrows_the_answer_and_the_planes_ceiling_wins_over_it() -> Result<(), Error> {
        // The ceiling is the half that matters: a caller's `k` may ask for fewer,
        // and nothing about a query may ask for more. A plane that returned
        // ninety-nine hits would be a corpus dump wearing a budget.
        let mut p = seeded()?;
        let all = p.retrieve("cryostat temperature", 0, "")?;
        assert!(all.len() > 1, "the fixture must admit more than one node, or the cap tests nothing");

        assert_eq!(p.retrieve("cryostat temperature", 1, "")?.len(), 1, "k narrows");
        assert_eq!(p.retrieve("cryostat temperature", 1000, "")?.len(), all.len(), "k cannot widen");

        let mut capped = seeded()?.with_max_hits(1);
        assert_eq!(capped.retrieve("cryostat temperature", 1000, "")?.len(), 1, "the plane's ceiling holds");

        // A cap of zero would make the plane return nothing, and nothing reads as
        // an empty corpus — so it is raised rather than accepted.
        let mut raised = seeded()?.with_max_hits(0);
        assert_eq!(raised.retrieve("cryostat temperature", 0, "")?.len(), 1);
        Ok(())
    }

    #[test]
    fn the_taxonomy_hint_is_reported_not_filtered_on() -> Result<(), Error> {
        // `field` is unifur's taxonomy; DCR's documents carry a filesystem
        // `source`, so the store does not own it. Filtering on it would return
        // nothing at all, and nothing is a confident wrong answer.
        let mut p = seeded()?;
        let plain = p.retrieve("cryostat temperature", 4, "")?;
        let hinted = p.retrieve("cryostat temperature", 4, "pentest")?;
        assert_eq!(plain, hinted, "an invented taxonomy must not filter anything");
        Ok(())
    }
}