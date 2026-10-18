//! The reward half of the harness-edit loop.
//!
//! The loop is: freeze a target, batch its failures into a
//! [`omni_core::HookPacket`], give the batch to a harness engineer, sandbox the
//! patch, rerun the same target on the same tasks, and score the patch by the
//! *difference of measured reruns*. This crate exists because of one sentence
//! in AGENTS.md:
//!
//! > Identities are content-derived, never positional — **matching integer
//! > indices is not proof of a paired comparison.**
//!
//! So a baseline and a patch run are two [`Batch`]es keyed by
//! [`Identity::id`], [`reward`] pairs them by that key only, and whatever
//! could not be paired is **named** rather than dropped: a task missing from
//! the patched run is an unpaired-by-side list entry, a task whose run hit an
//! environment failure is reported in its own bucket, and a structurally
//! invalid patch is a zero by definition — not a missing evaluation, and not a
//! no-op that looks like one.
//!
//! Nothing here computes a reward a human has to trust. Every number has a
//! `matched` count next to it, and every count has a list of what it excluded.

#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
#![forbid(unsafe_code)]

use std::collections::{BTreeMap, BTreeSet};

use omni_core::Error;
use serde::{Deserialize, Serialize};

// ── Identity ────────────────────────────────────────────────────────────────

/// The identity of one (task, configuration) pair, content-derived.
///
/// Everything that can change a run's output belongs here, per rule 8. The
/// task's own bytes are the largest part; the knobs and the model seat next to
/// them because two "matched" runs that differ in one environment variable are
/// the worst kind of unmatched — the kind that looks paired.
///
/// The id is the canonical hash of the whole fingerprint. Two callers who build
/// the same fingerprint with the same knobs in a different insertion order get
/// the same id; two callers who differ in one knob character do not.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Identity {
    /// The task as the target's own interface sees it.
    pub input: String,
    /// Every output-changing environment knob, by name and resolved value.
    pub knobs: BTreeMap<String, String>,
    /// The model identity, when the run went through a model seam.
    pub model: Option<String>,
    /// The completion budget, when it was pinned.
    pub max_tokens: Option<usize>,
}

impl Identity {
    /// An identity for a task with default knobs.
    #[must_use]
    pub fn new(input: &str) -> Self {
        Self {
            input: input.to_string(),
            knobs: BTreeMap::new(),
            model: None,
            max_tokens: None,
        }
    }

    /// Add or replace one knob. Order of insertion cannot matter: the
    /// fingerprint hashes a canonical form, and the map is ordered.
    #[must_use]
    pub fn with_knob(mut self, name: &str, value: &str) -> Self {
        self.knobs.insert(name.to_string(), value.to_string());
        self
    }

    /// Pin the model identity.
    #[must_use]
    pub fn with_model(mut self, model: &str) -> Self {
        self.model = Some(model.to_string());
        self
    }

    /// Pin the completion budget.
    #[must_use]
    pub fn with_max_tokens(mut self, n: usize) -> Self {
        self.max_tokens = Some(n);
        self
    }

    /// The content-derived id, over the canonical fingerprint.
    ///
    /// # Errors
    ///
    /// [`Error::Format`] if the fingerprint cannot be canonicalized, which the
    /// string-only shape cannot produce today — kept as `Result` so a future
    /// fingerprint field that can carry a number inherits the refusal for free
    /// rather than silently hashing one.
    pub fn id(&self) -> Result<String, Error> {
        let knobs_json = serde_json::Value::Object(
            self.knobs
                .iter()
                .map(|(k, v)| (k.clone(), serde_json::Value::String(v.clone())))
                .collect(),
        );
        let body = serde_json::json!({
            "input": self.input,
            "knobs": knobs_json,
            "model": self.model,
            "max_tokens": self.max_tokens,
        });
        let bytes = omni_core::canonical_bytes(&body)?;
        Ok(omni_core::content_id(&bytes))
    }
}

// ── Batches ─────────────────────────────────────────────────────────────────

/// One rerun's outcome.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RunOutcome {
    /// The task ran and produced its measure.
    Scored(f64),
    /// The environment failed before a score existed. Reported separately;
    /// never folded into a zero.
    MissingEnv(String),
}

/// The record of one arm of the comparison, keyed by identity id.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Batch {
    runs: BTreeMap<String, RunOutcome>,
}

impl Batch {
    /// An empty batch.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Record one run under its identity id.
    pub fn record(&mut self, id: impl Into<String>, outcome: RunOutcome) {
        self.runs.insert(id.into(), outcome);
    }

    /// How many runs the batch holds, scored or not.
    #[must_use]
    pub fn len(&self) -> usize {
        self.runs.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.runs.is_empty()
    }

    /// One run's outcome under an id.
    #[must_use]
    pub fn outcome_of(&self, id: &str) -> Option<&RunOutcome> {
        self.runs.get(id)
    }

    /// Every identity id this batch holds, in order.
    #[must_use]
    pub fn ids(&self) -> Vec<String> {
        self.runs.keys().cloned().collect()
    }

    /// Ids whose run hit an environment failure.
    #[must_use]
    pub fn missing_env(&self) -> Vec<String> {
        self.runs
            .iter()
            .filter(|(_, o)| matches!(o, RunOutcome::MissingEnv(_)))
            .map(|(id, _)| id.clone())
            .collect()
    }

    /// Build a batch from failure-packet episodes, which is how the harness
    /// feeds frozen runs in: the packet already carries per-episode task ids
    /// and scores.
    ///
    /// The `status` vocabulary decides the outcome: `"env"` marks a run whose
    /// environment failed (the episode's `transcript` carries the reason), and
    /// every other status carries its `score`.
    ///
    /// # Errors
    ///
    /// [`Error::Format`] when a scored episode's `score` is not finite — a NaN
    /// measure cannot be allowed into a mean, and a serde `null` from a
    /// corrupted number would quietly do exactly that.
    pub fn from_episodes(episodes: &[omni_core::HookEpisode]) -> Result<Self, Error> {
        let mut batch = Self::new();
        for (i, e) in episodes.iter().enumerate() {
            if e.status == "env" {
                batch.record(e.task_id.clone(), RunOutcome::MissingEnv(e.transcript.clone()));
                continue;
            }
            if !e.score.is_finite() {
                return Err(Error::Format(format!(
                    "episode {i} (task {:?}) carries a non-finite score; a corrupted \
                     measure cannot enter a delta, so the batch is refused",
                    e.task_id
                )));
            }
            batch.record(e.task_id.clone(), RunOutcome::Scored(e.score));
        }
        Ok(batch)
    }
}

// ── The reward ──────────────────────────────────────────────────────────────

/// What one comparison measured, and what it had to leave out to measure it.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Reward {
    /// Ids scored on both sides. Every number below is over exactly these.
    pub matched: usize,
    /// mean(patched) − mean(baseline) over the matched pairs, in arrival
    /// order of the baseline for stability.
    pub delta: f64,
    /// Ids with an environment failure on either side. The delta never
    /// includes them, and their *presence here* is the guarantee that they
    /// were excluded rather than treated as zeros.
    pub skipped_for_env: Vec<String>,
    /// Ids the baseline holds that the patch run never ran.
    pub unpaired_baseline: Vec<String>,
    /// Ids the patch run holds that the baseline never ran.
    pub unpaired_patched: Vec<String>,
}

impl Reward {
    /// The zero reward: what an invalid patch is worth.
    #[must_use]
    pub fn zero() -> Self {
        Self {
            matched: 0,
            delta: 0.0,
            skipped_for_env: Vec::new(),
            unpaired_baseline: Vec::new(),
            unpaired_patched: Vec::new(),
        }
    }

    /// Whether the delta was computed over anything at all. A reward with no
    /// matched pairs is not an improvement and not a regression — it is *not a
    /// measurement*, and a caller printing a sign for it is lying.
    #[must_use]
    pub fn is_empty_measurement(&self) -> bool {
        self.matched == 0
    }
}

/// Pair two batches by identity and score the difference.
///
/// Only ids present and **scored on both sides** enter the delta. Everything
/// else is named in its own bucket:
///
/// | situation | where it is reported | what it is *not* |
/// |---|---|---|
/// | scored on both | `matched`, into `delta` | — |
/// | environment failure on either side | `skipped_for_env` | never a zero |
/// | missing from the patch run | `unpaired_baseline` | never a zero |
/// | missing from the baseline | `unpaired_patched` | never a zero |
#[must_use]
pub fn reward(baseline: &Batch, patched: &Batch) -> Reward {
    // Walk the baseline's own order so the delta is a fact about the runs and
    // not about the map's iteration order, which callers may reason about in a
    // debugger.
    let mut matched_scores: Vec<(f64, f64)> = Vec::new();
    let mut skipped: Vec<String> = Vec::new();
    let mut unpaired_patched: BTreeSet<String> = patched.runs.keys().cloned().collect();

    for (id, base) in &baseline.runs {
        let Some(patch) = patched.runs.get(id) else {
            // Baseline holds an id the patch side never ran.
            continue;
        };
        unpaired_patched.remove(id);
        match (base, patch) {
            (RunOutcome::Scored(b), RunOutcome::Scored(p)) => {
                matched_scores.push((*b, *p));
            }
            (RunOutcome::MissingEnv(_), _) | (_, RunOutcome::MissingEnv(_)) => {
                skipped.push(id.clone());
            }
        }
    }

    let unpaired_baseline: Vec<String> = baseline
        .runs
        .keys()
        .filter(|id| !patched.runs.contains_key(*id))
        .cloned()
        .collect();

    // Means in a canonical summation order: the baseline's own key order, which
    // is what `matched_scores` was pushed in. A different traversal order could
    // produce a different last bit for the same matched set, and the reward is
    // exactly the place that must not happen.
    let n = matched_scores.len();
    let delta = if n == 0 {
        0.0
    } else {
        let b: f64 = matched_scores.iter().map(|(b, _)| b).sum::<f64>() / n as f64;
        let p: f64 = matched_scores.iter().map(|(_, p)| p).sum::<f64>() / n as f64;
        p - b
    };

    Reward {
        matched: n,
        delta,
        skipped_for_env: skipped,
        unpaired_baseline,
        unpaired_patched: unpaired_patched.into_iter().collect(),
    }
}

// ── The patch and its one-pass honesty check ────────────────────────────────

/// A patch as the harness engineer wrote it, and whether it is even a patch.
///
/// The grammar this crate can check is one thing only: a patch is one or more
/// `<patch>…</patch>` blocks carrying non-whitespace. Everything deeper —
/// parse, validate, execute in the sandbox — is [`crate`]'s neighbour
/// `omni-hook`'s job, and this check exists so nothing here can stand in for
/// it. An invalid state is *structural*, and structurally invalid patches are
/// worth a zero delta, never a missing evaluation, never a retry.
#[derive(Debug, Clone, PartialEq)]
pub struct Patch {
    text: String,
    state: PatchState,
}

#[derive(Debug, Clone, PartialEq)]
pub enum PatchState {
    /// Structurally a patch. The sandbox has not seen it yet.
    Structural,
    /// Not a patch at all, with the reason.
    Invalid(String),
}

impl Patch {
    /// Check the one thing this layer can check.
    #[must_use]
    pub fn parse(text: &str) -> Self {
        let open = text.matches("<patch>").count();
        let close = text.matches("</patch>").count();
        if open == 0 || close == 0 || open != close {
            return Self {
                text: text.to_string(),
                state: PatchState::Invalid(format!(
                    "expected one or more <patch>…</patch> blocks with balanced \
                     markers, found {open} open and {close} close"
                )),
            };
        }
        for block in text.split("</patch>").filter_map(|m| m.split_once("<patch>")) {
            let inner = block.1;
            if inner.trim().is_empty() {
                return Self {
                    text: text.to_string(),
                    state: PatchState::Invalid("empty <patch>…</patch> block".to_string()),
                };
            }
        }
        Self { text: text.to_string(), state: PatchState::Structural }
    }

    /// The text the engineer submitted, whole.
    #[must_use]
    pub fn text(&self) -> &str {
        &self.text
    }

    /// Whether this is structurally a patch. Not whether it is a *good* patch.
    #[must_use]
    pub fn is_structural(&self) -> bool {
        matches!(self.state, PatchState::Structural)
    }

    /// The invalid reason, when there is one.
    #[must_use]
    pub fn invalid_reason(&self) -> Option<&str> {
        match &self.state {
            PatchState::Invalid(r) => Some(r),
            PatchState::Structural => None,
        }
    }
}

/// The full step: pair, score, and let a structurally invalid patch be zero.
///
/// This is the one function a caller should treat as *the* rule-8 sentence in
/// executable form: an invalid patch is a no-patch and scores zero; an
/// environment failure is a missing evaluation and is reported separately.
#[must_use]
pub fn score_patch(baseline: &Batch, patched: Option<&Batch>, patch: &Patch) -> Reward {
    if !patch.is_structural() {
        return Reward::zero();
    }
    // A valid patch with nothing rerun is *not* an invalid patch: it is an
    // empty patched batch, and `reward` reports every baseline id as
    // unpaired — the missing-evaluation reporting, done honestly.
    reward(baseline, patched.unwrap_or(&Batch::new()))
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use omni_core::HookEpisode;

    fn batch(pairs: &[(&str, Result<f64, &str>)]) -> Batch {
        let mut b = Batch::new();
        for (id, r) in pairs {
            let outcome = match r {
                Ok(v) => RunOutcome::Scored(*v),
                Err(reason) => RunOutcome::MissingEnv((*reason).to_string()),
            };
            b.record(*id, outcome);
        }
        b
    }

    #[test]
    fn an_identity_is_content_derived_and_knobs_are_fingerprinted() -> Result<(), Error> {
        let a = Identity::new("scan 10.0.0.1 for open ports").id()?;
        let b = Identity::new("scan 10.0.0.1 for open ports").with_knob("COLI_KV_DTYPE", "f16").id()?;
        assert_ne!(a, b, "a knob that can change output changes the identity");

        let same = Identity::new("x").with_knob("A", "1").with_knob("B", "2").id()?;
        let same_other_order = Identity::new("x").with_knob("B", "2").with_knob("A", "1").id()?;
        assert_eq!(same, same_other_order, "insertion order cannot affect the fingerprint");

        let model = Identity::new("x").with_model("qwen3-4b").id()?;
        let no_model = Identity::new("x").id()?;
        assert_ne!(model, no_model, "the model identity is part of the fingerprint");

        let budget = Identity::new("x").with_max_tokens(64).id()?;
        assert_ne!(budget, no_model, "a pinned completion budget is part of the fingerprint");
        Ok(())
    }

    #[test]
    fn a_batch_built_from_episodes_matches_the_packet_vocabulary() -> Result<(), Error> {
        let episodes = vec![
            HookEpisode { task_id: "t1".into(), score: 0.5, status: "ok".into(), ..Default::default() },
            HookEpisode { task_id: "t2".into(), score: 7.25, status: "tool_error".into(), ..Default::default() },
            HookEpisode { task_id: "t3".into(), score: 0.0, status: "env".into(), transcript: "nfs flaked".into(), ..Default::default() },
        ];
        let b = Batch::from_episodes(&episodes)?;
        assert_eq!(b.len(), 3);
        assert_eq!(b.outcome_of("t2"), Some(&RunOutcome::Scored(7.25)));
        assert_eq!(b.outcome_of("t3"), Some(&RunOutcome::MissingEnv("nfs flaked".into())));
        assert_eq!(b.missing_env(), ["t3"]);
        Ok(())
    }

    #[test]
    fn a_non_finite_episode_score_is_refused_not_coerced() -> Result<(), Error> {
        // A NaN/HUGE score would enter a mean and corrupt the delta; the batch
        // refuses at the door instead. The value comes from the wire, which is
        // why it can be non-finite at all.
        let episodes = vec![HookEpisode {
            task_id: "t1".into(),
            score: f64::NAN,
            status: "ok".into(),
            turns: 1,
            transcript: String::new(),
        }];
        let r = Batch::from_episodes(&episodes);
        assert!(r.is_err(), "a non-finite score must be refused");
        Ok(())
    }

    #[test]
    fn the_delta_is_the_difference_of_matched_means() -> Result<(), Error> {
        let ids: Vec<String> = (1..=4usize)
            .map(|i| Identity::new(&format!("task-{i}")).id().unwrap_or_default())
            .collect();
        let baseline = batch(&[
            (&ids[0].clone(), Ok(1.0)),
            (&ids[1].clone(), Ok(2.0)),
            (&ids[2].clone(), Ok(4.0)),
            (&ids[3].clone(), Ok(4.0)),
        ]);
        // Same set of ids, different *arrival*: pairing is by content, so the
        // delta cannot care.
        let patched = batch(&[
            (&ids[3].clone(), Ok(3.0)),
            (&ids[0].clone(), Ok(1.5)),
            (&ids[2].clone(), Ok(2.0)),
            (&ids[1].clone(), Ok(4.0)),
        ]);
        let r = reward(&baseline, &patched);
        assert_eq!(r.matched, 4);
        // patched mean = (1.5 + 4 + 2 + 3)/4 = 2.625; baseline mean = 11/4 = 2.75
        assert!((r.delta - (2.625 - 2.75)).abs() < 1e-12, "delta = {}", r.delta);
        assert!(!r.is_empty_measurement());
        assert!(r.skipped_for_env.is_empty());
        assert!(r.unpaired_baseline.is_empty());
        assert!(r.unpaired_patched.is_empty());
        Ok(())
    }

    #[test]
    fn missing_and_unpaired_are_named_not_averaged() -> Result<(), Error> {
        let a = Identity::new("t-a").id()?;
        let b = Identity::new("t-b").id()?;
        let c = Identity::new("t-c").id()?;
        let baseline = batch(&[(&a.clone(), Ok(1.0)), (&b.clone(), Ok(2.0)), (&c.clone(), Ok(4.0))]);
        // The patch run: `b` hit an environment failure, `c` never ran, `d`
        // is a task the baseline never had.
        let d = Identity::new("t-d").id()?;
        let patched = batch(&[(&a.clone(), Ok(4.0)), (&b.clone(), Err("disk full")), (&d.clone(), Ok(9.0))]);

        let r = reward(&baseline, &patched);
        assert_eq!(r.matched, 1, "only t-a entered the delta");
        assert!((r.delta - 3.0).abs() < 1e-12, "delta over the one matched pair: {}", r.delta);
        assert_eq!(r.skipped_for_env, [b.as_str()], "the env failure is named, not zeroed");
        assert_eq!(r.unpaired_baseline, [c.as_str()], "a task the patch never ran is named");
        assert_eq!(r.unpaired_patched, [d.as_str()], "a task with no baseline is named");
        Ok(())
    }

    #[test]
    fn two_batches_sharing_no_ids_empty_the_measurement() -> Result<(), Error> {
        let a = Identity::new("x").id()?;
        let b = Identity::new("y").id()?;
        let r = reward(&batch(&[(&a.clone(), Ok(1.0))]), &batch(&[(&b.clone(), Ok(100.0))]));
        assert!(r.is_empty_measurement(), "nothing matched; there is nothing to sign");
        assert_eq!(r.delta, 0.0, "an unmatched delta is zero by shape, not by claim");
        assert_eq!(r.unpaired_baseline, [a]);
        assert_eq!(r.unpaired_patched, [b]);
        Ok(())
    }

    #[test]
    fn an_invalid_patch_scores_zero_and_is_never_a_missing_evaluation() -> Result<(), Error> {
        let a = Identity::new("t").id()?;
        let baseline = batch(&[(&a.clone(), Ok(1.0))]);

        let empty = Patch::parse("<patch>   </patch>");
        assert!(invalid_reason_present(&empty), "an all-whitespace block is not a patch");
        let unbalanced = Patch::parse("half a block <patch>");
        assert!(invalid_reason_present(&unbalanced));
        let absent = Patch::parse("just prose, no blocks at all");

        for p in [&empty, &unbalanced, &absent] {
            // With reruns available *and* without — the patch itself is the
            // discontinuity, so the treatment cannot depend on what was run.
            let r0 = score_patch(&baseline, None, p);
            let r1 = score_patch(&baseline, Some(&batch(&[(&a.clone(), Ok(2.0))])), p);
            assert_eq!(r0, r1);
            assert!(r0.is_empty_measurement());
            assert_eq!(r0.delta, 0.0);
            assert!(r0.unpaired_baseline.is_empty(), "not reported as unpaired reruns");
            assert!(r0.skipped_for_env.is_empty(), "not reported as environment failures");
        }

        let good = Patch::parse("attempt something\n<patch>\nshader work\n</patch>");
        assert!(good.is_structural(), "a real patch passes the structural gate");
        assert!(score_patch(&baseline, Some(&batch(&[(&a.clone(), Ok(2.0))])), &good).delta > 0.0);
        Ok(())
    }

    fn invalid_reason_present(p: &Patch) -> bool {
        !p.is_structural() && p.invalid_reason().is_some_and(|r| !r.is_empty())
    }

    #[test]
    fn a_valid_patch_with_no_reruns_is_missing_evaluations_not_a_zero() -> Result<(), Error> {
        // The distinction rule 8 exists to keep alive: an invalid patch IS a
        // zero; a valid patch whose reruns never happened is a set of named
        // missing evaluations.
        let a = Identity::new("t").id()?;
        let baseline = batch(&[(&a.clone(), Ok(1.0))]);
        let valid = Patch::parse("<patch>x</patch>");
        let r = score_patch(&baseline, None, &valid);
        assert!(valid.is_structural());
        assert!(r.is_empty_measurement());
        assert_eq!(r.unpaired_baseline, [a.as_str()], "the rerun is missing, and named");
        assert_eq!(r.delta, 0.0);
        Ok(())
    }

    #[test]
    fn a_batch_survives_a_round_trip_through_the_wire() -> Result<(), Error> {
        // A batch written to disk by one run and read back by the next must
        // pair with batches built in memory — the serialised form is a
        // storage format, not a second dialect.
        let id = Identity::new("probe").with_knob("COLI_DSA", "1").id()?;
        let mut b = Batch::new();
        b.record(id.clone(), RunOutcome::Scored(0.5));
        b.record(Identity::new("no env").id()?, RunOutcome::MissingEnv("disk full".into()));

        let json = serde_json::to_string(&b)?;
        let back: Batch = serde_json::from_str(&json)?;
        assert_eq!(back, b, "the round trip is the same batch");
        assert_eq!(back.missing_env(), b.missing_env());
        Ok(())
    }
}
