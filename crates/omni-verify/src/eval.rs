//! `eval` — the reward is computed over identities, and identity is content.
//!
//! The harness-improvement loop's most load-bearing number is `patched −
//! baseline` over paired reruns. Every scheme that ruins such a system does it
//! the same way: it pairs runs that are not the same task and averages them
//! anyway. This group pins rule 8 to residuals: pairing is by content-derived
//! identity, an environment failure and a missing rerun are named rather than
//! folded into zero, and a structurally invalid patch is a zero — not a
//! missing evaluation.

use omni_eval::{Batch, Identity, Patch, RunOutcome};

use crate::{Certificate, cert, exact, failed, with_cases};

/// Run the `eval` group.
#[must_use]
pub fn run() -> Vec<Certificate> {
    vec![
        pairing_is_by_identity_not_position(),
        a_knob_change_breaks_the_pair(),
        an_invalid_patch_scores_zero_not_missing(),
    ]
}

/// Whether the id-keyed reward said exactly what the fixture holds it to.
///
/// Side-effectually the same shape as `canonical`'s case loop: every deviation
/// prints its reason, and the boolean is what the certificate carries.
fn reward_matches(
    baseline: &Batch,
    patched: &Batch,
    expected_matched: usize,
    expected_delta: f64,
    skipped: &[&str],
    unpaired_baseline: &[&str],
    unpaired_patched: &[&str],
) -> bool {
    let r = omni_eval::reward(baseline, patched);
    let mut ok = r.matched == expected_matched;
    if !ok {
        eprintln!(
            "omni-verify: eval: paired {} runs but expected {}; reported delta {}",
            r.matched, expected_matched, r.delta
        );
    }
    // The delta bound: the expected value is computed the same way the code
    // computes it, so this is about bookkeeping and not about a rounding
    // cliff. A mispairing that changed the matched *set* moves the delta by
    // visibly more than this — that is why such a tight bound is honest here.
    if (r.delta - expected_delta).abs() > 1e-9 {
        ok = false;
        eprintln!(
            "omni-verify: eval: delta {} but expected {} over {} matched",
            r.delta, expected_delta, expected_matched
        );
    }
    for (got, want, what) in [
        (&r.skipped_for_env, skipped, "skipped_for_env"),
        (&r.unpaired_baseline, unpaired_baseline, "unpaired_baseline"),
        (&r.unpaired_patched, unpaired_patched, "unpaired_patched"),
    ] {
        let got: Vec<&str> = got.iter().map(|s| s.as_str()).collect();
        if got != want {
            ok = false;
            eprintln!("omni-verify: eval: {what} reports {got:?} but expected {want:?}");
        }
    }
    ok
}

/// Whether the forbidden scheme — zipping runs by arrival index — gets the
/// wrong answer on this fixture. The certificate's control: at least one
/// fixture where the schemes differ, else the check could pass under a
/// positional pairing and has certified nothing.
fn positional_index_reward(baseline: &Batch, patched: &Batch) -> (usize, f64) {
    let base_ids = baseline.ids();
    let patch_ids = patched.ids();
    let n = base_ids.len().min(patch_ids.len());
    let mut bs = 0.0;
    let mut ps = 0.0;
    let mut pairs = 0usize;
    for i in 0..n {
        if let (Some(RunOutcome::Scored(b)), Some(RunOutcome::Scored(p))) =
            (baseline.outcome_of(&base_ids[i]), patched.outcome_of(&patch_ids[i]))
        {
            bs += *b;
            ps += *p;
            pairs += 1;
        }
    }
    let delta = if pairs == 0 { 0.0 } else { ps / pairs as f64 - bs / pairs as f64 };
    (pairs, delta)
}

fn batch(pairs: &[(&str, RunOutcome)]) -> Batch {
    let mut b = Batch::new();
    for (id, o) in pairs {
        b.record(*id, o.clone());
    }
    b
}

fn pairing_is_by_identity_not_position() -> Certificate {
    // Four tasks, scored on both arms. The fixtures are the arrival orders a
    // real rerun produces: different order, partial coverage, a task with no
    // baseline, an environment failure mid-run.
    let baseline = batch(&[
        ("task-1", RunOutcome::Scored(1.0)),
        ("task-2", RunOutcome::Scored(2.0)),
        ("task-3", RunOutcome::Scored(3.0)),
        ("task-4", RunOutcome::Scored(4.0)),
    ]);

    struct Probe {
        label: &'static str,
        patched: Batch,
        matched: usize,
        delta: f64,
        skipped: &'static [&'static str],
        unpaired_baseline: &'static [&'static str],
        unpaired_patched: &'static [&'static str],
        positional_must_differ: bool,
    }

    let probes = [
        // Same ids, arrived out of order. The id-keyed delta is unchanged,
        // which is the property a positional scheme routinely breaks.
        Probe {
            label: "same ids, shuffled arrival",
            patched: batch(&[
                ("task-4", RunOutcome::Scored(5.0)),
                ("task-1", RunOutcome::Scored(1.5)),
                ("task-2", RunOutcome::Scored(4.0)),
                ("task-3", RunOutcome::Scored(2.0)),
            ]),
            matched: 4,
            // patched 12.5/4 = 3.125, baseline 10/4 = 2.5
            delta: 0.625,
            skipped: &[],
            unpaired_baseline: &[],
            unpaired_patched: &[],
            positional_must_differ: false,
        },
        // The patch rerun covered two of four tasks, one of which the
        // baseline never ran. A positional scheme zips task-1 onto task-2
        // and task-3 onto task-5 and calls those pairs.
        Probe {
            label: "partial coverage, one new task",
            patched: batch(&[
                ("task-2", RunOutcome::Scored(5.0)),
                ("task-5", RunOutcome::Scored(9.0)),
            ]),
            matched: 1,
            delta: 3.0,
            skipped: &[],
            unpaired_baseline: &["task-1", "task-3", "task-4"],
            unpaired_patched: &["task-5"],
            positional_must_differ: true,
        },
        // An environment failure on the patch side takes the pair out and
        // *says so*, instead of contributing a fabricated 0 − 2. The patch
        // rerun here never reached task-3/task-4, so those are named as
        // missing from the patch arm on top of it.
        Probe {
            label: "an environment failure is named",
            patched: batch(&[
                ("task-1", RunOutcome::Scored(1.0)),
                ("task-2", RunOutcome::MissingEnv("disk full".into())),
            ]),
            matched: 1,
            delta: 0.0,
            skipped: &["task-2"],
            unpaired_baseline: &["task-3", "task-4"],
            unpaired_patched: &[],
            // The arrival orders happen to agree here — the control does not
            // need every fixture to differ, only that at least one does.
            positional_must_differ: false,
        },
    ];

    let mut correct = 0usize;
    let mut differing = 0usize;
    for p in &probes {
        if reward_matches(&baseline, &p.patched, p.matched, p.delta, p.skipped, p.unpaired_baseline, p.unpaired_patched) {
            correct += 1;
        }
        let (pos_pairs, pos_delta) = positional_index_reward(&baseline, &p.patched);
        let id = omni_eval::reward(&baseline, &p.patched);
        let differs = pos_pairs != id.matched || (pos_delta - id.delta).abs() > 1e-9;
        if differs {
            differing += 1;
        } else if p.positional_must_differ {
            eprintln!("omni-verify: eval: {}: the positional control did not differ; the check was vacuous", p.label);
        }
    }

    let cases = probes.len();
    with_cases(
        cert(
            "eval",
            "pairing_is_by_identity_not_position",
            "rerun pairs are matched by content-derived identity, so arrival \
             order cannot change the reward and a positional index-zip cannot \
             pass for pairing where the compared sets differ",
            exact(correct == cases && differing >= 1),
            0.0,
        ),
        cases,
    )
}

fn a_knob_change_breaks_the_pair() -> Certificate {
    match a_knob_body() {
        Ok(c) => c,
        Err(e) => fixture_refused("a_knob_change_breaks_the_pair", &e),
    }
}

fn a_knob_body() -> Result<Certificate, omni_core::Error> {
    // Rule 8: every knob that can change a run's output belongs in the
    // fingerprint. The failure the fixtures hold: two runs of the same task
    // between which the engineer changed an environment variable, averaged
    // into a "pair" by a scheme with no fingerprint.
    let task = "scan target for exposed ports";
    let plain = Identity::new(task).id()?;
    let knob_only = Identity::new(task).with_knob("COLI_KV_DTYPE", "f16").id()?;
    let model_only = Identity::new(task).with_model("qwen3-4b").id()?;
    let both = Identity::new(task).with_knob("COLI_KV_DTYPE", "f16").with_model("qwen3-4b").id()?;
    // The same configuration, assembled in the other knob order: content is
    // what hashes, not construction order.
    let both_reordered = Identity::new(task).with_model("qwen3-4b").with_knob("COLI_KV_DTYPE", "f16").id()?;

    let mut arm_one = Batch::new();
    arm_one.record(plain.clone(), RunOutcome::Scored(1.0));
    let mut arm_two = Batch::new();
    arm_two.record(both.clone(), RunOutcome::Scored(4.0));
    let r = omni_eval::reward(&arm_one, &arm_two);

    let cases: [(&str, bool); 2] = [
        // Distinct fingerprint fields each carry weight independently, and
        // assembly order cannot: content is the identity, not construction.
        (
            "each output-changing field moves the identity, assembly order cannot",
            plain != knob_only
                && plain != model_only
                && knob_only != model_only
                && knob_only != both
                && model_only != both
                && both == both_reordered,
        ),
        // The reward cannot pair what the identity separates: matched 0,
        // delta 0, both sides named, no fabricated measure from either.
        (
            "the reward reports an identity mismatch as unpaired, not as a measure",
            r.is_empty_measurement()
                && r.delta == 0.0
                && r.unpaired_baseline == [plain.clone()]
                && r.unpaired_patched == [both.clone()],
        ),
    ];
    let ok = cases.iter().all(|(_, v)| *v);
    if !ok {
        for (what, v) in &cases {
            if !v {
                eprintln!("omni-verify: eval: knob identity: {what} did not hold");
            }
        }
    }
    Ok(with_cases(
        cert(
            "eval",
            "a_knob_change_breaks_the_pair",
            "two runs differing in any output-changing knob are different \
             identities, and the reward reports them unpaired rather than \
             averaging them as matched",
            exact(ok),
            0.0,
        ),
        cases.len(),
    ))
}

fn fixture_refused(name: &'static str, e: &omni_core::Error) -> Certificate {
    // A fixture that cannot hash is not a skip: the certificate fails with
    // the reason, which is how this suite always treats a check that could
    // not run.
    with_cases(
        cert(name, name, "the fixture could not hash, so the claim could not be evaluated", failed(name, &e.to_string()), 0.0),
        0,
    )
}

fn an_invalid_patch_scores_zero_not_missing() -> Certificate {
    let baseline = batch(&[("task-1", RunOutcome::Scored(1.0))]);
    // Reruns that exist and point clearly up: an invalid patch must produce
    // zero *even though good news was on the table*.
    let patched = batch(&[("task-1", RunOutcome::Scored(8.0))]);

    let invalids = [
        Patch::parse("<patch>   </patch>"),
        Patch::parse("half a block <patch>"),
        Patch::parse("plain prose, no markers at all"),
    ];
    let valid = Patch::parse("<patch>\nshader work\n</patch>");

    let mut cases = 0usize;
    let mut ok = true;
    for p in &invalids {
        // With reruns and without: the patch's own structural verdict is the
        // discontinuity, so the treatment must not depend on what was run.
        let r0 = omni_eval::score_patch(&baseline, None, p);
        let r1 = omni_eval::score_patch(&baseline, Some(&patched), p);
        let reason = p.invalid_reason().unwrap_or("<no reason>");
        cases += 1;
        if !(!p.is_structural() && r0 == omni_eval::Reward::zero() && r0 == r1) {
            ok = false;
            eprintln!("omni-verify: eval: invalid patch {reason:?} did not score an exact zero on both calls");
        }
    }

    // The controls, both in the other direction. A structurally valid patch
    // is NOT a zero-by-decree: with reruns it is a real measurement, and
    // without them it is the named missing evaluations. If this distinction
    // ever collapses, rule 8 has collapsed with it.
    let measured = omni_eval::score_patch(&baseline, Some(&patched), &valid);
    cases += 1;
    if !(valid.is_structural() && measured.matched == 1 && (measured.delta - 7.0).abs() < 1e-12) {
        ok = false;
        eprintln!("omni-verify: eval: a valid patch with reruns must be a real measurement, got matched {} delta {}", measured.matched, measured.delta);
    }

    let missing = omni_eval::score_patch(&baseline, None, &valid);
    cases += 1;
    if !(
        valid.is_structural()
            && missing.matched == 0
            && missing.delta == 0.0
            && missing.unpaired_baseline == ["task-1".to_string()]
            && missing.skipped_for_env.is_empty()
    ) {
        ok = false;
        eprintln!("omni-verify: eval: a valid patch with no reruns must report the missing evaluations, not a bare zero");
    }

    with_cases(
        cert(
            "eval",
            "an_invalid_patch_scores_zero_not_missing",
            "a structurally invalid patch is worth an exact zero whether or \
             not reruns exist, while a valid patch with no reruns is a set of \
             named missing evaluations — the two are never conflated",
            exact(ok),
            0.0,
        ),
        cases,
    )
}

#[cfg(test)]
mod tmp_dbg {
    use super::*;
    #[test]
    fn dbg_probes() {
        let baseline = batch(&[("task-1", RunOutcome::Scored(1.0)), ("task-2", RunOutcome::Scored(2.0)), ("task-3", RunOutcome::Scored(3.0)), ("task-4", RunOutcome::Scored(4.0))]);
        for (label, p, e) in [
            ("probe2: unpaired_baseline", batch(&[("task-2", RunOutcome::Scored(5.0)), ("task-5", RunOutcome::Scored(9.0))]), vec!["task-1", "task-3", "task-4"]),
            ("probe1", batch(&[("task-4", RunOutcome::Scored(5.0)), ("task-1", RunOutcome::Scored(1.5)), ("task-2", RunOutcome::Scored(4.0)), ("task-3", RunOutcome::Scored(2.0))]), vec![]),
        ] {
            let r = omni_eval::reward(&baseline, &p);
            eprintln!("{label}: matched={} delta={} ub={:?} up={:?} skip={:?} (want ub={e:?})", r.matched, r.delta, r.unpaired_baseline, r.unpaired_patched, r.skipped_for_env);
        }
    }
}
