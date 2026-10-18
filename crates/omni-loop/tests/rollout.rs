//! The first end-to-end pass through the harness-edit loop: a baseline
//! rollout, its failure packet, a hand-authored patch (the harness-engineer
//! model does not exist — every patch is hand-authored until it does), the
//! sandbox, the rerun, and `omni-eval`'s identity-paired reward.
//!
//! Demonstration-grade, stated plainly: the frozen agent is a scripted
//! playbook, so this proves the *pipeline* — identity pinning, sandbox
//! verdicts, paired deltas. It does not prove a patch improves a real agent.

use omni_core::{HookEpisode, HookPacket};
use omni_eval::reward;
use omni_loop::rollout::{PatchVerdict, Step, Task, evaluate_patch, rollout};

/// The frozen target: two tasks whose playbooks each contain one trap step.
/// Bare, the trap runs, the next step's precondition never arrives, and the
/// agent stalls to the turn budget — status `loop`, partial credit. This is
/// the failure the packet will name and the patch will gate.
fn tasks() -> Vec<Task> {
    vec![
        Task {
            input: "reconcile the ledger after the failed import".to_string(),
            steps: vec![
                step("read_state", 1.0, None, None),
                step("purge_cache", 1.0, Some("skip_cache_rebuild"), None),
                step("reconcile", 1.0, None, Some("skip_cache_rebuild")),
            ],
        },
        Task {
            input: "rebuild the index after the schema migration".to_string(),
            steps: vec![
                step("read_state", 1.0, None, None),
                step("purge_cache", 1.0, Some("skip_cache_rebuild"), None),
                step("reindex", 1.0, None, Some("skip_cache_rebuild")),
            ],
        },
    ]
}

/// A step with the same shape every playbook uses. The detour tool is the
/// one the patch's gate sends the agent to.
fn step(tool: &str, credit: f64, fallback: Option<&str>, requires: Option<&str>) -> Step {
    Step {
        tool: tool.to_string(),
        credit,
        fallback: fallback.map(str::to_string),
        requires: requires.map(str::to_string),
    }
}

/// The trap fires bare: no hooks, `purge_cache` runs, the final step stalls.
#[test]
fn the_baseline_loops_on_the_trap() -> Result<(), omni_core::Error> {
    let tasks = tasks();
    let (baseline, episodes) = rollout(&tasks, None)?;
    for e in &episodes {
        assert_eq!(e.status, "loop", "bare agent hits the trap: {}", e.transcript);
        assert_eq!(e.turns, 6);
        assert!((e.score - 2.0).abs() < 1e-9, "two steps of three-credit runs");
    }
    assert_eq!(baseline.len(), 2);
    Ok(())
}

#[test]
fn a_gate_patch_closes_the_trap_and_the_reward_prices_it() -> Result<(), omni_core::Error> {
    let tasks = tasks();
    let (baseline, baseline_episodes) = rollout(&tasks, None)?;
    let packet = HookPacket::new("pentest", "both tasks loop on purge_cache", baseline_episodes);

    // The hand-authored patch: every patch is hand-authored until the
    // harness-engineer model exists. It refuses the trap and names why.
    let patch = r#"{
        "version": 1, "name": "do-not-purge-before-reconcile",
        "rules": [
            {"at": "on_before_action", "when": {"kind": "status_equals", "status": "purge_cache"},
             "effect": {"kind": "deny", "reason": "purging the cache invalidated the reconciliation state in every failed episode; use the skip path"}},
            {"at": "on_init", "when": {"kind": "signal_at_least", "key": "status:loop", "at_least": 2},
             "effect": {"kind": "annotate", "note": "baseline looped on both tasks; gating the trap step"}}
        ]
    }"#;

    let verdict = evaluate_patch((&baseline, &packet), patch, &tasks)?;
    assert!(verdict.installed, "the patch must install: {:?}", verdict.refusal);
    assert!(verdict.reward.matched == 2, "both tasks pair by identity");
    assert!(
        verdict.reward.delta > 0.0,
        "gating the trap must price positive: delta = {}",
        verdict.reward.delta
    );
    for e in &verdict.episodes {
        assert_eq!(e.status, "done", "the detour completes: {}", e.transcript);
        assert!(e.transcript.contains("[detour] purge_cache refused"));
        assert!((e.score - 3.0).abs() < 1e-9);
        assert_eq!(e.turns, 3);
    }
    // The refusal was named in the reward's other half: the delta is the
    // whole 1.0 of recovered credit per task.
    assert!((verdict.reward.delta - 1.0).abs() < 1e-9);
    Ok(())
}

#[test]
fn an_invalid_patch_is_a_measured_zero_not_an_error() -> Result<(), omni_core::Error> {
    let tasks = tasks();
    let (baseline, baseline_episodes) = rollout(&tasks, None)?;
    let packet = HookPacket::new("pentest", "evidence", baseline_episodes);
    // A deny at a position that cannot carry it: refused at install, scored
    // as a zero by the loop — the "invalid patch is a no-patch" rule, priced.
    let broken = r#"{
        "version": 1, "name": "wrong-position",
        "rules": [
            {"at": "make_pre_hint", "when": {"kind": "always"},
             "effect": {"kind": "deny", "reason": "no"}}
        ]
    }"#;
    let verdict: PatchVerdict = evaluate_patch((&baseline, &packet), broken, &tasks)?;
    assert!(!verdict.installed);
    assert!(verdict
        .refusal
        .as_deref()
        .map(|r| r.contains("deny is not valid at make_pre_hint"))
        .unwrap_or(false));
    assert!(verdict.reward.is_empty_measurement());
    assert_eq!(verdict.reward.delta, 0.0);
    Ok(())
}

#[test]
fn an_abort_patch_ends_runaways_early_and_prices_the_trade() -> Result<(), omni_core::Error> {
    // A patch that aborts any looping episode after the fact: the episode
    // ends early (fewer turns burned) at the same partial credit — a cheaper
    // failure, priced as zero delta, which is itself the honest number.
    let tasks = tasks();
    let (baseline, baseline_episodes) = rollout(&tasks, None)?;
    let packet = HookPacket::new("pentest", "evidence", baseline_episodes);
    let patch = r#"{
        "version": 1, "name": "abort-runaways",
        "rules": [
            {"at": "on_post_step", "when": {"kind": "status_equals", "status": "loop"},
             "effect": {"kind": "abort", "reason": "the episode is looping; end it and report"}}
        ]
    }"#;
    let verdict = evaluate_patch((&baseline, &packet), patch, &tasks)?;
    assert!(verdict.installed);
    assert_eq!(verdict.reward.matched, 2);
    // The stall still stalls — same credit, fewer turns on the ones that
    // tripped the abort.
    let looped: Vec<&HookEpisode> = verdict
        .episodes
        .iter()
        .filter(|e| e.status == "aborted")
        .collect();
    assert!(!looped.is_empty(), "the abort must land");
    for e in &looped {
        assert!(e.turns < 6, "aborted early: {}", e.turns);
    }
    assert!((verdict.reward.delta).abs() < 1e-9, "same credit, cheaper failure");
    Ok(())
}

#[test]
fn the_reward_pairs_by_identity_across_reruns() -> Result<(), omni_core::Error> {
    // Two identical rollouts must produce two identical batches, and the
    // reward of running the comparison must be an exact zero — the control
    // that proves the pairing is real and not positional.
    let tasks = tasks();
    let (baseline_a, _) = rollout(&tasks, None)?;
    let (baseline_b, _) = rollout(&tasks, None)?;
    let r = reward(&baseline_a, &baseline_b);
    assert_eq!(r.matched, 2);
    assert!((r.delta).abs() < 1e-12);
    assert!(r.unpaired_baseline.is_empty());
    assert!(r.unpaired_patched.is_empty());
    assert!(r.skipped_for_env.is_empty());
    Ok(())
}
