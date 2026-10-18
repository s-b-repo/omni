//! End-to-end behaviour of the hook sandbox: what installs, what is refused
//! and by name, what the four lifecycle positions produce, and the two
//! properties the reward depends on — determinism across reruns and a gate
//! that answers dry-run exactly as it answers dispatch.

use omni_core::{Error, HookEpisode, HookPacket};
use omni_model::Gate;
use omni_hook::{Flow, HookPatch, HookRuntime, MAX_STATE_ENTRIES};

fn episode(status: &str, turns: usize, transcript: &str) -> HookEpisode {
    HookEpisode {
        task_id: format!("task-{status}-{turns}"),
        score: 0.0,
        status: status.to_string(),
        turns,
        transcript: transcript.to_string(),
    }
}

/// A packet whose deterministic signals no episode would naturally produce,
/// so signal matchers can be tested against known values.
fn packet_with(signals: &[(&str, u64)], episodes: Vec<HookEpisode>) -> HookPacket {
    let mut packet = HookPacket::new("pentest", "test packet", episodes);
    packet.signals = signals
        .iter()
        .map(|(k, v)| ((*k).to_string(), *v))
        .collect();
    packet.signals.sort();
    packet.signals.dedup();
    packet
}

fn expect_refusal(text: &str, needle: &str) -> Result<(), Error> {
    match HookPatch::install(text) {
        Ok(_) => Err(Error::Format(format!(
            "expected the patch to be refused with '{needle}', but it installed"
        ))),
        Err(e) => {
            let rendered = e.to_string();
            assert!(
                rendered.contains(needle),
                "refusal must name the failure: wanted '{needle}' in '{rendered}'"
            );
            Ok(())
        }
    }
}

#[test]
fn a_missing_version_is_refused_like_an_unknown_one() -> Result<(), Error> {
    expect_refusal(
        r#"{"name": "no-version", "rules": []}"#,
        "contract version 0",
    )
}

#[test]
fn an_unknown_version_is_refused_naming_both_numbers() -> Result<(), Error> {
    expect_refusal(
        r#"{"version": 99, "name": "from-the-future", "rules": []}"#,
        "contract version 99",
    )
}

#[test]
fn a_positionally_invalid_effect_is_refused_by_name() -> Result<(), Error> {
    expect_refusal(
        r#"{
            "version": 1,
            "name": "deny-at-wrong-position",
            "rules": [
                {"at": "make_pre_hint", "when": {"kind": "always"},
                 "effect": {"kind": "deny", "reason": "no"}}
            ]
        }"#,
        "effect deny is not valid at make_pre_hint",
    )
}

#[test]
fn an_episode_matcher_at_action_time_is_refused() -> Result<(), Error> {
    // `on_before_action` runs before any episode exists, so a turns matcher
    // there could only ever silently never fire. The sandbox refuses it
    // instead.
    expect_refusal(
        r#"{
            "version": 1,
            "name": "turns-at-action",
            "rules": [
                {"at": "on_before_action", "when": {"kind": "turns_at_least", "n": 3},
                 "effect": {"kind": "deny", "reason": "late"}}
            ]
        }"#,
        "matcher turns_at_least is not evaluable at on_before_action",
    )
}

#[test]
fn budget_exceedance_is_refused_at_install() -> Result<(), Error> {
    let hint = "x".repeat(2001);
    expect_refusal(
        &format!(
            r#"{{"version": 1, "name": "too-long",
                "rules": [{{"at": "make_pre_hint", "when": {{"kind": "always"}},
                           "effect": {{"kind": "inject_hint", "text": "{hint}"}}}}]}}"#
        ),
        "exceeds 2000 characters",
    )
}

#[test]
fn a_reasonless_refusal_is_refused() -> Result<(), Error> {
    expect_refusal(
        r#"{
            "version": 1, "name": "blank-deny",
            "rules": [
                {"at": "on_before_action", "when": {"kind": "always"},
                 "effect": {"kind": "deny", "reason": "   "}}
            ]
        }"#,
        "without a reason",
    )
}

#[test]
fn too_many_rules_is_refused() -> Result<(), Error> {
    let mut rules = String::new();
    for i in 0..65 {
        rules.push_str(&format!(
            r#"{{"at": "on_init", "when": {{"kind": "always"}},
                "effect": {{"kind": "annotate", "note": "rule {i}"}}}},"#
        ));
    }
    expect_refusal(
        &format!(
            r#"{{"version": 1, "name": "too-many", "rules": [{}]}}"#,
            rules.trim_end_matches(',')
        ),
        "exceeds the budget of 64",
    )
}

#[test]
fn a_combined_prehint_over_the_total_cap_is_refused() -> Result<(), Error> {
    // Three hints, each under its own 2,000-character cap, together over the
    // 4,000-character session cap: the per-rule budget alone would let a
    // patch hand the model a wall of text no one can read whole.
    let one = "a".repeat(1600);
    let two = "b".repeat(1600);
    let three = "c".repeat(1600);
    expect_refusal(
        &format!(
            r#"{{"version": 1, "name": "wall-of-text", "rules": [
                {{"at": "make_pre_hint", "when": {{"kind": "always"}},
                 "effect": {{"kind": "inject_hint", "text": "{one}"}}}},
                {{"at": "make_pre_hint", "when": {{"kind": "always"}},
                 "effect": {{"kind": "inject_hint", "text": "{two}"}}}},
                {{"at": "make_pre_hint", "when": {{"kind": "always"}},
                 "effect": {{"kind": "inject_hint", "text": "{three}"}}}}
            ]}}"#
        ),
        "combined pre-hint exceeds",
    )
}

#[test]
fn pre_hints_concatenate_in_declaration_order() -> Result<(), Error> {
    let patch = HookPatch::install(r#"{
        "version": 1, "name": "two-hints",
        "rules": [
            {"at": "make_pre_hint", "when": {"kind": "signal_at_least", "key": "loops", "at_least": 2},
             "effect": {"kind": "inject_hint", "text": "first"}},
            {"at": "make_pre_hint", "when": {"kind": "always"},
             "effect": {"kind": "inject_hint", "text": "second"}}
        ]
    }"#)?;
    let mut runtime = HookRuntime::new(patch);
    let quiet = packet_with(&[("loops", 1)], vec![episode("fail", 1, "nothing")]);
    assert_eq!(runtime.make_pre_hint(&quiet), Some("second".to_string()));
    let loud = packet_with(&[("loops", 3)], vec![episode("fail", 1, "nothing")]);
    assert_eq!(runtime.make_pre_hint(&loud), Some("first\n\nsecond".to_string()));
    Ok(())
}

#[test]
fn no_firing_rules_mean_no_hint() -> Result<(), Error> {
    let patch = HookPatch::install(r#"{
        "version": 1, "name": "quiet",
        "rules": [
            {"at": "make_pre_hint", "when": {"kind": "signal_at_least", "key": "missing", "at_least": 1},
             "effect": {"kind": "inject_hint", "text": "never"}}
        ]
    }"#)?;
    let mut runtime = HookRuntime::new(patch);
    let packet = packet_with(&[("loops", 1)], vec![episode("fail", 1, "nothing")]);
    assert_eq!(runtime.make_pre_hint(&packet), None);
    Ok(())
}

#[test]
fn first_matching_deny_wins_and_the_gate_answers_dry_run_like_dispatch() -> Result<(), Error> {
    let patch = HookPatch::install(r#"{
        "version": 1, "name": "gate",
        "rules": [
            {"at": "on_before_action", "when": {"kind": "transcript_contains", "text": "exec"},
             "effect": {"kind": "deny", "reason": "execution tools are out of scope"}},
            {"at": "on_before_action", "when": {"kind": "always"},
             "effect": {"kind": "deny", "reason": "blanket veto"}}
        ]
    }"#)?;
    let runtime = HookRuntime::new(patch);
    let first = runtime.evaluate("run_exec", &serde_json::json!({"cmd": "ls"}));
    assert!(!first.allowed);
    assert_eq!(first.reason, "execution tools are out of scope");
    let second = runtime.evaluate("read_file", &serde_json::json!({"path": "a"}));
    assert!(!second.allowed);
    assert_eq!(second.reason, "blanket veto");
    // Dry-runnable: the same question twice, one answer.
    let again = runtime.evaluate("run_exec", &serde_json::json!({"cmd": "ls"}));
    assert_eq!(first, again);
    // Purity: evaluating a gate verdict appends nothing to the audit trail.
    assert!(runtime.audit().is_empty());
    Ok(())
}

#[test]
fn post_step_can_remember_and_abort() -> Result<(), Error> {
    let patch = HookPatch::install(r#"{
        "version": 1, "name": "post",
        "rules": [
            {"at": "on_post_step", "when": {"kind": "status_equals", "status": "loop"},
             "effect": {"kind": "remember", "key": "seen_loop", "value": "yes"}},
            {"at": "on_post_step", "when": {"kind": "turns_at_least", "n": 8},
             "effect": {"kind": "abort", "reason": "episode budget exhausted"}}
        ]
    }"#)?;
    let mut runtime = HookRuntime::new(patch);
    let flow = runtime.on_post_step(&episode("loop", 9, "transcript"));
    assert_eq!(flow, Flow::Abort("episode budget exhausted".to_string()));
    assert_eq!(
        runtime.state().get("seen_loop").map(String::as_str),
        Some("yes")
    );
    // A short, successful episode continues.
    let flow = runtime.on_post_step(&episode("done", 2, "transcript"));
    assert_eq!(flow, Flow::Continue);
    Ok(())
}

#[test]
fn a_state_budget_refusal_is_visible_not_swallowed() -> Result<(), Error> {
    let mut rules = String::new();
    for i in 0..(MAX_STATE_ENTRIES + 1) {
        rules.push_str(&format!(
            r#"{{"at": "on_init", "when": {{"kind": "always"}},
                "effect": {{"kind": "remember", "key": "k{i}", "value": "v"}}}},"#
        ));
    }
    let text = format!(
        r#"{{"version": 1, "name": "filler", "rules": [{}]}}"#,
        rules.trim_end_matches(',')
    );
    let mut runtime = HookRuntime::new(HookPatch::install(&text)?);
    runtime.on_init(&packet_with(&[], vec![]));
    assert_eq!(runtime.state().len(), MAX_STATE_ENTRIES);
    let skipped: Vec<_> = runtime.audit().iter().filter(|e| e.skipped).collect();
    assert_eq!(skipped.len(), 1, "the refused write must be audited");
    assert!(
        skipped[0].detail.contains("state budget"),
        "the refusal names its budget: {}",
        skipped[0].detail
    );
    Ok(())
}

#[test]
fn the_same_patch_and_inputs_produce_the_same_trace() -> Result<(), Error> {
    // The reward pairs reruns by task identity and scores the difference.
    // A patch whose hosting is not deterministic would make that difference
    // measure noise. This is the property omni-eval leans on.
    let text = r#"{
        "version": 1, "name": "det",
        "rules": [
            {"at": "make_pre_hint", "when": {"kind": "signal_at_least", "key": "loops", "at_least": 2},
             "effect": {"kind": "inject_hint", "text": "hint"}},
            {"at": "on_post_step", "when": {"kind": "status_equals", "status": "fail"},
             "effect": {"kind": "remember", "key": "failed", "value": "once"}}
        ]
    }"#;
    let packet = packet_with(&[("loops", 4)], vec![episode("fail", 3, "transcript")]);

    let mut a = HookRuntime::new(HookPatch::install(text)?);
    a.on_init(&packet);
    let hint_a = a.make_pre_hint(&packet);
    let flow_a = a.on_post_step(&episode("fail", 3, "transcript"));

    let mut b = HookRuntime::new(HookPatch::install(text)?);
    b.on_init(&packet);
    let hint_b = b.make_pre_hint(&packet);
    let flow_b = b.on_post_step(&episode("fail", 3, "transcript"));

    assert_eq!(hint_a, hint_b);
    assert_eq!(flow_a, flow_b);
    assert_eq!(a.audit(), b.audit());
    assert_eq!(a.patch_id()?, b.patch_id()?);
    Ok(())
}

#[test]
fn on_init_seeds_state_from_packet_signals() -> Result<(), Error> {
    let patch = HookPatch::install(r#"{
        "version": 1, "name": "seed",
        "rules": [
            {"at": "on_init", "when": {"kind": "signal_at_least", "key": "repeated_refusals", "at_least": 3},
             "effect": {"kind": "remember", "key": "mode", "value": "de-escalate"}},
            {"at": "on_init", "when": {"kind": "signal_at_least", "key": "repeated_refusals", "at_least": 3},
             "effect": {"kind": "annotate", "note": "packet shows 3+ repeated refusals; de-escalation mode on"}}
        ]
    }"#)?;
    let mut runtime = HookRuntime::new(patch);
    runtime.on_init(&packet_with(
        &[("repeated_refusals", 4)],
        vec![episode("refused", 2, "t")],
    ));
    assert_eq!(
        runtime.state().get("mode").map(String::as_str),
        Some("de-escalate")
    );
    let annotations: Vec<_> = runtime
        .audit()
        .iter()
        .filter(|e| e.effect == "annotation")
        .collect();
    assert_eq!(annotations.len(), 1);
    assert!(annotations[0].detail.contains("de-escalation mode on"));
    Ok(())
}
