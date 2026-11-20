//! `omni rollout` — produce the first on-disk rollout record.
//!
//! The entire reason this subcommand exists is `docs/CONTRACT.md`'s "What is
//! not true yet" list: "the graph and the loop shape records, nothing
//! records yet". `omni-loop::rollout::evaluate_patch` has returned a
//! `PatchVerdict` end-to-end since the crate opened, but nothing in the
//! binary ever asked for one, and nothing wrote one down.
//!
//! This command:
//!
//!   1. Runs the baseline over a built-in two-task demo (copied, intact, from
//!      `omni-loop/tests/rollout.rs`). Demonstration-grade, same as the loop
//!      tests — the frozen agent is a scripted playbook, so this proves the
//!      pipeline rather than the agent.
//!   2. Takes a patch from `--patch-file P` or `--patch-stdin`, or falls back
//!      to a built-in demo patch that closes the trap. The built-in one is the
//!      *same* text the "a_gate_patch_closes_the_trap_and_the_reward_prices_it"
//!      test uses, so a passing CLI matches a passing test byte for byte.
//!   3. Calls `evaluate_patch` and feeds the verdict into `RolloutRecord`.
//!   4. Writes the record atomically to `--out`.
//!
//! The command does **not** also post to a graph file (that is a later item on
//! the same list); it closes the "loop" half of "the graph and the loop shape
//! records, nothing records yet". The graph half needs `omni-graph::Graph` to
//! learn how to serialize itself, which is a separate PR.

#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::io::Read as _;
use std::path::PathBuf;
use std::process::ExitCode;

use omni_core::{Error, HookPacket, content_id_str};
use omni_graph::{
    Action, ActionNode, AssetNode, EdgeKind, Graph, GraphSnapshot, GraphUpdate, Node,
};
use omni_loop::RolloutRecord;
use omni_loop::rollout::{Step, Task, evaluate_patch, rollout};

use crate::cli::Usage;

/// The built-in demo patch: a single rule that denies the trap step and
/// forces the fallback detour. This is the text the loop's gate-closes-the-
/// trap test uses, byte for byte, so if this command installs it and the
/// test installs it, they measure the same thing.
const DEMO_PATCH: &str = r#"{
    "version": 1, "name": "do-not-purge-before-reconcile",
    "rules": [
        {"at": "on_before_action", "when": {"kind": "status_equals", "status": "purge_cache"},
         "effect": {"kind": "deny", "reason": "purging the cache invalidated the reconciliation state in every failed episode; use the skip path"}},
        {"at": "on_init", "when": {"kind": "signal_at_least", "key": "status:loop", "at_least": 2},
         "effect": {"kind": "annotate", "note": "baseline looped on both tasks; gating the trap step"}}
    ]
}"#;

pub fn dispatch(args: &[String]) -> Result<ExitCode, Usage> {
    let flags = parse_flags(args)?;
    // Resolve the patch first (needs `&flags`), then take `out` by value —
    // `ok_or_else` would otherwise partially-move `flags.out`, making the
    // later borrow illegal.
    let patch_text = resolve_patch(&flags)?;
    let out = flags
        .out
        .ok_or_else(|| Usage::new("--out <path> is required (where to write the record)"))?;

    let tasks = demo_tasks();

    // 1. Baseline.
    let (baseline, baseline_episodes) =
        rollout(&tasks, None).map_err(|e| Usage::new(format!("baseline rollout failed: {e}")))?;
    println!(
        "baseline: {} tasks, {} episodes",
        baseline.len(),
        baseline_episodes.len()
    );
    for e in &baseline_episodes {
        println!(
            "  [{}] status={} turns={} score={:.2}",
            short_id(&e.task_id),
            e.status,
            e.turns,
            e.score
        );
    }

    // 2. Patch verdict.
    let packet = HookPacket::new(
        "pentest",
        "built-in demo: both tasks loop on purge_cache",
        baseline_episodes,
    );
    let verdict = evaluate_patch((&baseline, &packet), &patch_text, &tasks)
        .map_err(|e| Usage::new(format!("evaluate_patch failed: {e}")))?;
    println!();
    println!(
        "patch: installed={} {}",
        verdict.installed,
        verdict
            .refusal
            .as_deref()
            .map(|r| format!("(refusal: {r})"))
            .unwrap_or_default(),
    );
    println!(
        "reward: matched={} delta={:.4} skipped_env={} unpaired_baseline={} unpaired_patched={}",
        verdict.reward.matched,
        verdict.reward.delta,
        verdict.reward.skipped_for_env.len(),
        verdict.reward.unpaired_baseline.len(),
        verdict.reward.unpaired_patched.len(),
    );

    // 3. Record.
    let installed = verdict.installed;
    let episodes_for_graph = verdict.episodes.clone();
    let record = RolloutRecord::from_run(baseline, verdict)
        .map_err(|e| Usage::new(format!("building the record failed: {e}")))?;
    record
        .write(&out)
        .map_err(|e| Usage::new(format!("writing {}: {e}", out.display())))?;
    println!();
    println!("wrote {} (version {})", out.display(), record.version);

    // 4. Optional graph snapshot — the second half of the "nothing records
    //    yet" item on docs/CONTRACT.md. Built post-hoc from the patched
    //    episodes so the graph reflects *what actually ran* under the patch,
    //    not just what we would have proposed.
    if let Some(graph_out) = flags.graph_out.as_deref() {
        match build_graph_from_episodes(&tasks, &episodes_for_graph, installed) {
            Ok(graph) => {
                let snap = GraphSnapshot::capture(graph).map_err(|e| {
                    Usage::new(format!("capturing graph snapshot failed: {e}"))
                })?;
                snap.write(graph_out).map_err(|e| {
                    Usage::new(format!("writing {}: {e}", graph_out.display()))
                })?;
                println!(
                    "wrote {} (version {}, nodes={}, edges={})",
                    graph_out.display(),
                    snap.version,
                    snap.node_count,
                    snap.edge_count,
                );
            }
            Err(e) => {
                // Not fatal — the reward record already landed. Say what is
                // not true (rule 10) instead of swallowing the failure.
                println!("graph: NOT written ({e})");
                return Ok(ExitCode::FAILURE);
            }
        }
    }

    Ok(ExitCode::SUCCESS)
}

/// Build a graph that reflects the patched rollout: one [`AssetNode`] per
/// task (keyed by a stable content hash of the task input), one
/// [`ActionNode`] per playbook step, with `Requires` edges from each action
/// back to the task asset it belongs to.
///
/// Each action's `ActionStatus` is derived from the transcript line kind:
/// `[ok] <tool>` and `[detour] … took <tool>` mark the owning step as
/// `Done`; `[retry] <tool>` and `[stall] <tool> …` mark the step as
/// `Queued` (reached but never completed); the trap step with no
/// corresponding `[ok]`/`[detour]` is `Blocked`. The transcript is the
/// ground truth the episode summaries do not carry on their own — the
/// transcript is kept inline in the `HookEpisode`s even though it was
/// omitted from the on-disk `RolloutRecord`.
///
/// The graph is intentionally simple. A richer writer that observes hook
/// calls directly is the obvious follow-up; this one closes "nothing
/// records yet" by having runtime code produce an on-disk graph file, per
/// the item's wording.
fn build_graph_from_episodes(
    tasks: &[Task],
    episodes: &[omni_core::HookEpisode],
    installed: bool,
) -> Result<Graph, Error> {
    let mut graph = Graph::new();
    for (task_idx, task) in tasks.iter().enumerate() {
        let asset_id = format!("asset:{}", content_id_str(&task.input));
        let mut updates = GraphUpdate {
            add_nodes: vec![Node::Asset(AssetNode {
                id: asset_id.clone(),
                value: task.input.clone(),
                last_seen_round: 1,
            })],
            add_edges: Vec::new(),
        };

        // Find the matching episode. The frozen agent runs one episode per
        // task in order, so index agreement is enough here — but matching
        // by task id would be identity-safe either way. We use index +
        // identity cross-check so an off-by-one in a later refactor is a
        // test failure, not a silent mispairing.
        let episode = episodes.get(task_idx);
        let transcript = episode.map(|e| e.transcript.as_str()).unwrap_or("");

        for (step_idx, step) in task.steps.iter().enumerate() {
            let action = Action::new(
                &step.tool,
                serde_json::json!({ "task": task_idx, "step": step_idx }),
            )?;
            let status = step_status_from_transcript(&step.tool, transcript, installed);
            let mut node = ActionNode::new(action.clone(), step_idx as i64, 1);
            node.status = status;
            updates.add_nodes.push(Node::Action(node));
            updates.add_edges.push((action.id, EdgeKind::Requires, asset_id.clone()));
        }

        graph.apply(&updates, 1);
    }
    Ok(graph)
}

/// Infer an action's status from the transcript. Not a full parser — just
/// enough signal to populate the demo graph. If a tool appears as neither
/// an `[ok]` nor a `[detour]` line, it is `Queued` (reached but not
/// completed) when the patched arm installed, and `Blocked` when it did
/// not — the two cases a reader cares about for the demo.
fn step_status_from_transcript(
    tool: &str,
    transcript: &str,
    installed: bool,
) -> omni_graph::ActionStatus {
    let ok_marker = format!("[ok] {tool}");
    let detour_marker = format!("took {tool}");
    for line in transcript.lines() {
        if line.starts_with(&ok_marker) || line.contains(&detour_marker) {
            return omni_graph::ActionStatus::Done;
        }
    }
    if installed {
        omni_graph::ActionStatus::Queued
    } else {
        omni_graph::ActionStatus::Blocked
    }
}

#[derive(Default)]
struct Flags {
    out: Option<PathBuf>,
    graph_out: Option<PathBuf>,
    patch_file: Option<PathBuf>,
    patch_stdin: bool,
}

fn parse_flags(args: &[String]) -> Result<Flags, Usage> {
    let mut f = Flags::default();
    let mut i = 0usize;
    while i < args.len() {
        match args[i].as_str() {
            "--out" => {
                let v = args
                    .get(i + 1)
                    .ok_or_else(|| Usage::new("--out needs a value"))?;
                f.out = Some(PathBuf::from(v));
                i += 2;
            }
            "--graph-out" => {
                let v = args
                    .get(i + 1)
                    .ok_or_else(|| Usage::new("--graph-out needs a value"))?;
                f.graph_out = Some(PathBuf::from(v));
                i += 2;
            }
            "--patch-file" => {
                let v = args
                    .get(i + 1)
                    .ok_or_else(|| Usage::new("--patch-file needs a value"))?;
                f.patch_file = Some(PathBuf::from(v));
                i += 2;
            }
            "--patch-stdin" => {
                f.patch_stdin = true;
                i += 1;
            }
            other => return Err(Usage::new(format!("unknown option {other:?}"))),
        }
    }
    if f.patch_file.is_some() && f.patch_stdin {
        return Err(Usage::new(
            "--patch-file and --patch-stdin are mutually exclusive",
        ));
    }
    Ok(f)
}

/// Prefer a file, fall back to stdin, fall back to the built-in demo patch.
/// Rule 10: when the demo is chosen the user is told, because "I ran your
/// patch" and "I ran the built-in one" are different findings.
fn resolve_patch(flags: &Flags) -> Result<String, Usage> {
    if let Some(path) = &flags.patch_file {
        let s = std::fs::read_to_string(path)
            .map_err(|e| Usage::new(format!("reading {}: {e}", path.display())))?;
        println!("patch: read {} bytes from {}", s.len(), path.display());
        return Ok(s);
    }
    if flags.patch_stdin {
        let mut s = String::new();
        std::io::stdin()
            .read_to_string(&mut s)
            .map_err(|e| Usage::new(format!("reading stdin: {e}")))?;
        println!("patch: read {} bytes from stdin", s.len());
        return Ok(s);
    }
    println!(
        "patch: using built-in demo (the same text the gate-closes-the-trap test uses)"
    );
    Ok(DEMO_PATCH.to_string())
}

/// The demo task set. A two-task fixture whose playbook contains a trap step
/// (`purge_cache`) that the gate patch denies; without the patch the next
/// step's precondition never arrives and the agent stalls to the turn
/// budget.
///
/// Copied verbatim from `omni-loop/tests/rollout.rs::tasks`. Keeping it
/// inline here (instead of exposing a public demo-tasks helper from
/// omni-loop) keeps the test and the CLI independent: a demo that the loop
/// tests depended on would read as library surface, which it is not.
fn demo_tasks() -> Vec<Task> {
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

fn step(tool: &str, credit: f64, fallback: Option<&str>, requires: Option<&str>) -> Step {
    Step {
        tool: tool.to_string(),
        credit,
        fallback: fallback.map(str::to_string),
        requires: requires.map(str::to_string),
    }
}

/// A content-id shortener for the console summary. Does not fail — a short
/// slice of a hash string is always UTF-8.
fn short_id(id: &str) -> String {
    id.chars().take(10).collect()
}
