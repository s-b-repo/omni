//! The rollout stage of the harness-edit loop: run a frozen agent over a
//! frozen task set twice — once bare, once hosting a patch — and hand the two
//! batches to `omni-eval`'s identity-paired reward.
//!
//! This is the wiring that turns "the crates exist" into "a reward was
//! computed". It is also, stated plainly, **demonstration-grade**: the frozen
//! agent is a scripted playbook, not a live model, so the reward proves the
//! pipeline — identity pinning, the sandbox's verdicts, the paired delta —
//! and it does not yet prove that a patch improves a *real* agent. The
//! evidence-grade rollout needs the model seam wired into this loop, which is
//! the same blocker as the harness-engineer model. Every number here carries
//! its matched count; a reader who trusts the sign without the counts is
//! trusting the plumbing, not the agent.
//!
//! What a patch can and cannot do to this agent:
//!
//! - **Gate a trap** (`on_before_action` deny): the playbook's fallback path
//!   is taken instead, and the episode completes where the baseline stalled.
//! - **Abort a runaway** (`on_post_step` abort): the episode ends early with
//!   the turns it actually spent.
//! - **Guide in prose** (`make_pre_hint`): called and audited, but a scripted
//!   agent cannot read prose. Its behavioral effect needs a model that reads
//!   prompts — the harness-engineer-model blocker, named rather than faked.
//!
//! The agent is *frozen*: the playbooks are identical in both arms, and only
//! the harness differs. A reward that changed the agent would measure the
//! change, not the patch.

#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
#![forbid(unsafe_code)]

use omni_core::{Error, HookEpisode, HookPacket};
use omni_eval::{Batch, Identity, RunOutcome, Reward, reward};
use omni_model::Gate;
use omni_hook::{Flow, HookPatch, HookRuntime};

/// How many turns a stalled episode burns before it is called a loop. The
/// baseline arm's failure mode: the playbook's precondition never arrives,
/// the agent retries, the turn count climbs, the score does not.
pub const MAX_TURNS: usize = 6;

// ── the frozen target ───────────────────────────────────────────────────────

/// One step of a frozen playbook: the action the agent intends, the detour
/// it can take when the harness refuses the intent, and the precondition it
/// cannot itself satisfy.
#[derive(Debug, Clone)]
pub struct Step {
    /// The tool this step dispatches.
    pub tool: String,
    /// The step's partial credit when it completes — by any path.
    pub credit: f64,
    /// The tool to fall back to when the gate refuses `tool`. `None` means
    /// this step has no detour: refused here, the agent stalls, and a stalled
    /// agent burns turns until [`MAX_TURNS`] ends the episode as a loop.
    pub fallback: Option<String>,
    /// A precondition the agent cannot work around: the named tool must have
    /// *actually run* earlier in this episode. A step whose precondition is
    /// unmet stalls exactly like a refused step with no detour — the
    /// destructive-action trap is modelled here, in the frozen agent, where a
    /// real workflow bug lives, rather than invented by the harness.
    pub requires: Option<String>,
}

/// One task of the frozen target. `input` is the task's content — it feeds
/// the identity both arms are paired on, never a position.
#[derive(Debug, Clone)]
pub struct Task {
    pub input: String,
    pub steps: Vec<Step>,
}

/// What one episode produced.
#[derive(Debug, Clone)]
pub struct Episode {
    pub status: String,
    pub turns: usize,
    pub score: f64,
    pub transcript: String,
}

/// The scripted frozen agent. It executes its playbook exactly: one turn per
/// step, the fallback when the gate refuses the intent, a stall-and-loop when
/// there is no fallback and the gate refuses, and an early end when the
/// harness aborts the episode. It reads nothing else.
#[must_use]
pub fn run_task(task: &Task, hooks: &mut Option<&mut HookRuntime>, max_turns: usize) -> Episode {
    let mut transcript = String::new();
    let mut credit = 0.0f64;
    let mut turns = 0usize;
    let mut status = "done".to_string();
    let mut completed: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();

    'playbook: for (i, step) in task.steps.iter().enumerate() {
        if turns >= max_turns {
            status = "loop".to_string();
            break;
        }
        // A step whose precondition is unmet stalls exactly like a refused
        // step with no detour: the agent cannot fix it, it can only retry
        // until the turn budget ends the episode. The destructive-action trap
        // lives here, in the frozen agent, where a real workflow bug lives —
        // not invented by the harness.
        let unmet = match &step.requires {
            Some(required) if !completed.contains(required) => Some(required.clone()),
            _ => None,
        };
        if let Some(required) = unmet {
            while turns < max_turns {
                turns += 1;
                transcript
                    .push_str(&format!("[stall] {} waits on {required}\n", step.tool));
                // A retry is a turn: the harness observes it like any other,
                // and an abort rule fires mid-stall rather than after it.
                if let Some(h) = hooks.as_mut().as_mut() {
                    let outcome = HookEpisode {
                        task_id: task.input.clone(),
                        score: credit,
                        status: "loop".to_string(),
                        turns,
                        transcript: transcript.clone(),
                    };
                    if let Flow::Abort(reason) = h.on_post_step(&outcome) {
                        status = "aborted".to_string();
                        transcript.push_str(&format!("[aborted] {reason}\n"));
                        break 'playbook;
                    }
                }
            }
            status = "loop".to_string();
            break 'playbook;
        }
        // The gate verdict: the same call the dispatcher makes, dry-run or
        // not — the sandbox's Gate impl is pure, so there is one answer.
        let refused = hooks.as_ref().as_ref()
            .map(|h| !h.evaluate(&step.tool, &serde_json::json!({ "step": i })).allowed)
            .unwrap_or(false);
        turns += 1;
        if refused {
            match &step.fallback {
                Some(detour) => {
                    credit += step.credit;
                    completed.insert(detour.clone());
                    transcript.push_str(&format!(
                        "[detour] {} refused; took {detour}\n",
                        step.tool
                    ));
                }
                None => {
                    // No detour: the agent retries its intent until the turn
                    // budget ends the episode. This is the "loop" the failure
                    // packet exists to name. Each retry is a turn the harness
                    // observes, so an abort rule fires mid-stall.
                    while turns < max_turns {
                        turns += 1;
                        transcript.push_str(&format!("[retry] {}\n", step.tool));
                        if let Some(h) = hooks.as_mut().as_mut() {
                            let outcome = HookEpisode {
                                task_id: task.input.clone(),
                                score: credit,
                                status: "loop".to_string(),
                                turns,
                                transcript: transcript.clone(),
                            };
                            if let Flow::Abort(reason) = h.on_post_step(&outcome) {
                                status = "aborted".to_string();
                                transcript.push_str(&format!("[aborted] {reason}\n"));
                                break 'playbook;
                            }
                        }
                    }
                    status = "loop".to_string();
                    break 'playbook;
                }
            }
        } else {
            credit += step.credit;
            completed.insert(step.tool.clone());
            transcript.push_str(&format!("[ok] {}\n", step.tool));
        }
        // Post-step hooks may end the episode early.
        if let Some(h) = hooks.as_mut().as_mut() {
            let outcome = HookEpisode {
                task_id: task.input.clone(),
                score: credit,
                status: status.clone(),
                turns,
                transcript: transcript.clone(),
            };
            if let Flow::Abort(reason) = h.on_post_step(&outcome) {
                status = "aborted".to_string();
                transcript.push_str(&format!("[aborted] {reason}\n"));
                break 'playbook;
            }
        }
    }

    Episode {
        status,
        turns,
        score: credit,
        transcript,
    }
}

// ── the two arms ────────────────────────────────────────────────────────────

/// One arm of the comparison: every task, one episode each, scored into a
/// [`Batch`] keyed by content identity, with the episodes that feed the next
/// failure packet.
///
/// The identity pins the task content, the frozen agent's name and the turn
/// budget — the knobs that can change a run. The patch is the *treatment*, so
/// it must not appear in the identity: both arms must carry the same ids or
/// the pairing is a fiction.
pub fn rollout(
    tasks: &[Task],
    mut hooks: Option<&mut HookRuntime>,
) -> Result<(Batch, Vec<HookEpisode>), Error> {
    let mut batch = Batch::new();
    let mut episodes = Vec::new();
    for task in tasks {
        let episode = run_task(task, &mut hooks, MAX_TURNS);
        let id = Identity::new(&task.input)
            .with_model("frozen-scripted")
            .with_knob("max_turns", &MAX_TURNS.to_string())
            .id()?;
        batch.record(&id, RunOutcome::Scored(episode.score));
        episodes.push(HookEpisode {
            task_id: id,
            score: episode.score,
            status: episode.status,
            turns: episode.turns,
            transcript: episode.transcript,
        });
    }
    Ok((batch, episodes))
}

/// The verdict of one patch: whether it installed, why not, the patched
/// batch, and the paired reward against the baseline.
#[derive(Debug)]
pub struct PatchVerdict {
    /// False when the sandbox refused the patch. An uninstalled patch is a
    /// no-patch and scores zero — never an error at this layer, because
    /// "the engineer wrote a bad patch" is a measured outcome of the loop,
    /// the thing the reward exists to price.
    pub installed: bool,
    /// The refusal, named, when `installed` is false.
    pub refusal: Option<String>,
    pub reward: Reward,
    pub patched: Batch,
    pub episodes: Vec<HookEpisode>,
}

/// The full loop over one patch: baseline (already run), failure packet (the
/// evidence the engineer conditioned on), patch (hand-authored until the
/// engineer model exists), sandbox, rerun, paired reward.
///
/// # Errors
/// Only a malformed identity can error here; a malformed *patch* is a
/// measured zero, not an error.
pub fn evaluate_patch(
    baseline: (&Batch, &HookPacket),
    patch_text: &str,
    tasks: &[Task],
) -> Result<PatchVerdict, Error> {
    let (baseline_batch, _) = baseline;
    let patch = match HookPatch::install(patch_text) {
        Ok(p) => p,
        Err(e) => {
            return Ok(PatchVerdict {
                installed: false,
                refusal: Some(e.to_string()),
                reward: Reward::zero(),
                patched: Batch::new(),
                episodes: Vec::new(),
            });
        }
    };
    let mut hooks = HookRuntime::new(patch);
    // The packet the engineer conditioned on is what on_init and the pre-hint
    // rules read. Baseline signals are the patch's evidence, not its identity.
    hooks.on_init(baseline.1);
    let _pre_hint = hooks.make_pre_hint(baseline.1);
    let (patched, episodes) = rollout(tasks, Some(&mut hooks))?;
    let reward = reward(baseline_batch, &patched);
    Ok(PatchVerdict {
        installed: true,
        refusal: None,
        reward,
        patched,
        episodes,
    })
}
