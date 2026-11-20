//! `records` — the on-disk record types round-trip through disk unchanged,
//! and the readers refuse a file whose version is absent, wrong, or whose
//! mirror counts disagree with the body.
//!
//! Why this group exists even though the shipping crates already have
//! unit tests for the same claims:
//!
//! AGENTS.md rule 6: "a load-bearing claim reducible to a residual is
//! not a certificate" unless it is certified independently. Anything that
//! resumes a rollout or replays a frontier reads these two files, so a
//! silent shape drift (e.g. a serde rename, a dropped mirror check) is
//! the exact failure mode the gate exists to catch. The unit tests live
//! in the shipping crate and could be deleted in the same PR that
//! introduces the drift. The certificate lives here, is run by the
//! separate `scripts/verify-certificates.sh` gate, and so the two paths
//! would have to be broken in coordination for the drift to pass.
//!
//! Each certificate is a `match` over the expected outcome: `0.0` when
//! the claim holds, `f64::INFINITY` (via [`failed`]) when a precondition
//! did not run, and `1.0` (via [`exact(false)`]) when the claim was
//! evaluated and did not hold. A widening tolerance would absorb a real
//! break, so every residual here is exact.

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use omni_core::{HookEpisode, write_atomic};
use omni_eval::{Batch, Reward, RunOutcome};
use omni_graph::{Action, ActionNode, EdgeKind, Graph, GraphSnapshot, GraphUpdate, Node};
use omni_loop::RolloutRecord;
use omni_loop::rollout::PatchVerdict;

use crate::{Certificate, cert, exact, failed};

/// Run the `records` group.
#[must_use]
pub fn run() -> Vec<Certificate> {
    vec![
        rollout_record_round_trips(),
        rollout_record_missing_version_is_refused(),
        rollout_record_wrong_version_is_refused(),
        graph_snapshot_round_trips(),
        graph_snapshot_missing_version_is_refused(),
        graph_snapshot_tampered_mirror_is_refused(),
    ]
}

// ---- scratch dir plumbing, mirroring the atomic group's -----------------

static SCRATCH_SEQ: AtomicU64 = AtomicU64::new(0);

fn scratch(name: &str) -> PathBuf {
    let seq = SCRATCH_SEQ.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!(
        "omni-verify-records-{}-{name}-{seq}",
        std::process::id()
    ));
    if let Err(e) = std::fs::remove_dir_all(&dir) {
        if e.kind() != std::io::ErrorKind::NotFound {
            omni_core::note_advisory(
                &format!("clearing scratch {}", dir.display()),
                &omni_core::Error::from(e),
            );
        }
    }
    if let Err(e) = std::fs::create_dir_all(&dir) {
        omni_core::note_advisory(
            &format!("creating scratch {}", dir.display()),
            &omni_core::Error::from(e),
        );
    }
    dir
}

// ---- fixtures -----------------------------------------------------------

fn sample_verdict() -> PatchVerdict {
    let mut patched = Batch::new();
    patched.record("id-a", RunOutcome::Scored(2.0));
    patched.record("id-b", RunOutcome::Scored(3.0));
    PatchVerdict {
        installed: true,
        refusal: None,
        reward: Reward {
            matched: 2,
            delta: 0.5,
            skipped_for_env: Vec::new(),
            unpaired_baseline: Vec::new(),
            unpaired_patched: Vec::new(),
        },
        patched,
        episodes: vec![
            HookEpisode {
                task_id: "id-a".to_string(),
                score: 2.0,
                status: "done".to_string(),
                turns: 3,
                transcript: "[ok] reconcile".to_string(),
            },
            HookEpisode {
                task_id: "id-b".to_string(),
                score: 3.0,
                status: "done".to_string(),
                turns: 3,
                transcript: "[ok] reindex".to_string(),
            },
        ],
    }
}

fn sample_baseline() -> Batch {
    let mut b = Batch::new();
    b.record("id-a", RunOutcome::Scored(1.5));
    b.record("id-b", RunOutcome::Scored(2.5));
    b
}

fn sample_graph() -> Result<Graph, omni_core::Error> {
    let mut g = Graph::new();
    let a1 = Action::new("read_state", serde_json::json!({ "step": 0 }))?;
    let a2 = Action::new("purge_cache", serde_json::json!({ "step": 1 }))?;
    g.apply(
        &GraphUpdate {
            add_nodes: vec![
                Node::Action(ActionNode::new(a1.clone(), 0, 1)),
                Node::Action(ActionNode::new(a2.clone(), 1, 1)),
            ],
            add_edges: vec![(a2.id.clone(), EdgeKind::Requires, a1.id.clone())],
        },
        1,
    );
    Ok(g)
}

// ---- RolloutRecord -------------------------------------------------------

fn rollout_record_round_trips() -> Certificate {
    let theorem =
        "a RolloutRecord written to disk via write_atomic reads back bit-equal to the original";
    let dir = scratch("rr-roundtrip");
    let path = dir.join("rec.json");
    let residual = match RolloutRecord::from_run(sample_baseline(), sample_verdict()) {
        Ok(rec) => match rec.write(&path) {
            Ok(()) => match RolloutRecord::read(&path) {
                Ok(back) => exact(rec == back),
                Err(e) => failed("rollout_record_round_trips: read", &e.to_string()),
            },
            Err(e) => failed("rollout_record_round_trips: write", &e.to_string()),
        },
        Err(e) => failed("rollout_record_round_trips: build", &e.to_string()),
    };
    if let Err(e) = std::fs::remove_dir_all(&dir) {
        omni_core::note_advisory(
            &format!("cleaning up {}", dir.display()),
            &omni_core::Error::from(e),
        );
    }
    cert(
        "records",
        "rollout_record_round_trips",
        theorem,
        residual,
        0.0,
    )
}

fn rollout_record_missing_version_is_refused() -> Certificate {
    let theorem = "a rollout record file with no `version` field is refused by the reader, \
                   identically to one with an unknown version (AGENTS.md rule 5)";
    let dir = scratch("rr-no-version");
    let path = dir.join("bad.json");
    // A record body that looks valid except for the missing `version`.
    let body = serde_json::json!({
        "recorded_at_unix": 1,
        "baseline": { "runs": {} },
        "installed": false,
        "refusal": null,
        "reward": {
            "matched": 0, "delta": 0.0,
            "skipped_for_env": [], "unpaired_baseline": [], "unpaired_patched": [],
        },
        "patched": { "runs": {} },
        "episode_summaries": [],
    });
    let residual = match serde_json::to_string(&body) {
        Ok(s) => match write_atomic(&path, &s) {
            Ok(()) => match RolloutRecord::read(&path) {
                Err(omni_core::Error::Format(msg)) => exact(msg.contains("no `version`")),
                Err(other) => {
                    failed(
                        "rollout_record_missing_version_is_refused: unexpected error",
                        &format!("{other}"),
                    )
                }
                Ok(_) => exact(false),
            },
            Err(e) => failed(
                "rollout_record_missing_version_is_refused: write_atomic",
                &e.to_string(),
            ),
        },
        Err(e) => failed(
            "rollout_record_missing_version_is_refused: serde",
            &e.to_string(),
        ),
    };
    if let Err(e) = std::fs::remove_dir_all(&dir) {
        omni_core::note_advisory(
            &format!("cleaning up {}", dir.display()),
            &omni_core::Error::from(e),
        );
    }
    cert(
        "records",
        "rollout_record_missing_version_is_refused",
        theorem,
        residual,
        0.0,
    )
}

fn rollout_record_wrong_version_is_refused() -> Certificate {
    let theorem = "a rollout record file whose `version` is not this build's is refused, \
                   with the mismatch naming both numbers";
    let dir = scratch("rr-v99");
    let path = dir.join("v99.json");
    let body = serde_json::json!({
        "version": 99,
        "recorded_at_unix": 1,
        "baseline": { "runs": {} },
        "installed": false,
        "refusal": null,
        "reward": {
            "matched": 0, "delta": 0.0,
            "skipped_for_env": [], "unpaired_baseline": [], "unpaired_patched": [],
        },
        "patched": { "runs": {} },
        "episode_summaries": [],
    });
    let residual = match serde_json::to_string(&body) {
        Ok(s) => match write_atomic(&path, &s) {
            Ok(()) => match RolloutRecord::read(&path) {
                Err(omni_core::Error::Format(msg)) => exact(
                    msg.contains("version 99") && msg.contains("supports version 1"),
                ),
                Err(other) => failed(
                    "rollout_record_wrong_version_is_refused: unexpected error",
                    &format!("{other}"),
                ),
                Ok(_) => exact(false),
            },
            Err(e) => failed(
                "rollout_record_wrong_version_is_refused: write_atomic",
                &e.to_string(),
            ),
        },
        Err(e) => failed(
            "rollout_record_wrong_version_is_refused: serde",
            &e.to_string(),
        ),
    };
    if let Err(e) = std::fs::remove_dir_all(&dir) {
        omni_core::note_advisory(
            &format!("cleaning up {}", dir.display()),
            &omni_core::Error::from(e),
        );
    }
    cert(
        "records",
        "rollout_record_wrong_version_is_refused",
        theorem,
        residual,
        0.0,
    )
}

// ---- GraphSnapshot -------------------------------------------------------

fn graph_snapshot_round_trips() -> Certificate {
    let theorem =
        "a GraphSnapshot written via write_atomic reads back bit-equal to the original";
    let dir = scratch("gs-roundtrip");
    let path = dir.join("snap.json");
    let residual = match sample_graph() {
        Ok(g) => match GraphSnapshot::capture(g) {
            Ok(snap) => match snap.write(&path) {
                Ok(()) => match GraphSnapshot::read(&path) {
                    Ok(back) => exact(snap == back),
                    Err(e) => failed(
                        "graph_snapshot_round_trips: read",
                        &e.to_string(),
                    ),
                },
                Err(e) => failed("graph_snapshot_round_trips: write", &e.to_string()),
            },
            Err(e) => failed("graph_snapshot_round_trips: capture", &e.to_string()),
        },
        Err(e) => failed("graph_snapshot_round_trips: fixture", &e.to_string()),
    };
    if let Err(e) = std::fs::remove_dir_all(&dir) {
        omni_core::note_advisory(
            &format!("cleaning up {}", dir.display()),
            &omni_core::Error::from(e),
        );
    }
    cert(
        "records",
        "graph_snapshot_round_trips",
        theorem,
        residual,
        0.0,
    )
}

fn graph_snapshot_missing_version_is_refused() -> Certificate {
    let theorem = "a graph snapshot with no `version` field is refused by the reader";
    let dir = scratch("gs-no-version");
    let path = dir.join("bad.json");
    let body = serde_json::json!({
        "recorded_at_unix": 1,
        "node_count": 0,
        "edge_count": 0,
        "last_round": 0,
        "graph": { "nodes": {}, "edges": [], "last_round": 0 },
    });
    let residual = match serde_json::to_string(&body) {
        Ok(s) => match write_atomic(&path, &s) {
            Ok(()) => match GraphSnapshot::read(&path) {
                Err(omni_core::Error::Format(msg)) => exact(msg.contains("no `version`")),
                Err(other) => failed(
                    "graph_snapshot_missing_version_is_refused: unexpected error",
                    &format!("{other}"),
                ),
                Ok(_) => exact(false),
            },
            Err(e) => failed(
                "graph_snapshot_missing_version_is_refused: write_atomic",
                &e.to_string(),
            ),
        },
        Err(e) => failed(
            "graph_snapshot_missing_version_is_refused: serde",
            &e.to_string(),
        ),
    };
    if let Err(e) = std::fs::remove_dir_all(&dir) {
        omni_core::note_advisory(
            &format!("cleaning up {}", dir.display()),
            &omni_core::Error::from(e),
        );
    }
    cert(
        "records",
        "graph_snapshot_missing_version_is_refused",
        theorem,
        residual,
        0.0,
    )
}

fn graph_snapshot_tampered_mirror_is_refused() -> Certificate {
    let theorem = "a graph snapshot whose mirror count disagrees with its own graph body \
                   is refused (a truncated BTreeMap on replay would look exactly like this)";
    let dir = scratch("gs-tampered");
    let path = dir.join("tampered.json");
    // `node_count = 99` is a lie; the embedded graph has zero nodes.
    let body = serde_json::json!({
        "version": 1,
        "recorded_at_unix": 1,
        "node_count": 99,
        "edge_count": 0,
        "last_round": 0,
        "graph": { "nodes": {}, "edges": [], "last_round": 0 },
    });
    let residual = match serde_json::to_string(&body) {
        Ok(s) => match write_atomic(&path, &s) {
            Ok(()) => match GraphSnapshot::read(&path) {
                Err(omni_core::Error::Format(msg)) => exact(msg.contains("declares node_count=99")),
                Err(other) => failed(
                    "graph_snapshot_tampered_mirror_is_refused: unexpected error",
                    &format!("{other}"),
                ),
                Ok(_) => exact(false),
            },
            Err(e) => failed(
                "graph_snapshot_tampered_mirror_is_refused: write_atomic",
                &e.to_string(),
            ),
        },
        Err(e) => failed(
            "graph_snapshot_tampered_mirror_is_refused: serde",
            &e.to_string(),
        ),
    };
    if let Err(e) = std::fs::remove_dir_all(&dir) {
        omni_core::note_advisory(
            &format!("cleaning up {}", dir.display()),
            &omni_core::Error::from(e),
        );
    }
    cert(
        "records",
        "graph_snapshot_tampered_mirror_is_refused",
        theorem,
        residual,
        0.0,
    )
}
