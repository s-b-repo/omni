//! On-disk rollout records: the one place a baseline/patch comparison lives
//! after `evaluate_patch` returns.
//!
//! What this closes: `docs/CONTRACT.md` "What is not true yet" has listed
//! "the graph and the loop shape records, nothing records yet" since the
//! crate was opened. The verdict existed as a `PatchVerdict` in memory; a
//! reader outside the process had no way to see it. This is the smallest
//! shape that turns an in-memory verdict into an on-disk, versioned,
//! round-trippable record.
//!
//! Discipline:
//!
//! * **Versioned.** A record carries `version = 1`, and the reader refuses
//!   any other value — including a missing field, which `#[serde(default)]`
//!   would silently turn into `0`. AGENTS.md rule 5: "a missing version is
//!   refused identically" to an unknown one. The field is required, and the
//!   reader validates the integer before `serde_json::from_value` touches
//!   the rest of the document.
//!
//! * **Atomic.** The writer uses [`omni_core::write_atomic`]: temp-file +
//!   rename, so a reader polling the directory sees either the previous
//!   file or the whole new one, never a truncated prefix.
//!
//! * **Round-trippable.** The round-trip through disk is a certificate (see
//!   `omni-verify`'s `rollout_records` group, added in the same commit as
//!   this module). A record that drops a field or reshuffles a map in
//!   serialization is caught before it reaches a consumer.
//!
//! * **Rule 10.** `Batch` and `Reward` are reported as they are, never
//!   polished. A reward with zero matched pairs serializes as
//!   `matched: 0` and the record stays honest about what it measured.

#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
#![forbid(unsafe_code)]

use std::path::Path;

use omni_core::{Context, Error, write_atomic};
use omni_eval::{Batch, Reward};
use serde::{Deserialize, Serialize};

use crate::rollout::PatchVerdict;

/// Version of the rollout record format. Bumped on any breaking shape
/// change. Readers refuse any other value, missing included.
pub const ROLLOUT_RECORD_VERSION: u32 = 1;

/// One episode's measurable summary. The full transcript is intentionally
/// omitted from the record — a transcript is useful for human debugging but
/// is also the largest field and the one that would blow a polled-directory
/// reader's buffer. A caller that wants transcripts should keep its own log
/// beside this record.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EpisodeSummary {
    pub task_id: String,
    pub score: f64,
    pub status: String,
    pub turns: usize,
}

/// The on-disk shape of one baseline + patched comparison.
///
/// Every field is a scalar or a serde-deriving type from `omni_eval`; the
/// record itself derives `Serialize`/`Deserialize`, which keeps the on-disk
/// shape and the in-memory shape one definition — a drift the "drop a
/// field, parse nothing" failure thrives on.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RolloutRecord {
    /// Record format version; always [`ROLLOUT_RECORD_VERSION`].
    pub version: u32,
    /// UTC Unix epoch seconds at the moment the record was built. A
    /// clock-before-epoch system (impossible in practice) makes `from_run`
    /// return [`Error::Format`] rather than lie with a zero.
    pub recorded_at_unix: u64,
    /// Baseline run outcomes, keyed by content-derived identity.
    pub baseline: Batch,
    /// Whether the sandbox accepted the patch. `false` is not an error at
    /// this layer: "the engineer wrote a bad patch" is a measured outcome.
    pub installed: bool,
    /// The named refusal when `installed` is `false`.
    pub refusal: Option<String>,
    /// Identity-paired reward over the two batches below.
    pub reward: Reward,
    /// Patched run outcomes, keyed by the same identities.
    pub patched: Batch,
    /// One summary per episode in the patched arm. Baseline summaries are
    /// out of scope: the baseline's episodes live in its own packet, and
    /// duplicating them in the record would invite one half to stay
    /// updated while the other rotted.
    pub episode_summaries: Vec<EpisodeSummary>,
}

impl RolloutRecord {
    /// Assemble a record from a baseline batch and a patch verdict.
    ///
    /// # Errors
    ///
    /// [`Error::Format`] when `SystemTime::now()` precedes `UNIX_EPOCH`.
    /// This is never actually observed on any system that completed its
    /// own boot; it is checked rather than skipped because a zero in the
    /// timestamp field would be indistinguishable from a successfully
    /// recorded run that claimed to happen in 1970.
    pub fn from_run(baseline: Batch, verdict: PatchVerdict) -> Result<Self, Error> {
        let recorded_at_unix = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_err(|e| {
                Error::Format(format!(
                    "system clock is before UNIX_EPOCH ({e}); a zero timestamp \
                     would misrepresent the record's age so it is refused"
                ))
            })?
            .as_secs();

        let episode_summaries = verdict
            .episodes
            .iter()
            .map(|e| EpisodeSummary {
                task_id: e.task_id.clone(),
                score: e.score,
                status: e.status.clone(),
                turns: e.turns,
            })
            .collect();

        Ok(Self {
            version: ROLLOUT_RECORD_VERSION,
            recorded_at_unix,
            baseline,
            installed: verdict.installed,
            refusal: verdict.refusal,
            reward: verdict.reward,
            patched: verdict.patched,
            episode_summaries,
        })
    }

    /// Serialize and write atomically.
    ///
    /// # Errors
    ///
    /// Whatever [`write_atomic`] returns — a missing parent directory, a
    /// disk-full write, a failed rename — plus a [`Error::Json`] if
    /// serialization fails (which it will not, for a record built from
    /// plain scalars and `BTreeMap`s).
    pub fn write(&self, path: &Path) -> Result<(), Error> {
        let json = serde_json::to_string_pretty(self)?;
        write_atomic(path, &json)
    }

    /// Read and deserialize, refusing any record whose `version` is not
    /// this build's.
    ///
    /// # Errors
    ///
    /// [`Error::Io`] if `path` cannot be read. [`Error::Format`] when the
    /// top-level JSON is not an object or when `version` is missing, is
    /// not an integer, or is not [`ROLLOUT_RECORD_VERSION`].
    /// [`Error::Json`] when the body parses as JSON but not as a record.
    pub fn read(path: &Path) -> Result<Self, Error> {
        let bytes = std::fs::read_to_string(path)
            .ctx(|| format!("reading {}", path.display()))?;
        let value: serde_json::Value = serde_json::from_str(&bytes)?;
        if !value.is_object() {
            return Err(Error::Format(format!(
                "{} is not a JSON object",
                path.display()
            )));
        }
        let version_raw = value.get("version").ok_or_else(|| {
            Error::Format(format!(
                "{} has no `version` field. AGENTS.md rule 5: a missing \
                 version is refused identically to an unknown one.",
                path.display()
            ))
        })?;
        let version = version_raw.as_u64().ok_or_else(|| {
            Error::Format(format!(
                "{} `version` is not an integer: {version_raw}",
                path.display()
            ))
        })?;
        if u32::try_from(version).ok() != Some(ROLLOUT_RECORD_VERSION) {
            return Err(Error::Format(format!(
                "{} has version {version}, this build supports version {}",
                path.display(),
                ROLLOUT_RECORD_VERSION
            )));
        }
        let record: RolloutRecord = serde_json::from_value(value)?;
        Ok(record)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use omni_core::HookEpisode;
    use omni_eval::RunOutcome;

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

    #[test]
    fn a_record_round_trips_through_disk() -> Result<(), Error> {
        let record = RolloutRecord::from_run(sample_baseline(), sample_verdict())?;
        let dir = tempdir()?;
        let path = dir.join("record.json");
        record.write(&path)?;
        let read_back = RolloutRecord::read(&path)?;
        if record != read_back {
            return Err(Error::Format(format!(
                "round trip changed the record. wrote {record:?} read back {read_back:?}"
            )));
        }
        Ok(())
    }

    #[test]
    fn a_missing_version_is_refused() -> Result<(), Error> {
        let dir = tempdir()?;
        let path = dir.join("bad.json");
        // Hand-build a JSON that looks like a record but omits `version`.
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
        write_atomic(&path, &serde_json::to_string(&body)?)?;
        match RolloutRecord::read(&path) {
            Err(Error::Format(msg)) => {
                if !msg.contains("no `version`") {
                    return Err(Error::Format(format!(
                        "expected a `no version` diagnosis, got: {msg}"
                    )));
                }
                Ok(())
            }
            Ok(r) => Err(Error::Format(format!(
                "expected Err(Format) for a missing-version record; got Ok({r:?})"
            ))),
            Err(other) => Err(Error::Format(format!(
                "expected Err(Format) for a missing-version record; got {other:?}"
            ))),
        }
    }

    #[test]
    fn a_wrong_version_is_refused() -> Result<(), Error> {
        let dir = tempdir()?;
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
        write_atomic(&path, &serde_json::to_string(&body)?)?;
        match RolloutRecord::read(&path) {
            Err(Error::Format(msg)) => {
                if !msg.contains("version 99") || !msg.contains("supports version 1") {
                    return Err(Error::Format(format!(
                        "expected a version-mismatch diagnosis naming both numbers, got: {msg}"
                    )));
                }
                Ok(())
            }
            Ok(r) => Err(Error::Format(format!(
                "expected Err(Format) for a v99 record; got Ok({r:?})"
            ))),
            Err(other) => Err(Error::Format(format!(
                "expected Err(Format) for a v99 record; got {other:?}"
            ))),
        }
    }

    #[test]
    fn a_non_object_top_level_is_refused() -> Result<(), Error> {
        let dir = tempdir()?;
        let path = dir.join("array.json");
        write_atomic(&path, "[1, 2, 3]")?;
        match RolloutRecord::read(&path) {
            Err(Error::Format(msg)) => {
                if !msg.contains("not a JSON object") {
                    return Err(Error::Format(format!(
                        "expected a `not a JSON object` diagnosis, got: {msg}"
                    )));
                }
                Ok(())
            }
            Ok(r) => Err(Error::Format(format!(
                "expected Err(Format) for a top-level array; got Ok({r:?})"
            ))),
            Err(other) => Err(Error::Format(format!(
                "expected Err(Format) for a top-level array; got {other:?}"
            ))),
        }
    }

    #[test]
    fn a_refused_patch_still_records() -> Result<(), Error> {
        // A record whose `installed = false` is the honest shape: the
        // baseline was captured, the patch was refused, the reward is
        // zero. Writing nothing in this case would be the exact failure
        // mode rule 10 exists to prevent — a reward that was never
        // computed getting lost because it was also not a success.
        let verdict = PatchVerdict {
            installed: false,
            refusal: Some("budget exceeded".to_string()),
            reward: Reward::zero(),
            patched: Batch::new(),
            episodes: Vec::new(),
        };
        let record = RolloutRecord::from_run(sample_baseline(), verdict)?;
        let dir = tempdir()?;
        let path = dir.join("refused.json");
        record.write(&path)?;
        let read_back = RolloutRecord::read(&path)?;
        if read_back.installed {
            return Err(Error::Format("`installed` should have round-tripped false".to_string()));
        }
        if read_back.refusal.as_deref() != Some("budget exceeded") {
            return Err(Error::Format(format!(
                "refusal did not round-trip: got {:?}",
                read_back.refusal
            )));
        }
        Ok(())
    }

    /// Minimal temp directory helper. Uses the pid + a monotonic counter
    /// so parallel tests cannot collide; a `Drop` impl removes the tree.
    struct TempDir {
        path: std::path::PathBuf,
    }
    impl TempDir {
        fn join(&self, name: &str) -> std::path::PathBuf {
            self.path.join(name)
        }
    }
    impl Drop for TempDir {
        fn drop(&mut self) {
            if let Err(e) = std::fs::remove_dir_all(&self.path) {
                omni_core::note_advisory(
                    &format!("removing temp dir {}", self.path.display()),
                    &Error::from(e),
                );
            }
        }
    }
    fn tempdir() -> Result<TempDir, Error> {
        static SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let seq = SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let pid = std::process::id();
        let path = std::env::temp_dir().join(format!("omni-loop-record-{pid}-{seq}"));
        std::fs::create_dir_all(&path)?;
        Ok(TempDir { path })
    }
}
