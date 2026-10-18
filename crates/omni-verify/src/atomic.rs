//! `atomic` — a polling reader never sees a fragment.
//!
//! omni writes into directories another process polls. If a reader can observe
//! a half-written envelope it will see truncated JSON, and — worse — it may see
//! a *valid prefix*, because a JSON object closes early. The rename is what
//! makes "the file exists" and "the file is complete" the same statement.
//!
//! This group checks the property directly rather than trusting the
//! implementation: it writes through [`omni_core::write_atomic`] while a reader
//! thread reads, and asserts that every observation is one of the two complete
//! states.

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use omni_core::write_atomic;

use crate::{Certificate, cert, exact, failed, with_cases};

/// Run the `atomic` group.
#[must_use]
pub fn run() -> Vec<Certificate> {
    vec![no_partial_reads_are_observable(), temps_are_not_left_behind()]
}

/// Serial number for scratch directories.
///
/// Required, not decorative. `cargo test` runs test functions on parallel
/// threads inside one process, so a pid alone does not make a scratch directory
/// unique: two threads calling the same certificate at the same time get the
/// same pid, and the second one's `remove_dir_all` deletes the directory the
/// first one's reader is watching. That showed up here as the atomicity check
/// reporting that the file "vanished" — a false failure caused entirely by the
/// harness disagreeing with itself. Counter plus pid, which is the same rule
/// peregrine states for temp files.
static SCRATCH_SEQ: AtomicU64 = AtomicU64::new(0);

fn scratch(name: &str) -> PathBuf {
    let seq = SCRATCH_SEQ.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!(
        "omni-verify-atomic-{}-{name}-{seq}",
        std::process::id()
    ));
    // A stale directory from a previous run must go, but its absence is the
    // normal case and is not news. Recorded, not discarded.
    if let Err(e) = std::fs::remove_dir_all(&dir) {
        if e.kind() != std::io::ErrorKind::NotFound {
            omni_core::note_advisory(&format!("clearing scratch {}", dir.display()), &omni_core::Error::from(e));
        }
    }
    if let Err(e) = std::fs::create_dir_all(&dir) {
        omni_core::note_advisory(&format!("creating scratch {}", dir.display()), &omni_core::Error::from(e));
    }
    dir
}

fn no_partial_reads_are_observable() -> Certificate {
    // Two complete contents, written repeatedly, while a reader samples the
    // file as fast as it can. Every sample must be exactly one of them.
    const A: &str = r#"{"version":1,"id":"r1","hits":[]}"#;
    const B: &str = r#"{"version":1,"id":"r2","hits":[{"id":"a"},{"id":"b"},{"id":"c"}]}"#;

    let dir = scratch("race");
    let dest = dir.join("channel.response.json");
    if let Err(e) = write_atomic(&dest, A) {
        return cert("atomic", "no_partial_reads", "a polling reader observes only complete files", failed("atomic", &e.to_string()), 0.0);
    }

    let stop = Arc::new(AtomicBool::new(false));
    let reader_stop = Arc::clone(&stop);
    let reader_dest = dest.clone();

    // The reader runs on its own thread and reports the first state that is
    // neither A nor B. It never stops on success — it runs for the duration of
    // the write loop, because a torn read is a rare event and a single sample
    // would miss it by construction.
    let reader = std::thread::spawn(move || {
        let mut observations = 0u64;
        let mut bad: Option<String> = None;
        let mut read_errors = 0u64;
        let mut first_error: Option<String> = None;
        while !reader_stop.load(Ordering::Relaxed) {
            match std::fs::read_to_string(&reader_dest) {
                Ok(seen) => {
                    observations += 1;
                    if seen != A && seen != B && bad.is_none() {
                        bad = Some(seen.chars().take(80).collect());
                    }
                }
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                    // The file is never unlinked by write_atomic, so its absence
                    // means something is wrong beyond tearing.
                    if bad.is_none() {
                        bad = Some(format!("file vanished mid-read: {e}"));
                    }
                }
                Err(e) => {
                    // Any other read failure is counted rather than discarded.
                    // A reader that could not open the file most of the time
                    // has observed nothing, and "observed nothing" must not be
                    // reported as "observed nothing torn".
                    read_errors += 1;
                    if first_error.is_none() {
                        first_error = Some(e.to_string());
                    }
                }
            }
        }
        (observations, bad, read_errors, first_error)
    });

    // Enough writes that a torn read has a real chance of occurring if one is
    // possible. fsync per write makes this slow; the loop count is chosen for
    // coverage, not for speed, and the group is not on a hot path.
    //
    // A failed write is counted, not discarded: a run where every write failed
    // would still report "no torn reads observed", which is a vacuous pass.
    let mut write_failures = 0usize;
    for i in 0..200usize {
        if let Err(e) = write_atomic(&dest, if i % 2 == 0 { B } else { A }) {
            write_failures += 1;
            omni_core::note_advisory("writing the race fixture", &e);
        }
    }
    stop.store(true, Ordering::Relaxed);

    let (observations, bad, read_errors, first_read_error) = match reader.join() {
        Ok(v) => v,
        Err(_) => {
            return cert("atomic", "no_partial_reads", "a polling reader observes only complete files", failed("atomic", "reader thread panicked"), 0.0)
        }
    };

    if read_errors > 0 {
        eprintln!(
            "omni-verify: atomic: reader hit {read_errors} read error(s), first: {}",
            first_read_error.unwrap_or_else(|| "unrecorded".to_string())
        );
    }

    if let Some(observed) = bad {
        eprintln!("omni-verify: atomic: reader observed a non-atomic state: {observed}");
        return with_cases(
            cert("atomic", "no_partial_reads", "a polling reader observes only complete files", 1.0, 0.0),
            observations as usize,
        );
    }

    with_cases(
        cert(
            "atomic",
            "no_partial_reads",
            "a concurrent reader of an atomically written file observes only a \
             complete previous or complete new file, and its reads actually \
             succeeded — so a reader that saw nothing is not reported as a \
             reader that saw nothing torn",
            exact(observations > 0 && write_failures == 0 && read_errors == 0),
            0.0,
        ),
        observations as usize,
    )
}

fn temps_are_not_left_behind() -> Certificate {
    // A `.tmp-*` file left in a directory another process polls is
    // indistinguishable from a live write to anything inspecting the directory,
    // and a crashed writer leaves one forever.
    const ROUNDS: usize = 20;
    let dir = scratch("temps");
    let mut leftovers = 0usize;
    let mut written = 0usize;

    for i in 0..ROUNDS {
        let dest = dir.join(format!("f{i}.json"));
        if write_atomic(&dest, "{}").is_ok() {
            written += 1;
        }
    }

    if let Ok(entries) = std::fs::read_dir(&dir) {
        for e in entries.flatten() {
            if e.file_name().to_string_lossy().contains(".tmp-") {
                leftovers += 1;
            }
        }
    }

    with_cases(
        cert(
            "atomic",
            "temps_are_not_left_behind",
            "a completed atomic write leaves no temporary file behind, because a \
             stale `.tmp-*` in a polled directory is indistinguishable from a \
             live write",
            exact(written == ROUNDS && leftovers == 0),
            0.0,
        ),
        ROUNDS,
    )
}

#[cfg(test)]
mod tests {
    use crate::run_group;

    #[test]
    fn the_group_runs_and_passes() -> Result<(), String> {
        let report = run_group("atomic").map_err(|e| e.to_string())?;
        assert!(report.passed(), "{}", report.render());
        Ok(())
    }
}