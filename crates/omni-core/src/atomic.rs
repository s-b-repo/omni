//! Atomic file writes: write to a sibling temporary, then rename over the
//! destination.
//!
//! # Why
//!
//! omni writes into directories other processes are polling. A reader that
//! observes a half-written envelope sees a truncated JSON document and — worse —
//! may see a *valid* prefix of one, because JSON objects close early. The
//! rename is what makes "the file exists" and "the file is complete" the same
//! statement.
//!
//! Both sides of the deployed unifur/subnext channel state this rule
//! independently, in `contract.json` and in the responder's module docs. This
//! is the third implementation of it, and the only one whose tests check the
//! rename actually happened.
//!
//! # The temporary name
//!
//! `<name>.tmp-<pid>`. The pid is there so two processes writing the same
//! destination cannot collide on the temporary. It is not a security boundary
//! and is not treated as one: a hostile process on the same host can already
//! write into this directory.

use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use crate::error::{Error, note_advisory};

/// Monotonic suffix so two writes in the same process do not reuse a temporary.
static WRITE_SEQ: AtomicU64 = AtomicU64::new(0);

fn temp_path(dest: &Path) -> PathBuf {
    let seq = WRITE_SEQ.fetch_add(1, Ordering::Relaxed);
    let pid = std::process::id();
    let mut name = dest.file_name().map_or_else(String::new, |s| s.to_string_lossy().into_owned());
    name.push_str(&format!(".tmp-{pid}-{seq}"));
    dest.with_file_name(name)
}

/// Write `contents` to `dest` such that a concurrent reader sees either the
/// previous file or the complete new one, never a partial write.
///
/// # Errors
///
/// Returns [`Error::Io`] if the temporary cannot be written, renamed, or if the
/// destination's parent does not exist. The temporary is removed on failure —
/// leaving it behind would accumulate debris in a directory another process
/// polls, and a stale `.tmp-*` is indistinguishable from a live write to
/// anything inspecting the directory.
pub fn write_atomic(dest: &Path, contents: &str) -> Result<(), Error> {
    let parent = dest.parent().unwrap_or_else(|| Path::new("."));
    if !parent.is_dir() {
        return Err(Error::Format(format!(
            "cannot write {}: parent directory {} does not exist",
            dest.display(),
            parent.display()
        )));
    }

    let tmp = temp_path(dest);
    let write_result = (|| -> std::io::Result<()> {
        let mut f = std::fs::File::create(&tmp)?;
        f.write_all(contents.as_bytes())?;
        // Flush before the rename. Without it the rename can land while the
        // bytes are still in the page cache, and a reader on another host sees
        // the rename with no content — the exact failure atomicity exists to
        // prevent. fsync is the honest boundary; it is also the slow one, which
        // is why this is not the default for the reward cache.
        f.sync_all()?;
        Ok(())
    })();

    if let Err(e) = write_result {
        cleanup(&tmp, "after a failed write");
        return Err(Error::from(e).annotated(|| format!("writing {}", tmp.display())));
    }

    if let Err(e) = std::fs::rename(&tmp, dest) {
        cleanup(&tmp, "after a failed rename");
        return Err(Error::from(e).annotated(|| format!("renaming {} onto {}", tmp.display(), dest.display())));
    }
    Ok(())
}

/// Remove a temporary, recording rather than discarding a failure.
///
/// A leftover `.tmp-*` in a directory another process polls is
/// indistinguishable from a live write, so a cleanup that silently failed
/// leaves the exact ambiguity the atomic rename exists to prevent.
fn cleanup(tmp: &Path, when: &str) {
    if let Err(e) = std::fs::remove_file(tmp) {
        note_advisory(&format!("removing {when}: {}", tmp.display()), &Error::from(e));
    }
}

/// Write only if the destination does not already exist.
///
/// Returns `false` when `dest` was already present and left untouched. This is
/// the idempotence rule the channel depends on: a responder that comes up late
/// may answer a request whose response is already on disk, and must not
/// rewrite it.
pub fn write_atomic_if_absent(dest: &Path, contents: &str) -> Result<bool, Error> {
    if dest.exists() {
        return Ok(false);
    }
    write_atomic(dest, contents)?;
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::Context;

    /// A directory under the target dir, named after the test so parallel test
    /// threads cannot collide.
    fn scratch(name: &str) -> Result<PathBuf, Error> {
        let dir = std::env::temp_dir().join(format!("omni-atomic-{}-{name}", std::process::id()));
        // A leftover directory from a previous run has to go. Its absence is
        // the normal case and is not news; any other failure is.
        match std::fs::remove_dir_all(&dir) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => note_advisory(&format!("clearing scratch {}", dir.display()), &Error::from(e)),
        }
        std::fs::create_dir_all(&dir).ctx(|| format!("creating scratch {}", dir.display()))?;
        Ok(dir)
    }

    #[test]
    fn writes_the_whole_contents_and_leaves_no_temp_behind() -> Result<(), Error> {
        let dir = scratch("whole")?;
        let dest = dir.join("a.json");
        write_atomic(&dest, "{\"version\":1}")?;

        assert_eq!(std::fs::read_to_string(&dest)?, "{\"version\":1}");

        let leftovers: Vec<String> = std::fs::read_dir(&dir)?
            .filter_map(|e| e.ok())
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .filter(|n| n.contains(".tmp-"))
            .collect();
        assert!(leftovers.is_empty(), "temps left behind: {leftovers:?}");
        Ok(())
    }

    #[test]
    fn a_second_write_replaces_rather_than_appends() -> Result<(), Error> {
        let dir = scratch("replace")?;
        let dest = dir.join("a.json");
        write_atomic(&dest, "first")?;
        write_atomic(&dest, "second")?;
        assert_eq!(std::fs::read_to_string(&dest)?, "second");
        Ok(())
    }

    #[test]
    fn if_absent_leaves_an_existing_file_untouched() -> Result<(), Error> {
        let dir = scratch("absent")?;
        let dest = dir.join("a.json");
        write_atomic(&dest, "original")?;

        let wrote = write_atomic_if_absent(&dest, "replacement")?;
        assert!(!wrote, "the second write must report that it did nothing");
        assert_eq!(
            std::fs::read_to_string(&dest)?,
            "original",
            "an existing response must never be rewritten: a late responder may \
             answer a request that is already answered"
        );
        Ok(())
    }

    #[test]
    fn a_missing_parent_is_an_error_not_a_panic() -> Result<(), Error> {
        let dest = Path::new("/nonexistent-omni-parent-dir-xyz/a.json");
        let Err(err) = write_atomic(dest, "x") else {
            return Err(Error::Format("writing into a missing parent must fail".into()));
        };
        match err {
            Error::Format(msg) => {
                assert!(msg.contains("parent"), "unhelpful message: {msg}");
                Ok(())
            }
            other => Err(Error::Format(format!("expected a Format error naming the parent, got {other}"))),
        }
    }
}