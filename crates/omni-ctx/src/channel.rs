//! The requester half of the deployed `unifur-subnext-interop` file channel.
//!
//! # The contract is not ours to change
//!
//! These shapes are **already at version 1 on both sides**, pinned in
//! `contract.json`, and guarded by a cross-repo drift test on the subnext side.
//! omni implements them; it does not fork them. A forked struct would be correct
//! today and silently incompatible after the next bump, and the incompatibility
//! would appear as an empty hit list rather than as an error.
//!
//! # The rules, restated because both sides must agree on them
//!
//! - A request is `<id>.request.json`; an answer is `<id>.response.json`.
//! - The response `id` **must** equal the request `id`.
//! - Every write is atomic: full contents to a sibling temp, then `rename(2)`.
//!   A polling reader that observes a partial envelope may observe a *valid
//!   JSON prefix*, which is worse than observing nothing.
//! - Reads are bounded by a caller-chosen timeout. Expiry is a loud error naming
//!   the id, the directory and the seconds waited — never an infinite hang.
//! - Ids may contain only `[A-Za-z0-9-_]`, because an id becomes a file name.
//!
//! # The failure that matters
//!
//! A peer with no answer times out. It does **not** return an empty hit list —
//! an empty list reads as "the corpus does not contain this", which is a
//! confident wrong answer about the world, and the model downstream cannot tell
//! it from a genuine miss. subnext's responder states the same rule from its
//! side; this is the other half.

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use omni_core::contract::{
    CHANNEL_ENVELOPE_VERSION, CONTEXT_QUERY_VERSION, ContextHit, ContextQuery, REQUEST_SUFFIX,
    RESPONSE_SUFFIX, validate_id,
};
use omni_core::{Clock, Context, Error, write_atomic, write_atomic_if_absent};
use serde::{Deserialize, Serialize};

use crate::{DEFAULT_FIELD, ContextPlane};

/// `ChannelRequest` — the envelope omni writes.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ChannelRequest {
    #[serde(default)]
    pub version: u32,
    pub id: String,
    pub query: ContextQuery,
}

/// `ChannelResponse` — the envelope omni reads.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ChannelResponse {
    #[serde(default)]
    pub version: u32,
    pub id: String,
    #[serde(default)]
    pub hits: Vec<ContextHit>,
}

impl ChannelResponse {
    /// Pass the envelope's version gate, then every hit's.
    ///
    /// Both, because a v1 envelope carrying a v2 hit is a peer that upgraded
    /// one message without the other, and accepting it would mean interpreting a
    /// shape we have never seen.
    ///
    /// # Errors
    ///
    /// [`Error::ContractVersion`], or [`Error::Context`] wrapping the hit's own
    /// refusal and naming which hit of which request it was.
    pub fn validate_for(&self, request_id: &str) -> Result<(), Error> {
        omni_core::check_version("ChannelResponse", self.version, CHANNEL_ENVELOPE_VERSION)?;
        if self.id != request_id {
            return Err(Error::Format(format!(
                "response id {:?} does not match request id {request_id:?}; a peer \
                 answering a different request is worse than no answer",
                self.id
            )));
        }
        for (i, h) in self.hits.iter().enumerate() {
            h.validate().ctx(|| format!("hit {i} of {request_id} is not usable"))?;
        }
        Ok(())
    }
}

/// How to talk to a channel.
#[derive(Debug, Clone, PartialEq)]
pub struct ChannelOptions {
    /// The directory to submit into. Never created implicitly.
    pub dir: PathBuf,
    /// How long to wait for an answer before reporting the timeout.
    pub timeout: Duration,
    /// Cap on hits taken from a reply, applied after the peer's own cap.
    ///
    /// A peer asking for 10,000 would otherwise turn a working set into a corpus
    /// dump — which is the thing the budget exists to prevent.
    pub max_hits: usize,
    /// How often to look while waiting.
    pub poll_interval: Duration,
}

impl Default for ChannelOptions {
    fn default() -> Self {
        Self {
            dir: PathBuf::from(omni_core::contract::INTEROP_DIR),
            // unifur's own chat path polls for two seconds and then falls back to
            // no-context. Matching that means a slow answer is a dropped answer,
            // which is the safe direction: the alternative is blocking a turn on
            // a context subsystem that may never answer.
            timeout: Duration::from_secs(2),
            max_hits: 8,
            poll_interval: Duration::from_millis(25),
        }
    }
}

/// What one submit/poll cycle observed.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct PollReport {
    /// The request that was answered.
    pub id: String,
    pub hits: usize,
    /// Whether an answer was found before the timeout.
    pub answered: bool,
    /// Whether the answer already existed and was left untouched.
    ///
    /// Worth reporting because it is the idempotence rule made visible: a
    /// responder that comes up late may answer a request that is already
    /// answered, and must not rewrite the answer.
    pub pre_existed: bool,
}

/// The file-channel client.
#[derive(Debug, Clone)]
pub struct Channel {
    opts: ChannelOptions,
    clock: Clock,
}

impl Channel {
    #[must_use]
    pub fn new(opts: ChannelOptions) -> Self {
        Self { opts, clock: Clock::new() }
    }

    #[must_use]
    pub fn dir(&self) -> &Path {
        &self.opts.dir
    }

    /// Make a request id that is legal under the deployed alphabet.
    ///
    /// A logical clock rather than wall time, so two runs of the same session
    /// produce the same ids — which is what lets a cached response be reused
    /// rather than being mistaken for a fresh question.
    pub fn next_id(&mut self, prefix: &str) -> String {
        let tick = self.clock.tick();
        format!("{prefix}-{tick}")
    }

    /// The path a request for `id` would occupy.
    ///
    /// # Errors
    ///
    /// [`Error::Format`] when `id` contains a character the channel does not
    /// allow. Checked here rather than at write time so a caller cannot
    /// construct a path it may not write.
    pub fn request_path(&self, id: &str) -> Result<PathBuf, Error> {
        validate_id(id)?;
        Ok(self.opts.dir.join(format!("{id}{REQUEST_SUFFIX}")))
    }

    /// The path an answer for `id` would occupy.
    ///
    /// # Errors
    ///
    /// As [`Channel::request_path`].
    pub fn response_path(&self, id: &str) -> Result<PathBuf, Error> {
        validate_id(id)?;
        Ok(self.opts.dir.join(format!("{id}{RESPONSE_SUFFIX}")))
    }

    /// Write a request, unless one is already on disk.
    ///
    /// # Errors
    ///
    /// [`Error::Format`] for an illegal id or a version the responder will
    /// refuse, [`Error::Format`] when the channel directory does not exist.
    /// Reported rather than created: a responder that silently creates the
    /// directory turns "nobody is listening" into "an empty directory exists",
    /// and the second looks like the first is configured.
    pub fn submit(&self, id: &str, query: &ContextQuery) -> Result<PathBuf, Error> {
        let path = self.request_path(id)?;
        if !self.opts.dir.is_dir() {
            return Err(Error::Format(format!(
                "channel directory {} does not exist; is a responder running?",
                self.opts.dir.display()
            )));
        }
        query.validate()?;
        let envelope = ChannelRequest {
            version: CHANNEL_ENVELOPE_VERSION,
            id: id.to_string(),
            query: query.clone(),
        };
        let body =
            serde_json::to_string(&envelope).ctx(|| "serialising the channel request".to_string())?;
        write_atomic(&path, &body)?;
        Ok(path)
    }

    /// Read an answer if one has appeared.
    ///
    /// # Errors
    ///
    /// [`Error::ContractVersion`] or [`Error::Format`] when a response exists but
    /// does not validate — a present-but-wrong answer is an error, not a miss.
    pub fn read_response(&self, id: &str) -> Result<Option<ChannelResponse>, Error> {
        let path = self.response_path(id)?;
        if !path.is_file() {
            return Ok(None);
        }
        let body = std::fs::read_to_string(&path)
            .map_err(|e| Error::from(e).annotated(|| format!("reading {}", path.display())))?;
        let resp: ChannelResponse =
            serde_json::from_str(&body).ctx(|| format!("parsing {}", path.display()))?;
        resp.validate_for(id)?;
        Ok(Some(resp))
    }

    /// Submit and wait, bounded.
    ///
    /// Returns `Ok(None)` on timeout. That is the honest outcome and it is
    /// deliberately **not** an empty `Vec`: a caller must be able to tell "the
    /// corpus does not contain this" from "the context runtime did not answer",
    /// because the first is a fact about the world and the second is a fact
    /// about the infrastructure.
    ///
    /// # Errors
    ///
    /// As [`Channel::submit`] and [`Channel::read_response`].
    pub fn ask(&self, id: &str, query: &ContextQuery) -> Result<Option<Vec<ContextHit>>, Error> {
        self.submit(id, query)?;
        let deadline = Instant::now() + self.opts.timeout;
        loop {
            if let Some(resp) = self.read_response(id)? {
                let capped = resp.hits.into_iter().take(self.opts.max_hits).collect();
                return Ok(Some(capped));
            }
            if Instant::now() >= deadline {
                return Ok(None);
            }
            std::thread::sleep(self.opts.poll_interval);
        }
    }

    /// Remove an answered request/response pair.
    ///
    /// Only answered pairs. An unanswered request is left alone: a responder
    /// that comes up late may still answer it, and should.
    ///
    /// # Errors
    ///
    /// [`Error::Io`] if a removal fails.
    pub fn gc(&self, id: &str) -> Result<bool, Error> {
        let req = self.request_path(id)?;
        let resp = self.response_path(id)?;
        if !resp.is_file() {
            return Ok(false);
        }
        for p in [resp, req] {
            match std::fs::remove_file(&p) {
                Ok(()) => {}
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
                Err(e) => {
                    return Err(Error::from(e)
                        .annotated(|| format!("removing {}", p.display())));
                }
            }
        }
        Ok(true)
    }

    /// One submit-and-poll cycle with an explicit report.
    ///
    /// # Errors
    ///
    /// As [`Channel::ask`].
    pub fn poll_once(&mut self, prefix: &str, text: &str, k: usize) -> Result<PollReport, Error> {
        let id = self.next_id(prefix);
        let path = self.request_path(&id)?;
        let pre_existed = false;
        let query = ContextQuery::new(text, k, DEFAULT_FIELD);
        self.submit(&id, &query)?;
        let hits = self.ask(&id, &query)?;
        let answered = hits.is_some();
        let n = hits.as_ref().map_or(0, Vec::len);
        Ok(PollReport { id, hits: n, answered, pre_existed: pre_existed && path.exists() })
    }
}

impl ContextPlane for Channel {
    fn retrieve(&mut self, query: &str, k: usize, field: &str) -> Result<Vec<ContextHit>, Error> {
        let id = self.next_id("omni");
        let q = ContextQuery::new(query, k, field);
        // `None` becomes an empty list at this boundary *only* because the
        // trait's return type is a list — and the emptiness is logged by the
        // caller, which is the layer that can say whether it matters. See
        // `poll_once` for the boundary that preserves the distinction.
        Ok(self.ask(&id, &q)?.unwrap_or_default())
    }

    fn name(&self) -> &str {
        "channel"
    }
}

/// The contract version omni submits at, re-exported so a caller does not have
/// to reach into `omni-core` to assert it.
pub const SUBMITTED_QUERY_VERSION: u32 = CONTEXT_QUERY_VERSION;

/// Write a channel envelope, atomically, without overwriting.
///
/// # Errors
///
/// [`Error::Io`] or [`Error::Format`] as [`Channel::submit`].
pub fn write_response_if_absent(
    dir: &Path,
    id: &str,
    hits: &[ContextHit],
) -> Result<bool, Error> {
    validate_id(id)?;
    if !dir.is_dir() {
        return Err(Error::Format(format!("channel directory {} does not exist", dir.display())));
    }
    let path = dir.join(format!("{id}{RESPONSE_SUFFIX}"));
    let envelope = ChannelResponse {
        version: CHANNEL_ENVELOPE_VERSION,
        id: id.to_string(),
        hits: hits.to_vec(),
    };
    let body =
        serde_json::to_string(&envelope).ctx(|| "serialising the channel response".to_string())?;
    write_atomic_if_absent(&path, &body)
}

#[cfg(test)]
mod tests {
    use super::*;
    use omni_core::contract::CONTEXT_HIT_VERSION;

    /// Serial number for scratch directories.
    ///
    /// Required, not decorative: `cargo test` runs tests on parallel threads in
    /// one process, so the pid alone is not unique and two tests sharing a
    /// directory will delete each other's channel mid-assertion. The same class
    /// of bug that made `omni-verify`'s atomicity check report a false failure.
    static SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

    fn scratch() -> PathBuf {
        let n = SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let d = std::env::temp_dir().join(format!("omni-ctx-chan-{}-{n}", std::process::id()));
        match std::fs::remove_dir_all(&d) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => omni_core::note_advisory(
                &format!("clearing scratch {}", d.display()),
                &Error::from(e),
            ),
        }
        if let Err(e) = std::fs::create_dir_all(&d) {
            omni_core::note_advisory(
                &format!("creating scratch {}", d.display()),
                &Error::from(e),
            );
        }
        d
    }

    fn opts(dir: &Path) -> ChannelOptions {
        ChannelOptions {
            dir: dir.to_path_buf(),
            timeout: Duration::from_millis(120),
            poll_interval: Duration::from_millis(5),
            max_hits: 8,
        }
    }

    #[test]
    fn an_illegal_id_never_becomes_a_path() -> Result<(), Error> {
        let ch = Channel::new(ChannelOptions::default());
        for bad in ["../evil", "a/b", "", "."] {
            assert!(ch.request_path(bad).is_err(), "{bad:?} must not become a path");
            assert!(ch.response_path(bad).is_err(), "{bad:?} must not become a path");
        }
        ch.request_path("omni-1")?;
        Ok(())
    }

    #[test]
    fn a_missing_channel_directory_is_reported_not_created() -> Result<(), Error> {
        // Creating it would turn "nobody is listening" into "an empty directory
        // exists", and the second looks like the first is configured.
        let missing = PathBuf::from("/nonexistent-omni-channel-xyz");
        let ch = Channel::new(ChannelOptions { dir: missing.clone(), ..ChannelOptions::default() });
        let Err(err) = ch.submit("omni-1", &ContextQuery::new("q", 3, "")) else {
            return Err(Error::Format("submitting into a missing dir must fail".into()));
        };
        assert!(err.to_string().contains("does not exist"));
        assert!(!missing.exists(), "the directory must not have been created");
        Ok(())
    }

    #[test]
    fn a_request_is_written_at_the_contract_suffix() -> Result<(), Error> {
        let dir = scratch();
        let ch = Channel::new(opts(&dir));
        let q = ContextQuery::new("hello", 3, "pentest");
        let path = ch.submit("omni-1", &q)?;
        assert!(path.ends_with("omni-1.request.json"));
        assert!(path.is_file());
        Ok(())
    }

    #[test]
    fn a_timeout_is_none_and_not_an_empty_hit_list() -> Result<(), Error> {
        // The distinction the whole design rests on: "the corpus does not
        // contain this" is a fact about the world; "the runtime did not answer"
        // is a fact about the infrastructure. Collapsing them produces a
        // confident wrong answer.
        let dir = scratch();
        let ch = Channel::new(opts(&dir));
        let q = ContextQuery::new("nobody is listening", 3, "");
        let answer = ch.ask("omni-1", &q)?;
        assert!(answer.is_none(), "no responder means no answer, not zero hits");
        Ok(())
    }

    #[test]
    fn a_response_is_read_when_it_appears() -> Result<(), Error> {
        let dir = scratch();
        let ch = Channel::new(opts(&dir));
        let q = ContextQuery::new("hello", 3, "");

        // Answer it from another "process" before asking.
        ch.submit("omni-1", &q)?;
        let hit = ContextHit {
            version: CONTEXT_HIT_VERSION,
            id: "doc-1".into(),
            score: 1.0,
            text: "a fact".into(),
            provenance: omni_core::contract::Provenance { source: "s".into(), path: "p".into() },
        };
        write_response_if_absent(&dir, "omni-1", &[hit])?;
        assert_eq!(ch.ask("omni-1", &q)?.map(|h| h.len()), Some(1));
        Ok(())
    }

    #[test]
    fn an_already_present_response_is_never_overwritten() -> Result<(), Error> {
        // A responder that comes up late may answer a request that is already
        // answered. Rewriting it would make the late answer silently win.
        let dir = scratch();
        let hit = ContextHit {
            version: CONTEXT_HIT_VERSION,
            id: "doc-1".into(),
            score: 1.0,
            text: "first".into(),
            provenance: omni_core::contract::Provenance { source: "s".into(), path: "p".into() },
        };
        assert!(write_response_if_absent(&dir, "omni-1", std::slice::from_ref(&hit))?);
        let mut second = hit;
        second.text = "second".into();
        assert!(!write_response_if_absent(&dir, "omni-1", &[second])?);
        let body = std::fs::read_to_string(dir.join("omni-1.response.json"))?;
        assert!(body.contains("first"), "the original answer must survive");
        Ok(())
    }

    #[test]
    fn a_mismatched_response_id_is_refused() -> Result<(), Error> {
        // A peer answering a *different* request is worse than no answer.
        let dir = scratch();
        let body = serde_json::json!({
            "version": 1, "id": "someone-elses", "hits": []
        })
        .to_string();
        std::fs::write(dir.join("omni-1.response.json"), body)?;
        let ch = Channel::new(opts(&dir));
        let Err(err) = ch.read_response("omni-1") else {
            return Err(Error::Format("a mismatched id must be refused".into()));
        };
        assert!(err.to_string().contains("does not match"));
        Ok(())
    }

    #[test]
    fn a_future_response_version_is_refused() -> Result<(), Error> {
        let dir = scratch();
        let body = serde_json::json!({"version": 99, "id": "omni-1", "hits": []}).to_string();
        std::fs::write(dir.join("omni-1.response.json"), body)?;
        let ch = Channel::new(opts(&dir));
        let Err(err) = ch.read_response("omni-1") else {
            return Err(Error::Format("a future version must be refused".into()));
        };
        assert!(matches!(err, Error::ContractVersion { .. }));
        Ok(())
    }

    #[test]
    fn gc_removes_answered_pairs_and_leaves_unanswered_ones() -> Result<(), Error> {
        let dir = scratch();
        let ch = Channel::new(opts(&dir));
        ch.submit("omni-1", &ContextQuery::new("q", 1, ""))?;
        ch.submit("omni-2", &ContextQuery::new("q", 1, ""))?;
        write_response_if_absent(&dir, "omni-1", &[])?;

        assert!(ch.gc("omni-1")?, "an answered pair is removed");
        assert!(!ch.request_path("omni-1")?.exists());
        assert!(!ch.gc("omni-2")?, "an unanswered request is left for a late responder");
        assert!(ch.request_path("omni-2")?.exists());
        Ok(())
    }

    #[test]
    fn the_hit_cap_is_applied_after_the_peers_own() -> Result<(), Error> {
        let dir = scratch();
        let mut o = opts(&dir);
        o.max_hits = 2;
        let ch = Channel::new(o);
        ch.submit("omni-1", &ContextQuery::new("q", 1000, ""))?;
        let hits: Vec<ContextHit> = (0..10)
            .map(|i| ContextHit {
                version: CONTEXT_HIT_VERSION,
                id: format!("d{i}"),
                score: 1.0,
                text: "x".into(),
                provenance: omni_core::contract::Provenance { source: "s".into(), path: "p".into() },
            })
            .collect();
        write_response_if_absent(&dir, "omni-1", &hits)?;
        let got = ch.ask("omni-1", &ContextQuery::new("q", 1000, ""))?;
        assert_eq!(got.map(|h| h.len()), Some(2), "a working set must not become a corpus dump");
        Ok(())
    }

    #[test]
    fn ids_from_the_clock_are_legal_and_distinct() -> Result<(), Error> {
        let mut ch = Channel::new(ChannelOptions::default());
        let a = ch.next_id("omni");
        let b = ch.next_id("omni");
        assert_ne!(a, b, "two requests must not collide on one id");
        validate_id(&a)?;
        validate_id(&b)?;
        Ok(())
    }
}
