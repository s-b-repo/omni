//! The error type every `omni` crate returns.
//!
//! # Why one type and not `anyhow`
//!
//! A harness that installs model-written code is a harness whose failures must
//! be classifiable: a hook that blew its instruction budget, a contract whose
//! version is from the future, and a disk that is full are three different bugs
//! with three different fixes, and a stringly-typed error erases that
//! distinction exactly where it costs the most.
//!
//! So [`Error`] is a closed enum. [`Context`] exists to attach *where* a
//! failure happened without discarding *what* it was — the `source` survives,
//! which is the whole point of the `Context` variant as opposed to
//! `Format(String)`.
//!
//! ```
//! use omni_core::{Context, Error};
//!
//! fn read_len(path: &std::path::Path) -> Result<u64, Error> {
//!     let bytes = std::fs::read(path).ctx(|| format!("reading {}", path.display()))?;
//!     Ok(bytes.len() as u64)
//! }
//! # let missing = std::path::Path::new("/nonexistent/omni-doc-example");
//! # match read_len(missing) {
//! #     Err(Error::Context { .. }) => {}
//! #     other => panic!("expected an annotated io error, got {other:?}"),
//! # }
//! # Ok::<(), Error>(())
//! ```

/// Every failure omni can report.
///
/// The `Io`/`Json` variants exist so `?` composes; the rest are omni's own
/// taxonomy. Adding a variant is a deliberate act — it means a new way for the
/// harness to be wrong that a caller might want to branch on.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),

    #[error("json: {0}")]
    Json(#[from] serde_json::Error),

    /// Malformed input: a bad contract envelope, an out-of-range budget, a
    /// patch that is not shaped like a patch. The message is the diagnosis.
    #[error("{0}")]
    Format(String),

    /// A cross-process contract whose `version` this build does not speak.
    ///
    /// Separate from [`Error::Format`] because the response is different: a
    /// version mismatch is never a retry and never a silent reinterpretation,
    /// it is a loud stop naming both numbers. Every side of the unifur/subnext
    /// channel states this rule independently; a certificate pins it.
    #[error("contract version {got} for {kind} is not supported by this build (supported: {supported}; a missing version parses as 0 and is rejected the same way)")]
    ContractVersion {
        kind: &'static str,
        got: u32,
        supported: u32,
    },

    /// A harness hook failed to compile, or exceeded its budget while running.
    ///
    /// Carries the hook name and the budget that was exceeded, because "hook
    /// failed" without "which hook, which budget" is not an actionable log line.
    #[error("hook {hook} failed: {reason}")]
    Hook {
        hook: String,
        reason: String,
    },

    /// A peer reported that it is at capacity.
    ///
    /// Its own signal rather than a guess from a status code the harness
    /// invented: the server said so. Separate from a transport failure because
    /// the right answer differs — this is worth retrying after a backoff, and
    /// an `io::Error` dressed up to look like one would make the caller retry
    /// connection resets and overloads on the same schedule.
    #[error("peer is overloaded (HTTP {status}): {detail}")]
    Overloaded { status: u16, detail: String },

    /// A request exceeded the peer's declared input ceiling.
    ///
    /// Never retryable, and retrying is the tempting mistake: a 413 looks like
    /// a server-side limit, and the obvious "try again" makes a loop that can
    /// never succeed. Shrinking the prompt is the only fix.
    #[error("prompt exceeds the server's limit (HTTP {status}): {detail}")]
    PromptTooLong { status: u16, detail: String },

    /// An inner error annotated with what was being attempted.
    ///
    /// The single `map_err` in the codebase lives in [`Context::ctx`]; this
    /// variant is its result.
    #[error("{msg}: {source}")]
    Context { msg: String, source: Box<Error> },
}

/// Record a best-effort operation that failed, and carry on.
///
/// # The rule this exists to satisfy
///
/// Nothing is swallowed. An error the caller has *decided* is optional still
/// has to be written down somewhere, because "we tried to clean up a temporary
/// and failed" and "we never tried" are different states of the world, and only
/// one of them leaves a stray file for the next process to trip over.
///
/// `let _ = f()?` discards that distinction, which is why
/// `scripts/audit-bad-patterns.sh` counts it as a swallowed error (section B)
/// with a budget of zero. This function is the sanctioned way to say "yes, and
/// here is what it was".
///
/// It prints rather than accumulating into a report because the alternative —
/// threading a diagnostics sink through every call site — buys a log line that
/// nothing reads. If that changes, this changes with it.
///
/// ```
/// # use omni_core::{Context, Error};
/// # fn demo() -> Result<(), Error> {
/// if let Err(e) = std::fs::remove_file("/tmp/does-not-exist-omni") {
///     omni_core::note_advisory("removing a leftover scratch file", &Error::from(e));
/// }
/// # Ok(())
/// # }
/// ```
pub fn note_advisory(what: &str, err: &Error) {
    eprintln!("omni: advisory: {what}: {err}");
}

/// Attach a context message to any error that converts into [`Error`], turning
///
/// ```text
/// foo().ctx(|| format!("reading {path}"))?
/// ```
///
/// into a [`Error::Context`] that names the operation and preserves the cause
/// as `source`. Encapsulating the `map_err` here is what keeps error-mapping
/// closures out of every call site in the workspace.
pub trait Context<T> {
    fn ctx(self, f: impl FnOnce() -> String) -> Result<T, Error>;
}

impl<T, E: Into<Error>> Context<T> for Result<T, E> {
    fn ctx(self, f: impl FnOnce() -> String) -> Result<T, Error> {
        self.map_err(|e| Error::Context { msg: f(), source: Box::new(e.into()) })
    }
}

impl Error {
    /// Annotate an already-constructed error with the operation that failed.
    ///
    /// [`Context::ctx`] is the `Result`-shaped form and is what a call site
    /// almost always wants. This one exists for the two cases `ctx` cannot
    /// reach: an error that was built with `Error::from` rather than propagated
    /// by `?`, and a branch that returns an `Err` it just constructed.
    #[must_use]
    pub fn annotated(self, f: impl FnOnce() -> String) -> Error {
        Error::Context { msg: f(), source: Box::new(self) }
    }

    /// Walk the `Context` chain down to the root cause, discarding the
    /// annotations.
    ///
    /// Used by callers that log once: the chain is worth printing in full at
    /// the boundary, but a retry loop wants the unannotated error to compare
    /// against, and comparing annotated strings is how a retry loop ends up
    /// retrying two different failures forever.
    pub fn root(&self) -> &Error {
        let mut cur = self;
        while let Error::Context { source, .. } = cur {
            cur = source;
        }
        cur
    }

    /// The full chain, outermost first, as `(annotation, root)` pairs.
    ///
    /// `root` is the final element's error; the annotations are the messages
    /// wrapped along the way.
    pub fn chain(&self) -> (Vec<&str>, &Error) {
        let mut notes = Vec::new();
        let mut cur = self;
        while let Error::Context { msg, source } = cur {
            notes.push(msg.as_str());
            cur = source;
        }
        notes.reverse();
        (notes, cur)
    }

    /// True when the failure is worth retrying.
    ///
    /// Deliberately conservative and deliberately narrow. Three sources, and no
    /// more:
    ///
    /// - I/O that is *plausibly* transient (`Interrupted`, `TimedOut`,
    ///   `WouldBlock`, a reset or broken pipe);
    /// - [`Error::Overloaded`], which is the peer stating its own load, so the
    ///   decision is made from the wire rather than inferred from a status code
    ///   the harness made up;
    /// - [`Error::Context`] chains wrapping either of the above, since
    ///   `retryable` looks through the annotations.
    ///
    /// Notably **not** retryable: a version mismatch (the peer will still speak
    /// it next time), a format error (it will still be malformed), a hook that
    /// blew its instruction budget (it will blow it again), and
    /// [`Error::PromptTooLong`] (a bigger prompt is not a shorter one).
    pub fn retryable(&self) -> bool {
        match self.root() {
            Error::Overloaded { .. } => true,
            Error::Io(e) => matches!(
                e.kind(),
                std::io::ErrorKind::Interrupted
                    | std::io::ErrorKind::TimedOut
                    | std::io::ErrorKind::WouldBlock
                    | std::io::ErrorKind::ConnectionReset
                    | std::io::ErrorKind::BrokenPipe
            ),
            _ => false,
        }
    }
}