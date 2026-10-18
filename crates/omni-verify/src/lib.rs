//! The numerical certificate suite — one of omni's two gates.
//!
//! # What "certified" means here
//!
//! Nothing in this crate is a proof in the formal sense. Rust is not a proof
//! assistant, and a floating-point implementation of a real-valued identity can
//! at best be *correct to a stated tolerance*. What this crate does instead is
//! make every load-bearing numerical claim in the workspace **falsifiable and
//! continuously checked**:
//!
//! - the claim is written down as a theorem string, in prose, next to the code
//!   that checks it, so a reader can judge whether it is worth checking
//!   independently of whether it currently passes;
//! - the check produces a **residual**: a single number that is zero exactly
//!   when the claim holds;
//! - the residual is compared against a **tolerance chosen from the arithmetic**,
//!   never from whatever the code currently happens to produce.
//!
//! A claim that cannot be reduced to a residual is not listed.
//!
//! # Which claims are load-bearing
//!
//! Load-bearing means: **if this stopped holding, the harness would produce wrong
//! output silently rather than fail.** That is the only bar, and it is what
//! separates a certificate from a unit test. It is also why the group list is
//! short. A bad length, a malformed envelope, a version mismatch — those return
//! an `Err`, and an `Err` is already a test failure. What is left is the class
//! of claim that produces a *plausible wrong answer*:
//!
//! | Group | What is certified |
//! |---|---|
//! | `contract` | the gated table really describes the workspace: every message `omni-core` declares has a row, every row's deployed document is accepted by that row's own gate and refused one version ahead, every row's gate rejects a document whose fields are not the Rust types that row names while still accepting the deployed one, an unknown version and a missing version are rejected identically because `#[serde(default)]` makes them the same input, and channel ids cannot escape the channel directory |
//! | `canonical` | key order does not change a value's identity, array order does, and a non-finite number is refused rather than coerced to `null` |
//! | `atomic` | a concurrent reader sees the whole previous file or the whole new one, never a fragment |
//! | `peregrine` | the dialect reaches the wire unchanged: temperature is sent and defaults to 0.0, `top_k` appears only when enabled, tools are advertised but never forced, fields the server discards are not sent, and a base URL normalizes to exactly one separator — plus the event stream, one certificate per failure mode it handles: read boundaries, keep-alives, the `[DONE]` sentinel, a stream that stops early, a mid-stream error, fragmented tool calls, and the absence of token usage |
//!
//! # Running it
//!
//! ```text
//! cargo run -p omni-verify --release -- verify
//! cargo run -p omni-verify --release -- verify --group contract
//! cargo run -p omni-verify --release -- list
//! ```
//!
//! `verify` exits non-zero the moment any identity the implementation rests on
//! stops holding, so a regression shows up as a named failure rather than as a
//! reward that quietly drifts.

#![forbid(unsafe_code)]
#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

pub mod atomic;
pub mod canonical;
pub mod contract;
pub mod eval;
pub mod peregrine;

use std::fmt;

/// One checked claim.
#[derive(Debug, Clone)]
pub struct Certificate {
    pub group: &'static str,
    pub name: &'static str,
    /// The claim, stated so a reader can judge whether checking it is
    /// worthwhile independently of whether it currently passes.
    pub theorem: &'static str,
    /// Measured deviation from the claim; zero exactly when it holds.
    pub residual: f64,
    /// Largest residual consistent with the claim, given the arithmetic.
    pub tolerance: f64,
    /// How many cases the residual aggregates. A `0.0` residual over 64 cases
    /// says far more than the same number over one, and the distinction is
    /// invisible in a table row without it.
    pub cases: usize,
}

impl Certificate {
    #[must_use]
    pub fn passed(&self) -> bool {
        self.residual.is_finite() && self.residual <= self.tolerance
    }
}

impl fmt::Display for Certificate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}/{}: {} (residual {:e}, tolerance {:e})",
            self.group, self.name, self.theorem, self.residual, self.tolerance
        )
    }
}

/// The outcome of running one or more groups.
#[derive(Debug, Clone, Default)]
pub struct Report {
    pub certificates: Vec<Certificate>,
    /// Groups that could not be run at all. Never empty on success — a group
    /// that failed to load is a failure, not a skip.
    pub errors: Vec<String>,
}

impl Report {
    /// Certificates that did not pass, in group then name order.
    #[must_use]
    pub fn failures(&self) -> Vec<&Certificate> {
        self.certificates.iter().filter(|c| !c.passed()).collect()
    }

    /// True when every certificate passed and nothing errored.
    ///
    /// Named `passed` rather than `ok` on purpose: `ok` on a `Report` reads as
    /// `Result::ok` at every call site, and a gate whose success test is
    /// ambiguous with the stdlib method it shadows is a gate someone will
    /// misread.
    #[must_use]
    pub fn passed(&self) -> bool {
        self.errors.is_empty() && self.failures().is_empty()
    }

    /// A fixed-width table, plus the totals.
    ///
    /// The `cases` column is here for a reason: a residual of `0` aggregated
    /// over one case and the same residual over ten thousand are different
    /// statements, and a reader cannot tell them apart from the number alone.
    #[must_use]
    pub fn render(&self) -> String {
        let mut out = String::new();
        out.push_str(&format!(
            "{:<12} {:<34} {:>10} {:>10} {:>9} {:>7}  {}\n",
            "group", "name", "residual", "tolerance", "cases", "status", "theorem"
        ));
        for c in &self.certificates {
            out.push_str(&format!(
                "{:<12} {:<34} {:>10.3e} {:>10.3e} {:>9} {:>7}  {}\n",
                c.group,
                c.name,
                c.residual,
                c.tolerance,
                c.cases,
                if c.passed() { "ok" } else { "FAIL" },
                c.theorem
            ));
        }
        out.push_str(&format!(
            "\n{} certificates, {} failing, {} group errors\n",
            self.certificates.len(),
            self.failures().len(),
            self.errors.len()
        ));
        for e in &self.errors {
            out.push_str(&format!("group error: {e}\n"));
        }
        out
    }
}

/// The certificate groups, in the order `list` reports them.
pub const GROUPS: &[&str] = &["contract", "canonical", "atomic", "peregrine", "eval"];

/// Run one group.
///
/// Returns `Err` naming the group when it does not exist, because a typo'd
/// `--group` that silently verifies everything is a gate that passes when the
/// thing you wanted to check is not there.
pub fn run_group(group: &str) -> Result<Report, String> {
    let mut report = Report::default();
    match group {
        "contract" => report.certificates.extend(contract::run()),
        "canonical" => report.certificates.extend(canonical::run()),
        "atomic" => report.certificates.extend(atomic::run()),
        "peregrine" => report.certificates.extend(peregrine::run()),
        "eval" => report.certificates.extend(eval::run()),
        other => {
            return Err(format!(
                "unknown certificate group {other:?}; known groups are {GROUPS:?}"
            ))
        }
    }
    Ok(report)
}

/// Run every group, collecting failures rather than stopping at the first.
pub fn run_all() -> Report {
    let mut report = Report::default();
    for g in GROUPS {
        match run_group(g) {
            Ok(r) => report.certificates.extend(r.certificates),
            Err(e) => report.errors.push(e),
        }
    }
    report
}

/// The cheapest checks: the ones that must hold before anything else is
/// trustworthy.
///
/// Called "preflight" because it is meant to run before a long job, not instead
/// of the full suite — it is a smoke alarm, not the fire alarm.
#[must_use]
pub fn preflight() -> Report {
    let mut report = Report::default();
    if let Ok(r) = run_group("contract") {
        report.certificates.extend(r.certificates);
    } else {
        report.errors.push("preflight: contract group unavailable".to_string());
    }
    report
}

/// Build one certificate.
///
/// Every certificate in the workspace goes through here, so `tolerance` is
/// always written next to the claim that justifies it.
#[must_use]
pub fn cert(
    group: &'static str,
    name: &'static str,
    theorem: &'static str,
    residual: f64,
    tolerance: f64,
) -> Certificate {
    Certificate { group, name, theorem, residual, tolerance, cases: 1 }
}

/// Record how many cases a residual aggregates. A free function rather than a
/// method because `with_cases` takes `self` by value, and a by-value `with_*`
/// reads like a builder on a borrowed value.
#[must_use]
pub fn with_cases(c: Certificate, cases: usize) -> Certificate {
    Certificate { cases, ..c }
}

/// The residual of a check that could not run: infinite, so the certificate
/// fails, with the reason on stderr rather than a panic and rather than a skip.
/// A certificate that cannot be evaluated is not a passing certificate.
#[must_use]
pub fn failed(what: &str, err: &str) -> f64 {
    eprintln!("omni-verify: {what}: {err}");
    f64::INFINITY
}

/// Turn an exact boolean claim into a residual: `0.0` when it holds, `1.0` when
/// it does not. Measured this way rather than as "a near-zero" so that a future
/// tolerance cannot quietly absorb a broken identity — an audit failure has to
/// be a number that is clearly out of bounds.
#[must_use]
pub fn exact(holds: bool) -> f64 {
    f64::from(u8::from(!holds))
}

/// `mismatch_count` for `f32`, compared through `to_bits` so that a sign of
/// zero, a NaN payload, or a `-0.0` where `0.0` was required is a failure
/// rather than an equality.
#[must_use]
pub fn mismatch_bits(expected: &[f32], observed: &[f32]) -> f64 {
    if expected.len() != observed.len() {
        return f64::INFINITY;
    }
    let bad = expected
        .iter()
        .zip(observed.iter())
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count();
    bad as f64
}

/// The largest absolute deviation between two `f32` sequences, as `f64`. `inf`
/// when the lengths differ, which is a shape bug and not a numerical one.
#[must_use]
pub fn max_abs_diff(expected: &[f32], observed: &[f32]) -> f64 {
    if expected.len() != observed.len() {
        return f64::INFINITY;
    }
    expected
        .iter()
        .zip(observed.iter())
        .map(|(a, b)| (*a as f64 - *b as f64).abs())
        .fold(0.0, f64::max)
}

/// `f32::EPSILON` as an `f64`, the unit roundoff bound for one f32 operation.
pub const EPS_F32: f64 = f32::EPSILON as f64;

/// A deterministic pseudo-random `f32` in `[-1, 1)`, seeded so every run of the
/// suite sees the same fixture. A certificate whose input changes between runs
/// is a certificate you cannot debug; the xorshift is six lines and has no
/// dependency.
#[must_use]
pub fn rng() -> Rng {
    Rng { state: 0x9E37_79B9_7F4A_7C15 }
}

/// A xorshift64 source. Not cryptographic and not meant to be; it exists so a
/// certificate's fixture is reproducible.
pub struct Rng {
    state: u64,
}

impl Rng {
    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    /// Uniform in `[0, 1)`.
    #[must_use]
    pub fn unit(&mut self) -> f32 {
        ((self.next_u64() >> 11) as f64 / (1u64 << 53) as f64) as f32
    }

    /// Uniform in `[-1, 1)`.
    #[must_use]
    pub fn sym(&mut self) -> f32 {
        self.unit() * 2.0 - 1.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_group_runs_and_passes_on_a_clean_tree() {
        let report = run_all();
        assert!(report.errors.is_empty(), "group errors: {:?}", report.errors);
        assert!(report.passed(), "{}", report.render());
        assert!(
            !report.certificates.is_empty(),
            "an empty suite passes vacuously; it must contain claims"
        );
    }

    #[test]
    fn an_unknown_group_is_an_error_not_an_empty_pass() -> Result<(), String> {
        // The failure this guards: a typo'd --group silently verifies whatever
        // default branch runs, and the gate reports success for a check that
        // never happened.
        match run_group("contractt") {
            Err(msg) => {
                assert!(msg.contains("unknown certificate group"), "unhelpful: {msg}");
                Ok(())
            }
            Ok(r) => Err(format!("a typo'd group name must not produce a report, got {}", r.render())),
        }
    }

    #[test]
    fn a_check_that_cannot_run_is_infinite_residual_not_a_skip() {
        let residual = failed("self-test", "deliberate");
        assert!(!residual.is_finite());
        let c = cert("x", "y", "a claim that could not be evaluated", residual, 0.0);
        assert!(!c.passed(), "an unevaluable certificate must not pass");
    }

    #[test]
    fn a_nan_residual_never_passes() {
        // The subtle one. `NaN <= tolerance` is false, so comparing alone would
        // reject a NaN — but `INFINITY <= INFINITY` is *true*, so comparing
        // alone would also *accept* an infinite residual. `passed()` therefore
        // checks finiteness first rather than relying on the comparison.
        let nan = f64::NAN;
        assert!(nan.is_nan());
        let inf = f64::INFINITY;
        assert!(inf <= inf, "comparison alone would accept this");
        let c = cert("x", "y", "t", inf, inf);
        assert!(!c.passed(), "passed() must reject a non-finite residual");
    }

    #[test]
    fn bitwise_mismatch_catches_signed_zero() {
        assert_eq!(mismatch_bits(&[0.0], &[-0.0]), 1.0, "-0.0 is not 0.0 here");
        assert_eq!(mismatch_bits(&[1.0, 2.0], &[1.0, 2.0]), 0.0);
        assert!(mismatch_bits(&[1.0], &[1.0, 2.0]).is_infinite(), "a length mismatch is a shape bug");
    }

    #[test]
    fn the_rng_is_reproducible_across_instances() {
        let a: Vec<f32> = (0..8).map(|_| rng().unit()).collect();
        // Fresh instances, same seed — a certificate whose fixture moves
        // between runs is one you cannot debug.
        let b: Vec<f32> = (0..8).map(|_| rng().unit()).collect();
        assert_eq!(a, b);
    }
}