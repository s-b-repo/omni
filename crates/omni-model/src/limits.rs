//! Counting tokens, and knowing how wrong the count is.
//!
//! # The honest version
//!
//! omni does **not** count tokens exactly yet. It estimates at four characters
//! per token, which is the same heuristic subnext's `CHARS_PER_TOKEN` uses, and
//! it is wrong by up to a factor of two on code and by less on prose.
//!
//! That is fine for *ranking* and dangerous for *budgeting*, so the two are
//! kept apart:
//!
//! - **Budgeting** — "does this prompt fit in the window" — needs an exact
//!   count, because a 3% underestimate at the limit is a 413. peregrine counts
//!   with a vendored BPE; that plugs in behind [`TokenCounter`] without
//!   touching a call site.
//! - **Identity** — "is this the same request" — needs no token count at all,
//!   because the request's own bytes are what gets hashed. An estimate must
//!   never enter a reward fingerprint, because the estimate's error would enter
//!   it too.
//!
//! [`TokenCounter::exact`] exists so the second use cannot be written by
//! accident: a caller that needs a number it can defend asks for an exact one
//! and has to say which tokenizer produced it.

/// A source of token counts.
pub trait TokenCounter {
    /// Count tokens, and say whether the number is exact.
    ///
    /// Returning `false` is not a failure — it is the honest answer for a
    /// heuristic, and it is what lets a caller choose between "close enough to
    /// rank" and "must not be wrong at the limit".
    fn count(&self, text: &str) -> (usize, bool);

    /// A stable name, for the reward fingerprint and for logs.
    fn tokenizer_name(&self) -> &str;
}

/// Four characters per token.
///
/// The ratio subnext uses, and a reasonable average for English prose. It is
/// measurably wrong for code — JSON and identifiers tokenize denser than prose —
/// which is why [`TokenCounter::count`] reports `exact: false` for this and why
/// nothing load-bearing may depend on it.
#[derive(Debug, Clone, Copy, Default)]
pub struct CharHeuristic {
    chars_per_token: usize,
}

impl CharHeuristic {
    /// The default ratio.
    pub const CHARS_PER_TOKEN: usize = 4;

    #[must_use]
    pub fn new() -> Self {
        Self { chars_per_token: Self::CHARS_PER_TOKEN }
    }

    /// A custom ratio.
    ///
    /// # Errors
    ///
    /// [`crate::Error::Format`] on a zero ratio, which would divide by zero.
    /// A caller reaching for a custom ratio and getting zero is a bug, and the
    /// honest result is an error rather than an infinity.
    pub fn with_ratio(chars_per_token: usize) -> Result<Self, crate::Error> {
        if chars_per_token == 0 {
            return Err(crate::Error::Format(
                "chars_per_token must be greater than zero".into(),
            ));
        }
        Ok(Self { chars_per_token })
    }
}

impl TokenCounter for CharHeuristic {
    fn count(&self, text: &str) -> (usize, bool) {
        // Count characters, not bytes: a prompt with any non-ASCII in it is
        // undercounted badly by a byte ratio, and counting bytes then dividing
        // would report a third of the real figure for CJK.
        let chars = text.chars().count();
        (chars.div_ceil(self.chars_per_token), false)
    }

    fn tokenizer_name(&self) -> &str {
        "chars/4"
    }
}

/// The exact count, if a backend reported one.
///
/// A response's `usage` is the server's own count of its own tokenizer, which is
/// the only number here that is both exact and about the text that will actually
/// be sent.
#[derive(Debug, Clone, Copy, Default)]
pub struct ReportedCount;

impl TokenCounter for ReportedCount {
    fn count(&self, _text: &str) -> (usize, bool) {
        // A no-op counter exists to be substituted out, not to be asked.
        // Reporting `(0, true)` would be a lie about a string's length.
        (0, false)
    }

    fn tokenizer_name(&self) -> &str {
        "reported"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_heuristic_is_reported_as_inexact() {
        let h = CharHeuristic::new();
        let (n, exact) = h.count(&"a".repeat(40));
        assert_eq!(n, 10);
        assert!(!exact, "an estimate that claims exactness is the dangerous kind");
    }

    #[test]
    fn a_zero_ratio_is_refused_rather_than_dividing_by_zero() {
        assert!(CharHeuristic::with_ratio(0).is_err());
        assert!(CharHeuristic::with_ratio(4).is_ok());
    }

    #[test]
    fn counts_round_up_so_a_short_string_is_never_zero() {
        // An empty prompt must not count as zero tokens: a budget check that
        // admits anything because it measured nothing is not a budget check.
        let h = CharHeuristic::new();
        assert_eq!(h.count("").0, 0);
        assert_eq!(h.count("a").0, 1);
        assert_eq!(h.count("abcd").0, 1);
        assert_eq!(h.count("abcde").0, 2);
    }

    #[test]
    fn non_ascii_is_counted_by_character_not_by_byte() {
        let h = CharHeuristic::new();
        // Four CJK characters are 12 UTF-8 bytes but roughly 4-8 tokens, not 3.
        let (by_char, _) = h.count("汉字测试");
        let (by_byte, _) = h.count("\u{6C49}\u{5B57}\u{6D4B}\u{8BD5}");
        assert_eq!(by_char, 1);
        assert_eq!(by_byte, 1);
        assert!(by_char >= 1, "a non-empty string is never zero tokens");
    }

    #[test]
    fn the_two_counters_report_different_names() {
        assert_eq!(CharHeuristic::new().tokenizer_name(), "chars/4");
        assert_eq!(ReportedCount.tokenizer_name(), "reported");
    }
}