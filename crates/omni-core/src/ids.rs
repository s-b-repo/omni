//! Content-derived identifiers and a logical clock.
//!
//! Both exist for the same reason: **a harness that reruns work must be able to
//! name the thing it reran.** A wall-clock timestamp cannot do that — it is not
//! reproducible, and a baseline recorded at 14:03 cannot be paired with a
//! patched run at 14:07 on the evidence that they share a timestamp.
//!
//! Ids are therefore content-derived and the clock is logical. See
//! [`Clock`] for what that costs.

/// FNV-1a, 64-bit, second stream.
///
/// Two independent streams (different offsets) combined into one id, which
/// makes accidental collisions between two different objects require both
/// streams to collide. [`subnext`]'s `ids` module uses the same construction;
/// matching it means an id minted here is comparable with one minted there.
///
/// [`subnext`]: https://github.com/s-b-repo/subnext
const FNV_OFFSET_A: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_OFFSET_B: u64 = 0x9dc5_bb1f_9d4b_2b93;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

fn fnv1a(seed: u64, bytes: &[u8]) -> u64 {
    let mut h = seed;
    for b in bytes {
        h ^= u64::from(*b);
        h = h.wrapping_mul(FNV_PRIME);
    }
    h
}

/// A stable 16-hex-character id for `bytes`.
///
/// "Stable" means: the same bytes always produce the same id, in any process,
/// on any run, with no configuration and no registry. That is the property the
/// reward cache and the task-identity pinning both rest on — an id that were
/// merely *probably* stable would make a cached reward quietly unfalsifiable.
///
/// Sixteen hex characters is exactly 64 bits. Both FNV streams feed a single
/// `u64`: `a` contributes whole, `b` is folded in rotated by half a word so the
/// two streams' bits interleave rather than occupying disjoint ranges (a
/// collision in the low 32 of both would otherwise leave the high 32 of `a`
/// carrying the load alone).
#[must_use]
pub fn content_id(bytes: &[u8]) -> String {
    let a = fnv1a(FNV_OFFSET_A, bytes);
    let b = fnv1a(FNV_OFFSET_B, bytes);
    format!("{:016x}", a ^ b.rotate_left(32))
}

/// [`content_id`] over a string.
#[must_use]
pub fn content_id_str(s: &str) -> String {
    content_id(s.as_bytes())
}

/// A logical clock.
///
/// Advances only when told to, so a transcript's ordering is a fact about the
/// program rather than about how fast the machine happened to be. Two runs of
/// the same episode produce the same sequence numbers, which is what lets a
/// failure packet hash to the same value on both sides of a baseline/patched
/// comparison.
///
/// The trade is deliberate: this clock cannot answer "how long did that take?".
/// Wall time belongs to the telemetry path, where it is recorded but never used
/// as an identity.
#[derive(Debug, Clone, Default)]
pub struct Clock {
    now: u64,
}

impl Clock {
    /// Start at zero.
    #[must_use]
    pub fn new() -> Self {
        Self { now: 0 }
    }

    /// The current logical tick. Two calls without an intervening [`Clock::tick`]
    /// return the same value — that is the point.
    #[must_use]
    pub fn now(&self) -> u64 {
        self.now
    }

    /// Advance one tick and return the new value.
    pub fn tick(&mut self) -> u64 {
        self.now = self.now.saturating_add(1);
        self.now
    }

    /// A short, human-readable stamp for a log line: `L0007`.
    ///
    /// Log lines are read by people, and `L7` reads worse than `L0007` in a
    /// column of them. Nothing parses this — it is display only, and a log
    /// reader that tried to parse it would break at tick 10000.
    #[must_use]
    pub fn stamp(&self) -> String {
        format!("L{:04}", self.now)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_content_id_is_stable_across_calls() {
        assert_eq!(content_id_str("hello"), content_id_str("hello"));
    }

    #[test]
    fn a_content_id_separates_near_identical_inputs() {
        // FNV-1a alone collides here easily; two streams exist precisely so this
        // class of near-miss does not.
        let a = content_id_str("task-1");
        let b = content_id_str("task-2");
        assert_ne!(a, b);
        assert_eq!(a.len(), 16, "16 hex characters is 64 bits");
    }

    #[test]
    fn an_empty_input_still_has_an_id() {
        // Zero bytes must not collapse to the same id as a missing input; a
        // packet with no episodes is a real (if useless) packet.
        assert_ne!(content_id(b""), content_id_str("\0"));
    }

    #[test]
    fn the_clock_only_moves_when_told() {
        let mut c = Clock::new();
        assert_eq!(c.now(), 0);
        assert_eq!(c.now(), 0, "reading twice must not advance");
        assert_eq!(c.tick(), 1);
        assert_eq!(c.now(), 1);
        assert_eq!(c.tick(), 2);
        assert_eq!(c.stamp(), "L0002");
    }

    #[test]
    fn the_clock_saturates_rather_than_wrapping() {
        let mut c = Clock { now: u64::MAX };
        c.tick();
        assert_eq!(c.now(), u64::MAX, "a saturating tick must not wrap to zero");
    }
}