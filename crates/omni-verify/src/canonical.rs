//! `canonical` — a value's identity is a function of its content.
//!
//! The reward cache, the Merkle-chained audit log and the task-identity
//! manifests are all keyed on "the hash of this value". If the canonical form
//! is not stable, none of them mean anything, and none of them fail: they
//! quietly miss. This group is the check that they mean something.

use omni_core::canonical::{f, obj, s, value_id};
use serde_json::Value;

use crate::{Certificate, cert, exact, failed, with_cases};

/// Run the `canonical` group.
#[must_use]
pub fn run() -> Vec<Certificate> {
    vec![
        key_order_is_not_identity(),
        array_order_is_identity(),
        non_finite_is_refused(),
        duplicate_keys_are_refused(),
        nested_objects_are_sorted(),
    ]
}

fn key_order_is_not_identity() -> Certificate {
    // Two JSON documents that differ only in member order are the same value.
    // If they hashed differently, every reward fingerprint built from an
    // object literal would be sensitive to the order somebody happened to type
    // the fields in — and the cache would miss on every call while reporting
    // nothing wrong.
    const PAIRS: &[(&str, &str)] = &[
        (r#"{"b":1,"a":2}"#, r#"{"a":2,"b":1}"#),
        (r#"{"z":0,"m":{"y":1,"x":2}}"#, r#"{"m":{"x":2,"y":1},"z":0}"#),
        (r#"{"k":[3,1,2]}"#, r#"{"k":[3,1,2]}"#),
    ];
    let mut agreed = 0usize;
    for (a, b) in PAIRS {
        let (va, vb) = match (serde_json::from_str::<Value>(a), serde_json::from_str::<Value>(b)) {
            (Ok(x), Ok(y)) => (x, y),
            _ => {
                eprintln!("omni-verify: canonical: fixture {a} / {b} does not parse");
                continue;
            }
        };
        match (value_id(&va), value_id(&vb)) {
            (Ok(ia), Ok(ib)) if ia == ib => agreed += 1,
            _ => eprintln!("omni-verify: canonical: key order changed identity for {a} vs {b}"),
        }
    }
    with_cases(
        cert(
            "canonical",
            "key_order_is_not_identity",
            "two JSON documents differing only in member order canonicalize \
             identically, so a fingerprint does not depend on the order a \
             caller happened to build the object in",
            exact(agreed == PAIRS.len()),
            0.0,
        ),
        PAIRS.len(),
    )
}

fn array_order_is_identity() -> Certificate {
    // The deliberate opposite of the object rule. An array is a sequence, and a
    // harness that treats `[a,b]` and `[b,a]` as the same knob value has a bug
    // that a canonicalization scheme is exactly where it would hide.
    //
    // The second case is a control in the *other* direction: two empty arrays
    // must agree. Without it this check would also pass under a scheme that
    // called every array distinct, which certifies nothing.
    let cases: [(&str, &str, bool); 3] = [
        ("[1,2]", "[2,1]", true),
        ("[]", "[]", false),
        (r#"["a","b"]"#, r#"["b","a"]"#, true),
    ];
    let mut correct = 0usize;
    for (a, b, should_differ) in cases {
        let (Ok(va), Ok(vb)) = (serde_json::from_str::<Value>(a), serde_json::from_str::<Value>(b))
        else {
            eprintln!("omni-verify: canonical: array fixture {a} does not parse");
            continue;
        };
        let (Ok(ia), Ok(ib)) = (value_id(&va), value_id(&vb)) else {
            eprintln!("omni-verify: canonical: array fixture {a} did not hash");
            continue;
        };
        if (ia != ib) == should_differ {
            correct += 1;
        } else {
            eprintln!(
                "omni-verify: canonical: {a} vs {b} should{} differ, but they {}",
                if should_differ { "" } else { " not" },
                if ia == ib { "agreed" } else { "differed" }
            );
        }
    }
    with_cases(
        cert(
            "canonical",
            "array_order_is_identity",
            "array element order changes a value's identity while equal arrays \
             agree, so the check cannot pass under a scheme that merely called \
             every array distinct",
            exact(correct == cases.len()),
            0.0,
        ),
        cases.len(),
    )
}

fn non_finite_is_refused() -> Certificate {
    // `serde_json` maps NaN and the infinities to `null` inside
    // `From<f64> for Value`, with no error and no warning. So the real gate is
    // the *checked constructor*, which refuses at the point where the caller
    // still has a stack frame naming the offending knob.
    //
    // What must not happen is a non-finite knob landing in a fingerprint as a
    // null and hashing identically to an absent knob: that pairs a baseline
    // with a run differing in exactly one field while sharing an identity, and
    // produces no error while doing it.
    const PROBES: [f64; 3] = [f64::NAN, f64::INFINITY, f64::NEG_INFINITY];

    let Ok(null_id) = value_id(&Value::Null) else {
        return cert(
            "canonical",
            "non_finite_is_refused",
            "a non-finite number cannot enter a fingerprint",
            failed("canonical", "null did not hash"),
            0.0,
        );
    };

    let mut satisfied = 0usize;
    for v in PROBES {
        match f(v) {
            Err(_) => satisfied += 1,
            Ok(built) => match value_id(&built) {
                Ok(id) if id != null_id => satisfied += 1,
                _ => eprintln!("omni-verify: canonical: {v} was admitted and hashes as null"),
            },
        }
    }

    // Positive control: the guard must not be a blanket refusal. A finite float
    // builds and hashes distinctly, or this check would also pass against a
    // constructor that rejected every float.
    match f(0.5) {
        Ok(built) => match value_id(&built) {
            Ok(id) if id != null_id => satisfied += 1,
            _ => eprintln!("omni-verify: canonical: a finite float was not given its own identity"),
        },
        Err(_) => eprintln!("omni-verify: canonical: the constructor refuses every float"),
    }

    with_cases(
        cert(
            "canonical",
            "non_finite_is_refused",
            "NaN and the infinities are refused at construction and never hash \
             as `null`, while a finite float still builds — so a corrupted \
             value cannot masquerade as an absent one and the guard is not a \
             blanket refusal",
            exact(satisfied == PROBES.len() + 1),
            0.0,
        ),
        PROBES.len() + 1,
    )
}

fn duplicate_keys_are_refused() -> Certificate {
    // A duplicate key in a builder is last-wins in every builder that accepts
    // one. Forgetting one knob out of a reward fingerprint therefore produces
    // a fingerprint that is quietly missing a knob — which is exactly the
    // "matched baseline that is not matched" failure this workspace exists to
    // prevent, and it produces no error at all.
    match obj([("knob", serde_json::Value::from(1u64)), ("knob", serde_json::Value::from(2u64))]) {
        Ok(_) => cert(
            "canonical",
            "duplicate_keys_are_refused",
            "a duplicate key in a canonical object builder is refused",
            1.0,
            0.0,
        ),
        Err(_) => cert(
            "canonical",
            "duplicate_keys_are_refused",
            "a duplicate key in a canonical object builder is refused",
            0.0,
            0.0,
        ),
    }
}

fn nested_objects_are_sorted() -> Certificate {
    // The object rule has to recurse. A scheme that sorts only the top level
    // produces a stable id for a flat object and an unstable one for a nested
    // one, and reward fingerprints are nested by construction.
    let inner_a = obj([("temp", Value::from(0.0)), ("depth", Value::from(1u64))]);
    let inner_b = obj([("depth", Value::from(1u64)), ("temp", Value::from(0.0))]);

    let residual = match (inner_a, inner_b) {
        (Ok(a), Ok(b)) => {
            let outer_a = obj([("target", s("example.test")), ("knobs", a)]);
            let outer_b = obj([("knobs", b), ("target", s("example.test"))]);
            match (outer_a, outer_b) {
                (Ok(x), Ok(y)) => match (value_id(&x), value_id(&y)) {
                    (Ok(ix), Ok(iy)) => exact(ix == iy),
                    _ => failed("canonical", "nested fixture did not hash"),
                },
                _ => failed("canonical", "nested fixture failed to build"),
            }
        }
        _ => failed("canonical", "nested fixture failed to build"),
    };

    with_cases(
        cert(
            "canonical",
            "nested_objects_are_sorted",
            "object member order is normalized at every depth, so a nested \
             reward fingerprint is as stable as a flat one",
            residual,
            0.0,
        ),
        1,
    )
}

#[cfg(test)]
mod tests {
    use crate::run_group;

    #[test]
    fn the_group_runs_and_passes() -> Result<(), String> {
        let report = run_group("canonical").map_err(|e| e.to_string())?;
        assert!(report.passed(), "{}", report.render());
        Ok(())
    }
}