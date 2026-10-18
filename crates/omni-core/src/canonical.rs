//! Canonical JSON: one byte string per value, so a hash of a value is a hash of
//! *that value* rather than of whatever key order a serializer happened to emit.
//!
//! # Why this is load-bearing
//!
//! The reward cache is keyed on a hash of the patch plus the full set of knobs
//! that can change a run's output. If the canonical form were not stable, two
//! semantically identical requests would hash differently, the cache would miss
//! on every call, and the suite would look like it was paying for recomputation
//! that no test could explain. Worse, a cache that *sometimes* hits is a cache
//! nobody trusts.
//!
//! So the rules are:
//!
//! 1. Object keys are sorted, lexicographically, as byte strings.
//! 2. Numbers are emitted in a single form; non-finite values are rejected
//!    rather than coerced, because `NaN` has no JSON spelling and a silent
//!    `null` would turn a corrupted reward into a plausible one.
//! 3. No insignificant whitespace.
//!
//! This is deliberately not "whatever `serde_json` does". It is a spec we
//! control, and [`canonicalize`] is the only thing in the workspace allowed to
//! decide what a value hashes as.

use serde_json::{Map, Value};

use crate::error::{Context, Error};

/// Serialize `value` to its canonical JSON text.
///
/// # Why the non-finite check is here *and* in [`f`]
///
/// `serde_json` renders `NaN` and the infinities as `null`. The consequence is
/// that a number and an absent value become indistinguishable to anything that
/// hashes this output. That conversion happens inside `From<f64> for Value`,
/// long before this function is reached, so refusing here alone would defend
/// against a value that cannot arrive while missing the one that can.
///
/// The two checks therefore cover different doors: [`f`] refuses to *build* the
/// value, and this refuses to *emit* one that was built elsewhere — by a
/// `json!` macro, by parsing a number from the wire, or by any future caller
/// that reaches for `Value::from` directly. Both are cheap; keeping only one
/// leaves a hole.
///
/// # Errors
///
/// [`Error::Format`] when the value contains a non-finite number.
pub fn canonicalize(value: &Value) -> Result<String, Error> {
    let mut out = String::new();
    write_value(value, &mut out)?;
    Ok(out)
}

/// The canonical bytes of `value`.
///
/// Hashing the canonical text rather than a re-serialization keeps one
/// definition of identity in the workspace: two values are the same value iff
/// their canonical texts are byte-equal, and everything else — cache keys,
/// Merkle leaves, identity manifests — derives from that single fact.
pub fn canonical_bytes(value: &Value) -> Result<Vec<u8>, Error> {
    canonicalize(value).map(String::into_bytes)
}

fn write_value(value: &Value, out: &mut String) -> Result<(), Error> {
    match value {
        Value::Null => out.push_str("null"),
        Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Value::Number(n) => {
            let f = n.as_f64().unwrap_or(f64::NAN);
            if !f.is_finite() {
                return Err(Error::Format(format!(
                    "cannot canonicalize a non-finite number ({f}); JSON has no \
                     spelling for it and coercing it to null would make a \
                     corrupted value hash as an absent one"
                )));
            }
            out.push_str(&n.to_string());
        }
        Value::String(s) => write_string(s, out),
        Value::Array(items) => {
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                write_value(item, out)?;
            }
            out.push(']');
        }
        Value::Object(map) => {
            // Sorted, because two objects that differ only in insertion order
            // are the same object for every purpose the workspace hashes one
            // for. `Map` iteration order is not stable across construction
            // paths, so sorting is doing real work rather than restating an
            // accident.
            let mut keys: Vec<&String> = map.keys().collect();
            keys.sort();
            out.push('{');
            for (i, key) in keys.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                write_string(key, out);
                out.push(':');
                let Some(child) = map.get(*key) else {
                    // Unreachable for a well-formed map, but a missing key here
                    // would silently drop a field from a hash — which is the
                    // failure this whole module exists to prevent. Fail loudly
                    // rather than emit a shorter object.
                    return Err(Error::Format(format!("canonicalize: key {key:?} vanished mid-iteration")));
                };
                write_value(child, out)?;
            }
            out.push('}');
        }
    }
    Ok(())
}

fn write_string(s: &str, out: &mut String) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
}

/// Build a [`Value::Object`] from pairs, rejecting duplicates.
///
/// A duplicate key in a map literal is a silent last-wins in every builder that
/// accepts one, and a last-wins in a *hashed identity* means the earlier value
/// contributes nothing while still appearing in the source. Forgetting one knob
/// out of a reward fingerprint would produce exactly the "matched baseline that
/// is not matched" bug this workspace exists to prevent.
///
/// # Errors
///
/// [`Error::Format`] naming the duplicated key.
pub fn obj<const N: usize>(pairs: [(&str, Value); N]) -> Result<Value, Error> {
    let mut map = Map::with_capacity(N);
    for (k, v) in pairs {
        if map.contains_key(k) {
            return Err(Error::Format(format!("duplicate key {k:?} in a canonical object")));
        }
        map.insert(k.to_string(), v);
    }
    Ok(Value::Object(map))
}

/// `Value::String`, for building fingerprints.
pub fn s(v: &str) -> Value {
    Value::String(v.to_string())
}

/// `Value::from(u64)`, for building fingerprints.
pub fn n(v: u64) -> Value {
    Value::from(v)
}

/// `Value::from(f64)`, for building fingerprints — but **checked**.
///
/// `serde_json`'s `From<f64> for Value` maps `NaN` and the infinities to
/// `null`, silently and without a warning. That is the actual shape of this
/// hazard, not a hypothetical: a knob read from the environment as `NaN` would
/// land in a reward fingerprint as `null`, hash identically to an absent knob,
/// and produce a "matched baseline" that differs from its pair in exactly one
/// field while sharing its identity. That is worse than a crash, because it
/// produces no error at all.
///
/// So the conversion that reaches for a non-finite value refuses instead, at
/// the point where the caller still has a stack frame naming the knob.
///
/// # Errors
///
/// [`Error::Format`] naming the value.
pub fn f(v: f64) -> Result<Value, Error> {
    if !v.is_finite() {
        return Err(Error::Format(format!(
            "refusing to build a fingerprint value from the non-finite number \
             {v}; serde_json would have rendered it as null, which hashes \
             identically to an absent field"
        )));
    }
    Ok(Value::from(v))
}

/// Hash of a value's canonical form, as 16 hex characters.
///
/// Uses [`crate::ids::content_id`] so a fingerprint and an object id are the same
/// width and the same alphabet — a log line holding both is readable.
pub fn value_id(value: &Value) -> Result<String, Error> {
    let text = canonicalize(value).ctx(|| "canonicalizing a value for hashing".to_string())?;
    Ok(crate::ids::content_id_str(&text))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parsed(s: &str) -> Result<Value, Error> {
        serde_json::from_str(s).ctx(|| "parsing test json".to_string())
    }

    #[test]
    fn key_order_does_not_change_the_canonical_form() -> Result<(), Error> {
        let a = parsed(r#"{"b":1,"a":2}"#)?;
        let b = parsed(r#"{"a":2,"b":1}"#)?;
        assert_eq!(canonicalize(&a)?, canonicalize(&b)?);
        assert_eq!(value_id(&a)?, value_id(&b)?);
        Ok(())
    }

    #[test]
    fn nested_objects_are_sorted_too() -> Result<(), Error> {
        let a = parsed(r#"{"outer":{"z":1,"a":2}}"#)?;
        let b = parsed(r#"{"outer":{"a":2,"z":1}}"#)?;
        assert_eq!(canonicalize(&a)?, r#"{"outer":{"a":2,"z":1}}"#);
        assert_eq!(value_id(&a)?, value_id(&b)?);
        Ok(())
    }

    #[test]
    fn array_order_does_change_the_canonical_form() -> Result<(), Error> {
        // The opposite of the object rule, on purpose: an array is a sequence,
        // and a harness that treats [a,b] and [b,a] as the same knob value has
        // a bug.
        let a = parsed("[1,2]")?;
        let b = parsed("[2,1]")?;
        assert_ne!(value_id(&a)?, value_id(&b)?);
        Ok(())
    }

    #[test]
    fn a_non_finite_number_never_reaches_the_fingerprint_as_null() -> Result<(), Error> {
        // The hazard, demonstrated: serde_json's obvious conversion turns a
        // non-finite float into a null, and a null hashes identically to an
        // absent knob. A reward fingerprint carrying one would pair a baseline
        // with a patched run that differs in exactly one field while sharing an
        // identity — and produce no error doing it.
        let smuggled = Value::from(f64::NAN);
        assert!(
            smuggled.is_null(),
            "serde_json maps NaN to null; this is the behaviour being defended against"
        );

        // The defence is at construction, where the caller still has a stack
        // frame naming the knob that carried the bad value.
        for v in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let Err(err) = f(v) else {
                return Err(Error::Format(format!("f({v}) must refuse")));
            };
            assert!(err.to_string().contains("non-finite"), "unhelpful: {err}");
        }

        // A finite value still builds, and builds to what was asked for.
        assert_eq!(f(0.5)?, Value::from(0.5));
        Ok(())
    }

    #[test]
    fn a_number_built_by_the_naive_conversion_is_indistinguishable_from_an_absent_field() -> Result<(), Error> {
        // Why the constructor guard exists, stated as a failing property of the
        // status quo rather than as an aspiration. If serde_json ever stopped
        // coercing, this test fails and the guard in `f` becomes redundant
        // rather than silently wrong — which is the direction worth failing in.
        let coerced_null = Value::from(f64::INFINITY);
        let absent = Value::Null;
        assert_eq!(
            value_id(&coerced_null)?,
            value_id(&absent)?,
            "a non-finite knob and an absent knob must not be distinguishable \
             by accident — this is why `f()` refuses to build one"
        );
        Ok(())
    }

    #[test]
    fn strings_are_escaped_so_a_hash_cannot_be_confused_by_delimiters() -> Result<(), Error> {
        let a = Value::String("a\"b".into());
        let b = Value::String("a\\b".into());
        assert_ne!(value_id(&a)?, value_id(&b)?);
        assert_eq!(canonicalize(&Value::String("a\nb".into()))?, r#""a\nb""#);
        Ok(())
    }

    #[test]
    fn a_duplicate_key_is_refused() -> Result<(), Error> {
        let Err(err) = obj([("k", n(1)), ("k", n(2))]) else {
            return Err(Error::Format("a duplicate key must be refused".into()));
        };
        assert!(err.to_string().contains("duplicate"), "unhelpful: {err}");
        Ok(())
    }

    #[test]
    fn objects_with_the_same_fields_hash_the_same_whatever_the_build_order() -> Result<(), Error> {
        let a = obj([("beta", n(2)), ("alpha", s("x"))])?;
        let b = obj([("alpha", s("x")), ("beta", n(2))])?;
        assert_eq!(value_id(&a)?, value_id(&b)?);
        Ok(())
    }
}