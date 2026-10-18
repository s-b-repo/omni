//! `contract` — the version gate, the list of what it gates, and the ids that
//! become file names.
//!
//! These are the rules both sides of every deployed channel state in prose and
//! none of them enforced with a certificate. If one stops holding, omni does not
//! fail: it silently misinterprets a peer, which is the expensive kind of wrong.
//!
//! # The table is the claim
//!
//! [`GATED`] is the one enumeration of what the version gate covers. It is
//! worthless without the property that makes it checkable — every row carries a
//! **wire document** and the **gate that must accept it**, so a row cannot name
//! a message nothing in this workspace can parse. A table of names would look
//! just as complete while certifying nothing, which is the failure this file
//! was last wrong about: `ContextHit` had a struct, a version constant and a
//! `validate()`, and no row; `ContextPut` and `MemoryAppend` had version
//! constants and nothing else — no struct, no gate, and no row.
//!
//! The other half of that claim is [`VERSIONED_MESSAGES`], which lives in
//! `omni-core` and is what omni-core *declares*. `every_declared_message_is_a_gated_row`
//! holds the two together, so the next message added without a row is a failing
//! certificate rather than a quietly shorter table.
//!
//! # Every row's gate is the owner's own code
//!
//! A row's `accept` deserializes the deployed document into the message type
//! that type-owning crate declares and then runs that type's own version gate.
//! Five of the seven types live in `omni-core`; the two file-channel envelopes
//! live in `omni-ctx`, which is why this crate depends on it.
//!
//! Two of the seven used to be checked *structurally* — a required-field list
//! and a version read out of a [`serde_json::Value`] — because `omni-ctx` was
//! not a workspace member and this crate could not name those types. That was a
//! weaker claim than the other five rows made, the code said so, and saying so
//! is what made it a finding rather than a design. It is fixed now: there is no
//! second, private refusal path left in this file, and
//! `gated_rows_parse_into_the_types_they_name` is what keeps it that way.

use omni_core::Error;
use omni_core::contract::{
    CHANNEL_ENVELOPE_VERSION, CONTEXT_HIT_VERSION, CONTEXT_PUT_VERSION, CONTEXT_QUERY_VERSION,
    ContextHit, ContextPut, ContextQuery, HOOK_PACKET_VERSION, HookPacket, MEMORY_APPEND_VERSION,
    MemoryAppend, VERSIONED_MESSAGES, check_version, validate_id,
};
use omni_ctx::channel::{ChannelRequest, ChannelResponse};
use serde_json::Value;

use crate::{Certificate, cert, exact, failed, with_cases};

/// One row of the enumeration.
///
/// A name and a version are not enough on their own. A row naming a message
/// that nothing here can deserialise would make the table look complete while
/// certifying nothing, so each row also carries the document the deployed peer
/// actually writes and the gate that has to accept it — which is the only thing
/// that ties a row to a shape that exists.
struct Gated {
    /// The message name, and the `kind` this build passes to
    /// [`check_version`]: the string that appears in the error, so a reader can
    /// match it to the contract document by hand.
    kind: &'static str,
    /// The version this build speaks.
    supported: u32,
    /// A minimal document of this shape, as it crosses the channel. Every field
    /// the deployed contract names, and nothing else — this is a fixture, not a
    /// corpus.
    sample: &'static str,
    /// The same document with one field of the wrong JSON type, and nothing else
    /// changed. Not a corruption probe: a probe asks whether the gate notices, and
    /// this asks whether the gate is *reading the bytes as the named type* at
    /// all. A gate that counted keys, or compared a raw `version` value, would
    /// accept every one of these — which is exactly what
    /// [`gated_rows_parse_into_the_types_they_name`] needs in order to mean
    /// something.
    malformed: &'static str,
    /// Deserialize a document into the real message type and run that type's
    /// own gate. [`Error`], not a `bool`, because a fixture that does not parse
    /// is a finding and not a missing case.
    accept: fn(&str) -> Result<(), Error>,
}

/// Define a [`Gated::accept`] for a type whose `validate()` is its gate.
///
/// One definition per message type, next to the row that uses it, rather than a
/// generic helper and a second table of type names — the row and the shape it
/// claims cannot then drift apart by editing one and not the other.
macro_rules! gate {
    ($name:ident, $ty:ty) => {
        /// Parse the document into the real type and run its own version gate.
        fn $name(json: &str) -> Result<(), Error> {
            let msg: $ty = serde_json::from_str(json)?;
            msg.validate()
        }
    };
}

gate!(accept_context_query, ContextQuery);
gate!(accept_context_hit, ContextHit);
gate!(accept_context_put, ContextPut);
gate!(accept_memory_append, MemoryAppend);
gate!(accept_hook_packet, HookPacket);

/// `ChannelRequest` — the envelope omni writes.
///
/// Deserializes into [`ChannelRequest`], the type `omni-ctx` actually serialises
/// in `Channel::submit`, then runs the same two checks that write path runs:
/// the envelope's version, and the version of the query nested inside it.
///
/// # The nested query is not decoration
///
/// `ChannelRequest` carries a `query` field, and a v1 envelope wrapping a
/// query from the future is a peer that upgraded one message without the other.
/// The old structural gate looked at the top-level `version` and the presence of
/// the keys `id` and `query`; it would have accepted that document. This gate
/// does not, and the refusal it produces is the one `Channel::submit` produces
/// on the way out.
fn accept_channel_request(json: &str) -> Result<(), Error> {
    let request: ChannelRequest = serde_json::from_str(json)?;
    // The envelope first, so a version problem on the envelope is reported as
    // *this row's* version problem rather than as the inner query's.
    check_version("ChannelRequest", request.version, CHANNEL_ENVELOPE_VERSION)?;
    request.query.validate()
}

/// `ChannelResponse` — the envelope omni reads.
///
/// Deserializes into [`ChannelResponse`] and runs
/// [`ChannelResponse::validate_for`], which is the gate `Channel::read_response`
/// runs on every answer it accepts: the envelope's version, and then **every
/// hit's own** version. A v1 envelope carrying a hit from the future is refused
/// here, which is the class of document the old structural gate accepted.
///
/// # What this does not check, stated rather than implied
///
/// `validate_for` takes the request id the answer must match. This row passes
/// the document's own id, so its id-comparison is satisfied by construction.
/// That is not a hole in the row — it is that the id rule is a *pairing* rule
/// between two documents rather than a property of one, and it is enforced
/// where the two actually meet, in `Channel::read_response(id)`, against the id
/// the request was submitted under. The version half — the half this table is
/// the enumeration of — is fully exercised here.
fn accept_channel_response(json: &str) -> Result<(), Error> {
    let response: ChannelResponse = serde_json::from_str(json)?;
    response.validate_for(&response.id)
}

/// Every known versioned message.
///
/// The list is the point. A version gate is only auditable if the set of gated
/// messages is itself enumerable — otherwise "did we gate everything?" has no
/// answer, and the certificate below can only ever cover the messages somebody
/// remembered to write down.
///
/// The order follows the deployed `contract.json`, so a reader comparing the two
/// documents by eye is not doing arithmetic first. `HookPacket` is last because
/// it is the only one omni defines itself.
const GATED: &[Gated] = &[
    Gated {
        kind: "ContextQuery",
        supported: CONTEXT_QUERY_VERSION,
        sample: r#"{"version":1,"text":"How do I bypass a file upload restriction in a web application?","k":6,"field":"pentest"}"#,
        malformed: r#"{"version":1,"text":42,"k":6,"field":"pentest"}"#,
        accept: accept_context_query,
    },
    Gated {
        kind: "ContextHit",
        supported: CONTEXT_HIT_VERSION,
        // Trimmed from the answer the deployed responder actually wrote for
        // `ctx-full-1`: same fields, same types, a shorter rendered line.
        sample: r#"{"version":1,"id":"goal_3efeef74fec8","score":1.6645095348358154,"text":"[goal_3efeef74fec8 L2] files and components whose corruption would cause instability","provenance":{"source":"cybercorpus/text/atomic-red-team/atomics/Indexes/index.yaml.txt","path":"cybercorpus/text/atomic-red-team/atomics/Indexes/index.yaml.txt"}}"#,
        malformed: r#"{"version":1,"id":"goal_3efeef74fec8","score":"1.6645095348358154","text":"[goal_3efeef74fec8 L2] files and components whose corruption would cause instability","provenance":{"source":"cybercorpus/text/atomic-red-team/atomics/Indexes/index.yaml.txt","path":"cybercorpus/text/atomic-red-team/atomics/Indexes/index.yaml.txt"}}"#,
        accept: accept_context_hit,
    },
    Gated {
        kind: "ContextPut",
        supported: CONTEXT_PUT_VERSION,
        sample: r#"{"version":1,"id":"doc-1","kind":"document","text":"a corpus document","meta":{"corpus":"cybercorpus"}}"#,
        malformed: r#"{"version":1,"id":7,"kind":"document","text":"a corpus document","meta":{"corpus":"cybercorpus"}}"#,
        accept: accept_context_put,
    },
    Gated {
        kind: "MemoryAppend",
        supported: MEMORY_APPEND_VERSION,
        sample: r#"{"version":1,"payload":{"kind":"chat","turns":42}}"#,
        // The weakest of the seven, and deliberately so rather than quietly: the
        // payload is an opaque `Value` because its shape belongs to the peer, so
        // the version is the only field on this row with a Rust type behind it.
        // That is the right design and it is also the thinnest thing this
        // certificate can say about this row: a document with any JSON payload
        // at all is correctly accepted.
        malformed: r#"{"version":"1","payload":{"kind":"chat","turns":42}}"#,
        accept: accept_memory_append,
    },
    Gated {
        kind: "ChannelRequest",
        supported: CHANNEL_ENVELOPE_VERSION,
        // The deployed request document, byte for byte.
        sample: r#"{"version":1,"id":"ctx-full-1","query":{"version":1,"text":"How do I bypass a file upload restriction in a web application?","k":6,"field":"pentest"}}"#,
        malformed: r#"{"version":1,"id":"ctx-full-1","query":"How do I bypass a file upload restriction in a web application?"}"#,
        accept: accept_channel_request,
    },
    Gated {
        kind: "ChannelResponse",
        supported: CHANNEL_ENVELOPE_VERSION,
        // The deployed answer, with its hit list trimmed: the envelope is what
        // this row is about, and the hits have their own row.
        sample: r#"{"version":1,"id":"ctx-full-1","hits":[]}"#,
        malformed: r#"{"version":1,"id":"ctx-full-1","hits":{"goal_3efeef74fec8":1.0}}"#,
        accept: accept_channel_response,
    },
    Gated {
        kind: "HookPacket",
        supported: HOOK_PACKET_VERSION,
        sample: r#"{"version":1,"profile":"pentest","summary":"recurring loop","episodes":[{"task_id":"t1","score":0.0,"status":"task_limit_reached","turns":12,"transcript":"..."}],"signals":[["episodes",1],["total_turns",12]]}"#,
        malformed: r#"{"version":1,"profile":"pentest","summary":"recurring loop","episodes":1,"signals":[["episodes",1],["total_turns",12]]}"#,
        accept: accept_hook_packet,
    },
];

/// Run the `contract` group.
#[must_use]
pub fn run() -> Vec<Certificate> {
    vec![
        version_rejects_unknown(),
        missing_version_is_an_unknown_version(),
        every_gated_message_is_at_version_one(),
        gated_rows_describe_real_messages(),
        gated_rows_parse_into_the_types_they_name(),
        every_declared_message_is_a_gated_row(),
        ids_cannot_escape_the_channel_directory(),
        ids_accept_the_deployed_responders_alphabets(),
    ]
}

/// The sample with its `version` field removed: the "the peer dropped a field"
/// path, as a document.
fn without_version(sample: &str) -> Result<String, Error> {
    let mut value: Value = serde_json::from_str(sample)?;
    let object = value.as_object_mut().ok_or_else(|| {
        Error::Format("the deployed document is not a JSON object, so it has no version to remove".into())
    })?;
    object.remove("version");
    Ok(value.to_string())
}

/// The sample with `version` set to `n`: the "the peer sent this number" path.
fn with_version(sample: &str, n: u32) -> Result<String, Error> {
    let mut value: Value = serde_json::from_str(sample)?;
    let object = value.as_object_mut().ok_or_else(|| {
        Error::Format("the deployed document is not a JSON object, so it has no version to set".into())
    })?;
    object.insert("version".to_string(), Value::from(n));
    Ok(value.to_string())
}

/// Whether this row's own gate refused `json` with [`Error::ContractVersion`]
/// naming this row's message.
///
/// `Ok` counts as a refusal check failing, never as a skip: a gate that accepts
/// a document from the future is the defect, and a check that could not be
/// evaluated reports [`failed`] rather than passing quietly.
fn refused_as_version_error(row: &Gated, json: &str) -> bool {
    match (row.accept)(json) {
        Err(Error::ContractVersion { kind, .. }) => kind == row.kind,
        Err(other) => {
            eprintln!("omni-verify: contract: {} refused a bad version for the wrong reason: {other}", row.kind);
            false
        }
        Ok(()) => false,
    }
}

fn version_rejects_unknown() -> Certificate {
    // Every gated message, probed at a version one ahead of what it speaks.
    // A gate that only rejects `0` is a gate that lets a *future* peer through,
    // and a future peer is precisely the case the rule exists for.
    let mut rejected = 0usize;
    for row in GATED {
        if check_version(row.kind, row.supported + 1, row.supported).is_err() {
            rejected += 1;
        } else {
            eprintln!("omni-verify: contract: {} accepted version {} it does not speak", row.kind, row.supported + 1);
        }
    }
    with_cases(
        cert(
            "contract",
            "unknown_version_rejected",
            "a message whose version is one ahead of this build's is refused, \
             for every versioned message the workspace speaks",
            exact(rejected == GATED.len()),
            0.0,
        ),
        GATED.len(),
    )
}

fn missing_version_is_an_unknown_version() -> Certificate {
    // `#[serde(default)]` makes an absent `version` field deserialize to 0.
    // So the "field omitted" path and the "field is zero" path are one path,
    // and a peer that drops a field is rejected for the same stated reason as a
    // peer speaking a newer dialect. If these ever diverge, one of them is
    // silently interpreting garbage.
    //
    // Both paths are checked for **every** gated message rather than for one
    // type, because the omission is a per-type decision: a message that forgot
    // `#[serde(default)]` on its version would turn the omitted path into a
    // parse error, which is loud, so the claim worth making is narrower and
    // stronger — both paths reach the *same named refusal* on each row.
    //
    // The other direction is `gated_rows_describe_real_messages`: the same gates
    // must still accept the complete document.
    let mut agreed = 0usize;
    for row in GATED {
        let (omitted, explicit) = match (without_version(row.sample), with_version(row.sample, 0)) {
            (Ok(a), Ok(b)) => (a, b),
            (Err(e), _) | (_, Err(e)) => {
                eprintln!("omni-verify: contract: {} fixture could not be rewritten: {e}", row.kind);
                continue;
            }
        };
        if refused_as_version_error(row, &omitted) && refused_as_version_error(row, &explicit) {
            agreed += 2;
        } else {
            eprintln!(
                "omni-verify: contract: {} does not refuse an omitted and an explicit version 0 the same way",
                row.kind
            );
        }
    }
    with_cases(
        cert(
            "contract",
            "missing_version_is_unknown_version",
            "for every gated message an omitted `version` and an explicit 0 are \
             both refused as an unknown version naming that message, so a peer \
             that drops a field cannot be mistaken for a peer speaking a newer \
             dialect",
            exact(agreed == 2 * GATED.len()),
            0.0,
        ),
        2 * GATED.len(),
    )
}

fn every_gated_message_is_at_version_one() -> Certificate {
    // omni joins five systems by versioned message. If this drifts above 1
    // without the deployed peers moving, omni stops being able to talk to them
    // — and it will fail at runtime, in a channel, under load. Checking it here
    // makes the drift a build failure instead.
    let at_one = GATED.iter().filter(|row| row.supported == 1).count();
    let residual = if at_one == GATED.len() { 0.0 } else { failed("contract", "a versioned message left version 1") };
    with_cases(
        cert(
            "contract",
            "deployed_contracts_are_version_one",
            "every versioned message omni speaks is at version 1, matching the \
             contract already deployed on both sides of the interop channel",
            residual,
            0.0,
        ),
        GATED.len(),
    )
}

fn gated_rows_describe_real_messages() -> Certificate {
    // Two cases per row, and the second is the control. A row whose sample was
    // merely *accepted* would also be satisfied by a gate that accepts
    // everything, and by a sample whose version nobody checks. So each row's own
    // gate must also refuse the same document one version ahead: acceptance and
    // refusal, per row, from the same function — the only way to satisfy both is
    // a gate that is reading the version rather than the bytes.
    let residual = if GATED.is_empty() {
        failed("contract", "the gated table is empty, so it describes no messages")
    } else {
        let mut correct = 0usize;
        for row in GATED {
            let accepted = match (row.accept)(row.sample) {
                Ok(()) => true,
                Err(e) => {
                    eprintln!("omni-verify: contract: {} refused its own deployed document: {e}", row.kind);
                    false
                }
            };
            let refused_next = match with_version(row.sample, row.supported + 1) {
                Ok(ahead) => (row.accept)(&ahead).is_err(),
                Err(e) => {
                    eprintln!("omni-verify: contract: {} fixture could not be moved forward: {e}", row.kind);
                    false
                }
            };
            if accepted && refused_next {
                correct += 2;
            } else {
                eprintln!(
                    "omni-verify: contract: {} did not both accept the deployed document and refuse the next version",
                    row.kind
                );
            }
        }
        exact(correct == 2 * GATED.len())
    };
    with_cases(
        cert(
            "contract",
            "gated_rows_describe_real_messages",
            "every row of the gated table is accepted by the gate that row names, \
             at the version the deployed peer writes, and refused by that same \
             gate one version ahead — so a row cannot pass by naming a message \
             nothing parses, or a gate that accepts everything",
            residual,
            0.0,
        ),
        2 * GATED.len(),
    )
}

/// Whether this row's gate refuses a document that is not this row's shape.
fn refuses_a_malformed_document(row: &Gated) -> bool {
    match (row.accept)(row.malformed) {
        Err(_) => true,
        Ok(()) => {
            eprintln!(
                "omni-verify: contract: {} accepted a document whose fields are not its own types",
                row.kind
            );
            false
        }
    }
}

fn gated_rows_parse_into_the_types_they_name() -> Certificate {
    // What `gated_rows_describe_real_messages` cannot see. That certificate asks
    // whether a row's gate reads the version; this one asks whether it reads the
    // *rest* of the document as the type the row names. A gate that counted keys,
    // or that compared a raw `version` value and stopped there, passes every
    // case there while accepting every document here — which is not a
    // hypothetical: the two file-channel rows used to be exactly such a gate,
    // structurally, because their structs lived in a crate this one could not
    // reach. They are real types now, and this is the certificate that says so
    // instead of leaving it as a comment.
    //
    // Two halves, both required, and either alone would pass degenerately:
    //
    //   - each row's `malformed` document is refused, which a key-counting or
    //     version-comparing gate would wave through;
    //   - CONTROL: each row's own deployed document is still accepted, so the
    //     only gate that satisfies this one is a gate that reads the type
    //     rather than a gate that refuses everything.
    //
    // `MemoryAppend`'s half is the thinnest of the seven and is annotated as
    // such on the row: its payload is an opaque `Value` by design, so the
    // version is the only field on it with a Rust type behind it.
    //
    // The empty-table guard is the same one the version certificate carries, and
    // for the same reason: `0 == 2 * 0` is true, so an empty table would satisfy
    // this residual while describing no messages at all. A check that cannot run
    // reports `failed`, which is infinite, and never passes.
    let residual = if GATED.is_empty() {
        failed("contract", "the gated table is empty, so it binds no rows to no types")
    } else {
        let mut correct = 0usize;
        for row in GATED {
            if refuses_a_malformed_document(row) {
                correct += 1;
            }
            match (row.accept)(row.sample) {
                Ok(()) => correct += 1,
                Err(e) => {
                    eprintln!("omni-verify: contract: {} refused its own deployed document: {e}", row.kind);
                }
            }
        }
        exact(correct == 2 * GATED.len())
    };
    with_cases(
        cert(
            "contract",
            "gated_rows_parse_into_the_types_they_name",
            "every row's gate rejects a document whose fields are not the Rust types \
             that row names, while still accepting that row's deployed document — so a \
             row cannot pass by counting keys, and cannot pass by refusing everything",
            residual,
            0.0,
        ),
        2 * GATED.len(),
    )
}

/// The messages `declared` lists that have no row in [`GATED`].
fn rows_missing_from_gated(declared: &[(&'static str, u32)]) -> Vec<&'static str> {
    let mut missing = Vec::new();
    for (name, _) in declared {
        if !GATED.iter().any(|row| row.kind == *name) {
            missing.push(*name);
        }
    }
    missing
}

/// The messages both sides name, at two different versions.
fn versions_that_disagree(declared: &[(&'static str, u32)]) -> Vec<&'static str> {
    let mut wrong = Vec::new();
    for (name, version) in declared {
        if GATED.iter().any(|row| row.kind == *name && row.supported != *version) {
            wrong.push(*name);
        }
    }
    wrong
}

fn every_declared_message_is_a_gated_row() -> Certificate {
    // The drift this group exists to catch: a message added to omni-core with
    // its own version constant, its own struct and its own `validate`, and
    // never added here. The table still reads as complete; the gate is just
    // covering less than the workspace declares, and nothing else in the
    // workspace notices.
    //
    // Control, in the other direction: the same comparison, pointed at a
    // message omni-core does not declare, must report it missing. Without it
    // this check would also pass against a comparison that always agrees —
    // two empty enumerations disagree about nothing.
    const PROBE: &[(&str, u32)] = &[("ContextQuery", 1), ("NotADeployedMessage", 1)];

    let residual = if GATED.is_empty() || VERSIONED_MESSAGES.is_empty() {
        failed("contract", "one of the two enumerations is empty, so their agreement is not a claim")
    } else {
        let missing = rows_missing_from_gated(VERSIONED_MESSAGES);
        let wrong = versions_that_disagree(VERSIONED_MESSAGES);
        for name in missing.iter().chain(wrong.iter()) {
            eprintln!("omni-verify: contract: {name} is declared by omni-core but is not gated at its version here");
        }
        let found = rows_missing_from_gated(PROBE);
        let control = found.len() == 1 && found[0] == "NotADeployedMessage";
        if !control {
            eprintln!(
                "omni-verify: contract: the coverage comparison reported {found:?} for a planted missing message"
            );
        }
        exact(missing.is_empty() && wrong.is_empty() && control)
    };
    with_cases(
        cert(
            "contract",
            "every_declared_message_is_a_gated_row",
            "every versioned message omni-core declares has a row in this table \
             at the same version, and the comparison that says so still reports a \
             message it was never given — so adding a message without a row is a \
             failing certificate rather than a quietly shorter table",
            residual,
            0.0,
        ),
        VERSIONED_MESSAGES.len() + 1,
    )
}

fn ids_cannot_escape_the_channel_directory() -> Certificate {
    // A channel id becomes a file name. These are the traversals and separators
    // that would let a peer write outside the directory it was given.
    const ATTACKS: &[&str] = &[
        "../evil", "..", ".", "a/b", "a\\b", "/abs", "a b", "a;b", "a\0b", "",
    ];
    let mut refused = 0usize;
    for bad in ATTACKS {
        if validate_id(bad).is_err() {
            refused += 1;
        }
    }
    with_cases(
        cert(
            "contract",
            "ids_cannot_escape_the_channel",
            "a channel id containing a path separator, a null, or a directory \
             reference is refused, because the id becomes a file name in a \
             directory other processes poll",
            exact(refused == ATTACKS.len()),
            0.0,
        ),
        ATTACKS.len(),
    )
}

fn ids_accept_the_deployed_responders_alphabets() -> Certificate {
    // The other half, and the one that bites in the other direction: narrowing
    // the alphabet on one side only makes the two halves disagree about what a
    // legal id is, and a legal id the responder rejects is a request that times
    // out rather than one that errors.
    const ACCEPTED: &[&str] = &[
        "r1",
        "A_b-9",
        "ctx-3336511-1791718557487399007-0",
        "b0001",
    ];
    let mut accepted = 0usize;
    for good in ACCEPTED {
        if validate_id(good).is_ok() {
            accepted += 1;
        }
    }
    with_cases(
        cert(
            "contract",
            "ids_match_the_deployed_alphabet",
            "every id shape the deployed channel uses is accepted, so a legal \
             id is not one side's business to reject",
            exact(accepted == ACCEPTED.len()),
            0.0,
        ),
        ACCEPTED.len(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::run_group;

    #[test]
    fn the_group_runs_and_passes() -> Result<(), String> {
        let report = run_group("contract").map_err(|e| e.to_string())?;
        assert!(report.passed(), "{}", report.render());
        Ok(())
    }

    #[test]
    fn a_hook_packet_built_through_its_constructor_is_valid() -> Result<(), String> {
        let packet = HookPacket::new("pentest", "no recurring pattern", Vec::new());
        packet.validate().map_err(|e| e.to_string())?;
        Ok(())
    }

    /// Every declared message has a row, at the declared version.
    ///
    /// The certificate above states the claim; this pins the table's contents so
    /// a row cannot be renamed without something noticing here.
    #[test]
    fn every_declared_message_has_its_row() -> Result<(), String> {
        for (name, version) in VERSIONED_MESSAGES {
            let row = GATED
                .iter()
                .find(|r| r.kind == *name)
                .ok_or_else(|| format!("{name} is declared by omni-core but has no row in GATED"))?;
            if row.supported != *version {
                return Err(format!("{name} is gated at {} but omni-core declares {version}", row.supported));
            }
        }
        Ok(())
    }

    /// A row that accepts its own document and refuses the next version, or it
    /// is not describing a real message.
    #[test]
    fn every_row_accepts_its_deployed_document_and_refuses_the_next_version() -> Result<(), String> {
        for row in GATED {
            (row.accept)(row.sample).map_err(|e| format!("{} refused its deployed document: {e}", row.kind))?;
            let ahead = with_version(row.sample, row.supported + 1).map_err(|e| e.to_string())?;
            if (row.accept)(&ahead).is_ok() {
                return Err(format!("{} accepted a document one version ahead", row.kind));
            }
        }
        Ok(())
    }

    /// Every row's gate reads the document as the Rust type the row names.
    ///
    /// The certificate states this as a residual; this pins it per row so a
    /// failing row is named in the test output rather than only in a table cell.
    #[test]
    fn every_row_refuses_a_document_that_is_not_its_own_shape() -> Result<(), String> {
        for row in GATED {
            (row.accept)(row.sample).map_err(|e| format!("{} refused its deployed document: {e}", row.kind))?;
            if (row.accept)(row.malformed).is_ok() {
                return Err(format!(
                    "{} accepted a document whose fields are not its own types",
                    row.kind
                ));
            }
        }
        Ok(())
    }

    /// The table names the six deployed messages and omni's own, and nothing
    /// else — the enumeration is the claim, so its size is worth stating.
    #[test]
    fn the_table_covers_the_deployed_contract_and_nothing_more() -> Result<(), String> {
        let names: Vec<&str> = GATED.iter().map(|r| r.kind).collect();
        for expected in [
            "ContextQuery",
            "ContextHit",
            "ContextPut",
            "MemoryAppend",
            "ChannelRequest",
            "ChannelResponse",
            "HookPacket",
        ] {
            if !names.contains(&expected) {
                return Err(format!("{expected} is missing from GATED"));
            }
        }
        if names.len() != 7 {
            return Err(format!("GATED has {} rows, expected 7: {names:?}", names.len()));
        }
        Ok(())
    }
}