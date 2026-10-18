#!/usr/bin/env bash
# verify-cross-repo.sh — omni's cross-repo drift gate.
#
# omni, subnext, unifur and peregrine were written separately and coordinate
# through versioned file-based contracts, never shared memory. That arrangement
# has one failure mode and no default defence: nothing notices when one side
# moves. A message added on one side and not the other is not an error at
# runtime — it is a document that parses on both sides and carries nothing, and
# the channel goes quiet rather than loud.
#
# So this gate exists to make the seams checkable from outside the language
# that has to change. Everything here is a file comparison or a constant
# comparison; nothing requires the other repos to build, to be a workspace
# member, or to know omni exists. That matters twice over:
#
#   1. The repos are not in one workspace. `peregrine-ctx` is a vendored port,
#      `dcr` is reached by absolute path. A drift check that needed a build
#      would need one of those to be true.
#   2. The most likely drift is a path that resolves to the *wrong tree*. There
#      are two subnext checkouts on this box and only one of them has a
#      responder; nothing at compile time can tell you which you got.
#
# Same shape as `audit-bad-patterns.sh` and `verify-certificates.sh` on purpose:
# one command, non-zero on failure, the reason on stderr. A gate whose internals
# you have to remember is a gate you skip.
#
# Usage:
#   verify-cross-repo.sh [--strict] [--self-test] [--quiet]
#
#   --strict      any finding exits non-zero. Without it the run reports and
#                 exits 0, the same ratchet shape as `audit-bad-patterns.sh`:
#                 findings stay visible while the gate is advisory, and the
#                 gate is only a gate when somebody runs it with --strict.
#   --self-test   plant each known-bad condition in a temp tree and assert the
#                 gate still catches it. Also asserts a clean tree passes.
#   --quiet       findings only; no section narrative on stdout.
#
# Exit: 0 clean, 1 a finding (in --strict) or a self-test failure,
#       2 usage error, 3 a check could not run — a missing tool or an input
#       this build cannot read. 3 is deliberately not 0: a check that cannot
#       run is not a passing check (AGENTS.md rule 6).

set -euo pipefail

SCRIPT="$(realpath "${BASH_SOURCE[0]}")"
ROOT="${VCX_ROOT:-$(cd "$(dirname "$SCRIPT")/.." && pwd)}"

# The deployed layout. Every one of these is overridable so `--self-test` can
# point the whole gate at a fixture instead of at the live machine; a real run
# uses these defaults, and a real run is the only run that means anything about
# the world.
SUBNEXT="${VCX_SUBNEXT:-/home/cortix/Downloads/subnext}"
SUBNEXT_STALE="${VCX_SUBNEXT_STALE:-/home/cortix/Downloads/inc/subnext}"
PEREGRINE_CTX="${VCX_PEREGRINE_CTX:-/home/cortix/Downloads/inc/peregrine/crates/peregrine-ctx}"
UNIFUR="${VCX_UNIFUR:-/home/cortix/Downloads/inc/unifur}"
INTEROP_DIR="${VCX_INTEROP_DIR:-/srv/m-sda/unifur/interop}"

OMNI_CTX_MANIFEST="$ROOT/crates/omni-ctx/Cargo.toml"
OMNI_CORE_CONTRACT="$ROOT/crates/omni-core/src/contract.rs"
OMNI_VERIFY_CONTRACT="$ROOT/crates/omni-verify/src/contract.rs"
LEDGER="${VCX_LEDGER:-$ROOT/docs/NEGATIVE-RESULTS.md}"

CONTRACT_JSON="$INTEROP_DIR/contract.json"
SUBNEXT_INTEROP="$SUBNEXT/src/interop.rs"
UNIFUR_INTEROP="$UNIFUR/src/interop.rs"

# Messages omni defines for itself. They are not in the deployed document
# because unifur never heard of them, and requiring them to be would make the
# deployed side uncheckable. This list is the whole of that exemption: an empty
# one would make the deployed-side comparison vacuous, which is why the gate
# asserts every name in it really is gated rather than trusting the list.
OMNI_OWN_MESSAGES="HookPacket"

# The deployed message names and the constant that carries each one's version.
# A name in contract.json that is not in this table is a finding, not a skip:
# it means a message exists on the wire that this gate has no version mapping
# for, and an unmapped message is an unchecked message.
message_const() {
    case "$1" in
        ContextQuery)    printf 'CONTEXT_QUERY_VERSION' ;;
        ContextHit)      printf 'CONTEXT_HIT_VERSION' ;;
        ContextPut)      printf 'CONTEXT_PUT_VERSION' ;;
        MemoryAppend)    printf 'MEMORY_APPEND_VERSION' ;;
        ChannelRequest)  printf 'CHANNEL_ENVELOPE_VERSION' ;;
        ChannelResponse) printf 'CHANNEL_ENVELOPE_VERSION' ;;
        *)               printf '' ;;
    esac
}

# The six messages whose field lists this gate compares across all three
# sources. `Provenance` is excluded on purpose: it is a nested value, not a
# versioned message, and no source declares it in the document.
CONTRACT_MESSAGES="ContextQuery ContextHit ContextPut MemoryAppend ChannelRequest ChannelResponse"

# The port is permitted to differ in exactly these files, and only there:
# `dcr::` appears in exactly two of subnext's files, so the crate-prefix rename
# the port documents (`dcr::` -> `peregrine_ctx::`) can only show up in those
# two. A third differing file is not a rename. A file missing from the port is
# never a rename.
PORT_ALLOWED_DIFFER="lib.rs main.rs"

STRICT=0
QUIET=0
SELFTEST=0

usage() { sed -n '/^# Usage:/,/^# Exit:/p' "$0" | sed 's/^# \?//; $d'; }

while [ "$#" -gt 0 ]; do
    case "$1" in
        --strict)    STRICT=1; shift ;;
        --quiet)     QUIET=1; shift ;;
        --self-test) SELFTEST=1; shift ;;
        -h|--help)   usage; exit 0 ;;
        *) echo "unknown option $1" >&2; usage >&2; exit 2 ;;
    esac
done

# ── Reporting ───────────────────────────────────────────────────────────────
#
# Findings go to stderr as they happen rather than in a block at the end. A gate
# that takes forty seconds and then prints a summary has told you nothing about
# where the forty seconds went, and a gate that fails on three independent
# seams should say so three times, not once.
FINDINGS=0
BLOCKED=0

section() { [ "$QUIET" = "1" ] || printf '\n── %s\n' "$1"; }
note()    { [ "$QUIET" = "1" ] || printf '   · %s\n' "$1"; }

# reason <what was checked> :: <what was found> :: <why it matters>
#
# Every finding carries all three. A gate that reports "mismatch" has made the
# reader do the work the gate exists to do.
reason() {
    printf 'verify-cross-repo: FAIL — %s\n' "$1" >&2
    [ -n "${2:-}" ] && printf 'verify-cross-repo:        what was found: %s\n' "$2" >&2
    [ -n "${3:-}" ] && printf 'verify-cross-repo:        why it matters: %s\n' "$3" >&2
    FINDINGS=$((FINDINGS + 1))
    return 0
}

# cannot <what could not run> :: <why> — a check that could not run is not a
# passing check, so this is a distinct exit code and never folded into a pass.
cannot() {
    printf 'verify-cross-repo: CANNOT RUN — %s\n' "$1" >&2
    [ -n "${2:-}" ] && printf 'verify-cross-repo:        why: %s\n' "$2" >&2
    BLOCKED=$((BLOCKED + 1))
    return 0
}

TMP="$(mktemp -d)"
cleanup() { rm -rf "$TMP"; }
trap cleanup EXIT

# ── Small readers ───────────────────────────────────────────────────────────
#
# These parse *source text*, not compiled artefacts, because the thing that
# drifts is the text: the peer repo is not in this workspace and may not build
# here at all. Every reader is anchored so it can only read what it claims to.

# const_u32 <file> <NAME> — the value of `pub const NAME: u32 = N;`, or empty.
const_u32() {
    # `-q` in sed stops at the first match, so nothing is left writing into a
    # `head` that has already exited. With `set -o pipefail` an EPIPE on the
    # other side of a pipeline is a failure, and a helper that fails when the
    # constant it was asked for is the *first* one is a helper that lies.
    sed -n "s/^[[:space:]]*pub const $2: u32 = \([0-9]\+\);.*/\1/p" "$1" | { read -r v || true; printf '%s' "$v"; } || true
}

# struct_fields <file> <StructName> — the field names of a plain `pub struct`,
# one per line, in declaration order.
#
# Reads declarations rather than the whole file so a field mentioned in a doc
# comment or in a test fixture cannot be mistaken for a field on the wire. The
# terminating `}` is matched at column zero, which is where every struct in both
# repositories ends; matching a closing brace at any indent would stop at the
# first nested block.
struct_fields() {
    awk -v want="$2" '
        $0 ~ ("^pub struct " want " \\{") { inside = 1; next }
        inside && $0 ~ "^\\}" { exit }
        inside {
            rest = $0
            while (match(rest, /pub [A-Za-z_][A-Za-z0-9_]*[ \t]*:/)) {
                f = substr(rest, RSTART + 4, RLENGTH - 5)
                gsub(/[ \t]/, "", f)
                print f
                rest = substr(rest, RSTART + RLENGTH)
            }
        }
    ' "$1"
}

# rust_struct_field_list <file> <Struct> — struct_fields minus the version field.
#
# `version` is excluded because the deployed document does not list it in
# `fields` for any message: it is carried by the envelope, not enumerated as
# payload. Including it on one side and not the others would make every message
# differ by exactly one field.
rust_struct_field_list() {
    struct_fields "$1" "$2" | grep -vx 'version' || true
}

# gated_rows <file> — the GATED table's (kind, version-constant) pairs, one
# `Name CONST` per line.
#
# Anchored to the table declaration so a `kind:` in a doc comment or a fixture
# elsewhere in the file cannot join the table.
gated_rows() {
    sed -n '/^const GATED/,/^];/p' "$1" \
        | grep -oE 'kind: "[A-Za-z][A-Za-z0-9]*",|supported: [A-Z_][A-Z0-9_]*,' \
        | sed -e 's/kind: "//' -e 's/",$//' -e 's/supported: //' -e 's/,$//' \
        | paste - - 2>/dev/null || true
}

# declared_messages <file> — omni-core's VERSIONED_MESSAGES as `Name CONST`.
declared_messages() {
    sed -n '/^pub const VERSIONED_MESSAGES/,/^];/p' "$1" \
        | grep -oE '\("[A-Za-z][A-Za-z0-9]*", [A-Z_][A-Z0-9_]*\)' \
        | sed -e 's/("/ /' -e 's/", / /' -e 's/)//' || true
}

# literal_str <file> <key> — the string on the first line that declares `<key>`
# and carries a string literal. Used for subnext's PROTOCOL and unifur's
# `protocol:` literal, which is a shape difference between the two repos rather
# than a defect between them.
#
# The `.*"` on the same line is load-bearing. A pattern requiring the quote to
# follow the colon immediately reads *nothing* out of
# `pub const PROTOCOL: &str = "..."`, and a reader that reads nothing looks
# exactly like a peer that agrees — which is how this case was caught: the
# self-test planted a renamed protocol and the gate said nothing.
literal_str() {
    # `grep -m1` rather than `head -1`: same reason as const_u32 — a truncated
    # pipe is a non-zero pipeline under pipefail, and this runs against a 94 KB
    # Rust file with several matches in it.
    grep -m1 -E "$2[[:space:]]*:.*\"" "$1" \
        | grep -m1 -oE '"[^"]*"' \
        | sed -e 's/^"//' -e 's/"$//' || true
}

# sorted_unique — stdin, sorted, deduplicated, blanks dropped.
sorted_unique() { sed -e 's/[[:space:]]*$//' -e '/^$/d' | sort -u; }

# only_in <file-a> <file-b> — the lines in a but not in b.
#
# `|| true` on every one of these is not defensiveness, it is `set -e` plus
# `pipefail`: `grep` and `comm` both answer "nothing left" with a non-zero
# status, and an empty result here is the normal, healthy case. Without it the
# gate exits silently at the first clean comparison and reports nothing at all.
only_in()    { sort -u "$1" | comm -23 - <(sort -u "$2") || true; }

# ── 1 · the canonical subnext checkout ───────────────────────────────────────
#
# Two checkouts of subnext exist on this box. The one at /home/cortix/Downloads
# carries `src/interop.rs`; the one under inc/ does not, and is behind upstream.
# Nothing about that is detectable from omni: cargo would happily resolve either,
# a path dependency is a path, and a build of the wrong tree succeeds and
# produces a runtime with no responder in it. The symptom is a channel that
# simply never answers — the worst available kind of quiet, because both
# processes report success for as long as anyone is watching.
check_canonical() {
    section "1 · canonical subnext checkout"

    if [ ! -d "$SUBNEXT" ]; then
        reason "canonical subnext checkout" \
               "$SUBNEXT does not exist" \
               "the ContextPlane has no runtime to sit on, and every check below would be reading a path that is not there"
        return 0
    fi

    if [ ! -f "$SUBNEXT_INTEROP" ]; then
        reason "canonical subnext checkout has the responder" \
               "$SUBNEXT_INTEROP does not exist" \
               "without src/interop.rs this tree has no ChannelRequest poller; a peer writing a .request.json would wait out its timeout"
    else
        note "canonical: $SUBNEXT/src/interop.rs"
    fi

    # The crate name matters as much as the path: a checkout that resolves but
    # does not publish `dcr` is not the crate omni-ctx asks for.
    if [ -f "$SUBNEXT/Cargo.toml" ]; then
        if ! grep -qE '^name[[:space:]]*=[[:space:]]*"dcr"' "$SUBNEXT/Cargo.toml"; then
            reason "canonical subnext checkout publishes the dcr crate" \
                   "$SUBNEXT/Cargo.toml does not declare name = \"dcr\"" \
                   "a path dependency resolves against the package name; the wrong package resolves to nothing usable"
        fi
    else
        reason "canonical subnext checkout is a cargo package" \
               "$SUBNEXT/Cargo.toml does not exist" \
               "a directory with src/interop.rs and no manifest is not the crate the workspace depends on"
    fi

    # The sibling is reported, never failed on. Its being stale is not drift in
    # anything; depending on it is. Printing it every run is the point: the
    # reason it matters is that it looks exactly like the right answer.
    if [ -d "$SUBNEXT_STALE" ]; then
        if [ -f "$SUBNEXT_STALE/src/interop.rs" ]; then
            note "sibling $SUBNEXT_STALE has src/interop.rs (it may no longer be stale; re-check what it is pinned to)"
        else
            note "sibling $SUBNEXT_STALE has NO src/interop.rs — not the responder; nothing may be wired to it"
        fi
    fi

    # And the dependency itself.
    if [ ! -f "$OMNI_CTX_MANIFEST" ]; then
        reason "omni-ctx points at the canonical subnext" \
               "$OMNI_CTX_MANIFEST does not exist" \
               "there is no declared link to either checkout, so the ContextPlane has no pinned runtime at all"
        return 0
    fi

    declared="$(sed -n 's/^dcr[[:space:]]*=[[:space:]]*{[[:space:]]*path[[:space:]]*=[[:space:]]*"\([^"]*\)".*/\1/p' "$OMNI_CTX_MANIFEST" | { read -r v || true; printf '%s' "$v"; })"
    if [ -z "$declared" ]; then
        reason "omni-ctx declares a path dependency on dcr" \
               "$OMNI_CTX_MANIFEST declares no \`dcr = { path = ... }\`" \
               "the runtime is meant to be pinned to one checkout; an unpinned or workspace-inherited dcr can become either one"
        return 0
    fi

    # Cargo accepts an absolute path, so the reader does too; joining blindly
    # would silently produce <dir>//abs/path and resolve somewhere else.
    case "$declared" in
        /*) resolved="$(realpath -m "$declared")" ;;
        *)  resolved="$(realpath -m "$(dirname "$OMNI_CTX_MANIFEST")/$declared")" ;;
    esac
    canonical_real="$(realpath -m "$SUBNEXT")"
    stale_real="$(realpath -m "$SUBNEXT_STALE")"

    note "omni-ctx dcr path: $declared -> $resolved"

    if [ ! -d "$resolved" ]; then
        reason "omni-ctx's declared dcr path resolves" \
               "\"$declared\" does not resolve — it lands on $resolved, which does not exist" \
               "cargo would reject this the first time omni-ctx is built; it has stayed invisible only because omni-ctx is not yet a workspace member"
    elif [ "$resolved" = "$stale_real" ]; then
        reason "omni-ctx depends on the canonical subnext" \
               "declared path resolves to the stale sibling $resolved, which has no src/interop.rs" \
               "the build succeeds and the channel never answers; this is the exact failure the sibling exists to tempt"
    elif [ "$resolved" != "$canonical_real" ]; then
        reason "omni-ctx depends on the canonical subnext" \
               "declared path resolves to $resolved, not the canonical $canonical_real" \
               "two subnext trees exist; the wrong one compiles and then answers nothing"
    fi

    if [ -d "$resolved" ] && [ ! -f "$resolved/src/interop.rs" ]; then
        reason "the tree omni-ctx depends on has the responder" \
               "$resolved/src/interop.rs does not exist" \
               "the dependency resolves, so the build is green, and the runtime has no poller for a .request.json"
    fi
}

# ── 2 · the DCR port has not silently diverged ──────────────────────────────
#
# `peregrine-ctx` is a vendored port of subnext's runtime. Its own documentation
# claims the files are "verbatim apart from the crate prefix and the CLI name",
# and `dcr::` appears in exactly two of subnext's source files, so exactly two
# files are permitted to differ. Everything else differing is the port having
# moved on its own — which is invisible from inside peregrine, because a port
# that drifts is still a port that compiles.
#
# A file *absent* from the port is the sharper version of the same failure: the
# port still builds, it just no longer has, say, the write channel.
check_port() {
    section "2 · DCR port (subnext/src -> peregrine-ctx/src)"

    if [ ! -d "$SUBNEXT/src" ]; then
        cannot "compare the DCR port" "$SUBNEXT/src does not exist"
        return 0
    fi
    if [ ! -d "$PEREGRINE_CTX/src" ]; then
        cannot "compare the DCR port" "$PEREGRINE_CTX/src does not exist"
        return 0
    fi

    # `diff -rq` exits 1 when there is a difference, which is not an error here.
    diff -rq "$SUBNEXT/src" "$PEREGRINE_CTX/src" > "$TMP/port.diff" 2>&1 || true

    local port_missing="" port_extra="" port_illegal="" differed=""
    local n_missing=0 n_extra=0 n_illegal=0 n_diff=0 line verb rest

    while IFS= read -r line; do
        [ -n "$line" ] || continue
        case "$line" in
            Files\ *\ and\ *\ differ)
                rest="${line#Files }"
                verb="${rest%% and *}"
                differed="$differed ${verb##*/}"
                n_diff=$((n_diff + 1))
                ;;
            Only\ in\ *:*)
                rest="${line#Only in }"
                verb="${rest%%:*}"
                case "$verb" in
                    "$SUBNEXT/src")
                        port_missing="$port_missing ${rest#*: }"
                        n_missing=$((n_missing + 1))
                        ;;
                    "$PEREGRINE_CTX/src")
                        port_extra="$port_extra ${rest#*: }"
                        n_extra=$((n_extra + 1))
                        ;;
                    *)
                        # A third directory in the comparison is itself a finding.
                        port_illegal="$port_illegal $line"
                        n_illegal=$((n_illegal + 1))
                        ;;
                esac
                ;;
            *)
                port_illegal="$port_illegal $line"
                n_illegal=$((n_illegal + 1))
                ;;
        esac
    done < "$TMP/port.diff"

    for f in $differed; do
        case " $PORT_ALLOWED_DIFFER " in
            *" $f "*) note "differs (permitted: prefix rename): $f" ;;
            *)          port_illegal="$port_illegal $f"; n_illegal=$((n_illegal + 1)) ;;
        esac
    done

    if [ "$n_missing" -eq 0 ] && [ "$n_extra" -eq 0 ] && [ "$n_illegal" -eq 0 ]; then
        note "port in sync; ${n_diff} file(s) differ, all of them permitted"
        return 0
    fi

    if [ "$n_missing" -gt 0 ]; then
        reason "every upstream file is present in the port" \
               "missing from $PEREGRINE_CTX/src:$port_missing" \
               "the port still compiles without these; it has quietly stopped carrying the behaviour they implement"
    fi
    if [ "$n_extra" -gt 0 ]; then
        reason "the port carries no file upstream does not" \
               "only in $PEREGRINE_CTX/src:$port_extra" \
               "an upstream deletion or rename that the port never took; the two files with the same name are no longer the same file"
    fi
    if [ "$n_illegal" -gt 0 ]; then
        reason "the DCR port differs only where the prefix rename allows" \
               "$port_illegal" \
               "the port's own docs say 'verbatim apart from the crate prefix (dcr:: -> peregrine_ctx::) and the CLI name', and dcr:: appears in only ${PORT_ALLOWED_DIFFER// /, }; a difference anywhere else is the port moving independently of upstream"
    fi
}

# ── 3 · the GATED table, omni-core's declarations, and the deployed document ─
#
# AGENTS.md rule 5 names the GATED table as the enumeration of what the version
# gate covers, and says a message added without a row makes the certificate no
# longer describe the workspace. That is checkable from out here, in three
# directions, and the middle one is the interesting one: the deployed document,
# what omni-core declares, and what omni-verify claims to gate must be the same
# set of deployed messages. A message present in one and absent from another is
# a shape one side will write and the other will not read.
check_gated_table() {
    section "3 · GATED table vs deployed contract vs omni-core"

    if [ ! -f "$CONTRACT_JSON" ]; then
        cannot "read the deployed contract document" "$CONTRACT_JSON does not exist"
        return 0
    fi
    if ! jq -e . "$CONTRACT_JSON" > /dev/null 2>&1; then
        reason "the deployed contract document parses" \
               "$CONTRACT_JSON is not valid JSON" \
               "the document both peers read to learn the contract is unreadable; nothing downstream of it can be trusted either"
        return 0
    fi
    if [ ! -f "$OMNI_CORE_CONTRACT" ]; then
        cannot "read omni-core's contract declarations" "$OMNI_CORE_CONTRACT does not exist"
        return 0
    fi
    if [ ! -f "$OMNI_VERIFY_CONTRACT" ]; then
        cannot "read omni-verify's GATED table" "$OMNI_VERIFY_CONTRACT does not exist"
        return 0
    fi

    jq -r '.messages[]? | .name' "$CONTRACT_JSON" | sorted_unique > "$TMP/doc.names"
    jq -r '.messages[]? | "\(.name)\t\(.version)"' "$CONTRACT_JSON" > "$TMP/doc.versions"
    jq -r '.messages[]? | "\(.name)\t\((.fields // []) | join(","))"' "$CONTRACT_JSON" > "$TMP/doc.fields"

    declared_messages "$OMNI_CORE_CONTRACT" | awk '{print $1}' | sorted_unique > "$TMP/core.names"
    gated_rows "$OMNI_VERIFY_CONTRACT" | awk '{print $1}' | sorted_unique > "$TMP/gated.names"

    # The anti-vacuity floor. Three empty tables would agree perfectly.
    if [ ! -s "$TMP/doc.names" ]; then
        reason "the deployed contract document declares messages" \
               "$CONTRACT_JSON declares none" \
               "every comparison below would pass on three empty sets, which is the shape of a gate that certifies nothing"
        return 0
    fi
    if [ ! -s "$TMP/core.names" ]; then
        reason "omni-core declares versioned messages" \
               "VERSIONED_MESSAGES in $OMNI_CORE_CONTRACT is empty or unreadable" \
               "the enumeration is the other half of the GATED table's claim; empty, it covers nothing"
        return 0
    fi
    if [ ! -s "$TMP/gated.names" ]; then
        reason "omni-verify's GATED table has rows" \
               "GATED in $OMNI_VERIFY_CONTRACT is empty or unreadable" \
               "the table is the enumeration of what the version gate covers; empty, it covers nothing"
        return 0
    fi

    # omni's own messages are exempt from the deployed document by construction.
    printf '%s\n' $OMNI_OWN_MESSAGES | sorted_unique > "$TMP/own.names"
    only_in "$TMP/doc.names" "$TMP/core.names" > "$TMP/doc_not_core"
    only_in "$TMP/core.names" "$TMP/doc.names" > "$TMP/core_not_doc"
    # Strip the omni-own exemption AFTER the set difference, never before it:
    # subtracting first would let an omni-own name that the document happens to
    # adopt silently disappear from the check.
    if [ -s "$TMP/core_not_doc" ]; then
        grep -vxF -f "$TMP/own.names" "$TMP/core_not_doc" > "$TMP/core_not_doc.f" || true
        mv "$TMP/core_not_doc.f" "$TMP/core_not_doc"
    fi

    if [ -s "$TMP/doc_not_core" ]; then
        reason "every message in the deployed contract is declared by omni-core" \
               "declared on the wire, absent from VERSIONED_MESSAGES: $(tr '\n' ' ' < "$TMP/doc_not_core")" \
               "a message the peers exchange that this build has no struct and no version constant for is a message this build would reinterpret by accident"
    fi

    if [ -s "$TMP/core_not_doc" ]; then
        reason "every message omni-core declares as a deployed contract is in the deployed document" \
               "declared in VERSIONED_MESSAGES, absent from contract.json (and not listed as omni's own: ${OMNI_OWN_MESSAGES// /, }): $(tr '\n' ' ' < "$TMP/core_not_doc")" \
               "either the document is stale or omni forked a deployed message; the first is a silently refused peer and the second is two shapes with one name"
    fi

    only_in "$TMP/doc.names" "$TMP/gated.names" > "$TMP/doc_not_gated"
    only_in "$TMP/gated.names" "$TMP/core.names" > "$TMP/gated_not_core"
    only_in "$TMP/core.names" "$TMP/gated.names" > "$TMP/core_not_gated"

    if [ -s "$TMP/doc_not_gated" ]; then
        reason "every deployed message has a row in the GATED table" \
               "in contract.json, no GATED row: $(tr '\n' ' ' < "$TMP/doc_not_gated")" \
               "the version gate has no row to consult for these, so an unknown version on them is not refused by anything in this workspace"
    fi
    if [ -s "$TMP/gated_not_core" ]; then
        reason "every GATED row names a message omni-core declares" \
               "GATED rows with no declaration: $(tr '\n' ' ' < "$TMP/gated_not_core")" \
               "a row that names nothing this build can deserialise makes the table look complete while certifying nothing"
    fi
    if [ -s "$TMP/core_not_gated" ]; then
        reason "every message omni-core declares has a row in the GATED table" \
               "declared in VERSIONED_MESSAGES, no GATED row: $(tr '\n' ' ' < "$TMP/core_not_gated")" \
               "this is AGENTS.md rule 5 stated as a check: a message added to the enumeration and forgotten in the table is a certificate that no longer describes the workspace"
    fi

    # Versions, per message, across all three sources.
    local name const doc_ver core_ver gated_ver
    while IFS=$'\t' read -r name doc_ver; do
        [ -n "$name" ] || continue
        const="$(message_const "$name")"
        if [ -z "$const" ]; then
            reason "every deployed message has a version mapping in this gate" \
                   "contract.json declares $name, which this gate has no version constant for" \
                   "an unmapped message is an unchecked message: this run would have said nothing about its version"
            continue
        fi
        core_ver="$(declared_messages "$OMNI_CORE_CONTRACT" | awk -v n="$name" '$1 == n {print $2; exit}')"
        core_ver="$(const_u32 "$OMNI_CORE_CONTRACT" "$core_ver")"
        gated_ver="$(gated_rows "$OMNI_VERIFY_CONTRACT" | awk -v n="$name" '$1 == n {print $2; exit}')"
        gated_ver="$(const_u32 "$OMNI_CORE_CONTRACT" "$gated_ver")"

        if [ "$core_ver" != "$doc_ver" ] || [ "$gated_ver" != "$doc_ver" ]; then
            reason "$name is at one version everywhere it is declared" \
                   "contract.json says $doc_ver, omni-core's ${const} says ${core_ver:-<absent>}, the GATED row's supported says ${gated_ver:-<absent>}" \
                   "three numbers for one message means one peer will refuse what another writes, and the refusal is the kind that only shows up in a file"
        fi
    done < "$TMP/doc.versions"

    note "deployed messages: $(tr '\n' ' ' < "$TMP/doc.names")"
    note "omni-core declares: $(tr '\n' ' ' < "$TMP/core.names")"
    note "GATED rows: $(tr '\n' ' ' < "$TMP/gated.names")"
}

# ── 4 · the deployed contract document is not stale ──────────────────────────
#
# A version number that has moved on one side and not the other is the quietest
# failure in this family: both processes start, both report success, and the
# only symptom is a peer that stopped answering hours later. So the deployed
# document's own numbers are checked against the constants on both sides of the
# channel, and against the one version they are allowed to be.
check_document_fresh() {
    section "4 · deployed contract document is current"

    if [ ! -f "$CONTRACT_JSON" ]; then
        cannot "check the deployed contract document for staleness" "$CONTRACT_JSON does not exist"
        return 0
    fi
    if [ ! -f "$SUBNEXT_INTEROP" ]; then
        cannot "compare the deployed document against subnext" "$SUBNEXT_INTEROP does not exist"
        return 0
    fi
    if [ ! -f "$UNIFUR_INTEROP" ]; then
        cannot "compare the deployed document against unifur" "$UNIFUR_INTEROP does not exist"
        return 0
    fi

    local doc_version doc_protocol
    doc_version="$(jq -r '.version // "<absent>"' "$CONTRACT_JSON")"
    doc_protocol="$(jq -r '.protocol // "<absent>"' "$CONTRACT_JSON")"

    if [ "$doc_version" != "1" ]; then
        reason "the deployed document's own version is 1" \
               "$CONTRACT_JSON declares version $doc_version" \
               "the document version gates whether a reader will read it at all; a value other than 1 means either it was bumped without the readers or it was edited by hand"
    fi

    local doc_unifur_doc_version
    doc_unifur_doc_version="$(const_u32 "$UNIFUR_INTEROP" CONTRACT_DOC_VERSION)"
    if [ -n "$doc_unifur_doc_version" ] && [ "$doc_unifur_doc_version" != "$doc_version" ]; then
        reason "the deployed document's version matches the writer that generated it" \
               "on disk: $doc_version, unifur's CONTRACT_DOC_VERSION: $doc_unifur_doc_version" \
               "contract.json is generated from that constant; a different number on disk means the file was hand-edited and the next write will silently revert it"
    fi
    # subnext has no CONTRACT_DOC_VERSION: the document is owned by the side that
    # writes it. Said here rather than left to look like an omission.

    # Protocol, three sources: the document, unifur's literal, subnext's constant.
    local subnext_protocol unifur_protocol
    subnext_protocol="$(literal_str "$SUBNEXT_INTEROP" PROTOCOL)"
    unifur_protocol="$(literal_str "$UNIFUR_INTEROP" protocol)"
    if [ "$doc_protocol" != "unifur-subnext-interop" ]; then
        reason "the deployed document declares the deployed protocol" \
               "$CONTRACT_JSON declares protocol \"$doc_protocol\"" \
               "a reader that trusts the document's protocol name is looking for a conversation that is not this one"
    fi
    if [ -n "$subnext_protocol" ] && [ "$subnext_protocol" != "$doc_protocol" ]; then
        reason "subnext records the same protocol name as the deployed document" \
               "document: \"$doc_protocol\", subnext's PROTOCOL: \"$subnext_protocol\"" \
               "the name is how a reader knows the document is about the channel it is holding"
    fi
    if [ -n "$unifur_protocol" ] && [ "$unifur_protocol" != "$doc_protocol" ]; then
        reason "unifur records the same protocol name as the deployed document" \
               "document: \"$doc_protocol\", unifur's contract_doc(): \"$unifur_protocol\"" \
               "unifur writes this file; the constant and the file disagreeing means the next write changes the protocol out from under every reader"
    fi

    # Per-message versions, against both peers' constants.
    local name ver const sub_v unifur_v
    while IFS=$'\t' read -r name ver; do
        [ -n "$name" ] || continue
        const="$(message_const "$name")"
        if [ -z "$const" ]; then
            reason "every deployed message has a version mapping in this gate" \
                   "contract.json declares $name, unmapped here" \
                   "an unmapped message is an unchecked message"
            continue
        fi
        if [ "$ver" != "1" ]; then
            reason "$name is deployed at version 1" \
                   "contract.json declares $name at version $ver" \
                   "AGENTS.md rule 5: implement the deployed shapes, do not redefine them; a version-2 message on the wire is refused by every version-1 peer"
        fi
        sub_v="$(const_u32 "$SUBNEXT_INTEROP" "$const")"
        unifur_v="$(const_u32 "$UNIFUR_INTEROP" "$const")"
        if [ -z "$sub_v" ]; then
            reason "subnext declares a version constant for $name" \
                   "$SUBNEXT_INTEROP defines no $const" \
                   "the responder would accept a document it cannot name a supported version for; the gate is the only thing that would have noticed"
        elif [ "$sub_v" != "$ver" ]; then
            reason "$name's deployed version agrees with subnext's constant" \
                   "document: $ver, subnext's $const: $sub_v" \
                   "one side bumped and the other did not; the peer refuses the message and reports nothing but a timeout"
        fi
        if [ -z "$unifur_v" ]; then
            reason "unifur declares a version constant for $name" \
                   "$UNIFUR_INTEROP defines no $const" \
                   "the side that generates contract.json does not have a constant to generate it from, so the document is maintained by hand"
        elif [ "$unifur_v" != "$ver" ]; then
            reason "$name's deployed version agrees with unifur's constant" \
                   "document: $ver, unifur's $const: $unifur_v" \
                   "unifur's next write regenerates contract.json at a version the responder does not speak"
        fi
    done < "$TMP/doc.versions"

    note "document version $doc_version, protocol \"$doc_protocol\", $(wc -l < "$TMP/doc.versions") messages"
}

# ── 5 · both peers' contract files agree with each other ────────────────────
#
# Field lists, not just names. A responder that writes `{"id":…,"score":…}`
# without `provenance` produces a document that parses on the other side and
# loses its provenance — the document looks fine and the evidence is gone.
#
# Order is deliberately NOT compared. JSON object order carries no meaning to
# either reader, so a reordering is a documentation change, not a drift; the
# gate compares the sets and says so.
check_field_lists() {
    section "5 · unifur's and subnext's message shapes agree"

    if [ ! -f "$UNIFUR_INTEROP" ]; then
        cannot "compare unifur's message shapes" "$UNIFUR_INTEROP does not exist"
        return 0
    fi
    if [ ! -f "$SUBNEXT_INTEROP" ]; then
        cannot "compare subnext's message shapes" "$SUBNEXT_INTEROP does not exist"
        return 0
    fi
    if [ ! -f "$CONTRACT_JSON" ] || ! jq -e . "$CONTRACT_JSON" > /dev/null 2>&1; then
        cannot "compare the message shapes against the deployed document" "$CONTRACT_JSON is missing or unparseable"
        return 0
    fi

    local m
    for m in $CONTRACT_MESSAGES; do
        struct_fields "$UNIFUR_INTEROP" "$m" | { grep -vx 'version' || true; } | sorted_unique > "$TMP/u.$m"
        struct_fields "$SUBNEXT_INTEROP" "$m" | { grep -vx 'version' || true; } | sorted_unique > "$TMP/s.$m"
        awk -F'\t' -v n="$m" '$1 == n {print $2}' "$TMP/doc.fields" \
            | tr ',' '\n' | sorted_unique > "$TMP/d.$m"

        # An empty field list is not "no disagreement", it is no evidence.
        if [ ! -s "$TMP/u.$m" ] || [ ! -s "$TMP/s.$m" ] || [ ! -s "$TMP/d.$m" ]; then
            reason "$m's field list is readable on all three sides" \
                   "unifur: $(tr '\n' ' ' < "$TMP/u.$m" | sed 's/ $//') | subnext: $(tr '\n' ' ' < "$TMP/s.$m" | sed 's/ $//') | document: $(tr '\n' ' ' < "$TMP/d.$m" | sed 's/ $//')" \
                   "one side parsed to nothing, which this gate cannot distinguish from agreeing; a check that cannot read its input is not a check"
            continue
        fi

        if ! diff -q "$TMP/u.$m" "$TMP/s.$m" > /dev/null; then
            reason "$m's field list is the same in unifur and subnext" \
                   "only in unifur: $(only_in "$TMP/u.$m" "$TMP/s.$m" | tr '\n' ' '); only in subnext: $(only_in "$TMP/s.$m" "$TMP/u.$m" | tr '\n' ' ')" \
                   "two repos, two shapes, one name; the document still parses on both sides and one field of it is simply never spoken"
        fi
        if ! diff -q "$TMP/u.$m" "$TMP/d.$m" > /dev/null; then
            reason "$m's field list matches the deployed document" \
                   "only in unifur: $(only_in "$TMP/u.$m" "$TMP/d.$m" | tr '\n' ' '); only in contract.json: $(only_in "$TMP/d.$m" "$TMP/u.$m" | tr '\n' ' ')" \
                   "the document is what a third implementation would build itself from, and it is not what either peer writes"
        fi
        if ! diff -q "$TMP/s.$m" "$TMP/d.$m" > /dev/null; then
            reason "$m's field list matches the deployed document" \
                   "only in subnext: $(only_in "$TMP/s.$m" "$TMP/d.$m" | tr '\n' ' '); only in contract.json: $(only_in "$TMP/d.$m" "$TMP/s.$m" | tr '\n' ' ')" \
                   "the responder is the half that writes; a field the document promises and the responder does not send is a promise no peer can keep"
        fi
    done

    # The write half is an extension subnext added and the document does not
    # declare. That is documented, deliberate, and worth printing every run so
    # its absence is not mistaken for a hole in the check.
    if grep -q 'pub const WRITE_ACK_VERSION' "$SUBNEXT_INTEROP"; then
        note "subnext also implements WRITE_ACK_VERSION ($(const_u32 "$SUBNEXT_INTEROP" WRITE_ACK_VERSION)), which the deployed document does not declare — an extension, not a divergence"
    fi

    note "field lists compared for: $CONTRACT_MESSAGES"
}

# ── 6 · the negative-results ledger's citations have not rotted ──────────────
#
# A closed result is only as good as the line it points at. This checks that
# every cited source file still exists and that every cited line is still inside
# it. It does not check that the line says what the entry claims — that is a
# reading, and a reading is a reviewer's job, not a grep's.
#
# It also checks that every row carries a reopening condition. An entry with an
# empty one is not a closed result; it is a rumour with a citation, and the next
# person to read it has no way to tell the difference.
check_ledger() {
    section "6 · docs/NEGATIVE-RESULTS.md citations"

    if [ ! -f "$LEDGER" ]; then
        reason "the negative-results ledger exists" \
               "$LEDGER does not exist" \
               "without it, a settled experiment is re-run by whoever next thinks of it, and the most expensive habit in this family stays unpriced"
        return 0
    fi

    local rows
    rows="$(grep -cE '^\| N-[0-9]+ \|' "$LEDGER" || true)"
    if [ "$rows" -lt 3 ]; then
        reason "the ledger holds at least three closed results" \
               "it holds $rows row(s) starting with '| N-NN |'" \
               "a ledger this small passes every check below while recording nothing; the citation check must not be satisfiable by an empty table"
        return 0
    fi

    # Row shape and the reopening condition.
    local seen_ids="$TMP/ledger.ids"
    : > "$seen_ids"
    local line id cells reopen src
    while IFS= read -r line; do
        cells="$(printf '%s' "$line" | awk -F'|' '{print NF}')"
        if [ "$cells" != "7" ]; then
            reason "every ledger row is ID | Claim | Result | Source | Reopen-when" \
                   "a row has $cells pipe-separated cells, not 6: $(printf '%s' "$line" | cut -c1-60)..." \
                   "the citation reader and the reopening-condition reader both key off this shape; a row that does not fit is not being read"
            continue
        fi
        id="$(printf '%s' "$line" | awk -F'|' '{gsub(/^[ \t]+|[ \t]+$/, "", $2); print $2}')"
        reopen="$(printf '%s' "$line" | awk -F'|' '{v=$(NF-1); gsub(/^[ \t]+|[ \t]+$/, "", v); print v}')"
        src="$(printf '%s' "$line" | awk -F'|' '{v=$(NF-2); gsub(/^[ \t]+|[ \t]+$/, "", v); print v}')"

        if grep -qxF "$id" "$seen_ids"; then
            reason "ledger ids are unique" \
                   "$id appears more than once" \
                   "two results under one id means a later reader cannot tell which measurement they are looking at"
        fi
        printf '%s\n' "$id" >> "$seen_ids"

        case "$reopen" in
            ""|-|--|—|TBD|tbd|n/a|N/A)
                reason "$id carries a reopening condition" \
                       "its 'reopen only if' cell is \"$reopen\"" \
                       "a closed result without a condition is a rumour with a citation; it closes the question and leaves nothing to check against next time"
                ;;
        esac
        if ! printf '%s' "$src" | grep -qE '`/[^`]+:[0-9]+`'; then
            reason "$id cites a source file and line" \
                   "its source cell is \"$src\"" \
                   "the row states a measurement nobody can go and look at"
        fi
    done < <(grep -E '^\| N-[0-9]+ \|' "$LEDGER")

    # Every citation anywhere in the file, prose included.
    local total=0 rotted=0 cite target lineno flen
    while IFS= read -r cite; do
        total=$((total + 1))
        target="${cite%:*}"
        lineno="${cite##*:}"
        if [ ! -f "$target" ]; then
            reason "every cited source file still exists" \
                   "$target (cited at line $lineno)" \
                   "a citation to a file that is gone is worse than no citation: it reads as evidence"
            rotted=$((rotted + 1))
            continue
        fi
        flen="$(wc -l < "$target")"
        if [ "$lineno" -lt 1 ] || [ "$lineno" -gt "$flen" ]; then
            reason "every cited line is still inside its file" \
                   "$target:$lineno, but the file is $flen lines long" \
                   "the cited line moved or the file was rewritten; the row is now quoting a line number rather than a result"
            rotted=$((rotted + 1))
            continue
        fi
        if [ -z "$(sed -n "${lineno}p" "$target")" ]; then
            reason "every cited line says something" \
                   "$target:$lineno is blank" \
                   "a citation to a blank line cannot be checked by a reader, which is the only reason to cite one"
            rotted=$((rotted + 1))
        fi
    done < <(grep -oE '`/[^`]+:[0-9]+`' "$LEDGER" | tr -d '`' | sort -u)

    if [ "$total" -eq 0 ]; then
        reason "the ledger contains at least one resolvable citation" \
               "no backticked /absolute/path:line was found anywhere in $LEDGER" \
               "the citation checks are satisfied by a ledger that cites nothing, which is a check with no subject"
    else
        note "$rows row(s), $total distinct citation(s) verified against the files on disk"
    fi
}

# ── Self-test ───────────────────────────────────────────────────────────────
#
# "A gate that only ever passes is indistinguishable from one whose check rotted."
# subnext ships the same idea (audit-selftest.sh) and peregrine's audit script
# found a real hole in its own regex this way. Every check here is exercised by
# planting the specific defect it exists to catch and asserting the gate is both
# non-zero AND names that defect — a gate that fails for the wrong reason proves
# nothing.
#
# The fixture is a tree in the state this gate requires, not a copy of the live
# one. The live tree is what the gate reports on; grading the gate against the
# world would only re-run check 1 twice.
#
# The positive control matters as much as the negative ones: a gate that fails
# on everything is exactly as useless as a gate that passes on everything, and
# only a clean fixture can tell those apart.
build_fixture() {
    local fx="$1"
    # The fixture mirrors the real directory *depth*, not just the file names.
    # It has to: the manifest's path dependency is relative, and the thing this
    # gate has to be able to catch is a relative path that is off by one `..`.
    # A flat fixture would resolve any depth correctly and grade nothing.
    mkdir -p "$fx/inc/omni/crates/omni-core/src" "$fx/inc/omni/crates/omni-verify/src" \
             "$fx/inc/omni/crates/omni-ctx" "$fx/inc/omni/docs" \
             "$fx/inc/peregrine/crates/peregrine-ctx" "$fx/inc/unifur/src" \
             "$fx/subnext" "$fx/subnext-stale" "$fx/interop"

    cp -a "$SUBNEXT/src" "$fx/subnext/src"
    # The manifest too: check 1 reads the crate name out of it, and a fixture
    # without one would grade that check on a tree it can never have produced.
    cp "$SUBNEXT/Cargo.toml" "$fx/subnext/Cargo.toml"
    cp "$UNIFUR_INTEROP" "$fx/inc/unifur/src/interop.rs"
    cp "$CONTRACT_JSON" "$fx/interop/contract.json"
    cp "$OMNI_CORE_CONTRACT" "$fx/inc/omni/crates/omni-core/src/contract.rs"
    cp "$OMNI_VERIFY_CONTRACT" "$fx/inc/omni/crates/omni-verify/src/contract.rs"
    # Conditional, not `&&`: the ledger is an input to check 6 and a missing one
    # is a finding that run will report on its own terms.
    if [ -f "$LEDGER" ]; then
        cp "$LEDGER" "$fx/inc/omni/docs/NEGATIVE-RESULTS.md"
    fi

    # The port, in sync: the canonical tree with the documented prefix rename
    # applied to the only two files where `dcr::` occurs. This is what a
    # re-ported peregrine-ctx is supposed to look like byte for byte.
    cp -a "$fx/subnext/src" "$fx/inc/peregrine/crates/peregrine-ctx/src"
    sed -i 's/\bdcr::/peregrine_ctx::/g' \
        "$fx/inc/peregrine/crates/peregrine-ctx/src/lib.rs" \
        "$fx/inc/peregrine/crates/peregrine-ctx/src/main.rs"

    # The stale sibling: the canonical tree with the responder removed. Exactly
    # the trap this gate exists to catch.
    cp -a "$fx/subnext/src" "$fx/subnext-stale/src"
    rm -f "$fx/subnext-stale/src/interop.rs"

    # A relative path, at the depth the real manifest *should* have used: four
    # levels up from crates/omni-ctx to the directory holding both trees.
    printf '[package]\nname = "omni-ctx"\n\n[dependencies]\ndcr = { path = "../../../../subnext" }\n' \
        > "$fx/inc/omni/crates/omni-ctx/Cargo.toml"
}

run_gate_on() {
    # run_gate_on <fixture-root>; sets RG_STATUS and RG_OUTPUT.
    local fx="$1"
    set +e
    RG_OUTPUT="$(VCX_ROOT="$fx/inc/omni" \
        VCX_SUBNEXT="$fx/subnext" \
        VCX_SUBNEXT_STALE="$fx/subnext-stale" \
        VCX_PEREGRINE_CTX="$fx/inc/peregrine/crates/peregrine-ctx" \
        VCX_UNIFUR="$fx/inc/unifur" \
        VCX_INTEROP_DIR="$fx/interop" \
        VCX_LEDGER="$fx/inc/omni/docs/NEGATIVE-RESULTS.md" \
        "$SCRIPT" --strict --quiet 2>&1)"
    RG_STATUS=$?
    set -e
}

self_test() {
    local base="$TMP/selftest"
    local fx
    build_fixture "$base"
    # The pristine copy lives OUTSIDE the fixture directory: a copy inside it
    # would be copied into itself, which `cp -a` refuses and which would look
    # like a self-test failure rather than the scripting error it is.
    local pristine="$TMP/selftest-pristine"
    cp -a "$base" "$pristine"

    local pass=0 fail_count=0
    local failures=""

    grade() {
        # grade <case-name> <expected-substring> <want: catch|clean>
        local name="$1" needle="$2" want="$3"
        if [ "$want" = "catch" ]; then
            if [ "$RG_STATUS" -eq 0 ]; then
                fail_count=$((fail_count + 1)); failures="$failures
  $name: gate exited 0 on a tree that has the defect"
                return 0
            fi
            if ! printf '%s' "$RG_OUTPUT" | grep -qF "$needle"; then
                fail_count=$((fail_count + 1)); failures="$failures
  $name: gate failed, but never named \"$needle\"
$(printf '%s' "$RG_OUTPUT" | sed 's/^/      | /')"
                return 0
            fi
        else
            if [ "$RG_STATUS" -ne 0 ]; then
                fail_count=$((fail_count + 1)); failures="$failures
  $name: gate failed on a tree with no defect planted
$(printf '%s' "$RG_OUTPUT" | sed 's/^/      | /')"
                return 0
            fi
        fi
        pass=$((pass + 1))
    }

    fresh() { fx="$base/case-$1"; rm -rf "$fx"; cp -a "$pristine" "$fx"; }

    # ── positive control ─────────────────────────────────────────────────────
    run_gate_on "$pristine"
    grade "control: an in-sync tree passes" "" clean

    # ── 1 · canonical checkout ───────────────────────────────────────────────
    fresh no-interop; rm -f "$fx/subnext/src/interop.rs"
    run_gate_on "$fx"; grade "canonical src/interop.rs removed" "does not exist" catch

    fresh wrong-name; sed -i 's/^name = "dcr"/name = "dcr_old"/' "$fx/subnext/Cargo.toml"
    run_gate_on "$fx"; grade "canonical tree does not publish dcr" 'name = "dcr"' catch

    fresh points-stale
    printf '[package]\nname = "omni-ctx"\n\n[dependencies]\ndcr = { path = "../../../../subnext-stale" }\n' \
        > "$fx/inc/omni/crates/omni-ctx/Cargo.toml"
    run_gate_on "$fx"; grade "omni-ctx points at the stale sibling" "stale sibling" catch

    fresh path-unresolvable
    printf '[package]\nname = "omni-ctx"\n\n[dependencies]\ndcr = { path = "../../../Downloads/subnext" }\n' \
        > "$fx/inc/omni/crates/omni-ctx/Cargo.toml"
    run_gate_on "$fx"; grade "omni-ctx path does not resolve" "does not resolve" catch

    fresh no-path-dep
    printf '[package]\nname = "omni-ctx"\n\n[dependencies]\ndcr = { workspace = true }\n' \
        > "$fx/inc/omni/crates/omni-ctx/Cargo.toml"
    run_gate_on "$fx"; grade "omni-ctx declares no path for dcr" "no \`dcr = { path = ... }\`" catch

    fresh no-manifest; rm -f "$fx/inc/omni/crates/omni-ctx/Cargo.toml"
    run_gate_on "$fx"; grade "omni-ctx has no manifest at all" "does not exist" catch

    # ── 2 · the port ─────────────────────────────────────────────────────────
    fresh port-third-file; printf '\n// drifted\n' >> "$fx/inc/peregrine/crates/peregrine-ctx/src/indexer.rs"
    run_gate_on "$fx"; grade "port's indexer.rs drifts" "indexer.rs" catch

    fresh port-missing-file; rm -f "$fx/inc/peregrine/crates/peregrine-ctx/src/trust.rs"
    run_gate_on "$fx"; grade "port is missing trust.rs" "trust.rs" catch

    fresh port-extra-file; printf '// local\n' > "$fx/inc/peregrine/crates/peregrine-ctx/src/local_only.rs"
    run_gate_on "$fx"; grade "port has a file upstream does not" "local_only.rs" catch

    fresh port-src-gone; rm -rf "$fx/inc/peregrine/crates/peregrine-ctx/src"
    run_gate_on "$fx"; grade "port has no src/ at all" "CANNOT RUN" catch

    # ── 3 · GATED table vs the document vs omni-core ──────────────────────────
    fresh doc-version-bumped
    jq '.messages |= map(if .name == "MemoryAppend" then .version = 2 else . end)' \
        "$fx/interop/contract.json" > "$fx/interop/c.json" && mv "$fx/interop/c.json" "$fx/interop/contract.json"
    run_gate_on "$fx"; grade "MemoryAppend bumped to 2 in contract.json" "MemoryAppend is at one version everywhere" catch

    fresh doc-message-dropped
    jq 'del(.messages[] | select(.name == "MemoryAppend"))' "$fx/interop/contract.json" \
        > "$fx/interop/c.json" && mv "$fx/interop/c.json" "$fx/interop/contract.json"
    run_gate_on "$fx"; grade "MemoryAppend dropped from contract.json" "absent from contract.json" catch

    fresh core-entry-dropped
    sed -i '/("MemoryAppend", MEMORY_APPEND_VERSION),/d' "$fx/inc/omni/crates/omni-core/src/contract.rs"
    run_gate_on "$fx"; grade "MemoryAppend dropped from VERSIONED_MESSAGES" "absent from VERSIONED_MESSAGES" catch

    fresh gated-row-dropped
    sed -i '/kind: "ChannelResponse"/,+3d' "$fx/inc/omni/crates/omni-verify/src/contract.rs"
    run_gate_on "$fx"; grade "ChannelResponse has no GATED row" "ChannelResponse" catch

    fresh gated-row-renamed
    sed -i 's/kind: "MemoryAppend"/kind: "MemoryPush"/' "$fx/inc/omni/crates/omni-verify/src/contract.rs"
    run_gate_on "$fx"; grade "GATED row renamed to a message nothing declares" "MemoryPush" catch

    fresh doc-messages-empty
    jq '.messages = []' "$fx/interop/contract.json" > "$fx/interop/c.json" && mv "$fx/interop/c.json" "$fx/interop/contract.json"
    run_gate_on "$fx"; grade "contract.json declares no messages" "declares none" catch

    # ── 4 · the document is current ──────────────────────────────────────────
    fresh doc-top-version
    jq '.version = 2' "$fx/interop/contract.json" > "$fx/interop/c.json" && mv "$fx/interop/c.json" "$fx/interop/contract.json"
    run_gate_on "$fx"; grade "contract.json's own version bumped" "own version is 1" catch

    fresh doc-protocol-typo
    jq '.protocol = "unifur-subnext-interop-v2"' "$fx/interop/contract.json" > "$fx/interop/c.json" && mv "$fx/interop/c.json" "$fx/interop/contract.json"
    run_gate_on "$fx"; grade "contract.json protocol renamed" "declares protocol" catch

    fresh subnext-constant-bumped
    sed -i 's/^pub const CONTEXT_HIT_VERSION: u32 = 1;/pub const CONTEXT_HIT_VERSION: u32 = 2;/' "$fx/subnext/src/interop.rs"
    run_gate_on "$fx"; grade "subnext bumps CONTEXT_HIT_VERSION alone" "CONTEXT_HIT_VERSION" catch

    fresh unifur-constant-bumped
    sed -i 's/^pub const MEMORY_APPEND_VERSION: u32 = 1;/pub const MEMORY_APPEND_VERSION: u32 = 3;/' "$fx/inc/unifur/src/interop.rs"
    run_gate_on "$fx"; grade "unifur bumps MEMORY_APPEND_VERSION alone" "MEMORY_APPEND_VERSION" catch

    # ── 5 · the two contract files agree ─────────────────────────────────────
    fresh unifur-field-renamed
    sed -i 's/^    pub field: String,$/    pub taxonomy: String,/' "$fx/inc/unifur/src/interop.rs"
    run_gate_on "$fx"; grade "unifur renames ContextQuery.field" "ContextQuery" catch

    fresh subnext-field-removed
    sed -i '/^    pub provenance: Provenance,$/d' "$fx/subnext/src/interop.rs"
    run_gate_on "$fx"; grade "subnext drops ContextHit.provenance" "provenance" catch

    fresh doc-field-added
    jq '.messages |= map(if .name == "MemoryAppend" then .fields = ["payload","nonce"] else . end)' \
        "$fx/interop/contract.json" > "$fx/interop/c.json" && mv "$fx/interop/c.json" "$fx/interop/contract.json"
    run_gate_on "$fx"; grade "contract.json promises a field neither peer sends" "nonce" catch

    fresh subnext-protocol-split
    sed -i 's/^pub const PROTOCOL: &str = "unifur-subnext-interop";$/pub const PROTOCOL: \&str = "unifur-subnext-interop-2";/' "$fx/subnext/src/interop.rs"
    run_gate_on "$fx"; grade "subnext's PROTOCOL diverges" "PROTOCOL" catch

    # ── 6 · the ledger ───────────────────────────────────────────────────────
    fresh ledger-citation-past-eof
    printf '\n| N-90 | planted: a citation past the end of its file | planted | `%s/src/interop.rs:99999` | planted condition |\n' "$SUBNEXT" \
        >> "$fx/inc/omni/docs/NEGATIVE-RESULTS.md"
    run_gate_on "$fx"; grade "ledger cites a line that does not exist" "but the file is" catch

    fresh ledger-citation-missing-file
    printf '\n| N-91 | planted: a citation to a file that is gone | planted | `/nonexistent/repo/docs/gone.md:12` | planted condition |\n' \
        >> "$fx/inc/omni/docs/NEGATIVE-RESULTS.md"
    run_gate_on "$fx"; grade "ledger cites a file that does not exist" "cited source file still exists" catch

    fresh ledger-no-reopen
    printf '\n| N-92 | planted: a closed result with no reopening condition | planted | `%s/src/interop.rs:139` | — |\n' "$SUBNEXT" \
        >> "$fx/inc/omni/docs/NEGATIVE-RESULTS.md"
    run_gate_on "$fx"; grade "ledger row with an empty reopen cell" "reopening condition" catch

    fresh ledger-duplicate-id
    printf '\n| N-01 | planted: a duplicate id | planted | `%s/src/interop.rs:139` | planted condition |\n' "$SUBNEXT" \
        >> "$fx/inc/omni/docs/NEGATIVE-RESULTS.md"
    run_gate_on "$fx"; grade "ledger with a duplicate id" "are unique" catch

    fresh ledger-row-wrong-shape
    printf '\n| N-93 | planted: a row that is not the 5-column shape | planted | %s/src/interop.rs:139 |\n' "$SUBNEXT" \
        >> "$fx/inc/omni/docs/NEGATIVE-RESULTS.md"
    run_gate_on "$fx"; grade "ledger row with the wrong column count" "cells" catch

    fresh ledger-missing; rm -f "$fx/inc/omni/docs/NEGATIVE-RESULTS.md"
    run_gate_on "$fx"; grade "ledger absent entirely" "does not exist" catch

    # ── verdict ──────────────────────────────────────────────────────────────
    if [ "$fail_count" -ne 0 ]; then
        {
            echo "CROSS-REPO SELF-TEST FAILED: $fail_count of $((pass + fail_count)) cases did not behave as specified:" >&2
            printf '%s\n' "$failures" >&2
            echo "verify-cross-repo: fix the check, or the fixture — do not relax the" >&2
            echo "verify-cross-repo: expectation. The expectations are the list of things" >&2
            echo "verify-cross-repo: this gate exists to catch, and shortening it is how a" >&2
            echo "verify-cross-repo: gate rots while still reporting success." >&2
        }
        exit 1
    fi
    echo "CROSS-REPO SELF-TEST OK ($pass cases: 1 positive control, $((pass - 1)) planted defects, each caught and named)"
    exit 0
}

# ── main ─────────────────────────────────────────────────────────────────────

if ! command -v jq > /dev/null 2>&1; then
    echo "verify-cross-repo: CANNOT RUN — jq is not installed" >&2
    echo "verify-cross-repo:        why: the deployed contract document is JSON and reading it with" >&2
    echo "verify-cross-repo:        grep would check the file's punctuation, not its contents." >&2
    echo "verify-cross-repo:        A gate that cannot read its input must not report a pass." >&2
    exit 3
fi

if [ "$SELFTEST" = "1" ]; then
    self_test
fi

section "verify-cross-repo — drift between omni, subnext, unifur and peregrine"
note "canonical subnext : $SUBNEXT"
note "stale sibling     : $SUBNEXT_STALE"
note "vendored port     : $PEREGRINE_CTX"
note "unifur            : $UNIFUR"
note "deployed contract : $CONTRACT_JSON"

check_canonical
check_port
check_gated_table
check_document_fresh
check_field_lists
check_ledger

echo

if [ "$BLOCKED" -gt 0 ]; then
    printf 'verify-cross-repo: %d check(s) could not run and %d finding(s) were recorded\n' "$BLOCKED" "$FINDINGS" >&2
    echo "verify-cross-repo: a check that cannot run is not a passing check (AGENTS.md rule 6)." >&2
    echo "verify-cross-repo: every 'could not run' line above names the path or tool that is absent." >&2
    exit 3
fi

if [ "$FINDINGS" -eq 0 ]; then
    echo "CROSS-REPO OK"
    exit 0
fi

printf 'verify-cross-repo: %d finding(s) above\n' "$FINDINGS" >&2

if [ "$STRICT" = "1" ]; then
    echo "verify-cross-repo: the repos have drifted apart. Each finding names what was checked," >&2
    echo "verify-cross-repo: what was found, and why it matters. Fix the drifted side or re-point" >&2
    echo "verify-cross-repo: the dependency — do not edit the gate to make a run green." >&2
    exit 1
fi

echo "verify-cross-repo: advisory run (no --strict); run with --strict to fail on these." >&2
exit 0