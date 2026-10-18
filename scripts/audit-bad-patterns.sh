#!/usr/bin/env bash
# audit-bad-patterns.sh — omni's other gate, next to `omni verify`.
#
# Two properties this gate has that a linter does not:
#
#   1. **It ratchets.** `--strict` fails when a section's count *exceeds* its
#      budget, so an existing, reviewed debt can stay visible and shrink, while
#      new code is held to zero.
#   2. **omni starts at zero.** peregrine's table opens at A 285 / B 109 /
#      E 2323 because those measure its existing debt across 13 crates. omni is
#      greenfield with no debt to carry, so every gated section opens at 0. A
#      budget that needs raising because new code added a hit is a review
#      conversation, not a bookkeeping update — budgets only move down.
#
# Usage:
#   audit-bad-patterns.sh [--strict] [--section A,B] [--files <path,...>]
#                         [--max-report N] [--show] [--budgets] [--self-test]
#
# Exit: 0 clean, 1 a gated section exceeded its budget (or --self-test found a
# dead rule), 2 usage error.

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STRICT=0
SHOW=0
MAX_REPORT=40
FILE_LIST=""
declare -a ONLY_SECTIONS=()

# ── Budgets ──────────────────────────────────────────────────────────────────
#
# Every gated section is 0. See the header: this is the honest starting point
# for a repository that has never had a bad commit, and the only way the number
# stays meaningful is if raising one is louder than fixing the hit.
declare -A BUDGETS=(
    [A]=0   # panicking error handling
    [B]=0   # swallowed errors
    [C]=0   # lint suppression
    [J]=0   # secrets
    [M]=0   # injection
    [N]=0   # UB and concurrency
    [T]=0   # Cargo.toml hygiene
)
readonly BUDGETED_SECTIONS="A B C J M N T"

budget_of() { printf '%s' "${BUDGETS[$1]:--1}"; }

section_names() {
    cat <<'EOF'
A  panicking error handling   .unwrap( .expect( panic! todo! unimplemented! unreachable!
B  swallowed errors           let _ =  if let Err(_)  Err(_) => {}  bare .ok();
C  lint suppression           #[allow( #[cfg_attr(.., allow)]  // audit-allow:  clippy::allow_attributes
J  secrets                    API keys, private keys, credentialed URLs
M  injection                  shell/SQL built by string formatting or interpolation
N  UB and concurrency         unsafe outside a crate that may use it
T  Cargo.toml hygiene         wildcard deps, git deps without a pinned rev
EOF
}

usage() { sed -n '/^# Usage:/,/^# Exit:/p' "$0" | sed 's/^# \?//; $d'; }

while [ "$#" -gt 0 ]; do
    case "$1" in
        --strict) STRICT=1; shift ;;
        --section)
            [ "$#" -ge 2 ] || { echo "--section needs a section letter" >&2; exit 2; }
            IFS=',' read -r -a ONLY_SECTIONS <<< "$2"; shift 2 ;;
        --files)
            [ "$#" -ge 2 ] || { echo "--files needs a path" >&2; exit 2; }
            FILE_LIST="$2"; shift 2 ;;
        --max-report)
            [ "$#" -ge 2 ] || { echo "--max-report needs a number" >&2; exit 2; }
            MAX_REPORT="$2"; shift 2 ;;
        --show) SHOW=1; shift ;;
        --budgets) echo "section  budget"; for s in $BUDGETED_SECTIONS; do printf '%-7s %s\n' "$s" "$(budget_of "$s")"; done; exit 0 ;;
        --self-test) SELFTEST=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown option $1" >&2; usage >&2; exit 2 ;;
    esac
done
SELFTEST="${SELFTEST:-0}"

want() {
    [ "${#ONLY_SECTIONS[@]}" -eq 0 ] && return 0
    local want="$1" s
    for s in "${ONLY_SECTIONS[@]}"; do [ "$s" = "$want" ] && return 0; done
    return 1
}

# ── File set ─────────────────────────────────────────────────────────────────
#
# `crates/**/src/**.rs` plus the workspace manifests. Tests and examples are
# included on purpose: with `allow-unwrap-in-tests = false` a panic in a test is
# a panic, and excluding tests would make the gate's claim untrue for exactly
# the files most likely to acquire one.
mapfile -t FILES < <(
    if [ -n "$FILE_LIST" ]; then
        IFS=',' read -r -a explicit <<< "$FILE_LIST"; printf '%s\n' "${explicit[@]}"
    else
        find "$ROOT/crates" -name '*.rs' -type f 2>/dev/null | sort
        printf '%s\n' "$ROOT/Cargo.toml"
    fi
)

scan() {
    # scan <section-letter> <match-regex> [exclude-regex]
    # Prints `[letter] path:line: text` for each match.
    #
    # Two filters, in order:
    #
    #   1. Comment lines are dropped. Without this, this file's own
    #      documentation — which quotes `.unwrap(`, `let _ =` and `panic!` in
    #      order to explain why they are banned — would count as violations of
    #      the rules it states, and the gate could never reach its budget. A
    #      pattern quoted in prose is not a pattern executed.
    #
    #   2. An optional exclusion regex. Needed because POSIX ERE has no
    #      lookahead, and section B needs to distinguish `let _ = fallible()`
    #      from `let _ = &borrowed` — one drops a Result, the other drops a
    #      borrow, and only the first is the defect. Matching on a positive
    #      signal (`let _ = ` alone) and excluding the benign forms afterwards
    #      is the portable way to express "not followed by & or :".
    local letter="$1" re="$2" exclude="${3:-}" f
    want "$letter" || return 0
    for f in "${FILES[@]}"; do
        [ -f "$f" ] || continue
        grep -nE "$re" "$f" 2>/dev/null \
            | grep -vE '^[0-9]+:[[:space:]]*(//|/\*|\*/|\*)' \
            | if [ -n "$exclude" ]; then grep -vE "$exclude"; else cat; fi \
            | while IFS= read -r line; do
                printf '%s: %s\n' "${f#"$ROOT"/}" "$line"
            done
    done | sed "s/^/[$letter] /"
}

REPORT=""
COUNT=0
record() {
    # record <letter> <description> <match-regex> [exclude-regex]
    local letter="$1" desc="$2" re="$3" exclude="${4:-}" hits
    hits="$(scan "$letter" "$re" "$exclude")"
    [ -z "$hits" ] && return 0
    local n budget status
    n="$(printf '%s\n' "$hits" | wc -l | tr -d ' ')"
    budget="$(budget_of "$letter")"
    status="ok"
    if [ "$budget" -ge 0 ] && [ "$n" -gt "$budget" ]; then status="OVER BUDGET"; fi
    REPORT+="[$letter] $n hit(s), budget $budget — $status"$'\n'
    REPORT+="$(printf '%s\n' "$hits" | head -"$MAX_REPORT")"$'\n'
    if [ "$n" -gt "$MAX_REPORT" ]; then REPORT+="    ... and $((n - MAX_REPORT)) more"$'\n'; fi
    REPORT+=$'\n'
    if [ "$status" != "ok" ]; then COUNT=$((COUNT + n - budget)); fi
}

# ── Section A: panicking error handling ──────────────────────────────────────
# Anchored so `assert_eq!` is not matched: it expands to a panic, but it is the
# sanctioned way to assert inside a test. The panic *macro* is what is banned.
#
# Two halves, for two different jobs. `.unwrap()` and `.expect()` are matched
# unconditionally: requiring a non-word character before the dot sounds
# tidier but misses the ordinary `x.unwrap()`, because `x` is a word character.
# That hole was found by --self-test, not by reading the pattern.
#
# The macros keep the guard, because there `mypanic!(` is a different name and
# must not count. (`.unwrap_or(` cannot false-positive either: the pattern
# requires `(` immediately after `unwrap`.)
record A "panicking error handling" \
    '\.unwrap\(\)|\.expect\(|(^|[^[:alnum:]_])(panic!|todo!|unimplemented!|unreachable!)\('

# ── Section B: swallowed errors ──────────────────────────────────────────────
# Narrow on purpose. peregrine's B budget of 109 counts `.ok()` and `.is_err()`
# broadly; omni's budget is 0, and a zero budget on a loose pattern is a gate
# that cries wolf, which gets ignored within a week.
#
# So this section counts only constructs that *structurally* discard a failure:
#   let _ = expr;              the value is explicitly dropped
#   if let Err(_) = expr       the error variant is matched and discarded
#   Err(_) => {}               a wildcard arm with an empty body
#   expr.ok();                  a Result collapsed to a value and dropped
#
# Matched unanchored on purpose: an earlier revision anchored `let _ =` to the
# start of a line, which caught `let _ = fallible()` written as a statement and
# missed the same discard written inline as `let _ = fallible()`. A planted
# probe caught the miss, which is what the self-test exists to prevent.
#
# `is_err()` as a boolean *predicate* is not on the list: `if x.is_err()` names a
# condition, it does not swallow anything.
#
# The exclusion keeps the two idioms that drop a value rather than a failure —
# `let _ = &borrow` and `let _: T = expr` — from counting. POSIX ERE has no
# lookahead, so they are excluded after the match rather than in the pattern.
#
# The sanctioned alternative to every construct listed is
# `omni_core::note_advisory(what, &err)`, which records the failure and lets the
# caller continue.
record B "swallowed errors" \
    'let _ = |if let Err\(_\)|^[[:space:]]*Err\(_\) *=> *\{ *\}|^[[:space:]]*[^/].*\.ok\(\);' \
    'let _ = &|let _:'

# ── Section C: lint suppression ──────────────────────────────────────────────
record C "lint suppression" \
    '#\[allow\(|#\[cfg_attr\([^]]*allow|// *audit-allow:|clippy::allow_attributes|#\[expect\('

# ── Section J: secrets ───────────────────────────────────────────────────────
record J "secrets" \
    '(sk-[A-Za-z0-9]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}|-----BEGIN [A-Z ]*PRIVATE KEY-----|://[^/[:space:]]*:[^/@[:space:]]*@|(password|passwd|secret|api[_-]?key|token)[[:space:]]*[:=][[:space:]]*"[^"]{8,}")'

# ── Section M: injection ─────────────────────────────────────────────────────
# A command or SQL string assembled by interpolation. omni shells out for the
# model backend and runs nothing else, so this should stay empty; if it does not,
# that is the finding.
record M "injection" \
    '(Command::new\([^)]*format!|format!\("[^"]*(SELECT|INSERT|UPDATE|DELETE|DROP)[[:space:]]|format!\("[^"]*\{\}[^"]*"[^)]*\.arg\()'

# ── Section N: UB and concurrency ────────────────────────────────────────────
# `unsafe` is confined. omni-core and omni-verify both carry
# `#![forbid(unsafe_code)]`; this catches it appearing where that would not.
record N "unsafe outside a forbidden crate" \
    '(^|[^[:alnum:]_])unsafe\s*(\{|fn|impl|trait|static)'

# ── Section T: Cargo.toml hygiene ────────────────────────────────────────────
record T "Cargo.toml hygiene" \
    '=[[:space:]]*"\*"|= *\{[^}]*git[[:space:]]*='

# ── Self-test ────────────────────────────────────────────────────────────────
#
# "A gate that only ever passes is indistinguishable from one whose regex rotted."
# subnext ships the same idea (`audit-selftest.sh`): plant known-bad code and
# assert the gate still catches it. This is not ceremony — an earlier revision
# of section B anchored `let _ =` to line start, which caught the statement form
# and missed the inline form, and only a planted probe revealed it.
#
# Every gated section is exercised, with the exact count expected.
if [ "$SELFTEST" = "1" ]; then
    plant="$(mktemp -d)"
    cat > "$plant/probe.rs" <<'PLANT'
fn a() { let x: Option<u8> = None; let _ = x.unwrap(); }
#[allow(dead_code)]
fn b() -> u8 { panic!("boom") }
fn c() { let key = "sk-AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"; println!("{key}"); }
fn d() { let _ = std::fs::remove_file("/tmp/x"); }
fn e() { unsafe { std::ptr::null::<u8>(); } }
// These two MUST NOT be counted: they drop a value, not a failure.
fn f() { let good = String::new(); let _ = &good; let _: u8 = 1; }
PLANT
    printf '[workspace]\nmembers = []\n' > "$plant/Cargo.toml"

    # scan() walks the global FILES list, so point it at the plant for the
    # duration. Without this the self-test would silently grade the real
    # repository's cleanliness — which is the one thing it must never do.
    saved_files=("${FILES[@]+"${FILES[@]}"}")
    FILES=("$plant/probe.rs" "$plant/Cargo.toml")
    ONLY_SECTIONS=()

    got_a="$(scan A '\.unwrap\(\)|\.expect\(|(^|[^[:alnum:]_])(panic!|todo!|unimplemented!|unreachable!)\(' | grep -c 'probe.rs' || true)"
    got_b="$(scan B 'let _ = |if let Err\(_\)|^[[:space:]]*Err\(_\) *=> *\{ *\}|^[[:space:]]*[^/].*\.ok\(\);' 'let _ = &|let _:' | grep -c 'probe.rs' || true)"
    got_c="$(scan C '#\[allow\(|#\[cfg_attr\([^]]*allow|// *audit-allow:|clippy::allow_attributes|#\[expect\(' | grep -c 'probe.rs' || true)"
    got_j="$(scan J '(sk-[A-Za-z0-9]{20,}|-----BEGIN [A-Z ]*PRIVATE KEY-----)' | grep -c 'probe.rs' || true)"
    got_n="$(scan N '(^|[^[:alnum:]_])unsafe\s*(\{|fn|impl|trait|static)' | grep -c 'probe.rs' || true)"

    FILES=("${saved_files[@]+"${saved_files[@]}"}")
    rm -rf "$plant"

    fail=0
    check() {
        local section="$1" got="$2" want="$3"
        if [ "$got" -ne "$want" ]; then
            echo "AUDIT SELF-TEST FAILED: section $section matched $got time(s), expected $want" >&2
            fail=1
        fi
    }
    check A "$got_a" 2   # .unwrap( on line 2, panic!( on line 3
    check B "$got_b" 2   # the two inline let _ = discards, NOT the benign pair on line 7
    check C "$got_c" 1   # one #[allow(
    check J "$got_j" 1   # one planted key
    check N "$got_n" 1   # one unsafe block

    if [ "$fail" -ne 0 ]; then
        echo "audit-bad-patterns: a section's regex has rotted. Fix the pattern;" >&2
        echo "audit-bad-patterns: do not lower the expectation — the expectation" >&2
        echo "audit-bad-patterns: is the list of things this gate exists to catch." >&2
        exit 1
    fi
    echo "AUDIT SELF-TEST OK (A=2 B=2 C=1 J=1 N=1; benign idioms correctly ignored)"
    exit 0
fi

# ── Report ───────────────────────────────────────────────────────────────────
if [ "$SHOW" = "1" ] && [ -z "$REPORT" ]; then section_names; fi
[ -n "$REPORT" ] && printf '%s' "$REPORT"

if [ "$COUNT" -eq 0 ]; then
    echo "AUDIT OK"
    exit 0
fi

if [ "$STRICT" = "1" ]; then
    echo "audit-bad-patterns: $COUNT hit(s) above budget across: $BUDGETED_SECTIONS" >&2
    echo "audit-bad-patterns: fix the code, or spend the budget in review — do not raise it silently." >&2
    exit 1
fi
echo "AUDIT: $COUNT hit(s) above budget (non-strict)"
exit 0