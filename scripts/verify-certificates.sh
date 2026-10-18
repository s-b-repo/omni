#!/usr/bin/env bash
# verify-certificates.sh — omni's certificate gate.
#
# One command, non-zero on failure, reason on stderr. Same shape as
# audit-bad-patterns.sh on purpose: a gate you have to remember the internals of
# is a gate you skip.
#
# Usage:
#   verify-certificates.sh [--debug]
#
# `--debug` builds the suite without optimisations. The default is release
# because two certificates (the atomic write race and the 200-write fixture) are
# timed loosely rather than exactly, and a debug build makes them several times
# slower for no extra confidence.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROFILE="release"

while [ "$#" -gt 0 ]; do
    case "$1" in
        --debug) PROFILE="debug"; shift ;;
        -h|--help) sed -n '/^# Usage:/,/^# `--debug`/p' "$0" | sed 's/^# \?//; $d'; exit 0 ;;
        *) echo "unknown option $1" >&2; exit 2 ;;
    esac
done

cd "$ROOT"

# `--debug` is honoured by swapping the profile; the release default keeps the
# two timing-sensitive certificates (the atomic write race, the 200-write
# fixture) quick without weakening what they check. `-q` keeps cargo's progress
# noise out of the certificate table, which is the only thing a reader of this
# gate's output actually wants to see.
if ! cargo run -p omni-verify --profile="$PROFILE" --quiet -- verify; then
    echo "CERTIFICATES FAILED" >&2
    echo "verify-certificates: a failing certificate means a load-bearing identity" >&2
    echo "verify-certificates: stopped holding. Read the theorem column — it states" >&2
    echo "verify-certificates: what the claim is, so you can judge whether the check" >&2
    echo "verify-certificates: is worth keeping before you widen its tolerance." >&2
    exit 1
fi

echo "CERTIFICATES OK"