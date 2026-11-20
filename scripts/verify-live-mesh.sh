#!/usr/bin/env bash
# verify-live-mesh.sh — exercise `omni model peregrine --complete` against the
# mesh-shim, end to end.
#
# What this gate turns from a claim into evidence:
#
#   `omni model peregrine --complete` is the live-integration gate for the
#   model seam. docs/CONTRACT.md "What is not true yet" says plainly: "No
#   peregrine server has been exercised. No peregrine-serve is running on
#   this machine, so the twenty-seven peregrine certificates and the client
#   tests prove the dialect and the transport, not peregrine's behaviour.
#   The transport was proven end to end against ollama on :11434, which is
#   OpenAI-compatible but not peregrine — so 'it works against a compatible
#   server' is a weaker claim than 'it works against peregrine', and it is
#   the weaker claim that is true. No live integration test runs in CI."
#
# This script boots the sibling `mesh-shim` (Python stdlib, shells out to
# `dblocks`), drives the omni binary at it, and asserts the three findings a
# live run needs to produce: a `would POST`, a `health: {...}` reachable line,
# and a `finish_reason:` from the real round trip. It is the minimum shape of
# "the live path works", and it fails loudly when any of those are missing.
#
# Same shape as `audit-bad-patterns.sh` and `verify-certificates.sh`: one
# command, non-zero on failure, the reason on stderr. A gate whose internals
# you have to remember is a gate you skip.
#
# Usage:
#   verify-live-mesh.sh [--port N] [--keep-shim]
#
# Exit:
#   0 the live path answered and every required marker is present
#   1 a required marker was missing, or `omni model peregrine --complete`
#     exited non-zero
#   2 a usage error
#   3 a prerequisite is absent (dblocks binary, checkpoint, mesh-shim,
#     python3). Deliberately not 0: "a check that cannot run is not a
#     passing check" (AGENTS.md rule 6).

set -uo pipefail

SCRIPT="$(realpath "${BASH_SOURCE[0]}")"
ROOT="$(cd "$(dirname "$SCRIPT")/.." && pwd)"
WORKSPACE="$(cd "$ROOT/.." && pwd)"

PORT=0
KEEP=0

usage() {
    cat >&2 <<EOF
usage: $(basename "$0") [--port N] [--keep-shim]

  --port N      pin the shim to port N instead of asking the OS for a free one.
                Defaults to picking a port the kernel says is unused.
  --keep-shim   leave the shim running after the test finishes. Useful for
                prodding the live path by hand without re-booting. Prints the
                shim pid on stderr.
EOF
}

while (($#)); do
    case "$1" in
    --port)
        shift || { usage; exit 2; }
        PORT="$1"
        ;;
    --keep-shim) KEEP=1 ;;
    -h | --help) usage; exit 0 ;;
    *)
        echo "unknown flag: $1" >&2
        usage
        exit 2
        ;;
    esac
    shift
done

say()  { printf 'verify-live-mesh: %s\n' "$*" >&2; }
fail() { printf 'verify-live-mesh: FAIL — %s\n' "$*" >&2; }
missing() {
    printf 'verify-live-mesh: could not run — %s\n' "$*" >&2
    exit 3
}

# ---- preflight ---------------------------------------------------------------

OMNI_BIN="$ROOT/target/debug/omni"
MESH_SHIM="$WORKSPACE/mesh-shim/mesh_shim.py"
DBLOCKS="$WORKSPACE/unifur/target/debug/dblocks"
CHECKPOINT="$WORKSPACE/nina-data/tiny-model/lm-12f979ad92611014.mpk"

command -v python3 >/dev/null || missing "python3 not on PATH"
[[ -f "$MESH_SHIM"  ]] || missing "mesh-shim not at $MESH_SHIM"
[[ -x "$DBLOCKS"    ]] || missing "dblocks not executable at $DBLOCKS (build unifur first)"
[[ -f "$CHECKPOINT" ]] || missing "checkpoint not at $CHECKPOINT (mint one with dblocks lm train --tiny)"

if [[ ! -x "$OMNI_BIN" ]]; then
    say "building omni..."
    (cd "$ROOT" && cargo build --bin omni --quiet) || { fail "omni build failed"; exit 1; }
fi

# ---- pick a port -------------------------------------------------------------

if [[ "$PORT" -eq 0 ]]; then
    # Ask the kernel for a free port and release it immediately. A gap between
    # releasing and the shim binding is a race, but on localhost the window is
    # vanishingly small, and a collision here fails loudly instead of silently.
    PORT="$(python3 -I -c "
import socket
s = socket.socket()
s.bind(('127.0.0.1', 0))
print(s.getsockname()[1])
s.close()
")" || missing "could not ask the kernel for a free port"
fi

# ---- boot the shim -----------------------------------------------------------

LOG_DIR="$(mktemp -d -t mesh-shim.XXXXXX)"
SHIM_LOG="$LOG_DIR/shim.log"
OMNI_LOG="$LOG_DIR/omni.log"

say "booting mesh-shim on 127.0.0.1:$PORT (log: $SHIM_LOG)"

python3 -I "$MESH_SHIM" \
    --backend lm-generate \
    --dblocks "$DBLOCKS" \
    --checkpoint "$CHECKPOINT" \
    --host 127.0.0.1 \
    --port "$PORT" \
    --max-new-tokens 24 \
    --log-level INFO \
    >"$SHIM_LOG" 2>&1 &
SHIM_PID=$!

cleanup() {
    local rc=$?
    if ((KEEP)) && ((rc == 0)); then
        say "leaving shim running (pid=$SHIM_PID, port=$PORT) because --keep-shim was passed"
        say "log at $SHIM_LOG"
        return "$rc"
    fi
    if kill -0 "$SHIM_PID" 2>/dev/null; then
        kill -TERM "$SHIM_PID" 2>/dev/null || true
        # Give it up to two seconds to close on its own, then SIGKILL.
        for _ in 1 2 3 4; do
            kill -0 "$SHIM_PID" 2>/dev/null || break
            sleep 0.5
        done
        kill -KILL "$SHIM_PID" 2>/dev/null || true
    fi
    rm -rf "$LOG_DIR"
    return "$rc"
}
trap cleanup EXIT

# Wait for the shim to accept connections. A gate that hard-sleeps "long
# enough" is a gate that flakes on slow hardware; probe until /health answers
# or the shim dies.
for _ in $(seq 1 40); do
    if ! kill -0 "$SHIM_PID" 2>/dev/null; then
        fail "shim exited during startup"
        sed -n '1,80p' "$SHIM_LOG" >&2
        exit 1
    fi
    if python3 -I -c "
import sys, urllib.request
try:
    urllib.request.urlopen('http://127.0.0.1:$PORT/health', timeout=1).read()
except Exception:
    sys.exit(1)
" 2>/dev/null; then
        break
    fi
    sleep 0.25
done

if ! python3 -I -c "
import sys, urllib.request
try:
    urllib.request.urlopen('http://127.0.0.1:$PORT/health', timeout=1).read()
except Exception:
    sys.exit(1)
" 2>/dev/null; then
    fail "shim did not accept /health within 10s"
    sed -n '1,80p' "$SHIM_LOG" >&2
    exit 1
fi

# ---- drive omni --------------------------------------------------------------

say "driving omni model peregrine --complete --url http://127.0.0.1:$PORT"
"$OMNI_BIN" model peregrine --complete --url "http://127.0.0.1:$PORT" >"$OMNI_LOG" 2>&1
OMNI_RC=$?

if ((OMNI_RC != 0)); then
    fail "omni model peregrine --complete exited $OMNI_RC"
    sed -n '1,120p' "$OMNI_LOG" >&2
    exit 1
fi

# ---- assert markers ----------------------------------------------------------

# Three markers the live run must produce. Each is a certificate reduced to
# a grep (AGENTS.md rule 6): a claim and a residual — the claim is the
# human string in `label`, the residual is 0 when the pattern matched and
# 1 when it did not, and neither is widened to make the suite green.
declare -a MARKERS=(
    'would POST http://127.0.0.1:'$PORT'/v1/chat/completions|the client built a POST to the expected URL'
    'health: {"ok": true|the shim was reachable on /health before the chat call'
    'sending\.\.\.|the --complete path actually dispatched the chat'
    'finish_reason:|the live round trip returned a parsed finish reason'
)

any_missing=0
for entry in "${MARKERS[@]}"; do
    pattern="${entry%%|*}"
    label="${entry##*|}"
    if grep -Eq "$pattern" "$OMNI_LOG"; then
        say "ok  — $label"
    else
        fail "marker missing: $label (pattern: $pattern)"
        any_missing=1
    fi
done

if ((any_missing)); then
    echo "---- omni output ----" >&2
    sed -n '1,120p' "$OMNI_LOG" >&2
    exit 1
fi

# ---- report ------------------------------------------------------------------

REPLY_LINE="$(grep -E '^(reply|finish_reason|usage):' "$OMNI_LOG" | tr '\n' ' ')"
say "LIVE OK  — $REPLY_LINE"
exit 0
