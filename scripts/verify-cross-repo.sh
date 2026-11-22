#!/usr/bin/env bash
set -euo pipefail
scripts="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ "${1:-}" == --legacy-interop ]]; then
    shift
    exec "$scripts/verify-deployed-interop.sh" "$@"
fi
exec python3 "$scripts/verify_cross_repo.py" "$@"
