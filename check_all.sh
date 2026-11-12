#!/usr/bin/env bash
# One command to confirm the install is healthy. Takes ~1-2 minutes. Must end with "ALL CHECKS PASSED".
set -euo pipefail
cd "$(dirname "$0")"

# FIX: on success, print only the summary line; on failure, print the whole log
# so the failing check is visible (the old `| tail -1` hid it).
run_check() {
    local script="$1"
    local log
    log="$(mktemp)"
    if ! python3 "$script" > "$log" 2>&1; then
        echo "--- $script FAILED; full output below ---"
        cat "$log"
        rm -f "$log"
        return 1
    fi
    tail -1 "$log"
    rm -f "$log"
}

run_check controls.py
run_check test_pipeline.py
run_check test_fixes.py
run_check test_revision.py
python3 -m unittest test_primary_results
echo "ALL CHECKS PASSED"
