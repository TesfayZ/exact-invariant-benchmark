#!/usr/bin/env bash
# Offline paper reproduction, or a fresh numerical baseline in a new file.
set -euo pipefail
cd "$(dirname "$0")"
export ALGEBRAIC_RL_REPO="${ALGEBRAIC_RL_REPO:-$PWD/vendor/algebraicRLtest}"
case "${1:-offline}" in
  offline)
    python3 env_report.py
    bash check_all.sh
    bash regrade_all.sh
    make -C docs paper figure
    ;;
  sindy)
    out="${2:?Supply a new output path for the numerical baseline}"
    test ! -e "$out" || { echo "Output already exists: $out" >&2; exit 1; }
    OPENBLAS_NUM_THREADS=1 python3 sr_sindy.py --systems all --seeds 15 --out "$out"
    ;;
  *) echo "Usage: bash reproduce_all.sh [offline | sindy NEW_OUTPUT_PATH]" >&2; exit 2 ;;
esac
