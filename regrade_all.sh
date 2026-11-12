#!/usr/bin/env bash
# Regrade only the explicitly selected paper experiments. No API calls.
set -euo pipefail
cd "$(dirname "$0")"
python3 primary_results.py
pairs=(primary/raw_groq:primary/graded_groq primary/raw_cerebras:primary/graded_cerebras primary/raw_gemini:primary/graded_gemini raw_sindy15:graded_sindy15 raw_rational15:graded_rational15)
pids=()
for pair in "${pairs[@]}"; do
  (
    target="results/${pair##*:}.jsonl"
    python3 grade_outputs.py --raw "results/${pair%%:*}.jsonl" --out "$target.tmp" > /dev/null
    mv "$target.tmp" "$target"
  ) &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
if (( failed )); then
  echo "A grader failed; combined results were not rebuilt." >&2
  exit 1
fi
cat results/primary/graded_groq.jsonl results/primary/graded_cerebras.jsonl results/primary/graded_gemini.jsonl results/graded_sindy15.jsonl > results/graded_all.jsonl
python3 analysis.py results/graded_all.jsonl > results/analysis_all.txt
python3 analysis.py results/graded_sindy15.jsonl > results/analysis_sindy15.txt
python3 sensitivity.py > /dev/null
python3 policy_audit.py > /dev/null
python3 summarize_results.py
python3 verify_claims.py > results/verify_claims.txt
echo "regrade_all: done"
