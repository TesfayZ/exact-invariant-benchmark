# Offline reproduction

The reported primary dataset is exactly 1,200 LLM responses (400 each from
Groq, Cerebras and Gemini) plus 150 sparse numerical-baseline outputs. A further
150 rational multi-output numerical-baseline outputs are analysed separately.
No additional LLM experiment is used by the manuscript or submission package.

## Commands

The pinned dependencies require Python 3.12 or later. The recorded environment
uses Python 3.14.6; other interpreter/platform combinations are not guaranteed
to reproduce numerical discovery coefficients bit-for-bit.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
bash check_all.sh
bash regrade_all.sh
```

The expected summaries are `ALL CHECKS PASSED`, `regrade_all: done`, and
`0 mismatch(es)` in `results/verify_claims.txt`. No API calls are made.
The reference environment uses Python 3.14.6 on Ubuntu 26.04 LTS, Linux x86_64,
with glibc 2.43 and GCC 14.3.0. The lock file does not install the interpreter
or operating system. `python3 env_report.py` reports the active environment.

The required `algebraicRLtest` modules are bundled at commit
`687c868c204b8ec537451b5e046532de80197d9d`. No API key or repository clone is
needed for offline reproduction. The checked Python environment is recorded in
`requirements-lock.txt`.
Installing dependencies may require network access. TeX Live is required for
compiling the manuscript. Exact rational grading uses sympy; numerical discovery
additionally uses numpy, scipy and pysindy.

## Expected results

| Quantity | Count |
|---|---:|
| Primary LLM responses | 1,200 |
| Primary sparse-baseline outputs | 150 |
| LLM reference recoveries (affine-span or curve ideal equality) | 853 |
| Additional non-reference valid LLM answers | 36 |
| LLM task-valid, first/all rule | 889 / 889 |
| LLM cells with a reference/task-valid gap | 13 / 60 |
| Nonzero gap range | 5–30 percentage points |
| LLM empty / refused / unparseable | 82 / 40 / 30 |
| LLM reconstructed successes | 0 |
| Sparse-baseline reference / additional valid / reconstructed | 8 / 14 / 29 |
| Rational baseline first / any / all valid | 134 / 135 / 98 |
| Rational baseline complete curve generating sets | 30 / 30 |
| Independent CAS literal / reconstructed checks | 1,198 / 29 |
| Independent CAS disagreements | 0 |

Thirty-five non-reference LLM answers are explained by quadratic products of
references and one is exactly the cube `(A-B)**3`. None establishes an independent
invariant. All 30 unparseable LLM records are retained under the declared variable
contract; alternative notation does not silently alter primary scores.

## Evidence and grading

`primary_results.py` validates all selected slot identities, stored prompt hashes,
recorded generation settings, raw hashes and the 21-slot Groq follow-up audit.
`regrade_all.sh` grades only its explicit input list, then rebuilds
`results/graded_all.jsonl`, the analyses, policy audit, diagnostic sensitivity,
`results/paper_summary.json` and generated TeX claims. `paper_claims.json` pins
expected results; `verify_claims.py` compares recomputation with those claims
and checks CAS coverage. It also checks `docs/claims_generated.tex`, which supplies
numerical claim macros to the manuscript. Result tables and the figure are
generated from the grading records. The check does not parse numbers in prose.

Literal decimals and scientific notation are exact rational coefficients. A
separate denominator-capped reconstruction can yield a useful candidate but never
counts as literal success. Drift diagnostics centre the quotient representative;
constants and known constraints do not improve that algebraic diagnostic.
Graded records store outer verdicts and reconstructed sub-verdicts at each
denominator cap. The sensitivity flag detects changes in either, so a change
from reconstructed incomplete to reconstructed complete remains visible even
when the outer verdict stays `exact_after_rounding`.

The raw prompt strings are authoritative. Regenerating trajectories can differ
slightly across numerical-library/platform builds; tests allow only numeric-only
drift below 1e-3 relative. Set `STRICT_PROMPT_HASH=1` for exact regenerated hashes.

## Independent CAS

Singular 4.4.1 was used on the 1,350-row primary dataset. With
Singular installed, regenerate its input and comparison using:

```bash
python3 crosscheck_singular.py --singular Singular
python3 summarize_results.py
python3 verify_claims.py
```

The CAS script and report are included in the package. The report binds itself
to current raw-file hashes and records 1,198 literal and 29 reconstructed checks,
with zero mismatches; 152 records require no algebraic decision. Shared parsing
and system definitions limit independence. Singular is optional for core offline
regrading, which verifies the stored CAS report and generated script hashes.

## Runtime and monomial-order diagnostics

`results/grader_diagnostics.json` records one timed complete grading call for
each of the 1,198 parseable primary submissions, grouped by system and submitted
degree. It binds the raw files by SHA-256 and checks every timed verdict against
the archived grade. It also compares literal conservation and quotient
constancy under grevlex and lex, and their centred drift on all 227 conservation
failures with a stored score. All 977 conservation decisions agree; the maximum
drift difference and all three threshold-flip counts are zero. No timed call
was unavailable. The grader has no timeout; zero observed failures does not
bound worst-case runtime. Initial numerical setup is excluded, and shared
process caches can affect these single-pass measurements.

To rerun explicitly (overwrites machine-dependent timings), use:

```bash
OPENBLAS_NUM_THREADS=1 python3 grader_diagnostics.py
python3 make_tables.py
```

Ordinary offline reproduction preserves the archived timings. The script and
report are included in the submission supplement.

## Fresh numerical outputs

Raw discovery outputs are immutable paper evidence. Use new paths:

```bash
mkdir -p runs
OPENBLAS_NUM_THREADS=1 python3 sr_sindy.py --systems all --seeds 15 --out runs/fresh_sindy.jsonl
python3 sr_rational.py --seeds 15 --out runs/fresh_rational.jsonl
python3 grade_outputs.py --raw runs/fresh_rational.jsonl --out runs/fresh_rational_graded.jsonl
```

The rational baseline fixes relative SVD tolerance 1e-7 and maximum denominator
1000; raw records contain singular values and all emitted rational expressions.
Numerical recollection need not reproduce coefficients bit-for-bit across BLAS
builds. The numerical algorithms operate on sampled states and do not read LLM prompts.

## Manuscript figures and tables

```bash
make -C docs paper figure
```

This regenerates the tables and figure sources, then compiles the named paper.
TeX Live with PDFLaTeX, BibTeX, pgfplots, standalone and xurl is required.
