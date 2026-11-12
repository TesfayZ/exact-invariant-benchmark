# Exact Grading of Proposed Polynomial Invariants

Python code for exact grading of polynomial constraints and conserved quantities
over the rationals. Proposals can come from a language model, a numerical method
or another source. The grader checks the declared task; it does not determine
whether a correct answer was recalled or discovered.

The grader reports literal task validity, reference recovery, rational
reconstruction and output-set completeness separately. Reference recovery means
affine-span membership for conservation tasks and ideal equality for curve tasks.
Decimal literals are treated as exact rational coefficients. The accompanying
audit uses five system families and their parameter or coordinate variants.
The algebraic tests are established methods; the contribution is the grading
protocol and measurement of how its reporting choices affect the same outputs.
See [vendor/NOTICE.md](vendor/NOTICE.md) for backend attribution and licensing.

## Installation and offline use

Python 3.12 or later is required; the reference environment uses Python 3.14.6.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
bash check_all.sh
bash regrade_all.sh
```

Cached reproduction needs no API credentials or external backend checkout.
[REPRODUCE.md](REPRODUCE.md) gives expected results, optional CAS checks and
commands for fresh numerical outputs.

## Code and data

| Path | Purpose |
|---|---|
| `systems.py`, `data.py`, `fresh_systems.py` | Exact system definitions, sampling and parameter variants |
| `grade.py`, `grade_outputs.py` | Symbolic grading and JSONL batch interface |
| `prompts.py`, `run_llm.py` | Prompt generation and model collection interface |
| `sr_sindy.py`, `sr_rational.py` | Sparse and rational numerical proposal baselines |
| `primary_results.py` | Dataset selection, slot identities and evidence hash validation |
| `analysis.py`, `policy_audit.py`, `sensitivity.py` | Aggregate scores and paired grading-policy analyses |
| `summarize_results.py`, `verify_claims.py`, `paper_claims.json` | Derived summaries and expected-value checks |
| `crosscheck_singular.py` | Independent CAS cross-check |
| `grader_diagnostics.py`, `time_grader.py` | Runtime and monomial-order diagnostics |
| `make_tables.py`, `make_figures.py` | Generate TeX tables and figure sources from results |
| `results/primary/` | Stored prompts, settings, raw responses, grades and provenance |
| `results/raw_sindy15.jsonl`, `results/raw_rational15.jsonl` | Numerical discovery outputs |
| `results/graded_all.jsonl` | Combined LLM and sparse-baseline grading records |
| `vendor/` | Licensed algebraic backend and external criterion comparison code |
| `docs/` | Shared manuscript and generated results for the paper |

The evidence contains 400 responses per deployment (Groq, Cerebras and Gemini),
1,200 total, and two numerical collections of 150 outputs each. The Groq
transport audit preserves 21 initially failed slots and their follow-up attempts.
Returned empty, refused, invalid and truncated answers remain in the dataset.
Stored prompt strings are authoritative; regenerated numerical observations may
differ slightly across numerical libraries. `STRICT_PROMPT_HASH=1` requires
byte-identical regenerated prompts.

## Grading an output file

Provide a JSONL file with one record per response. `key` selects a definition in
`systems.REGISTRY`; `output` is the returned JSON **string**, containing a
`polynomials` list and a `refuse` flag. The other required fields identify the
proposer and experimental condition. For example:

```json
{"model":"my-proposer","system":"rigid_body_1_2_3","key":"rigid_body","condition":"A","noise":0.0,"sample":0,"output":"{\"polynomials\":[\"w1**2+2*w2**2+3*w3**2\"],\"refuse\":false}"}
```

Save it as `responses.jsonl`, then run:

```bash
python3 grade_outputs.py --raw responses.jsonl --out runs/my-proposer-graded.jsonl
```

Use a new output path for prospective experiments. Keep returned responses and
transport failures distinct, and retain the exact prompts and service settings.
The collection utilities `collect_primary.py`, `recover_primary.py` and
`prepare_primary.py` support planned collection and evidence selection.

## Paper reproducibility

The paper's LaTeX sources are included in `docs/` so the reproducibility scripts
can compare recomputed experimental results with the numerical values used in
the paper. Tables, the figure and numerical claim macros are generated from the
grading records. `verify_claims.py` checks recomputed summaries against the
paper's pinned values and generated LaTeX macros. These checks cover structured
numerical values rather than the manuscript prose.

Readers can reproduce the reported results using the code, cached evidence and
commands in [REPRODUCE.md](REPRODUCE.md). Submission packaging and publisher
source archives are separate release materials; rebuilding them is not needed
to reproduce the paper's results.

To rebuild the paper with TeX Live installed:

```bash
make -C docs paper figure
```

## License

MIT; third-party notices and licenses are retained under `vendor/`.
