# SRBench evaluator source snapshots

Unmodified files from cavalab/srbench, release v2.0, commit
`e5ded4715ed5721703353d3500a2fdb99004faf1`:
https://github.com/cavalab/srbench/tree/e5ded4715ed5721703353d3500a2fdb99004faf1

`assess_symbolic_model.py` and `symbolic_utils.py` come from `experiment/`;
`collate_groundtruth_results.py` comes from `postprocessing/`. The upstream
GPL-3.0 licence is preserved in LICENSE. These snapshots document provenance;
they are not imported by the audit and their full training pipeline is not run.

The audit adapts the published difference/ratio symbolic criterion to polynomial
conservation outputs, against each listed reference, using the shared exact parser.
It does not reproduce the regression R2 prefilter, estimator-specific cleaning,
or float rounding. This is explicitly a criterion-level transfer, not a claim
to reproduce SRBench scores or identify defects in its regression task.
