"""Offline runtime and grevlex/lex audit of every primary algebraic submission.

Run explicitly: timings are machine dependent. Raw discovery evidence is read only.
Numerical setup is excluded; each complete grade call uses D=1000, including
reconstruction and drift when needed. Order comparisons use literal polynomials.
"""
import hashlib
import json
import platform
import statistics
import time
from pathlib import Path

import numpy as np
import sympy as sp
import grade
import grade_outputs
from primary_results import PRIMARY_PAIRS


def run():
    records, hashes = [], {}
    bases = {}
    for raw, graded in PRIMARY_PAIRS:
        path = Path('results') / (raw + '.jsonl')
        hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        originals = [json.loads(line) for line in path.read_text().splitlines()]
        expected = [json.loads(line) for line in (Path('results') / (graded + '.jsonl')).read_text().splitlines()]
        assert len(originals) == len(expected)
        for index, (row, archived) in enumerate(zip(originals, expected)):
            assert all(row[k] == archived[k] for k in ('model', 'system', 'condition', 'sample'))
            texts, _ = grade_outputs.extract(row['output'])
            if not texts:
                continue
            system, kw = grade_outputs.ctx(row['key'])
            texts = texts if system.vanishing_task else texts[:1]
            polys = [grade.parse(text, system)[0] for text in texts]
            if any(p is None for p in polys):
                continue
            start = time.perf_counter()
            try:
                result = grade.grade(texts if system.vanishing_task else texts[0], system, D=1000, **kw)
            except Exception as error:
                result = dict(verdict='grader_unavailable', note=str(error))
            elapsed = 1000 * (time.perf_counter() - start)
            assert result['verdict'] == archived['verdict'], (raw, index)
            record = dict(source=raw, row=index, system=row['key'],
                          degree=max(sp.Poly(p, *system.variables).total_degree() for p in polys),
                          milliseconds=elapsed, verdict=result['verdict'])
            # Vanishing residuals do not reduce modulo an order-dependent basis.
            if not system.vanishing_task:
                p = polys[0]
                derivative = grade.lie_derivative_numerator(p, system)
                decisions, scores = {}, {}
                for order in ('grevlex', 'lex'):
                    key = (row['key'], order)
                    if key not in bases:
                        bases[key] = (sp.groebner(system.constraints, *system.variables,
                                                 order=order, domain='QQ') if system.constraints else None)
                    gb = bases[key]
                    reduce = lambda q: gb.reduce(q)[1] if gb is not None else sp.expand(q)
                    representative = reduce(p)
                    decisions[order] = (reduce(derivative) == 0,
                                        sp.Poly(representative, *system.variables).total_degree() <= 0)
                    if 'relative_drift' in archived['detail']:
                        centred = representative - sp.Poly(representative, *system.variables).TC()
                        # Same cached six trajectories as the primary grader.
                        # Reuse the public diagnostic with an already reduced representative.
                        scores[order] = kw['drift_fn'].legacy(centred)
                assert decisions['grevlex'] == decisions['lex'], (raw, index)
                record['order_decisions'] = decisions
                if scores:
                    assert np.isclose(scores['grevlex'], archived['detail']['relative_drift'], rtol=1e-12, atol=1e-14)
                    record['drift_by_order'] = scores
            records.append(record)
    def describe(selected):
        times = [r['milliseconds'] for r in selected]
        return dict(n=len(times), min_degree=min(r['degree'] for r in selected),
                    max_degree=max(r['degree'] for r in selected), median_ms=statistics.median(times),
                    p95_ms=float(np.percentile(times, 95)), max_ms=max(times))
    groups, degrees = [], []
    for key in sorted({r['system'] for r in records}):
        selected = [r for r in records if r['system'] == key]
        groups.append(dict(system=key, **describe(selected)))
        for degree in sorted({r['degree'] for r in selected}):
            degrees.append(dict(system=key, degree=degree,
                                **describe([r for r in selected if r['degree'] == degree])))
    scored = [r['drift_by_order'] for r in records if 'drift_by_order' in r]
    summary = dict(n=len(records), unavailable=sum(r['verdict'] == 'grader_unavailable' for r in records),
                   order_checks=sum('order_decisions' in r for r in records),
                   drift_checks=len(scored), max_drift_difference=max((abs(s['lex']-s['grevlex']) for s in scored), default=0),
                   threshold_flips={str(t): sum((s['lex'] < t) != (s['grevlex'] < t) for s in scored)
                                    for t in (1e-3, 1e-2, 5e-2)})
    report = dict(python=platform.python_version(), platform=platform.platform(), sympy=sp.__version__,
                  raw_sha256=hashes, timeout_policy='No timeout; exceptions recorded as unavailable.',
                  summary=summary, by_system=groups, by_system_degree=degrees, records=records)
    Path('results/grader_diagnostics.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    run()
