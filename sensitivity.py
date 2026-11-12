"""Recompute paired statistical checks and numerical-policy sensitivity from graded records."""
import json
from pathlib import Path
from scipy.stats import binomtest


def former_coefficient_counts():
    """Reproduce the former evaluated-float conversion on the same baseline outputs.

    This is an explicitly historical sensitivity calculation, never the current
    acceptance rule. It needs only the original expression strings and no API.
    """
    import sympy as sp
    from sympy.parsing.sympy_parser import (parse_expr, standard_transformations,
        implicit_multiplication_application, convert_xor)
    import systems, grade, grade_outputs
    counts = {"strict": 0, "reconstructed": 0}
    transforms = standard_transformations + (implicit_multiplication_application, convert_xor)
    for line in Path('results/raw_sindy15.jsonl').read_text().splitlines():
        row = json.loads(line)
        system = systems.REGISTRY[row['key']]()
        texts, _ = grade_outputs.extract(row['output'])
        polys, err = [], 0.0
        for text in texts:
            expr = parse_expr(grade.normalise_text(text),
                local_dict={str(v): v for v in system.variables},
                transformations=transforms, evaluate=True)
            for f in expr.atoms(sp.Float):
                r = sp.Rational(float(f)).limit_denominator(1000)
                err = max(err, abs(float(f)-float(r))/abs(float(f)) if f else 0.0)
                expr = expr.subs(f, r)
            polys.append(sp.expand(expr))
        if system.vanishing_task:
            result = grade._grade_vanishing_polys(polys, [], system)
        else:
            result = grade._grade_conserved_poly(polys[0], system)
        if result['verdict'] in grade._RECONSTRUCTABLE and err > 1e-9:
            counts['reconstructed'] += 1
        elif result['verdict'] == 'exactly_correct':
            counts['strict'] += 1
    return counts


def compute():
    rows = [json.loads(line) for line in Path('results/graded_all.jsonl').read_text().splitlines()]
    groups = [('LLMs', [r for r in rows if not r['model'].startswith('implicit')]),
              ('SINDy', [r for r in rows if r['model'].startswith('implicit')])]
    summary = {}
    for name, group in groups:
        pairs = []
        for row in group:
            detail = row['detail']
            if detail.get('literal_verdict', row['verdict']) not in ('plausible_but_wrong', 'implausible'):
                continue
            current = detail.get('relative_drift', detail.get('max_data_residual'))
            old = detail.get('legacy_relative_drift', current)
            if current is not None:
                pairs.append((old, current))
        summary[name] = {'literal_failures_with_fit': len(pairs)}
        for label, index in [('legacy', 0), ('centred', 1)]:
            summary[name][label] = {str(t): sum(p[index] < t for p in pairs) for t in (1e-3, 1e-2, 5e-2)}
    a = [json.loads(l) for l in Path('results/primary/graded_groq.jsonl').read_text().splitlines()]
    b = [json.loads(l) for l in Path('results/primary/graded_cerebras.jsonl').read_text().splitlines()]
    a = {r['sample']: r['verdict'] == 'refused' for r in a if r['system'] == 'twisted_cubic' and r['condition'] == 'B'}
    b = {r['sample']: r['verdict'] == 'refused' for r in b if r['system'] == 'twisted_cubic' and r['condition'] == 'B'}
    assert a.keys() == b.keys()
    n10 = sum(a[k] and not b[k] for k in a)
    n01 = sum(b[k] and not a[k] for k in a)
    summary['former_baseline_coefficient_policy'] = former_coefficient_counts()
    summary['paired_refusal'] = {'groq_only': n10, 'cerebras_only': n01,
        'two_sided_p': binomtest(n10, n10+n01, p=0.5).pvalue if n10+n01 else 1.0}
    return summary


if __name__ == '__main__':
    summary = compute()
    Path('results/sensitivity.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2))
