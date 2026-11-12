"""Complete paired audit; never modifies raw outputs or the primary grading policy.

SRBench comparison: transfer its constant-difference/constant-ratio criterion
to each conservation reference. Shared literal parsing; no quotient reduction,
R2 gate, estimator cleaning or float rounding. See vendor/srbench/NOTICE.md.
"""
import collections
import json
from pathlib import Path
import statistics
import sympy as sp
import systems
import grade
from grade_outputs import extract
from primary_results import PRIMARY_PAIRS

GOOD = {'exactly_correct', 'valid_non_reference'}
BASELINE = 'implicit-sindy(pysindy-lib)'


def reference_criterion(p, references):
    """SRBench symbolic acceptance criterion on already parsed exact expressions."""
    if p == 0:
        return False
    for ref in references:
        variables = sorted(p.free_symbols | ref.free_symbols, key=str)
        if not variables or sp.Poly(ref-p, *variables).total_degree() <= 0:
            return True
        pc, rc = sp.Poly(p, *variables).as_dict(), sp.Poly(ref, *variables).as_dict()
        ratio = pc.get(next(iter(rc)), 0)/next(iter(rc.values()))
        if ratio != 0 and pc == {m: ratio*c for m,c in rc.items()}:
            return True
    return False


def run():
    graded = []
    records = []
    failures = []
    cache = {}
    for raw_name, grade_name in PRIMARY_PAIRS:
        raw = [json.loads(l) for l in Path(f'results/{raw_name}.jsonl').read_text().splitlines()]
        gs = [json.loads(l) for l in Path(f'results/{grade_name}.jsonl').read_text().splitlines()]
        if len(raw) != len(gs):
            raise ValueError('Raw/graded row count mismatch')
        for index, (r, g) in enumerate(zip(raw, gs)):
            assert all(r[k] == g[k] for k in ('model', 'system', 'condition', 'sample', 'noise'))
            if r['key'] not in cache:
                cache[r['key']] = systems.REGISTRY[r['key']]()
            s = cache[r['key']]
            polys, note = extract(r['output'])
            first = g['verdict'] in GOOD
            all_valid = first and all(v in GOOD for v in (g.get('verdicts_all') or []))
            result = dict(raw_file=f'results/{raw_name}.jsonl', row=index+1,
                          model=r['model'], system=r['system'], condition=r['condition'],
                          noise=r['noise'], sample=r['sample'], reference=g['verdict']=='exactly_correct',
                          task_valid=first, all_valid=all_valid, criterion=None)
            if not s.vanishing_task and polys:
                p, parse_note = grade.parse(polys[0], s)
                if p is not None:
                    result['criterion'] = reference_criterion(p, s.conserved_ref)
                    result['expression'] = polys[0]
                    result['verdict'] = g['verdict']
            if g['verdict'] == 'unparseable':
                details = []
                for text in polys or []:
                    p, why = grade.parse(text, s)
                    entry = dict(text=text, parser_note=why)
                    # Diagnostic only: never silently fix or regrade the archive.
                    if p is None and '=' in text:
                        lhs, rhs = text.split('=', 1)
                        if lhs.strip() not in {str(v) for v in s.variables}:
                            q, qnote = grade.parse(rhs, s)
                            if q is not None:
                                entry['label_rhs_verdict'] = grade.grade(rhs, s)['verdict']
                    details.append(entry)
                for entry in details:
                    if 'ω' in entry['text']:
                        alias = entry['text'].replace('ω', 'w')
                        q, _ = grade.parse(alias,s)
                        if q is not None:
                            entry['notation_only_w_alias_verdict'] = grade.grade(alias,s)['verdict']
                failures.append(dict(raw_file=result['raw_file'], row=index+1,
                                     model=r['model'], system=r['system'], condition=r['condition'],
                                     output=r['output'], extraction_note=note, details=details,
                                     grader_detail=g['detail']))
            records.append(result)
            graded.append(g)
    cells = []
    groups = collections.defaultdict(list)
    for r, g in zip(records, graded):
        groups[(r['model'], r['system'], r['condition'], r['noise'])].append((r, g))
    for key, pairs in sorted(groups.items()):
        n = len(pairs)
        ref = sum(r['reference'] for r, _ in pairs)
        valid = sum(r['task_valid'] for r, _ in pairs)
        cells.append(dict(model=key[0], system=key[1], condition=key[2], noise=key[3], n=n,
                          reference=ref, task_valid=valid, all_valid=sum(r['all_valid'] for r,_ in pairs),
                          permissive=sum(g['verdict'] in GOOD | {'correct_but_trivial','sound_incomplete','exact_after_rounding'} for _,g in pairs),
                          gap_points=100*(valid-ref)/n))
    comparisons = []
    for model in sorted({r['model'] for r in records}):
        rs = [r for r in records if r['model']==model and r['criterion'] is not None]
        comparisons.append(dict(model=model, n=len(rs), criterion=sum(r['criterion'] for r in rs),
                                task_valid=sum(r['task_valid'] for r in rs),
                                both=sum(r['criterion'] and r['task_valid'] for r in rs),
                                criterion_only=sum(r['criterion'] and not r['task_valid'] for r in rs),
                                task_only=sum(r['task_valid'] and not r['criterion'] for r in rs)))
    llm_cells = [c for c in cells if c['model'] != BASELINE]
    gaps = [c['gap_points'] for c in llm_cells]
    summary = dict(llm_cells=len(gaps), nonzero_gap_cells=sum(g>0 for g in gaps),
                   median_gap_points=statistics.median(gaps),
                   median_nonzero_gap_points=statistics.median(g for g in gaps if g>0),
                   maximum_gap_points=max(gaps), unparseable=len(failures),
                   llm_first_valid=sum(r['task_valid'] for r in records if r['model']!=BASELINE),
                   llm_all_valid=sum(r['all_valid'] for r in records if r['model']!=BASELINE))
    output = dict(summary=summary, cells=cells, comparisons=comparisons,
                  criterion_scope='parseable first-polynomial conserved outputs; exact-parser transfer of SRBench symbolic criterion; no R2 gate/float rounding/quotient reduction',
                  disagreements=[r for r in records if r['criterion'] is not None and r['criterion'] != r['task_valid']],
                  unparseable_audit=failures)
    Path('results/policy_audit.json').write_text(json.dumps(output, indent=2)+'\n')
    Path('results/policy_audit_records.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
    print(json.dumps(summary, indent=2))
    print(json.dumps(comparisons, indent=2))


if __name__ == '__main__':
    run()
