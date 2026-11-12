"""Regenerate data-dependent prose counts and TeX macros from selected results."""
import collections
import json
from pathlib import Path
import re
import statistics

from primary_results import LLM_PAIRS, rows, validate


def compute():
    validate()
    llm = [r for _, path in LLM_PAIRS for r in rows('results/'+path+'.jsonl')]
    raw = [r for path, _ in LLM_PAIRS for r in rows('results/'+path+'.jsonl')]
    baseline = rows('results/graded_sindy15.jsonl')
    audit = json.loads(Path('results/policy_audit.json').read_text())
    sensitivity = json.loads(Path('results/sensitivity.json').read_text())
    count = lambda records, verdict: sum(r['verdict'] == verdict for r in records)
    good = {'exactly_correct', 'valid_non_reference'}
    multi = [r for r in llm if r.get('verdicts_all')]
    gaps = [r['gap_points'] for r in audit['cells'] if r['model'] != 'implicit-sindy(pysindy-lib)']
    nonzero = [g for g in gaps if g > 0]
    out = dict(nLLM=len(llm), nBaseline=len(baseline), nPrimary=len(llm)+len(baseline),
               nExact=count(llm, 'exactly_correct'), nNonReference=count(llm, 'valid_non_reference'),
               nFirstValid=audit['summary']['llm_first_valid'], nAllValid=audit['summary']['llm_all_valid'],
               nGapCells=len(nonzero), nZeroGapCells=len(gaps)-len(nonzero),
               minGap=min(nonzero), maxGap=max(nonzero), medianGap=statistics.median(gaps),
               medianPositiveGap=statistics.median(nonzero), nMulti=len(multi),
               nAllLoss=sum(r['verdict'] in good and any(v not in good for v in r['verdicts_all']) for r in multi),
               nBestGain=sum(r['verdict'] not in good and any(v in good for v in r['verdicts_all']) for r in multi),
               nUnparseable=count(llm, 'unparseable'),
               nDecimal=sum(bool(re.search(r'\d\.\d', r['output'] or '')) for r in raw),
               nCriterionTaskOnly=sum(r['task_only'] for r in audit['comparisons'] if r['model'] != 'implicit-sindy(pysindy-lib)'),
               nCriterionOnly=sum(r['criterion_only'] for r in audit['comparisons']),
               nFit=sensitivity['LLMs']['literal_failures_with_fit'],
               nFitLow=sensitivity['LLMs']['centred']['0.01'], nFitMid=sensitivity['LLMs']['centred']['0.05'],
               nLegacyLow=sensitivity['LLMs']['legacy']['0.01'], nLegacyMid=sensitivity['LLMs']['legacy']['0.05'],
               llm_verdicts=dict(collections.Counter(r['verdict'] for r in llm)),
               baseline_verdicts=dict(collections.Counter(r['verdict'] for r in baseline)),
               deployments={}, multi_cells=[])
    for model, label in [('openai/gpt-oss-120b', 'Groq'), ('gpt-oss-120b', 'Cerebras'), ('gemini-3.5-flash-lite', 'Gemini')]:
        rs = [r for r in llm if r['model'] == model]
        rr = [r for r in raw if r['model'] == model]
        ms = [r for r in multi if r['model'] == model]
        out['deployments'][label] = dict(n=len(rs), verdicts=dict(collections.Counter(r['verdict'] for r in rs)),
              truncated=sum(r['finish_reason'] in ('length', 'MAX_TOKENS') for r in rr),
              multi=len(ms), multi_all_valid=sum(all(v in good for v in r['verdicts_all']) for r in ms))
    for cell in audit['cells']:
        if cell['model'] == 'implicit-sindy(pysindy-lib)':
            continue
        loss = cell['task_valid']-cell['all_valid']
        if loss:
            out['multi_cells'].append(dict(model=cell['model'], system=cell['system'], condition=cell['condition'], loss=loss))
    # Classify every recovered Kepler reference explicitly; do not infer by exclusion.
    import grade, systems
    kepler = collections.Counter()
    for r in llm:
        if r['system'] in ('kepler', 'kepler_pert') and r['verdict'] == 'exactly_correct':
            s = systems.REGISTRY[r['system']]()
            expression, note = grade.parse(r['detail']['poly'], s)
            assert expression is not None, note
            matches = [label for label, ref in zip(('E', 'L', 'Ax', 'Ay'), s.conserved_ref)
                       if grade.affine_span_membership(expression, [ref], s)]
            kepler[matches[0] if len(matches) == 1 else 'other'] += 1
    out['kepler_references'] = dict(kepler)
    out['nKepler'] = sum(kepler.values())
    out['nKeplerEnergy'] = kepler['E']
    out['nKeplerAngular'] = kepler['L']
    acro = [r for r in llm+baseline if r['system'].startswith('acrobot')]
    out['nAcrobot'] = len(acro)
    out['nAcrobotExact'] = count(acro, 'exactly_correct')
    out['quadratic_nonreference'] = sum(r['verdict']=='valid_non_reference' and
        r['detail'].get('relation') == 'polynomial (deg<=2) in references' for r in llm)
    cubic = []
    for r in llm:
        if r['verdict'] != 'valid_non_reference' or r['detail'].get('relation') == 'polynomial (deg<=2) in references':
            continue
        s = next(factory() for factory in systems.REGISTRY.values() if factory().name == r['system'])
        expression, _ = grade.parse(r['detail']['poly'], s)
        matches = [str(ref) for ref in s.conserved_ref
                   if grade.nf(expression-ref**3, s.constraints, s.variables) == 0]
        assert matches, 'Non-reference answer requires a new dependence analysis'
        cubic.append(dict(model=r['model'], system=r['system'], condition=r['condition'],
                          sample=r['sample'], polynomial=r['detail']['poly'], cubed_reference=matches[0]))
    out['cubic_nonreference'] = cubic
    rational = rows('results/graded_rational15.jsonl')
    out['rational_baseline'] = dict(n=len(rational),
        first_valid=sum(r['verdict'] in good for r in rational),
        any_valid=sum(any(v in good for v in (r.get('verdicts_all') or [r['verdict']])) for r in rational),
        all_valid=sum(all(v in good for v in (r.get('verdicts_all') or [r['verdict']])) for r in rational),
        curve_complete=sum(r['verdict']=='exactly_correct' for r in rational if r['system'].startswith('twisted')))
    cas_path = Path('results/singular_crosscheck.json')
    if cas_path.exists():
        cas = json.loads(cas_path.read_text())
        out.update(nCASLiteral=cas['checked_literal_rows'], nCASReconstructed=cas['checked_reconstructed_rows'],
                   nCASNonalgebraic=cas['nonalgebraic_rows'])
    return out


def main():
    summary = compute()
    Path('results/paper_summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    macros = []
    for name, value in summary.items():
        if isinstance(value, (int, float)) and name.isalpha():
            macros.append('\\newcommand{\\'+name+'}{'+f'{value:g}'+'}')
    Path('docs').mkdir(exist_ok=True)
    Path('docs/claims_generated.tex').write_text('\n'.join(macros)+'\n')
    print('Rebuilt paper summary and numeric macros from corrected results.')


if __name__ == '__main__':
    main()
