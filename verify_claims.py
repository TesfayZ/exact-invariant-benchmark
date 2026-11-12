"""Check pinned paper claims, exact dataset scope and independently checked evidence."""
import json
from pathlib import Path
import sys
import sympy as sp
import grade
import systems
from primary_results import PRIMARY_PAIRS, digest, rows, validate
from summarize_results import compute


def main():
    checks = []
    def check(name, actual, expected=True):
        checks.append((name, actual == expected))
        print(('ok  ' if actual == expected else 'FAIL')+' '+name+
              ('' if actual == expected else f' computed={actual!r} expected={expected!r}'))
    validate()
    actual = compute()
    pinned = json.loads(Path('paper_claims.json').read_text())
    for key in sorted(set(actual) | set(pinned)):
        check('Paper claim '+key, actual.get(key), pinned.get(key))
    check('Summary artifact matches recomputation', json.loads(Path('results/paper_summary.json').read_text()), actual)
    macros = ''.join('\\newcommand{\\'+key+'}{'+f'{value:g}'+'}\n'
                     for key,value in actual.items() if isinstance(value, (int,float)) and key.isalpha())
    check('Manuscript numeric macros match data', Path('docs/claims_generated.tex').read_text(), macros)
    combined = []
    for raw_name, grade_name in PRIMARY_PAIRS:
        raw, graded = rows('results/'+raw_name+'.jsonl'), rows('results/'+grade_name+'.jsonl')
        check('Raw/graded identities '+raw_name,
              [(r['model'],r['system'],r['condition'],r['noise'],r['sample']) for r in raw],
              [(r['model'],r['system'],r['condition'],r['noise'],r['sample']) for r in graded])
        combined.extend(graded)
    check('Combined dataset includes only selected primary runs', rows('results/graded_all.jsonl'), combined)
    rational = rows('results/graded_rational15.jsonl')
    for raw_name, grade_name in [('raw_sindy15', 'graded_sindy15'),
                                 ('raw_rational15', 'graded_rational15')]:
        raw, graded = rows('results/'+raw_name+'.jsonl'), rows('results/'+grade_name+'.jsonl')
        check('Numerical raw/graded identities '+raw_name,
              [(r['model'], r['system'], r['condition'], r['noise'], r['sample']) for r in raw],
              [(r['model'], r['system'], r['condition'], r['noise'], r['sample']) for r in graded])
        check('Numerical system/seed coverage '+raw_name,
              sorted((r['key'], r['condition'], r['noise'], r['sample']) for r in raw),
              sorted((key, 'B', 0.0, seed) for key in systems.REGISTRY for seed in range(15)))
    check('Numerical baseline scope', actual['rational_baseline'],
          dict(n=150,first_valid=134,any_valid=135,all_valid=98,curve_complete=30))
    cas = json.loads(Path('results/singular_crosscheck.json').read_text())
    check('Independent CAS covers selected raw files',cas['source_raw_sha256'],
          {rn:digest('results/'+rn+'.jsonl') for rn,_ in PRIMARY_PAIRS})
    check('Independent CAS rows', cas['primary_rows'], 1350)
    check('Independent CAS mismatches', cas['mismatches'], [])
    check('Independent CAS generated script hash', cas['script_sha256'], digest('results/singular_checks.sing'))
    for key in ('acrobot','acrobot_pert'):
        s = systems.REGISTRY[key]()
        aux = sp.Symbol('saturation_aux')
        basis = sp.groebner(s.constraints+[1-aux*s.field_den], aux,*s.variables,order='lex',domain='QQ')
        saturated = [p.as_expr() for p in basis.polys if aux not in p.as_expr().free_symbols]
        original = sp.groebner(s.constraints,*s.variables,order='lex',domain='QQ')
        check(key+' denominator saturation unchanged', saturated, [p.as_expr() for p in original.polys])
        z=s.variables[2]
        den=sp.Poly(s.field_den,z)
        check(key+' denominator positive on circle',den.degree()==2 and den.coeff_monomial(z)==0
              and den.coeff_monomial(z**2)<0 and s.field_den.subs(z,1)>0)
        g1,g2=s.constraints
        check(key+' constraint combinations are trivial', grade.grade_conserved(str(g1**2+g1*g2+7),s,D=None)['verdict'], 'correct_but_trivial')
    check('All LLM non-reference answers explained by reference products/cube',
          actual['quadratic_nonreference']+len(actual['cubic_nonreference']), actual['nNonReference'])
    check('Kepler classifications have no ambiguous residual category',actual['kepler_references'].get('other',0),0)
    check('LLM exact-after-reconstruction count',actual['llm_verdicts'].get('exact_after_rounding',0),0)
    failures=sum(not good for _,good in checks)
    print(f'\n{failures} mismatch(es)')
    return bool(failures)


if __name__=='__main__':
    sys.exit(main())
