"""Cross-check primary algebraic decisions in an independent CAS (Singular).

Shared restricted parsing and system definitions; independent differentiation,
Groebner bases, ideal equality, quotient triviality and affine-reference span.
Numerical diagnostics and extraction/refusal decisions are outside this check.
Usage: python crosscheck_singular.py --singular /path/to/Singular
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import sympy as sp
import grade
import systems
from grade_outputs import extract
from primary_results import PRIMARY_PAIRS

PROCEDURES = r'''
proc inspan(poly p, ideal refs)
{
  list basis;
  poly q;
  int i,j,changed;
  for(i=1;i<=size(refs);i++)
  {
    q=refs[i]; changed=1;
    while(q!=0 && changed)
    {
      changed=0;
      for(j=1;j<=size(basis);j++)
      {
        if(leadmonom(q)==leadmonom(basis[j]))
        {q=q-leadcoef(q)/leadcoef(basis[j])*basis[j]; changed=1; break;}
      }
    }
    if(q!=0){basis[size(basis)+1]=q;}
  }
  changed=1;
  while(p!=0 && changed)
  {
    changed=0;
    for(j=1;j<=size(basis);j++)
    {
      if(leadmonom(p)==leadmonom(basis[j]))
      {p=p-leadcoef(p)/leadcoef(basis[j])*basis[j]; changed=1; break;}
    }
  }
  return(p==0);
}
proc conserved(poly p, ideal constraints, ideal field, ideal refs)
{
  ideal gb=std(constraints);
  poly q=reduce(p,gb);
  if(deg(q)<=0){return("trivial");}
  poly derivative=0;
  int i;
  for(i=1;i<=nvars(basering);i++){derivative=derivative+diff(p,var(i))*field[i];}
  if(reduce(derivative,gb)!=0){return("failure");}
  ideal rr=1,reduce(refs,gb);
  if(inspan(q,rr)){return("reference");}
  return("valid");
}
proc vanishing(ideal candidates, ideal constraints)
{
  ideal gb=std(constraints);
  if(size(reduce(candidates,gb))!=0){return("failure");}
  if(size(reduce(constraints,std(candidates)))==0){return("reference");}
  return("incomplete");
}
'''


def polynomial(p):
    p=sp.sympify(p)
    variables=sorted(p.free_symbols,key=str)
    if not variables: return '('+str(p)+')'
    terms=[]
    for exponents,c in sp.Poly(p,*variables,domain='QQ').terms():
        factors=['('+str(c)+')']
        factors += ['('+str(v)+'^'+str(e)+')' for v,e in zip(variables,exponents) if e]
        terms.append('*'.join(factors))
    return '('+'+'.join(terms)+')'


def expected(verdict):
    return {'exactly_correct':'reference', 'valid_non_reference':'valid',
            'correct_but_trivial':'trivial', 'sound_incomplete':'incomplete',
            'plausible_but_wrong':'failure', 'implausible':'failure'}[verdict]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--singular', default='Singular')
    a = parser.parse_args()
    commands = [PROCEDURES]
    cases = []
    skipped = []
    ring_keys = {}
    primary_count = 0
    source_hashes = {}
    for rn, gn in PRIMARY_PAIRS:
        raw = [json.loads(l) for l in Path(f'results/{rn}.jsonl').read_text().splitlines()]
        gs = [json.loads(l) for l in Path(f'results/{gn}.jsonl').read_text().splitlines()]
        primary_count += len(raw)
        source_hashes[rn] = hashlib.sha256(Path(f'results/{rn}.jsonl').read_bytes()).hexdigest()
        assert len(raw)==len(gs)
        for index,(r,g) in enumerate(zip(raw,gs)):
            assert all(r[k]==g[k] for k in ('model','system','sample','condition','noise'))
            s = systems.REGISTRY[r['key']]()
            strings, note = extract(r['output'])
            if not strings or g['verdict'] in {'unparseable','grader_unavailable'}:
                skipped.append(dict(file=rn,row=index+1,verdict=g['verdict']))
                continue
            if r['key'] not in ring_keys:
                ring = 'r'+str(len(ring_keys))
                ring_keys[r['key']] = ring
                commands.append(f'ring {ring}=0,({",".join(map(str,s.variables))}),dp;')
                commands.append('ideal constraints='+','.join(map(polynomial,s.constraints or [0]))+';')
                commands.append('ideal field='+','.join(map(polynomial,s.field or [0]))+';')
                commands.append('ideal refs='+','.join(map(polynomial,s.conserved_ref or [0]))+';')
            commands.append('setring '+ring_keys[r['key']]+';')
            variants = [(None,g['detail'].get('literal_verdict',g['verdict']))]
            if g['verdict']=='exact_after_rounding':
                variants.append((1000,g['detail']['rounded_verdict']))
            for D, verdict in variants:
                ps = [grade.parse(t,s,D=D)[0] for t in (strings if s.vanishing_task else strings[:1])]
                if any(p is None for p in ps):
                    raise ValueError('A previously parseable expression failed to parse')
                ident = len(cases)
                if s.vanishing_task:
                    commands.append('ideal candidates='+','.join(map(polynomial,ps))+';')
                    expr = 'vanishing(candidates,constraints)'
                else:
                    expr = f'conserved({polynomial(ps[0])},constraints,field,refs)'
                commands.append(f'print("CHECK {ident} "+{expr});')
                cases.append(dict(file=rn,row=index+1,reconstruction_D=D,expected=expected(verdict)))
    commands.append('quit;')
    script='\n'.join(commands)+'\n'
    Path('results/singular_checks.sing').write_text(script)
    with tempfile.TemporaryDirectory() as tmp:
        path=Path(tmp)/'audit.sing'; path.write_text(script)
        run=subprocess.run([a.singular,'-q',str(path)],capture_output=True,text=True,timeout=600)
    if run.returncode or '?' in run.stdout or run.stderr.strip():
        Path('/tmp/singular-crosscheck-errors.txt').write_text(run.stdout+'\n'+run.stderr)
        raise RuntimeError(run.stdout[:4000]+'\n'+run.stderr[-2000:])
    answers={}
    for line in run.stdout.splitlines():
        if line.startswith('CHECK '):
            _,i,label=line.split(); answers[int(i)]=label
    mismatches=[dict(case,actual=answers.get(i)) for i,case in enumerate(cases) if answers.get(i)!=case['expected']]
    version=subprocess.run([a.singular,'--version'],capture_output=True,text=True).stdout.splitlines()[0]
    report=dict(cas=version, primary_rows=primary_count, source_raw_sha256=source_hashes,
                checked_literal_rows=sum(c['reconstruction_D'] is None for c in cases),
                checked_reconstructed_rows=sum(c['reconstruction_D'] is not None for c in cases),
                nonalgebraic_rows=len(skipped), checks=len(cases), mismatches=mismatches,
                scope='Shared parser and system definitions; independent Singular differentiation, ideal membership/equality, quotient triviality and affine reference-span test. Numerical proximity and nonalgebraic verdicts excluded.',
                script_sha256=hashlib.sha256(script.encode()).hexdigest())
    Path('results/singular_crosscheck.json').write_text(json.dumps(report,indent=2)+'\n')
    Path('results/singular_checks.sing').write_text(script)
    print(json.dumps(report,indent=2))
    if mismatches: raise SystemExit(1)


if __name__=='__main__':
    main()
