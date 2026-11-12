"""Scientific regression checks for the expanded policy audit and baseline."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from unittest.mock import patch
import numpy as np
import sympy as sp
import grade
import systems
from fresh_systems import REGISTRY
from policy_audit import reference_criterion
from sr_rational import rational_basis
from grade_outputs import extract


def main():
    s=systems.free_rigid_body()
    E,L=s.conserved_ref
    assert reference_criterion(3*E,[E])
    assert reference_criterion(E+17,[E])
    assert not reference_criterion(0,[E])
    assert not reference_criterion(E+L,[E,L])
    assert not reference_criterion(E**2,[E,L])
    assert grade.grade(str(E+L),s)['verdict']=='exactly_correct'
    assert grade.grade(str(E**2),s)['verdict']=='valid_non_reference'
    assert grade.parse('E = '+str(E),s)[0]==E
    for invalid in ['{"polynomials":"w1"}', '{"polynomials":{"x":"w1"}}',
                    '{"polynomials":[true]}', '{"polynomials":null}',
                    '{"polynomials":0}', '{"polynomials":""}']:
        assert extract(invalid)==([], 'invalid_polynomials_schema')
    for invalid in ['{"polynomials":["w1"],"refuse":"false"}',
                    '{"polynomials":[],"refuse":0}']:
        assert extract(invalid)==([], 'invalid_refusal_schema')
    assert extract('{"polynomials":[],"refuse":false}') == (None, 'refused')
    # A failed reference-span calculation must not become a non-reference claim.
    with patch.object(grade, 'affine_span_membership', side_effect=ArithmeticError('unavailable')):
        assert grade.grade(str(E), s, D=None)['verdict'] == 'grader_unavailable'
    unavailable = grade._reconstruction_result(
        {'verdict': 'grader_unavailable'}, {'verdict': 'exactly_correct'}, 1e-6)
    assert unavailable['verdict'] == 'grader_unavailable'
    with patch.object(grade, 'nf', return_value=None):
        try:
            grade.affine_span_membership(E, [E], s)
        except ArithmeticError:
            pass
        else:
            raise AssertionError('Unavailable normal form reported as nonmembership')
    # Failure of the optional product explanation cannot invalidate a proved law.
    with patch.object(grade, 'affine_span_membership', side_effect=[None, ArithmeticError('unavailable')]):
        result = grade.grade(str(E**2), s, D=None)
        assert result['verdict'] == 'valid_non_reference' and 'unavailable' in result['relation']
    # A known rational two-dimensional nullspace, recovered without references.
    rng=np.random.default_rng(9)
    X=rng.normal(size=(80,2))
    A=np.column_stack([X[:,0],X[:,1],2*X[:,0]+3*X[:,1],X[:,0]-X[:,1]])
    recovered,info=rational_basis(A)
    exact=sp.Matrix([[1,0,2,1],[0,1,3,-1]])
    assert info['nullity']==2
    assert all(exact*sp.Matrix(v)==sp.zeros(2,1) for v in recovered)
    assert sp.Matrix.hstack(*map(sp.Matrix,recovered)).rank()==2
    # Wide matrices need the right-nullspace directions omitted by reduced SVD.
    for array in (np.array([[1.,2.,3.]]), np.array([[1.,2.,3.],[2.,4.,6.]]),
                  np.zeros((1,3))):
        recovered,info=rational_basis(array)
        exact=sp.Matrix(array.astype(int))
        expected_nullity=array.shape[1]-exact.rank()
        assert info['nullity']==expected_nullity
        assert all(exact*sp.Matrix(v)==sp.zeros(array.shape[0],1) for v in recovered)
        assert sp.Matrix.hstack(*map(sp.Matrix,recovered)).rank()==expected_nullity
    for key,factory in REGISTRY.items():
        s=factory()
        assert all(grade.is_conserved(p,s)[0] for p in s.conserved_ref)
        jac=sp.Matrix([[sp.diff(p,v) for v in s.variables] for p in s.conserved_ref])
        point=dict(zip(s.variables,range(1,len(s.variables)+1)))
        assert jac.subs(point).rank()>=2
        assert any(not grade.is_conserved(v,s)[0] for v in s.variables)
    with tempfile.TemporaryDirectory() as tmp:
        raw=Path(tmp)/'raw.jsonl'; graded=Path(tmp)/'graded.jsonl'
        subprocess.run([sys.executable,'run_llm.py','--provider','mock','--model','mock',
                        '--suite','planted','--conditions','A','--n','1','--out',str(raw)],check=True,capture_output=True)
        rs=[json.loads(l) for l in raw.read_text().splitlines()]
        assert len(rs)==6 and all(r['transport_attempts'] and r['max_attempts']==6 for r in rs)
        subprocess.run([sys.executable,'grade_outputs.py','--raw',str(raw),'--out',str(graded)],check=True,capture_output=True)
        assert len(graded.read_text().splitlines())==6
        # Complete/incomplete reconstruction can change beneath one outer label.
        s=systems.twisted_cubic()
        row=dict(model='regression',key='twisted_cubic',system=s.name,
                 condition='B',noise=0.,sample=0,error=None,
                 output=json.dumps(dict(polynomials=['y-x**2+0.0001','0.004*(z-x**3)'],refuse=False)))
        raw.write_text(json.dumps(row)+'\n')
        subprocess.run([sys.executable,'grade_outputs.py','--raw',str(raw),'--out',str(graded)],check=True,capture_output=True)
        result=json.loads(graded.read_text())
        assert set(result['verdict_by_D'].values())=={'exact_after_rounding'}
        assert result['reconstructed_verdict_by_D']=={
            '10':'sound_incomplete','100':'sound_incomplete','1000':'exactly_correct'}
        assert result['rationalisation_sensitive']
        # An unchanged literal coefficient at one cap is not an outcome change.
        s=systems.free_rigid_body()
        row.update(key='rigid_body',system=s.name,
                   output=json.dumps(dict(polynomials=['1.234*w1'],refuse=False)))
        raw.write_text(json.dumps(row)+'\n')
        subprocess.run([sys.executable,'grade_outputs.py','--raw',str(raw),'--out',str(graded)],check=True,capture_output=True)
        result=json.loads(graded.read_text())
        assert set(result['verdict_by_D'].values())=={'implausible'}
        assert result['reconstructed_verdict_by_D']=={
            '10':'implausible','100':'implausible','1000':None}
        assert not result['rationalisation_sensitive']
    print('Revision scientific regressions passed')


if __name__=='__main__': main()
