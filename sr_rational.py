"""Numerical multi-output nullspace baseline with explicit rational recovery.

Uses sampled states/finite differences only, never reference polynomials or the
symbolic field. QR fixes coordinates in the estimated nullspace before rational
recovery, avoiding an arbitrary SVD mixture. All recovered basis vectors are
returned, including possible trivial constraints; the grader decides validity.
"""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from scipy.linalg import qr
import sympy as sp
import systems
import data
from sr_sindy import DEG


def rational_basis(A, denominator=1000, rtol=1e-7):
    norms = np.linalg.norm(A, axis=0)
    # Truly zero columns represent exact numerical null directions too.
    scales = np.where(norms > 0, norms, 1.)
    # A reduced SVD omits n-m right-nullspace directions when m < n.
    # Keep the reduced form for tall discovery matrices to avoid a large U.
    _, singular, vt = np.linalg.svd(A/scales, full_matrices=A.shape[0] < A.shape[1])
    rank = int(np.sum(singular > rtol*singular[0])) if singular[0] else 0
    V = vt[rank:].T / scales[:, None]
    if V.shape[1] == 0:
        return [], dict(nullity=0, rank=rank, singular_values=singular.tolist())
    _, _, pivots = qr(V.T, pivoting=True, mode='economic')
    basis = np.linalg.solve(V[pivots[:V.shape[1]], :].T, V.T).T
    recovered = [[sp.Rational(float(c)).limit_denominator(denominator) for c in v]
                 for v in basis.T]
    residuals = [float(np.linalg.norm(A@np.array(v, dtype=float))/
                       max(np.linalg.norm(A)*np.linalg.norm(np.array(v, dtype=float)), 1e-300))
                 for v in recovered]
    return recovered, dict(nullity=V.shape[1], rank=rank, singular_values=singular.tolist(),
                           recovered_relative_residuals=residuals)


def discover(s, seed, denominator=1000, rtol=1e-7):
    import pysindy as ps
    lib = ps.PolynomialLibrary(degree=DEG[s.family], include_bias=s.vanishing_task)
    if s.vanishing_task:
        points = data.curve_points(s, n=600, seed=seed)
        lib.fit(points)
        A = np.asarray(lib.transform(points))
    else:
        trajectories = data.trajectories(s, n_traj=6, T=2., n_pts=2000, seed=seed)
        rows = []
        for X in trajectories:
            lib.fit(X)
            theta = np.asarray(lib.transform(X))
            rows.append(np.asarray(ps.FiniteDifference(order=4)._differentiate(theta, 2./1999))[5:-5])
        A = np.vstack(rows)
    monomials = [sp.prod(v**int(e) for v,e in zip(s.variables, powers)) for powers in lib.powers_]
    basis, info = rational_basis(A, denominator, rtol)
    polynomials = [str(sp.expand(sum(c*m for c,m in zip(vector, monomials)))) for vector in basis]
    return polynomials, info


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--systems', default='all')
    p.add_argument('--seeds', type=int, default=15)
    p.add_argument('--denominator', type=int, default=1000)
    p.add_argument('--rtol', type=float, default=1e-7)
    p.add_argument('--out', required=True, help='New file only; never overwrite archived runs')
    a = p.parse_args()
    keys = list(systems.REGISTRY) if a.systems == 'all' else a.systems.split(',')
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, 'x') as f:
        for key in keys:
            s = systems.REGISTRY[key]()
            for seed in range(a.seeds):
                start = time.monotonic()
                polys, info = discover(s, seed, a.denominator, a.rtol)
                r = dict(model='rational-nullspace', provider='sr', key=key, system=s.name,
                         condition='B', sample=seed, noise=0., temperature=0., prompt='',
                         output=json.dumps(dict(polynomials=polys, refuse=not polys)), error=None,
                         settings=dict(degree=DEG[s.family], denominator=a.denominator,
                                       relative_svd_threshold=a.rtol, n_traj=6, T=2., n_pts=2000,
                                       differentiation_order=4, trim=5, curve_points=600),
                         info=info, elapsed_seconds=time.monotonic()-start, t=time.time())
                f.write(json.dumps(r)+'\n'); f.flush()
                print(key, seed, 'nullity', info['nullity'], flush=True)


if __name__ == '__main__':
    main()
