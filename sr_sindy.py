"""Symbolic-regression tier: implicit-SINDy-style discovery on pysindy's PolynomialLibrary
and finite-difference differentiation, followed by a sequentially thresholded sparse
nullspace (the implicit formulation: find c with Theta(x) c ~ 0, or dTheta/dt c ~ 0 for a
conserved quantity). NOTE: this is our implementation of the implicit formulation on top
of pysindy components, not a call to a pysindy estimator; the posing is documented as a
threat to validity. No knowledge of the constraints is given (no deflation), as for a
generic tool. Writes records in run_llm's raw format so grading is shared."""
import argparse, json, os, time
import numpy as np, sympy as sp
import systems, data

DEG = {"twisted_cubic": 3, "rigid_body": 2, "kepler": 3, "moiety": 2, "acrobot": 3}

def sparse_null(A, thr=0.05, iters=8):
    """Smallest right singular vector, then threshold-and-refit on the surviving support."""
    A = np.asarray(A); norms = np.linalg.norm(A, axis=0) + 1e-30; A = A / norms          # column-normalised, as in the paper's Sec. 4
    supp = np.arange(A.shape[1]); c = None
    for iteration in range(iters):
        _, S, Vt = np.linalg.svd(A[:, supp], full_matrices=False)
        v = Vt[-1]; v = v / np.max(np.abs(v))
        keep = np.abs(v) > thr
        if keep.all() or keep.sum() < 2: c = np.zeros(A.shape[1]); c[supp] = v; break
        if iteration == iters - 1:
            c = np.zeros(A.shape[1]); c[supp] = v
            break
        supp = supp[keep]
    if c is None: c = np.zeros(A.shape[1]); c[supp] = v
    c = c / norms; c = c / np.max(np.abs(c))            # back to monomial coordinates (paper, App. C)
    return c, S[-1], (S[-2] if len(S) > 1 else np.nan)

def discover(system, noise=0.0, seed=0):
    import pysindy as ps
    vs = system.variables; deg = DEG[system.family]
    lib = ps.PolynomialLibrary(degree=deg, include_bias=False)
    if system.vanishing_task:
        X = data.curve_points(system, n=600, seed=seed)
        X = X + np.random.default_rng(seed).normal(0, noise, X.shape) if noise else X
        lib.fit(X); Th = np.asarray(lib.transform(X)); A = np.hstack([Th, np.ones((len(X), 1))])
    else:
        traj = data.trajectories(system, n_traj=6, T=2.0, n_pts=2000, seed=seed)
        rng = np.random.default_rng(seed); rows = []
        fd = ps.FiniteDifference(order=4)
        for tr in traj:
            X = tr + rng.normal(0, noise, tr.shape) if noise else tr
            lib.fit(X); Th = np.asarray(lib.transform(X))
            dt = 2.0 / 1999
            dTh = np.asarray(fd._differentiate(Th, dt))
            rows.append(dTh[5:-5])
        A = np.vstack(rows)
    names = list(lib.get_feature_names([str(v) for v in vs])) + (["1"] if system.vanishing_task else [])
    c, smin, snext = sparse_null(A)
    terms = [f"({float(ci):.12g})*{n.replace(' ', '*').replace('^', '**')}"
             for ci, n in zip(c, names) if abs(ci) > 1e-8]
    return " + ".join(terms) if terms else None, dict(smin=float(smin), snext=float(snext))

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--noise", type=float, default=0.0)
    ap.add_argument("--seeds", type=int, default=3); ap.add_argument("--out", default="results/new_sindy_raw.jsonl")
    ap.add_argument("--systems", default=",".join(systems.DEFAULT), help="keys; use all or perturbed for variants"); a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "a") as f:
        for key in (list(systems.REGISTRY) if a.systems == "all" else systems.PERTURBED if a.systems == "perturbed" else a.systems.split(",")):
            s = systems.REGISTRY[key]()
            for i in range(a.seeds):
                t0 = time.time(); expr, info = discover(s, a.noise, i)
                out = json.dumps({"polynomials": [expr] if expr else [], "refuse": expr is None})
                f.write(json.dumps(dict(model="implicit-sindy(pysindy-lib)", provider="sr", system=s.name, key=key,
                                        condition="B", noise=a.noise, sample=i, temperature=0, prompt="",
                                        output=out, error=None, info=info, t=time.time())) + "\n")
                print(key, i, round(time.time() - t0, 1), "s;", (expr or "")[:90], info)

if __name__ == "__main__":
    main()
