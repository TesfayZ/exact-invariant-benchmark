"""Trajectory generation (high-accuracy integration of the system's own vector field)
and the data-fit functions used by the 'plausible but wrong' bucket."""
import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp
import ideal_diff


def _rhs(system):
    vs = system.variables
    f = sp.lambdify(vs, [e / system.field_den for e in system.field], "numpy")
    return lambda t, y: np.array(f(*y), dtype=float)


def initial_states(system, n, rng):
    name = system.family
    if name == "rigid_body":
        return [rng.uniform(-2, 2, 3) for _ in range(n)]
    if name == "kepler":
        out = []
        while len(out) < n:
            r0 = rng.uniform(0.8, 1.5); th = rng.uniform(0, 2 * np.pi)
            x, y = r0 * np.cos(th), r0 * np.sin(th)
            v = rng.uniform(0.5, 0.9) / np.sqrt(r0)   # bound orbits
            ph = rng.uniform(0, 2 * np.pi)
            out.append(np.array([x, y, v * np.cos(ph), v * np.sin(ph), 1 / r0]))
        return out
    if name == "moiety":
        return [rng.uniform(0.2, 2.0, 4) for _ in range(n)]
    if name == "acrobot":
        out = []
        for _ in range(n):
            t1, t2 = rng.uniform(-1, 1, 2); w = rng.uniform(-1, 1, 2)
            out.append(np.array([np.cos(t1), np.sin(t1), np.cos(t2), np.sin(t2), *w]))
        return out
    raise ValueError(name)


def trajectories(system, n_traj=8, T=5.0, n_pts=400, seed=0):
    rng = np.random.default_rng(seed)
    rhs = _rhs(system)
    ts = np.linspace(0, T, n_pts)
    trajs = []
    for y0 in initial_states(system, n_traj, rng):
        sol = solve_ivp(rhs, (0, T), y0, t_eval=ts, method="DOP853", rtol=1e-12, atol=1e-13)
        if not sol.success or not np.all(np.isfinite(sol.y)):
            raise RuntimeError(f"solve_ivp failed for {system.name} from y0={y0}: {sol.message}")
        trajs.append(sol.y.T)
    return trajs


def make_drift_fn(system, trajs):
    """Drift on the fixed quotient representative after removing its constant.

    This is invariant to adding constants or multiples of known constraints.
    The legacy score is retained only for sensitivity analysis.
    """
    vs = system.variables
    gb = sp.groebner(system.constraints, *vs, order="grevlex", domain="QQ") if system.constraints else None
    def fn(p):
        p = sp.expand(p)
        representative = gb.reduce(p)[1] if gb is not None else p
        centred = sp.Poly(representative, *vs, domain="QQ")
        q = sp.expand(representative - centred.TC())
        return ideal_diff.relative_drift(q, trajs, vs)
    fn.legacy = lambda p: ideal_diff.relative_drift(sp.sympify(p), trajs, vs)
    return fn


def make_residual_fn(system, pts):
    def fn(p):
        return ideal_diff.relative_residual(sp.sympify(p), pts, system.variables)
    return fn


def curve_points(system, n=2000, seed=0):
    """Sample points on the variety for vanishing tasks (twisted cubic)."""
    rng = np.random.default_rng(seed)
    t = rng.uniform(-1.5, 1.5, n)
    return np.stack([np.asarray(c, dtype=float) * np.ones_like(t) for c in system.curve(t)], axis=1)
