"""Ground-truth systems for the exact-verification benchmark.

Each System carries: variables, a vanishing-constraint generating set (an ideal, canonical), a
polynomial vector field for the Lie-derivative conservation test, and reference conserved
quantities, all exact over QQ. CANONICAL systems are textbook forms; PERTURBED systems change
parameters/structure and rename variables (parameter and coordinate variants).
"""
import os, sys
from dataclasses import dataclass
import sympy as sp

_REPO = os.environ.get("ALGEBRAIC_RL_REPO") or str(
    __import__("pathlib").Path(__file__).resolve().parent / "vendor" / "algebraicRLtest")
if not os.path.isfile(os.path.join(_REPO, "ideal_diff.py")):
    raise RuntimeError("algebraicRLtest backend missing: use the bundled vendor/algebraicRLtest "
                       "directory or set ALGEBRAIC_RL_REPO to an existing checkout.")
sys.path.insert(0, _REPO)  # github.com/TesfayZ/algebraicRLtest
# Propagate the backend path to child processes (test_pipeline.py spawns
# run_llm.py / grade_outputs.py, which do not import systems first).
os.environ["PYTHONPATH"] = _REPO + os.pathsep + os.environ.get("PYTHONPATH", "")

R = sp.Rational


@dataclass
class System:
    name: str
    family: str                     # sampler dispatch: rigid_body | kepler | moiety | acrobot | twisted_cubic
    variables: list
    constraints: list               # generators of the vanishing ideal (may be [])
    field: list = None              # d(var)/dt numerators
    field_den: object = 1           # dvar/dt = field / field_den
    conserved_ref: list = None      # reference conserved quantities
    vanishing_task: bool = False    # task = state generators of `constraints`
    description: str = ""
    perturbed: bool = False
    curve: object = None            # vanishing tasks: t -> point on the variety
    curve_text: str = ""


def _syms(names):
    return sp.symbols(" ".join(names), real=True)


def twisted_cubic(names=("x", "y", "z"), perturbed=False):
    x, y, z = _syms(names)
    if not perturbed:
        cons, curve, txt = [y - x**2, z - x**3], (lambda t: (t, t**2, t**3)), "(t, t^2, t^3)"
    else:   # (t, 2t^2 - 1, t^3 + t)
        cons, curve, txt = [y - 2 * x**2 + 1, z - x**3 - x], (lambda t: (t, 2 * t**2 - 1, t**3 + t)), "(t, 2t^2 - 1, t^3 + t)"
    return System("twisted_cubic" + ("_pert" if perturbed else ""), "twisted_cubic", [x, y, z], cons,
                  vanishing_task=True, perturbed=perturbed, curve=curve, curve_text=txt,
                  description=(f"Points ({names[0]}, {names[1]}, {names[2]}) in three-dimensional space. "
                    f"State generators of all polynomial constraints."))


def free_rigid_body(I=(1, 2, 3), names=("w1", "w2", "w3"), perturbed=False):
    w1, w2, w3 = _syms(names)
    I1, I2, I3 = [sp.Integer(i) for i in I]
    f = [(I2 - I3) / I1 * w2 * w3, (I3 - I1) / I2 * w3 * w1, (I1 - I2) / I3 * w1 * w2]
    return System(f"rigid_body_{I[0]}_{I[1]}_{I[2]}" + ("_pert" if perturbed else ""), "rigid_body", [w1, w2, w3], [],
                  field=[sp.nsimplify(e) for e in f], conserved_ref=[I1 * w1**2 + I2 * w2**2 + I3 * w3**2,
                                                                    I1**2 * w1**2 + I2**2 * w2**2 + I3**2 * w3**2],
                  perturbed=perturbed, description=f"Torque-free Euler equations, principal axes, inertias {I}.")


def kepler(mu=R(1), names=("x", "y", "vx", "vy", "u"), perturbed=False):
    """Planar Kepler, lifted with u = 1/r (constraint u^2 (x^2+y^2) = 1)."""
    x, y, vx, vy, u = _syms(names)
    mu = sp.nsimplify(mu)
    fld = [vx, vy, -mu * x * u**3, -mu * y * u**3, -(x * vx + y * vy) * u**3]
    L = x * vy - y * vx
    E = R(1, 2) * (vx**2 + vy**2) - mu * u
    Ax, Ay = vy * L - mu * x * u, -vx * L - mu * y * u          # Laplace-Runge-Lenz
    nm = ", ".join(names)
    return System("kepler" + ("_pert" if perturbed else ""), "kepler", [x, y, vx, vy, u], [u**2 * (x**2 + y**2) - 1],
                  field=fld, conserved_ref=[E, L, Ax, Ay], perturbed=perturbed,
                  description=f"Planar Kepler (mu={mu}) in ({nm}) where {names[4]} = 1/r and {names[4]}^2({names[0]}^2+{names[1]}^2) = 1.")


def moiety_network(variant=0, names=("A", "B", "C", "D"), perturbed=False):
    A, B, C, D = _syms(names)
    if variant == 0:   # A + B <-> C, C -> D
        r1, r2 = 2 * A * B - C, R(3, 2) * C
        fld, refs, desc = [-r1, -r1, r1 - r2, r2], [A - B, A + C + D], "A+B<->C, C->D mass action, rate constants 2,1,3/2."
    else:              # 2A + B <-> C, C -> D
        r1, r2 = R(3, 2) * A**2 * B - C, 2 * C
        fld, refs, desc = [-2 * r1, -r1, r1 - r2, r2], [A - 2 * B, B + C + D], "2A+B<->C, C->D mass action, rate constants 3/2,1,2."
    if perturbed: desc = desc.replace("A", names[0]).replace("B", names[1]).replace("C", names[2]).replace("D", names[3])
    return System("moiety_network" + ("_pert" if perturbed else ""), "moiety", [A, B, C, D], [], field=fld,
                  conserved_ref=refs, perturbed=perturbed, description=desc)


def acrobot(m1=1, m2=1, l1=1, lc1=R(1, 2), lc2=R(1, 2), I1=1, I2=1, g=R(49, 5), names=("c1", "s1", "c2", "s2", "w1", "w2"), perturbed=False):
    """Passive Acrobot (book model) in the 6-d observation, derived symbolically from the parameters.
    With default parameters the energy equals the reference repo's polynomial (checked in test_pipeline.py)."""
    c1, s1, c2, s2, w1, w2 = _syms(names)
    m1, m2, l1, I1, I2 = [sp.nsimplify(v) for v in (m1, m2, l1, I1, I2)]
    lc1, lc2, g = sp.nsimplify(lc1), sp.nsimplify(lc2), sp.nsimplify(g)
    d1 = m1 * lc1**2 + m2 * (l1**2 + lc2**2 + 2 * l1 * lc2 * c2) + I1 + I2
    d2 = m2 * (lc2**2 + l1 * lc2 * c2) + I2
    d22 = m2 * lc2**2 + I2
    h1 = -m2 * l1 * lc2 * s2 * (w2**2 + 2 * w1 * w2)
    h2 = m2 * l1 * lc2 * s2 * w1**2
    p1 = (m1 * lc1 + m2 * l1) * g * s1 + m2 * g * lc2 * (s1 * c2 + c1 * s2)
    p2 = m2 * g * lc2 * (s1 * c2 + c1 * s2)
    r1, r2 = -(h1 + p1), -(h2 + p2)
    det = sp.expand(d1 * d22 - d2**2)
    a1, a2 = sp.expand(d22 * r1 - d2 * r2), sp.expand(-d2 * r1 + d1 * r2)
    fld = [-s1 * w1 * det, c1 * w1 * det, -s2 * w2 * det, c2 * w2 * det, a1, a2]
    E = sp.expand(R(1, 2) * d1 * w1**2 + d2 * w1 * w2 + R(1, 2) * d22 * w2**2
                  - (m1 * lc1 + m2 * l1) * g * c1 - m2 * g * lc2 * (c1 * c2 - s1 * s2))
    return System("acrobot" + ("_pert" if perturbed else ""), "acrobot", [c1, s1, c2, s2, w1, w2],
                  [c1**2 + s1**2 - 1, c2**2 + s2**2 - 1], field=[sp.expand(e) for e in fld], field_den=det,
                  conserved_ref=[E], perturbed=perturbed,
                  description=f"Passive Acrobot (book model) in ({', '.join(names)}); m1={m1}, m2={m2}, l1={l1}, lc1={lc1}, lc2={lc2}, I1={I1}, I2={I2}, g={g}.")


REGISTRY = {
    "twisted_cubic": twisted_cubic, "rigid_body": free_rigid_body, "kepler": kepler, "moiety": moiety_network, "acrobot": acrobot,
    "twisted_cubic_pert": lambda: twisted_cubic(("u", "v", "w"), True),
    "rigid_body_pert": lambda: free_rigid_body((3, 5, 11), ("p", "q", "r"), True),
    "kepler_pert": lambda: kepler(R(3, 2), ("X", "Y", "Vx", "Vy", "Z"), True),
    "moiety_pert": lambda: moiety_network(1, ("P", "Q", "S", "T"), True),
    "acrobot_pert": lambda: acrobot(1, R(13, 10), R(6, 5), R(1, 2), R(3, 5), 1, R(3, 2), R(49, 5), ("k1", "n1", "k2", "n2", "a1", "a2"), True),
}
CANONICAL = ["twisted_cubic", "rigid_body", "kepler", "moiety", "acrobot"]
PERTURBED = [k for k in REGISTRY if k.endswith("_pert")]
DEFAULT = CANONICAL