"""Prompt construction. A supplies equations; B supplies sampled states and a
system description (parameters and mechanism). The unevaluated Bd option omits
the description but retains declared variables and known constraints. For the
curve tasks B and Bd coincide. Canonical and perturbed systems are defined in
systems.REGISTRY; no tool execution is part of the reported experiment."""
import numpy as np
import sympy as sp
import data

FORMAT = """Answer with a single JSON object and nothing else:
{"polynomials": ["<poly 1>", "<poly 2>", ...], "refuse": false}
Rules: each entry is a polynomial in exactly the variables listed, using + - * ** and
exact rational numbers (write 3/4, not 0.75). No trig functions, roots, or division by
variables. If you cannot determine the answer, return {"polynomials": [], "refuse": true}."""


def _eqs(s):
    # FIX: write the common denominator explicitly. The old form put
    # "[divided by D = det]" on the same line as the numerator, which is ambiguous.
    lines = []
    D = sp.expand(s.field_den)
    for v, f in zip(s.variables, s.field):
        if D == 1:
            lines.append(f"d({v})/dt = {sp.expand(f)}")
        else:
            lines.append(f"d({v})/dt = ({sp.expand(f)}) / ({D})")
    return "\n".join(lines)


def _task_text(s, historical=False):
    if s.vanishing_task and not historical:
        return ("State a generating set for the ideal of all polynomial relations holding on the "
                "entire underlying parametrised curve, not the ideal of the finite displayed samples. "
                "When only sampled points are supplied, treat their decimal coordinates as rounded "
                "observations and infer a candidate curve; finite samples alone do not uniquely "
                "determine an unrestricted curve. If you cannot determine a candidate generating "
                "set, refuse. Return exact rational polynomials for the inferred curve.")
    if s.vanishing_task:
        v = ", ".join(map(str, s.variables))
        return (f"State a generating set for the ideal of ALL polynomial relations satisfied by the "
                f"points ({v}). Give polynomials p with p({v}) = 0 on the data; the set should "
                f"generate every such relation.")
    task = ("Find one non-constant polynomial Q in the variables that is exactly conserved along "
            "every solution (dQ/dt = 0 identically on the constraint set), and is not itself one of "
            "the listed constraints or a constant.")
    if not historical:
        task += " Return exactly one polynomial in the polynomials list."
    return task


def build(system, condition="A", n_rows=25, seed=0, noise=0.0, *, historical=False):
    """Build corrected prompts by default; historical=True reproduces archived wording."""
    s = system
    data_only = condition == "Bd" and not s.vanishing_task
    head = ("" if data_only else f"System: {s.description}\n") + f"Variables: {', '.join(map(str, s.variables))}\n"
    if s.constraints and not s.vanishing_task:   # never reveal the answer
        head += "Known algebraic constraints (always hold): " + "; ".join(f"{sp.expand(g)} = 0" for g in s.constraints) + "\n"
    if condition == "A":
        if s.field is None:
            body = f"The state is ({', '.join(map(str, s.variables))}) = {s.curve_text} for real t.\n"
        else:
            body = "Equations of motion:\n" + _eqs(s) + "\n"
    else:
        rng = np.random.default_rng(seed)
        if s.field is None:
            pts = data.curve_points(s, n=n_rows, seed=seed)
            pts = pts + rng.normal(0, noise, pts.shape) if noise else pts
            rows = pts
        else:
            traj = data.trajectories(s, n_traj=2, T=3.0, n_pts=n_rows, seed=seed)[0]
            rows = traj + rng.normal(0, noise, traj.shape) if noise else traj
        body = "Sampled states (one row per time sample, columns = variables in the order above):\n"
        if not historical:
            body = ("Sampled points" if s.field is None else "Sampled states (one row per time sample)")
            body += ", columns = variables in the order above. Decimal coordinates are rounded observations, not exact rational states:\n"
        body += "\n".join(", ".join(f"{x:.10g}" for x in r) for r in rows) + "\n"
    return head + body + "\n" + _task_text(s, historical=historical) + "\n\n" + FORMAT
