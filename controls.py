"""Grader validation. Run BEFORE any model output is graded. Exits nonzero on failure."""
import sys, sympy as sp
import systems, grade, data

fails = []
def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond: fails.append(name)

def V(s): return {str(v): v for v in s.variables}

# ---- conserved-quantity systems --------------------------------------------
for key in [k for k, f in systems.REGISTRY.items() if f().conserved_ref]:
    s = systems.REGISTRY[key]()
    trajs = data.trajectories(s, n_traj=6, T=4.0)
    drift = data.make_drift_fn(s, trajs)
    refs = s.conserved_ref
    E = refs[0]
    kw = dict(system=s, drift_fn=drift)

    # positive controls: same invariant written differently
    check(f"{key}: reference is exactly_correct", grade.grade(str(E), **kw)["verdict"] == "exactly_correct")
    check(f"{key}: scaled+shifted reference is exactly_correct", grade.grade(f"7/3*({E}) - 5", **kw)["verdict"] == "exactly_correct")
    if s.constraints:
        g = s.constraints[0]
        check(f"{key}: reference + multiple of constraint is exactly_correct",
              grade.grade(f"({E}) + ({s.variables[0]}+2)*({g})", **kw)["verdict"] == "exactly_correct")
    if len(refs) > 1:
        check(f"{key}: sum of two references is exactly_correct", grade.grade(f"({refs[0]})+({refs[1]})", **kw)["verdict"] == "exactly_correct")
    check(f"{key}: E^2 is valid_non_reference", grade.grade(f"({E})**2", **kw)["verdict"] == "valid_non_reference")

    # negative controls: perturb one coefficient
    x0 = s.variables[-1]
    for eps in ["1/10**12", "1/10**6", "1/10**3"]:
        r = grade.grade(f"({E}) + ({eps})*{x0}**2", **kw)
        check(f"{key}: perturbation {eps} rejected", r["verdict"] in ("plausible_but_wrong", "implausible"))
    # null controls
    check(f"{key}: constant is trivial", grade.grade("42", **kw)["verdict"] == "correct_but_trivial")
    if s.constraints:
        check(f"{key}: known constraint is trivial", grade.grade(str(s.constraints[0]), **kw)["verdict"] == "correct_but_trivial")
    check(f"{key}: refusal", grade.grade("REFUSE", **kw)["verdict"] == "refused")
    check(f"{key}: transcendental rejected", grade.grade(f"sin({s.variables[0]})", **kw)["verdict"] == "unparseable")
    # float input: rationalisation
    r = grade.grade(str(sp.N(sp.expand(E), 15)), **kw)
    check(f"{key}: decimal reference is literal-exact or reconstructs explicitly",
          r["verdict"] == "exactly_correct" or
          (r["verdict"] == "exact_after_rounding" and r["rounded_verdict"] == "exactly_correct"))
    # Float tolerance policy: a LOSSY float (2.0004 for 2) must NOT be exactly_correct, whatever D is,
    # while the same error written as an exact rational is rejected outright (consistent, never accepted).
    P_ = sp.Poly(sp.expand(E), *s.variables)
    terms_ = [(m, float(c) * (1 + 2e-4 if i == 0 else 1)) for i, (m, c) in enumerate(P_.terms())]
    lossy_txt = " + ".join(f"({c!r})*" + "*".join(f"({v})**{e}" for v, e in zip(s.variables, m)) for m, c in terms_)
    rl = grade.grade(lossy_txt, **kw)
    check(f"{key}: lossy float (2e-4 off one coefficient) is never exactly_correct", rl["verdict"] != "exactly_correct")

# ---- vanishing tasks: twisted cubic (canonical and perturbed) --------------
for key in ["twisted_cubic", "twisted_cubic_pert"]:
    s = systems.REGISTRY[key](); vs = s.variables
    pts = data.curve_points(s)
    kw = dict(system=s, data_residual_fn=data.make_residual_fn(s, pts))
    true = [str(g) for g in s.constraints]
    gb = [str(e) for e in sp.groebner(s.constraints, *vs, order="grevlex", domain="QQ").exprs]
    check(f"{key}: true generators exactly_correct", grade.grade(true, **kw)["verdict"] == "exactly_correct")
    check(f"{key}: reduced GB ({len(gb)} elements) exactly_correct", grade.grade(gb, **kw)["verdict"] == "exactly_correct")
    check(f"{key}: one generator is sound_incomplete", grade.grade(true[:1], **kw)["verdict"] == "sound_incomplete")
    extra = [e for e in gb if e not in true]
    if extra:
        check(f"{key}: a non-generator GB element alone is sound_incomplete", grade.grade(extra[:1], **kw)["verdict"] == "sound_incomplete")
    x0 = vs[0]
    r = grade.grade([true[0], f"({true[1]}) + 1/10**6*{x0}"], **kw)
    check(f"{key}: 1e-6 wrong generator is plausible_but_wrong", r["verdict"] == "plausible_but_wrong")
    r = grade.grade([true[0], f"{vs[2]} - {x0}**2"], **kw)
    check(f"{key}: grossly wrong is implausible", r["verdict"] == "implausible")
    check(f"{key}: refusal", grade.grade("REFUSE", **kw)["verdict"] == "refused")
    # FIX: a true generating set plus one unparseable entry must not be exactly_correct.
    # This guards the grade_vanishing fix: previously, unparseable entries were silently
    # dropped and the parseable remainder could still return exactly_correct.
    r = grade.grade(true + ["not_a_polynomial("], **kw)
    check(f"{key}: unparseable entry prevents exactly_correct", r["verdict"] != "exactly_correct")

import prompts
for k, f in systems.REGISTRY.items():
    for cond in ("A", "B", "Bd"):
        s = f(); txt = prompts.build(s, cond, n_rows=3)
        if s.vanishing_task:
            leaks = [str(sp.expand(g)) for g in s.constraints] + [str(sp.expand(-g)) for g in s.constraints]
            check(f"{k}/{cond}: prompt does not leak constraints", "Known algebraic" not in txt and not any(l in txt for l in leaks))
            if cond in ("B", "Bd") and s.curve_text:
                check(f"{k}/{cond}: prompt does not leak the parametrisation",
                      s.curve_text not in txt and "for real t" not in txt)
        else:
            check(f"{k}/{cond}: prompt does not leak the reference invariants",
                  not any(str(sp.expand(r)) in txt for r in s.conserved_ref))
            if cond == "Bd":
                check(f"{k}/Bd: data-only prompt omits the system description (parameters/mechanism)",
                      s.description not in txt and "System:" not in txt)
            if cond == "B":
                check(f"{k}/B: sampled-data prompt does include the description (B is NOT data-only)", s.description in txt)

print(f"\n{len(fails)} failure(s)")
sys.exit(1 if fails else 0)