"""Independent correctness tests (beyond controls.py). Exit code nonzero on any failure.
  1. field tangent to every constraint ideal (otherwise the Lie test is meaningless)
  2. symbolic Lie derivative agrees with NUMERIC differentiation along integrated trajectories
  3. fuzz: random rational combinations of references are accepted; random polynomials rejected
  4. text normalisation / extraction robustness
  5. end-to-end with the mock model
"""
import sys, json, os, random, subprocess, tempfile
import numpy as np, sympy as sp
import systems, grade, data, grade_outputs

# test_pipeline.py needs bugs.py and envs.py from $ALGEBRAIC_RL_REPO for the
# independent-derivation checks in section 2b. Give a clear error rather than a bare
# ModuleNotFoundError at import time, since the rest of the suite is independently runnable.
try:
    import bugs, envs
except ImportError as e:
    sys.stderr.write(
        f"FAIL: {e}\n"
        "test_pipeline.py needs bugs.py and envs.py from $ALGEBRAIC_RL_REPO\n"
        "(they provide bugs.acrobot_reference() and envs.acrobot_dsdt() for the\n"
        "independent-derivation checks). Clone\n"
        "    https://github.com/TesfayZ/algebraicRLtest\n"
        "and re-export ALGEBRAIC_RL_REPO so that $ALGEBRAIC_RL_REPO/bugs.py and\n"
        "$ALGEBRAIC_RL_REPO/envs.py exist; see README §1.\n"
    )
    sys.exit(1)

fails = []
def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond: fails.append(name)

# ... rest of file unchanged ...

random.seed(1); rng = np.random.default_rng(1)

# 1. tangency: Lie derivative of each constraint lies in the ideal
for key, f in systems.REGISTRY.items():
    s = f()
    if s.field is None or not s.constraints: continue
    for g in s.constraints:
        L = grade.lie_derivative_numerator(g, s)
        check(f"{key}: field tangent to constraint {sp.expand(g)}", grade.nf(L, s.constraints, s.variables) == 0)

# 2. numeric cross-check of the symbolic Lie derivative (independent of Groebner/normal forms)
for key in [k for k, f in systems.REGISTRY.items() if f().conserved_ref]:
    s = systems.REGISTRY[key](); vs = s.variables
    fld = sp.lambdify(vs, [e / s.field_den for e in s.field], "numpy")
    trajs = data.trajectories(s, n_traj=3, T=2.0, n_pts=50, seed=3)
    worst = 0.0
    for Q in s.conserved_ref:
        grad = sp.lambdify(vs, [sp.diff(Q, v) for v in vs], "numpy")
        for tr in trajs:
            for y in tr[::7]:
                worst = max(worst, abs(float(np.dot(grad(*y), fld(*y)))))
    check(f"{key}: numeric dQ/dt of references ~ 0 on integrated states (max {worst:.1e})", worst < 1e-8)
    # a non-conserved polynomial must have clearly nonzero numeric derivative AND be rejected symbolically
    Qbad = vs[0] ** 2 + vs[-1]
    gb = sp.lambdify(vs, [sp.diff(Qbad, v) for v in vs], "numpy")
    num = max(abs(float(np.dot(gb(*y), fld(*y)))) for tr in trajs for y in tr[::7])
    check(f"{key}: non-conserved example has nonzero numeric derivative ({num:.2e}) and is rejected",
          num > 1e-6 and grade.is_conserved(Qbad, s)[0] is False)

# 2b. independent derivation check: parametrised Acrobot with default parameters == reference repo
import bugs, envs
G, _, _ = bugs.acrobot_reference(); a0 = systems.acrobot()
sub = dict(zip(bugs.ACRO_SYMS, a0.variables))
check("acrobot: derived energy equals the reference repo's polynomial", sp.expand(a0.conserved_ref[0] - G[2].subs(sub, simultaneous=True)) == 0)
fl = sp.lambdify(a0.variables, [e / a0.field_den for e in a0.field], "numpy"); worst = 0.0
for _ in range(100):
    th1, th2 = rng.uniform(-3, 3, 2); w1, w2 = rng.uniform(-3, 3, 2)
    mine = np.array(fl(np.cos(th1), np.sin(th1), np.cos(th2), np.sin(th2), w1, w2))
    d = envs.acrobot_dsdt(np.array([th1, th2, w1, w2, 0.0]))
    ref = np.array([-np.sin(th1) * d[0], np.cos(th1) * d[0], -np.sin(th2) * d[1], np.cos(th2) * d[1], d[2], d[3]])
    worst = max(worst, np.max(np.abs(mine - ref)))
check(f"acrobot: vector field equals the reference repo dynamics (max diff {worst:.1e})", worst < 1e-9)
# perturbed variants really differ from canonical (not the same system renamed)
for k in systems.PERTURBED:
    base = k[:-5]; sp_, sb = systems.REGISTRY[k](), systems.REGISTRY[base]()
    check(f"{k}: differs from canonical in names and in the system itself",
          [str(v) for v in sp_.variables] != [str(v) for v in sb.variables] and sp_.description != sb.description)

# 3. fuzz
for key in [k for k, f in systems.REGISTRY.items() if f().conserved_ref]:
    s = systems.REGISTRY[key](); vs = s.variables; refs = s.conserved_ref
    ok_acc = ok_rej = 0; N = 12
    for _ in range(N):
        cs = [sp.Rational(random.randint(-9, 9), random.randint(1, 9)) for _ in refs] + [sp.Rational(random.randint(-5, 5))]
        if all(c == 0 for c in cs[:-1]): cs[0] = 1
        Q = sum(c * r for c, r in zip(cs, refs)) + cs[-1]
        v = grade.grade(str(Q), s)["verdict"]
        ok_acc += v == "exactly_correct"
        # random sparse polynomial: must not be accepted as correct (conserved) unless it truly is
        mons = [sp.Mul(*[x ** random.randint(0, 2) for x in random.sample(vs, 2)]) for _ in range(3)]
        R = sum(sp.Rational(random.randint(1, 9)) * m for m in mons)
        cons, _ = grade.is_conserved(R, s)
        v2 = grade.grade(str(R), s)["verdict"]
        ok_rej += (not cons and v2 in ("implausible", "plausible_but_wrong")) or (cons and v2 in ("exactly_correct", "valid_non_reference", "correct_but_trivial"))
    check(f"{key}: fuzz accepts {ok_acc}/{N} random rational combinations of references", ok_acc == N)
    check(f"{key}: fuzz verdicts consistent with the exact conservation test ({ok_rej}/{N})", ok_rej == N)

# 4. parsing / extraction robustness
s = systems.free_rigid_body()
for txt, exp in [("w1^2 + 2*w2^2 + 3*w3^2", "exactly_correct"), ("w1\u00b2 + 2 w2\u00b2 + 3 w3\u00b2", "exactly_correct"),
                 ("w1**2+2*w2**2+3*w3**2 = 0", "exactly_correct"), ("$w_1^2$", "unparseable"),
                 ("0.5*w1**2 + 1.0*w2**2 + 1.5*w3**2", "exactly_correct"), ("sqrt(w1)", "unparseable"),
                 ("w1**2/w2", "unparseable"), ("", "refused")]:
    got = grade.grade(txt, s)["verdict"]
    check(f"parse {txt!r} -> {exp}", got == exp)
check("extract: fenced json with stray braces",
      grade_outputs.extract('I think {x} then:\n```json\n{"polynomials":["w1**2"],"refuse":false}\n```')[0] == ["w1**2"])
check("extract: refusal", grade_outputs.extract('{"polynomials": [], "refuse": true}')[0] is None)

# FIX: null / whitespace-only completions must not be classified as refusals.
# These are the notes the grader relies on to route to empty_output, not refused.
check("extract: None output -> no_output", grade_outputs.extract(None) == (None, "no_output"))
check("extract: empty string -> empty_output", grade_outputs.extract("") == (None, "empty_output"))
check("extract: whitespace-only -> empty_output", grade_outputs.extract("   \n\t ") == (None, "empty_output"))

# FIX: in a vanishing task, a true generating set plus one unparseable entry must not be exactly_correct.
s_tc = systems.REGISTRY["twisted_cubic"]()
kw_tc = dict(data_residual_fn=data.make_residual_fn(s_tc, data.curve_points(s_tc)))
true_tc = [str(g) for g in s_tc.constraints]
r = grade.grade(true_tc + ["not_a_poly("], s_tc, **kw_tc)
check("vanishing: unparseable entry prevents exactly_correct", r["verdict"] != "exactly_correct")
r = grade.grade(true_tc, s_tc, **kw_tc)
check("vanishing: clean true generating set is exactly_correct", r["verdict"] == "exactly_correct")

# 5. end-to-end with mock model
with tempfile.TemporaryDirectory(prefix='exact-invariant-test-') as d:
    raw, gr = os.path.join(d, "raw.jsonl"), os.path.join(d, "g.jsonl")
    r1 = subprocess.run([sys.executable, "run_llm.py", "--provider", "mock", "--model", "mock", "--n", "2", "--out", raw], capture_output=True, text=True)
    r2 = subprocess.run([sys.executable, "grade_outputs.py", "--raw", raw, "--out", gr], capture_output=True, text=True)
    r3 = subprocess.run([sys.executable, "analysis.py", gr], capture_output=True, text=True)
    n = sum(1 for _ in open(gr)) if os.path.exists(gr) else 0
    check(f"end-to-end mock pipeline runs and grades {n} rows", r1.returncode == r2.returncode == r3.returncode == 0 and n == 2 * 2 * len(systems.DEFAULT))
    # resumability: rerunning must not duplicate rows
    subprocess.run([sys.executable, "run_llm.py", "--provider", "mock", "--model", "mock", "--n", "2", "--out", raw], capture_output=True)
    r4 = subprocess.run([sys.executable, "run_llm.py", "--provider", "mock", "--model", "mock", "--n", "1", "--systems", "perturbed", "--conditions", "A", "--out", os.path.join(d, "p.jsonl")], capture_output=True, text=True)
    check("run_llm accepts --systems perturbed", r4.returncode == 0 and sum(1 for _ in open(os.path.join(d, "p.jsonl"))) == len(systems.PERTURBED))
    check("run_llm is resumable (no duplicate rows)", sum(1 for _ in open(raw)) == n)

print(f"\n{len(fails)} failure(s)")
sys.exit(1 if fails else 0)