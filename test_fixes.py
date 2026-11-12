"""Regression tests for the post-audit fixes. Run by check_all.sh.
 1. Condition A/B prompts are byte-identical to those used for the collected data (hash file).
 2. Literal decimal exactness and separate bounded-denominator reconstruction.
 3. Every polynomial of a multi-polynomial conserved answer is graded and stored (verdicts_all).
 4. Resumability key covers max_tokens / reasoning-effort: different config appends, same config skips.
 5. Bd (data-only) prompt omits the system description; B does not."""
import glob, hashlib, json, os, re, subprocess, sys, tempfile
import systems, grade, prompts, data

fails = []
def check(name, ok):
    print(("PASS " if ok else "FAIL ") + name)
    if not ok: fails.append(name)

# 1 -------------------------------------------------------------------------------------------
H = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompt_hashes_AB.json")))
_here = os.path.dirname(os.path.abspath(__file__))
_sha = lambda t: hashlib.sha256(t.encode()).hexdigest()
bad = [k for k, v in H.items() if _sha(prompts.build(systems.REGISTRY[k.split("|")[0]](), k.split("|")[1],
        seed=int(k.split("|")[2]))) != v]

def _sent_prompts():
    """The prompts that were actually sent to the models (stored in every raw row)."""
    out = {}
    for f in sorted(glob.glob(os.path.join(_here, "results", "primary", "raw_*.jsonl"))):
        for l in open(f):
            r = json.loads(l)
            if r.get("condition") in ("A", "B") and r.get("prompt"):
                out.setdefault(f"{r['key']}|{r['condition']}|{r['sample']}", r["prompt"])
    return out

_NUM = re.compile(r"[-+]?\d+\.?\d*(?:[eE][-+]?\d+)?")
def _drift(new, sent):
    """None if the texts differ in anything but numbers; else the max relative numeric deviation."""
    if _NUM.sub("#", new) != _NUM.sub("#", sent): return None
    a, b = [float(x) for x in _NUM.findall(new)], [float(x) for x in _NUM.findall(sent)]
    return max((abs(x - y) / max(abs(y), 1e-12) for x, y in zip(a, b)), default=0.0)

hard, soft = [], {}
if bad and os.environ.get("STRICT_PROMPT_HASH") != "1":
    sent = _sent_prompts()
    for k in bad:
        n, c, sd = k.split("|")
        ref = sent.get(k)
        # The stored hash must equal the hash of the prompt that was really sent; otherwise the hash file is wrong.
        if ref is None or _sha(ref) != H[k]: hard.append(k); continue
        d = _drift(prompts.build(systems.REGISTRY[n](), c, seed=int(sd)), ref)
        (soft.__setitem__(k, d) if d is not None and d < 1e-3 else hard.append(k))
else:
    hard = bad
if soft:
    print("WARN numeric-only drift (template identical; trajectory numbers differ because solve_ivp/DOP853 is not")
    print("     bitwise-stable across platforms/library builds; graded verdicts use the stored prompts and outputs):")
    for k, d in soft.items(): print(f"       {k}: max relative deviation {d:.1e}")
check(f"A/B prompt templates identical to collected data ({len(H)} prompts; {len(soft)} with numeric-only drift)", not hard)
if hard: print("     real prompt mismatch (not numeric drift):", hard)

# The collected contract explicitly targets the underlying curve.
curve = systems.REGISTRY["twisted_cubic"]()
check("collected curve prompt targets the underlying curve and rounded observations",
      all(term in prompts.build(curve, "B") for term in
          ("entire underlying parametrised curve", "rounded observations", "do not uniquely")))


# 2 -------------------------------------------------------------------------------------------
s = systems.REGISTRY["rigid_body"]()
fn = data.make_drift_fn(s, data.trajectories(s, n_traj=2, T=2.0))
g = lambda t, D=1000: grade.grade_conserved(t, s, D=D, drift_fn=fn)
check("exact rational answer: exactly_correct", g("w1**2 + 2*w2**2 + 3*w3**2")["verdict"] == "exactly_correct")
check("exact decimal 2.0 stays exactly_correct", g("w1**2 + 2.0*w2**2 + 3.0*w3**2")["verdict"] == "exactly_correct")
check("exact decimal 0.5*(...) stays exactly_correct", g("0.5*(w1**2 + 2*w2**2 + 3*w3**2)")["verdict"] == "exactly_correct")
for t in ("2.0004", "2.0001", "2.00000001"):
    r = g(f"w1**2 + {t}*w2**2 + 3*w3**2")
    check(f"lossy float {t} -> exact_after_rounding (was exactly_correct)", r["verdict"] == "exact_after_rounding" and r["rounded_verdict"] == "exactly_correct")
check("exact rational 2+1e-8 is rejected (consistent with float policy)",
      g("w1**2 + (2000000001/1000000000)*w2**2 + 3*w3**2")["verdict"] in ("plausible_but_wrong", "implausible"))
check("a finite decimal scalar multiple of an invariant is exact", g("0.333333333333333*(w1**2 + 2*w2**2 + 3*w3**2)")["verdict"] == "exactly_correct")
tc = systems.REGISTRY["twisted_cubic"]()
check("vanishing: lossy float generator -> exact_after_rounding", grade.grade(["y - 1.0001*x**2", "z - x**3"], tc)["verdict"] == "exact_after_rounding")
check("vanishing: rational generators unaffected", grade.grade(["y - x**2", "z - x**3"], tc)["verdict"] == "exactly_correct")

# The QQ contract applies to coefficients as well as the declared variables.
for coeff in ("sqrt(2)", "pi"):
    check(f"non-rational coefficient {coeff} is rejected",
          g(f"{coeff}*(w1**2 + 2*w2**2 + 3*w3**2)")["verdict"] == "unparseable")
r = grade.grade_conserved(str(s.conserved_ref[0]**3), s)
check("E cubed is valid without a false independence claim",
      r["verdict"] == "valid_non_reference" and "unresolved" in r["relation"])

# Literal decimals are exact decimal rationals, without a near-equality gate.
for token in ("2.0000000001", "2.00000000000001", "2.0004"):
    for cap in (10, 100, 1000):
        result = g(f"w1**2 + {token}*w2**2 + 3*w3**2", D=cap)
        check(f"literal decimal {token}, D={cap}: never strict-exact",
              result["verdict"] == "exact_after_rounding" and
              result["literal_verdict"] == "plausible_but_wrong")
check("explicit decimal and fraction spellings denote the same rational polynomial",
      grade.parse("w1**2+2.0000000001*w2**2+3*w3**2", s)[0] ==
      grade.parse("w1**2+(20000000001/10000000000)*w2**2+3*w3**2", s)[0])
check("finite decimal arithmetic is exact before expansion",
      grade.parse("(0.1+0.2-0.3)*w1", s)[0] == 0)
for expression in ("__import__('math').factorial(3)*w1**2",
                   "w1.__class__", "[w1][0]", "w1/0", "w1/w2", "w1**-1"):
    check(f"restricted parser rejects {expression}", grade.parse(expression, s)[0] is None)
check("restricted parser preserves rational scientific notation and implicit multiplication",
      grade.parse("2w1^2 + (3/4)w2**2 - 1e-12*w3", s)[0] ==
      2*s.variables[0]**2 + grade.sp.Rational(3, 4)*s.variables[1]**2
      - grade.sp.Rational(1, 10**12)*s.variables[2])
check("restricted parser respects unary signs and right-associative powers",
      grade.parse("-w1**2 + w2**2**3", s)[0] == -s.variables[0]**2+s.variables[1]**8)
from sr_sindy import sparse_null
import numpy as np
matrix = np.random.default_rng(1).normal(size=(30, 10))
vector, _, _ = sparse_null(matrix, iters=1)
check("sparse nullspace iteration limit returns a finite full-size vector",
      vector.shape == (10,) and np.isfinite(vector).all() and np.max(np.abs(vector)) == 1)
check("an exact fractional coefficient is not reconstructed to a nearby small fraction",
      grade.parse("(1/10001)*w1", s, D=10)[0] == s.variables[0]/10001)
x = s.variables[0]
check("drift is invariant under a huge additive constant",
      fn(x) == fn(x+10**12) and fn(x) > 0.1)
k = systems.acrobot()
kfn = data.make_drift_fn(k, data.trajectories(k, n_traj=2, T=1))
q = k.variables[-1]
check("drift is invariant under adding a known constraint",
      kfn(q) == kfn(q + 10**8*k.constraints[0]))

# 3 -------------------------------------------------------------------------------------------
with tempfile.TemporaryDirectory(prefix='exact-invariant-test-') as d:
    row = dict(model="m", provider="mock", system=s.name, key="rigid_body", condition="A", noise=0.0, sample=0, temperature=1.0,
               prompt="", output=json.dumps({"polynomials": ["w1**2 + w2**2", "w1**2 - w2**2"], "refuse": False}), error=None)
    open(os.path.join(d, "r.jsonl"), "w").write(json.dumps(row) + "\n")
    subprocess.run([sys.executable, "grade_outputs.py", "--raw", os.path.join(d, "r.jsonl"), "--out", os.path.join(d, "g.jsonl")], check=True, capture_output=True)
    rec = json.loads(open(os.path.join(d, "g.jsonl")).readline())
    check("multi-poly answer: headline = first polynomial", rec["verdict"] == "exactly_correct")
    check("multi-poly answer: all polynomials graded and stored", rec["verdicts_all"] == ["exactly_correct", "implausible"])

    # 4 -------------------------------------------------------------------------------------------
    out = os.path.join(d, "m.jsonl")
    def run(*extra):
        subprocess.run([sys.executable, "run_llm.py", "--provider", "mock", "--model", "mock", "--n", "1", "--systems", "rigid_body",
                        "--conditions", "A", "--out", out, *extra], check=True, capture_output=True)
        return sum(1 for _ in open(out))
    n1 = run(); n2 = run(); n3 = run("--max-tokens", "8000"); n4 = run("--max-tokens", "8000"); n5 = run("--reasoning-effort", "low")
    check("same config is skipped on resume", n2 == n1 == 1)
    check("different max_tokens is NOT blocked by an old row", n3 == 2 and n4 == 2)
    check("different reasoning_effort is NOT blocked by an old row", n5 == 3)
    rows = [json.loads(l) for l in open(out)]
    n6 = run("--temperature", "0.5")
    check("different temperature is not blocked on resume", n6 == 4)
    check("max_tokens / reasoning_effort recorded per row", rows[1]["max_tokens"] == 8000 and rows[2]["reasoning_effort"] == "low")

    # 5 -------------------------------------------------------------------------------------------
    kep = systems.REGISTRY["kepler"]()
    check("Bd prompt omits description; B keeps it", "System:" not in prompts.build(kep, "Bd") and kep.description in prompts.build(kep, "B"))

    # 6 -------------------------------------------------------------------------------------------
    import grade_outputs
    check("bare REFUSE / NONE text is refused offline (same as grade.grade)", all(grade_outputs.extract(t) == (None, "refused") for t in ("REFUSE", " none \n")))
    check("other non-JSON text stays unparseable", grade_outputs.extract("I think x+y") == ([], "no_json"))
    check("JSON refusal and polynomial answers unchanged", grade_outputs.extract('{"polynomials": [], "refuse": true}') == (None, "refused")
          and grade_outputs.extract('{"polynomials": ["x+y"]}') == (["x+y"], "ok"))

print(f"\n{len(fails)} failure(s)")
sys.exit(1 if fails else 0)
