"""Tables with Wilson 95% intervals. Per (model, system, condition); never pooled across systems."""
import json, math, sys, collections

def wilson(k, n, z=1.96):
    if n == 0: return (float("nan"),) * 2
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0, c - h), min(1, c + h)

def main(path="results/graded_all.jsonl"):
    rows = [json.loads(l) for l in open(path)]
    cells = collections.defaultdict(list)
    for r in rows: cells[(r["model"], r["system"], r["condition"], r["noise"])].append(r)
    order = ["exactly_correct", "exact_after_rounding", "correct_but_trivial", "valid_non_reference", "sound_incomplete",
         "plausible_but_wrong", "implausible", "refused", "empty_output", "unparseable",
         "grader_unavailable"]
    # FIX: column widths widened so long model/system names do not collide.
    # FIX: `empty_output` is not a stated answer; excluded from the plausible-but-wrong denominator.
    print(f"{'model':<28}{'system':<24}{'cond':<5}{'n':>4}  " + "  ".join(f"{v[:9]:>9}" for v in order) + "   exact% [95% CI]   sens")
    for k in sorted(cells):
        rs = cells[k]; n = len(rs); c = collections.Counter(r["verdict"] for r in rs)
        lo, hi = wilson(c["exactly_correct"], n); sens = sum(r["rationalisation_sensitive"] for r in rs)
        print(f"{k[0]:<28}{k[1]:<24}{k[2]:<5}{n:>4}  " + "  ".join(f"{c[v]:>9}" for v in order)
              + f"   {100*c['exactly_correct']/n:5.1f} [{100*lo:.0f},{100*hi:.0f}]   {sens}")
    # Apply the three reporting policies to the same expressions, and distinguish
    # reference mismatch from permissive acceptance of algebraic/task failures.
    print("\nScoring policies on identical outputs (rates in %, gaps in percentage points):")
    print("  Reconstruction, trivial and incomplete columns are disjoint additional accepted counts.")
    for k in sorted(cells):
        rs = cells[k]; n = len(rs); c = collections.Counter(r["verdict"] for r in rs)
        strict = c["exactly_correct"]
        task_valid = strict + (0 if k[1].startswith("twisted_cubic") else c["valid_non_reference"])
        reconstructed = c["exact_after_rounding"]
        trivial = c["correct_but_trivial"]
        incomplete = c["sound_incomplete"]
        naive = task_valid + reconstructed + trivial + incomplete
        lo, hi = wilson(task_valid, n)
        print(f"  {k[0]:<28}{k[1]:<24}{k[2]:<4} strict {100*strict/n:5.1f}%"
              f"  task-valid {100*task_valid/n:5.1f}% [{100*lo:.0f},{100*hi:.0f}]"
              f"  naive {100*naive/n:5.1f}%"
              f"  valid-strict {100*(task_valid-strict)/n:4.1f} pts"
              f"  naive-valid {100*(naive-task_valid)/n:4.1f} pts"
              f"  reconstructed {reconstructed}/{n}  trivial {trivial}/{n}  incomplete {incomplete}/{n}")
    # First-polynomial rule audit: rows where several polynomials were returned for a conserved task.
    print("\nFirst-polynomial rule audit (conserved tasks, answers with >1 polynomial):")
    mp = collections.defaultdict(lambda: collections.Counter())
    for r in rows:
        va = r.get("verdicts_all")
        if not va: continue
        key = r["model"]; mp[key]["multi"] += 1
        good = ("exactly_correct", "valid_non_reference")
        if r["verdict"] not in good and any(v in good for v in va): mp[key]["first_wrong_other_right"] += 1
        if r["verdict"] in good and any(v not in good for v in va): mp[key]["first_right_other_wrong"] += 1
        if all(v in good for v in va): mp[key]["all_valid"] += 1
    for m, c in sorted(mp.items()): print(f"  {m:<28} {dict(c)}")
    # Plausible-but-wrong is defined by a data-fit threshold; show how the count moves with it.
    print("\nPlausible-but-wrong count vs data-fit threshold (stated, non-conserved answers with a recorded fit value):")
    for m in sorted({r["model"] for r in rows}):
        vals = []
        for r in rows:
            if r["model"] != m or r["detail"].get("literal_verdict", r["verdict"]) not in ("plausible_but_wrong", "implausible"): continue
            d = r["detail"]; v = d.get("relative_drift", d.get("max_data_residual"))
            if v is not None: vals.append(v)
        if vals: print(f"  {m:<28} n={len(vals):>3}  <1e-3: {sum(v<1e-3 for v in vals):>3}  <1e-2: {sum(v<1e-2 for v in vals):>3}  <5e-2: {sum(v<5e-2 for v in vals):>3}  min={min(vals):.2e}")
    # headline: among answers that were stated (not refused/unparseable/empty/unavailable),
    # fraction plausible-but-wrong
    print("\nPlausible-but-wrong among stated answers (per model, per system):")
    for k in sorted(cells):
        rs = cells[k]
        stated = [r for r in rs if r["verdict"] not in ("refused", "unparseable", "grader_unavailable", "empty_output")]
        if not stated: continue
        pw = sum(r["verdict"] == "plausible_but_wrong" for r in stated); lo, hi = wilson(pw, len(stated))
        print(f"  {k[0]:<28}{k[1]:<24}{k[2]:<4} {pw}/{len(stated)} = {100*pw/len(stated):.0f}% [{100*lo:.0f},{100*hi:.0f}]")
    # which terms are wrong
    tc = collections.Counter()
    for r in rows:
        for t in r["detail"].get("remainder_terms", []): tc[(r["system"], t)] += 1
    if tc:
        print("\nMost frequent monomials in nonzero Lie-derivative remainders:")
        for (s, t), n in tc.most_common(8): print(f"  {s:<24}{t:<28}{n}")

if __name__ == "__main__":
    main(*sys.argv[1:])