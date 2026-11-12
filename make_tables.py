"""Generate the paper's result tables from results/graded_all.jsonl and numerical baseline files so
that table numbers can never drift from the data.  Writes docs/tables_generated.tex  (\\input by the paper).
Usage: python make_tables.py"""
import json, math, collections, glob, os

S = "implicit-sindy(pysindy-lib)"; GROQ = "openai/gpt-oss-120b"; CER = "gpt-oss-120b"; GEM = "gemini-3.5-flash-lite"
rows = [json.loads(l) for l in open("results/graded_all.jsonl")]

def wilson(k, n, z=1.96):
    if n == 0: return (float("nan"),) * 2
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0, c - h), min(1, c + h)

def cell(m, s, c): return [r for r in rows if r["model"] == m and r["system"] == s and r["condition"] == c]
def pct(k, n): return 100 * k / n if n else float("nan")
def fmt(x): return f"{x:.0f}" if abs(x - round(x)) < 0.05 else f"{x:.1f}"

ORDER = [("acrobot", "Acrobot"), ("acrobot_pert", "Acrobot\\_p"), ("kepler", "Kepler"), ("kepler_pert", "Kepler\\_p"),
         ("moiety_network", "Moiety"), ("moiety_network_pert", "Moiety\\_p"), ("rigid_body_1_2_3", "Rigid"),
         ("rigid_body_3_5_11_pert", "Rigid\\_p"), ("twisted_cubic", "Twisted"), ("twisted_cubic_pert", "Twisted\\_p")]
out = []

# Deployment inventory: one readable panel per deployment, with both rates and intervals.
for macro, model in (("tabLLMGroq", GROQ), ("tabLLMCerebras", CER), ("tabLLMGemini", GEM)):
    out.append("\\newcommand{\\" + macro + "}{%")
    out.append(r"\begin{tabular}{@{}l l c r l l@{}}\toprule")
    out.append(r"System & Variant & Input & $n$ & Reference recovery [95\% CI] & Task-valid [95\% CI] \\\midrule")
    for key, lab in ORDER:
        for cond in "AB":
            rs = cell(model, key, cond)
            n = len(rs); c = collections.Counter(r["verdict"] for r in rs)
            ex = c["exactly_correct"]; tv = ex + c["valid_non_reference"]
            lo, hi = wilson(ex, n); vlo, vhi = wilson(tv, n)
            variant = "Twin" if key.endswith("_pert") else "Canonical"
            label = lab.replace(r"\_p", "")
            out.append(f"{label} & {variant} & {cond} & {n} & {fmt(pct(ex,n))} [{100*lo:.0f},{100*hi:.0f}] & {fmt(pct(tv,n))} [{100*vlo:.0f},{100*vhi:.0f}]" + r" \\")
    out.append(r"\bottomrule\end{tabular}}")

# Diagnostic baseline summary; full exclusive verdicts remain in the cached records.
out.append(r"\newcommand{\tabSINDy}{%")
out.append(r"\begin{tabular}{@{}l l r r r r r r@{}}\toprule")
out.append(r"System & Variant & $n$ & Reference & Task-valid & Reconstructed & Incomplete & Trivial \\\midrule")
SY = [("twisted_cubic", "Twisted cubic"), ("rigid_body_1_2_3", "Rigid body"), ("kepler", "Kepler"), ("moiety_network", "Moiety"), ("acrobot", "Acrobot")]
SYP = [("twisted_cubic_pert", "Twisted cubic"), ("rigid_body_3_5_11_pert", "Rigid body"), ("kepler_pert", "Kepler"), ("moiety_network_pert", "Moiety"), ("acrobot_pert", "Acrobot")]
for title, lst in (("Canonical", SY), ("Twin", SYP)):
    for key, lab in lst:
        rs = cell(S, key, "B"); n = len(rs); c = collections.Counter(r["verdict"] for r in rs)
        ex = c['exactly_correct']; tv = ex + c['valid_non_reference']
        out.append(f"{lab} & {title} & {n} & {ex} & {tv} & {c['exact_after_rounding']} & {c['sound_incomplete']} & {c['correct_but_trivial']}" + r" \\")
    if title == "Canonical": out.append(r"\midrule")
out.append(r"\bottomrule\end{tabular}}")

# ---- Table: taxonomy effect: naive membership-only vs task-valid ------------------------------
infl = []
for m, tag in ((GROQ, "Groq"), (CER, "Cerebras"), (GEM, "Gemini"), (S, "SINDy")):
    for key, lab in ORDER:
        for cond in ("A", "B"):
            rs = cell(m, key, cond)
            if not rs: continue
            n = len(rs); c = collections.Counter(r["verdict"] for r in rs)
            tv = c["exactly_correct"] + (c["valid_non_reference"] if not key.startswith("twisted") else 0)
            naive = tv + c["exact_after_rounding"] + c["correct_but_trivial"] + c["sound_incomplete"] + (c["valid_non_reference"] if key.startswith("twisted") else 0)
            infl.append((100 * (naive - tv) / n, tag, lab, cond, pct(naive, n), pct(tv, n), c["exact_after_rounding"], c["correct_but_trivial"], c["sound_incomplete"], n))
infl.sort(reverse=True)
out.append(r"\newcommand{\tabInfl}{%")
out.append(r"\begin{tabular}{@{}l l r r r r r r@{}}\toprule Tier & Cell & Permissive & Valid & Gap & Recon. & Trivial & Incompl. \\\midrule")
for d, tag, lab, cond, nv, tv, recon, trivial, incomplete, n in infl[:10]:
    out.append(f"{tag} & {lab}, {cond} & {nv:.1f}\\% & {tv:.1f}\\% & {d:.1f} pts & {recon}/{n} & {trivial}/{n} & {incomplete}/{n}" + r" \\")
out.append(r"\bottomrule\end{tabular}}")
llm_infl = [x for x in infl if x[1] != "SINDy"]
sin_infl = [x for x in infl if x[1] == "SINDy"]
out.append(r"\newcommand{\inflLLMmax}{%.0f}" % max(x[0] for x in llm_infl))
out.append(r"\newcommand{\inflSINDymax}{%.0f}" % max(x[0] for x in sin_infl))
out.append(r"\newcommand{\inflSINDymin}{%.0f}" % min(x[0] for x in sin_infl))
out.append(r"\newcommand{\inflLLMnonzero}{%d}" % sum(1 for x in llm_infl if x[0] > 0))
out.append(r"\newcommand{\inflLLMcells}{%d}" % len(llm_infl))
# Literal-policy sensitivity and current reference timing rows.
summary = json.load(open("results/sensitivity.json"))
out += [r"\newcommand{\tabSensitivity}{%",
        r"\begin{tabular}{@{}l l r r r r@{}}\toprule",
        r"Outputs & Diagnostic & $n$ & $<10^{-3}$ & $<10^{-2}$ & $<5\times10^{-2}$ \\\midrule"]
for name in ("LLMs", "SINDy"):
    for metric in ("legacy", "centred"):
        g = summary[name]
        counts = [g[metric][str(t)] for t in (1e-3, 1e-2, 5e-2)]
        out.append(f"{name} & {'uncentred' if metric == 'legacy' else metric} & {g['literal_failures_with_fit']} & " + " & ".join(map(str, counts)) + r" \\")
out.append(r"\bottomrule\end{tabular}}")
out.append(r"\newcommand{\tabRuntimeRows}{%")
for line in open("results/grade_times.txt"):
    key, value, _ = line.split()
    family = key.removesuffix("_pert")
    label = {"twisted_cubic": "Twisted cubic", "rigid_body": "Rigid body",
             "kepler": "Kepler", "moiety": "Moiety", "acrobot": "Acrobot"}[family]
    label += " (twin)" if key.endswith("_pert") else " (canonical)"
    out.append(label + " & " + value + r" \\")
out.append("}")
audit = json.load(open('results/policy_audit.json'))
labels = {GROQ:'Groq', CER:'Cerebras', GEM:'Gemini', S:'Sparse float'}
out += [r'\newcommand{\tabPolicySummary}{%', r'\begin{tabular}{@{}lrrrrrr@{}}\toprule',
        r'Deployment & $n$ & Reference & Task-valid & Additional valid & Gap cells & Max. gap \\\midrule']
for model in (GROQ,CER,GEM):
    cs = [c for c in audit['cells'] if c['model']==model]
    reference = sum(c['reference'] for c in cs)
    valid = sum(c['task_valid'] for c in cs)
    out.append(f"{labels[model]} & {sum(c['n'] for c in cs)} & {reference} & {valid} & {valid-reference} & {sum(c['gap_points']>0 for c in cs)}/20 & {max(c['gap_points'] for c in cs):.1f}"+r' \\')
out.append(r'\bottomrule\end{tabular}}')
out += [r'\newcommand{\tabNonReference}{%', r'\begin{tabular}{@{}lrrr@{}}\toprule',
        r'Deployment & All outputs & Non-reference valid & Archive prevalence \\ \midrule']
for model in (GROQ, CER, GEM):
    rs = [r for r in rows if r['model'] == model]
    count = sum(r['verdict'] == 'valid_non_reference' for r in rs)
    out.append(f"{labels[model]} & {len(rs)} & {count} & {100*count/len(rs):.2f}\\%" + r' \\')
out.append(r'\bottomrule\end{tabular}}')
out += [r'\newcommand{\tabExternalCriterion}{%',r'\begin{tabular}{@{}lrrrrr@{}}\toprule',
        r'Source & Parseable $n$ & Criterion & Task-valid & Criterion only & Task only \\\midrule']
for model in (GROQ,CER,GEM,S):
    c = next(c for c in audit['comparisons'] if c['model']==model)
    out.append(f"{labels[model]} & {c['n']} & {c['criterion']} & {c['task_valid']} & {c['criterion_only']} & {c['task_only']}"+r' \\')
out.append(r'\bottomrule\end{tabular}}')
out += [r'\newcommand{\tabRationalBaseline}{%',r'\begin{tabular}{@{}llrrrr@{}}\toprule',
        r'System & Variant & $n$ & First valid & Any valid & All valid \\\midrule']
rational = [json.loads(l) for l in open('results/graded_rational15.jsonl')]
for key, lab in ORDER:
    rs = [r for r in rational if r['system']==key]
    good = {'exactly_correct','valid_non_reference'}
    first = sum(r['verdict'] in good for r in rs)
    any_valid = sum(any(v in good for v in (r.get('verdicts_all') or [r['verdict']])) for r in rs)
    all_valid = sum(all(v in good for v in (r.get('verdicts_all') or [r['verdict']])) for r in rs)
    out.append(f"{lab.replace(chr(92)+'_p','')} & {'Twin' if key.endswith('_pert') else 'Canonical'} & {len(rs)} & {first} & {any_valid} & {all_valid}"+r' \\')
out.append(r'\bottomrule\end{tabular}}')
diagnostics = json.load(open('results/grader_diagnostics.json'))
out += [r'\newcommand{\tabSubmissionRuntime}{%',
        r'\begin{tabular}{@{}lrrrrr@{}}\toprule',
        r'System & $n$ & Degree range & Median ms & 95th pct. ms & Max. ms \\\midrule']
runtime_labels = {'acrobot': 'Acrobot', 'kepler': 'Kepler', 'moiety': 'Moiety',
                  'rigid_body': 'Rigid body', 'twisted_cubic': 'Twisted cubic'}
for r in diagnostics['by_system']:
    label = runtime_labels[r['system'].removesuffix('_pert')]
    if r['system'].endswith('_pert'):
        label += ' (twin)'
    out.append(f"{label} & {r['n']} & {r['min_degree']}--{r['max_degree']} & {r['median_ms']:.1f} & {r['p95_ms']:.1f} & {r['max_ms']:.1f}" + r' \\')
out.append(r'\bottomrule\end{tabular}}')
os.makedirs("docs", exist_ok=True)
open("docs/tables_generated.tex", "w").write("\n".join(out) + "\n")
print("wrote docs/tables_generated.tex;  top inflation:"); [print("  ", x) for x in infl[:10]]
print("LLM max infl", max(x[0] for x in llm_infl), " nonzero cells", sum(1 for x in llm_infl if x[0] > 0), "/", len(llm_infl))
print("SINDy infl range", min(x[0] for x in sin_infl), max(x[0] for x in sin_infl))
