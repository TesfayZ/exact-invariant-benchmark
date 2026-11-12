"""Offline grading of cached raw outputs. Re-runnable; never calls a model.
Reports outer and reconstructed verdicts at caps D in {10,100,1000}. A change in
either flags rationalisation_sensitive; the headline verdict uses D=1000."""
import argparse, json, re, os
import systems, grade, data

FIT_TOL = 1e-3          # declared data-fit threshold for 'plausible_but_wrong' (results are also reported at 1e-2 and 5e-2)
DS = (10, 100, 1000)
_cache = {}

def ctx(key):
    if key not in _cache:
        if key.startswith('planted_'):
            from fresh_systems import REGISTRY
            systems.REGISTRY.update(REGISTRY)
        s = systems.REGISTRY[key]()
        if s.vanishing_task:
            fn = data.make_residual_fn(s, data.curve_points(s)); _cache[key] = (s, dict(data_residual_fn=fn))
        elif s.family == 'planted':
            _cache[key] = (s, {})  # prospective equation-only suite: exact grading, no drift claim
        else:
            fn = data.make_drift_fn(s, data.trajectories(s, n_traj=6, T=4.0)); _cache[key] = (s, dict(drift_fn=fn))
    return _cache[key]

def extract(output):
    """Return (list_of_polys | None for refusal, parse_note)."""
    if output is None: return None, "no_output"
    if not output.strip(): return None, "empty_output"
    # robust: scan every '{', decode a JSON object there, keep the LAST one that has a 'polynomials' key
    dec, j, i = json.JSONDecoder(), None, 0
    while True:
        i = output.find("{", i)
        if i < 0: break
        try:
            obj, end = dec.raw_decode(output[i:])
            if isinstance(obj, dict) and "polynomials" in obj: j = obj
            i += max(end, 1)
        except Exception:
            i += 1
    if j is None:
        # Consistent with grade.grade: a bare REFUSE / NONE string is a refusal, not unparseable.
        if output.strip().upper() in ("REFUSE", "NONE"): return None, "refused"
        return [], "no_json"
    if not isinstance(j['polynomials'], list) or not all(isinstance(x,str) for x in j['polynomials']):
        return [], 'invalid_polynomials_schema'
    if 'refuse' in j and not isinstance(j['refuse'], bool):
        return [], 'invalid_refusal_schema'
    if j.get("refuse", False) or not j['polynomials']:
        return None, "refused"
    return j['polynomials'], "ok"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="results/primary/raw_groq.jsonl"); ap.add_argument("--out", default="results/primary/graded_groq.jsonl")
    a = ap.parse_args()
    if not os.path.exists(a.raw): raise SystemExit(f"{a.raw} not found: run run_llm.py or sr_sindy.py first")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    out = open(a.out, "w")
    for line in open(a.raw):
        r = json.loads(line)
        s, kw = ctx(r["key"])
        polys, note = extract(r["output"])
        if r.get("error"):
            res = {D: dict(verdict="grader_unavailable", note=r["error"][:200]) for D in DS}
        elif polys is None:
            # FIX: null content ("no_output") is an empty completion, not a refusal.
            # Refused = explicit {"refuse": true} / empty polynomial list, or a bare REFUSE/NONE string.
            # Any other non-JSON text is "unparseable" (see extract).
            v = "empty_output" if note in ("empty_output", "no_output") else "refused"
            res = {D: dict(verdict=v) for D in DS}
        elif not polys:
            res = {D: dict(verdict="unparseable", note=note) for D in DS}
        else:
            res = {}
            for D in DS:
                arg = polys if s.vanishing_task else polys[0]
                try:
                    res[D] = grade.grade(arg, s, D=D, fit_tol=FIT_TOL, **kw)
                except Exception as e:
                    res[D] = dict(verdict="grader_unavailable", note=str(e)[:200])
        verdicts = {D: res[D]["verdict"] for D in DS}
        reconstructed = {D: res[D].get("reconstructed_verdict") for D in DS}
        # An unchanged expression needs no separate reconstructed sub-verdict.
        outcomes = {(verdicts[D], reconstructed[D] or verdicts[D]) for D in DS}
        rec = {k: r[k] for k in ("model", "system", "condition", "noise", "sample")}
        # Policy (documented in the paper, Sec. 4): the headline verdict of a conserved-quantity task
        # is that of the FIRST polynomial (the prompt asks for one). Every returned polynomial is also
        # graded at D=1000 and stored in `verdicts_all`, so the first-only rule can be audited.
        verdicts_all = None
        if polys and not s.vanishing_task and len(polys) > 1:
            verdicts_all = []
            for q in polys:
                try: verdicts_all.append(grade.grade(q, s, D=1000, fit_tol=FIT_TOL, **kw)["verdict"])
                except Exception: verdicts_all.append("grader_unavailable")
        rec.update(verdicts_all=verdicts_all, rounding_err=res[1000].get("rounding_err", 0.0),
                   coefficient_policy="literal-decimal-rational-v1",
                   drift_policy="centred-quotient-grevlex-v1")
        rec.update(verdict=verdicts[1000], verdict_by_D=verdicts,
                   reconstructed_verdict_by_D=reconstructed,
                   rationalisation_sensitive=len(outcomes) > 1,
                   detail=res[1000], n_polys=0 if polys is None else len(polys),
                   multi_poly_conserved=(not s.vanishing_task and polys is not None and len(polys) > 1))
        out.write(json.dumps(rec, default=str) + "\n")
    print("graded ->", a.out)

if __name__ == "__main__":
    main()
