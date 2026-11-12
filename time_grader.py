"""Median grade time per system, one reference per system, 5 repetitions.
Setup (trajectories / curve points) is built once outside the timed loop, so the
number measures grade.grade (Gröbner reduction / Lie-derivative normal form).
Writes results/grade_times.txt. No API calls."""
import time, statistics, os
import systems, grade, data

def one(s):
    if s.vanishing_task:
        arg = [str(g) for g in s.constraints]
        kw = {"data_residual_fn": data.make_residual_fn(s, data.curve_points(s))}
    else:
        arg = str(s.conserved_ref[0])
        kw = {"drift_fn": data.make_drift_fn(s, data.trajectories(s, n_traj=2, T=2.0, n_pts=50))}
    ts = []
    for _ in range(5):
        t0 = time.perf_counter()
        grade.grade(arg, s, **kw)
        ts.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(ts)

os.makedirs("results", exist_ok=True)
rows = []
for k, f in systems.REGISTRY.items():
    ms = one(f())
    rows.append(f"{k:<28} {ms:8.1f} ms")
print("\n".join(rows))
open("results/grade_times.txt", "w").write("\n".join(rows) + "\n")