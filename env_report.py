"""Print the environment (paste this into any bug report / reproduction claim).
Usage: python env_report.py > results/environment_<machine>.txt"""
import platform, sys, os, subprocess
print("python   ", sys.version.split()[0], platform.platform(), platform.machine())
for m in ("numpy", "scipy", "sympy", "pysindy"):
    try: print(f"{m:<9}", __import__(m).__version__)
    except Exception as e: print(f"{m:<9} not importable ({type(e).__name__})")
try:
    import numpy; cfg = numpy.show_config(mode="dicts")
    b = cfg.get("Build Dependencies", {}).get("blas", {})
    print("blas     ", b.get("name"), b.get("version"))
except Exception: pass
from pathlib import Path
r = Path(os.environ.get("ALGEBRAIC_RL_REPO") or Path(__file__).resolve().parent / "vendor/algebraicRLtest").resolve()
if (r / ".git").exists():
    try:
        print("algebraicRLtest commit", subprocess.check_output(
            ["git", "-C", str(r), "rev-parse", "HEAD"], text=True).strip())
    except subprocess.CalledProcessError:
        print("algebraicRLtest commit: unavailable")
else:
    print("algebraicRLtest bundled commit 687c868c204b8ec537451b5e046532de80197d9d")
