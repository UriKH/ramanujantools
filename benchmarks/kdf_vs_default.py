"""KdF (native FLINT steps / inverse) vs the default CMF path on the same sympy matrices (= ramanujantools before
this branch). Each measurement in its own process, capped. Results: benchmarks/kdf_vs_default.jsonl"""
import json, os, subprocess, sys
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
OUT = os.path.join(HERE, "kdf_vs_default.jsonl"); CAP = 300
CASES = [  # label, counts, arguments
    ("(1,1,0,1,1,0) generic, symbolic x,y", (1,1,0,1,1,0), "x,y"),
    ("(2,1,0,1,1,0) generic, 1/7,1/11", (2,1,0,1,1,0), "1/7,1/11"),
    ("(2,1,1,1,1,1) generic, 1/7,1/11", (2,1,1,1,1,1), "1/7,1/11"),
    ("(2,1,0,1,1,0) specialized, x=y=1", (2,1,0,1,1,0), "1,1"),
    ("(2,1,1,1,1,1) specialized, x=y=1", (2,1,1,1,1,1), "1,1"),
    ("(2,2,1,2,1,0) specialized, x=y=1/3", (2,2,1,2,1,0), "1/3,1/3"),
    ("(3,2,1,3,1,0) specialized, x=y=1/4 (L2chi)", (3,2,1,3,1,0), "1/4,1/4"),
]
WORKER = r'''
import json, sys, time, sympy as sp
from sympy.abc import x, y
from ramanujantools import Position
from ramanujantools.cmf import CMF, KdF
counts = tuple(json.loads(sys.argv[1])); args = [sp.sympify(v) for v in sys.argv[2].split(",")]
what, impl = sys.argv[3], sys.argv[4]
cmf = KdF(*counts, *args)
target = cmf if impl == "new" else CMF(dict(cmf.matrices), validate=False)
axes = list(cmf.matrices.keys())
start = Position({a: sp.Rational(2 * k + 3, 7) * (-1 if str(a).startswith("n") else 1) for k, a in enumerate(axes)})
began = time.time()
if what == "trajectory, positive steps":
    target.trajectory_matrix(Position({a: 1 for a in axes}), start)
elif what == "trajectory, negative steps":
    target.trajectory_matrix(Position({a: (-1) ** k * (k % 2 + 1) for k, a in enumerate(axes)}), start)
elif what == "M(nd0, sign=False)":
    target.M(sp.Symbol("nd0"), False)
print(json.dumps(time.time() - began))
'''
done = set()
if os.path.exists(OUT):
    for l in open(OUT): r = json.loads(l); done.add((r["case"], r["what"], r["impl"]))
for label, counts, args in CASES:
    for what in ("trajectory, positive steps", "trajectory, negative steps", "M(nd0, sign=False)"):
        for impl in ("original", "new"):
            if (label, what, impl) in done: continue
            try:
                p = subprocess.run([sys.executable, "-c", WORKER, json.dumps(counts), args, what, impl], capture_output=True,
                                   text=True, timeout=CAP, cwd=ROOT, env=dict(os.environ, PYTHONPATH=ROOT))
                seconds = json.loads(p.stdout.strip().splitlines()[-1]) if p.returncode == 0 else None
                status = "ok" if p.returncode == 0 else "error: " + p.stderr.strip().splitlines()[-1][:120]
            except subprocess.TimeoutExpired:
                seconds, status = None, f"timeout {CAP}s"
            with open(OUT, "a") as f:
                f.write(json.dumps(dict(case=label, what=what, impl=impl, seconds=seconds, status=status)) + "\n")
            print(label, what, impl, seconds, status, flush=True)
