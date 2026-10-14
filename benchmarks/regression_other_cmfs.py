"""Same measurements on this branch and on master, for CMFs other than KdF. Each run in a fresh process,
best of 3. Usage: regression_other_cmfs.py <label> <repository root>. Results: benchmarks/regression.jsonl"""
import json, os, subprocess, sys
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "regression.jsonl")
label, root = sys.argv[1], sys.argv[2]
CASES = {
    "pFq(2,1,1/2) trajectory_matrix": "from ramanujantools.cmf import pFq; c=pFq(2,1,sp.Rational(1,2)); x0,x1,y0=sp.symbols('x0 x1 y0'); f=lambda: c.trajectory_matrix({x0:1,x1:2,y0:3},{x0:1,x1:2,y0:3})",
    "pFq(3,2,1) trajectory_matrix": "from ramanujantools.cmf import pFq; c=pFq(3,2,1); x=sp.symbols('x0:3'); y=sp.symbols('y0:2'); t={**{a:1 for a in x},**{b:2 for b in y}}; f=lambda: c.trajectory_matrix(t, {**{a:1 for a in x},**{b:2 for b in y}})",
    "pFq(3,2,1) negative steps": "from ramanujantools.cmf import pFq; c=pFq(3,2,1); x=sp.symbols('x0:3'); y=sp.symbols('y0:2'); t={x[0]:1,x[1]:-1,x[2]:1,y[0]:-1,y[1]:2}; f=lambda: c.trajectory_matrix(t, {**{a:3 for a in x},**{b:5 for b in y}})",
    "pFq(3,2,1) walk 200": "from ramanujantools.cmf import pFq; c=pFq(3,2,1); x=sp.symbols('x0:3'); y=sp.symbols('y0:2'); t={**{a:1 for a in x},**{b:2 for b in y}}; f=lambda: c.walk(t, 200, {**{a:1 for a in x},**{b:2 for b in y}})",
    "MeijerG(1,1,2,2) trajectory_matrix": "from ramanujantools.cmf import MeijerG; c=MeijerG(1,1,2,2,sp.Rational(1,3)); s=sp.symbols('a0 a1 b0 b1'); f=lambda: c.trajectory_matrix({a:1 for a in s},{a:sp.Rational(k+2,3) for k,a in enumerate(s)})",
    "MeijerG(2,2,3,3) trajectory_matrix": "from ramanujantools.cmf import MeijerG; c=MeijerG(2,2,3,3,sp.Rational(1,3)); s=sp.symbols('a0 a1 a2 b0 b1 b2'); f=lambda: c.trajectory_matrix({a:(-1)**k for k,a in enumerate(s)},{a:sp.Rational(k+2,3) for k,a in enumerate(s)})",
    "zeta3 trajectory_matrix": "from ramanujantools.cmf import zeta3; from sympy.abc import x, y; c=zeta3(); f=lambda: c.trajectory_matrix({x:1,y:1},{x:1,y:1})",
    "pi walk 1000": "from ramanujantools.cmf import pi; from sympy.abc import x, y; c=pi(); f=lambda: c.walk({x:1,y:1}, 1000, {x:1,y:1})",
}
WORKER = "import time, sympy as sp\n{setup}\n_began=time.time(); f(); print(time.time()-_began)"
done = {(r['tree'], r['case']) for r in map(json.loads, open(OUT))} if os.path.exists(OUT) else set()
for name, setup in CASES.items():
    if (label, name) in done: continue
    times = []
    for _ in range(3):
        p = subprocess.run([sys.executable, "-c", WORKER.format(setup=setup)], capture_output=True, text=True,
                           cwd=root, env=dict(os.environ, PYTHONPATH=root), timeout=600)
        times.append(float(p.stdout.strip().splitlines()[-1]) if p.returncode == 0 else None)
        if p.returncode: print(name, p.stderr[-300:])
    best = min(t for t in times if t is not None) if any(t is not None for t in times) else None
    with open(OUT, "a") as f: f.write(json.dumps(dict(tree=label, case=name, seconds=best)) + "\n")
    print(label, name, best, flush=True)
