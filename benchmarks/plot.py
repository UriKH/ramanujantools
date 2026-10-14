"""Charts for the PR, from benchmarks/kdf_vs_default.jsonl and benchmarks/regression.jsonl (measured runs only)."""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__))
BLUE, ORANGE, INK, MUTED, SURFACE, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#fcfcfb", "#e6e5e0"
plt.rcParams.update({"font.size": 9, "text.color": INK, "axes.labelcolor": MUTED, "xtick.color": MUTED,
                     "ytick.color": INK, "axes.edgecolor": GRID, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE})
CAP = 300

def bars(ax, labels, series, title, xlabel, log=True):
    import numpy as np
    y = np.arange(len(labels)); h = 0.36
    for k, (name, values, color) in enumerate(series):
        shown = [v if v is not None else CAP for v in values]
        rects = ax.barh(y + (k - 0.5) * h, shown, h * 0.92, color=color, label=name,
                        hatch=["////" if v is None else "" for v in values], edgecolor=SURFACE, linewidth=0)
        for r, v in zip(rects, values):
            text = f"> {CAP} s (stopped)" if v is None else (f"{v:.2f} s" if v >= 0.1 else f"{v * 1000:.0f} ms")
            ax.text(r.get_width() * (1.08 if log else 1.01), r.get_y() + r.get_height() / 2, text, va="center", fontsize=7.5, color=MUTED)
    ax.set_yticks(y, labels); ax.invert_yaxis()
    if log: ax.set_xscale("log"); ax.set_xlim(1e-3, CAP * 30)
    ax.set_title(title, loc="left", fontsize=10.5, color=INK, pad=8)
    ax.set_xlabel(xlabel); ax.grid(axis="x", color=GRID, linewidth=0.6); ax.set_axisbelow(True)
    for s in ("top", "right", "left"): ax.spines[s].set_visible(False)

rows = [json.loads(l) for l in open(os.path.join(HERE, "kdf_vs_default.jsonl"))]
cases = list(dict.fromkeys(r["case"] for r in rows))
whats = ["trajectory, negative steps", "M(nd0, sign=False)", "trajectory, positive steps"]
titles = {"trajectory, negative steps": "trajectory_matrix with backward steps",
          "M(nd0, sign=False)": "M(nd0, sign=False): the step back",
          "trajectory, positive steps": "trajectory_matrix with forward steps only"}
get = {(r["case"], r["what"], r["impl"]): r["seconds"] for r in rows}
fig, axes = plt.subplots(3, 1, figsize=(9, 11), constrained_layout=True)
for ax, what in zip(axes, whats):
    bars(ax, cases, [("before (sympy step matrices / inverse)", [get[(c, what, "original")] for c in cases], ORANGE),
                     ("this PR (FLINT)", [get[(c, what, "new")] for c in cases], BLUE)], titles[what], "seconds (log scale)")
axes[2].legend(loc="lower right", frameon=False, fontsize=8.5)
fig.suptitle("KdF CMFs: time per call, before and after (one process per run, cap 300 s)", x=0.01, ha="left", fontsize=12)
fig.savefig(os.path.join(HERE, "kdf_speedup.png"), dpi=160)

reg = [json.loads(l) for l in open(os.path.join(HERE, "regression.jsonl"))]
names = list(dict.fromkeys(r["case"] for r in reg))
value = {(r["tree"], r["case"]): r["seconds"] for r in reg}
fig, ax = plt.subplots(figsize=(9, 4.6), constrained_layout=True)
bars(ax, names, [("master", [value[("master", n)] for n in names], ORANGE),
                 ("this PR", [value[("branch", n)] for n in names], BLUE)],
     "Other CMFs: no regression (best of 3 fresh processes)", "seconds", log=False)
ax.set_xlim(0, 0.1); ax.legend(loc="lower right", frameon=False, fontsize=8)
fig.savefig(os.path.join(HERE, "regression_other_cmfs.png"), dpi=160)
print("ok")
