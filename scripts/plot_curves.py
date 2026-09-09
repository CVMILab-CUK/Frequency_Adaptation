#!/usr/bin/env python3
"""Plot the operating curves from one or more eval `results.json` files.

Reads only what the evaluator measured. A metric recorded as null is left out
of the plot rather than interpolated, so a gap in a curve means "not measured"
and nothing else.
"""
import argparse, json, os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("results", nargs="+", help="paths to results.json")
ap.add_argument("--labels", default=None, help="comma-separated series labels")
ap.add_argument("--out", default="./results/figures")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

runs = []
for i, p in enumerate(a.results):
    with open(p) as f:
        runs.append((json.load(f), p))
labels = (a.labels.split(",") if a.labels
          else [os.path.basename(os.path.dirname(p)) for _, p in runs])

METRICS = [("SC", "Structure consistency", "up"),
           ("EdgeF1", "Edge F1", "up"),
           ("LPIPS", "LPIPS to reference", "down"),
           ("CLIP", "CLIP score", "up"),
           ("Diversity", "Sample diversity (LPIPS)", "up"),
           ("FID", "FID", "down")]

fig, axes = plt.subplots(2, 3, figsize=(16, 9))
for ax, (key, title, direction) in zip(axes.ravel(), METRICS):
    for (res, _), lab in zip(runs, labels):
        xs, ys, es = [], [], []
        for r, v in sorted(res["per_cutoff"].items(), key=lambda kv: float(kv[0])):
            m = v.get(key)
            if m is None:
                continue
            xs.append(float(r))
            if isinstance(m, dict):
                ys.append(m["mean"]); es.append(m.get("sem", 0.0))
            else:
                ys.append(float(m)); es.append(0.0)
        if not xs:
            continue
        ax.errorbar(xs, ys, yerr=es, marker="o", capsize=3, label=lab)
    ax.set_xlabel("cutoff  r"); ax.set_ylabel(title)
    ax.set_title(f"{title}  ({'higher' if direction=='up' else 'lower'} is better)")
    ax.grid(alpha=0.3)
    if len(runs) > 1:
        ax.legend(fontsize=8)
fig.suptitle("Frequency Adaptation — operating curves", fontsize=14)
fig.tight_layout()
p1 = os.path.join(a.out, "operating_curves.png")
fig.savefig(p1, dpi=150); print("wrote", p1)

# Headline: the trade-off itself, parameterised by r.
fig2, ax = plt.subplots(figsize=(7, 6))
plotted = False
for (res, _), lab in zip(runs, labels):
    pts = []
    for r, v in sorted(res["per_cutoff"].items(), key=lambda kv: float(kv[0])):
        if v.get("FID") is None or v.get("SC") is None:
            continue
        pts.append((float(r), v["SC"]["mean"], v["FID"]))
    if not pts:
        continue
    plotted = True
    rs, sc, fid = zip(*pts)
    ax.plot(sc, fid, "-o", label=lab)
    for r_, s_, f_ in pts:
        ax.annotate(f"r={r_}", (s_, f_), fontsize=7,
                    textcoords="offset points", xytext=(4, 4))
ax.set_xlabel("Structure consistency  (higher = obeys the condition)")
ax.set_ylabel("FID  (lower = more realistic)")
ax.set_title("Structure-fidelity vs realism, traced by one model")
ax.grid(alpha=0.3)
if plotted:
    ax.legend()
    p2 = os.path.join(a.out, "tradeoff.png")
    fig2.tight_layout(); fig2.savefig(p2, dpi=150); print("wrote", p2)
else:
    print("trade-off plot skipped: no run has both FID and SC measured")
