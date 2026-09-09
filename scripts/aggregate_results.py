#!/usr/bin/env python3
"""Collect every results.json under a directory into one table.

Runs that have not been evaluated simply do not appear; cells for metrics that
were not computed print as an empty marker. Nothing is filled in.
"""
import argparse, glob, json, os

ap = argparse.ArgumentParser()
ap.add_argument("--root", default="./results")
ap.add_argument("--csv", default=None, help="also write a CSV here")
a = ap.parse_args()

rows = []
for path in sorted(glob.glob(os.path.join(a.root, "**", "results.json"), recursive=True)):
    with open(path) as f:
        res = json.load(f)
    run = os.path.basename(os.path.dirname(path))
    prov = res.get("provenance", {})
    block = res.get("per_cutoff") or res.get("per_setting") or {}
    for setting, v in sorted(block.items(), key=lambda kv: str(kv[0])):
        def g(k):
            m = v.get(k)
            if m is None:
                return None
            return m["mean"] if isinstance(m, dict) else m
        rows.append({
            "run": run, "setting": setting,
            "SC": g("SC"), "EdgeF1": g("EdgeF1"), "LPIPS": g("LPIPS"),
            "CLIP": g("CLIP"), "Diversity": g("Diversity"), "FID": g("FID"),
            "n": (v.get("SC") or v.get("EdgeF1") or {}).get("n"),
            "ckpt": os.path.basename(str(prov.get("adapter", prov.get("repo", "")))),
        })

if not rows:
    raise SystemExit(f"no results.json found under {a.root} -- nothing has been evaluated yet")

cols = ["run", "setting", "SC", "EdgeF1", "LPIPS", "CLIP", "Diversity", "FID", "n"]
w = {c: max(len(c), *(len(f"{r[c]:.4f}" if isinstance(r[c], float) else str(r[c] if r[c] is not None else "--")) for r in rows)) for c in cols}
print(" ".join(c.rjust(w[c]) for c in cols))
print("-" * (sum(w.values()) + len(cols) - 1))
for r in rows:
    cells = []
    for c in cols:
        v = r[c]
        cells.append(("--" if v is None else (f"{v:.4f}" if isinstance(v, float) else str(v))).rjust(w[c]))
    print(" ".join(cells))
print("\n'--' means the metric was not computed, not that it is zero.")

if a.csv:
    import csv
    with open(a.csv, "w", newline="") as f:
        wtr = csv.DictWriter(f, fieldnames=cols + ["ckpt"])
        wtr.writeheader()
        for r in rows:
            wtr.writerow(r)
    print(f"wrote {a.csv}")
