#!/usr/bin/env python3
"""A chance floor for the out-of-domain SC_vs_filtered readings.

Sec. 4.6 reports that in-domain and out-of-domain runs follow the filtered
condition about equally closely at every cutoff. That only means something if
the numbers are above chance, and at r=0.3 the condition is nearly flat. This
re-scores the saved generations of E21 against a *different* input's filtered
condition (a fixed derangement), which is the same floor scripts/fixed_
instrument.py uses for the main sweep.
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
from PIL import Image
from datalibs.frequency import high_pass_torch, rgb_to_gray_torch
from eval.metrics import structure_consistency

ap = argparse.ArgumentParser()
ap.add_argument("--runs", nargs="+",
                default=["results/E21_indomain_photos", "results/E21_ood_photos"])
ap.add_argument("--batch", type=int, default=25)
ap.add_argument("--out", default="results/E25_ood_floor/results.json")
a = ap.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"

def load(paths):
    xs = [np.asarray(Image.open(p).convert("RGB"), np.float32) / 255.0 for p in paths]
    return torch.from_numpy(np.stack(xs)).permute(0, 3, 1, 2).to(dev)

out = {"provenance": {"floor": "generation i against the filtered condition of a "
                               "different input, fixed derangement (seed 0)",
                      "device": dev}}
for run in a.runs:
    prov = json.load(open(os.path.join(run, "results.json")))["provenance"]
    src = prov["sketch_dir"]
    files = sorted(f for f in os.listdir(src) if f.lower().endswith((".png", ".jpg", ".jpeg")))
    files = files[: prov["n_images"]]
    n = len(files)
    perm = np.roll(np.arange(n), n // 2)          # a derangement, deterministic
    for cut in prov["cutoffs"]:
        gdir = os.path.join(run, f"r{cut}")
        if not os.path.isdir(gdir): continue
        gnames = sorted(f for f in os.listdir(gdir) if f.endswith(".png"))
        if len(gnames) != n: print("skip", gdir, len(gnames), "vs", n); continue
        same, shuf = [], []
        for i in range(0, n, a.batch):
            j = min(i + a.batch, n)
            x = load([os.path.join(src, f) for f in files[i:j]])
            xs = load([os.path.join(src, files[p]) for p in perm[i:j]])
            g = load([os.path.join(gdir, f) for f in gnames[i:j]])
            def cond(t):
                if cut <= 0.0:
                    return rgb_to_gray_torch(t).repeat(1, 3, 1, 1).clamp(0, 1)
                f_ = high_pass_torch(rgb_to_gray_torch(t), cut)
                return f_.repeat(1, 3, 1, 1).clamp(0, 1) if f_.shape[1] == 1 else f_.clamp(0, 1)
            same += list(structure_consistency(g, cond(x), 0.0))
            shuf += list(structure_consistency(g, cond(xs), 0.0))
        key = f"{os.path.basename(run)}/r{cut}"
        s, f_ = np.array(same), np.array(shuf)
        out[key] = {"SC_vs_filtered": {"mean": float(s.mean()), "sem": float(s.std(ddof=1) / np.sqrt(n))},
                    "SC_vs_filtered_floor": {"mean": float(f_.mean()), "sem": float(f_.std(ddof=1) / np.sqrt(n))},
                    "n": n}
        d = (s.mean() - f_.mean()) / (s.std(ddof=1) / np.sqrt(n))
        out[key]["above_floor_sems"] = float(d)
        print(f"{key}: {s.mean():.4f} vs floor {f_.mean():.4f}  ({d:.1f} SEM above)", flush=True)

os.makedirs(os.path.dirname(a.out), exist_ok=True)
json.dump(out, open(a.out, "w"), indent=1)
print("wrote", a.out)
