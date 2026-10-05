#!/usr/bin/env python3
"""E26 -- are equal steps in r equal steps in the picture?

Limitations says this is not measured: "whether equal steps in $r$ are perceived
as equal changes ... is not measured here". A user study is the right instrument
and is not in reach yet. This is the proxy that is: LPIPS between the
generations at consecutive cutoffs, same image, same seed, so the only thing
that differs between the two pictures is r.

  step_lpips[a->b]     perceptual distance between the generations at a and b
  step_rate[a->b]      that distance divided by (b - a), so a dial that moved
                       the picture at a constant rate would give a flat column
  cum_lpips[r]         distance from the r=0 generation, the dial's travel

A flat step_rate would mean r is already a perceptually uniform parameter. A
falling one means the early part of the range does most of the work and a user
would want a non-linear mapping on the slider. Either answer is publishable;
what the paper cannot keep doing is not answering.

Reads only generations that eval/run_eval.py already saved.
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("--run", default="results/E10b")
ap.add_argument("--out", default="results/E26_dial_linearity/results.json")
ap.add_argument("--cutoffs", default="0.0,0.05,0.1,0.2,0.3,0.5,0.7,1.0")
ap.add_argument("--n", type=int, default=500)
ap.add_argument("--batch", type=int, default=25)
ap.add_argument("--device", default="cuda")
a = ap.parse_args()

import lpips
net = lpips.LPIPS(net="alex").to(a.device).eval()

R = [float(x) for x in a.cutoffs.split(",")]
names = sorted(os.listdir(os.path.join(a.run, "reference")))[:a.n]

def load(cut, batch_names):
    d = os.path.join(a.run, f"gen_r{cut}")
    ims = [np.asarray(Image.open(os.path.join(d, nm)).convert("RGB"), dtype=np.float32) / 255.0
           for nm in batch_names]
    x = torch.from_numpy(np.stack(ims)).permute(0, 3, 1, 2).to(a.device)
    return x * 2.0 - 1.0                      # lpips wants [-1, 1]

def fmt(c):
    """The directory names use the same spelling as the cutoff list."""
    return c

def pairwise(ca, cb):
    vals = []
    with torch.no_grad():
        for s in range(0, len(names), a.batch):
            nb = names[s:s + a.batch]
            d = net(load(ca, nb), load(cb, nb)).flatten()
            vals.append(d.cpu().numpy())
    return np.concatenate(vals)

def stat(v):
    return {"mean": float(v.mean()),
            "sem": float(v.std(ddof=1) / np.sqrt(v.size)),
            "n": int(v.size)}

cuts = [x.strip() for x in a.cutoffs.split(",")]
out = {"provenance": {
    "run": a.run, "n_images": len(names), "metric": "LPIPS AlexNet (eval/metrics.py's network)",
    "definition": __doc__.strip(),
    "note": "re-scores saved generations; no new inference"}}

print(f"{'step':>14} {'dr':>6} {'LPIPS':>18} {'per unit r':>12}")
for i in range(len(cuts) - 1):
    ca, cb = cuts[i], cuts[i + 1]
    dr = R[i + 1] - R[i]
    v = pairwise(ca, cb)
    s = stat(v)
    out[f"step_lpips@{ca}->{cb}"] = s
    out[f"step_rate@{ca}->{cb}"] = {"mean": s["mean"] / dr, "sem": s["sem"] / dr, "n": s["n"]}
    print(f"{ca+'->'+cb:>14} {dr:>6.2f} {s['mean']:>11.4f}±{s['sem']:.4f} {s['mean']/dr:>12.3f}", flush=True)

print()
for i in range(1, len(cuts)):
    v = pairwise(cuts[0], cuts[i])
    s = stat(v)
    out[f"cum_lpips@{cuts[i]}"] = s
    print(f"  from r=0 to r={cuts[i]:>4}: {s['mean']:.4f}±{s['sem']:.4f}", flush=True)

os.makedirs(os.path.dirname(a.out), exist_ok=True)
json.dump(out, open(a.out, "w"), indent=1)
print(f"\nwrote {a.out}")
