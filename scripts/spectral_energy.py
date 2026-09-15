#!/usr/bin/env python3
"""E22 -- how much of the spectrum the condition filter removes, measured.

The paper quotes how fast the condition empties as r grows. Those figures used
to be typed by hand; this script measures them on the held-out reference
photographs with the exact mask the condition uses (datalibs.frequency), so the
text can take them from results/ like every other number.

  energy_removed[r]   fraction of grayscale spectral energy |F|^2 (DC included)
                      inside the cut disc of normalised radius r
  energy_retained[r]  1 - energy_removed[r]
  cond_std[r]         standard deviation of the condition the model receives
                      (high_pass_torch output, per-image min-max normalised)
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
from PIL import Image
from datalibs.frequency import high_pass_torch, rgb_to_gray_torch, radial_mask_torch

ap = argparse.ArgumentParser()
ap.add_argument("--ref", default="results/E10b/reference")
ap.add_argument("--out", default="results/E22_spectral_energy/results.json")
a = ap.parse_args()

R = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
dev = "cuda" if torch.cuda.is_available() else "cpu"
names = sorted(f for f in os.listdir(a.ref) if f.endswith(".png"))
acc = {f"{k}@{r}": [] for r in R for k in ("energy_removed", "cond_std")}
for i in range(0, len(names), 32):
    x = np.stack([np.asarray(Image.open(os.path.join(a.ref, f)).convert("RGB")) for f in names[i:i + 32]])
    g = rgb_to_gray_torch(torch.from_numpy(x).permute(0, 3, 1, 2).float().div(255).to(dev))
    _, _, h, w = g.shape
    e = torch.fft.fftshift(torch.fft.fft2(g), dim=(-2, -1)).abs() ** 2
    tot = e.sum(dim=(-3, -2, -1))
    for r in R:
        keep = radial_mask_torch(h, w, r, dev, "high_pass")      # 0 inside the disc
        removed = 1.0 - (e * keep).sum(dim=(-3, -2, -1)) / tot
        acc[f"energy_removed@{r}"] += removed.cpu().tolist()
        acc[f"cond_std@{r}"] += high_pass_torch(g, r).flatten(1).std(1).cpu().tolist()

res = {"provenance": {"ref": a.ref, "n_images": len(names), "definition": __doc__.strip()}}
for k, v in acc.items():
    v = np.asarray(v, np.float64)
    res[k] = {"mean": float(v.mean()), "sem": float(v.std() / np.sqrt(len(v))), "first_image": float(v[0])}
for r in R:
    res[f"energy_retained@{r}"] = {"mean": 1.0 - res[f"energy_removed@{r}"]["mean"]}
os.makedirs(os.path.dirname(a.out), exist_ok=True)
json.dump(res, open(a.out, "w"), indent=1)
for r in R:
    print(f"r={r}: removed {res[f'energy_removed@{r}']['mean']:.4f}  retained {res[f'energy_retained@{r}']['mean']:.5f}  "
          f"cond_std {res[f'cond_std@{r}']['mean']:.3f} (first {res[f'cond_std@{r}']['first_image']:.3f})")
