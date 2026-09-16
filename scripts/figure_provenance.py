#!/usr/bin/env python3
"""T02: match every generated tile of figures/dial.png and figures/strength.png
against the saved evaluation generations, so each tile's run is identified.

Tiles are cropped with the geometry the caption code uses (dial: 384 px tiles,
6 px gaps, 3x9; strength: 256 px tiles, no gap, 2x9). Candidates are every
results/<run>/gen_r<r>/<idx>.png plus the reference sets, downsized to the tile
size and compared with SSIM; the best match per tile is written to
papers/figures/provenance_check.csv together with the runner-up.
"""
import csv, glob, os, re, sys
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr

R = "results"
FIGS = {
    "dial":     dict(path="papers/figures/dial.png", tile=384, gap=6, rows=3, cols=9,
                     labels=["reference", "r=0", "r=0.05", "r=0.1", "r=0.2", "r=0.3", "r=0.5", "r=0.7", "r=1"],
                     idx=["00000", "00001", "00002"]),
    "strength": dict(path="papers/figures/strength.png", tile=256, gap=0, rows=2, cols=9,
                     labels=["reference", "fixed 1.00", "fixed 0.75", "fixed 0.50", "fixed 0.25",
                             "rand 0.50", "rand 0.25", "rand r=0.1", "rand r=0.3"],
                     idx=["00000", "00001"]),
}

def candidates(idx):
    out = []
    for d in sorted(glob.glob(f"{R}/*/gen_r*")) + sorted(glob.glob(f"{R}/*/reference")):
        p = os.path.join(d, f"{idx}.png")
        if os.path.exists(p):
            run = os.path.relpath(d, R)
            out.append((run, p))
    return out

def main():
    rows_out = []
    for fig, cfg in FIGS.items():
        im = Image.open(cfg["path"]).convert("RGB")
        step = cfg["tile"] + cfg["gap"]
        for r_i, idx in enumerate(cfg["idx"]):
            cands = candidates(idx)
            cache = []
            for run, p in cands:
                a = np.asarray(Image.open(p).convert("RGB").resize((cfg["tile"], cfg["tile"]), Image.LANCZOS))
                cache.append((run, a))
            print(f"[{fig}] row {r_i} idx {idx}: {len(cache)} candidates", flush=True)
            for c_i in range(cfg["cols"]):
                tile = np.asarray(im.crop((c_i * step, r_i * step, c_i * step + cfg["tile"], r_i * step + cfg["tile"])))
                scored = sorted(((ssim(tile, a, channel_axis=2), run, a) for run, a in cache), key=lambda t: -t[0])
                (s1, run1, a1), (s2, run2, _) = scored[0], scored[1]
                rows_out.append(dict(figure=fig, row=r_i, col=c_i, label=cfg["labels"][c_i],
                                     best_match_run=run1, image_idx=idx,
                                     ssim=f"{s1:.4f}", psnr=f"{psnr(tile, a1):.2f}",
                                     second_run=run2, second_ssim=f"{s2:.4f}"))
                print(f"  {fig} r{r_i}c{c_i} {cfg['labels'][c_i]:12s} -> {run1} ssim={s1:.4f} (2nd {run2} {s2:.4f})", flush=True)
    with open("papers/figures/provenance_check.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys())); w.writeheader(); w.writerows(rows_out)
    print("wrote papers/figures/provenance_check.csv", len(rows_out), "rows")

if __name__ == "__main__":
    main()
