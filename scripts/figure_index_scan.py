#!/usr/bin/env python3
"""T02 follow-up: dial.png rows 2-3 match no saved generation at their own image
index. Scan every test index of the candidate runs to find what the tiles are."""
import glob, os, sys
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim

TILE, GAP = 384, 6
im = Image.open("papers/figures/dial.png").convert("RGB")
def tile(r, c):
    s = TILE + GAP
    return np.asarray(im.crop((c * s, r * s, c * s + TILE, r * s + TILE)))

RUNS = ["E10b", "E13_seed2027", "E13_seed2028", "E16_generic_caption"]
COLS = {1: "0.0", 2: "0.05", 3: "0.1", 4: "0.2", 5: "0.3", 6: "0.5", 7: "0.7", 8: "1.0"}

for row in (1, 2):
    probe = tile(row, 3)          # r=0.1 column
    best = []
    for run in RUNS:
        d = f"results/{run}/gen_r0.1"
        if not os.path.isdir(d):
            continue
        for p in sorted(glob.glob(d + "/*.png")):
            a = np.asarray(Image.open(p).convert("RGB").resize((TILE, TILE), Image.LANCZOS))
            best.append((ssim(probe, a, channel_axis=2), run, os.path.basename(p)[:-4]))
    best.sort(reverse=True)
    print(f"\n=== dial row {row}, r=0.1 column: top 5 over all indices")
    for s, run, idx in best[:5]:
        print(f"   {s:.4f}  {run}  idx {idx}")
    s0, run0, idx0 = best[0]
    print(f"   -> verifying idx {idx0} of {run0} across all cutoffs:")
    for c, r in COLS.items():
        p = f"results/{run0}/gen_r{r}/{idx0}.png"
        if os.path.exists(p):
            a = np.asarray(Image.open(p).convert("RGB").resize((TILE, TILE), Image.LANCZOS))
            print(f"      col{c} r={r:4s} ssim={ssim(tile(row, c), a, channel_axis=2):.4f}")
    # is the reference column of this row the reference of idx0?
    for ref_idx in [f"{row:05d}", idx0]:
        p = f"results/E10b/reference/{ref_idx}.png"
        if os.path.exists(p):
            a = np.asarray(Image.open(p).convert("RGB").resize((TILE, TILE), Image.LANCZOS))
            print(f"      reference col vs reference {ref_idx}: ssim={ssim(tile(row, 0), a, channel_axis=2):.4f}")
