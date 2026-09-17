#!/usr/bin/env python3
"""Redraw Figure 1 (figures/mask.png) from datalibs.frequency.

Four rows over the eight cutoffs the paper sweeps: the radial mask, the
log-magnitude spectrum after masking, the condition the model receives, and the
same condition displayed at +/-2 sigma. Every row goes through the same
functions the dataloader uses, so the picture cannot drift from the method.

The figure was originally rasterised at 196 px per cell, about 225 dpi at
\\textwidth. --tile sets the cell content size; the default 380 gives
3136 x 1568, twice the old file and over 300 dpi.
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from PIL import Image
from datalibs.frequency import radial_mask_numpy, high_pass_numpy, RGB2GRAY_WEIGHTS

ap = argparse.ArgumentParser()
ap.add_argument("--image", default="/home/work/data/ffhq512/images/00000.png")
ap.add_argument("--out", default="papers/figures/mask.png")
ap.add_argument("--tile", type=int, default=380, help="cell content size in px")
ap.add_argument("--gap", type=int, default=12, help="white gap between cells")
ap.add_argument("--work", type=int, default=512, help="resolution the filter runs at")
a = ap.parse_args()

R = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]

rgb = np.asarray(Image.open(a.image).convert("RGB").resize((a.work, a.work), Image.LANCZOS),
                 dtype=np.float32) / 255.0
gray = (rgb * np.array(RGB2GRAY_WEIGHTS, np.float32)).sum(-1)      # as the dataloader does

def u8(x):
    return Image.fromarray((np.clip(x, 0, 1) * 255 + 0.5).astype(np.uint8))

def fit(img):
    return np.asarray(img.resize((a.tile, a.tile), Image.LANCZOS), dtype=np.uint8)

rows = [[], [], [], []]
for r in R:
    mask = radial_mask_numpy(a.work, a.work, r, "high_pass")
    rows[0].append(fit(u8(mask)))                                   # 1: the mask, black is removed

    f = np.fft.fftshift(np.fft.fft2(gray)) * mask                   # 2: what is left of the spectrum
    mag = np.log1p(np.abs(f))
    rows[1].append(fit(u8((mag - mag.min()) / (mag.max() - mag.min() + 1e-9))))

    cond = high_pass_numpy(gray, r)[:, :, 0]                        # 3: the condition itself
    rows[2].append(fit(u8(cond)))

    s = cond.std()                                                  # 4: the same data at +/-2 sigma
    rows[3].append(fit(u8((cond - cond.mean()) / (4 * s + 1e-9) + 0.5)))

t, g = a.tile, a.gap
W = 8 * t + 7 * g + g
H = 4 * t + 3 * g + g
canvas = np.full((H, W), 255, np.uint8)
for i, row in enumerate(rows):
    for j, cell in enumerate(row):
        y, x = g // 2 + i * (t + g), g // 2 + j * (t + g)
        canvas[y:y + t, x:x + t] = cell
Image.fromarray(canvas, "L").save(a.out)
print(f"{a.out}: {W}x{H} L, {a.image} at {a.work} px, cells {t} px")
