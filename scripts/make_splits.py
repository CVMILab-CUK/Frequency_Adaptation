#!/usr/bin/env python3
"""Regenerate train/valid/test splits from the images actually present on disk.

The splits shipped in config/data/ referenced a dataset copy that no longer
exists on this machine, so they are rebuilt here from the real file listing
with a fixed seed -- the split has to be reproducible for the paper.
"""
import argparse, os, random

ap = argparse.ArgumentParser()
ap.add_argument("--img_dir", default="/home/work/data/ffhq512/images")
ap.add_argument("--out_dir", default="./config/data")
ap.add_argument("--val_rate", type=float, default=0.02)
ap.add_argument("--test_rate", type=float, default=0.05)
ap.add_argument("--seed", type=int, default=2026)
a = ap.parse_args()

files = sorted(f for f in os.listdir(a.img_dir) if f.lower().endswith((".png", ".jpg", ".jpeg")))
if not files:
    raise SystemExit(f"no images in {a.img_dir}")
random.Random(a.seed).shuffle(files)

n = len(files)
n_test = int(n * a.test_rate)
n_val = int(n * a.val_rate)
splits = {
    "test": files[:n_test],
    "valid": files[n_test:n_test + n_val],
    "train": files[n_test + n_val:],
}
os.makedirs(a.out_dir, exist_ok=True)
for name, lst in splits.items():
    with open(os.path.join(a.out_dir, f"{name}.txt"), "w") as f:
        f.write("\n".join(lst) + "\n")
    print(f"{name:6s} {len(lst):6d}")
print(f"total  {n:6d}  seed={a.seed}")
