"""Does the scanned annotation on the source drawings inflate the sketch SC?

Thirty of the forty forensic drawings in results/E6_face/input carry a filename
stamped across the top of the scan, and the model reproduces it. That is a
faithful response to the condition it was given, but it means part of the
structure score is agreement on text rather than on a face. This recomputes the
score with the whole top fifth of the frame removed -- a deliberate
over-estimate of the band -- so the paper can state the size of the effect
instead of asserting there isn't one.

Reads only images already on disk; no GPU, no generation.
"""
import json, os, sys
import numpy as np, torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eval.metrics import structure_consistency

ROOT = "results/E6_face"
MASK_FRAC = 0.20            # everything above this line is discarded

files = sorted(os.listdir(f"{ROOT}/input"))

def load(sub):
    a = np.stack([np.asarray(Image.open(f"{ROOT}/{sub}/{f}").convert("RGB"))
                  for f in files])
    return torch.from_numpy(a).permute(0, 3, 1, 2).float() / 255.0

cond, gen = load("input"), load("generated")
H = cond.shape[2]
k = int(round(H * MASK_FRAC))

def agg(x):
    return {"mean": float(x.mean()), "std": float(x.std()),
            "sem": float(x.std() / np.sqrt(len(x))), "n": int(len(x))}

full   = structure_consistency(gen, cond, 0.0)
masked = structure_consistency(gen[:, :, k:, :].contiguous(),
                               cond[:, :, k:, :].contiguous(), 0.0)

out = {"note": "SC recomputed on the saved images with the top MASK_FRAC of "
                "each frame removed, to bound the contribution of the scanned "
                "filename annotation on the source drawings.",
       "mask_frac": MASK_FRAC, "rows_removed": k, "height": H,
       "SC_full_frame": agg(full), "SC_band_removed": agg(masked),
       "delta": float(masked.mean() - full.mean())}

with open(f"{ROOT}/band_check.json", "w") as f:
    json.dump(out, f, indent=1)

print(f"full frame        SC = {full.mean():.4f} +- {out['SC_full_frame']['sem']:.4f}")
print(f"top {k:3d} rows cut  SC = {masked.mean():.4f} +- {out['SC_band_removed']['sem']:.4f}")
print(f"difference           = {out['delta']:+.4f}  "
      f"({abs(out['delta']) / out['SC_full_frame']['sem']:.2f} SEM)")
