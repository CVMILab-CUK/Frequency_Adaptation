#!/usr/bin/env python3
"""E19 -- score saved generations with an instrument that does not move with r.

Structure consistency re-extracts at the same cutoff the model was given, so a
row-to-row difference mixes model behaviour with a change of instrument. This
rescoring holds the instrument fixed and asks two things of the saved images,
against the 500 reference photographs, never regenerating anything:

  sc_fix[r_m]  Pearson correlation of high_pass(gen, r_m) with
               high_pass(reference, r_m) at fixed r_m in {0.05, 0.1, 0.3}
  band[k]      Pearson correlation of the band-pass components of gen and
               reference in radial band [k/10, (k+1)/10) of the normalised
               frequency radius used by the condition filter

Both come with a floor: the same statistic between gen_i and the reference of a
different image (a fixed derangement), because aligned FFHQ faces agree at low
frequency whatever the model does. Generation directories must hold PNGs named
exactly like the reference set; a mismatch aborts that directory.
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
from PIL import Image
from datalibs.frequency import high_pass_torch, rgb_to_gray_torch, radial_mask_torch

ap = argparse.ArgumentParser()
ap.add_argument("--dirs", nargs="+", required=True, help="results/<run> directories")
ap.add_argument("--ref", default="results/E10b/reference")
ap.add_argument("--out", default="results/E19_fixed_instrument/results.json")
ap.add_argument("--n", type=int, default=None, help="limit images (smoke test only)")
ap.add_argument("--batch", type=int, default=32)
a = ap.parse_args()

RM = [0.0, 0.05, 0.1, 0.3]   # 0.0 = grayscale identity, the SC_vs_drawing instrument
NB = 10
dev = "cuda"
ref_names = sorted(f for f in os.listdir(a.ref) if f.endswith(".png"))
if a.n: ref_names = ref_names[:a.n]
N = len(ref_names)
rng = np.random.default_rng(0)
perm = rng.permutation(N)
while np.any(perm == np.arange(N)):          # derangement: never pair an image with itself
    perm = rng.permutation(N)

def load(d, names):
    x = np.stack([np.asarray(Image.open(os.path.join(d, f)).convert("RGB")) for f in names])
    return rgb_to_gray_torch(torch.from_numpy(x).permute(0, 3, 1, 2).float().div(255).to(dev))

def pearson(u, v):
    u = u.flatten(1); v = v.flatten(1)
    u = u - u.mean(1, keepdim=True); v = v - v.mean(1, keepdim=True)
    return ((u * v).sum(1) / (u.norm(dim=1) * v.norm(dim=1)).clamp_min(1e-8)).cpu().numpy()

H = W = None
def bands(x):
    """(B,1,H,W) -> list of NB band-pass real images."""
    global H, W
    b, _, H, W = x.shape
    f = torch.fft.fftshift(torch.fft.fft2(x), dim=(-2, -1))
    out = []
    for k in range(NB):
        lo = radial_mask_torch(H, W, k / NB, dev, "low_pass")
        hi = radial_mask_torch(H, W, (k + 1) / NB, dev, "low_pass")
        m = (hi - lo) if k > 0 else hi          # band 0 includes DC
        out.append(torch.fft.ifft2(torch.fft.ifftshift(f * m, dim=(-2, -1))).real)
    return out

def stats(v):
    v = np.asarray(v, dtype=np.float64)
    return {"mean": float(v.mean()), "sem": float(v.std() / np.sqrt(len(v))), "n": int(len(v))}

import hashlib
def _md5(path):
    return hashlib.md5(open(path, "rb").read()).hexdigest()
_REF_MD5 = {f: _md5(os.path.join(a.ref, f)) for f in ref_names}
_checked = {}
def same_reference(d):
    """A generation is only comparable if it was made from the same test images."""
    if d not in _checked:
        names = sorted(f for f in os.listdir(d) if f.endswith(".png"))
        if a.n: names = names[:a.n]
        _checked[d] = names == ref_names and all(_md5(os.path.join(d, f)) == _REF_MD5[f] for f in names)
    return _checked[d]

ref_cache = {}
def ref_feats(i0, i1, idx):
    key = (i0, i1, idx is None)
    names = [ref_names[j] for j in (range(i0, i1) if idx is None else idx)]
    return load(a.ref, names)

results = json.load(open(a.out)) if os.path.exists(a.out) else {}
for d in a.dirs:
    for sub in sorted(os.listdir(d)):
        if not (sub.startswith("gen_r") and os.path.isdir(os.path.join(d, sub))):
            continue
        gdir = os.path.join(d, sub)
        own_ref = os.path.join(d, "reference")
        if os.path.isdir(own_ref) and not same_reference(own_ref):
            print(f"SKIP {gdir}: its reference/ differs from {a.ref}", flush=True)
            continue
        names = sorted(f for f in os.listdir(gdir) if f.endswith(".png"))
        if a.n: names = names[:a.n]
        if names != ref_names:
            print(f"SKIP {gdir}: {len(names)} files, names do not match the reference set", flush=True)
            continue
        acc = {f"sc_fix@{r}": [] for r in RM} | {f"sc_fix@{r}_floor": [] for r in RM}
        acc |= {f"band{k}": [] for k in range(NB)} | {f"band{k}_floor": [] for k in range(NB)}
        for i in range(0, N, a.batch):
            j = min(i + a.batch, N)
            g = load(gdir, names[i:j])
            rt = load(a.ref, ref_names[i:j])
            rs = load(a.ref, [ref_names[p] for p in perm[i:j]])
            for r in RM:
                hg, ht, hs = (high_pass_torch(x, r) for x in (g, rt, rs))
                acc[f"sc_fix@{r}"] += list(pearson(hg, ht))
                acc[f"sc_fix@{r}_floor"] += list(pearson(hg, hs))
            bg, bt, bs = bands(g), bands(rt), bands(rs)
            for k in range(NB):
                acc[f"band{k}"] += list(pearson(bg[k], bt[k]))
                acc[f"band{k}_floor"] += list(pearson(bg[k], bs[k]))
        key = f"{os.path.basename(os.path.normpath(d))}/{sub}"
        results[key] = {m: stats(v) for m, v in acc.items()}
        print(f"{key}: sc_fix@0.1 {results[key]['sc_fix@0.1']['mean']:.4f} "
              f"(floor {results[key]['sc_fix@0.1_floor']['mean']:.4f}) "
              f"band1 {results[key]['band1']['mean']:.3f} band5 {results[key]['band5']['mean']:.3f}", flush=True)
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        tmp = a.out + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(results, fh, indent=1)
        os.replace(tmp, a.out)                     # atomic: a crash never leaves half a file
print(f"wrote {a.out} ({len(results)} entries)")
