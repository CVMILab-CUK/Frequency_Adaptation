#!/usr/bin/env python3
"""E23b -- E23 again, paired and with a chance floor under every column.

Two things were wrong with the first pass and both cut the same way.

1. The cutoffs are scored on the same 500 images with the same reference and
   the same seed, so the change from one cutoff to another is a *paired*
   quantity. The first pass combined two independent SEMs, which is the
   unpaired two-sample error and is far too wide. Paired differences are
   computed here per image.

2. Only the colour column was given a derangement floor. FFHQ is pose-aligned,
   so two unrelated faces already land in nearly the same box: a raw face-IoU
   near 0.87 mostly measures the dataset's alignment, not preserved layout.
   Every column now gets a floor -- the same generation scored against a
   *different* reference -- and the informative quantity is the lift over it.

The floor is averaged over K derangements rather than one, because a single
permutation moves the floor by more than the effect being measured.

Writes per-image vectors to an .npz next to the JSON so any later statistic can
be recomputed without re-reading 66 GB of PNGs.
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--run", default="results/E10b")
ap.add_argument("--out", default="results/E23b_layout_appearance_paired")
ap.add_argument("--cutoffs", default="0.0,0.05,0.1,0.2,0.3,0.5,0.7,1.0")
ap.add_argument("--n", type=int, default=500)
ap.add_argument("--derangements", type=int, default=20)
ap.add_argument("--seed", type=int, default=2026)
ap.add_argument("--device", default="cuda")
a = ap.parse_args()

CASCADE = os.path.join(os.path.dirname(cv2.__file__), "data", "haarcascade_frontalface_default.xml")
detector = cv2.CascadeClassifier(CASCADE)

def largest_face(gray):
    b = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))
    return None if len(b) == 0 else max(b, key=lambda x: x[2] * x[3])

def iou(p, q):
    px, py, pw, ph = p; qx, qy, qw, qh = q
    x0, y0 = max(px, qx), max(py, qy)
    x1, y1 = min(px + pw, qx + qw), min(py + ph, qy + qh)
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    return inter / (pw * ph + qw * qh - inter)

def centre(p, q):
    pcx, pcy = p[0] + p[2] / 2, p[1] + p[3] / 2
    qcx, qcy = q[0] + q[2] / 2, q[1] + q[3] / 2
    return float(np.hypot(qcx - pcx, qcy - pcy) / p[2])

def ab_hist(bgr, bins=32):
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    h = cv2.calcHist([lab], [1, 2], None, [bins, bins], [0, 256, 0, 256])
    return h / h.sum()

class Clip:
    def __init__(self, device):
        import open_clip, torch
        self.torch = torch
        self.model, _, self.pre = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="openai", device=device)
        self.model.eval(); self.device = device
    def embed(self, bgr):
        from PIL import Image
        x = self.pre(Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))).unsqueeze(0).to(self.device)
        with self.torch.no_grad():
            f = self.model.encode_image(x)
        return (f / f.norm(dim=-1, keepdim=True)).squeeze(0).float().cpu().numpy()

names = sorted(os.listdir(os.path.join(a.run, "reference")))[:a.n]
N = len(names)
clip = Clip(a.device)

print(f"caching {N} references", flush=True)
ref_box, ref_hist, ref_clip = [], [], []
for nm in names:
    bgr = cv2.imread(os.path.join(a.run, "reference", nm))
    ref_box.append(largest_face(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)))
    ref_hist.append(ab_hist(bgr)); ref_clip.append(clip.embed(bgr))
ref_clip = np.stack(ref_clip)

rng = np.random.default_rng(a.seed)
perms = []
while len(perms) < a.derangements:
    p = rng.permutation(N)
    if not np.any(p == np.arange(N)):
        perms.append(p)

cuts = [x.strip() for x in a.cutoffs.split(",")]
store, summary = {}, {"provenance": {
    "run": a.run, "n_images": N, "derangements": a.derangements, "seed": a.seed,
    "detector": os.path.basename(CASCADE), "opencv": cv2.__version__,
    "clip": "open_clip ViT-B-32 / openai",
    "definition": __doc__.strip(),
    "note": "paired statistics; every column carries a derangement floor"}}

for c in cuts:
    gd = os.path.join(a.run, f"gen_r{c}")
    if not os.path.isdir(gd):
        continue
    gen_box, gen_hist, gen_clip = [], [], []
    for nm in names:
        bgr = cv2.imread(os.path.join(gd, nm))
        gen_box.append(largest_face(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)))
        gen_hist.append(ab_hist(bgr)); gen_clip.append(clip.embed(bgr))
    gen_clip = np.stack(gen_clip)

    found = np.array([b is not None for b in gen_box], dtype=np.float64)
    ok = np.array([(ref_box[i] is not None) and (gen_box[i] is not None) for i in range(N)])

    m_iou = np.full(N, np.nan); m_ctr = np.full(N, np.nan)
    for i in range(N):
        if ok[i]:
            m_iou[i] = iou(ref_box[i], gen_box[i]); m_ctr[i] = centre(ref_box[i], gen_box[i])
    m_col = np.array([np.minimum(ref_hist[i], gen_hist[i]).sum() for i in range(N)])
    m_clip = np.einsum("ij,ij->i", ref_clip, gen_clip)

    f_iou = np.zeros((a.derangements, N)); f_ctr = np.zeros((a.derangements, N))
    f_col = np.zeros((a.derangements, N)); f_clip = np.zeros((a.derangements, N))
    f_ok = np.zeros((a.derangements, N), dtype=bool)
    for k, p in enumerate(perms):
        for i in range(N):
            j = p[i]                                   # generation i vs reference j
            if ref_box[j] is not None and gen_box[i] is not None:
                f_iou[k, i] = iou(ref_box[j], gen_box[i])
                f_ctr[k, i] = centre(ref_box[j], gen_box[i]); f_ok[k, i] = True
            f_col[k, i] = np.minimum(ref_hist[j], gen_hist[i]).sum()
        f_clip[k] = np.einsum("ij,ij->i", ref_clip[p], gen_clip)

    store[f"matched_iou@{c}"] = m_iou; store[f"matched_centre@{c}"] = m_ctr
    store[f"matched_colour@{c}"] = m_col; store[f"matched_clip@{c}"] = m_clip
    store[f"found@{c}"] = found; store[f"ok@{c}"] = ok
    store[f"floor_iou@{c}"] = np.where(f_ok, f_iou, np.nan)
    store[f"floor_centre@{c}"] = np.where(f_ok, f_ctr, np.nan)
    store[f"floor_colour@{c}"] = f_col; store[f"floor_clip@{c}"] = f_clip

    def s(v):
        v = v[~np.isnan(v)]
        return {"mean": float(v.mean()), "sem": float(v.std(ddof=1) / np.sqrt(v.size)), "n": int(v.size)}
    fl_iou = np.nanmean(np.where(f_ok, f_iou, np.nan), axis=0)
    fl_col = f_col.mean(axis=0); fl_clip = f_clip.mean(axis=0)
    summary[f"iou@{c}"]  = {"matched": s(m_iou),  "floor": s(fl_iou),  "lift": s(m_iou - fl_iou)}
    summary[f"colour@{c}"] = {"matched": s(m_col), "floor": s(fl_col), "lift": s(m_col - fl_col)}
    summary[f"clip@{c}"] = {"matched": s(m_clip), "floor": s(fl_clip), "lift": s(m_clip - fl_clip)}
    summary[f"found@{c}"] = s(found)
    print(f"  r={c}: IoU {s(m_iou)['mean']:.4f} (floor {s(fl_iou)['mean']:.4f}, lift {s(m_iou-fl_iou)['mean']:+.4f}) | "
          f"CLIP {s(m_clip)['mean']:.4f} (floor {s(fl_clip)['mean']:.4f}, lift {s(m_clip-fl_clip)['mean']:+.4f})", flush=True)

os.makedirs(a.out, exist_ok=True)
np.savez_compressed(os.path.join(a.out, "per_image.npz"), **store)
json.dump(summary, open(os.path.join(a.out, "results.json"), "w"), indent=1)
print(f"wrote {a.out}/results.json and per_image.npz")
