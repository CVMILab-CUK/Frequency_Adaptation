#!/usr/bin/env python3
"""E23 -- does the face layout really hold while appearance drifts?

The paper says it in the abstract, the introduction and the conclusion: as the
cutoff grows the face layout stays put while background, clothing and colour
leave the reference. Limitations admits the evidence is the samples in Figures 3
and 5, because neither structure consistency nor Edge-F1 separates the two --
both fall together across the same range.

This script measures the two sides apart, on the generations the paper already
saved, so it adds no inference and no training:

  layout, from the geometry of the face
    face_found     fraction of generations with a detectable frontal face
    face_iou       IoU of the largest face box, generation against reference
    face_center    centre displacement, in units of the reference face width
    face_scale     |log(w_gen / w_ref)|, so 0 is the same size

  appearance, from colour and from what CLIP sees
    colour_hist    Lab (a,b) histogram intersection, 1.0 is the same palette
    chroma_shift   L2 distance between mean (a,b)
    clip_img       CLIP ViT-B/32 image-image cosine (the paper's CLIP model)

The claim predicts the layout columns stay flat over r in [0.05, 0.3] while the
appearance columns fall. The detector is the frontal-face cascade that ships
with OpenCV: it is coarse, but it needs no download and it is reproducible from
a clean checkout. It finds a face in 39 of the first 40 references, so the
geometry columns are read over the images where both sides detect, and
`face_found` reports how often the generation has a face at all.
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--run", default="results/E10b", help="run directory with reference/ and gen_r*/")
ap.add_argument("--out", default="results/E23_layout_appearance/results.json")
ap.add_argument("--cutoffs", default="0.0,0.05,0.1,0.2,0.3,0.5,0.7,1.0")
ap.add_argument("--n", type=int, default=500)
ap.add_argument("--device", default="cuda")
a = ap.parse_args()

CASCADE = os.path.join(os.path.dirname(cv2.__file__), "data", "haarcascade_frontalface_default.xml")
detector = cv2.CascadeClassifier(CASCADE)

def largest_face(gray):
    """(x, y, w, h) of the largest frontal face, or None."""
    boxes = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))
    if len(boxes) == 0:
        return None
    return max(boxes, key=lambda b: b[2] * b[3])

def iou(p, q):
    px, py, pw, ph = p; qx, qy, qw, qh = q
    x0, y0 = max(px, qx), max(py, qy)
    x1, y1 = min(px + pw, qx + qw), min(py + ph, qy + qh)
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    return inter / (pw * ph + qw * qh - inter)

def lab_ab_hist(bgr, bins=32):
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    h = cv2.calcHist([lab], [1, 2], None, [bins, bins], [0, 256, 0, 256])
    return h / h.sum()

class Clip:
    """The paper's CLIP: open_clip ViT-B-32, OpenAI weights (eval/metrics.py:107)."""
    def __init__(self, device):
        import open_clip, torch
        self.torch = torch
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="openai", device=device)
        self.model.eval()
        self.device = device

    def embed(self, bgr):
        from PIL import Image
        img = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        x = self.preprocess(img).unsqueeze(0).to(self.device)
        with self.torch.no_grad():
            f = self.model.encode_image(x)
        return (f / f.norm(dim=-1, keepdim=True)).squeeze(0).float().cpu().numpy()

def stat(vals):
    v = np.asarray([x for x in vals if x is not None], dtype=np.float64)
    if v.size == 0:
        return None
    return {"mean": float(v.mean()),
            "sem": float(v.std(ddof=1) / np.sqrt(v.size)) if v.size > 1 else 0.0,
            "n": int(v.size)}

ref_dir = os.path.join(a.run, "reference")
names = sorted(os.listdir(ref_dir))[:a.n]
clip = Clip(a.device)

# The reference side is read once and reused for every cutoff.
ref_cache = {}
for nm in names:
    bgr = cv2.imread(os.path.join(ref_dir, nm))
    ref_cache[nm] = (largest_face(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)),
                     lab_ab_hist(bgr),
                     cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)[:, :, 1:].reshape(-1, 2).mean(0),
                     clip.embed(bgr))

out = {"provenance": {
    "run": a.run, "ref": ref_dir, "n_images": len(names),
    "detector": os.path.basename(CASCADE), "opencv": cv2.__version__,
    "clip": "open_clip ViT-B-32 / openai",
    "definition": __doc__.strip(),
    "note": "re-scores generations already saved by eval/run_eval.py; no new inference"}}

for c in [x.strip() for x in a.cutoffs.split(",")]:
    gen_dir = os.path.join(a.run, f"gen_r{c}")
    if not os.path.isdir(gen_dir):
        print(f"  skip r={c}: {gen_dir} missing", flush=True)
        continue
    found, ious, centers, scales, hists, chromas, clips = [], [], [], [], [], [], []
    for nm in names:
        p = os.path.join(gen_dir, nm)
        if not os.path.exists(p):
            continue
        bgr = cv2.imread(p)
        rb, rh, rc, rf = ref_cache[nm]
        gb = largest_face(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY))
        found.append(1.0 if gb is not None else 0.0)
        if rb is not None and gb is not None:
            ious.append(iou(rb, gb))
            rcx, rcy = rb[0] + rb[2] / 2, rb[1] + rb[3] / 2
            gcx, gcy = gb[0] + gb[2] / 2, gb[1] + gb[3] / 2
            centers.append(float(np.hypot(gcx - rcx, gcy - rcy) / rb[2]))
            scales.append(float(abs(np.log(gb[2] / rb[2]))))
        gh = lab_ab_hist(bgr)
        hists.append(float(np.minimum(rh, gh).sum()))
        gc = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)[:, :, 1:].reshape(-1, 2).mean(0)
        chromas.append(float(np.linalg.norm(gc - rc)))
        clips.append(float(np.dot(rf, clip.embed(bgr))))
    for key, vals in [("face_found", found), ("face_iou", ious), ("face_center", centers),
                      ("face_scale", scales), ("colour_hist", hists),
                      ("chroma_shift", chromas), ("clip_img", clips)]:
        s = stat(vals)
        if s is not None:
            out[f"{key}@{c}"] = s
    print(f"  r={c}: face_found {np.mean(found):.3f}  iou {np.mean(ious) if ious else float('nan'):.3f} "
          f"colour {np.mean(hists):.3f}  clip {np.mean(clips):.3f}", flush=True)

os.makedirs(os.path.dirname(a.out), exist_ok=True)
json.dump(out, open(a.out, "w"), indent=1)
print(f"wrote {a.out}")
