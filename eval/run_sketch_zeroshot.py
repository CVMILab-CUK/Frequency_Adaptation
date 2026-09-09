#!/usr/bin/env python3
"""E6 -- zero-shot on real hand-drawn sketches.

Real sketches carry no cutoff r: the model has to infer structure density from
the drawing itself, which is only possible because the condition is per-image
min-max normalised. There is no ground-truth photo for a drawing, so the
metrics are:
  SC     against the *input sketch* -- did the output follow what was drawn
  CLIP   against the prompt
  FID    against the training distribution (FFHQ), i.e. does it look like a face
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("HF_HOME", "/home/work/data/hf_cache")

import cv2, numpy as np, torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from PIL import Image

from datalibs.frequency import to_model_range, rgb_to_gray_torch, high_pass_torch
from eval.metrics import CLIPScore, structure_consistency, fid_from_dirs

ap = argparse.ArgumentParser()
ap.add_argument("-c", "--config", required=True)
ap.add_argument("--adapter", required=True)
ap.add_argument("--sketch_dir", default="/home/work/data/sketches/images")
ap.add_argument("--ref_dir", default=None, help="real photos for FID (defaults to the test reference set)")
ap.add_argument("--out", default="results/E6_sketch")
ap.add_argument("--n_images", type=int, default=200)
ap.add_argument("--steps", type=int, default=50)
ap.add_argument("--cfg", type=float, default=7.5)
ap.add_argument("--batch", type=int, default=8)
ap.add_argument("--seed", type=int, default=2026)
ap.add_argument("--prompt", default="a high-quality photo of a face")
a = ap.parse_args()

cfg = OmegaConf.load(a.config)
res = int(cfg.datasets.img_size)
os.makedirs(a.out, exist_ok=True)
gen_dir = os.path.join(a.out, "generated"); os.makedirs(gen_dir, exist_ok=True)
in_dir = os.path.join(a.out, "input"); os.makedirs(in_dir, exist_ok=True)

from trainer.fa_trainer import Trainer
tr = Trainer(a.config); tr.device = 0
tr.model_define(0)
tr.model.load_adapter(a.adapter)
tr.model.unet.to(dtype=tr.weight_dtype)
tr.model.unet.eval(); tr.model.vae.eval(); tr.model.text_encoder.eval()
pipe = tr.build_pipeline()
clip_m = CLIPScore(0)

files = sorted(f for f in os.listdir(a.sketch_dir)
               if f.lower().endswith((".png", ".jpg", ".jpeg")))[:a.n_images]
print(f"[e6] {len(files)} real sketches from {a.sketch_dir}", flush=True)

SC, CL = [], []
for s in range(0, len(files), a.batch):
    chunk = files[s:s + a.batch]
    ims = []
    for f in chunk:
        im = cv2.cvtColor(cv2.imread(os.path.join(a.sketch_dir, f)), cv2.COLOR_BGR2RGB)
        im = cv2.resize(im, (res, res), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        ims.append(im)
    x = torch.from_numpy(np.stack(ims)).permute(0, 3, 1, 2).cuda()
    # a drawing is already a structure map; collapse to gray and replicate, the
    # same shape the model was trained on, but with no filtering applied
    cond_unit = rgb_to_gray_torch(x).repeat(1, 3, 1, 1).clamp(0, 1)
    cond = to_model_range(cond_unit).to(dtype=tr.weight_dtype)

    g = torch.Generator(device="cpu").manual_seed(a.seed + s)
    out = pipe([a.prompt] * len(chunk), cond, height=res, width=res,
               num_inference_steps=a.steps, num_images_per_prompt=1,
               guidance_scale=a.cfg, generator=g, output_type="pt").images.float().clamp(0, 1)

    # SC against the drawing itself, at the lowest cutoff (the drawing IS the condition)
    SC.append(structure_consistency(out, cond_unit, 0.0))
    CL.append(clip_m.score(out, [a.prompt] * len(chunk)))
    for j in range(out.shape[0]):
        Image.fromarray((out[j] * 255).byte().cpu().numpy().transpose(1, 2, 0)
                        ).save(os.path.join(gen_dir, f"{s + j:04d}.png"))
        Image.fromarray((cond_unit[j] * 255).byte().cpu().numpy().transpose(1, 2, 0)
                        ).save(os.path.join(in_dir, f"{s + j:04d}.png"))
    print(f"[e6] {min(s + a.batch, len(files))}/{len(files)}", flush=True)

SC = np.concatenate(SC); CL = np.concatenate(CL)
fid = None
ref = a.ref_dir or "results/E10b/reference"
try:
    fid = fid_from_dirs(gen_dir, ref, device="cuda")
except Exception as e:
    print(f"[e6] FID not computed: {e}", flush=True)

def agg(x):
    return {"mean": float(x.mean()), "std": float(x.std()),
            "sem": float(x.std() / np.sqrt(len(x))), "n": int(len(x))}

res_json = {"provenance": {"adapter": os.path.abspath(a.adapter), "sketch_dir": a.sketch_dir,
                           "n_images": len(files), "steps": a.steps, "cfg_scale": a.cfg,
                           "seed": a.seed, "prompt": a.prompt, "fid_reference": ref,
                           "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")},
            "SC_vs_input_sketch": agg(SC), "CLIP": agg(CL), "FID_vs_FFHQ": fid}
with open(os.path.join(a.out, "results.json"), "w") as f:
    json.dump(res_json, f, indent=2)
print(f"\n[e6] SC vs sketch {SC.mean():.4f}  CLIP {CL.mean():.2f}  FID {fid}")
