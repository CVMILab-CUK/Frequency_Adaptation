#!/usr/bin/env python3
"""D-17 -- what the dial means for a hand drawing.

The manuscript sweeps r over conditions extracted from photographs, but a
drawing arrives with no r attached, so the dial has no defined meaning there.
This fixes the definition an artist can actually act on: the artist filters
their own drawing at cutoff r and hands the model the result, exactly the
operation the training conditions went through.

Two scores per setting, because they answer different questions:
  SC_vs_filtered  did the model follow the condition it was handed
  SC_vs_drawing   did the output still follow the artist's original drawing

The second is what the artist cares about, and it is the one that should fall
as r rises if the dial does on drawings what it does on photographs.
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("HF_HOME", "/home/work/data/hf_cache")

import numpy as np, torch
import cv2
from omegaconf import OmegaConf
from PIL import Image

from datalibs.frequency import to_model_range, rgb_to_gray_torch, high_pass_torch
from eval.metrics import CLIPScore, structure_consistency, fid_from_dirs

ap = argparse.ArgumentParser()
ap.add_argument("-c", "--config", required=True)
ap.add_argument("--adapter", required=True)
ap.add_argument("--sketch_dir", default="/home/work/data/sketches/face")
ap.add_argument("--ref_dir", default="results/E10b/reference")
ap.add_argument("--out", default="results/E17_sketch_dial")
ap.add_argument("--cutoffs", default="0.0,0.05,0.1,0.2,0.3,0.5,0.7,1.0")
ap.add_argument("--n_images", type=int, default=40)
ap.add_argument("--steps", type=int, default=50)
ap.add_argument("--cfg", type=float, default=7.5)
ap.add_argument("--batch", type=int, default=8)
ap.add_argument("--seed", type=int, default=2026)
ap.add_argument("--prompt", default="a high-quality photo of a face")
a = ap.parse_args()

cfg = OmegaConf.load(a.config)
res = int(cfg.datasets.img_size)
cuts = [float(x) for x in a.cutoffs.split(",")]
os.makedirs(a.out, exist_ok=True)

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
print(f"[e17] {len(files)} drawings from {a.sketch_dir}, "
      f"{len(cuts)} cutoffs", flush=True)


def load(chunk):
    ims = []
    for f in chunk:
        im = cv2.cvtColor(cv2.imread(os.path.join(a.sketch_dir, f)), cv2.COLOR_BGR2RGB)
        im = cv2.resize(im, (res, res), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        ims.append(im)
    return torch.from_numpy(np.stack(ims)).permute(0, 3, 1, 2).cuda()


def agg(x):
    return {"mean": float(x.mean()), "std": float(x.std()),
            "sem": float(x.std() / np.sqrt(len(x))), "n": int(len(x))}


results = {"provenance": {
    "adapter": os.path.abspath(a.adapter), "sketch_dir": a.sketch_dir,
    "n_images": len(files), "steps": a.steps, "cfg_scale": a.cfg,
    "seed": a.seed, "prompt": a.prompt, "cutoffs": cuts,
    "definition": "the artist high-pass filters their own drawing at cutoff r; "
                  "that filtered drawing is the condition",
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}, "per_cutoff": {}}

for cut in cuts:
    gen_dir = os.path.join(a.out, f"r{cut}"); os.makedirs(gen_dir, exist_ok=True)
    SCf, SCd, CL = [], [], []
    for s in range(0, len(files), a.batch):
        chunk = files[s:s + a.batch]
        x = load(chunk)
        drawing = rgb_to_gray_torch(x).repeat(1, 3, 1, 1).clamp(0, 1)
        # the drawing IS a structure map, so it is filtered directly rather than
        # re-derived from a photograph -- the same chain, one step shorter
        if cut <= 0.0:
            cond_unit = drawing
        else:
            f_ = high_pass_torch(rgb_to_gray_torch(x), cut)
            cond_unit = f_.repeat(1, 3, 1, 1).clamp(0, 1) if f_.shape[1] == 1 else f_.clamp(0, 1)
        cond = to_model_range(cond_unit).to(dtype=tr.weight_dtype)

        g = torch.Generator(device="cpu").manual_seed(a.seed + s)
        out = pipe([a.prompt] * len(chunk), cond, height=res, width=res,
                   num_inference_steps=a.steps, num_images_per_prompt=1,
                   guidance_scale=a.cfg, generator=g, output_type="pt").images.float().clamp(0, 1)

        SCf.append(structure_consistency(out, cond_unit, 0.0))
        SCd.append(structure_consistency(out, drawing, 0.0))
        CL.append(clip_m.score(out, [a.prompt] * len(chunk)))
        for j in range(out.shape[0]):
            Image.fromarray((out[j] * 255).byte().cpu().numpy().transpose(1, 2, 0)
                            ).save(os.path.join(gen_dir, f"{s + j:04d}.png"))
        print(f"[e17] r={cut} {min(s + a.batch, len(files))}/{len(files)}", flush=True)

    fid = None
    try:
        fid = fid_from_dirs(gen_dir, a.ref_dir, device="cuda")
    except Exception as e:
        print(f"[e17] FID not computed at r={cut}: {e}", flush=True)

    results["per_cutoff"][str(cut)] = {
        "SC_vs_filtered": agg(np.concatenate(SCf)),
        "SC_vs_drawing":  agg(np.concatenate(SCd)),
        "CLIP":           agg(np.concatenate(CL)),
        "FID":            fid}
    with open(os.path.join(a.out, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"[e17] r={cut} done: {results['per_cutoff'][str(cut)]}", flush=True)

print("[e17] complete", flush=True)
