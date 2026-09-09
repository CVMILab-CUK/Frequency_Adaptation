#!/usr/bin/env python3
"""E7 + paper figures: the dial, held at a fixed seed.

Panel 1  one image, seed fixed, r swept -> the continuous morph from
         colourisation to free generation. This is the paper's money figure.
Panel 2  the same sweep for several images, one row each.
Panel 3  real sketches: drawing next to what the model made of it.
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("HF_HOME", "/home/work/data/hf_cache")

import cv2, numpy as np, torch
from omegaconf import OmegaConf
from PIL import Image

from datalibs.frequency import make_condition_torch, to_model_range, rgb_to_gray_torch

ap = argparse.ArgumentParser()
ap.add_argument("-c", "--config", required=True)
ap.add_argument("--adapter", required=True)
ap.add_argument("--out", default="results/figures")
ap.add_argument("--cutoffs", default="0.0,0.05,0.1,0.2,0.3,0.5,0.7,1.0")
ap.add_argument("--n_rows", type=int, default=6)
ap.add_argument("--steps", type=int, default=50)
ap.add_argument("--cfg", type=float, default=7.5)
ap.add_argument("--seed", type=int, default=2026)
ap.add_argument("--sketch_dir", default="/home/work/data/sketches/images")
a = ap.parse_args()

cfg = OmegaConf.load(a.config); res = int(cfg.datasets.img_size)
os.makedirs(a.out, exist_ok=True)

from trainer.fa_trainer import Trainer
tr = Trainer(a.config); tr.device = 0
tr.model_define(0); tr.model.load_adapter(a.adapter)
tr.model.unet.to(dtype=tr.weight_dtype)
tr.model.unet.eval(); tr.model.vae.eval(); tr.model.text_encoder.eval()
tr.makeDatasets(cfg.datasets.data_path, cfg.datasets.img_path, frequency_rate=0.0,
                img_size=res, frequency_img=cfg.datasets.frequency_img, mean=0.5, std=0.5,
                caption_path=getattr(cfg.datasets, "caption_path", None), ddp=False)
pipe = tr.build_pipeline()
cuts = [float(x) for x in a.cutoffs.split(",")]

def strip(tiles, path):
    h = tiles[0].shape[0]
    canvas = np.concatenate([np.pad(t, ((0,0),(2,2),(0,0)), constant_values=255) for t in tiles], axis=1)
    Image.fromarray(canvas).save(path); print("wrote", path, flush=True)

def to_np(t):
    return (t.clamp(0,1)*255).byte().cpu().numpy().transpose(1,2,0)

rows = []
for i in range(a.n_rows):
    d = tr.test_dataset[i]
    gt = ((d["gt"] + 1.0) / 2.0).clamp(0, 1).unsqueeze(0).cuda()
    cap = d["caption"]
    tiles = [to_np(gt[0])]
    for c in cuts:
        cond_unit = make_condition_torch(gt, c, frequency_img=cfg.datasets.frequency_img)
        cond = to_model_range(cond_unit).to(dtype=tr.weight_dtype)
        g = torch.Generator(device="cpu").manual_seed(a.seed)   # SAME seed across r
        out = pipe([cap], cond, height=res, width=res, num_inference_steps=a.steps,
                   num_images_per_prompt=1, guidance_scale=a.cfg, generator=g,
                   output_type="pt").images.float().clamp(0,1)
        tiles.append(to_np(out[0]))
    rows.append(tiles)
    strip(tiles, os.path.join(a.out, f"dial_row{i}.png"))
    print(f"[fig] row {i+1}/{a.n_rows}", flush=True)

grid = np.concatenate([np.concatenate([np.pad(t,((2,2),(2,2),(0,0)),constant_values=255)
                                       for t in r], axis=1) for r in rows], axis=0)
Image.fromarray(grid).save(os.path.join(a.out, "dial_grid.png"))
print("wrote", os.path.join(a.out, "dial_grid.png"), flush=True)
print("columns: GT | " + " | ".join(f"r={c}" for c in cuts))

# real sketches next to their generations
if os.path.isdir(a.sketch_dir):
    fs = sorted(os.listdir(a.sketch_dir))[:a.n_rows]
    tiles = []
    for f in fs:
        im = cv2.cvtColor(cv2.imread(os.path.join(a.sketch_dir,f)), cv2.COLOR_BGR2RGB)
        im = cv2.resize(im,(res,res),interpolation=cv2.INTER_AREA).astype(np.float32)/255.
        x = torch.from_numpy(im).permute(2,0,1).unsqueeze(0).cuda()
        cond_unit = rgb_to_gray_torch(x).repeat(1,3,1,1).clamp(0,1)
        g = torch.Generator(device="cpu").manual_seed(a.seed)
        out = pipe(["a high-quality photo of a face"],
                   to_model_range(cond_unit).to(dtype=tr.weight_dtype),
                   height=res, width=res, num_inference_steps=a.steps,
                   num_images_per_prompt=1, guidance_scale=a.cfg, generator=g,
                   output_type="pt").images.float().clamp(0,1)
        tiles += [to_np(cond_unit[0]), to_np(out[0])]
    strip(tiles, os.path.join(a.out, "sketch_pairs.png"))
