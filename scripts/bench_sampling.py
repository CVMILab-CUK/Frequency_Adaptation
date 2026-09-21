#!/usr/bin/env python3
"""Measure sampling wall-clock per image, so Table 14's Compute row is measured.

Mirrors eval/run_eval.py's pipeline exactly -- same trainer, same condition
builder, same sampler settings -- and times the generation call only, after a
warm-up pass. Reports the adapter on and the adapter off (both injection paths
scaled to 0), which is the cost the adapter adds over the frozen backbone.
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("HF_HOME", "/home/work/data/hf_cache")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from omegaconf import OmegaConf
from datalibs.frequency import make_condition_torch, to_model_range

ap = argparse.ArgumentParser()
ap.add_argument("-c", "--config", default="./config/generated/E10b_skip_inject.yaml")
ap.add_argument("--adapter", default="./ckpt_dir/E10b_skip_inject/final/adapter.safetensors")
ap.add_argument("--cutoff", type=float, default=0.1)
ap.add_argument("--steps", type=int, default=50)
ap.add_argument("--cfg", type=float, default=7.5)
ap.add_argument("--batch", type=int, default=1)
ap.add_argument("--n", type=int, default=8, help="timed images per setting")
ap.add_argument("--warmup", type=int, default=2)
ap.add_argument("--seed", type=int, default=2026)
ap.add_argument("--out", default="results/E23_sampling_cost/results.json")
a = ap.parse_args()

from trainer.fa_trainer import Trainer
cfg = OmegaConf.load(a.config)
res = int(cfg.datasets.img_size)
tr = Trainer(a.config); tr.device = 0
tr.model_define(0)
tr.model.load_adapter(a.adapter)
tr.model.unet.to(dtype=tr.weight_dtype)
tr.model.unet.eval(); tr.model.vae.eval(); tr.model.text_encoder.eval()
tr.makeDatasets(cfg.datasets.data_path, cfg.datasets.img_path,
                frequency_rate=0.0, mode=cfg.datasets.frequency_mode,
                img_size=res, source_size=getattr(cfg.datasets, "source_size", None),
                frequency_img=cfg.datasets.frequency_img, mean=0.5, std=0.5,
                caption_path=getattr(cfg.datasets, "caption_path", None), ddp=False)
pipe = tr.build_pipeline()
device = "cuda"

gts, caps = [], []
for i in range(a.n + a.warmup):
    d = tr.test_dataset[i]
    gts.append(((d["gt"] + 1.0) / 2.0).clamp(0, 1)); caps.append(d["caption"])
gts = torch.stack(gts)

from models.attention_processor import StandAloneAttnProcessor as _SA
def set_scale(v):
    n = 0
    for pr in tr.model.unet.attn_processors.values():
        if isinstance(pr, _SA): pr.scale = float(v); n += 1
    inj = getattr(tr.model, "skip_injector", None)
    if inj is not None: inj.scale = float(v)
    return n

def run_one(i):
    gt_b = gts[i:i + a.batch].to(device)
    cond_unit = make_condition_torch(gt_b, a.cutoff,
        frequency_img=cfg.datasets.frequency_img, mode=cfg.datasets.frequency_mode)
    cond = to_model_range(cond_unit).to(dtype=tr.weight_dtype)
    g = torch.Generator(device="cpu").manual_seed(a.seed + i)
    pipe(caps[i:i + a.batch], cond, height=res, width=res,
         num_inference_steps=a.steps, num_images_per_prompt=1,
         guidance_scale=a.cfg, generator=g, output_type="pt")

out = {"provenance": {
        "adapter": os.path.abspath(a.adapter), "config": os.path.abspath(a.config),
        "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
        "sampler": "DDIM", "steps": a.steps, "cfg_scale": a.cfg, "batch": a.batch,
        "resolution": res, "cutoff": a.cutoff, "n_timed": a.n, "warmup": a.warmup,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}}

for label, scale in (("adapter_on", 1.0), ("adapter_off", 0.0)):
    sites = set_scale(scale)
    for i in range(a.warmup): run_one(i)
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    ts = []
    for i in range(a.warmup, a.warmup + a.n):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        run_one(i)
        torch.cuda.synchronize(); ts.append(time.perf_counter() - t0)
    ts_img = [t / a.batch for t in ts]
    out[label] = {"scale": scale, "attn_sites": sites,
                  "seconds_per_image": {"mean": sum(ts_img) / len(ts_img),
                                        "min": min(ts_img), "max": max(ts_img)},
                  "peak_gpu_gb": torch.cuda.max_memory_allocated() / 1e9}
    print(label, out[label], flush=True)

os.makedirs(os.path.dirname(a.out), exist_ok=True)
json.dump(out, open(a.out, "w"), indent=1)
print("wrote", a.out)
