#!/usr/bin/env python3
"""Measure real throughput and peak memory per batch size, so the training
config is set from measurement rather than from a guess."""
import argparse, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("HF_HOME", "/home/work/data/hf_cache")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from omegaconf import OmegaConf

ap = argparse.ArgumentParser()
ap.add_argument("-c", "--config", default="./config/smoke.yaml")
ap.add_argument("--batches", default="16,32,48,64")
ap.add_argument("--steps", type=int, default=12)
a = ap.parse_args()

from trainer.fa_trainer import Trainer
from diffusers.training_utils import compute_snr

print(f"{'batch':>6} {'img/s':>9} {'peak GB':>9} {'status':>10}")
print("-" * 38)
for bs in [int(x) for x in a.batches.split(",")]:
    cfg = OmegaConf.load(a.config)
    cfg.trainer.batch_size = bs
    tmp = f"/tmp/bench_{bs}.yaml"; OmegaConf.save(cfg, tmp)
    tr = Trainer(tmp); tr.device = 0
    try:
        tr.initialize(0, 1)
        tr.model_define(0)
        tr.makeDatasets(cfg.datasets.data_path, cfg.datasets.img_path,
                        frequency_rate="random", img_size=int(cfg.datasets.img_size),
                        frequency_img=cfg.datasets.frequency_img, mean=0.5, std=0.5,
                        caption_path=getattr(cfg.datasets, "caption_path", None), ddp=False)
        accel = tr.model.accelerator
        tr.model.unet, tr.model.optimizer, tr.loader_train = accel.prepare(
            tr.model.unet, tr.model.optimizer, tr.loader_train)
        torch.cuda.reset_peak_memory_stats()
        it = iter(tr.loader_train)
        n_img, t0 = 0, None
        for i in range(a.steps):
            data = next(it)
            img = data["gt"].to(0, dtype=tr.weight_dtype)
            fil = data["filtered_image"].to(0, dtype=tr.weight_dtype)
            with torch.no_grad():
                lat = tr.model.vae.encode(img).latent_dist.sample() * tr.model.vae.config.scaling_factor
                cond = tr.encode_condition(fil)
                ehs = tr.encode_text(data["caption"])
            noise = torch.randn_like(lat)
            t = torch.randint(0, 1000, (lat.shape[0],), device=lat.device).long()
            noisy = tr.model.noise_scheduler.add_noise(lat, noise, t)
            pred = tr.model.unet(noisy, t, ehs, return_dict=False,
                                 cross_attention_kwargs={"ip_hidden_states": cond})[0]
            loss = torch.nn.functional.mse_loss(pred.float(), noise.float())
            accel.backward(loss)
            tr.model.optimizer.step(); tr.model.optimizer.zero_grad(set_to_none=True)
            if i == 2:  # skip warmup
                torch.cuda.synchronize(); t0 = time.time(); n_img = 0
            elif t0 is not None:
                n_img += lat.shape[0]
        torch.cuda.synchronize()
        ips = n_img / (time.time() - t0)
        peak = torch.cuda.max_memory_allocated() / 1e9
        print(f"{bs:>6} {ips:>9.1f} {peak:>9.1f} {'ok':>10}", flush=True)
    except torch.cuda.OutOfMemoryError:
        print(f"{bs:>6} {'-':>9} {'-':>9} {'OOM':>10}", flush=True)
    except Exception as e:
        print(f"{bs:>6} {'-':>9} {'-':>9} {type(e).__name__:>10}: {e}", flush=True)
    finally:
        for attr in ("model", "loader_train"):
            if hasattr(tr, attr): delattr(tr, attr)
        del tr; import gc; gc.collect(); torch.cuda.empty_cache()
