#!/usr/bin/env python3
"""Evaluate pretrained ControlNet / T2I-Adapter baselines on the same protocol.

Two conditioning regimes, kept clearly separate because they are not equally
fair:

  --cond native   each baseline gets the condition it was trained on (Canny /
                  HED). This is the comparison to quote.
  --cond freq     each baseline gets *our* frequency condition. The baselines
                  were never trained on it, so this is a zero-shot transfer
                  result and is labelled as such -- it flatters us and must not
                  be presented as the headline.

Same test images, same captions, same seeds, same metrics as run_eval.py.
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("HF_HOME", "/home/work/data/hf_cache")

import cv2
import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image

from datalibs.frequency import make_condition_torch
from eval.metrics import (LPIPSMetric, CLIPScore, structure_consistency,
                          edge_f1, fid_from_dirs)

BASELINES = {
    # name: (repo, condition the checkpoint expects, adapter family)
    "controlnet_canny": ("lllyasviel/sd-controlnet-canny", "canny", "controlnet"),
    "controlnet_hed":   ("lllyasviel/sd-controlnet-hed", "canny", "controlnet"),  # HED detector optional
    # T2I-Adapter is the size-matched comparison: a small side network rather
    # than a duplicated encoder, which is the design our adapter belongs to.
    "t2iadapter_canny": ("TencentARC/t2iadapter_canny_sd15v2", "canny", "t2iadapter"),
    "t2iadapter_sketch":("TencentARC/t2iadapter_sketch_sd15v2", "canny", "t2iadapter"),
}


def canny_cond(gt_unit):
    out = []
    arr = (gt_unit.clamp(0, 1) * 255).byte().cpu().numpy().transpose(0, 2, 3, 1)
    for im in arr:
        e = cv2.Canny(cv2.cvtColor(im, cv2.COLOR_RGB2GRAY), 100, 200)
        out.append(np.stack([e] * 3, -1))
    return torch.from_numpy(np.stack(out)).permute(0, 3, 1, 2).float() / 255.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-c", "--config", default="./config/fa_sd15_ffhq512.yaml")
    p.add_argument("--baseline", default="controlnet_canny", choices=list(BASELINES))
    p.add_argument("--cond", default="native", choices=["native", "freq"])
    p.add_argument("--cutoffs", default="0.1,0.3,0.5",
                   help="only used when --cond freq")
    p.add_argument("--out", default="./results/baselines")
    p.add_argument("--n_images", type=int, default=500)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--cfg", type=float, default=7.5)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--seed", type=int, default=2026)
    a = p.parse_args()

    from diffusers import (StableDiffusionControlNetPipeline, ControlNetModel,
                           StableDiffusionAdapterPipeline, T2IAdapter)
    cfg = OmegaConf.load(a.config)
    device = "cuda"
    res = int(cfg.datasets.img_size)
    tag = f"{a.baseline}_{a.cond}"
    out_root = os.path.join(a.out, tag)
    os.makedirs(out_root, exist_ok=True)

    repo, _, family = BASELINES[a.baseline]
    if family == "controlnet":
        side = ControlNetModel.from_pretrained(repo, torch_dtype=torch.float16)
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            cfg.model.model_id, controlnet=side, torch_dtype=torch.float16,
            safety_checker=None, requires_safety_checker=False).to(device)
    else:
        side = T2IAdapter.from_pretrained(repo, torch_dtype=torch.float16)
        pipe = StableDiffusionAdapterPipeline.from_pretrained(
            cfg.model.model_id, adapter=side, torch_dtype=torch.float16,
            safety_checker=None, requires_safety_checker=False).to(device)
    pipe.set_progress_bar_config(disable=True)
    n_params = sum(q.numel() for q in side.parameters())
    print(f"[base] {a.baseline}: {n_params:,} side-network params ({family})", flush=True)

    from trainer.base_trainer import BaseTrainer
    bt = BaseTrainer(".", ".", batch_size=a.batch, num_workers=4)
    bt.makeDatasets(cfg.datasets.data_path, cfg.datasets.img_path,
                    frequency_rate=0.0, img_size=res,
                    frequency_img=cfg.datasets.frequency_img, mean=0.5, std=0.5,
                    caption_path=getattr(cfg.datasets, "caption_path", None), ddp=False)

    n = min(a.n_images, len(bt.test_dataset))
    gts, caps = [], []
    ref_dir = os.path.join(a.out, "reference"); os.makedirs(ref_dir, exist_ok=True)
    for i in range(n):
        d = bt.test_dataset[i]
        g = ((d["gt"] + 1.0) / 2.0).clamp(0, 1)
        gts.append(g); caps.append(d["caption"])
        pth = os.path.join(ref_dir, f"{i:05d}.png")
        if not os.path.exists(pth):
            Image.fromarray((g * 255).byte().numpy().transpose(1, 2, 0)).save(pth)
    gts = torch.stack(gts)

    lp, cl = LPIPSMetric(0), CLIPScore(0)
    settings = ([("native", None)] if a.cond == "native"
                else [("freq", float(x)) for x in a.cutoffs.split(",")])

    results = {"provenance": {
        "baseline": a.baseline, "repo": repo, "cond_regime": a.cond,
        "family": family, "side_params": int(n_params),
        "controlnet_params": int(n_params), "n_images": n, "steps": a.steps,
        "cfg_scale": a.cfg, "seed": a.seed, "resolution": res,
        "gpu": torch.cuda.get_device_name(0),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "note": ("baselines run on their native condition"
                 if a.cond == "native" else
                 "ZERO-SHOT transfer: baselines were not trained on this condition"),
    }, "per_setting": {}}

    for name, cut in settings:
        key = name if cut is None else f"freq_r{cut}"
        gen_dir = os.path.join(out_root, key); os.makedirs(gen_dir, exist_ok=True)
        SC, EF, LP, CL = [], [], [], []
        for s in range(0, n, a.batch):
            gt_b = gts[s:s + a.batch].to(device)
            cap_b = caps[s:s + a.batch]
            if cut is None:
                cond_unit = canny_cond(gt_b).to(device)
            else:
                cond_unit = make_condition_torch(gt_b, cut,
                    frequency_img=cfg.datasets.frequency_img, mode=cfg.datasets.frequency_mode)
            g = torch.Generator(device="cpu").manual_seed(a.seed + s)
            # T2I-Adapter canny/sketch checkpoints read one grayscale channel
            # (pixel-unshuffle 8x -> 64 channels); ControlNet reads three.
            cond_in = cond_unit[:, :1] if family == "t2iadapter" else cond_unit
            imgs = pipe(cap_b, image=cond_in.to(torch.float16),
                        num_inference_steps=a.steps, guidance_scale=a.cfg,
                        generator=g, output_type="pt").images
            # StableDiffusionAdapterPipeline in diffusers 0.32 ignores
            # output_type="pt" and returns an (N, H, W, C) numpy array in [0, 1].
            if isinstance(imgs, np.ndarray):
                imgs = torch.from_numpy(imgs).permute(0, 3, 1, 2)
            imgs = imgs.to(device).float().clamp(0, 1)
            if cut is not None:
                SC.append(structure_consistency(imgs, cond_unit, cut))
            EF.append(edge_f1(imgs, gt_b)); LP.append(lp.distance(imgs, gt_b))
            CL.append(cl.score(imgs, cap_b))
            for j in range(imgs.shape[0]):
                Image.fromarray((imgs[j] * 255).byte().cpu().numpy().transpose(1, 2, 0)
                                ).save(os.path.join(gen_dir, f"{s + j:05d}.png"))
            print(f"[base] {key} {min(s + a.batch, n)}/{n}", flush=True)

        def agg(x):
            x = np.concatenate(x)
            return {"mean": float(x.mean()), "std": float(x.std()),
                    "sem": float(x.std() / np.sqrt(len(x))), "n": int(len(x))}

        fid = None
        try:
            fid = fid_from_dirs(gen_dir, ref_dir, device="cuda")
        except Exception as e:
            print(f"[base] FID not computed for {key}: {e}", flush=True)

        results["per_setting"][key] = {
            "SC": agg(SC) if SC else None, "EdgeF1": agg(EF),
            "LPIPS": agg(LP), "CLIP": agg(CL), "FID": fid,
        }
        with open(os.path.join(out_root, "results.json"), "w") as f:
            json.dump(results, f, indent=2)
        print(f"[base] {key} done -> {results['per_setting'][key]}", flush=True)


if __name__ == "__main__":
    main()
