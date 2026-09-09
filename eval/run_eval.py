#!/usr/bin/env python3
"""Measure the structure-fidelity / realism / diversity curve of a trained adapter.

For every cutoff r the script generates images from the *same* test images with
the *same* seeds, then reports, per r:

    SC        structure consistency  (corr. between given and re-extracted condition)
    EdgeF1    Canny edge F1 vs the reference image
    LPIPS     perceptual distance to the reference image
    CLIP      image-text alignment against the image's own caption
    Diversity mean pairwise LPIPS over K samples of one condition
    FID       clean-FID of the generated set against the reference set

Provenance (checkpoint, seed, counts, sampler settings, git commit) is written
alongside the numbers. Any metric that could not be computed is recorded as
null -- never as a stand-in value.
"""
import argparse, json, os, subprocess, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image

from datalibs.frequency import make_condition_torch, to_model_range
from eval.metrics import (LPIPSMetric, CLIPScore, structure_consistency,
                          edge_f1, fid_from_dirs)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("-c", "--config", default="./config/fa_sd15_ffhq512.yaml")
    p.add_argument("--adapter", required=True, help="path to adapter.safetensors")
    p.add_argument("--out", default="./results/eval")
    p.add_argument("--cutoffs", default="0.0,0.05,0.1,0.2,0.3,0.5,0.7,1.0")
    p.add_argument("--n_images", type=int, default=500, help="test images per cutoff")
    p.add_argument("--n_div", type=int, default=50, help="images used for the diversity estimate")
    p.add_argument("--k_div", type=int, default=4, help="samples per image for diversity")
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--cfg", type=float, default=7.5)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--skip_fid", action="store_true")
    p.add_argument("--adapter_scale", type=float, default=None,
                   help="override the adapter's injection scale. 0.0 turns the "
                        "adapter off entirely, which is the control that says "
                        "whether a weak result means the adapter is inert or harmful.")
    p.add_argument("--skip_scale", type=float, default=None,
                   help="override the skip injector's scale independently, so the two "
                        "injection paths can be ablated apart. Defaults to --adapter_scale.")
    return p.parse_args()


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"],
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


def save_png(t, path):
    arr = (t.clamp(0, 1) * 255).byte().cpu().numpy().transpose(1, 2, 0)
    Image.fromarray(arr).save(path)


def main():
    a = parse_args()
    cfg = OmegaConf.load(a.config)
    device = 0
    os.makedirs(a.out, exist_ok=True)

    from trainer.fa_trainer import Trainer
    tr = Trainer(a.config)
    tr.device = device
    tr.model_define(device)
    tr.model.load_adapter(a.adapter)
    # Standalone eval runs outside accelerate's autocast, so the UNet has to be
    # cast explicitly to match the bf16 inputs the VAE/text-encoder produce.
    tr.model.unet.to(dtype=tr.weight_dtype)
    tr.model.unet.eval(); tr.model.vae.eval(); tr.model.text_encoder.eval()

    if a.adapter_scale is not None or a.skip_scale is not None:
        from models.attention_processor import StandAloneAttnProcessor as _SA
        n = 0
        if a.adapter_scale is not None:
            for pr in tr.model.unet.attn_processors.values():
                if isinstance(pr, _SA):
                    pr.scale = float(a.adapter_scale); n += 1
            print(f"[eval] cross-attention adapter scale -> {a.adapter_scale} at {n} sites", flush=True)
        # The skip injector is a separate path. Scaling only the attention sites
        # leaves it running, so an "adapter off" control that forgets it is
        # really measuring "skip injector alone" -- which is a useful ablation,
        # but not the control it claims to be.
        inj = getattr(tr.model, "skip_injector", None)
        if inj is not None:
            inj.scale = float(a.skip_scale if a.skip_scale is not None else a.adapter_scale)
            print(f"[eval] skip injector scale -> {inj.scale}", flush=True)

    tr.makeDatasets(cfg.datasets.data_path, cfg.datasets.img_path,
                    frequency_rate=0.0, mode=cfg.datasets.frequency_mode,
                    img_size=int(cfg.datasets.img_size),
                    source_size=getattr(cfg.datasets, "source_size", None),
                    frequency_img=cfg.datasets.frequency_img, mean=0.5, std=0.5,
                    caption_path=getattr(cfg.datasets, "caption_path", None), ddp=False)
    pipe = tr.build_pipeline()

    lpips_m = LPIPSMetric(device)
    clip_m = CLIPScore(device)
    res = int(cfg.datasets.img_size)

    n = min(a.n_images, len(tr.test_dataset))
    idxs = list(range(n))
    cutoffs = [float(x) for x in a.cutoffs.split(",")]

    # ---- reference set (written once; also the FID reference) ----
    ref_dir = os.path.join(a.out, "reference")
    os.makedirs(ref_dir, exist_ok=True)
    gts, caps = [], []
    for i in idxs:
        d = tr.test_dataset[i]
        gt_unit = ((d["gt"] + 1.0) / 2.0).clamp(0, 1)
        gts.append(gt_unit); caps.append(d["caption"])
        p = os.path.join(ref_dir, f"{i:05d}.png")
        if not os.path.exists(p):
            save_png(gt_unit, p)
    gts = torch.stack(gts)
    print(f"[eval] {n} reference images at {res}px -> {ref_dir}", flush=True)

    results = {
        "provenance": {
            "adapter": os.path.abspath(a.adapter),
            "config": os.path.abspath(a.config),
            "git_commit": git_commit(),
            "n_images": n, "steps": a.steps, "cfg_scale": a.cfg, "seed": a.seed,
            "k_div": a.k_div, "n_div": a.n_div, "resolution": res,
            "adapter_scale": a.adapter_scale,
            "skip_scale": a.skip_scale,
            "gpu": torch.cuda.get_device_name(0),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "per_cutoff": {},
    }

    for cut in cutoffs:
        t0 = time.time()
        gen_dir = os.path.join(a.out, f"gen_r{cut}")
        os.makedirs(gen_dir, exist_ok=True)
        SC, EF, LP, CL = [], [], [], []

        for s in range(0, n, a.batch):
            gt_b = gts[s:s + a.batch].to(device)
            cap_b = caps[s:s + a.batch]
            cond_unit = make_condition_torch(gt_b, cut,
                frequency_img=cfg.datasets.frequency_img, mode=cfg.datasets.frequency_mode)
            cond = to_model_range(cond_unit).to(dtype=tr.weight_dtype)

            g = torch.Generator(device="cpu").manual_seed(a.seed + s)
            out = pipe(cap_b, cond, height=res, width=res,
                       num_inference_steps=a.steps, num_images_per_prompt=1,
                       guidance_scale=a.cfg, generator=g,
                       output_type="pt").images.float().clamp(0, 1)

            SC.append(structure_consistency(out, cond_unit, cut))
            EF.append(edge_f1(out, gt_b))
            LP.append(lpips_m.distance(out, gt_b))
            CL.append(clip_m.score(out, cap_b))
            for j in range(out.shape[0]):
                save_png(out[j], os.path.join(gen_dir, f"{s + j:05d}.png"))
            print(f"[eval] r={cut} {min(s + a.batch, n)}/{n}", flush=True)

        SC, EF, LP, CL = map(np.concatenate, (SC, EF, LP, CL))

        # ---- diversity: K samples of the same condition, different seeds ----
        div = []
        for i in range(min(a.n_div, n)):
            gt_1 = gts[i:i + 1].to(device)
            cond_unit = make_condition_torch(gt_1, cut,
                frequency_img=cfg.datasets.frequency_img, mode=cfg.datasets.frequency_mode)
            cond = to_model_range(cond_unit).to(dtype=tr.weight_dtype)
            g = torch.Generator(device="cpu").manual_seed(a.seed + 100000 + i)
            samp = pipe([caps[i]], cond, height=res, width=res,
                        num_inference_steps=a.steps, num_images_per_prompt=a.k_div,
                        guidance_scale=a.cfg, generator=g,
                        output_type="pt").images.float().clamp(0, 1)
            div.append(lpips_m.diversity(samp))
        div = np.array(div)

        fid = None
        if not a.skip_fid:
            try:
                fid = fid_from_dirs(gen_dir, ref_dir, device="cuda")
            except Exception as e:
                print(f"[eval] FID not computed at r={cut}: {e}", flush=True)

        def agg(x):
            return {"mean": float(np.mean(x)), "std": float(np.std(x)),
                    "sem": float(np.std(x) / np.sqrt(len(x))), "n": int(len(x))}

        results["per_cutoff"][str(cut)] = {
            "SC": agg(SC), "EdgeF1": agg(EF), "LPIPS": agg(LP),
            "CLIP": agg(CL), "Diversity": agg(div),
            "FID": fid, "seconds": round(time.time() - t0, 1),
        }
        np.savez(os.path.join(a.out, f"raw_r{cut}.npz"),
                 SC=SC, EdgeF1=EF, LPIPS=LP, CLIP=CL, Diversity=div)
        with open(os.path.join(a.out, "results.json"), "w") as f:
            json.dump(results, f, indent=2)
        print(f"[eval] r={cut} done  SC={SC.mean():.4f}  EdgeF1={EF.mean():.4f}  "
              f"LPIPS={LP.mean():.4f}  CLIP={CL.mean():.2f}  Div={div.mean():.4f}  "
              f"FID={fid}", flush=True)

    print(f"\n[eval] wrote {os.path.join(a.out, 'results.json')}")
    hdr = f"{'r':>6} {'SC':>8} {'EdgeF1':>8} {'LPIPS':>8} {'CLIP':>8} {'Div':>8} {'FID':>8}"
    print(hdr); print("-" * len(hdr))
    for k, v in results["per_cutoff"].items():
        fid_s = f"{v['FID']:.2f}" if v["FID"] is not None else "n/a"
        print(f"{k:>6} {v['SC']['mean']:>8.4f} {v['EdgeF1']['mean']:>8.4f} "
              f"{v['LPIPS']['mean']:>8.4f} {v['CLIP']['mean']:>8.2f} "
              f"{v['Diversity']['mean']:>8.4f} {fid_s:>8}")


if __name__ == "__main__":
    main()
