#!/usr/bin/env python3
"""How much of the structure condition survives the frozen SD VAE?

The adapter reads the condition as a VAE latent. If the encode->decode round
trip does not preserve the condition, no adapter can carry information that is
no longer there -- so this bounds what any design reading that latent can do.

Score is the Pearson correlation between the condition fed in and the same
condition re-extracted from the round-tripped image, which is exactly the
`structure_consistency` metric used everywhere else, with the VAE standing in
for the generator.

`--source_size` routes the image through `datalibs.compose.SourceResolution`,
the *same* transform the training pipeline uses, rather than reimplementing it
here: the image is downscaled and restored to the working resolution, and only
then is the condition built. Reimplementing it is how the first version of this
script produced numbers that disagreed with the pipeline it was meant to
describe.
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("HF_HOME", "/home/work/data/hf_cache")

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from datalibs.compose import SourceResolution
from datalibs.frequency import make_condition_torch, to_model_range
from eval.metrics import structure_consistency

ap = argparse.ArgumentParser()
ap.add_argument("--img_dir", default="/home/work/data/ffhq512/images")
ap.add_argument("--n", type=int, default=24)
ap.add_argument("--res", type=int, default=512)
ap.add_argument("--cutoffs", default="0.0,0.05,0.1,0.2,0.3,0.5,0.7,1.0")
ap.add_argument("--source_size", type=int, default=None,
                help="build the condition at this size then upscale to --res")
ap.add_argument("--out", default=None)
# Which autoencoder. Defaults reproduce the original SD1.5 measurement exactly.
ap.add_argument("--vae_id", default="stable-diffusion-v1-5/stable-diffusion-v1-5")
ap.add_argument("--subfolder", default="vae", help="'' for a repo that is the VAE itself")
ap.add_argument("--dtype", default="fp32", choices=["fp32", "bf16"])
ap.add_argument("--batch", type=int, default=16,
                help="encode/decode in chunks: 500 images at 512^2 in one call OOMs")
a = ap.parse_args()

from diffusers import AutoencoderKL
_dt = torch.float32 if a.dtype == "fp32" else torch.bfloat16
vae = AutoencoderKL.from_pretrained(a.vae_id, subfolder=(a.subfolder or None),
                                    torch_dtype=_dt).eval().cuda()
print(f"VAE {a.vae_id}/{a.subfolder}: latent_channels={vae.config.latent_channels}, "
      f"scaling_factor={vae.config.scaling_factor}", flush=True)

files = sorted(os.listdir(a.img_dir))[:a.n]
ims = np.stack([cv2.cvtColor(cv2.imread(os.path.join(a.img_dir, f)), cv2.COLOR_BGR2RGB)
                .astype(np.float32) / 255.0 for f in files])
ims = torch.from_numpy(ims).permute(0, 3, 1, 2)
if ims.shape[-1] != a.res:
    ims = F.interpolate(ims, size=(a.res, a.res), mode="bilinear", align_corners=False)

out = {}
for cut in [float(x) for x in a.cutoffs.split(",")]:
    src = ims
    if a.source_size and a.source_size != a.res:
        # exactly what the dataloader does, via the shared transform
        sr = SourceResolution(a.source_size)
        arr = src.permute(0, 2, 3, 1).cpu().numpy()
        arr = np.stack([sr({"gt": x, "filtered_image": x.copy()})["filtered_image"] for x in arr])
        src = torch.from_numpy(arr).permute(0, 3, 1, 2)
    scs = []
    for i in range(0, src.shape[0], a.batch):
        cond = make_condition_torch(src[i:i + a.batch].cuda(), cut)   # [0,1]
        with torch.no_grad():
            # .mode() not .sample(): this number goes in a table, so it must be
            # deterministic. Sampling moved it by ~0.05 at mid r between runs.
            z = vae.encode(to_model_range(cond).to(_dt)).latent_dist.mode()
            rt = vae.decode(z).sample.float()                        # [-1,1]
        rt = ((rt + 1) / 2).clamp(0, 1)
        scs.append(structure_consistency(rt, cond.float(), cut))
    sc = np.concatenate(scs)
    out[str(cut)] = {"mean": float(sc.mean()), "sem": float(sc.std() / np.sqrt(len(sc))),
                     "n": int(len(sc))}
    print(f"  r={cut:<5} kept = {sc.mean():.4f} ± {sc.std()/np.sqrt(len(sc)):.4f}", flush=True)

if a.out:
    with open(a.out, "w") as f:
        json.dump({"provenance": {"n": len(files), "res": a.res, "source_size": a.source_size,
                                  "img_dir": a.img_dir, "vae_id": a.vae_id,
                                  "subfolder": a.subfolder, "dtype": a.dtype,
                                  "latent_channels": int(vae.config.latent_channels)},
                   "kept": out}, f, indent=2)
    print(f"wrote {a.out}")
