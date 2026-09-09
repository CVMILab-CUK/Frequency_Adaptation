"""Metrics for the structure-fidelity / realism / diversity trade-off.

Every function here returns a number computed from tensors it was handed.
Nothing is estimated, defaulted, or filled in: a metric that cannot be
computed raises, so a missing number can never silently become a plausible
one in a results table.
"""
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from datalibs.frequency import make_condition_torch, rgb_to_gray_torch


# --------------------------------------------------------------- structure
@torch.no_grad()
def structure_consistency(gen: torch.Tensor, cond_unit: torch.Tensor, cutoff: float,
                          frequency_img: str = "gray") -> np.ndarray:
    """Per-image Pearson correlation between the condition the model was given
    and the condition re-extracted from what it generated.

    This is the direct test of "did it obey the structure": run the *same*
    high-pass filter at the *same* cutoff over the generated image and see
    whether the result matches the input. Correlation (not L1) because the
    filter output is per-image min-max normalised, so absolute scale carries
    no information.

    Args:
        gen:       (B, 3, H, W) generated images in [0, 1]
        cond_unit: (B, 3, H, W) the conditioning image the model saw, in [0, 1]
        cutoff:    the cutoff that produced `cond_unit`
    Returns:
        (B,) correlations in [-1, 1]
    """
    assert gen.shape == cond_unit.shape, f"{gen.shape} vs {cond_unit.shape}"
    # Re-extract through the identical chain the condition was built with --
    # a plain channel mean would use different grayscale weights than cv2 and
    # make the correlation depend on the colour of the generated image.
    re_cond = make_condition_torch(gen, cutoff, frequency_img=frequency_img)
    re_cond = rgb_to_gray_torch(re_cond)
    tgt = rgb_to_gray_torch(cond_unit)

    a = re_cond.flatten(1).float()
    b = tgt.flatten(1).float()
    a = a - a.mean(dim=1, keepdim=True)
    b = b - b.mean(dim=1, keepdim=True)
    denom = a.norm(dim=1) * b.norm(dim=1)
    corr = (a * b).sum(dim=1) / denom.clamp_min(1e-8)
    return corr.cpu().numpy()


@torch.no_grad()
def edge_f1(gen: torch.Tensor, gt: torch.Tensor, tol: int = 2,
            low: int = 100, high: int = 200) -> np.ndarray:
    """Canny-edge F1 between generated and reference images, with a `tol`-pixel
    match tolerance (dilation), as used in the sketch-to-image literature."""
    import cv2
    out = []
    g = (gen.clamp(0, 1) * 255).byte().cpu().numpy().transpose(0, 2, 3, 1)
    t = (gt.clamp(0, 1) * 255).byte().cpu().numpy().transpose(0, 2, 3, 1)
    k = np.ones((2 * tol + 1, 2 * tol + 1), np.uint8)
    for i in range(len(g)):
        eg = cv2.Canny(cv2.cvtColor(g[i], cv2.COLOR_RGB2GRAY), low, high) > 0
        et = cv2.Canny(cv2.cvtColor(t[i], cv2.COLOR_RGB2GRAY), low, high) > 0
        if eg.sum() == 0 or et.sum() == 0:
            out.append(0.0); continue
        eg_d = cv2.dilate(eg.astype(np.uint8), k) > 0
        et_d = cv2.dilate(et.astype(np.uint8), k) > 0
        prec = (eg & et_d).sum() / eg.sum()
        rec = (et & eg_d).sum() / et.sum()
        out.append(0.0 if prec + rec == 0 else float(2 * prec * rec / (prec + rec)))
    return np.array(out)


# --------------------------------------------------------------- perceptual
class LPIPSMetric:
    def __init__(self, device, net="alex"):
        import lpips
        self.fn = lpips.LPIPS(net=net).to(device).eval()
        self.device = device

    @torch.no_grad()
    def distance(self, a: torch.Tensor, b: torch.Tensor) -> np.ndarray:
        """a, b in [0, 1] -> (B,) LPIPS distances."""
        d = self.fn(a.to(self.device) * 2 - 1, b.to(self.device) * 2 - 1)
        return d.flatten().cpu().numpy()

    @torch.no_grad()
    def diversity(self, samples: torch.Tensor) -> float:
        """Mean pairwise LPIPS among (N, 3, H, W) samples of one condition."""
        n = samples.shape[0]
        if n < 2:
            raise ValueError("diversity needs at least 2 samples per condition")
        ds = []
        for i in range(n):
            for j in range(i + 1, n):
                ds.append(self.distance(samples[i:i + 1], samples[j:j + 1])[0])
        return float(np.mean(ds))


# --------------------------------------------------------------- text align
class CLIPScore:
    """CLIP image-text cosine similarity (the ViT-B/32 convention, x100)."""

    def __init__(self, device, model_id="ViT-B-32", pretrained="openai"):
        import open_clip
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_id, pretrained=pretrained)
        self.model = self.model.to(device).eval()
        self.tokenizer = open_clip.get_tokenizer(model_id)
        self.device = device
        self.mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device).view(1, 3, 1, 1)

    @torch.no_grad()
    def score(self, images: torch.Tensor, captions: List[str]) -> np.ndarray:
        x = F.interpolate(images.to(self.device), size=224, mode="bicubic", align_corners=False)
        x = ((x.clamp(0, 1) - self.mean) / self.std)
        im = self.model.encode_image(x)
        tx = self.model.encode_text(self.tokenizer(captions).to(self.device))
        im = im / im.norm(dim=-1, keepdim=True)
        tx = tx / tx.norm(dim=-1, keepdim=True)
        return (100.0 * (im * tx).sum(dim=-1)).float().cpu().numpy()


# --------------------------------------------------------------- distribution
def fid_from_dirs(gen_dir: str, ref_dir: str, device="cuda", batch_size: int = 64) -> float:
    """Clean-FID between two folders of PNGs. Raises if either is too small."""
    import os
    from cleanfid import fid as cfid
    n_gen = len([f for f in os.listdir(gen_dir) if f.endswith(".png")])
    n_ref = len([f for f in os.listdir(ref_dir) if f.endswith(".png")])
    if min(n_gen, n_ref) < 200:
        raise ValueError(
            f"FID needs a meaningful sample count; got gen={n_gen} ref={n_ref}. "
            "Report it as not-computed rather than quoting an unstable value.")
    return float(cfid.compute_fid(gen_dir, ref_dir, device=device,
                                  batch_size=batch_size, verbose=False))
