"""Single source of truth for the frequency (high-pass) conditioning signal.

Both the CPU dataloader transform and the GPU validation path call into this
module, so the training-time and evaluation-time conditioning can never drift
apart again. (They previously did: the dataloader took `np.abs` of the inverse
FFT while validation took `.real`, giving two essentially uncorrelated signals.)

Parameterisation
----------------
`cutoff` is a *normalised* radius in [0, 1]:

    cut_radius_in_pixels = cutoff * (min(H, W) / 2)

so ``cutoff=0`` is the identity (no filtering) and ``cutoff=1`` removes every
frequency inside the disc inscribed in the spectrum. The definition is
resolution independent, which is what lets a curve measured at 512px be
compared against one measured at 256px.

Output convention
-----------------
`high_pass_*` returns the real part of the inverse transform, per-image
min-max normalised to [0, 1]. Callers that feed a VAE must map to [-1, 1] via
`to_model_range`, exactly as ground-truth images are normalised.
"""
from typing import Union

import numpy as np
import torch

__all__ = [
    "high_pass_numpy",
    "high_pass_torch",
    "make_condition_torch",
    "rgb_to_gray_torch",
    "to_model_range",
    "radial_mask_numpy",
    "radial_mask_torch",
    "RGB2GRAY_WEIGHTS",
]

# The exact weights cv2.COLOR_RGB2GRAY uses, so the GPU path reproduces the
# dataloader's grayscale step bit-for-bit rather than approximately.
RGB2GRAY_WEIGHTS = (0.299, 0.587, 0.114)


def to_model_range(x):
    """Map a [0, 1] signal to the [-1, 1] range the VAE expects."""
    return x * 2.0 - 1.0


# --------------------------------------------------------------------------
# numpy (dataloader / CPU workers)
# --------------------------------------------------------------------------
def radial_mask_numpy(h: int, w: int, cutoff: float, mode: str = "high_pass") -> np.ndarray:
    """(h, w) float32 binary mask. 0 inside the cut disc for 'high_pass'."""
    radius = float(cutoff) * (min(h, w) / 2.0)
    cy, cx = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    inside = (x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2
    if mode == "high_pass":
        mask = np.ones((h, w), np.float32)
        mask[inside] = 0.0
    elif mode == "low_pass":
        mask = np.zeros((h, w), np.float32)
        mask[inside] = 1.0
    else:
        raise ValueError(f"unknown mode {mode!r}")
    return mask


def high_pass_numpy(image: np.ndarray, cutoff: float, mode: str = "high_pass") -> np.ndarray:
    """Filter an (H, W, C) float image. Returns (H, W, C) float32 in [0, 1].

    `cutoff == 0` with mode 'high_pass' is the identity, and is returned
    without touching the FFT so that r=0 is exactly the original image.
    """
    if image.ndim == 2:
        image = image[:, :, None]
    image = image.astype(np.float32, copy=False)

    if mode == "high_pass" and cutoff <= 0.0:
        return _minmax_numpy(image)

    h, w = image.shape[:2]
    mask = radial_mask_numpy(h, w, cutoff, mode)[:, :, None]

    f = np.fft.fft2(image, axes=(0, 1))
    f = np.fft.fftshift(f, axes=(0, 1))
    f = f * mask
    f = np.fft.ifftshift(f, axes=(0, 1))
    out = np.fft.ifft2(f, axes=(0, 1))

    # The inverse of a real signal through a symmetric real mask is real up to
    # numerical error; `.real` keeps edge polarity, `np.abs` would fold it away.
    return _minmax_numpy(out.real.astype(np.float32))


def _minmax_numpy(x: np.ndarray) -> np.ndarray:
    lo, hi = float(x.min()), float(x.max())
    if hi - lo < 1e-6:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - lo) / (hi - lo)).astype(np.float32)


# --------------------------------------------------------------------------
# torch (GPU: validation, inference, on-the-fly augmentation)
# --------------------------------------------------------------------------
def radial_mask_torch(h: int, w: int, cutoff, device, mode: str = "high_pass") -> torch.Tensor:
    """Mask broadcastable against (B, C, H, W).

    `cutoff` may be a float or a (B,) tensor, giving a per-sample radius.
    """
    if not torch.is_tensor(cutoff):
        cutoff = torch.tensor([float(cutoff)], device=device)
    cutoff = cutoff.to(device=device, dtype=torch.float32).view(-1, 1, 1, 1)

    radius = cutoff * (min(h, w) / 2.0)
    cy, cx = h // 2, w // 2
    y, x = torch.meshgrid(
        torch.arange(h, device=device, dtype=torch.float32),
        torch.arange(w, device=device, dtype=torch.float32),
        indexing="ij",
    )
    d2 = ((y - cy) ** 2 + (x - cx) ** 2).view(1, 1, h, w)
    inside = d2 <= radius ** 2
    if mode == "high_pass":
        return (~inside).to(torch.float32)
    elif mode == "low_pass":
        return inside.to(torch.float32)
    raise ValueError(f"unknown mode {mode!r}")


@torch.no_grad()
def high_pass_torch(x: torch.Tensor, cutoff, mode: str = "high_pass") -> torch.Tensor:
    """Filter a (B, C, H, W) tensor in [0, 1]. Returns the same shape in [0, 1].

    Mirrors `high_pass_numpy` exactly, including the r=0 identity shortcut and
    the per-image min-max normalisation.
    """
    orig_dtype = x.dtype
    xf = x.float()
    b, c, h, w = xf.shape
    device = xf.device

    if not torch.is_tensor(cutoff):
        cutoff = torch.full((b,), float(cutoff), device=device)
    cutoff = cutoff.to(device=device, dtype=torch.float32)

    mask = radial_mask_torch(h, w, cutoff, device, mode)  # (B or 1, 1, H, W)

    f = torch.fft.fft2(xf)
    f = torch.fft.fftshift(f, dim=(-2, -1))
    f = f * mask
    f = torch.fft.ifftshift(f, dim=(-2, -1))
    out = torch.fft.ifft2(f).real

    out = _minmax_torch(out)

    if mode == "high_pass":
        # r == 0 is the identity; keep it bit-exact rather than round-tripping.
        identity = (cutoff <= 0.0).view(b, 1, 1, 1)
        out = torch.where(identity, _minmax_torch(xf), out)

    return out.to(dtype=orig_dtype)


def _minmax_torch(x: torch.Tensor) -> torch.Tensor:
    lo = torch.amin(x, dim=(-3, -2, -1), keepdim=True)
    hi = torch.amax(x, dim=(-3, -2, -1), keepdim=True)
    return torch.where(hi - lo < 1e-6, torch.zeros_like(x), (x - lo) / (hi - lo + 1e-12))


# --------------------------------------------------------------------------
# the whole conditioning chain, in one place
# --------------------------------------------------------------------------
def rgb_to_gray_torch(x: torch.Tensor) -> torch.Tensor:
    """(B, 3, H, W) -> (B, 1, H, W) using cv2's RGB2GRAY weights."""
    if x.shape[1] == 1:
        return x
    w = torch.tensor(RGB2GRAY_WEIGHTS, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    return (x * w).sum(dim=1, keepdim=True)


@torch.no_grad()
def make_condition_torch(images_unit: torch.Tensor,
                         cutoff,
                         frequency_img: str = "gray",
                         mode: str = "high_pass") -> torch.Tensor:
    """Build the conditioning image exactly as the dataloader does.

    This is the GPU twin of `datalibs.compose.Frequency_Filtering`, and it
    exists because unifying only the FFT was not enough: the dataloader
    converts to grayscale *before* filtering, and every evaluation call site
    was filtering RGB directly. That reintroduced the same train/eval split the
    shared filter was written to remove -- most visibly at r=0, where the
    dataloader hands the model a grayscale image and the evaluator was handing
    it a colour one.

    Args:
        images_unit: (B, 3, H, W) in [0, 1]
        cutoff: float or (B,) tensor, the normalised radius
        frequency_img: 'gray' (collapse colour first) or 'color' (per-channel)
    Returns:
        (B, 3, H, W) in [0, 1] -- map with `to_model_range` before the VAE.
    """
    if frequency_img not in ("gray", "color"):
        raise ValueError(f"frequency_img must be 'gray' or 'color', got {frequency_img!r}")

    x = rgb_to_gray_torch(images_unit) if frequency_img == "gray" else images_unit
    out = high_pass_torch(x, cutoff, mode=mode)
    if out.shape[1] == 1:
        out = out.repeat(1, 3, 1, 1)
    return out
