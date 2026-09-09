"""Backwards-compatible GPU FFT filters.

These are thin wrappers over `datalibs.frequency`, which both the dataloader
and the evaluation path share. They exist so older scripts keep importing
`GpuFixedFFT` / `GpuRandomFFT`; new code should call `high_pass_torch`
directly.

`fft_scale` here is the same normalised cutoff used everywhere else:
radius = fft_scale * min(H, W) / 2.
"""
import torch
import torch.nn as nn

from datalibs.frequency import high_pass_torch

__all__ = ["GpuFixedFFT", "GpuRandomFFT"]


class GpuFixedFFT(nn.Module):
    """Apply a fixed-cutoff high-pass filter to a (B, C, H, W) batch in [0, 1]."""

    def __init__(self, fft_scale: float = 0.5, mode: str = "high_pass"):
        super().__init__()
        assert 0.0 <= fft_scale <= 1.0, "fft_scale must be in [0, 1]"
        self.fft_scale = fft_scale
        self.mode = mode

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return high_pass_torch(x, self.fft_scale, mode=self.mode)


class GpuRandomFFT(nn.Module):
    """Per-sample random cutoff, for on-GPU randomised-bandwidth augmentation."""

    def __init__(self, fft_scale_range=(0.0, 1.0), mode: str = "high_pass"):
        super().__init__()
        lo, hi = fft_scale_range
        assert 0.0 <= lo <= hi <= 1.0, "fft_scale_range must satisfy 0 <= min <= max <= 1"
        self.fft_scale_range = (float(lo), float(hi))
        self.mode = mode

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lo, hi = self.fft_scale_range
        cutoff = torch.rand(x.shape[0], device=x.device) * (hi - lo) + lo
        return high_pass_torch(x, cutoff, mode=self.mode)
