"""A trainable conv stem for the structure condition, replacing the frozen VAE.

Why this exists (measured, 24 FFHQ-512 images): pushing the high-pass condition
through the frozen SD VAE destroys it, progressively with the cutoff r --
encode->decode correlation falls 0.93 (r=0.05) / 0.67 (r=0.2) / 0.54 (r=0.3) /
0.29 (r=0.5) / 0.06 (r=0.7) / 0.001 (r=1.0). Above r~0.5 the condition is
simply gone, and no adapter can carry information that is not there.

Below that the information does survive the VAE, but only the *decoder* -- a
large non-linear network -- can recover it. The adapter reads the latent with a
1x1 convolution per position, which sees far less. ControlNet and T2I-Adapter
both avoid the VAE for their condition entirely and train a small strided conv
stem on the raw condition image; this is that stem.

Output is (B, out_channels, H/8, W/8), matching the VAE latent grid so
everything downstream of it is unchanged.
"""
import torch
import torch.nn as nn

__all__ = ["ConditionEncoder"]


class ConditionEncoder(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 4, base: int = 32):
        super().__init__()
        self.out_channels = out_channels
        # GroupNorm after each stride keeps the activation scale from collapsing
        # through seven layers: without it the stem's output std measured 0.013
        # against the VAE latent's 0.66-0.84 (a 36-65x mismatch) and, worse, was
        # identical across cutoffs -- the output was bias-dominated and barely
        # responded to the input at all.
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, base, 3, padding=1),            nn.SiLU(),
            nn.Conv2d(base, base, 3, padding=1, stride=2),                      # /2
            nn.GroupNorm(8, base),                                 nn.SiLU(),
            nn.Conv2d(base, base * 2, 3, padding=1),               nn.SiLU(),
            nn.Conv2d(base * 2, base * 2, 3, padding=1, stride=2),              # /4
            nn.GroupNorm(8, base * 2),                             nn.SiLU(),
            nn.Conv2d(base * 2, base * 4, 3, padding=1),           nn.SiLU(),
            nn.Conv2d(base * 4, base * 4, 3, padding=1, stride=2),              # /8
            nn.GroupNorm(8, base * 4),                             nn.SiLU(),
            nn.Conv2d(base * 4, out_channels, 3, padding=1),
        )
        # Match the VAE latent scale the adapter was designed around, so
        # `cond_encoder: conv` is a swap of the encoder and not also a silent
        # change of the signal's magnitude.
        self.out_scale = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 3, H, W) in [-1, 1] -> (B, out_channels, H/8, W/8)."""
        return self.net(x.to(dtype=next(self.parameters()).dtype)) * self.out_scale
