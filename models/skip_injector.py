"""ControlNet-style skip-connection injection for the structure condition.

The cross-attention adapter injects the condition at 16 attention sites only.
ControlNet instead adds residuals to every skip connection the decoder consumes
and to the mid block, at all four resolutions. This module is the lightweight
version of that placement: a small strided conv encoder reads the *raw*
condition image (never the frozen VAE, which was measured to destroy the
condition above r ~ 0.5) and emits one residual per skip tensor.

Every output head is zero-initialised, so at step 0 the injector is an exact
no-op and the UNet is bit-identical to stock SD1.5 -- the invariant that
`scripts/check_conditioning.py` check [5] asserts.

The residuals are handed to `UNet2DConditionModel.forward` through its existing
`down_block_additional_residuals` / `mid_block_additional_residual` arguments,
so no part of the UNet is monkey-patched.
"""
from typing import List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["SkipInjector"]


def _conv(i, o, k=3, s=1):
    return nn.Conv2d(i, o, kernel_size=k, stride=s, padding=k // 2)


class SkipInjector(nn.Module):
    """Condition image -> one residual per UNet skip tensor.

    Args:
        spec: [(channels, spatial), ...] for each `down_block_res_samples`
            entry, in order, as reported by the UNet itself.
        mid_spec: (channels, spatial) for `mid_block_res_sample`.
        in_ch: channels of the conditioning image (3).
        base: width multiplier; the trunk runs at base*4 / base*8.
        scale: fixed multiplier applied to every residual.
    """

    def __init__(self,
                 spec: Sequence[Tuple[int, int]],
                 mid_spec: Tuple[int, int],
                 in_ch: int = 3,
                 base: int = 32,
                 scale: float = 1.0):
        super().__init__()
        self.spec = list(spec)
        self.mid_spec = tuple(mid_spec)
        self.scale = float(scale)

        b = base
        # 512 -> 64: the same 8x reduction the VAE performs, but trainable and
        # reading the image directly.
        self.stem = nn.Sequential(
            _conv(in_ch, b, 3, 2), nn.SiLU(),
            _conv(b, b, 3, 1), nn.SiLU(),
            _conv(b, b * 2, 3, 2), nn.SiLU(),
            _conv(b * 2, b * 2, 3, 1), nn.SiLU(),
            _conv(b * 2, b * 4, 3, 2), nn.SiLU(),
            _conv(b * 4, b * 4, 3, 1), nn.SiLU(),
        )
        self.down32 = nn.Sequential(_conv(b * 4, b * 8, 3, 2), nn.SiLU(),
                                    _conv(b * 8, b * 8, 3, 1), nn.SiLU())
        self.down16 = nn.Sequential(_conv(b * 8, b * 8, 3, 2), nn.SiLU(),
                                    _conv(b * 8, b * 8, 3, 1), nn.SiLU())
        self.down8 = nn.Sequential(_conv(b * 8, b * 8, 3, 2), nn.SiLU(),
                                   _conv(b * 8, b * 8, 3, 1), nn.SiLU())
        self.trunk_ch = {64: b * 4, 32: b * 8, 16: b * 8, 8: b * 8}

        def head(res_hw, out_ch):
            c = nn.Conv2d(self.trunk_ch[res_hw], out_ch, kernel_size=1)
            nn.init.zeros_(c.weight); nn.init.zeros_(c.bias)   # exact no-op at init
            return c

        self.heads = nn.ModuleList([head(s, c) for c, s in self.spec])
        self.mid_head = head(self.mid_spec[1], self.mid_spec[0])

    def forward(self, cond: torch.Tensor) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """cond: (B, 3, H, W) in [-1, 1]. Returns (down_residuals, mid_residual).

        Training feeds bf16 while this module holds fp32 weights, so the input is
        cast in and the residuals cast back out: they are added to UNet skip
        tensors and must not silently promote those to fp32.
        """
        in_dtype = cond.dtype
        w_dtype = next(self.parameters()).dtype
        cond = cond.to(dtype=w_dtype)
        f64 = self.stem(cond)
        f32 = self.down32(f64)
        f16 = self.down16(f32)
        f8 = self.down8(f16)
        feats = {64: f64, 32: f32, 16: f16, 8: f8}

        downs = []
        for head, (c, s) in zip(self.heads, self.spec):
            x = feats[s]
            if x.shape[-1] != s:                      # non-512 inputs
                x = F.interpolate(x, size=(s, s), mode="bilinear", align_corners=False)
            downs.append((head(x) * self.scale).to(dtype=in_dtype))
        m = feats[self.mid_spec[1]]
        if m.shape[-1] != self.mid_spec[1]:
            m = F.interpolate(m, size=(self.mid_spec[1],) * 2, mode="bilinear", align_corners=False)
        return downs, (self.mid_head(m) * self.scale).to(dtype=in_dtype)

    @staticmethod
    @torch.no_grad()
    def spec_from_unet(unet, latent_hw: int = 64) -> Tuple[List[Tuple[int, int]], Tuple[int, int]]:
        """Ask the UNet what shapes it expects, rather than assuming them."""
        dev = next(unet.parameters()).device
        dt = next(unet.parameters()).dtype
        lat = torch.zeros(1, unet.config.in_channels, latent_hw, latent_hw, device=dev, dtype=dt)
        t = torch.zeros(1, dtype=torch.long, device=dev)
        ehs = torch.zeros(1, 77, unet.config.cross_attention_dim, device=dev, dtype=dt)
        caught = {}
        orig = unet.mid_block.forward

        def spy(hidden_states, *a, **kw):
            out = orig(hidden_states, *a, **kw)
            caught["mid"] = (out.shape[1], out.shape[2])
            return out
        unet.mid_block.forward = spy
        try:
            unet(lat, t, ehs, return_dict=False)
        finally:
            unet.mid_block.forward = orig

        emb_dim = unet.config.block_out_channels[0] * 4
        emb = torch.zeros(1, emb_dim, device=dev, dtype=dt)
        h = unet.conv_in(lat)
        res = [h]
        for blk in unet.down_blocks:
            if getattr(blk, "has_cross_attention", False):
                h, s = blk(hidden_states=h, temb=emb, encoder_hidden_states=ehs)
            else:
                h, s = blk(hidden_states=h, temb=emb)
            res.extend(s)
        return [(t_.shape[1], t_.shape[2]) for t_ in res], caught["mid"]
