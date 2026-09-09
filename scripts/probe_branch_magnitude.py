#!/usr/bin/env python3
"""How loud is the structure branch compared to the text branch?

The adapter contributes `scale * out_spatial` to a residual stream that also
carries `text_out` and the residual itself. If that contribution is orders of
magnitude smaller than the others, no amount of training on the projections
will make the condition matter -- the signal is simply drowned, and the fix is
a magnitude/gating one rather than a capacity one.

Reports, per cross-attention site, the mean L2 norm per token of:
  residual   the stream the block adds back
  text       the text cross-attention output
  struct     the adapter's contribution
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("HF_HOME", "/home/work/data/hf_cache")

import torch
from trainer.fa_trainer import Trainer
from models.attention_processor import StandAloneAttnProcessor as SAProcessor

ap = argparse.ArgumentParser()
ap.add_argument("-c", "--config", default="config/fa_sd15_ffhq512.yaml")
ap.add_argument("--adapter", default=None)
a = ap.parse_args()

tr = Trainer(a.config); tr.device = 0
tr.initialize(0, 1); tr.model_define(0)
if a.adapter:
    tr.model.load_adapter(a.adapter)
unet = tr.model.unet.to(0).eval()

stats = {}
orig = SAProcessor.__call__

def probe(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None,
          temb=None, scale=1.0, ip_hidden_states=None):
    key = getattr(self, "_probe_name", "?")
    res = hidden_states
    out = orig(self, attn, hidden_states, encoder_hidden_states, attention_mask,
               temb, scale, ip_hidden_states)
    # out = text_out + scale*out_spatial ; recompute with the structure off
    saved, self.scale = self.scale, 0.0
    out_text = orig(self, attn, hidden_states, encoder_hidden_states, attention_mask,
                    temb, scale, ip_hidden_states)
    self.scale = saved
    struct = out - out_text
    stats[key] = (res.norm(dim=-1).mean().item(),
                  out_text.norm(dim=-1).mean().item(),
                  struct.norm(dim=-1).mean().item())
    return out

for name, pr in unet.attn_processors.items():
    if isinstance(pr, SAProcessor):
        pr._probe_name = name.replace(".processor", "")
SAProcessor.__call__ = probe

lat = torch.randn(1, 4, 64, 64, device=0)
t = torch.tensor([500], device=0)
ehs = torch.randn(1, 77, 768, device=0)
cond = torch.randn(1, 4, 64, 64, device=0)
with torch.no_grad():
    unet(lat, t, ehs, return_dict=False, cross_attention_kwargs={"ip_hidden_states": cond})
SAProcessor.__call__ = orig

print(f"{'site':<44} {'residual':>9} {'text':>9} {'struct':>9} {'struct/text':>12}")
print("-" * 88)
rs = ts = ss = 0.0
for k, (r, t_, s_) in stats.items():
    print(f"{k:<44} {r:>9.3f} {t_:>9.3f} {s_:>9.4f} {s_/max(t_,1e-9):>12.4f}")
    rs += r; ts += t_; ss += s_
n = len(stats)
print("-" * 88)
print(f"{'mean':<44} {rs/n:>9.3f} {ts/n:>9.3f} {ss/n:>9.4f} {ss/max(ts,1e-9):>12.4f}")
