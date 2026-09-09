#!/usr/bin/env python3
"""Gate check: does the structure condition actually reach and change the UNet?

If the adapter were silently disconnected -- a wrong kwarg name, a processor
that never fires -- training would still run and the loss would still fall,
because the frozen UNet alone can denoise. Everything downstream would then be
measuring nothing. This asserts, on real weights:

  1. the SAProcessor is installed at every cross-attention site
  2. changing the condition changes the output
  3. gradients reach every adapter tensor
  4. the condition is downsampled to each UNet resolution

Note on instrumentation: diffusers filters `cross_attention_kwargs` through
`inspect.signature(processor.__call__)`, so any wrapper placed on a processor
must preserve that signature or it silently drops `ip_hidden_states` -- which
looks exactly like a disconnected adapter. The resolution probe below therefore
runs as a separate pass using a forward hook on the Attention module, leaving
the processor's signature untouched.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("HF_HOME", "/home/work/data/hf_cache")

import torch
from trainer.fa_trainer import Trainer
from models.attention_processor import StandAloneAttnProcessor as SAProcessor

cfg_path = sys.argv[1] if len(sys.argv) > 1 else "config/fa_sd15_ffhq512.yaml"
tr = Trainer(cfg_path); tr.device = 0
tr.initialize(0, 1); tr.model_define(0)
unet = tr.model.unet.to(0)
ok = True

procs = unet.attn_processors
sa = [k for k, v in procs.items() if isinstance(v, SAProcessor)]
cross = [k for k in procs if not k.endswith("attn1.processor")]
print(f"[1] SAProcessor at {len(sa)}/{len(cross)} cross-attention sites")
ok &= (len(sa) == len(cross) and len(sa) > 0)

B = 2
lat = torch.randn(B, 4, 64, 64, device=0)
t = torch.tensor([500, 500], device=0)
ehs = torch.randn(B, 77, 768, device=0)
# Build the condition the way training does. With a trainable conv encoder the
# check must run through it, or the encoder's parameters look gradient-less
# simply because the probe fed the adapter a pre-made latent.
_enc = tr.model.cond_encoder
_res = int(tr.config.datasets.img_size)
_imgA = torch.randn(B, 3, _res, _res, device=0).clamp(-1, 1)
_imgB = torch.randn(B, 3, _res, _res, device=0).clamp(-1, 1)
if _enc is not None:
    cA = _enc(_imgA)
    cB = _enc(_imgB)
else:
    cA = torch.randn(B, 4, 64, 64, device=0)
    cB = torch.randn(B, 4, 64, 64, device=0)

# The probe has to exercise every path training does. A skip injector receives
# its input from the raw condition image, not the adapter latent, so a forward
# that omits its residuals makes its parameters look gradient-less for a reason
# that has nothing to do with the model.
def fwd(cond, img=None):
    extra = tr.skip_residuals(img if img is not None else _imgA) or {}
    return unet(lat, t, ehs, return_dict=False,
                cross_attention_kwargs={"ip_hidden_states": cond}, **extra)[0]

with torch.no_grad():
    oA = fwd(cA.detach(), _imgA)
    oB = fwd(cB.detach(), _imgB)
    oZ = fwd(torch.zeros_like(cA), torch.zeros_like(_imgA))
scale = oA.abs().mean().item()
d_ab = (oA - oB).abs().mean().item()
d_az = (oA - oZ).abs().mean().item()
print(f"[2] output scale {scale:.4f}")
print(f"    |out(A) - out(B)|    = {d_ab:.6f}  ({100*d_ab/scale:.2f}% of scale)")
print(f"    |out(A) - out(zero)| = {d_az:.6f}  ({100*d_az/scale:.2f}% of scale)")

# With a zero-initialised gate the adapter is an exact no-op at init by design,
# so "condition changes the output" is the wrong assertion to make on a fresh
# model. Open the gate to a small value and re-test: the path must be live once
# the gate is non-zero, and the gradient check below still has to pass either way.
gated = [pr for pr in procs.values()
         if isinstance(pr, SAProcessor) and getattr(pr, "out_gate", None) is not None]
if gated and d_ab <= 1e-4 * scale:
    print(f"    zero-init gate detected on {len(gated)} sites -- no-op at init is correct;")
    with torch.no_grad():
        for pr in gated:
            pr.out_gate.fill_(0.1)
    with torch.no_grad():
        oA2, oB2 = fwd(cA.detach()), fwd(cB.detach())
    d_ab = (oA2 - oB2).abs().mean().item()
    d_az = (oA2 - oZ).abs().mean().item()
    print(f"    re-tested with gate=0.1: |out(A)-out(B)| = {d_ab:.6f} "
          f"({100*d_ab/scale:.2f}% of scale)")
    with torch.no_grad():
        for pr in gated:
            pr.out_gate.zero_()

live = d_ab > 1e-4 * scale and d_az > 1e-4 * scale
print(f"    condition changes the output: {'YES' if live else 'NO -- ADAPTER DISCONNECTED'}")
ok &= live

def grad_coverage():
    for q in tr.model.params:
        q.grad = None
    cond = _enc(_imgA) if _enc is not None else cA
    fwd(cond, _imgA).float().pow(2).mean().backward()
    return [q for q in tr.model.params if q.grad is not None and q.grad.abs().sum() > 0]

n = len(tr.model.params)
g = grad_coverage()
print(f"[3] adapter tensors with non-zero gradient at init: {len(g)}/{n}")

if gated and len(g) < n:
    # Expected for a zero-init gate: with gate == 0 the projections behind it
    # see grad * gate == 0. ControlNet's zero convolutions behave the same way.
    # What matters is that one optimizer step lifts the gate off zero and the
    # whole adapter starts training -- so assert that, not the init state.
    print(f"    zero-init gate: {len(g)} tensors moving are the gates themselves.")
    opt = torch.optim.AdamW(tr.model.params, lr=1e-4)
    opt.step(); opt.zero_grad(set_to_none=True)
    max_gate = max(float(pr.out_gate.abs().max()) for pr in gated)
    g = grad_coverage()
    print(f"    after one optimizer step: max|gate| = {max_gate:.2e}, "
          f"gradient reaches {len(g)}/{n}")

ok &= (len(g) == n)

# resolution probe: hook the Attention modules, never touch processor signatures
seen = {}
handles = []
for name, mod in unet.named_modules():
    if type(mod).__name__ == "Attention" and isinstance(getattr(mod, "processor", None), SAProcessor):
        def hook(m, args, kwargs, out, nm=name):
            hs = kwargs.get("hidden_states", args[0] if args else None)
            ip = kwargs.get("ip_hidden_states")
            if hs is not None and ip is not None:
                seen[nm] = (int(hs.shape[1] ** 0.5), int(ip.shape[-1]))
        handles.append(mod.register_forward_hook(hook, with_kwargs=True))
with torch.no_grad():
    fwd(cA.detach())
for h in handles:
    h.remove()
pairs = sorted(set(seen.values()))
print(f"[4] (latent hw, condition hw fed in) at {len(seen)} sites: {pairs}")
ok &= (len(seen) == len(sa))

# [5] The invariant that would have caught the residual double-count on day one:
# an adapter is an *additive* modification, so switching it off must reproduce
# the untouched backbone exactly. The original processor returned
# `hidden_states + text_out` while BasicTransformerBlock adds the residual
# itself, so norm2(x) entered the stream twice at every cross-attention layer
# and scale=0 was still far from stock SD1.5.
saved = {}
for name, pr in procs.items():
    if isinstance(pr, SAProcessor):
        saved[name] = pr.scale
        pr.scale = 0.0
# "off" has to mean every injection path off, not just the attention one --
# otherwise a skip injector keeps contributing and the invariant reads as
# violated for the wrong reason.
_inj = getattr(tr.model, "skip_injector", None)
_inj_scale = _inj.scale if _inj is not None else None
if _inj is not None:
    _inj.scale = 0.0
with torch.no_grad():
    off = fwd(cA.detach(), _imgA)
if _inj is not None:
    _inj.scale = _inj_scale

from diffusers import UNet2DConditionModel as _U
stock = _U.from_pretrained(tr.config.model.model_id, subfolder="unet").eval().to(0)
with torch.no_grad():
    ref = stock(lat, t, ehs, return_dict=False)[0]
del stock
torch.cuda.empty_cache()

for name, pr in procs.items():
    if isinstance(pr, SAProcessor):
        pr.scale = saved[name]

dev = (ref - off).abs().max().item()
print(f"[5] max|stock UNet - adapter(scale=0)| = {dev:.3e}")
print("    adapter is purely additive: " + ("YES" if dev < 1e-3 else "NO -- IT PERTURBS THE BACKBONE"))
ok &= (dev < 1e-3)

# [6] One real training step, in the dtype and autocast the trainer actually
# uses. Checks [1]-[5] all run in fp32, which is how a bf16-only failure
# (an fp32 side module fed bf16 activations) reached five training attempts
# before anything noticed.
print("[6] one training-shaped step under autocast")
try:
    _b = 2
    _img = torch.randn(_b, 3, _res, _res, device=0, dtype=tr.weight_dtype).clamp(-1, 1)
    with torch.no_grad(), torch.autocast("cuda", dtype=tr.weight_dtype):
        _lat = tr.model.vae.encode(_img).latent_dist.sample() * tr.model.vae.config.scaling_factor
        _cond = tr.encode_condition(_img)
        _ehs = tr.encode_text(["a photo of a face"] * _b)
    _noise = torch.randn_like(_lat)
    _t = torch.randint(0, 1000, (_b,), device=0).long()
    _noisy = tr.model.noise_scheduler.add_noise(_lat, _noise, _t)
    for q in tr.model.params:
        q.grad = None
    with torch.autocast("cuda", dtype=tr.weight_dtype):
        _extra = tr.skip_residuals(_img) or {}
        _pred = tr.model.unet(_noisy, _t, _ehs, return_dict=False,
                              cross_attention_kwargs={"ip_hidden_states": _cond},
                              **_extra)[0]
    _loss = torch.nn.functional.mse_loss(_pred.float(), _noise.float())
    _loss.backward()
    _moved = sum(1 for q in tr.model.params if q.grad is not None and q.grad.abs().sum() > 0)
    print(f"    forward+backward OK in {tr.weight_dtype}, loss {_loss.item():.4f}, "
          f"gradient on {_moved}/{n} tensors")
except Exception as _e:
    print(f"    FAILED: {type(_e).__name__}: {_e}")
    ok = False

print("\n" + ("PASS - conditioning path is live" if ok else "FAIL - see above"))
sys.exit(0 if ok else 1)
