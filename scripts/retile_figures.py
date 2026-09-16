#!/usr/bin/env python3
"""T02 (branch A): rebuild figures/dial.png from the saved evaluation
generations and write a provenance manifest for both figure sheets.

Diagnosis (scripts/figure_provenance.py, figure_index_scan.py): every tile of
strength.png is pixel-identical to the saved evaluation generations, and so is
row 1 of dial.png, but rows 2-3 of dial.png are a different noise draw of the
same test images (SSIM 0.50-0.65 against results/E10b at their own index, and
lower against every other run). Evaluation draws noise per batch
(eval/run_eval.py:165, manual_seed(seed + batch_start)), so a figure script
that re-seeded per image reproduces image 0 and diverges afterwards. Re-tiling
from the saved generations removes the mismatch without any new inference.
"""
import hashlib, json, os
from PIL import Image

TILE, GAP, RESAMPLER = 384, 6, "LANCZOS"
CUTS = ["0.0", "0.05", "0.1", "0.2", "0.3", "0.5", "0.7", "1.0"]
IDS = ["00000", "00001", "00002"]
MAIN_CKPT = "ckpt_dir/E10b_skip_inject/final/adapter.safetensors"
SPEC_CKPT = "ckpt_dir/E2_fixed_r0.1/final/adapter.safetensors"
CAPTION = "BLIP caption of that test image (per-image, as in Table 1)"

def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()

def train_seed(ckpt_run):
    p = os.path.join("ckpt_dir", ckpt_run, "run_manifest.json")
    m = json.load(open(p))
    return m.get("trainer", {}).get("seed") or m.get("config", {}).get("trainer", {}).get("seed")

def sampling_seed(run):
    return json.load(open(f"results/{run}/results.json"))["provenance"]["seed"]

CK = {MAIN_CKPT: sha(MAIN_CKPT), SPEC_CKPT: sha(SPEC_CKPT)}
GEN_MODE = "eval/run_eval.py:165 - torch.Generator(cpu).manual_seed(seed + batch_start), one draw per batch"

def entry(fig, row, col, label, run, idx, ckpt, ckpt_run, r=None, scale=None, ref=False):
    src = f"results/{run}/{'reference' if ref else 'gen_r' + r}/{idx}.png"
    return dict(figure=fig, tile=[row, col], label=label, run_dir=f"results/{run}",
                checkpoint=None if ref else ckpt, checkpoint_sha256=None if ref else CK[ckpt],
                train_seed=None if ref else train_seed(ckpt_run),
                sampling_seed=None if ref else sampling_seed(run),
                generator_mode=None if ref else GEN_MODE,
                image_idx=idx, caption=None if ref else CAPTION, r=r, scale=scale,
                source_png=src, source_png_sha256=sha(src), resampler=RESAMPLER)

# ---- dial.png: re-tile from results/E10b -----------------------------------
step = TILE + GAP
canvas = Image.new("RGB", (9 * TILE + 8 * GAP, 3 * TILE + 2 * GAP), "white")
dial = []
for r_i, idx in enumerate(IDS):
    cols = [("reference", f"results/E10b/reference/{idx}.png", None)] + \
           [(f"r={c}", f"results/E10b/gen_r{c}/{idx}.png", c) for c in CUTS]
    for c_i, (label, path, c) in enumerate(cols):
        canvas.paste(Image.open(path).convert("RGB").resize((TILE, TILE), getattr(Image, RESAMPLER)),
                     (c_i * step, r_i * step))
        dial.append(entry("dial", r_i, c_i, label, "E10b", idx, MAIN_CKPT, "E10b_skip_inject",
                          r=c, ref=(c is None)))
assert canvas.size == (3504, 1164), canvas.size
canvas.save("papers/figures/dial.png", optimize=True)

# ---- strength.png: already the evaluation generations; record its manifest --
S_TILE = 256
S_COLS = [("reference", "E10b", "reference", None, None, None, True),
          ("fixed-r=0.1 scale 1.00", "E16_spec_scale1.0", "0.1", "0.1", "1.00", SPEC_CKPT, False),
          ("fixed-r=0.1 scale 0.75", "E16_spec_scale0.75", "0.1", "0.1", "0.75", SPEC_CKPT, False),
          ("fixed-r=0.1 scale 0.50", "E16_spec_scale0.5", "0.1", "0.1", "0.50", SPEC_CKPT, False),
          ("fixed-r=0.1 scale 0.25", "E16_spec_scale0.25", "0.1", "0.1", "0.25", SPEC_CKPT, False),
          ("randomised-r scale 0.50", "E12_scale0.5", "0.1", "0.1", "0.50", MAIN_CKPT, False),
          ("randomised-r scale 0.25", "E12_scale0.25", "0.1", "0.1", "0.25", MAIN_CKPT, False),
          ("randomised-r r=0.1 full strength", "E10b", "0.1", "0.1", "1.00", MAIN_CKPT, False),
          ("randomised-r r=0.3 full strength", "E10b", "0.3", "0.3", "1.00", MAIN_CKPT, False)]
strength = []
for r_i, idx in enumerate(IDS[:2]):
    for c_i, (label, run, sub, r, scale, ckpt, ref) in enumerate(S_COLS):
        ck_run = "E2_fixed_r0.1" if ckpt == SPEC_CKPT else "E10b_skip_inject"
        strength.append(entry("strength", r_i, c_i, label, run, idx, ckpt, ck_run, r=r, scale=scale, ref=ref))

for name, data in (("dial", dial), ("strength", strength)):
    out = f"papers/figures/{name}.manifest.json"
    json.dump({"figure": f"figures/{name}.png", "tile_px": TILE if name == "dial" else S_TILE,
               "gap_px": GAP if name == "dial" else 0, "resampler": RESAMPLER,
               "built_by": "scripts/retile_figures.py", "tiles": data},
              open(out, "w"), indent=1)
    print("wrote", out, len(data), "tiles")
im = Image.open("papers/figures/dial.png"); print("dial.png", im.size, im.mode)
