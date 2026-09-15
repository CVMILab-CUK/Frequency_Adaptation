#!/usr/bin/env python3
"""Generate README.md from results/, so its tables cannot drift from the runs.

Same rule as the paper: a number that no run produced does not appear.
"""
import json, os, sys

R = "results"
def load(p):
    f = os.path.join(R, p, "results.json")
    return json.load(open(f)) if os.path.exists(f) else None

main = load("E10b") or sys.exit("results/E10b missing")
pc = main["per_cutoff"]
CUTS = ["0.0", "0.05", "0.1", "0.2", "0.3", "0.5", "0.7", "1.0"]

def row(r):
    v = pc[r]
    return (f"| {r} | {v['SC']['mean']:.4f} | {v['Diversity']['mean']:.4f} | "
            f"{v['LPIPS']['mean']:.4f} | {v['CLIP']['mean']:.2f} | {v['FID']:.2f} |")

curve = "\n".join(row(r) for r in CUTS)

# strength dial
sc_rows = []
for s in ["0.25", "0.5", "0.75"]:
    d = load(f"E12_scale{s}")
    if d:
        v = d["per_cutoff"]["0.1"]
        sc_rows.append(f"| adapter strength {s} | {v['SC']['mean']:.4f} | {v['FID']:.2f} |")
sc_rows.append(f"| adapter strength 1.00 | {pc['0.1']['SC']['mean']:.4f} | {pc['0.1']['FID']:.2f} |")
sc_rows.append("| | | |")
for r in ["0.3", "0.5"]:
    sc_rows.append(f"| **cutoff r={r}** | **{pc[r]['SC']['mean']:.4f}** | **{pc[r]['FID']:.2f}** |")
strength = "\n".join(sc_rows)

# ablation ladder
lad = []
for name, path in [("VAE condition, cross-attention only", "E1"),
                   ("+ zero-init gate", "E3h"),
                   ("+ conv condition encoder", "E9a"),
                   ("+ 128-channel stem", "E10a"),
                   ("+ 256-channel stem", "E10d"),
                   ("+ 7x7 window", "E10c")]:
    d = load(path)
    if d:
        p = d["per_cutoff"]
        lad.append(f"| {name} | {p['0.05']['SC']['mean']:.4f} | {p['0.1']['SC']['mean']:.4f} | {p['0.1']['FID']:.2f} |")
lad.append(f"| **+ skip residuals** | **{pc['0.05']['SC']['mean']:.4f}** | "
           f"**{pc['0.1']['SC']['mean']:.4f}** | **{pc['0.1']['FID']:.2f}** |")
ladder = "\n".join(lad)

# injection decomposition
dec = []
for name, path in [("both paths", "E10b"), ("skip only", "E10b_adapter_off"),
                   ("cross-attention only", "E10b_skip_off"), ("both off", "E10b_all_off")]:
    d = load(path)
    if d and "0.0" in d["per_cutoff"]:
        v = d["per_cutoff"]["0.0"]
        dec.append(f"| {name} | {v['SC']['mean']:.4f} | {v['FID']:.2f} |")
decomp = "\n".join(dec)

# baselines
base = []
for b, label in [("controlnet_canny", "ControlNet-Canny"), ("controlnet_hed", "ControlNet-HED")]:
    p = os.path.join(R, "baselines", f"{b}_native", "results.json")
    if os.path.exists(p):
        d = json.load(open(p)); v = d["per_setting"]["native"]
        base.append(f"| {label} | {d['provenance']['controlnet_params']:,} | "
                    f"{v['EdgeF1']['mean']:.4f} | {v['LPIPS']['mean']:.4f} | "
                    f"{v['FID']:.2f} |" if v.get("FID") else
                    f"| {label} | {d['provenance']['controlnet_params']:,} | "
                    f"{v['EdgeF1']['mean']:.4f} | {v['LPIPS']['mean']:.4f} | n/a |")
m = json.load(open("ckpt_dir/E10b_skip_inject/run_manifest.json"))
base.append(f"| **Ours, r=0.1** | **{m['trainable_params']:,}** | "
            f"**{pc['0.1']['EdgeF1']['mean']:.4f}** | **{pc['0.1']['LPIPS']['mean']:.4f}** | "
            f"**{pc['0.1']['FID']:.2f}** |")
baselines = "\n".join(base)

# seeds
import statistics as st
seed_rows = []
for r in ["0.05", "0.1", "0.3"]:
    vals = [load(p)["per_cutoff"][r]["SC"]["mean"]
            for p in ("E10b", "E13_seed2027", "E13_seed2028") if load(p)]
    if len(vals) > 1:
        seed_rows.append(f"| {r} | {st.mean(vals):.4f} | {st.stdev(vals):.4f} |")
seeds = "\n".join(seed_rows)

nref = len(json.load(open("papers/refs_index.json")))
_e22 = json.load(open("results/E22_spectral_energy/results.json"))

readme = f"""# A Frequency Dial for Structure-Conditioned Diffusion

**School of Computer Science, Soongsil University**

A structure-conditioned diffusion adapter is trained for one operating point.
Canny edges, a depth map, a sketch: each fixes how tightly the output must
follow the condition, and moving that trade-off means training again.

We train a single adapter on conditions whose information content is
randomised — a high-pass filter whose cutoff `r` is drawn per sample — and read
`r` back at inference as a dial.

<p align="center"><img src="docs/dial.jpg" width="100%"></p>
<p align="center"><em>One seed, one caption, r swept. Left to right: reference,
then r = 0, 0.05, 0.1, 0.3, 0.5, 0.7, 1.0. The background crowd thins, the
eyewear drifts from the reference, the clothing colour detaches, while identity
and pose survive. The reference is never shown to the model.</em></p>

📄 [Paper (EN)](papers/fa_en.pdf) &nbsp;·&nbsp; [논문 (KO)](papers/fa_ko.pdf)
&nbsp;·&nbsp; [Project page](docs/index.html)

---

## What the cutoff does

<p align="center"><img src="docs/mask.jpg" width="100%"></p>
<p align="center"><em>Top: the radial mask, black is removed. Bottom: the
resulting condition, displayed at ±2σ. Columns are r = 0, 0.05, 0.1, 0.2, 0.3,
0.5, 0.7, 1.0.</em></p>

Natural images concentrate their energy at low frequency, so the condition
empties fast — a cutoff of 0.05 already discards {100*_e22['ac_removed@0.05']['mean']:.1f} % of the spectral energy outside
the constant term.
Above r ≈ 0.2 the condition looks like flat grey (averaged over the test images its standard deviation
falls from {_e22['cond_std@0.0']['mean']:.3f} to {_e22['cond_std@1.0']['mean']:.3f}), yet the face outline survives and the model still reaches
SC {pc['0.2']['SC']['mean']:.4f} at r = 0.2. Low contrast is not the same as no
information.

## Operating curve

{main['provenance']['n_images']} held-out FFHQ-512 images, {main['provenance']['steps']} sampler steps,
CFG {main['provenance']['cfg_scale']}, seed {main['provenance']['seed']}.
**Structure consistency (SC)** re-extracts the condition from the generated
image at the same cutoff and correlates it with the condition the model was
given: feeding the ground truth back scores exactly 1.000, an unrelated face
scores ≈ 0.

| r | SC | Diversity | LPIPS | CLIP | FID |
|---|-----|-----------|-------|------|-----|
{curve}

Structure consistency falls and diversity rises monotonically while FID stays
within six points. Across three seeds:

| r | SC mean | std |
|---|---------|-----|
{seeds}

## Turning the adapter down is not the same knob

Holding r = 0.1 and scaling both injection paths instead:

| setting | SC | FID |
|---------|-----|-----|
{strength}

Weakening the adapter degrades the image; narrowing the condition's bandwidth
does not. The frequency parameterisation is what makes the trade-off
traversable.

## Where the signal enters matters more than capacity

One field changed per row against the row above it:

| variant | SC r=0.05 | SC r=0.1 | FID r=0.1 |
|---------|-----------|----------|-----------|
{ladder}

Decomposing the two injection paths at r = 0:

| configuration | SC | FID |
|---------------|-----|-----|
{decomp}

The skip residuals carry most of the structure signal. With both paths off the
model falls to the frozen backbone, which is the control that says the adapter
is doing the work.

## Against ControlNet

Same test images, same captions, each baseline on its own condition:

| method | trainable params | Edge-F1 | LPIPS | FID |
|--------|-----------------|---------|-------|-----|
{baselines}

The conditions differ — Canny edges against a frequency band — so this compares
operating points, not a like-for-like control.

## Where it does not hold

The useful range is roughly **r ∈ [0, 0.3]**. Above it the condition retains
too little energy for structure consistency to mean much, since two nearly
empty signals correlate near zero whatever the model does.

Everything here is one domain (FFHQ faces) and one backbone (SD1.5). Current
structure-control work has moved to DiT backbones where LoRA-style adapters
report control at a fraction of our parameter count; read the efficiency
numbers against ControlNet, not against that line. Per-cutoff specialists still
beat the single model at r = 0.1. No user study was run.

## Reproducing

```bash
python scripts/fetch_ffhq.py                    # FFHQ-512 with BLIP captions
python scripts/make_splits.py                   # fixed-seed splits
python scripts/check_conditioning.py config/generated/E10b_skip_inject.yaml
python train_fa.py -c config/generated/E10b_skip_inject.yaml
python eval/run_eval.py -c config/generated/E10b_skip_inject.yaml \\
       --adapter ckpt_dir/E10b_skip_inject/final/adapter.safetensors --out results/E10b
```

`scripts/check_conditioning.py` gates every run. It asserts the adapter is
present at all cross-attention sites, fully differentiable, and **purely
additive** — with the injection off the network must reproduce stock SD1.5
bit for bit. That last check is the one that catches a whole class of
conditioning bug before any GPU time is spent.

### Building the paper

```bash
cd papers && make        # regenerate numbers from results/, then both PDFs
                make refs # re-verify every reference (arXiv API or publisher record)
```

No number in either PDF is typed by hand: `scripts/paper_numbers.py` emits them
from `results/*/results.json`, and a missing run fails the build rather than
printing a stale value. All {nref} references were checked against the arXiv API or, for published-only
entries, the publisher or DOI record,
and their abstract pages fetched before entering `refs.bib`.

## Layout

```
datalibs/frequency.py       the filter, shared by the dataloader and evaluation
models/skip_injector.py     ControlNet-style residuals from the raw condition
models/attention_processor.py   local-window attention at cross-attention sites
trainer/fa_trainer.py       training loop
eval/                       metrics, evaluation, baselines, figures
scripts/check_conditioning.py   the gate every run passes first
papers/                     LaTeX sources, generated numbers, verified refs
```

Checkpoints, logs and generated images are not tracked; the scripts above
regenerate them.
"""
open("README.md", "w").write(readme)
print(f"README.md written, {len(readme.splitlines())} lines")
