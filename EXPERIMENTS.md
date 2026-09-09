# Frequency Adaptation — Experimental Plan

## 1. The claim

> A single ~0.1 M-parameter local-window adapter, trained with **randomised
> frequency-bandwidth conditioning**, gives **continuous test-time control**
> over the structure-fidelity ↔ diversity trade-off — matching per-bandwidth
> specialist models at every operating point, at 3575× fewer trainable
> parameters than ControlNet, and transferring zero-shot to real sketches.

Three separable, individually falsifiable sub-claims:

| # | Sub-claim | Killed by |
|---|-----------|-----------|
| C1 | One random-r model ≈ K specialists, each at its own r | Any r where a specialist clearly wins |
| C2 | r is a smooth, monotone control dial at test time | A non-monotone or discontinuous SC(r) curve |
| C3 | 0.1 M params suffice for competitive quality | FID far above ControlNet at matched SC |

C3 is the one most likely to fail. The capacity ablation (E3b) exists to find
out *early* whether the 1×1 projection is the bottleneck.

## 2. Measured facts this plan is built on

Both measured on this machine, on FFHQ-512 (`scripts/` reproduce them):

**Trainable parameters** (SD1.5, 16 cross-attention sites):
`to_k/to_v` 99,840 + the 18 downsample convs actually reachable 1,224 = **101,064**
— 0.0118 % of the UNet's 859,622,028.

An earlier count of 104,192 included 3,128 parameters in downsample convolutions that
no attention site ever reached (`scripts/check_conditioning.py` found 68/160 tensors
receiving gradient). Each site is now built with exactly the number of halvings it
needs, and all 68 remaining tensors train.
→ **3575×** smaller than ControlNet-SD1.5 and **762×** smaller than T2I-Adapter-sketch-v2.
Both baseline counts were counted from the downloaded weights on this machine, not cited:
ControlNet 361,279,120 · T2I-Adapter 77,000,640 · ours 101,064.

**Retained spectral energy vs cutoff r** (64 images, grayscale):

| r | 0.00 | 0.05 | 0.10 | 0.20 | 0.30 | 0.50 | 0.70 | 1.00 |
|---|------|------|------|------|------|------|------|------|
| energy kept | .2289 | .0246 | .0118 | .0055 | .0033 | .0014 | .0006 | .00005 |

log₁₀(energy) is close to linear in r over [0.05, 0.7] (−0.32, −0.33, −0.22,
−0.37 dex per step), so **uniform sampling in r is already near-uniform in
information content** — that is why `rate_sampler: uniform` is the default
rather than an arbitrary choice. The exception is r < 0.05, where energy falls
~10× inside a very narrow band; `sqrt` and `log` samplers exist to test whether
that regime is undertrained (E3f).

### The grayscale half of the conditioning chain (found at step 2,000)

The first in-loop validation curve came back with r=0 as the **worst** operating
point (LPIPS 0.965 vs ~0.87 everywhere else), which is backwards: r=0 hands the
model the grayscale image, the most informative condition there is.

Cause: unifying the FFT was not enough. `Frequency_Filtering` converts to
grayscale *before* filtering, but every evaluation call site was filtering RGB
directly — so at r=0 training fed a grayscale image and evaluation fed a colour
one. The same class of defect as the `abs`/`real` split, one stage earlier in
the chain.

Fixed by moving the whole chain — grayscale, filter, channel expansion — into
`datalibs.frequency.make_condition_torch`, and routing training, in-loop
validation, `run_eval.py`, `run_baselines.py` and `structure_consistency`
through it. Verified against the CPU dataloader: max difference **3e-6** across
r ∈ {0, 0.05, 0.1, 0.3, 0.5, 1.0}.

Training was never affected — the dataloader path was correct throughout — so
E1 resumed from its step-2,000 checkpoint rather than restarting.

### Mid-run finding at step 4,000: the adapter barely uses r > 0

Measured on the step-4,000 checkpoint (48 test images, 25 sampler steps, cfg 7.5):

| r | SC | ±sem | SC for an unrelated image | SC ceiling |
|---|-----|------|---------------------------|------------|
| 0.0 | **0.2271** | 0.036 | 0.122 | 1.0 |
| 0.1 | 0.0301 | 0.015 | −0.018 | 1.0 |
| 0.3 | 0.0304 | 0.014 | −0.016 | 1.0 |
| 0.7 | 0.0170 | 0.012 | −0.020 | 1.0 |

The metric was validated first: feeding the ground truth back gives SC = 1.000
at every r, an unrelated image gives ≈0 (0.122 at r=0, because faces share
gross structure), and a flat grey image gives 0.000. So SC reads directly as
the fraction of achievable structure obedience.

**The model uses the condition at r=0 and almost ignores it for r > 0** — about
3 % of ceiling, which is the regime the whole method exists for.

Re-measured at step 8,000 to test whether it was simply early:

| r | SC @4,000 | SC @8,000 | Δ | sem |
|---|-----------|-----------|---|-----|
| 0.0 | 0.2271 | 0.2767 | +0.0496 | 0.033 |
| 0.1 | 0.0301 | 0.0343 | +0.0042 | 0.014 |
| 0.3 | 0.0304 | 0.0300 | −0.0004 | 0.015 |
| 0.7 | 0.0170 | 0.0209 | +0.0038 | 0.014 |

Doubling the training moved r > 0 by less than a third of one standard error.
**"Too early" is dead.** r=0 (colourisation) is progressing; the structure
regime is flat, and there is no basis for expecting the remaining 3× of
training to reach a useful value.

One mechanism is testable and cheap: **the adapter has no zero-initialisation.**
ControlNet's zero convolutions, LoRA's B=0 and IP-Adapter all start the
injection at exactly zero. Here the injection is `hidden_states + 1.0 *
out_spatial` with random init and a fixed scale, so the adapter perturbs a
frozen UNet from step 0 (measured: 2.0 % of output scale) and early training
goes into undoing that. **E3h** tests it.

### Gate check before any training

`scripts/check_conditioning.py` asserts, on real weights, that the structure
condition reaches the UNet and that every adapter tensor trains. It is a gate,
not a diagnostic: a disconnected adapter would still show a falling loss,
because the frozen UNet alone can denoise, and every number downstream would
then be measuring nothing.

Current status: **PASS** — 16/16 cross-attention sites carry the adapter,
swapping the condition moves the output by 1.9 % of its scale, all 68 adapter
tensors receive gradient, and the condition is halved correctly for each of the
four latent resolutions (64, 32, 16, 8).


## ROOT CAUSE — the residual was counted twice (found 2026-08-29 04:40)

The E1 result below was measured on a **defective architecture**. It is kept
because it is what led to the defect, and because a recorded measurement is not
deleted just because its configuration turned out to be wrong.

The adapter-off control is what exposed it. Forcing the injection scale to 0
should leave a well-formed model; instead:

| | SC (r=0) | LPIPS | CLIP | FID |
|---|---------|-------|------|-----|
| adapter on | 0.2831 | 0.7111 | 29.38 | 75.07 |
| adapter off | 0.0038 | 0.9661 | 19.27 | **332.67** |

Turning the adapter off *destroyed* the model, which is the opposite of an
inert adapter. Cause: SD1.5 cross-attention has `residual_connection = False`,
so `BasicTransformerBlock` adds the residual itself —

```python
attn_output = self.attn2(norm_hidden_states, ...)
hidden_states = attn_output + hidden_states     # the block adds it
```

— and a processor must therefore return only the attention output.
`StandAloneAttnProcessor` returned `hidden_states + text_out`, so
`norm2(hidden_states)` entered the residual stream a second time at every one
of the 16 cross-attention layers (measured magnitude 7.19 against a residual of
17.88, a ~40 % perturbation).

**The adapter's 101,064 parameters spent 26,420 steps learning to cancel that
term rather than to carry structure.** That is why SC stayed at ~0 for r > 0
while removing the adapter collapsed the model: the structure branch had become
a negative copy of the spurious term.

Fixed: the processor now returns `text_out + scale * out_spatial` only. The
structure signal still shapes the text query and is still added residually
once, exactly as the original design intended.

Verified, and now asserted by the gate check on every run:
**`max |stock SD1.5 − adapter(scale=0)| = 0.000e+00`** — the adapter is a
purely additive modification. This invariant would have caught the defect
before any GPU time was spent, and it is check [5] in
`scripts/check_conditioning.py`.

E1 and E3h are both being re-run on the fixed architecture. E3h's first attempt
(1,416 steps) was discarded: it was testing zero-initialisation on top of the
same defect.

## E1 RESULT (PRE-FIX, defective architecture) — measured, complete

Full test protocol: 500 test images, 50 sampler steps,
cfg 7.5, seed 2026, 512px, NVIDIA H200.
Checkpoint `FA_SD15_r0-1/final`,
git `59ada210a83f`, run 2026-08-29 03:08:18.

| r | SC | ±sem | EdgeF1 | LPIPS | CLIP | Diversity | FID |
|---|-----|------|--------|-------|------|-----------|-----|
| 0.0 | 0.2831 | 0.0081 | 0.2830 | 0.7111 | 29.38 | 0.7162 | 75.07 |
| 0.05 | 0.0205 | 0.0028 | 0.3037 | 0.7181 | 29.19 | 0.7227 | 73.30 |
| 0.1 | 0.0140 | 0.0028 | 0.3098 | 0.7169 | 29.30 | 0.7185 | 71.58 |
| 0.2 | 0.0214 | 0.0028 | 0.3139 | 0.7117 | 29.42 | 0.7153 | 71.94 |
| 0.3 | 0.0140 | 0.0028 | 0.3101 | 0.7149 | 29.41 | 0.7200 | 71.33 |
| 0.5 | 0.0106 | 0.0031 | 0.3073 | 0.7200 | 29.42 | 0.7249 | 71.77 |
| 0.7 | 0.0127 | 0.0033 | 0.3023 | 0.7291 | 29.43 | 0.7314 | 73.30 |
| 1.0 | 0.0003 | 0.0004 | 0.2989 | 0.7411 | 29.31 | 0.7379 | 76.74 |

**This was a null result for the central claim — on an architecture that
turned out to be defective.** It is reported as measured, and it is not
evidence about the method. SC at r > 0 sits at
0.010–0.021 against a ceiling of 1.0 and an unrelated-image floor of ≈0 — at
r=1.0 it is 0.0003. The model generates faces from the caption and does not use
the structure condition. CLIP is flat at 29.2–29.4, so text conditioning works
normally; it is specifically the adapter path that is inert.

Only r=0 shows real conditioning (SC 0.2831), and that is the colourisation
regime where the condition is the grayscale image itself.

Secondary observations, all weak but internally consistent with "the condition
is being ignored": FID has a shallow minimum near r=0.3 (71.33) and rises at
both ends (75.07 at r=0, 76.74 at r=1); diversity rises monotonically with r
(0.7153 → 0.7379), as expected when the condition constrains less.

Caveat on the schedule: E1 ran to step 26,420 rather than the planned 24,420,
because the resume path restored `global_step` but restarted the epoch loop.
The in-loop curve is flat from step 8,000 onward, so the extra 8 % changes
nothing, but the discrepancy is recorded rather than smoothed over. The step
budget is now enforced across resumes.

**Interpretation is blocked on one control**: FID 71–77 could mean the adapter
is inert *or* actively harmful. `results/E1_adapter_off` re-runs the identical
protocol with the adapter's injection scale forced to 0.

## 3. Datasets

| Tag | Data | Size | Role |
|-----|------|------|------|
| D1 | FFHQ-512 + BLIP captions (`Ryan-sjtu/ffhq512-caption`) | 70 k | main training + in-domain test |
| D2 | CelebA-HQ 512 | 30 k | near-domain transfer (no training) |
| D3 | COCO-2017 val, 512 centre-crop | 5 k | far-domain transfer + a second full training run |
| D4 | Real hand-drawn sketches | ~1 k | zero-shot, the practical payoff |

Splits are regenerated from the files actually on disk with a fixed seed
(`scripts/make_splits.py`, seed 2026): 93 % train / 2 % valid / 5 % test.
D1 test = 3 500 images; metrics use the first 500 unless stated.

**D3 matters most for acceptance.** A faces-only paper reads as a case study;
one general-domain training run is what makes the mechanism look general.

## 4. Metrics

Per operating point r, over the same images with the same seeds:

| Metric | What it tests | Direction |
|--------|---------------|-----------|
| **SC** structure consistency | re-extract the condition from the *generated* image at the same r, correlate with the given condition | ↑ |
| **EdgeF1** | Canny edge F1 vs reference, 2 px tolerance | ↑ |
| **LPIPS** | perceptual distance to reference | ↓ |
| **CLIP** | alignment with the image's own caption | ↑ |
| **Diversity** | mean pairwise LPIPS over K=4 samples of one condition | ↑ |
| **FID** | clean-FID, generated set vs reference set | ↓ |

SC is the metric the paper turns on: it is the only one that measures
obedience to the *given* condition rather than similarity to the ground truth,
so it stays meaningful at large r where the reference is barely recoverable.

Every number carries `{mean, std, sem, n}` and raw per-sample arrays
(`raw_r*.npz`), with checkpoint / seed / sampler settings / git commit in
`results.json:provenance`. Metrics that cannot be computed are recorded as
`null`, never as a stand-in.

## 5. Experiments

### E1 — Main result: the operating curve  *(running first)*
Train on D1, r ~ U(0,1). Evaluate at r ∈ {0, .05, .1, .2, .3, .5, .7, 1}.
**Headline figure:** FID vs SC as a parametric curve in r — our single model
traces the whole curve; every baseline is one point.

### E2 — Specialists  *(the experiment reviewers will demand)*
Train 4 separate adapters at fixed r ∈ {0.1, 0.3, 0.5, 0.7}, identical budget.
Compare each specialist at its own r against the E1 model at that r. This is
the direct test of C1. Report the gap with error bars; if a specialist wins
anywhere, say so and quantify it.

### E3 — Ablations (all on D1, shortened schedule)
| id | axis | values |
|----|------|--------|
| a | local window k | 1, 3, 5, 7, global cross-attn |
| b | adapter capacity `stem_channels` | 0, 64, 128 |
| c | downsampler | avg, conv |
| d | injection site | cross-attn only, + self-attn |
| e | conditioning colour | gray, color |
| f | cutoff sampler | uniform, sqrt, log |
| g | condition dropout | 0, 0.05, 0.1 |
| h | zero-initialised output gate | off (E1), on |

E3a is the architectural core: if global attention wins clearly, the
"locality is the right prior for structure" story is wrong and must be dropped.

E3h adds a zero-initialised per-channel gate on the adapter's output (+12,480
parameters, 113,544 total — still 3181× smaller than ControlNet). A per-channel
vector rather than a scalar, applied after the output projection: with a single
scalar at zero the projections behind it would receive no gradient at all.
Verified: the adapter is an exact no-op at init, one optimizer step lifts the
gate to 1e-4, and gradient then reaches all 84 tensors. Promoted from a routine
ablation to a priority run by the step-4,000 result above.

### E4 — Baselines
- **a. Native condition.** ControlNet-canny, ControlNet-hed, T2I-Adapter-sketch,
  on the same test images, captions, and seeds, with their own conditions.
- **b. Zero-shot matched condition.** Feed our frequency condition to the
  pretrained baselines. Expected to be unfavourable to them — reported as a
  transfer result, *not* as the headline comparison.
- **c. Trained matched condition (the fair one).** Train ControlNet from
  scratch on our frequency condition, same data and schedule. Expensive (361 M
  params) but this is the comparison that carries the parameter-efficiency
  claim. Budget one run.

### E5 — Cross-domain
E1 checkpoint evaluated zero-shot on D2 and D3; plus one full training run on
D3 to show the mechanism is not face-specific.

### E6 — Zero-shot real sketches (D4)
No ground truth, so: SC against the input sketch, CLIP, FID against the
training distribution, and qualitative panels.

### E7 — Control demonstrations
Fixed seed, sweep r → continuous morph from colourisation (r=0) to free
generation (r=1). Also r_train ≠ r_test robustness, and adapter `scale` as a
second, orthogonal dial at inference.

### E8 — Human study *(only if E1–E4 hold)*
Two-alternative forced choice on structure faithfulness and realism vs the
strongest baseline, ≥20 raters, ≥500 pairs.

## 6. Compute budget (1× H200, measured 28 img/s @ batch 8, 512 px)

| Block | Runs | Est. GPU-hours |
|-------|------|----------------|
| E1 | 1 | **6.6** (measured in production: 33.0 img/s steady state, 65,100 images, 12 epochs) |
| E2 | 4 | ~24 |
| E3 | ~12 | ~30 |
| E4c | 1 | ~12 |
| E5 | 1 + evals | ~10 |
| eval passes | — | ~10 |
| **total** | | **~95 GPU-h** (only the E1 row is a measured rate) |

Roughly four days of wall clock on one GPU. Throughput was benchmarked before launch at 32.3 / 33.2 / 33.5 img/s for batch
16 / 32 / 48 (peak 29.0 / 55.8 / 79.1 GB). It saturates — the job is compute-bound on
the UNet, not input-bound — so batch 32 is used: same speed as 48, more optimizer
steps, 87 GB of headroom.

The running job confirms it: **33.0 img/s at steady state**, matching the benchmark.
The first logging window read 27.2 img/s, which was startup — dataloader workers
spinning up and the NFS page cache filling — and it settled within a few hundred
steps. Only the steady-state figure is used for planning. Every other block's
estimate is still extrapolated and will be replaced as each run reports its own rate.

## 7. Risks, stated up front

1. **Capacity (C3).** 0.1 M params may simply be too few. E3b answers it; if
   so, the honest paper is "how little capacity suffices", with the curve of
   quality vs adapter size as the contribution.
2. **r=0 degeneracy.** With `frequency_img: gray`, r=0 is colourisation, not a
   copy — non-trivial, but it must be described as such rather than as
   "perfect reconstruction".
3. **Min-max normalisation** makes the condition's absolute scale
   uninformative, so the model must infer structure density from spatial
   statistics alone. This is what allows real sketches (which have no r) to
   work at all, but it should be stated as a design choice and ablated.
4. **Single seed.** Main results get 3 seeds before submission; ablations get 1.
5. **Baseline fairness.** E4b flatters us and will be labelled as transfer.
   E4c is the comparison that counts.

## 8. Ground rules

- No number appears in any table, plot, or message unless it came out of a
  script in this repo, on this machine, from a checkpoint that exists.
- Not-yet-run means **blank**, never estimated.
- Failed or unstable runs are reported as failed.
- Every result ships with its `results.json` provenance block.
