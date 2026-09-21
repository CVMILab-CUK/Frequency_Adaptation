#!/usr/bin/env python3
"""Emit every number the paper cites as LaTeX macros, straight from results/.

Nothing in the paper is typed by hand: if a run did not happen, its macro is
absent and the build fails loudly rather than printing a stale or invented
value.
"""
import json, os, statistics as st, sys

R = "results"
def load(p):
    f = os.path.join(R, p, "results.json")
    return json.load(open(f)) if os.path.exists(f) else None

out = []
def mac(name, val, fmt="{:.4f}"):
    out.append(f"\\newcommand{{\\{name}}}{{{fmt.format(val) if isinstance(val,float) else val}}}")

main = load("E10b")
if main is None:
    sys.exit("results/E10b missing -- refusing to emit paper numbers")
pc = main["per_cutoff"]
CUTS = ["0.0","0.05","0.1","0.2","0.3","0.5","0.7","1.0"]
key = {"0.0":"Zero","0.05":"ZeroFive","0.1":"OneZero","0.2":"TwoZero",
       "0.3":"ThreeZero","0.5":"FiveZero","0.7":"SevenZero","1.0":"OneHundred"}

for c in CUTS:
    v = pc[c]
    mac(f"scMain{key[c]}", v["SC"]["mean"])
    mac(f"semMain{key[c]}", v["SC"]["sem"])
    mac(f"divMain{key[c]}", v["Diversity"]["mean"])
    mac(f"lpipsMain{key[c]}", v["LPIPS"]["mean"])
    mac(f"clipMain{key[c]}", v["CLIP"]["mean"], "{:.2f}")
    mac(f"edgeMain{key[c]}", v["EdgeF1"]["mean"])
    if v.get("FID") is not None:
        mac(f"fidMain{key[c]}", v["FID"], "{:.2f}")

# seeds
seeds = [("E10b",2026),("E13_seed2027",2027),("E13_seed2028",2028)]
for c in ["0.05","0.1","0.2","0.3","0.5","0.7"]:
    vals=[load(p)["per_cutoff"][c]["SC"]["mean"] for p,_ in seeds if load(p)]
    if len(vals)>1:
        mac(f"seedMean{key[c]}", st.mean(vals))
        mac(f"seedStd{key[c]}", st.stdev(vals))

# ablations at r=0.1 and r=0.3
for tag, path in [("Ea","E9a"),("Ecap","E10a"),("Ecapb","E10d"),("Ewin","E10c"),
                  ("Ebase","E1"),("Egate","E3h"),("Eres","E9b")]:
    d = load(path)
    if not d: continue
    for c in ["0.05","0.1","0.3"]:
        if c in d["per_cutoff"]:
            mac(f"sc{tag}{key[c]}", d["per_cutoff"][c]["SC"]["mean"])
    if "0.1" in d["per_cutoff"] and d["per_cutoff"]["0.1"].get("FID"):
        mac(f"fid{tag}", d["per_cutoff"]["0.1"]["FID"], "{:.2f}")

# injection-path decomposition
for tag, path in [("Both","E10b"),("SkipOnly","E10b_adapter_off"),
                  ("AttnOnly","E10b_skip_off"),("AllOff","E10b_all_off")]:
    d = load(path)
    if not d: continue
    for c in ["0.0","0.1","0.3"]:
        if c in d["per_cutoff"]:
            mac(f"path{tag}{key[c]}", d["per_cutoff"][c]["SC"]["mean"])
    if "0.0" in d["per_cutoff"] and d["per_cutoff"]["0.0"].get("FID"):
        mac(f"pathFid{tag}", d["per_cutoff"]["0.0"]["FID"], "{:.2f}")

# strength dial vs frequency dial
for s,nm in [("0.25","QuarterScale"),("0.5","HalfScale"),("0.75","ThreeQScale")]:
    d = load(f"E12_scale{s}")
    if not d: continue
    v = d["per_cutoff"]["0.1"]
    mac(f"sc{nm}", v["SC"]["mean"]); mac(f"div{nm}", v["Diversity"]["mean"])
    if v.get("FID"): mac(f"fid{nm}", v["FID"], "{:.2f}")

# specialists
for r in ["0.1","0.3","0.5","0.7"]:
    d = load(f"E2_r{r}")
    if not d: continue
    s = d["per_cutoff"][r]["SC"]; g = pc[r]["SC"]
    mac(f"spec{key[r]}", s["mean"])
    mac(f"specZ{key[r]}", (s["mean"]-g["mean"])/((s["sem"]**2+g["sem"]**2)**0.5), "{:+.1f}")

# baselines
for b in ["controlnet_canny","controlnet_hed"]:
    p = os.path.join(R, "baselines", f"{b}_native", "results.json")
    if not os.path.exists(p): continue
    d = json.load(open(p)); v = d["per_setting"]["native"]
    nm = "Canny" if "canny" in b else "Hed"
    mac(f"base{nm}Edge", v["EdgeF1"]["mean"]); mac(f"base{nm}Lpips", v["LPIPS"]["mean"])
    mac(f"base{nm}Clip", v["CLIP"]["mean"], "{:.2f}")
    if v.get("FID"): mac(f"base{nm}Fid", v["FID"], "{:.2f}")
    mac("baseParams", f"{d['provenance']['controlnet_params']:,}")

# VAE survival
for tag, f in [("Native","vae_survival_512.json"),("Up","vae_survival_256up.json")]:
    p = os.path.join(R, f)
    if not os.path.exists(p): continue
    k = json.load(open(p))["kept"]
    for c in ["0.1","0.3","0.7"]:
        if c in k: mac(f"vae{tag}{key[c]}", k[c]["mean"])

# sketch zero-shot
for tag, path in [("Face","E6_face"),("Scene","E6_scene")]:
    d = load(path)
    if not d: continue
    mac(f"sketch{tag}SC", d["SC_vs_input_sketch"]["mean"])
    mac(f"sketch{tag}Clip", d["CLIP"]["mean"], "{:.2f}")
    mac(f"sketch{tag}N", d["provenance"]["n_images"])
    if d.get("FID_vs_FFHQ") is not None:
        mac(f"sketch{tag}Fid", d["FID_vs_FFHQ"], "{:.2f}")

# the scanned filename annotation on the forensic drawings: how much of the
# face SC is agreement on that band rather than on the face (scripts/sketch_band_check.py)
_bcf = os.path.join(R, "E6_face", "band_check.json")
_bc = json.load(open(_bcf)) if os.path.exists(_bcf) else None
if _bc:
    mac("bandFrac",  int(round(_bc["mask_frac"] * 100)))
    mac("bandRows",  _bc["rows_removed"])
    mac("bandSCcut", _bc["SC_band_removed"]["mean"])
    mac("bandDelta", _bc["delta"], "{:+.4f}")
    mac("bandSem",   abs(_bc["delta"]) / _bc["SC_full_frame"]["sem"], "{:.2f}")

# --- E14/E15: one-field ablations rebuilt on the working model (single seed) -
ABL = [("Kone","E14a_k1"),("Kfive","E14a_k5"),("Stemsf","E14b_stem64"),
       ("Davg","E14c_down_avg"),("Color","E14e_color"),("Sqrt","E14f_sqrt"),
       ("Log","E14f_log"),("Drop","E14g_conddrop0"),("Nogate","E14h_nogate"),
       ("Stemskip","E15_stem256_skip")]
for tag, path in ABL:
    d = load(path)
    if not d: continue
    for c in ["0.0","0.05","0.1","0.3","0.5"]:
        v = d["per_cutoff"].get(c)
        if not v: continue
        mac(f"abl{tag}{key[c]}", v["SC"]["mean"])
        if v.get("FID") is not None:
            mac(f"ablFid{tag}{key[c]}", v["FID"], "{:.2f}")

# --- D-06: strength scaling on the fixed-cutoff specialist (E16) -------------
for s_, nm in [("0.25","Quarter"),("0.5","Half"),("0.75","ThreeQ"),("1.0","One")]:
    d = load(f"E16_spec_scale{s_}")
    if not d: continue
    v = d["per_cutoff"]["0.1"]
    mac(f"sscSC{nm}", v["SC"]["mean"]); mac(f"sscDiv{nm}", v["Diversity"]["mean"])
    if v.get("FID") is not None: mac(f"sscFid{nm}", v["FID"], "{:.2f}")

# --- D-16: one generic caption for every image (E16) -------------------------
d = load("E16_generic_caption")
if d:
    for c in ["0.0","0.1","0.3"]:
        v = d["per_cutoff"].get(c)
        if not v: continue
        mac(f"genSC{key[c]}", v["SC"]["mean"])
        mac(f"genClip{key[c]}", v["CLIP"]["mean"], "{:.2f}")
        if v.get("FID") is not None: mac(f"genFid{key[c]}", v["FID"], "{:.2f}")

# --- D-17: the dial on hand drawings, filtered by the artist at r (E17) -------
_sd = os.path.join(R, "E17_sketch_dial", "results.json")
if os.path.exists(_sd):
    d = json.load(open(_sd))
    for c in CUTS:
        v = d["per_cutoff"].get(c)
        if not v: continue
        mac(f"sdial{key[c]}", v["SC_vs_drawing"]["mean"])
        mac(f"sdialSem{key[c]}", v["SC_vs_drawing"]["sem"])
        mac(f"sdialF{key[c]}", v["SC_vs_filtered"]["mean"])
    mac("sdialN", d["provenance"]["n_images"])

# --- D-07: T2I-Adapter, the size-matched side-network baseline (E17b) --------
# Only the canny checkpoint is reported: it received its native condition. The
# sketch checkpoint expects a PiDiNet sketch and was given Canny edges, so its
# numbers measure a condition mismatch, not the method.
_t = os.path.join(R, "baselines", "t2iadapter_canny_native", "results.json")
if os.path.exists(_t):
    d = json.load(open(_t)); v = d["per_setting"]["native"]
    mac("baseTEdge", v["EdgeF1"]["mean"]); mac("baseTLpips", v["LPIPS"]["mean"])
    mac("baseTClip", v["CLIP"]["mean"], "{:.2f}")
    if v.get("FID") is not None: mac("baseTFid", v["FID"], "{:.2f}")
    mac("baseTParams", f"{d['provenance']['side_params']:,}")

# model sizes
m = json.load(open("ckpt_dir/E10b_skip_inject/run_manifest.json"))
mac("ourParams", f"{m['trainable_params']:,}")
mac("ourParamsRatio", f"{361279120/m['trainable_params']:.0f}")
mac("nTrain", f"{m['n_train']:,}"); mac("nTest", f"{m['n_test']:,}")
mac("nValid", f"{m['n_valid']:,}")

# --- seed-level statistics and the corrected tests (review D-05) ------------
# The paper previously printed single-seed z values with no test named. These
# are recomputed across the three seeds and carry a stated test.
import math
SEEDS = ["E10b", "E13_seed2027", "E13_seed2028"]
def seed_vals(c):
    return [load(p)["per_cutoff"][c]["SC"]["mean"] for p in SEEDS if load(p)]

for c in CUTS:
    v = seed_vals(c)
    if len(v) > 1:
        m = sum(v) / len(v)
        sd = (sum((x - m) ** 2 for x in v) / (len(v) - 1)) ** 0.5
        mac(f"sMean{key[c]}", m); mac(f"sSd{key[c]}", sd)

# r=0 -> r=0.05 is a rise; state it with the paired within-checkpoint SE
a, b = pc["0.0"]["SC"], pc["0.05"]["SC"]
se = (a["sem"] ** 2 + b["sem"] ** 2) ** 0.5
mac("riseDelta", b["mean"] - a["mean"]); mac("riseZ", (b["mean"] - a["mean"]) / se, "{:.2f}")

# r=0.5 vs r=0.7 across seeds: paired t, df=2
d5, d7 = seed_vals("0.5"), seed_vals("0.7")
if len(d5) == len(d7) > 1:
    dd = [x - y for x, y in zip(d5, d7)]
    m = sum(dd) / len(dd)
    sd = (sum((x - m) ** 2 for x in dd) / (len(dd) - 1)) ** 0.5
    mac("tailT", m / (sd / math.sqrt(len(dd))), "{:.3f}")
    mac("tailCrit", "4.303")

# specialists, re-expressed against the three-seed spread of the single model
for r in ["0.1", "0.3", "0.5", "0.7"]:
    d = load(f"E2_r{r}")
    if not d: continue
    sp = d["per_cutoff"][r]["SC"]["mean"]
    v = seed_vals(r)
    m = sum(v) / len(v)
    sd = (sum((x - m) ** 2 for x in v) / (len(v) - 1)) ** 0.5
    mac(f"specD{key[r]}", sp - m, "{:+.4f}")
    mac(f"specR{key[r]}", (sp - m) / sd, "{:+.2f}")

# Diversity is nearly LPIPS shifted by a constant -- state the constant
diffs = [pc[c]["LPIPS"]["mean"] - pc[c]["Diversity"]["mean"] for c in CUTS]
md = sum(diffs) / len(diffs)
mac("divGap", md); mac("divGapSd", (sum((x - md) ** 2 for x in diffs) / (len(diffs) - 1)) ** 0.5)


# ---- E18a: the strength sweep filled in between the E12 points
SCALE_KEY = {"0.1":"A","0.2":"B","0.25":"C","0.35":"D","0.5":"E","0.6":"F","0.75":"G"}
for sc_, k in SCALE_KEY.items():
    for src, suffix in ((f"E12_scale{sc_}", ""), (f"E18a_scale{sc_}", "")):
        d = load(src)
        if d and "0.1" in d["per_cutoff"]:
            v = d["per_cutoff"]["0.1"]
            mac(f"sw{k}Sc", v["SC"]["mean"])
            if v.get("FID") is not None:
                mac(f"sw{k}Fid", v["FID"], "{:.2f}")
            mac(f"sw{k}Val", sc_)
            break
# seed-2027 replicas of the same sweep, where they exist
for sc_, k in [("0.25","C"), ("0.5","E"), ("0.75","G")]:
    d = load(f"E18a_scale{sc_}_s2027")
    if d and "0.1" in d["per_cutoff"]:
        mac(f"sw{k}ScB", d["per_cutoff"]["0.1"]["SC"]["mean"])

# ---- E18: the skip path trained alone, which separates placement from capacity
d = load("E18_skip_only")
if d:
    for c in ["0.0", "0.05", "0.1", "0.2", "0.3", "0.5", "0.7", "1.0"]:
        if c in d["per_cutoff"]:
            mac(f"skipOnly{key[c]}", d["per_cutoff"][c]["SC"]["mean"])
            if d["per_cutoff"][c].get("FID") is not None:
                mac(f"skipOnlyFid{key[c]}", d["per_cutoff"][c]["FID"], "{:.2f}")

# ---- E21: does the dial survive outside the training domain?
for tag, path in [("Ood", "E21_ood_photos"), ("Ind", "E21_indomain_photos")]:
    d = load(path)
    if not d:
        continue
    for c in ["0.05", "0.1", "0.2", "0.3"]:
        v = d["per_cutoff"].get(c)
        if v and v.get("SC_vs_drawing"):
            mac(f"ood{tag}{key[c]}", v["SC_vs_drawing"]["mean"])
            if v.get("FID") is not None:
                mac(f"ood{tag}Fid{key[c]}", v["FID"], "{:.2f}")
d = load("E21_ood_photos_off")
if d and "0.1" in d["per_cutoff"]:
    mac("oodOffOneZero", d["per_cutoff"]["0.1"]["SC_vs_drawing"]["mean"])
# The pre-registered quantity in results/DECISIONS.md is the total decline from
# r=0.05 to r=0.3, not a regression slope. Emitting an OLS slope here would put
# a second, silently different definition of the same claim into the paper.
def _drop(path):
    dd = load(path)
    if not dd:
        return None
    a = dd["per_cutoff"].get("0.05", {}).get("SC_vs_drawing")
    b = dd["per_cutoff"].get("0.3", {}).get("SC_vs_drawing")
    return a["mean"] - b["mean"] if a and b else None
so, si = _drop("E21_ood_photos"), _drop("E21_indomain_photos")
if so and si:
    mac("oodDrop", so); mac("oodDropIn", si); mac("oodDropRatio", so/si, "{:.3f}")

# ---- E19: the fixed-instrument reading of the dial
d = None
f19 = os.path.join(R, "E19_fixed_instrument", "results.json")
if os.path.exists(f19):
    d = json.load(open(f19))
if d:
    for c, k in [("0.05", "ZeroFive"), ("0.1", "OneZero"), ("0.3", "ThreeZero")]:
        col = d.get(f"E10b/gen_r{c}")
        if col and "sc_fix@0.1" in col:
            mac(f"fixInst{k}", col["sc_fix@0.1"]["mean"])

    col = d.get("E10b/gen_r0.2")
    if col and "sc_fix@0.1" in col:
        mac("fixInstTwoZero", col["sc_fix@0.1"]["mean"])
    # the unrelated-image anchor for SC: each generation against a deranged
    # reference, at the cutoff it was generated with
    for c, k, rm in [("0.0", "Zero", "0.0"), ("0.1", "OneZero", "0.1")]:
        col = d.get(f"E10b/gen_r{c}")
        if col and f"sc_fix@{rm}_floor" in col:
            mac(f"scFloor{k}", col[f"sc_fix@{rm}_floor"]["mean"])

# ---- review 2026-09-14 (T17): the no-adapter control at the cutoffs Table
# tab:where prints, not the r=0 value in every column
d = load("E10b_all_off_r0.05")
if d and "0.05" in d["per_cutoff"]:
    mac("pathAllOffZeroFive", d["per_cutoff"]["0.05"]["SC"]["mean"])
d = load("E10b_all_off")
if d and d["per_cutoff"].get("0.1", {}).get("FID") is not None:
    mac("pathFidAllOffOneZero", d["per_cutoff"]["0.1"]["FID"], "{:.2f}")

# ---- E19b: the strength sweeps read with the same fixed instrument (r_m=0.1)
f19b = os.path.join(R, "E19b_strength_fixed_instrument", "results.json")
if os.path.exists(f19b):
    d = json.load(open(f19b))
    for run, nm in [("E12_scale0.25", "QuarterScale"), ("E12_scale0.5", "HalfScale"),
                    ("E12_scale0.75", "ThreeQScale"), ("E16_spec_scale0.25", "SscQuarter"),
                    ("E16_spec_scale0.5", "SscHalf"), ("E16_spec_scale0.75", "SscThreeQ"),
                    ("E16_spec_scale1.0", "SscOne")]:
        col = d.get(f"{run}/gen_r0.1")
        if col and "sc_fix@0.1" in col:
            mac(f"fixInst{nm}", col["sc_fix@0.1"]["mean"])

# ---- trainable parameter counts per design, from each run's manifest (T18)
for nm, run in [("SkipOnly", "E18_skip_only"), ("AttnOnly", "E9a_conv_encoder"),
                ("Cap", "E10a_capacity128"), ("Capb", "E10d_capacity256"),
                ("Stemsf", "E14b_stem64"), ("Stemskip", "E15_stem256_skip")]:
    p = os.path.join("ckpt_dir", run, "run_manifest.json")
    if os.path.exists(p):
        mac(f"params{nm}", f"{json.load(open(p))['trainable_params']:,}")

# ---- E24: what the paper cost, from the training logs and a timed sampler
d = load("E24_compute_budget")
if d:
    mac("trainHoursMain", d["train_hours_main"], "{:.1f}")
    mac("trainPeakGb", d["train_peak_gb"], "{:.0f}")
    mac("trainImgSec", d["train_images_per_sec"], "{:.0f}")
    mac("trainHoursAll", d["train_hours_total"], "{:.0f}")
    mac("nTrainRuns", d["provenance"]["n_runs"])
    mac("sampleSecImg", d["sampling_seconds_per_image"], "{:.2f}")
    mac("samplePeakGb", d["sampling_peak_gb"], "{:.1f}")
    mac("sampleImages", f'{d["generated_images_saved"] + d["diversity_samples_unsaved"]:,}')
    mac("sampleHoursAll", d["sampling_hours_total_est"], "{:.0f}")
    mac("gpuHoursAll", d["gpu_hours_total_est"], "{:.0f}")

# ---- E22: how fast the condition empties, measured instead of typed
d = load("E22_spectral_energy")
if d:
    # non-constant energy only: the DC term dominates |F|^2 and the disc
    # removes it at every r, which would inflate the fraction removed
    mac("acRemovedZeroFive", 100 * d["ac_removed@0.05"]["mean"], "{:.1f}")
    mac("acRetainedFiveZero", d["ac_retained@0.5"]["mean"])
    mac("condStdZero", d["cond_std@0.0"]["mean"], "{:.3f}")
    mac("condStdOneHundred", d["cond_std@1.0"]["mean"], "{:.3f}")
    mac("energyN", d["provenance"]["n_images"])

# ---- the span of Diversity over the sweep, which Sec. Metrics quotes
_dv = [pc[c]["Diversity"]["mean"] for c in CUTS]
mac("divRange", max(_dv) - min(_dv), "{:.2f}")

# ---- E21 protocol details the OOD section states
for tag, path in [("Ood", "E21_ood_photos"), ("Ind", "E21_indomain_photos")]:
    d = load(path)
    if not d:
        continue
    if tag == "Ood":
        mac("oodN", d["provenance"]["n_images"])
    for c in ["0.05", "0.1", "0.2", "0.3"]:
        v = d["per_cutoff"].get(c)
        if v and v.get("SC_vs_filtered"):
            mac(f"ood{tag}F{key[c]}", v["SC_vs_filtered"]["mean"])
d = load("E21_ood_photos_off")
if d and "0.1" in d["per_cutoff"]:
    mac("oodOffFidOneZero", d["per_cutoff"]["0.1"]["FID"], "{:.2f}")

os.makedirs("papers", exist_ok=True)
with open("papers/numbers.tex","w") as f:
    f.write("% generated by scripts/paper_numbers.py -- do not edit\n")
    f.write("\n".join(sorted(set(out))) + "\n")
print(f"papers/numbers.tex: {len(set(out))} macros from measured results")
