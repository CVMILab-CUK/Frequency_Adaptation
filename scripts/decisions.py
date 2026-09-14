#!/usr/bin/env python3
"""Apply the pre-registered decision rules of ralplan v3 mechanically.

Thresholds are constants fixed before the runs they judge finished (see the
plan). Where a constant was derived from existing results, the script recomputes
it and refuses to run if the files no longer agree. It reads results/ only and
never edits the paper. Usage: decisions.py {s1,s2,s3,s4,s5,s6,t2,all}
"""
import json, math, os, sys
R = "results"
def J(p):
    p = os.path.join(R, p)
    return json.load(open(p)) if os.path.exists(p) else None
def rj(run): return J(f"{run}/results.json")
def sdv(v):
    m = sum(v) / len(v); return (sum((x - m) ** 2 for x in v) / (len(v) - 1)) ** 0.5

# ---- constants frozen in ralplan v3 ------------------------------------------
M_SC01, SD_SC01 = 0.62452, 0.005928
PI1, PI2 = 0.02945, 0.02329
LO_I, HI_I = 0.5951, 0.6540
E10D_SC01 = 0.2158
LO_III = 0.2658
FID01_MEAN, FID01_SD = 54.11, 0.809
SPEC_FID_BOUND = 50.09
DIAL_R = ["0.05", "0.1", "0.15", "0.2", "0.25", "0.3", "0.5", "0.7"]

out = []
def say(s): print(s); out.append(s)

def check_constants():
    v = [rj(r)["per_cutoff"]["0.1"]["SC"]["mean"] for r in ("E10b", "E13_seed2027", "E13_seed2028")]
    f = [rj(r)["per_cutoff"]["0.1"]["FID"] for r in ("E10b", "E13_seed2027", "E13_seed2028")]
    m, s = sum(v) / 3, sdv(v)
    assert abs(m - M_SC01) < 5e-5 and abs(s - SD_SC01) < 5e-6, (m, s)
    assert abs(4.303 * math.sqrt(4 / 3) * s - PI1) < 5e-5
    assert abs(sum(f) / 3 - FID01_MEAN) < 5e-3 and abs(sdv(f) - FID01_SD) < 5e-4, (sum(f) / 3, sdv(f))
    assert abs(rj("E10d")["per_cutoff"]["0.1"]["SC"]["mean"] - E10D_SC01) < 5e-5
    assert abs((FID01_MEAN - 4.303 * math.sqrt(4 / 3) * FID01_SD) - SPEC_FID_BOUND) < 5e-3

SCALE_RUNS = ["E18a_scale0.1", "E18a_scale0.2", "E12_scale0.25", "E18a_scale0.35", "E12_scale0.5", "E18a_scale0.6", "E12_scale0.75"]
S1_VERDICTS = {}

def s1_delta():
    base = rj("E10b"); d = []
    for a_, b_ in [("E12_scale0.25", "E18a_scale0.25_s2027"), ("E12_scale0.5", "E18a_scale0.5_s2027"),
                   ("E12_scale0.75", "E18a_scale0.75_s2027")]:
        A, B = rj(a_), rj(b_)
        if A and B: d.append(A["per_cutoff"]["0.1"]["FID"] - B["per_cutoff"]["0.1"]["FID"])
    D2 = rj("E18a_dial_s2027")
    if D2:
        for r in ("0.2", "0.3"): d.append(base["per_cutoff"][r]["FID"] - D2["per_cutoff"][r]["FID"])
    sd_samp = math.sqrt(sum(x * x for x in d) / len(d) / 2) if d else 0.0
    return 2 * max(FID01_SD, sd_samp), sd_samp, len(d)

def pipeline_ok():
    """E18a runs used the edited pipeline; E10b/E12 did not. Its seed-2027 dial
    points must land within the three training seeds' range (+-0.03)."""
    D2 = rj("E18a_dial_s2027")
    if not D2: return False, "E18a_dial_s2027 missing"
    msgs, ok = [], True
    for r in ("0.2", "0.3"):
        v = [rj(x)["per_cutoff"][r]["SC"]["mean"] for x in ("E10b", "E13_seed2027", "E13_seed2028")]
        got = D2["per_cutoff"][r]["SC"]["mean"]
        inside = min(v) - 0.03 <= got <= max(v) + 0.03; ok &= inside
        msgs.append(f"r={r}: {got:.4f} vs seeds [{min(v):.4f}, {max(v):.4f}] +-0.03 -> {'ok' if inside else 'OUTSIDE'}")
    order = [("E18a_scale0.1", 0.1), ("E18a_scale0.2", 0.2), ("E12_scale0.25", 0.25), ("E18a_scale0.35", 0.35),
             ("E12_scale0.5", 0.5), ("E18a_scale0.6", 0.6), ("E12_scale0.75", 0.75)]
    got = [(s_, rj(r)["per_cutoff"]["0.1"]["SC"]["mean"]) for r, s_ in order if rj(r)]
    mono = len(got) == len(order) and all(a[1] < b[1] for a, b in zip(got, got[1:]))
    ok &= mono
    msgs.append(f"scale SC monotone in s across E12/E18a ({len(got)}/{len(order)} present): {mono}")
    return ok, "; ".join(msgs)

def judge(label, curve, points, delta):
    """curve: [(r, sc, fid)] in r order; points: {run: (sc, fid)}."""
    sc = [c[1] for c in curve]
    if not all(x > y for x, y in zip(sc, sc[1:])):
        say(f"[{label}] dial not monotone in r -> not evaluable"); return {}
    say(f"[{label}] dial curve: " + ", ".join(f"({r}, {s_:.4f}, {f_:.2f})" for r, s_, f_ in curve))
    res = {}
    lo_sc, lo_fid, hi_sc = curve[-1][1], curve[-1][2], curve[0][1]
    for run, (s_, f_) in points.items():
        if lo_sc <= s_ <= hi_sc:
            for (r0, s0, f0), (r1, s1_, f1) in zip(curve, curve[1:]):
                if s1_ <= s_ <= s0:
                    fi = f1 + (f0 - f1) * (s_ - s1_) / (s0 - s1_); break
            diff = f_ - fi
            v = "dial" if diff > delta else "strength" if diff < -delta else "none"
            say(f"  [{label}] {run}: SC {s_:.4f} FID {f_:.2f}; dial FID at same SC {fi:.2f}; diff {diff:+.2f} -> "
                + {"dial": "dial lower FID", "strength": "strength lower FID", "none": "not separated"}[v])
        elif s_ < lo_sc:
            v = "dial" if f_ > lo_fid + delta else "nc"
            say(f"  [{label}] {run}: SC {s_:.4f} below dial range; FID {f_:.2f} vs {lo_fid:.2f}+{delta:.2f} -> "
                + ("dial dominates" if v == "dial" else "not comparable"))
        else:
            v = "nc"; say(f"  [{label}] {run}: SC {s_:.4f} above dial range -> not comparable")
        res[run] = v
    return res

def s1():
    say("## S1 C2 matched curve (strength vs cutoff)")
    ok, msg = pipeline_ok(); say(f"pipeline regression check: {msg}")
    if not ok: say("-> S1 not evaluable (pipeline mismatch)"); return
    delta, sd_samp, n = s1_delta()
    say(f"sampling FID sd from {n} seed pairs = {sd_samp:.3f}; delta = {delta:.3f}")
    base, extra = rj("E10b"), rj("E18a_dial_s2026")
    pc = dict(base["per_cutoff"]); pc.update(extra["per_cutoff"] if extra else {})
    if any(r not in pc for r in DIAL_R): say("dial points missing -> not evaluable"); return
    curve = [(float(r), pc[r]["SC"]["mean"], pc[r]["FID"]) for r in DIAL_R]
    pts = {run: (rj(run)["per_cutoff"]["0.1"]["SC"]["mean"], rj(run)["per_cutoff"]["0.1"]["FID"]) for run in SCALE_RUNS if rj(run)}
    S1_VERDICTS["moving"] = judge("moving instrument, pre-registered v3", curve, pts, delta)
    # co-primary (Architect iteration 2): the same rule on SC measured at a fixed
    # cutoff, floor-subtracted, from E19. Primary r_m=0.1; 0.05 and 0.3 robustness.
    E = J("E19_fixed_instrument/results.json")
    if not E: say("[fixed instrument] E19 missing -> not evaluable"); return
    def key_of(r):
        return f"E18a_dial_s2026/gen_r{r}" if r in ("0.15", "0.25") else f"E10b/gen_r{r}"
    for rm in ("0.1", "0.05", "0.3"):
        m = f"sc_fix@{rm}"
        try:
            fc = [(float(r), E[key_of(r)][m]["mean"] - E[key_of(r)][m + "_floor"]["mean"], pc[r]["FID"]) for r in DIAL_R]
            fp = {run: (E[f"{run}/gen_r0.1"][m]["mean"] - E[f"{run}/gen_r0.1"][m + "_floor"]["mean"],
                        rj(run)["per_cutoff"]["0.1"]["FID"]) for run in pts if f"{run}/gen_r0.1" in E}
        except KeyError as ex:
            say(f"[fixed r_m={rm}] missing {ex} -> not evaluable"); continue
        S1_VERDICTS[f"fixed{rm}"] = judge(f"fixed r_m={rm}" + (" (co-primary)" if rm == "0.1" else " (robustness)"), fc, fp, delta)
    # Pre-registered write-up per point (Critic iteration 2):
    #   both 'dial'                 -> C2 sentence allowed for this point
    #   exactly one 'dial'          -> "instrument-dependent": both values in the table, no C2 sentence
    #   either says 'strength'      -> must be reported in the main text
    #   r_m 0.05 / 0.3 disagree     -> footnote
    #   point absent from E19       -> printed as unscored, never silently dropped
    mv, fx = S1_VERDICTS.get("moving", {}), S1_VERDICTS.get("fixed0.1", {})
    for run in SCALE_RUNS:
        if run not in mv: say(f"  write-up {run}: no moving verdict (missing run)"); continue
        if run not in fx: say(f"  write-up {run}: unscored by the fixed instrument"); continue
        a_, b_ = mv[run], fx[run]
        if "strength" in (a_, b_): w = "strength favoured on at least one instrument: report in the main text"
        elif a_ == b_ == "dial": w = "C2 sentence allowed"
        elif "dial" in (a_, b_): w = "instrument-dependent: report both values, no C2 sentence for this point"
        else: w = "no C2 claim for this point"
        rob = [S1_VERDICTS.get(f"fixed{rm}", {}).get(run) for rm in ("0.05", "0.3")]
        foot = " | robustness disagreement -> footnote" if any(v is not None and v != b_ for v in rob) else ""
        say(f"  write-up {run}: moving={a_}, fixed0.1={b_} -> {w}{foot}")

def s2():
    say("## S2 VAE survival")
    base = J("vae_survival_sd15_n500.json")
    if not base: say("sd15 n500 missing -> not evaluable"); return
    b = base["kept"]
    for tag in ["sdxl", "flux16"]:
        d = J(f"vae_survival_{tag}_n500.json")
        if not d: say(f"- {tag}: missing (excluded)"); continue
        k = d["kept"]; d1 = k["0.1"]["mean"] - b["0.1"]["mean"]; d3 = k["0.3"]["mean"] - b["0.3"]["mean"]
        if tag == "flux16":
            v = ("narrow C3 to 4-channel SD-family VAEs" if d1 > 0.1 and d3 > 0.1 else
                 "generalise C3 to the 16-channel VAE" if d1 <= 0.1 and d3 <= 0.1 else "graded statement")
        else:
            v = ("C3 specific to the SD1.5 VAE (SDXL's 4-channel VAE preserves more)" if d1 > 0.1 and d3 > 0.1 else
                 "C3 holds for the SDXL VAE too" if d1 <= 0.1 and d3 <= 0.1 else "graded statement")
        p = d["provenance"]
        say(f"- {tag} ({p.get('vae_id')}/{p.get('subfolder')}, {p.get('latent_channels')}ch, n={p.get('n')}): "
            f"r=0.1 {k['0.1']['mean']:.4f} ({d1:+.4f}), r=0.3 {k['0.3']['mean']:.4f} ({d3:+.4f}) -> {v}")
    say(f"  sd15 reference: r=0.1 {b['0.1']['mean']:.4f}, r=0.3 {b['0.3']['mean']:.4f}")

def s3():
    say("## S3 out-of-domain photos (SC_vs_drawing, same script/prompt/cutoffs)")
    O, I = rj("E21_ood_photos"), rj("E21_indomain_photos")
    if not O or not I: say(f"missing: ood={bool(O)} in-domain={bool(I)} -> not evaluable"); return
    cs = ["0.05", "0.1", "0.2", "0.3"]
    def curve(X): return [(X["per_cutoff"][c]["SC_vs_drawing"]["mean"], X["per_cutoff"][c]["SC_vs_drawing"]["sem"]) for c in cs]
    o, i = curve(O), curve(I)
    mono = all(not (y[0] > x[0] + 2 * math.sqrt(x[1] ** 2 + y[1] ** 2)) for x, y in zip(o, o[1:]))
    slope_o, slope_i = o[0][0] - o[-1][0], i[0][0] - i[-1][0]
    sem_c = math.sqrt(o[0][1] ** 2 + o[-1][1] ** 2)
    ok = mono and slope_o >= 0.5 * slope_i and slope_o > 3 * sem_c
    say(f"OOD {[round(x[0],4) for x in o]}  in-domain {[round(x[0],4) for x in i]}")
    say(f"slope OOD {slope_o:.4f}, in-domain {slope_i:.4f}, ratio {slope_o/slope_i if slope_i else float('nan'):.3f}; "
        f"3*sem {3*sem_c:.4f}; monotone(2 sem tol) {mono} -> {'dial carries out of domain' if ok else 'weak transfer: report in Limitations'}")
    C = rj("E21_ood_photos_off")
    if C: say(f"  descriptive: adapter off at r=0.1 SC_vs_drawing {C['per_cutoff']['0.1']['SC_vs_drawing']['mean']:.4f}")

def case_of(x, lo, hi):
    return "0" if x > hi else "i" if x >= lo else "ii" if x > LO_III else "iii"
WORD = {"0": "skip-only higher; attention-path effect not separable from the absent structure dropout (cf. E14g_conddrop0)",
        "i": "not separated at this number of seeds",
        "ii": "paths complementary; placement dominates at matched capacity",
        "iii": "co-adaptation: weaken the placement sentence, state in Limitations"}

def s4():
    say("## S4 E18 skip-only trained from scratch")
    d = rj("E18_skip_only")
    if not d: say("E18 missing (excluded)"); return
    x = d["per_cutoff"]["0.1"]["SC"]["mean"]
    c = case_of(x, LO_I, HI_I)
    say(f"E18 SC@0.1 {x:.4f} (FID {d['per_cutoff']['0.1']['FID']}); bounds {LO_I}/{HI_I}/{LO_III} -> case ({c}): {WORD[c]}")

def t2():
    d = rj("E18_skip_only")
    if not d: print("SPEC2"); return
    x = d["per_cutoff"]["0.1"]["SC"]["mean"]
    near = min(abs(x - b) for b in (LO_I, HI_I, LO_III)) <= SD_SC01
    print("SEED2" if near else "SPEC2")

def s6():
    say("## S6 conditional slot")
    a, b = rj("E18_skip_only"), rj("E18_skip_only_s2027")
    if a and b:
        x = (a["per_cutoff"]["0.1"]["SC"]["mean"] + b["per_cutoff"]["0.1"]["SC"]["mean"]) / 2
        c = case_of(x, M_SC01 - PI2, M_SC01 + PI2)
        say(f"E18 two-seed mean SC@0.1 {x:.4f}; bounds {M_SC01-PI2:.4f}/{M_SC01+PI2:.4f}/{LO_III} -> case ({c}): {WORD[c]}")
    e = rj("E2_fixed_r0.1_s2027")
    if e:
        v = e["per_cutoff"]["0.1"]; s_, f_ = v["SC"]["mean"], v["FID"]
        r = ("specialist advantage replicated" if s_ > HI_I and f_ < SPEC_FID_BOUND else
             "replicated on SC only" if s_ > HI_I else
             "replicated on FID only" if f_ < SPEC_FID_BOUND else "advantage in one of two seeds only")
        say(f"E2_r0.1 seed 2027: SC@0.1 {s_:.4f}, FID {f_:.2f} (bounds SC>{HI_I}, FID<{SPEC_FID_BOUND}) -> {r}")
    if not (a and b) and not e: say("no S6 result (excluded)")

def s5():
    say("## S5 band selectivity (fixed instrument)")
    E = J("E19_fixed_instrument/results.json")
    if not E: say("E19 missing -> not evaluable"); return
    def valid_bands(anchor):
        vb = []
        for k in range(10):
            c = E[anchor][f"band{k}"]; den = c["mean"] - E[anchor][f"band{k}_floor"]["mean"]
            if den >= 0.05 and den >= 3 * c["sem"]: vb.append(k)
        return vb
    def A(key, k, anchor):
        den = E[anchor][f"band{k}"]["mean"] - E[anchor][f"band{k}_floor"]["mean"]
        return max(-0.5, min(1.5, (E[key][f"band{k}"]["mean"] - E[key][f"band{k}_floor"]["mean"]) / den))
    def dial_gap(run):
        anc, k1, k3 = f"{run}/gen_r0.0", f"{run}/gen_r0.1", f"{run}/gen_r0.3"
        if not all(k in E for k in (anc, k1, k3)): return None, "missing"
        vb = valid_bands(anc); lo = [k for k in vb if 1 <= k <= 2]; hi = [k for k in vb if k >= 3]
        if len(lo) < 2 or len(hi) < 2: return None, f"not evaluable (valid bands {vb})"
        dl = {k: A(k3, k, anc) - A(k1, k, anc) for k in vb}
        return sum(dl[k] for k in lo) / len(lo) - sum(dl[k] for k in hi) / len(hi), f"valid bands {vb}"
    # dial: primary on E10b (its values were seen in the 32-image smoke), confirmed
    # on the two training seeds that no smoke touched
    gaps = {}
    for run in ("E10b", "E13_seed2027", "E13_seed2028"):
        g, note = dial_gap(run); gaps[run] = g
        say(f"- dial r=0.3 vs r=0.1, {run}: gap {('%+.3f' % g) if g is not None else '-'} ({note})")
    if gaps["E10b"] is None: dial = "not evaluable"
    elif gaps["E10b"] > -0.2: dial = "not selective"
    elif all(gaps[r] is not None and gaps[r] <= -0.2 for r in ("E13_seed2027", "E13_seed2028")): dial = "selective (confirmed on held-out seeds)"
    else: dial = "selective on E10b only (exploratory)"
    say(f"  dial verdict: {dial}")
    # strength: every scale point is judged by the same mechanical rule; points whose
    # mean A over valid bands is below 0.2 carry too little agreement to discriminate
    anc = "E10b/gen_r0.0"; vb = valid_bands(anc) if anc in E else []
    base = {k: A("E10b/gen_r0.1", k, anc) for k in vb} if "E10b/gen_r0.1" in E else {}
    say("- E16_spec_scale*: not evaluable (anchor E2_r0.1/gen_r0.0 does not exist)")
    disc, broad = 0, True
    for run in ["E18a_scale0.1", "E18a_scale0.2", "E12_scale0.25", "E18a_scale0.35", "E12_scale0.5", "E18a_scale0.6", "E12_scale0.75"]:
        key = f"{run}/gen_r0.1"
        if key not in E: say(f"- {run}: unscored"); continue
        ks = [k for k in vb if base.get(k, 0) >= 0.1]
        l2 = [k for k in ks if 1 <= k <= 2]; h2 = [k for k in ks if k >= 3]
        mean_a = sum(A(key, k, anc) for k in vb) / len(vb) if vb else 0.0
        if len(l2) < 2 or len(h2) < 2: say(f"- {run}: not evaluable (bands)"); continue
        rat = {k: A(key, k, anc) / base[k] for k in ks}
        g = sum(rat[k] for k in l2) / len(l2) - sum(rat[k] for k in h2) / len(h2)
        if mean_a < 0.2:
            say(f"- {run}: mean A {mean_a:.3f} < 0.2 -> descriptive only; ratio gap {g:+.3f}"); continue
        disc += 1; broad &= abs(g) < 0.1
        say(f"- {run}: mean A {mean_a:.3f}; ratio[0.1,0.3) − ratio[0.3,1) = {g:+.3f}")
    strength = ("not evaluable (<2 discriminative scale points)" if disc < 2 else
                "broadband" if broad else "not broadband")
    say(f"  strength verdict: {strength} ({disc} discriminative points)")
    ok = dial.startswith("selective") and strength == "broadband"
    say(f"-> {('dial band-selective, strength broadband' + (' [exploratory]' if 'exploratory' in dial else '')) if ok else 'band selectivity not established as pre-registered'}")

if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    check_constants()
    if what == "t2": t2(); sys.exit(0)
    say("Disclosure (pre-registration timeline):")
    say("- Two smoke runs of scripts/fixed_instrument.py on E10b, all 8 cutoffs, bands 0-9 with floors: "
        "log_dirs/prereg_smoke/e19_smoke.json (22:14, 32 images, sc_fix@{0.05,0.1,0.3}) and e19_smoke0.json "
        "(22:15, 16 images, sc_fix@{0,0.05,0.1,0.3}), hashes in log_dirs/PREREG.txt. Both predate every frozen S5 rule.")
    say("- The S5 rule changed form after those smokes: v2 (22:10) required a gap >= 0.2 at r = 0.1, 0.2 and 0.3; "
        "v3 (22:19) uses the r=0.3-minus-r=0.1 difference as primary. Because the E10b values were seen, the dial "
        "verdict is confirmed only if the two untouched seeds E13_seed2027/2028 also pass; otherwise it is exploratory.")
    say("- Iteration 2 (22:41) added the S1 fixed-instrument co-primary; iteration 3 replaced a by-name exclusion of "
        "scale 0.25 with the mean-A < 0.2 rule after scale 0.25 and 0.35 moving-instrument SC were known, before E19 "
        "scored any scale directory.")
    say("- S1 ran across code versions: E18a_scale0.1 before the models.py edit (22:20); E18a_scale0.2 and 0.35 with an "
        "intermediate models/pipelines.py (class-name check); E18a_scale0.6 onward with the final pipelines.py (22:40:56).")
    for f in ([s1, s2, s3, s4, s5, s6] if what == "all" else [globals()[what]]):
        try: f()
        except Exception as ex: say(f"{f.__name__} error: {type(ex).__name__}: {ex}")
    if what == "all":
        with open(os.path.join(R, "DECISIONS.md"), "w") as fh:
            fh.write("# Pre-registered decisions (ralplan v3 + iteration 2), computed by scripts/decisions.py\n\n" + "\n".join(out) + "\n")
