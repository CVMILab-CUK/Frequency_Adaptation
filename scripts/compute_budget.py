#!/usr/bin/env python3
"""What this paper cost, measured rather than remembered.

Training wall-clock and peak memory come from each run's TensorBoard log, the
per-run spans that have no scalar log fall back to the interval between their
first and last checkpoint, and sampling cost is the measured seconds per image
(results/E23_sampling_cost) times the number of images actually generated.
The sampling total is therefore a product of two measurements, not a stopwatch
reading, and the paper says so.
"""
import glob, json, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
OUT = "results/E24_compute_budget/results.json"
MAIN = "E10b_skip_inject"

def tb_span(run):
    """(hours, peak GB, images/s) from the run's scalar log, or None."""
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except Exception:
        return None
    best = None
    for f in glob.glob(f"log_dirs/{run}/**/events.out.tfevents.*", recursive=True):
        ea = EventAccumulator(f, size_guidance={"scalars": 0})
        try: ea.Reload()
        except Exception: continue
        if "train/loss" not in ea.Tags()["scalars"]: continue
        s = ea.Scalars("train/loss")
        h = (s[-1].wall_time - s[0].wall_time) / 3600
        mem = max((x.value for x in ea.Scalars("perf/gpu_mem_gb")), default=0.0) \
            if "perf/gpu_mem_gb" in ea.Tags()["scalars"] else 0.0
        ips = max((x.value for x in ea.Scalars("perf/images_per_sec")), default=0.0) \
            if "perf/images_per_sec" in ea.Tags()["scalars"] else 0.0
        if best is None or h > best[0]: best = (h, mem, ips)
    return best

def ckpt_span(run):
    """Hours between the first and last checkpoint, plus one checkpoint interval
    for the steps before the first one was written."""
    d = os.path.join("ckpt_dir", run)
    steps = sorted(os.path.getmtime(p) for p in glob.glob(d + "/step-*") if os.path.isdir(p))
    fin = os.path.join(d, "final")
    if len(steps) < 2 or not os.path.isdir(fin): return None
    return (os.path.getmtime(fin) - steps[0] + (steps[1] - steps[0])) / 3600

def reported_runs():
    """Checkpoints that some results.json was generated from. A run whose output
    the paper never reports -- a development run, or one discarded for a bug --
    is not part of what the paper cost to produce."""
    out = set()
    for f in glob.glob("results/*/results.json"):
        try: d = json.load(open(f))
        except Exception: continue
        ad = d.get("provenance", {}).get("adapter")
        if ad: out.add(os.path.basename(os.path.dirname(os.path.dirname(ad))))
    return out

reported = reported_runs()
runs, train_hours, skipped = {}, 0.0, {}
for run in sorted(os.listdir("ckpt_dir")):
    if not os.path.isdir(os.path.join("ckpt_dir", run)): continue
    h = ckpt_span(run)
    if h is None: continue
    if run in reported:
        runs[run] = round(h, 3); train_hours += h
    else:
        skipped[run] = round(h, 3)

tb = tb_span(MAIN)
bench = json.load(open("results/E23_sampling_cost/results.json"))
sec = bench["adapter_on"]["seconds_per_image"]["mean"]

gen = 0
for root, _, files in os.walk("results"):
    b = os.path.basename(root)
    if b.startswith("gen_r") or b == "generated" or b.startswith(("r0.", "r1.")):
        gen += sum(1 for f in files if f.endswith(".png"))
div = 0
for f in glob.glob("results/*/results.json"):
    try: d = json.load(open(f))
    except Exception: continue
    p = d.get("provenance", {})
    if p.get("k_div") and p.get("n_div"):
        div += p["k_div"] * p["n_div"] * len(d.get("per_cutoff", {}))

sample_hours = (gen + div) * sec / 3600
out = {
    "provenance": {
        "train_hours_source": "TensorBoard train/loss span for the main run; "
                              "checkpoint timestamps for the per-run totals",
        "sampling_source": "results/E23_sampling_cost (measured) x images generated",
        "gpu": bench["provenance"]["gpu"],
        "n_runs": len(runs),
        "runs_not_reported": skipped,
    },
    "train_hours_main": round(tb[0], 3) if tb else runs.get(MAIN),
    "train_peak_gb": round(tb[1], 2) if tb else None,
    "train_images_per_sec": round(tb[2], 1) if tb else None,
    "train_hours_total": round(train_hours, 1),
    "train_hours_per_run": runs,
    "sampling_seconds_per_image": round(sec, 3),
    "sampling_peak_gb": round(bench["adapter_on"]["peak_gpu_gb"], 2),
    "generated_images_saved": gen,
    "diversity_samples_unsaved": div,
    "sampling_hours_total_est": round(sample_hours, 1),
    "gpu_hours_total_est": round(train_hours + sample_hours, 0),
}
os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(out, open(OUT, "w"), indent=1)
print(json.dumps({k: v for k, v in out.items() if k != "train_hours_per_run"}, indent=1))
