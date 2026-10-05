#!/usr/bin/env python3
"""Pull the scalars out of the TensorBoard logs and leave the images behind.

log_dirs/ is 14.2 GB, and almost all of it is the validation image grids each
run wrote; the scalars are a few hundred points per run. Table 14's Compute row
(hours per run, peak GB, images/s) and the validation curves come from those
scalars, so this writes them to JSON and the big event files become archive
material rather than something the repository needs.
"""
import glob, json, os, sys
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

OUT = "archive_meta/tb_scalars"
os.makedirs(OUT, exist_ok=True)
runs = sorted(d for d in glob.glob("log_dirs/*") if os.path.isdir(d))
total_src = total_dst = 0
for d in runs:
    run = os.path.basename(d)
    files = glob.glob(d + "/**/events.out.tfevents.*", recursive=True)
    if not files: continue
    src = sum(os.path.getsize(f) for f in files)
    out = {"run": run, "event_files": len(files), "source_bytes": src, "series": {}}
    for f in sorted(files):
        # images=1 keeps the accumulator from holding image payloads in memory
        ea = EventAccumulator(f, size_guidance={"scalars": 0, "images": 1,
                                               "histograms": 1, "audio": 1,
                                               "compressedHistograms": 1, "tensors": 1})
        try: ea.Reload()
        except Exception as e:
            out.setdefault("unreadable", []).append([os.path.basename(f), str(e)]); continue
        for tag in ea.Tags().get("scalars", []):
            pts = [[s.step, round(s.value, 6), round(s.wall_time, 3)] for s in ea.Scalars(tag)]
            out["series"].setdefault(tag, []).extend(pts)
    for tag in out["series"]: out["series"][tag].sort()
    p = os.path.join(OUT, run + ".json")
    json.dump(out, open(p, "w"), separators=(",", ":"))
    dst = os.path.getsize(p)
    total_src += src; total_dst += dst
    print(f"{run:34s} {src/1048576:8.0f} MB -> {dst/1024:7.1f} KB  "
          f"{len(out['series'])} series", flush=True)
print(f"\n{len(runs)} runs: {total_src/1073741824:.1f} GB -> {total_dst/1048576:.1f} MB")
