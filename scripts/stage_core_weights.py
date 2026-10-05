#!/usr/bin/env python3
"""Copy the adapters the paper's claims rest on into weights/, with provenance.

Only the runs a reader needs to check the paper are included. The strength
sweep needs no checkpoint of its own: those rows scale an existing adapter at
inference time, which the per-run provenance in results/ records.
"""
import hashlib, json, os, shutil

CORE = [
    ("E10b_skip_inject", "randomised-r adapter, training seed 2026. Table 1, Figs 2-5, "
                         "and the base for every scaled strength row."),
    ("E13_seed2027",     "the same configuration at training seed 2027; with 2026 and 2028 "
                         "it gives the three-seed means the abstract quotes."),
    ("E13_seed2028",     "training seed 2028, as above."),
    ("E2_fixed_r0.1",    "specialist trained at a fixed r=0.1 (Sec. 4.4), and the base for "
                         "the fixed-r strength rows."),
    ("E18_skip_only",    "the skip path trained alone (Sec. 5.3)."),
    ("E2_fixed_r0.3",    "specialist at r=0.3 (Table S5)."),
    ("E2_fixed_r0.5",    "specialist at r=0.5 (Table S5)."),
    ("E2_fixed_r0.7",    "specialist at r=0.7 (Table S5)."),
]
os.makedirs("weights", exist_ok=True)
man = {"note": "Adapter weights for the runs the paper cites. float32, as trained and as "
               "evaluated; evaluation uses the raw weights, never the EMA copy.",
       "load": "models/models.py -> load_adapter(); configs in config/generated/<run>.yaml",
       "adapters": []}
for run, why in CORE:
    src = f"ckpt_dir/{run}/final/adapter.safetensors"
    if not os.path.exists(src): print("MISSING", src); continue
    dst = f"weights/{run}.safetensors"
    shutil.copy2(src, dst)
    h = hashlib.sha256(open(dst, "rb").read()).hexdigest()
    cfg = f"config/generated/{run}.yaml"
    man["adapters"].append({"run": run, "file": os.path.basename(dst),
                            "bytes": os.path.getsize(dst), "sha256": h,
                            "config": cfg if os.path.exists(cfg) else None,
                            "why": why})
    print(f"{run:22s} {os.path.getsize(dst)/1048576:6.1f} MB  {h[:16]}")
json.dump(man, open("weights/MANIFEST.json", "w"), indent=1)
print(f"\n{len(man['adapters'])} adapters, "
      f"{sum(a['bytes'] for a in man['adapters'])/1048576:.0f} MB -> weights/")
