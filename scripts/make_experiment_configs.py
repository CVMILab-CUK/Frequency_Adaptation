#!/usr/bin/env python3
"""Generate the E2 (specialists) and E3 (ablations) configs from the E1 base.

Every variant differs from E1 in exactly one field, so a difference in results
is attributable. Configs are written to config/generated/ and each carries the
axis it varies in `model.name`, which is also the checkpoint directory name.
"""
import argparse, os
from omegaconf import OmegaConf

ap = argparse.ArgumentParser()
ap.add_argument("--base", default="./config/fa_sd15_ffhq512.yaml")
ap.add_argument("--out", default="./config/generated")
ap.add_argument("--epochs", type=int, default=6,
                help="shortened schedule for ablations (E1 runs the full one)")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

base = OmegaConf.load(a.base)

def emit(name, **overrides):
    c = OmegaConf.create(OmegaConf.to_container(base, resolve=True))
    c.model.name = name
    c.trainer.total_epoch = a.epochs
    for path, val in overrides.items():
        node = c
        parts = path.split(".")
        for p in parts[:-1]:
            node = node[p]
        node[parts[-1]] = val
    p = os.path.join(a.out, f"{name}.yaml")
    OmegaConf.save(c, p)
    return p

made = []

# --- E2: fixed-cutoff specialists -------------------------------------------
for r in (0.1, 0.3, 0.5, 0.7):
    made.append(("E2", emit(f"E2_fixed_r{r}", **{"datasets.frequency_rate": r})))

# --- E3a: local window size --------------------------------------------------
for k in (1, 5, 7):                      # 3 is E1
    made.append(("E3a", emit(f"E3a_k{k}", **{"model.sa_adapter.kernel_size": k})))

# --- E3b: adapter capacity ---------------------------------------------------
for ch in (64, 128):                     # 0 is E1
    made.append(("E3b", emit(f"E3b_stem{ch}", **{"model.sa_adapter.stem_channels": ch})))

# --- E3c: downsampler --------------------------------------------------------
made.append(("E3c", emit("E3c_down_avg", **{"model.sa_adapter.down_mode": "avg"})))

# --- E3e: conditioning colour ------------------------------------------------
made.append(("E3e", emit("E3e_color", **{"datasets.frequency_img": "color"})))

# --- E3f: cutoff sampler -----------------------------------------------------
for smp in ("sqrt", "log"):              # uniform is E1
    made.append(("E3f", emit(f"E3f_sampler_{smp}", **{"datasets.rate_sampler": smp})))

# --- E3g: condition dropout --------------------------------------------------
for p_ in (0.0, 0.1):                    # 0.05 is E1
    made.append(("E3g", emit(f"E3g_conddrop{p_}", **{"model.cond_dropout": p_})))

print(f"{'block':<6} {'config':<52}")
print("-" * 60)
for blk, path in made:
    print(f"{blk:<6} {path:<52}")
print(f"\n{len(made)} configs written to {a.out} (epochs={a.epochs})")
print("E3a k=3, E3b stem=0, E3c conv, E3e gray, E3f uniform, E3g 0.05 are the E1 run itself.")
