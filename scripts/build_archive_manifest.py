#!/usr/bin/env python3
"""Decide what stays on the server and what goes to Drive, and record both.

The split is by what the paper needs to stand up. Every number in it comes from
results/*/results.json through scripts/paper_numbers.py, and inference needs
adapter.safetensors; both are small. The bulk -- the generated PNGs, the
intermediate checkpoints, the optimiser state and the TensorBoard image grids --
is reproducible from those or is training scaffolding, so it is archived.
"""
import hashlib, json, os, sys

KEEP, SEND = [], []

def walk(top):
    for r, _, fs in os.walk(top):
        for f in fs: yield os.path.join(r, f)

def classify(p):
    parts = p.split(os.sep)
    top = parts[0]
    if top == "results":
        if p.endswith(("results.json", ".md")): return "keep", "per-run metrics, the source of every number"
        return "send", "generated images / raw arrays"
    if top == "ckpt_dir":
        base = os.path.basename(p)
        if "/final/" in "/" + p and base in ("adapter.safetensors", "meta.json"):
            return "keep", "adapter weights for inference"
        if "/final/" in "/" + p: return "send", "EMA copy / optimiser state"
        return "send", "intermediate checkpoint"
    if top == "log_dirs": return "send", "TensorBoard event file (scalars extracted to archive_meta/)"
    if top == "EG_submission": return "keep", "the submission bundle itself"
    if top in ("output", "log", "__pycache__", "LaTex"): return "send", "training scratch / vendored zip"
    return "keep", "source"

for top in sorted(os.listdir(".")):
    if top in (".git", "archive_meta") or not os.path.isdir(top): continue
    for p in walk(top):
        if "__pycache__" in p: continue
        try: size = os.path.getsize(p)
        except OSError: continue
        (KEEP if classify(p)[0] == "keep" else SEND).append((p, size, classify(p)[1]))

def tot(rows): return sum(s for _, s, _ in rows)
print(f"KEEP {len(KEEP):6d} files {tot(KEEP)/1073741824:8.2f} GB")
print(f"SEND {len(SEND):6d} files {tot(SEND)/1073741824:8.2f} GB")
print()
by = {}
for p, s, why in SEND:
    by.setdefault(p.split(os.sep)[0] + "  — " + why, [0, 0])
    by[p.split(os.sep)[0] + "  — " + why][0] += 1
    by[p.split(os.sep)[0] + "  — " + why][1] += s
for k, (n, s) in sorted(by.items(), key=lambda kv: -kv[1][1]):
    print(f"  {k:62s} {n:7d} files {s/1073741824:7.2f} GB")

os.makedirs("archive_meta", exist_ok=True)
with open("archive_meta/send_to_drive.tsv", "w") as f:
    f.write("path\tbytes\treason\n")
    for p, s, why in sorted(SEND): f.write(f"{p}\t{s}\t{why}\n")
with open("archive_meta/keep_on_server.tsv", "w") as f:
    f.write("path\tbytes\treason\n")
    for p, s, why in sorted(KEEP): f.write(f"{p}\t{s}\t{why}\n")
print("\nwrote archive_meta/send_to_drive.tsv and keep_on_server.tsv")
