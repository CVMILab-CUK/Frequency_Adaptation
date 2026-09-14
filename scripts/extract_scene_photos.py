#!/usr/bin/env python3
"""E21 input: the photographs of rhfeiyang/photo-sketch-pair-500 as 512^2 PNGs.

General scenes, far from FFHQ faces, used only to ask whether the dial carries
out of domain. Each photo is centre-cropped to a square and resized once with
Lanczos; the source file name is kept in index.json so every output traces back.
"""
import glob, io, json, os
import pyarrow.parquet as pq
from PIL import Image

SRC = sorted(glob.glob("/home/work/data/hf_cache/hub/datasets--rhfeiyang--photo-sketch-pair-500/**/*.parquet", recursive=True))
OUT = "/home/work/data/scene_photos512"
os.makedirs(OUT, exist_ok=True)
assert len(SRC) == 1, SRC
t = pq.read_table(SRC[0]).to_pydict()
index = {}
for i, (ph, fn) in enumerate(zip(t["photo"], t["file_name"])):
    im = Image.open(io.BytesIO(ph["bytes"])).convert("RGB")
    w, h = im.size; s = min(w, h)
    im = im.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s)).resize((512, 512), Image.LANCZOS)
    name = f"{i:04d}.png"
    im.save(os.path.join(OUT, name))
    index[name] = {"file_name": fn, "source_size": [w, h]}
json.dump({"source": SRC[0], "crop": "centre square", "resize": "512 Lanczos", "files": index},
          open(os.path.join(OUT, "index.json"), "w"), indent=1)
print(f"{len(index)} photos -> {OUT}")
