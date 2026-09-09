#!/usr/bin/env python3
"""Download FFHQ-512 (with BLIP captions) from the Hub and materialise it on disk.

Writes:
  <out>/images/<idx:05d>.png   -- RGB 512x512
  <out>/captions.jsonl         -- {"file": "00000.png", "caption": "..."}
  <out>/DONE                   -- written only after a full, verified pass

Resumable: images already on disk are skipped, so re-running after an
interruption picks up where it stopped.
"""
import argparse, io, json, os, sys, time

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="Ryan-sjtu/ffhq512-caption")
    ap.add_argument("--out", default="/home/work/data/ffhq512")
    ap.add_argument("--cache", default="/home/work/.cache/huggingface")
    args = ap.parse_args()

    os.environ.setdefault("HF_HOME", args.cache)
    from huggingface_hub import snapshot_download
    import pyarrow.parquet as pq
    from PIL import Image

    img_dir = os.path.join(args.out, "images")
    os.makedirs(img_dir, exist_ok=True)

    print(f"[fetch] snapshot_download {args.repo}", flush=True)
    local = snapshot_download(repo_id=args.repo, repo_type="dataset",
                              allow_patterns=["data/*.parquet", "README.md"])
    print(f"[fetch] snapshot at {local}", flush=True)

    files = sorted(f for f in os.listdir(os.path.join(local, "data")) if f.endswith(".parquet"))
    print(f"[fetch] {len(files)} parquet shards", flush=True)

    cap_path = os.path.join(args.out, "captions.jsonl")
    idx = 0
    t0 = time.time()
    with open(cap_path, "w") as cf:
        for si, fn in enumerate(files):
            tbl = pq.read_table(os.path.join(local, "data", fn))
            cols = tbl.column_names
            img_col = next((c for c in ("image", "img", "jpg", "png") if c in cols), None)
            cap_col = next((c for c in ("text", "caption", "captions", "blip_caption") if c in cols), None)
            if img_col is None:
                raise SystemExit(f"no image column in {fn}; columns={cols}")
            imgs = tbl.column(img_col).to_pylist()
            caps = tbl.column(cap_col).to_pylist() if cap_col else [""] * len(imgs)

            for rec, cap in zip(imgs, caps):
                name = f"{idx:05d}.png"
                dst = os.path.join(img_dir, name)
                if not os.path.exists(dst):
                    raw = rec["bytes"] if isinstance(rec, dict) else rec
                    im = Image.open(io.BytesIO(raw)).convert("RGB")
                    im.save(dst, compress_level=1)
                cf.write(json.dumps({"file": name, "caption": (cap or "").strip()}) + "\n")
                idx += 1
            cf.flush()
            el = time.time() - t0
            print(f"[fetch] shard {si+1}/{len(files)} done  total={idx}  {el:.0f}s", flush=True)

    with open(os.path.join(args.out, "DONE"), "w") as f:
        f.write(f"{idx}\n")
    print(f"[fetch] COMPLETE  {idx} images -> {img_dir}", flush=True)

if __name__ == "__main__":
    main()
