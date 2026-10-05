# What is on the server but not in git

`log_dirs/`, `ckpt_dir/` and the generated images under `results/` are
gitignored and large (115 GB). This directory records what they contain so the
repository is self-describing without them.

- `tb_scalars/<run>.json` — every scalar series from that run's TensorBoard
  logs: the loss and learning-rate curves, `perf/images_per_sec`,
  `perf/gpu_mem_gb`, and the per-cutoff validation metrics. 33 runs, 14.2 GB of
  event files reduced to 2.7 MB. Table 14's Compute row and the statement that
  training stops where the validation curve flattens both come from these. What
  is left behind in the event files is the validation image grids, which is
  what made them large.
- `send_to_drive.tsv`, `keep_on_server.tsv` — every file classified, with its
  size and the reason. Regenerate with `scripts/build_archive_manifest.py`.
- `tb_extract.log` — the extraction run.

Nothing here is a substitute for the checkpoints: `weights/` holds the adapters
the paper cites, and the rest of `ckpt_dir/` is intermediate checkpoints and
optimiser state that only matter for resuming training.
