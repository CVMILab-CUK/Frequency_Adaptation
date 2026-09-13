#!/bin/bash
# E17b -- rerun the D-07 T2I-Adapter baselines after the one-channel input fix.
# keep-alive is checked before each stage and restarted if down; never stopped.
set -u
cd /home/work/model/Frequency_Adaptation
export HF_HOME=/home/work/data/hf_cache TOKENIZERS_PARALLELISM=false
LOG=log_dirs/e17b.log
say(){ echo "[e17b] $(date '+%F %T') $*" | tee -a "$LOG"; }
ka(){ pgrep -f gpu_keepalive.py >/dev/null || { say "keep-alive down; restarting";
      bash /home/work/gpu_keepalive/ensure_running.sh >>"$LOG" 2>&1 || true; }; }
MAIN=config/generated/E10b_skip_inject.yaml
for b in t2iadapter_canny t2iadapter_sketch; do
  ka; say "D-07 baseline $b"
  python eval/run_baselines.py -c "$MAIN" --baseline "$b" --cond native \
    --out results/baselines --n_images 500 --steps 50 --batch 8 >>"$LOG" 2>&1 \
    && say "  $b done" || say "  $b FAILED"
done
ka; python scripts/paper_numbers.py >>"$LOG" 2>&1; say "E17b complete"
