#!/bin/bash
# E17 -- the two review items E16 does not cover.
#
#   D-07  T2I-Adapter as a size-matched baseline. ControlNet duplicates the
#         encoder; T2I-Adapter is a small side network, which is the design
#         family our adapter belongs to, so it is the fairer of the two.
#   D-17  what r means for a hand drawing. The manuscript says a drawing "has
#         no r"; this fixes a definition the artist can act on -- filter your
#         own drawing at cutoff r -- and sweeps it.
#
# keep-alive is checked before every stage and restarted if it is down; it is
# never stopped. This session is reaped below 1% GPU utilisation.
set -u
cd /home/work/model/Frequency_Adaptation
export HF_HOME=/home/work/data/hf_cache TOKENIZERS_PARALLELISM=false
LOG=log_dirs/e17.log; mkdir -p log_dirs results
say(){ echo "[e17] $(date '+%F %T') $*" | tee -a "$LOG"; }
ka(){ pgrep -f gpu_keepalive.py >/dev/null || { say "keep-alive down; restarting";
      bash /home/work/gpu_keepalive/ensure_running.sh >>"$LOG" 2>&1 || true; }; }

say "waiting for the E16 chain to finish"
while pgrep -f "run_e14|run_e16" >/dev/null; do ka; sleep 120; done
say "GPU free"

MAIN=config/generated/E10b_skip_inject.yaml
CK=ckpt_dir/E10b_skip_inject/final/adapter.safetensors

# ---- D-07: the size-matched baseline --------------------------------------
for b in t2iadapter_canny t2iadapter_sketch; do
  ka; say "D-07 baseline $b (native condition)"
  python eval/run_baselines.py -c "$MAIN" --baseline "$b" --cond native \
    --out results/baselines --n_images 500 --steps 50 --batch 8 \
    >>"$LOG" 2>&1 && say "  $b done" || say "  $b FAILED"
done

# ---- D-17: the dial on hand drawings --------------------------------------
if [ -f "$CK" ]; then
  ka; say "D-17 drawing dial sweep"
  python eval/run_sketch_dial.py -c "$MAIN" --adapter "$CK" \
    --sketch_dir /home/work/data/sketches/face --out results/E17_sketch_dial \
    --cutoffs 0.0,0.05,0.1,0.2,0.3,0.5,0.7,1.0 --n_images 40 --steps 50 --batch 8 \
    >>"$LOG" 2>&1 && say "D-17 done" || say "D-17 FAILED"
else say "D-17 skipped: no main checkpoint at $CK"; fi

ka; say "regenerating paper macros from results/"
python scripts/paper_numbers.py >>"$LOG" 2>&1 && say "macros regenerated"
ka; say "E17 complete"
