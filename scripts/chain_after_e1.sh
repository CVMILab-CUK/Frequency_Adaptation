#!/bin/bash
# Runs after E1: finish E1 properly (full evaluation), then train E3h.
#
#   1. wait for the E1 launcher to exit
#   2. full evaluation of E1's final checkpoint -> results/E1 (the baseline curve)
#   3. gate check + train E3h (zero-init gate), the only difference from E1
#   4. full evaluation of E3h -> results/E3h
#
# Nothing here touches /home/work/gpu_keepalive: the session is reaped below 1%
# GPU utilisation, so that watchdog must outlive every job.
set -u
cd /home/work/model/Frequency_Adaptation
export HF_HOME=/home/work/data/hf_cache
export TOKENIZERS_PARALLELISM=false
LOG=log_dirs/chain.log
mkdir -p log_dirs results
say(){ echo "[chain] $(date '+%F %T') $*" | tee -a "$LOG"; }

say "waiting for E1 to finish"
while pgrep -f "launch_e1" > /dev/null; do sleep 60; done
say "E1 launcher exited"

CK=ckpt_dir/FA_SD15_r0-1
FINAL=""
for cand in "$CK/final/adapter.safetensors" "$CK/$(cat $CK/LATEST 2>/dev/null)/adapter.safetensors"; do
    [ -f "$cand" ] && { FINAL="$cand"; break; }
done
if [ -z "$FINAL" ]; then
    say "FATAL: no E1 checkpoint found; not continuing."
    exit 1
fi
say "E1 checkpoint: $FINAL"

say "evaluating E1 (baseline curve)"
python eval/run_eval.py -c config/fa_sd15_ffhq512.yaml --adapter "$FINAL" \
    --out results/E1 --n_images 500 --n_div 50 --k_div 4 --steps 50 --batch 16 \
    >> "$LOG" 2>&1 && say "E1 evaluation done -> results/E1/results.json" \
                   || say "E1 evaluation FAILED (see $LOG); continuing to E3h"

CFG=config/generated/E3h_zero_init_gate.yaml
say "gate check for E3h"
if ! python scripts/check_conditioning.py "$CFG" >> "$LOG" 2>&1; then
    say "FATAL: E3h gate check failed -- refusing to train."
    exit 1
fi
say "E3h gate check PASSED"

for attempt in 1 2 3 4 5; do
    say "starting E3h (attempt $attempt)"
    python train_fa.py -c "$CFG" >> "$LOG" 2>&1 && { say "E3h finished cleanly"; break; }
    say "E3h exited with code $? -- resuming in 60s"
    sleep 60
done

E3=ckpt_dir/E3h_zero_init_gate
F3=""
for cand in "$E3/final/adapter.safetensors" "$E3/$(cat $E3/LATEST 2>/dev/null)/adapter.safetensors"; do
    [ -f "$cand" ] && { F3="$cand"; break; }
done
if [ -n "$F3" ]; then
    say "evaluating E3h"
    python eval/run_eval.py -c "$CFG" --adapter "$F3" \
        --out results/E3h --n_images 500 --n_div 50 --k_div 4 --steps 50 --batch 16 \
        >> "$LOG" 2>&1 && say "E3h evaluation done -> results/E3h/results.json" \
                       || say "E3h evaluation FAILED (see $LOG)"
fi
say "chain complete"
