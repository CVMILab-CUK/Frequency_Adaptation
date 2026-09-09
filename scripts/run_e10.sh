#!/bin/bash
# E10: three controlled experiments against E9a, run in sequence.
#   E10a  adapter capacity      stem_channels 0 -> 128
#   E10b  injection site        + ControlNet-style skip residuals
#   E10c  receptive field       3x3 local window -> 7x7
# Each differs from E9a in exactly one field, so a difference is attributable.
#
# Every stage is gated by scripts/check_conditioning.py, whose check [5] asserts
# the adapter is purely additive (scale=0 must reproduce stock SD1.5 exactly).
# /home/work/gpu_keepalive is never touched: below 1% GPU utilisation this
# session is reaped, so that watchdog must outlive every stage.
set -u
cd /home/work/model/Frequency_Adaptation
export HF_HOME=/home/work/data/hf_cache
export TOKENIZERS_PARALLELISM=false
LOG=log_dirs/e10.log
mkdir -p log_dirs results
say(){ echo "[e10] $(date '+%F %T') $*" | tee -a "$LOG"; }

stage () {
    local name="$1" out="$2" cfg="config/generated/$1.yaml"
    say "gate check: $name"
    python scripts/check_conditioning.py "$cfg" >> "$LOG" 2>&1 \
        || { say "FATAL: gate check failed for $name -- refusing to train."; return 1; }
    say "gate check PASSED: $name"

    local done=0
    for attempt in 1 2 3 4 5; do
        say "training $name (attempt $attempt)"
        python train_fa.py -c "$cfg" >> "$LOG" 2>&1 && { say "$name finished cleanly"; done=1; break; }
        say "$name exited non-zero -- resuming in 60s"; sleep 60
    done
    [ "$done" = 1 ] || { say "FATAL: $name gave up after 5 attempts"; return 1; }

    local ck="ckpt_dir/$name" final=""
    for c in "$ck/final/adapter.safetensors" "$ck/$(cat "$ck/LATEST" 2>/dev/null)/adapter.safetensors"; do
        [ -f "$c" ] && { final="$c"; break; }
    done
    [ -n "$final" ] || { say "FATAL: no checkpoint for $name"; return 1; }

    say "evaluating $name -> $out"
    python eval/run_eval.py -c "$cfg" --adapter "$final" --out "$out" \
        --n_images 500 --n_div 50 --k_div 4 --steps 50 --batch 16 >> "$LOG" 2>&1 \
        && say "$name evaluation done -> $out/results.json" || say "$name evaluation FAILED"

    say "adapter-off control for $name"
    python eval/run_eval.py -c "$cfg" --adapter "$final" --adapter_scale 0.0 \
        --out "${out}_adapter_off" --cutoffs 0.0,0.3 --n_images 500 --n_div 20 \
        --k_div 2 --steps 50 --batch 16 >> "$LOG" 2>&1 \
        && say "control done -> ${out}_adapter_off/results.json" || say "control FAILED"
    return 0
}

# a failing stage must not silently take the others down with it
stage E10a_capacity128 results/E10a || say "E10a did not complete; continuing"
stage E10b_skip_inject results/E10b || say "E10b did not complete; continuing"
stage E10c_window7     results/E10c || say "E10c did not complete; continuing"
say "E10 complete"
