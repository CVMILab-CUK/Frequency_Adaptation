#!/bin/bash
# E11: everything the paper still needs, in sequence, on the E10b architecture
# (skip injection + conv condition encoder) -- the configuration that works.
#
#   E4  baselines      ControlNet-canny / -hed, native condition, no training
#   E2  specialists    four adapters at fixed r, vs the one random-r model at
#                      the same r. The direct test of "one model replaces K".
#
# Every training stage is gated by scripts/check_conditioning.py, whose check
# [5] asserts the adapter is purely additive. /home/work/gpu_keepalive is never
# touched: this session is reaped below 1% GPU utilisation.
set -u
cd /home/work/model/Frequency_Adaptation
export HF_HOME=/home/work/data/hf_cache
export TOKENIZERS_PARALLELISM=false
LOG=log_dirs/e11.log
mkdir -p log_dirs results
say(){ echo "[e11] $(date '+%F %T') $*" | tee -a "$LOG"; }

CFG_MAIN=config/generated/E10b_skip_inject.yaml

# ---------------------------------------------------------------- E4 baselines
for b in controlnet_canny controlnet_hed; do
    say "baseline $b (native condition)"
    python eval/run_baselines.py -c "$CFG_MAIN" --baseline "$b" --cond native \
        --n_images 500 --steps 50 --batch 8 --out results/baselines >> "$LOG" 2>&1 \
        && say "baseline $b done" || say "baseline $b FAILED"
done

# --------------------------------------------------------------- E2 specialists
train_and_eval () {
    local name="$1" cfg="config/generated/$1.yaml" out="$2"
    say "gate check: $name"
    python scripts/check_conditioning.py "$cfg" >> "$LOG" 2>&1 \
        || { say "FATAL: gate check failed for $name"; return 1; }
    say "gate check PASSED: $name"

    local ok=0
    for attempt in 1 2 3 4 5; do
        say "training $name (attempt $attempt)"
        python train_fa.py -c "$cfg" >> "$LOG" 2>&1 && { say "$name finished cleanly"; ok=1; break; }
        say "$name exited non-zero -- resuming in 60s"; sleep 60
    done
    [ "$ok" = 1 ] || { say "FATAL: $name gave up after 5 attempts"; return 1; }

    local ck="ckpt_dir/$name" final=""
    for c in "$ck/final/adapter.safetensors" "$ck/$(cat "$ck/LATEST" 2>/dev/null)/adapter.safetensors"; do
        [ -f "$c" ] && { final="$c"; break; }
    done
    [ -n "$final" ] || { say "FATAL: no checkpoint for $name"; return 1; }

    say "evaluating $name -> $out"
    python eval/run_eval.py -c "$cfg" --adapter "$final" --out "$out" \
        --n_images 500 --n_div 50 --k_div 4 --steps 50 --batch 16 >> "$LOG" 2>&1 \
        && say "$name evaluation done" || say "$name evaluation FAILED"
    return 0
}

for r in 0.1 0.3 0.5 0.7; do
    train_and_eval "E2_fixed_r${r}" "results/E2_r${r}" || say "E2 r=${r} did not complete; continuing"
done

say "E11 complete"
