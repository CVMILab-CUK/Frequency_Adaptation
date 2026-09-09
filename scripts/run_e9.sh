#!/bin/bash
# E9b then E9a, each gated, trained, evaluated, and controlled.
#
#   E9b = E1  + source_size 256   -> does the earlier (lower) resolution explain it?
#   E9a = E3h + conv condition encoder -> does bypassing the frozen VAE fix it?
#
# Each differs from its control by exactly one field, so a difference in the
# result is attributable. /home/work/gpu_keepalive is never touched.
set -u
cd /home/work/model/Frequency_Adaptation
export HF_HOME=/home/work/data/hf_cache
export TOKENIZERS_PARALLELISM=false
LOG=log_dirs/e9.log
mkdir -p log_dirs results
say(){ echo "[e9] $(date '+%F %T') $*" | tee -a "$LOG"; }

stage () {
    local cfg="$1" name="$2" out="$3"
    say "gate check: $name"
    python scripts/check_conditioning.py "$cfg" >> "$LOG" 2>&1 \
        || { say "FATAL: gate check failed for $name"; return 1; }
    say "gate check PASSED: $name"

    local done=0
    for a in 1 2 3 4 5; do
        say "training $name (attempt $a)"
        python train_fa.py -c "$cfg" >> "$LOG" 2>&1 && { say "$name finished cleanly"; done=1; break; }
        say "$name exited non-zero -- resuming in 60s"; sleep 60
    done
    [ "$done" = 1 ] || { say "FATAL: $name gave up"; return 1; }

    local ck="ckpt_dir/$name" f=""
    for c in "$ck/final/adapter.safetensors" "$ck/$(cat "$ck/LATEST" 2>/dev/null)/adapter.safetensors"; do
        [ -f "$c" ] && { f="$c"; break; }
    done
    [ -n "$f" ] || { say "FATAL: no checkpoint for $name"; return 1; }

    say "evaluating $name"
    python eval/run_eval.py -c "$cfg" --adapter "$f" --out "$out" \
        --n_images 500 --n_div 50 --k_div 4 --steps 50 --batch 16 >> "$LOG" 2>&1 \
        && say "$name evaluation done -> $out/results.json" || say "$name evaluation FAILED"

    say "adapter-off control for $name"
    python eval/run_eval.py -c "$cfg" --adapter "$f" --adapter_scale 0.0 \
        --out "${out}_adapter_off" --cutoffs 0.0,0.3 --n_images 500 --n_div 20 \
        --k_div 2 --steps 50 --batch 16 >> "$LOG" 2>&1 \
        && say "control done -> ${out}_adapter_off/results.json" || say "control FAILED"
    return 0
}

stage config/generated/E9b_source256.yaml    E9b_source256    results/E9b || exit 1
stage config/generated/E9a_conv_encoder.yaml E9a_conv_encoder results/E9a || exit 1
say "E9 complete"
