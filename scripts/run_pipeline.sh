#!/bin/bash
# Full pipeline on the residual-fixed architecture:
#   E1 (baseline) -> evaluate -> E3h (zero-init gate) -> evaluate
#
# Each training stage is gated by scripts/check_conditioning.py, which now also
# asserts that the adapter is purely additive (scale=0 must reproduce stock
# SD1.5 exactly) -- the invariant that would have caught the residual
# double-count before any GPU time was spent.
#
# /home/work/gpu_keepalive is never touched: the session is reaped below 1% GPU
# utilisation, so that watchdog must outlive every stage.
set -u
cd /home/work/model/Frequency_Adaptation
export HF_HOME=/home/work/data/hf_cache
export TOKENIZERS_PARALLELISM=false
LOG=log_dirs/pipeline.log
mkdir -p log_dirs results
say(){ echo "[pipe] $(date '+%F %T') $*" | tee -a "$LOG"; }

run_stage () {   # $1 = config, $2 = run name, $3 = results dir
    local cfg="$1" name="$2" out="$3"
    say "gate check: $name"
    if ! python scripts/check_conditioning.py "$cfg" >> "$LOG" 2>&1; then
        say "FATAL: gate check failed for $name -- refusing to train."
        return 1
    fi
    say "gate check PASSED: $name"

    local done=0
    for attempt in 1 2 3 4 5; do
        say "training $name (attempt $attempt)"
        if python train_fa.py -c "$cfg" >> "$LOG" 2>&1; then
            say "$name finished cleanly"; done=1; break
        fi
        say "$name exited non-zero -- resuming in 60s"
        sleep 60
    done
    [ "$done" = 1 ] || { say "FATAL: $name gave up after 5 attempts"; return 1; }

    local ck="ckpt_dir/$name" final=""
    for cand in "$ck/final/adapter.safetensors" "$ck/$(cat "$ck/LATEST" 2>/dev/null)/adapter.safetensors"; do
        [ -f "$cand" ] && { final="$cand"; break; }
    done
    [ -n "$final" ] || { say "FATAL: no checkpoint for $name"; return 1; }

    say "evaluating $name -> $out"
    python eval/run_eval.py -c "$cfg" --adapter "$final" --out "$out" \
        --n_images 500 --n_div 50 --k_div 4 --steps 50 --batch 16 >> "$LOG" 2>&1 \
        && say "$name evaluation done -> $out/results.json" \
        || say "$name evaluation FAILED (see $LOG)"

    # adapter-off control: says whether a weak result means inert or harmful
    say "adapter-off control for $name"
    python eval/run_eval.py -c "$cfg" --adapter "$final" --adapter_scale 0.0 \
        --out "${out}_adapter_off" --cutoffs 0.0,0.3 --n_images 500 --n_div 20 \
        --k_div 2 --steps 50 --batch 16 >> "$LOG" 2>&1 \
        && say "control done -> ${out}_adapter_off/results.json" \
        || say "control FAILED (see $LOG)"
    return 0
}

run_stage config/fa_sd15_ffhq512.yaml FA_SD15_r0-1 results/E1 || exit 1
run_stage config/generated/E3h_zero_init_gate.yaml E3h_zero_init_gate results/E3h || exit 1
say "pipeline complete"
