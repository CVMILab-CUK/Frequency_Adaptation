#!/bin/bash
# Re-run E10b (it died on a bf16/fp32 mismatch that check [6] now catches),
# then extend the capacity ladder that E10a showed to be the live axis.
#   E10b  injection site   + ControlNet-style skip residuals   (control: E9a)
#   E10d  capacity         stem_channels 0 -> 256              (control: E9a)
# gpu_keepalive is left alone: below 1% utilisation this session is reaped.
set -u
cd /home/work/model/Frequency_Adaptation
export HF_HOME=/home/work/data/hf_cache
export TOKENIZERS_PARALLELISM=false
LOG=log_dirs/e10b.log
mkdir -p log_dirs results
say(){ echo "[e10b] $(date '+%F %T') $*" | tee -a "$LOG"; }

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
        && say "$name evaluation done" || say "$name evaluation FAILED"
    say "adapter-off control for $name"
    python eval/run_eval.py -c "$cfg" --adapter "$final" --adapter_scale 0.0 \
        --out "${out}_adapter_off" --cutoffs 0.0,0.3 --n_images 500 --n_div 20 \
        --k_div 2 --steps 50 --batch 16 >> "$LOG" 2>&1 \
        && say "control done" || say "control FAILED"
    return 0
}
stage E10b_skip_inject  results/E10b || say "E10b did not complete; continuing"
stage E10d_capacity256  results/E10d || say "E10d did not complete; continuing"
say "E10b/E10d complete"
