#!/bin/bash
# Everything Eurographics 2027 needs, in priority order, fully autonomous.
# Abstract deadline 2026-09-25, full paper 2026-10-01.
#
#   1  E6  zero-shot on real hand-drawn sketches   -- the application claim
#   2  E7  fixed-seed dial figures                 -- the money figure
#   3  E13 seeds 2027 / 2028 on the headline       -- variance
#   4  aggregate everything into a results table
#
# Waits for any chain already running rather than fighting it for the GPU.
# /home/work/gpu_keepalive is never touched: this session is reaped below 1%
# GPU utilisation, so that watchdog must outlive every stage. If it ever dies,
# ensure_running.sh brings it back -- we only check, never kill.
set -u
cd /home/work/model/Frequency_Adaptation
export HF_HOME=/home/work/data/hf_cache
export TOKENIZERS_PARALLELISM=false
LOG=log_dirs/eg.log
mkdir -p log_dirs results
say(){ echo "[eg] $(date '+%F %T') $*" | tee -a "$LOG"; }

keepalive_check(){
    pgrep -f "gpu_keepalive.py" >/dev/null && return 0
    say "WARNING: gpu keep-alive is not running -- restarting it"
    bash /home/work/gpu_keepalive/ensure_running.sh >> "$LOG" 2>&1 || true
}

CFG=config/generated/E10b_skip_inject.yaml
CK=ckpt_dir/E10b_skip_inject/final/adapter.safetensors

say "waiting for the E11 chain and the scale sweep to finish"
while pgrep -f "run_e11.sh|scale_sweep.sh" >/dev/null; do keepalive_check; sleep 120; done
say "GPU free; starting"

retry(){  # retry "<label>" <command...>
    local label="$1"; shift
    for attempt in 1 2 3; do
        keepalive_check
        say "$label (attempt $attempt)"
        "$@" >> "$LOG" 2>&1 && { say "$label done"; return 0; }
        say "$label failed; retrying in 60s"; sleep 60
    done
    say "GIVING UP on $label"; return 1
}

# 1 -- real sketches, zero shot
retry "E6 real-sketch zero-shot" \
    python eval/run_sketch_zeroshot.py -c "$CFG" --adapter "$CK" \
        --sketch_dir /home/work/data/sketches/images \
        --out results/E6_sketch --n_images 200 --steps 50 --batch 8 \
    || say "E6 incomplete; continuing"

# 2 -- the dial figure, fixed seed
retry "E7 dial figures" \
    python eval/make_figures.py -c "$CFG" --adapter "$CK" \
        --out results/figures --n_rows 6 --steps 50 \
    || say "E7 incomplete; continuing"

# 3 -- seed variance on the headline configuration
for s in 2027 2028; do
    NAME="E13_seed$s"; SCFG="config/generated/$NAME.yaml"
    keepalive_check
    say "gate check: $NAME"
    if ! python scripts/check_conditioning.py "$SCFG" >> "$LOG" 2>&1; then
        say "gate check FAILED for $NAME; skipping"; continue
    fi
    ok=0
    for attempt in 1 2 3 4 5; do
        keepalive_check
        say "training $NAME (attempt $attempt)"
        python train_fa.py -c "$SCFG" >> "$LOG" 2>&1 && { say "$NAME finished"; ok=1; break; }
        say "$NAME exited non-zero; resuming in 60s"; sleep 60
    done
    [ "$ok" = 1 ] || { say "GIVING UP on $NAME"; continue; }
    F="ckpt_dir/$NAME/final/adapter.safetensors"
    [ -f "$F" ] || F="ckpt_dir/$NAME/$(cat ckpt_dir/$NAME/LATEST 2>/dev/null)/adapter.safetensors"
    retry "evaluating $NAME" \
        python eval/run_eval.py -c "$SCFG" --adapter "$F" --out "results/$NAME" \
            --n_images 500 --n_div 50 --k_div 4 --steps 50 --batch 16 \
        || say "$NAME evaluation incomplete"
done

# 4 -- one table with everything measured so far
retry "aggregate" python scripts/aggregate_results.py || say "aggregate incomplete"
keepalive_check
say "EG chain complete"
