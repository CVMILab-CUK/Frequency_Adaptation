#!/bin/bash
# E14: ablations rebuilt on the configuration that works (E10b), one field each.
# The older E3* configs were branched off E1 and are not comparable to it.
#
# Each stage is gated by scripts/check_conditioning.py, whose check [5] asserts
# the adapter is purely additive. /home/work/gpu_keepalive is only ever checked
# and restarted, never stopped: this session is reaped below 1% GPU use.
set -u
cd /home/work/model/Frequency_Adaptation
export HF_HOME=/home/work/data/hf_cache TOKENIZERS_PARALLELISM=false
LOG=log_dirs/e14.log; mkdir -p log_dirs results
say(){ echo "[e14] $(date '+%F %T') $*" | tee -a "$LOG"; }
ka(){ pgrep -f gpu_keepalive.py >/dev/null || { say "keep-alive down; restarting";
      bash /home/work/gpu_keepalive/ensure_running.sh >>"$LOG" 2>&1 || true; }; }

stage(){
  local n="$1" cfg="config/generated/$1.yaml" out="results/$1"
  [ -f "$cfg" ] || { say "no config for $n"; return 1; }
  ka; say "gate check: $n"
  python scripts/check_conditioning.py "$cfg" >>"$LOG" 2>&1 \
    || { say "FATAL gate check failed: $n"; return 1; }
  local ok=0
  for a in 1 2 3; do
    ka; say "training $n (attempt $a)"
    python train_fa.py -c "$cfg" >>"$LOG" 2>&1 && { ok=1; break; }
    say "$n exited non-zero; retrying in 60s"; sleep 60
  done
  [ "$ok" = 1 ] || { say "GIVING UP: $n"; return 1; }
  local f="ckpt_dir/$n/final/adapter.safetensors"
  [ -f "$f" ] || f="ckpt_dir/$n/$(cat ckpt_dir/$n/LATEST 2>/dev/null)/adapter.safetensors"
  ka; say "evaluating $n"
  python eval/run_eval.py -c "$cfg" --adapter "$f" --out "$out" \
    --n_images 500 --n_div 50 --k_div 4 --steps 50 --batch 16 >>"$LOG" 2>&1 \
    && say "$n done" || say "$n evaluation FAILED"
}

for n in E14h_nogate E14a_k1 E14a_k5 E14b_stem64 E14c_down_avg \
         E14e_color E14f_sqrt E14f_log E14g_conddrop0; do
  stage "$n" || say "$n incomplete; continuing"
done
ka; say "E14 complete"
