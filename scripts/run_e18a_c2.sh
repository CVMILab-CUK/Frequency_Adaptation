#!/bin/bash
# S1 -- C2 matched curve (ralplan v2). Inference only, same flags as E12.
#   scale s at r=0.1 on the randomised model: 0.1 0.2 0.35 0.6 (seed 2026)
#   dial points r=0.15, 0.25 (seed 2026)
#   second sampling seed 2027: scales 0.25 0.5 0.75 at r=0.1; cutoffs 0.2, 0.3
#   E10b_all_off at r=0.05 (tab:where control row was labelled r=0.05 but held r=0)
# keep-alive is checked before every stage and restarted if down; never stopped.
set -u
cd /home/work/model/Frequency_Adaptation
export HF_HOME=/home/work/data/hf_cache TORCH_HOME=/home/work/data/cache/torch XDG_CACHE_HOME=/home/work/data/cache TOKENIZERS_PARALLELISM=false
LOG=log_dirs/e18a.log; mkdir -p log_dirs results
say(){ echo "[e18a] $(date '+%F %T') $*" | tee -a "$LOG"; }
ka(){ pgrep -f gpu_keepalive.py >/dev/null || { say "keep-alive down; restarting";
      bash /home/work/gpu_keepalive/ensure_running.sh >>"$LOG" 2>&1 || true; }; }
CFG=config/generated/E10b_skip_inject.yaml
CK=ckpt_dir/E10b_skip_inject/final/adapter.safetensors
ev(){ # out cutoffs seed n_div k_div [extra...]
  local out=$1 cuts=$2 seed=$3 nd=$4 kd=$5; shift 5
  [ -f "results/$out/results.json" ] && { say "skip $out (exists)"; return 0; }
  for a in 1 2 3; do
    ka; say "eval $out cutoffs=$cuts seed=$seed $* (attempt $a)"
    python eval/run_eval.py -c $CFG --adapter $CK --out results/$out --cutoffs $cuts \
      --n_images 500 --n_div $nd --k_div $kd --steps 50 --batch 16 --seed $seed "$@" >>"$LOG" 2>&1 \
      && { say "  $out done"; return 0; }
    say "  $out failed; retry in 60s"; sleep 60
  done
  say "  GIVING UP $out"; return 1
}
for s in 0.1 0.2 0.35 0.6; do ev E18a_scale$s 0.1 2026 50 4 --adapter_scale $s --skip_scale $s; done
ev E18a_dial_s2026 0.15,0.25 2026 50 4
for s in 0.25 0.5 0.75; do ev E18a_scale${s}_s2027 0.1 2027 50 4 --adapter_scale $s --skip_scale $s; done
ev E18a_dial_s2027 0.2,0.3 2027 50 4
ev E10b_all_off_r0.05 0.05 2026 20 2 --adapter_scale 0 --skip_scale 0
ka; say "S1 complete"
