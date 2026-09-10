#!/bin/bash
# E16 -- everything the peer review asks for that needs a GPU, plus the two
# E14 stages the (now corrected) gate check had wrongly blocked.
#
#   D-06  the experiment the text announces: fixed-cutoff training + inference
#         scaling, on the r=0.1 specialist checkpoint that already exists
#   D-16  generic caption sweep, to separate the condition's contribution from
#         caption and seed
#   D-15  re-render the dial figure with every cutoff and printed r labels
#   D-03  the missing 2x2 cell: capacity varied with skips on
#
# keep-alive is checked before every stage and restarted if down; it is never
# stopped. This session is reaped below 1% GPU utilisation.
set -u
cd /home/work/model/Frequency_Adaptation
export HF_HOME=/home/work/data/hf_cache TOKENIZERS_PARALLELISM=false
LOG=log_dirs/e16.log; mkdir -p log_dirs results
say(){ echo "[e16] $(date '+%F %T') $*" | tee -a "$LOG"; }
ka(){ pgrep -f gpu_keepalive.py >/dev/null || { say "keep-alive down; restarting";
      bash /home/work/gpu_keepalive/ensure_running.sh >>"$LOG" 2>&1 || true; }; }

say "waiting for the E14 chain to finish"
while pgrep -f run_e14 >/dev/null; do ka; sleep 120; done
say "GPU free"

MAIN=config/generated/E10b_skip_inject.yaml
CK=ckpt_dir/E10b_skip_inject/final/adapter.safetensors

# ---- E14 stages the old gate check blocked -------------------------------
for n in E14h_nogate E14a_k1; do
  cfg=config/generated/$n.yaml
  ka; say "gate check (re-run): $n"
  if python scripts/check_conditioning.py "$cfg" >>"$LOG" 2>&1; then
    say "gate PASSED: $n"
    ok=0
    for a in 1 2 3; do ka; say "training $n (attempt $a)"
      python train_fa.py -c "$cfg" >>"$LOG" 2>&1 && { ok=1; break; }
      say "$n non-zero exit; retry in 60s"; sleep 60; done
    if [ "$ok" = 1 ]; then
      f=ckpt_dir/$n/final/adapter.safetensors
      [ -f "$f" ] || f=ckpt_dir/$n/$(cat ckpt_dir/$n/LATEST 2>/dev/null)/adapter.safetensors
      ka; python eval/run_eval.py -c "$cfg" --adapter "$f" --out results/$n \
        --n_images 500 --n_div 50 --k_div 4 --steps 50 --batch 16 >>"$LOG" 2>&1 \
        && say "$n done" || say "$n eval FAILED"
    else say "GIVING UP: $n"; fi
  else say "gate still fails: $n -- skipping"; fi
done

# ---- D-06: the declared experiment, on the existing specialist ------------
SPEC=ckpt_dir/E2_fixed_r0.1/final/adapter.safetensors
if [ -f "$SPEC" ]; then
  for s in 0.25 0.5 0.75 1.0; do
    ka; say "D-06 specialist scale $s"
    python eval/run_eval.py -c config/generated/E2_fixed_r0.1.yaml --adapter "$SPEC" \
      --adapter_scale $s --skip_scale $s --out results/E16_spec_scale$s \
      --cutoffs 0.1 --n_images 500 --n_div 50 --k_div 4 --steps 50 --batch 16 \
      >>"$LOG" 2>&1 && say "  scale $s done" || say "  scale $s FAILED"
  done
else say "D-06 skipped: no specialist checkpoint"; fi

# ---- D-16: generic caption, to separate condition from caption/seed -------
ka; say "D-16 generic-caption sweep"
python eval/run_eval.py -c "$MAIN" --adapter "$CK" --out results/E16_generic_caption \
  --generic_caption "a photo of a person" --cutoffs 0.0,0.1,0.3,0.7 \
  --n_images 500 --n_div 50 --k_div 4 --steps 50 --batch 16 >>"$LOG" 2>&1 \
  && say "D-16 done" || say "D-16 FAILED (flag may be missing)"

# ---- D-15: dial figure with every cutoff and labels -----------------------
ka; say "D-15 re-render dial"
python eval/make_figures.py -c "$MAIN" --adapter "$CK" --out results/figures_full \
  --cutoffs 0.0,0.05,0.1,0.2,0.3,0.5,0.7,1.0 --n_rows 6 --steps 50 \
  >>"$LOG" 2>&1 && say "D-15 done" || say "D-15 FAILED"

# ---- D-03: the missing 2x2 cell -------------------------------------------
n=E15_stem256_skip; cfg=config/generated/$n.yaml
ka; say "gate check: $n"
if python scripts/check_conditioning.py "$cfg" >>"$LOG" 2>&1; then
  ok=0
  for a in 1 2 3; do ka; say "training $n (attempt $a)"
    python train_fa.py -c "$cfg" >>"$LOG" 2>&1 && { ok=1; break; }
    say "$n non-zero exit; retry in 60s"; sleep 60; done
  if [ "$ok" = 1 ]; then
    f=ckpt_dir/$n/final/adapter.safetensors
    [ -f "$f" ] || f=ckpt_dir/$n/$(cat ckpt_dir/$n/LATEST 2>/dev/null)/adapter.safetensors
    ka; python eval/run_eval.py -c "$cfg" --adapter "$f" --out results/$n \
      --n_images 500 --n_div 50 --k_div 4 --steps 50 --batch 16 >>"$LOG" 2>&1 \
      && say "$n done" || say "$n eval FAILED"
  fi
else say "gate failed: $n"; fi

ka; say "E16 complete"
