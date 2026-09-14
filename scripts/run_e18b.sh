#!/bin/bash
# ralplan v3 (+ Architect iteration-2 amendments), stages S2-S7, unattended
# after S1 (scripts/run_e18a_c2.sh).
#   S2 VAE survival on SD1.5 / SDXL / a 16-channel FLUX-family VAE (n=500)
#   S3 dial on 500 out-of-domain scene photos, in-domain comparator, adapter-off control
#   S4 E18: skip-only adapter trained from scratch (smoke load+eval at step 2000)
#   S5 E19: fixed-instrument and band rescoring of every saved generation
#   S6 conditional slot (E18 seed 2027 or E2_fixed_r0.1 seed 2027), start cutoff
#   S7 decisions.py -> results/DECISIONS.md; macros regenerated add-only; no prose edits
# keep-alive: a loop for the whole life of the chain plus a check before every
# stage; it is never stopped. Processes are only ever killed by PID.
set -u
cd /home/work/model/Frequency_Adaptation
export HF_HOME=/home/work/data/hf_cache TORCH_HOME=/home/work/data/cache/torch \
       XDG_CACHE_HOME=/home/work/data/cache TRITON_CACHE_DIR=/home/work/data/cache/triton \
       TOKENIZERS_PARALLELISM=false
LOG=log_dirs/e18b.log; mkdir -p log_dirs results
say(){ echo "[e18b] $(date '+%F %T') $*" | tee -a "$LOG"; }
ka(){ pgrep -f gpu_keepalive.py >/dev/null || { say "keep-alive down; restarting";
      bash /home/work/gpu_keepalive/ensure_running.sh >>"$LOG" 2>&1 || true; }; }
( while true; do ka; sleep 120; done ) &
KAP=$!
trap 'kill $KAP 2>/dev/null' EXIT

CFG=config/generated/E10b_skip_inject.yaml
CK=ckpt_dir/E10b_skip_inject/final/adapter.safetensors
disk_ok(){ # results and checkpoints live on the NFS mount; the local loop disk only gets a warning
  local kb lk
  kb=$(df --output=avail -k /home/work/model | tail -1)
  lk=$(df --output=avail -k /home/work | tail -1)
  [ "$lk" -ge 1048576 ] || say "warning: local /home/work has $((lk/1024)) MB free"
  [ "$kb" -ge 52428800 ] || { say "DISK GUARD: /home/work/model has $((kb/1048576)) GB free (< 50); skipping $1"; return 1; }
}

say "waiting for S1"
while pgrep -f run_e18a_c2.sh >/dev/null; do sleep 120; done
say "S1 finished"

# ---------------------------------------------------------------- S2: VAEs
vae(){ # tag repo subfolder [required_latent_channels]
  local tag=$1 id=$2 sub=$3 need=${4:-}
  local out=results/vae_survival_${tag}_n500.json
  [ -f "$out" ] && { say "S2 $tag exists"; return 0; }
  disk_ok "S2 $tag" || return 1
  ka; say "S2 $tag <- $id/$sub"
  python scripts/vae_condition_survival.py --n 500 --batch 16 --vae_id "$id" --subfolder "$sub" --out "$out" >>"$LOG" 2>&1 \
    || { say "  $tag failed ($id)"; rm -f "$out"; return 1; }
  if [ -n "$need" ]; then
    local ch; ch=$(python -c "import json;print(json.load(open('$out'))['provenance']['latent_channels'])")
    [ "$ch" = "$need" ] || { say "  $id has $ch latent channels, not $need; discarded"; rm -f "$out"; return 1; }
  fi
  say "  $tag done"
}
vae sd15 stable-diffusion-v1-5/stable-diffusion-v1-5 vae || true
vae sdxl stabilityai/sdxl-vae "" || true
vae flux16 ostris/Flex.1-alpha vae 16 || vae flux16 Tongyi-MAI/Z-Image-Turbo vae 16 \
  || vae flux16 AuraDiffusion/16ch-vae "" 16 || say "S2 no 16-channel VAE could be measured"

# ---------------------------------------------------------------- S3: photos
PH=/home/work/data/scene_photos512
sd(){ # out dir cutoffs [extra...]
  local out=$1 dir=$2 cuts=$3; shift 3
  [ -f "results/$out/results.json" ] && { say "S3 $out exists"; return 0; }
  disk_ok "S3 $out" || return 1
  ka; say "S3 $out"
  python eval/run_sketch_dial.py -c $CFG --adapter $CK --sketch_dir "$dir" --ref_dir "$dir" \
    --prompt "a high-quality photo" --cutoffs "$cuts" --n_images 500 --batch 16 --out "results/$out" "$@" >>"$LOG" 2>&1 \
    && say "  $out done" || say "  $out FAILED"
}
sd E21_ood_photos $PH 0.0,0.05,0.1,0.2,0.3
sd E21_indomain_photos results/E10b/reference 0.05,0.1,0.2,0.3
sd E21_ood_photos_off $PH 0.1 --adapter_scale 0 --skip_scale 0

# ---------------------------------------------------------------- training helper
smoke_state_ok(){ # step_dir cfg
  python scripts/check_ckpt_state.py "$1" >>"$LOG" 2>&1 && return 0
  grep -q "attn_inject: false" "$2" && return 1
  say "  optimizer-count mismatch logged only (two-path model)"; return 0
}
smoke_watch(){ # run cfg train_pid
  # Only evidence that the checkpoint itself is unusable stops training: the
  # model refuses to load it, or its optimizer state does not cover the saved
  # tensors. A failure later in generation is logged and training continues.
  local n=$1 cfg=$2 TP=$3 f=ckpt_dir/$1/step-2000 try
  while kill -0 "$TP" 2>/dev/null; do
    if [ -f "$f/adapter.safetensors" ] && [ -f "$f/train_state.pt" ]; then
      sleep 90
      for try in 1 2; do
        if python scripts/check_ckpt_state.py --load "$cfg" "$f/adapter.safetensors" >>"$LOG" 2>&1 \
           && smoke_state_ok "$f" "$cfg"; then
          touch "log_dirs/_smoke_$n.ok"; say "smoke OK: $n (checkpoint loads, optimizer state matches)"
          python eval/run_eval.py -c "$cfg" --adapter "$f/adapter.safetensors" --out "log_dirs/_smoke_$n" \
            --cutoffs 0.1 --n_images 16 --n_div 4 --k_div 2 --steps 50 --batch 8 --skip_fid >>"$LOG" 2>&1 \
            && say "smoke generation OK: $n" || say "warning: smoke generation failed for $n (training continues; full eval will show it)"
          return 0
        fi
        say "smoke load/state attempt $try failed: $n"; sleep 60
      done
      touch "log_dirs/_smoke_$n.fail"
      if kill -0 "$TP" 2>/dev/null; then say "SMOKE FAILED twice (load or optimizer state): $n -- stopping training PID $TP"; kill "$TP"; fi
      return 1
    fi
    sleep 60
  done
}
train_eval(){ # run_name
  local n=$1 cfg=config/generated/$1.yaml d=ckpt_dir/$1
  [ -f "results/$n/results.json" ] && { say "$n already evaluated"; return 0; }
  [ -f "$cfg" ] || { say "no config $cfg"; return 1; }
  disk_ok "$n" || return 1
  # Refuse only on evidence of someone else's state; an empty directory made by a
  # gate check's Trainer is not evidence, and our own partial run should resume.
  if [ -f "$d/run_manifest.json" ]; then
    python scripts/check_ckpt_state.py --manifest "$d" "$cfg" >>"$LOG" 2>&1 \
      || { say "FATAL $d holds a run with a different seed; refusing $n"; return 1; }
    say "$d holds this run's own state; training resumes from it"
  elif [ -e "$d/LATEST" ] || compgen -G "$d/step-*" >/dev/null; then
    say "FATAL $d has checkpoints but no manifest; refusing $n"; return 1
  fi
  ka; say "gate check: $n"
  python scripts/check_conditioning.py "$cfg" >>"$LOG" 2>&1 || { say "FATAL gate check failed: $n"; return 1; }
  local ok=0 a TP WP rc
  for a in 1 2 3; do
    rm -f "log_dirs/_smoke_$n.fail"
    ka; say "training $n (attempt $a)"
    python train_fa.py -c "$cfg" >>"$LOG" 2>&1 &
    TP=$!; WP=""
    if [ ! -f "log_dirs/_smoke_$n.ok" ]; then smoke_watch "$n" "$cfg" "$TP" & WP=$!; fi
    wait "$TP"; rc=$?
    [ -n "$WP" ] && wait "$WP" 2>/dev/null   # let an in-flight smoke finish before deciding
    if [ "$rc" = 0 ]; then ok=1; break; fi
    [ -f "log_dirs/_smoke_$n.fail" ] && { say "$n: checkpoint unusable (smoke failed twice); not retrying"; return 1; }
    say "$n exited $rc; retry in 60s (resumes from LATEST)"; sleep 60
  done
  [ "$ok" = 1 ] || { say "GIVING UP: $n"; return 1; }
  [ -f "log_dirs/_smoke_$n.fail" ] && say "warning: smoke failed but training completed; evaluation will surface load errors"
  python scripts/check_ckpt_state.py --manifest "$d" "$cfg" >>"$LOG" 2>&1 \
    || { say "FATAL manifest seed does not match $cfg; not evaluating $n"; return 1; }
  local fck=$d/final/adapter.safetensors
  [ -f "$fck" ] || fck=$d/$(cat "$d/LATEST" 2>/dev/null)/adapter.safetensors
  for a in 1 2; do
    ka; say "evaluating $n (attempt $a)"
    python eval/run_eval.py -c "$cfg" --adapter "$fck" --out "results/$n" \
      --n_images 500 --n_div 50 --k_div 4 --steps 50 --batch 16 >>"$LOG" 2>&1 && { say "$n done"; return 0; }
    sleep 60
  done
  say "$n evaluation FAILED"; return 1
}

# ---------------------------------------------------------------- S4: E18
if [ -f log_dirs/E18_CODE_READY ]; then
  train_eval E18_skip_only || say "S4 incomplete"
else
  say "S4 skipped: log_dirs/E18_CODE_READY missing"
fi

# ---------------------------------------------------------------- S5: E19
if disk_ok "S5"; then
  ka; say "S5 fixed-instrument rescoring"
  DIRS=""
  for x in E10b E13_seed2027 E13_seed2028 E12_scale0.25 E12_scale0.5 E12_scale0.75 \
           E18a_scale0.1 E18a_scale0.2 E18a_scale0.35 E18a_scale0.6 E18a_dial_s2026 \
           E18a_scale0.25_s2027 E18a_scale0.5_s2027 E18a_scale0.75_s2027 E18a_dial_s2027 \
           E16_spec_scale0.25 E16_spec_scale0.5 E16_spec_scale0.75 E16_spec_scale1.0 \
           E10b_skip_off E10b_adapter_off E10b_all_off E10b_all_off_r0.05 \
           E2_r0.1 E2_r0.3 E2_r0.5 E2_r0.7 E18_skip_only; do
    [ -d "results/$x" ] && DIRS="$DIRS results/$x" || say "  S5: results/$x absent, excluded"
  done
  python scripts/fixed_instrument.py --dirs $DIRS --batch 32 --out results/E19_fixed_instrument/results.json >>"$LOG" 2>&1 \
    && say "  S5 done" || say "  S5 FAILED"
fi

# ---------------------------------------------------------------- S6: conditional
CHOICE=$(python scripts/decisions.py t2 2>>"$LOG")
say "S6 choice: ${CHOICE:-<empty: decisions.py failed, defaulting to SPEC2>}"
if [ "$(date +%s)" -ge "$(TZ=Asia/Seoul date -d '2026-09-14 15:00' +%s)" ]; then
  say "S6 skipped: past the 2026-09-14 15:00 KST start cutoff"
elif [ "$CHOICE" = "SEED2" ] && [ -f log_dirs/E18_CODE_READY ]; then
  train_eval E18_skip_only_s2027 || say "S6 incomplete"
  python scripts/fixed_instrument.py --dirs results/E18_skip_only_s2027 --batch 32 \
    --out results/E19_fixed_instrument/results.json >>"$LOG" 2>&1 || true
else
  train_eval E2_fixed_r0.1_s2027 || say "S6 incomplete"
fi

# ---------------------------------------------------------------- S7
ka; say "S7 decisions + macros"
python scripts/decisions.py all >>"$LOG" 2>&1 && say "  results/DECISIONS.md written"
cp papers/numbers.tex log_dirs/numbers.before.tex
python scripts/paper_numbers.py >>"$LOG" 2>&1 && say "  macros regenerated"
python - <<'PYC' >>"$LOG" 2>&1 || { cp log_dirs/numbers.before.tex papers/numbers.tex; say "  numbers.tex would change existing macros; restored the previous file"; }
old = set(open("log_dirs/numbers.before.tex").read().splitlines()[1:])
new = set(open("papers/numbers.tex").read().splitlines()[1:])
gone = sorted(old - new)
print(f"numbers.tex: +{len(new - old)} lines, -{len(gone)} lines")
raise SystemExit(1 if gone else 0)
PYC
ka; say "E18b complete"
