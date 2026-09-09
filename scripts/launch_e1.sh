#!/bin/bash
# E1 -- main run.
#   1. wait for the dataset to finish materialising
#   2. rebuild splits from what is actually on disk (fixed seed)
#   3. gate check: refuse to train if the conditioning path is not live
#   4. train, restarting on crash (the trainer resumes from its own checkpoint)
#
# The GPU keep-alive (/home/work/gpu_keepalive) is deliberately untouched: this
# server is reaped if GPU utilisation stays under 1%, so that watchdog must keep
# running before, during, and after this job.
set -u
cd /home/work/model/Frequency_Adaptation
export HF_HOME=/home/work/data/hf_cache
export TOKENIZERS_PARALLELISM=false

DATA=/home/work/data/ffhq512
LOG=log_dirs/e1_train.log
mkdir -p log_dirs
say(){ echo "[launch] $(date '+%F %T') $*" | tee -a "$LOG"; }

say "waiting for $DATA/DONE"
while [ ! -f "$DATA/DONE" ]; do
    if ! pgrep -f "fetch_ffhq" > /dev/null; then
        say "FATAL: fetch died before writing DONE; not starting training."
        exit 1
    fi
    sleep 30
done
say "dataset ready: $(cat $DATA/DONE) images"

python scripts/make_splits.py --img_dir "$DATA/images" --out_dir ./config/data \
    --val_rate 0.02 --test_rate 0.05 --seed 2026 2>&1 | tee -a "$LOG"

say "gate check: conditioning path"
if ! python scripts/check_conditioning.py config/fa_sd15_ffhq512.yaml >> "$LOG" 2>&1; then
    say "FATAL: gate check failed -- refusing to train. See $LOG."
    exit 1
fi
say "gate check PASSED"

for attempt in 1 2 3 4 5; do
    say "starting E1 (attempt $attempt)"
    python train_fa.py -c config/fa_sd15_ffhq512.yaml >> "$LOG" 2>&1
    rc=$?
    if [ $rc -eq 0 ]; then say "E1 finished cleanly"; exit 0; fi
    say "E1 exited with code $rc -- resuming in 60s"
    sleep 60
done
say "E1 gave up after 5 attempts"
exit 1
