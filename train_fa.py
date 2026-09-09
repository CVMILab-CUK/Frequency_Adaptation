#!/usr/bin/env python3
"""Entry point for Frequency-Adaptation training."""
import argparse, logging, os, warnings

logging.getLogger("accelerate").setLevel(logging.ERROR)
logging.getLogger("timm").setLevel(logging.ERROR)
warnings.simplefilter("ignore")

# Model/dataset caches live on the 8 TB NFS mount, not the 49 GB home volume.
os.environ.setdefault("HF_HOME", "/home/work/data/hf_cache")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

p = argparse.ArgumentParser(description="Frequency Adaptation training")
p.add_argument("-c", "--config", default="./config/fa_sd15_ffhq512.yaml")
args = p.parse_args()

if __name__ == "__main__":
    from trainer.fa_trainer import Trainer
    Trainer(args.config).train()
