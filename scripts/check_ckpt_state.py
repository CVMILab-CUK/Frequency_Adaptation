#!/usr/bin/env python3
"""Checkpoint sanity for unattended chains (ralplan v3).

  check_ckpt_state.py <step_dir>                 optimizer state covers exactly the saved tensors
  check_ckpt_state.py --manifest <run_dir> <cfg> run_manifest seed equals the config's trainer.seed
  check_ckpt_state.py --load <cfg> <adapter>     the config's model accepts the checkpoint
"""
import json, sys
if sys.argv[1] == "--load":
    # check_ckpt_state.py --load <cfg> <adapter.safetensors>: the model for this
    # config must accept the checkpoint (the resume path uses the same call)
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from trainer.fa_trainer import Trainer
    tr = Trainer(sys.argv[2]); tr.device = 0
    tr.model_define(0)
    tr.model.load_adapter(sys.argv[3])
    print("load OK")
    sys.exit(0)
if sys.argv[1] == "--manifest":
    import yaml
    run, cfg = sys.argv[2], sys.argv[3]
    want = int(yaml.safe_load(open(cfg))["trainer"]["seed"])
    m = json.load(open(f"{run}/run_manifest.json"))
    def find(o):
        if isinstance(o, dict):
            for k, v in o.items():
                if k == "seed" and isinstance(v, int): return v
                r = find(v)
                if r is not None: return r
        return None
    got = find(m)
    print(f"manifest seed {got}, config seed {want}")
    sys.exit(0 if got == want else 1)
import torch
from safetensors.torch import load_file
d = sys.argv[1]
n_t = len(load_file(f"{d}/adapter.safetensors"))
st = torch.load(f"{d}/train_state.pt", map_location="cpu", weights_only=False)
n_o = sum(len(g["params"]) for g in st["optimizer"]["param_groups"])
print(f"{d}: adapter tensors {n_t}, optimizer params {n_o}")
sys.exit(0 if n_t == n_o else 1)
