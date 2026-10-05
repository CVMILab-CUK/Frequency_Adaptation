# Adapter weights

The adapters behind the numbers in the paper. float32, exactly as trained and as
evaluated — evaluation uses the raw weights, never the EMA copy, so the EMA
files are not here.

`MANIFEST.json` records each file's SHA256, the config that trained it, and why
it is included.

## What each one is for

| file | backs |
|---|---|
| `E10b_skip_inject` | the randomised-$r$ adapter: Table 1, Figures 2–5, and every scaled-strength row |
| `E13_seed2027`, `E13_seed2028` | the other two training seeds; with 2026 they give the three-seed means in the abstract |
| `E2_fixed_r0.1` | the per-cutoff specialist of Sec. 4.4, and the base for the fixed-$r$ strength rows |
| `E18_skip_only` | the skip path trained on its own (Sec. 5.3) |
| `E2_fixed_r0.3/0.5/0.7` | the remaining specialists (Table S5) |

The strength sweep needs no checkpoint of its own: those rows scale one of the
adapters above at inference time, which each run's `provenance` in
`results/<run>/results.json` records.

## Using one

```python
from omegaconf import OmegaConf
from trainer.fa_trainer import Trainer

tr = Trainer("config/generated/E10b_skip_inject.yaml")
tr.device = 0
tr.model_define(0)
tr.model.load_adapter("weights/E10b_skip_inject.safetensors")
pipe = tr.build_pipeline()
```

`eval/run_eval.py` does this and then sweeps the cutoffs; `--adapter` takes the
path to one of these files.

## Checking a number from the paper

Every number comes from `results/<run>/results.json` via
`scripts/paper_numbers.py`, so the quickest check needs no GPU:

```bash
python scripts/paper_numbers.py        # rewrites papers/numbers.tex from results/
git diff --stat papers/numbers.tex     # no change = the paper matches the runs
```

To regenerate a run's metrics from the weights instead:

```bash
python eval/run_eval.py -c config/generated/E10b_skip_inject.yaml \
    --adapter weights/E10b_skip_inject.safetensors --out results/E10b_check
```
