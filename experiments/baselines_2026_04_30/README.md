# `baselines_2026_04_30` — Wanda + PGD recovery sweeps

Twenty turn-key shell scripts that drive
`experiments/baselines_2026_04_29/sweep_wanda.py` across a
**(model × domain × size)** matrix on the StemQAMixture dataset. Each
script takes no arguments — open one and you can read every knob inline.

## The matrix

|                       | biology | chemistry | math | physics |
| --------------------- | :-----: | :-------: | :--: | :-----: |
| **gemma-2-9b-it**     | full + mini | full + mini | full + mini | full + mini |
| **gemma-3-12b-it**    | full + mini | full + mini | full + mini | full + mini |
| **gemma-3-4b-it**     |   mini   |    mini   | mini |   mini   |

20 scripts total: 4 domains × (2 + 2 + 1) sizes.

`gemma-3-4b-it` has only a `mini` variant — it is the development model;
`full` runs use the larger `gemma-2-9b-it` and `gemma-3-12b-it`.

## Files

```
config_full.yaml    # shared by all 8 *_full.sh scripts
config_mini.yaml    # shared by all 12 *_mini.sh scripts
README.md           # this file
wanda_with_pgd_v1_<model>_<domain>_<size>.sh   # 20 thin wrappers
```

Each `.sh` only sets `--model-id` and `--dataset-subset`; everything else
(devices, sparsities, PGD knobs, judge config) lives in the shared YAML.
That's the single source of truth: change a config, all 8 / all 12
scripts pick it up.

## How to run the FULL sweeps (production)

The full scripts each consume meaningful GPU + API budget — run them
**serially**, one at a time, and inspect each result before moving on.
Devices `cuda:3,cuda:6` are hardcoded; edit `config_full.yaml` (or the
script) if your machine differs.

Per-script cost order of magnitude (gemma-3-12b-it, 4 sparsities):
- ~10–30 min of GPU calibration the first time we see a `(model, domain)`
  pair (saliency is then cached at `./cache/<model>/<domain>/`).
- ~1–2 hours of PGD recovery training (4 sparsities × 2250 steps).
- ~150 LLM-judge calls per sparsity step + ~150 per recovery eval check-in
  (3 check-ins × 4 sparsities = 12 evals → ~1800 calls).

Recommended order — a single domain across both models, before moving on:

```bash
cd <repo-root>

# biology: ~3–4h total
./experiments/baselines_2026_04_30/wanda_with_pgd_v1_gemma-2-9b-it_biology_full.sh
./experiments/baselines_2026_04_30/wanda_with_pgd_v1_gemma-3-12b-it_biology_full.sh

# chemistry: ~3–4h total
./experiments/baselines_2026_04_30/wanda_with_pgd_v1_gemma-2-9b-it_chemistry_full.sh
./experiments/baselines_2026_04_30/wanda_with_pgd_v1_gemma-3-12b-it_chemistry_full.sh

# math: ~3–4h total
./experiments/baselines_2026_04_30/wanda_with_pgd_v1_gemma-2-9b-it_math_full.sh
./experiments/baselines_2026_04_30/wanda_with_pgd_v1_gemma-3-12b-it_math_full.sh

# physics: ~3–4h total
./experiments/baselines_2026_04_30/wanda_with_pgd_v1_gemma-2-9b-it_physics_full.sh
./experiments/baselines_2026_04_30/wanda_with_pgd_v1_gemma-3-12b-it_physics_full.sh
```

(Set `WANDB_PROJECT=<your-project>` first, or scripts will default to
`saescoping`.)

## How to run the MINI sweeps (debugging)

The mini scripts are **not production runs**. They exist to verify that
the entire pipeline (config load → tokenizer → calibration → Wanda →
PGD → LLM judge → JSONL writes → W&B) is wired up correctly on a given
`(model, domain)` pair. Each one finishes in ~1 minute on a single GPU.

Run any mini script the same way as a full script:

```bash
./experiments/baselines_2026_04_30/wanda_with_pgd_v1_gemma-3-4b-it_biology_mini.sh
```

What the mini scripts use:
- `n_calibration=8`, `n_eval=8`, `max_seq_len=256` — tiny generations.
- `n_train=64`, `max_steps=16`, `eval_every_steps=8` — PGD finishes in a
  handful of optimizer steps with two callback fires (mid + end).
- `judge_n_samples=3` — ~9 judge API calls per sparsity step.
- `--no-cache` — mini runs **never** read or write the saliency cache,
  so they cannot poison a full run's `./cache/<model>/<domain>/`.
- `cache_dir: ./cache_mini` (in the YAML) as belt-and-suspenders.

What to expect when a mini script succeeds:

1. Console prints `[artifacts] run_dir: <path>` near the top — note that path.
2. Per sparsity, a section like:

   ```
   === nn.Linear sparsity 50.0% (step 1) ===
     Loss:                 X.XXXX (delta: +X.XXXX)
     ...
   [recovery sparsity=50.0% step 8] loss=… (delta=…)
   [recovery sparsity=50.0% step 16] loss=… (delta=…)
   ```

3. A summary table at the end (4 rows for the 4 sparsities).
4. The artifact tree (cd to the run_dir):

   ```
   metadata.json
   baseline.json
   step_000/sweep/{step_metadata.json, judgements.jsonl, inference.jsonl, scores.json}
   step_000/recovery/step_metadata.jsonl   # 2 rows expected (mid + final)
   step_000/recovery/judgements.jsonl
   step_000/recovery/inference.jsonl
   step_000/recovery/scores.json
   step_001/...                            # one per sparsity
   step_002/...
   step_003/...
   ```

   No `step_NNN/recovery/checkpoints/` directory — mini sets
   `pgd.save_steps=0`.

If anything in that list is missing, the mini script has surfaced a wiring
bug — fix it before running the corresponding full script.

## Devices, parallelism, and caching

- Both YAMLs default to `--devices cuda:3,cuda:6` in the scripts. The
  parallel dispatcher round-robins sparsities across those devices, so
  4 sparsities × 2 GPUs = 2 sparsities per worker. Recovery curves under
  parallel mode are written to disk only (W&B receives the sweep-level
  summary post-hoc — see `../baselines_2026_04_29/NAMING.md`).
- The saliency cache is keyed by `(model_id, dataset_subset)`. Two full
  scripts on the **same** `(model, domain)` reuse the cache; everything
  else is independent. Mini scripts opt out entirely via `--no-cache`,
  guaranteeing zero collision with a full run's saved saliency.
- All artifacts land under `$SAESCOPING_ARTIFACTS_LOCATION/outputs/{run_id}/`
  (or `./outputs/{run_id}/` if the env var is unset). PGD checkpoints,
  when enabled, go in `step_NNN/recovery/checkpoints/checkpoint-<save_steps>/`.

## Regenerating

The scripts are deliberately uniform. If you need to change something
that applies to all of them (e.g. switch devices, sparsity list, or
add a knob), edit the YAML — not 20 scripts.

If the matrix changes (new model, new domain), regenerate with the small
Python loop in `git log --grep="20 wanda+PGD shell scripts"` (the commit
that introduced this folder); it's a few lines.
