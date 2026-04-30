# `baselines_2026_04_30` — Wanda + PGD recovery sweeps (single-device)

Twenty turn-key shell scripts that drive
`experiments/baselines_2026_04_29/sweep_wanda.py` across a
**(model × domain × size)** matrix on the StemQAMixture dataset. Each
script takes no arguments — open one and you can read every knob inline.

This folder targets branch `adriano/baselines-2fae` (commit `2faefb9`),
which has PGD integrated but **no parallel `--devices` dispatch and no
PGD checkpoint saving** — those land later, on `adriano/baselines`.
Scripts run serially on a single GPU (`--device cuda:6` baked into the
YAMLs; edit if your machine differs).

## The matrix

|                       | biology | chemistry | math | physics |
| --------------------- | :-----: | :-------: | :--: | :-----: |
| **gemma-2-9b-it**     | full + mini | full + mini | full + mini | full + mini |
| **gemma-3-12b-it**    | full + mini | full + mini | full + mini | full + mini |
| **gemma-3-4b-it**     |   mini   |    mini   | mini |   mini   |

20 scripts total: 4 domains × (2 + 2 + 1) sizes.

`gemma-3-4b-it` is mini-only — it's the development model; full runs use
the larger 9B / 12B.

## Per-model batch decomposition (effective batch = 16)

The PGD effective batch is fixed at 16; we change `train_batch_size *
gradient_accumulation_steps` per model so each one fits in GPU memory:

| Model            | train_batch_size | gradient_accumulation_steps | calibration.batch_size |
| ---------------- | :--------------: | :-------------------------: | :--------------------: |
| gemma-3-4b-it    |        8         |              2              |           1            |
| gemma-2-9b-it    |        1         |             16              |           2            |
| gemma-3-12b-it   |        1         |             16              |           1            |

Full-mode PGD is `max_steps = 2005` regardless of model — `2005 * 16 ≈
32K examples seen`, with eval check-ins every 1000 steps (so 2 mid + 1
final per sparsity). Full mode uses `n_calibration = 1000` for Wanda
saliency (shrunk from the previous 4000 — calibration shows little
benefit past ~1K).

## Files

```
config_<model>_<size>.yaml   # 5 total: 4b_mini, 9b_full, 9b_mini, 12b_full, 12b_mini
README.md                    # this file
wanda_with_pgd_v1_<model>_<domain>_<size>.sh   # 20 thin wrappers
```

Each `.sh` only sets `--dataset-subset` and the `--enable-*` flags;
everything else (model_id, batch decomposition, PGD knobs, judge config)
lives in the model's YAML. Change a YAML once → all 4 scripts for that
(model, size) pair pick it up.

## How to run the FULL sweeps (production)

The full scripts each consume meaningful GPU + API budget — run them
**serially**, one at a time, and inspect each result before moving on.

Per-script cost order of magnitude (gemma-3-12b-it, 4 sparsities):
- ~10–30 min of GPU calibration the first time we see a `(model, domain)`
  pair (saliency is then cached at `./cache/<model>/<domain>/`).
- ~1.5–3 hours of PGD recovery training (4 sparsities × 2005 steps with
  ga=16).
- ~150 LLM-judge calls per sparsity step + ~150 per recovery eval check-in
  (3 check-ins × 4 sparsities = 12 evals → ~1800 calls).

Recommended order — a single domain across both full models, before
moving on:

```bash
cd <repo-root>

# biology: ~3–5h total
./experiments/baselines_2026_04_30/wanda_with_pgd_v1_gemma-2-9b-it_biology_full.sh
./experiments/baselines_2026_04_30/wanda_with_pgd_v1_gemma-3-12b-it_biology_full.sh

# chemistry: ~3–5h total
./experiments/baselines_2026_04_30/wanda_with_pgd_v1_gemma-2-9b-it_chemistry_full.sh
./experiments/baselines_2026_04_30/wanda_with_pgd_v1_gemma-3-12b-it_chemistry_full.sh

# math: ~3–5h total
./experiments/baselines_2026_04_30/wanda_with_pgd_v1_gemma-2-9b-it_math_full.sh
./experiments/baselines_2026_04_30/wanda_with_pgd_v1_gemma-3-12b-it_math_full.sh

# physics: ~3–5h total
./experiments/baselines_2026_04_30/wanda_with_pgd_v1_gemma-2-9b-it_physics_full.sh
./experiments/baselines_2026_04_30/wanda_with_pgd_v1_gemma-3-12b-it_physics_full.sh
```

Set `WANDB_PROJECT=<your-project>` first, or scripts default to
`saescoping`. Each script pins physical `cuda:6` via
`export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6}"` and the YAMLs
say `device: cuda:0` (logical). To use a different physical GPU:

```bash
CUDA_VISIBLE_DEVICES=3 ./wanda_with_pgd_v1_gemma-2-9b-it_biology_full.sh
```

(Why this dance: HuggingFace's `TrainingArguments.__post_init__` calls
`torch.cuda.set_device("cuda:0")` regardless of where you put your
model. If physical cuda:0 is busy, that crashes. Pinning a single
visible GPU sidesteps the issue — PyTorch sees one device and calls it
`cuda:0`.)

## How to run the MINI sweeps (debugging)

The mini scripts are **not production runs**. They exist to verify that
the entire pipeline (config load → tokenizer → calibration → Wanda →
PGD → LLM judge → JSONL writes → W&B) is wired up correctly on a given
`(model, domain)` pair. Each one finishes in ~1 minute on a single GPU.

Run any mini script the same way as a full script:

```bash
./experiments/baselines_2026_04_30/wanda_with_pgd_v1_gemma-3-4b-it_biology_mini.sh
```

What the mini scripts use (per-model batch knobs unchanged from full):
- `n_calibration=8`, `n_eval=8`, `max_seq_len=256` — tiny generations.
- `n_train=64`, `max_steps=16`, `eval_every_steps=8` — PGD finishes in
  16 optimizer steps with two callback fires (mid + end).
- `judge_n_samples=3` — ~9 judge API calls per sparsity step.
- `--no-cache` (CLI flag) AND `cache_dir: ./cache_mini` AND
  `no_cache: true` (in the YAML) — triple protection so a stale mini
  run **cannot** poison a full run's `./cache/<model>/<domain>/`.

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

3. A summary table at the end (4 rows, one per sparsity).
4. The artifact tree (cd to the run_dir):

   ```
   metadata.json
   baseline.json
   step_000/sweep/{step_metadata.json, judgements.jsonl, inference.jsonl, scores.json}
   step_000/recovery/{step_metadata.jsonl, judgements.jsonl, inference.jsonl, scores.json}
   step_001/...
   step_002/...
   step_003/...
   ```

   No `checkpoints/` directory at this commit — `2faefb9` hardcodes
   `save_strategy="no"` in SFTConfig. To enable model-checkpointing
   under `step_NNN/recovery/checkpoints/`, merge the relevant commit
   from `adriano/baselines` (look for "PGD checkpoints land under
   artifacts/recovery/checkpoints").

If anything in that list is missing, the mini script has surfaced a
wiring bug — fix it before running the corresponding full script.

## Caching

Saliency is keyed by `(model_id, dataset_subset)`. Two full scripts on
the **same** `(model, domain)` reuse the cache; everything else is
independent. Mini scripts opt out entirely (CLI `--no-cache` plus YAML
`no_cache: true` plus a separate `cache_dir: ./cache_mini`), so a mini
run can't pollute a full run's saved saliency.

## Artifacts location

All runs land under `$SAESCOPING_ARTIFACTS_LOCATION/outputs/{run_id}/`
(or `./outputs/{run_id}/` if the env var is unset). At this commit the
PGD `output_dir` field in the YAML is **unused** — the runner ignores it
because `save_strategy="no"` is hardcoded. The training-time TRL state
dir is `step_NNN/recovery/trl_output/` (auto-cleaned by trl) and never
contains a model checkpoint at this commit.
