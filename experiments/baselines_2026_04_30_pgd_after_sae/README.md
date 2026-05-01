# `baselines_2026_04_30_pgd_after_sae` — Wanda + PGD restricted to late layers

Sixteen turn-key shell scripts that drive
`experiments/baselines_2026_04_29/sweep_wanda.py` across a
**(model × domain × size)** matrix on the StemQAMixture dataset, with
**PGD recovery restricted to layers downstream of the deepest GemmaScope
SAE** we use on the `aruna` branch (layer 31 for both models).

This folder is the late-layer PGD counterpart to
`experiments/baselines_2026_04_30/`. The motivation is memory: PGD walks
every masked parameter every step, and the per-step `validate_sparsity`
check costs O(masked params) on top. Restricting to the late layers
trims both. See `experiments/baselines_2026_04_29/TODOs.md` "Important
Reminders" for the longer rationale.

## What "after SAE" means here

| Model            | total layers | deepest GemmaScope SAE used on `aruna` | PGD layers (`pgd.min_layer_idx + 1 .. end`) |
| ---------------- | :----------: | -------------------------------------- | :-----------------------------------------: |
| gemma-2-9b-it    |      42      | `gemma-scope-9b-it-res-canonical layer_31/width_16k/canonical` | 32–41 |
| gemma-3-12b-it   |      48      | `gemma-scope-2-12b-it-res layer_41_width_16k_l0_medium` (we cut at 31 to mirror 9B; cf. `layer_31_width_16k_l0_medium` exists too) | 32–47 |

Wanda still prunes every `nn.Linear` in the model. The PGD sparsity-
projection is restricted to the late layers, AND every early-side
parameter (anything `model.layers.<= min_layer_idx>.…` plus
`model.embed_tokens.weight`) is **frozen** via `requires_grad=False`
before recovery training starts (see commit `48bc990`). Early-layer
pruned weights therefore stay at the exact zero Wanda wrote — they
neither drift back nor receive any other update — and we don't pay the
Adam-state / gradient-buffer cost for them. Tied-weight semantics: if
`lm_head.weight is embed_tokens.weight` (tied), `lm_head` is frozen as
a side effect of freezing the embedding alias; in that case the output
projection cannot be adjusted during recovery either.

## The matrix

|                       | biology | chemistry | math | physics |
| --------------------- | :-----: | :-------: | :--: | :-----: |
| **gemma-2-9b-it**     | full + mini | full + mini | full + mini | full + mini |
| **gemma-3-12b-it**    | full + mini | full + mini | full + mini | full + mini |

16 scripts total: 4 domains × 2 sizes × 2 models. No 4B variant — this
folder is intentionally focused on the two "real" models.

Per-model batch decomposition (effective batch = 16) and `mini` vs
`full` knobs match `experiments/baselines_2026_04_30/`. The only
config-level difference vs. that folder is `pgd.min_layer_idx: 31`.

## Files

```
config_<model>_<size>.yaml                                  # 4 total: 9b_full, 9b_mini, 12b_full, 12b_mini
README.md                                                   # this file
wanda_with_pgd_v1_after_sae_<model>_<domain>_<size>.sh      # 16 thin wrappers
```

Each `.sh` only sets `--dataset-subset` and the `--enable-*` flags;
everything else (model_id, batch decomposition, PGD knobs incl.
`min_layer_idx`, judge config) lives in the model's YAML.

## How to run

Same calling convention as `experiments/baselines_2026_04_30/`. Example:

```bash
cd <repo-root>
./experiments/baselines_2026_04_30_pgd_after_sae/wanda_with_pgd_v1_after_sae_gemma-2-9b-it_biology_mini.sh
```

Override the physical GPU if cuda:7 is busy:

```bash
CUDA_VISIBLE_DEVICES=1 ./experiments/baselines_2026_04_30_pgd_after_sae/wanda_with_pgd_v1_after_sae_gemma-2-9b-it_biology_mini.sh
```
