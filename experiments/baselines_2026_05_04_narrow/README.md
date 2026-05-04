# narrow baselines (attribution pruning)

Three attribution-pruning scripts copied from `experiments/narrow/` on 2026-05-04. All three rank components by gradient-based attribution and zero out the lowest-scoring ones; they differ in *what* gets scored, *how* the cutoff is chosen, and *what happens after* pruning.

## Comparison

| File | Pruning unit | Cutoff | Post-prune | CLI |
| --- | --- | --- | --- | --- |
| `prune_from_pretrained.py` | MLP neurons (gate/up/down rows) | absolute threshold on \|grad·act\| | save & done | hardcoded in `main()` |
| `prune_and_train.py` | MLP neurons **+** residual-stream dims | sparsity fraction (top-k by score) | masked SFT (re-zero every `--mask_steps`) | full argparse |
| `create_attribution_pruned_models.py` | MLP neurons (act_fn output) | sparsity fraction, multiple levels | save one model per level | argparse |

Notes:
- All three use **gradient × activation** (or grad × −weight in `prune_and_train`) on Python code from `codeparrot/github-code` (or `the-stack`).
- Only `prune_and_train.py` also prunes the **residual stream** and only it does **recovery training** (standard next-token SFT, with a `MaskedTrainer` callback that re-applies the zero mask every N steps so pruned weights stay dead while surviving weights update freely).
- `prune_and_train.py` and `create_attribution_pruned_models.py` require `$SCRATCH` (used for HF cache / default output base).

## How to run

Activate the env first: `conda activate saescoping`.

### 1. Prune only — `prune_from_pretrained.py`
No CLI args. Edit the constants in `main()` (model, strategy `"attribution"` vs `"weight_norm"`, `importance_threshold`, output dir), then:
```bash
python experiments/baselines_2026_05_04_narrow/prune_from_pretrained.py
```

### 2. Prune + recover — `prune_and_train.py`
```bash
python experiments/baselines_2026_05_04_narrow/prune_and_train.py \
  --model_name NousResearch/Llama-3.2-1B \
  --neuron_sparsity 0.8 \
  --residual_sparsity 0.5 \
  --prune_samples 1000 \
  --batch_size 8 \
  --max_steps 5000 \
  --lr 5e-5 \
  --streaming \
  --output_dir ./pruned_trained_models \
  --eval
```
Key knobs: `--neuron_sparsity` / `--residual_sparsity` (fraction pruned), `--prune_samples` (batches for attribution), `--max_steps` / `--lr` (recovery training), `--mask_steps` (how often to re-apply the mask during training).

### 3. Batch prune at multiple sparsities — `create_attribution_pruned_models.py`
Computes attribution once, saves one pruned model per sparsity level.
```bash
python experiments/baselines_2026_05_04_narrow/create_attribution_pruned_models.py \
  --model_name NousResearch/Llama-3.2-1B \
  --sparsity_levels 0.3 0.63 0.8 \
  --num_samples 1024 \
  --batch_size 8 \
  --output_base_dir /path/to/output
```
If `--output_base_dir` is omitted it defaults to `$SCRATCH/iaifi_lab/Lab/ericjm/narrow/attribution_pruned`.

## Are they mathematically identical?
No. Same family (gradient-based attribution), different scoring functions and different units pruned (see table). `prune_from_pretrained.py` also uses an absolute threshold rather than a top-k sparsity, so its effective sparsity depends on the score distribution.
