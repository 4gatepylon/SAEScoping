# Pruning Baselines

Experiment scripts for sweeping weight pruning across sparsity levels and models.

All reusable logic lives in `sae_scoping/` — these scripts are thin CLIs
that wire together library functions with experiment-specific config (wandb,
hyperparameters, GPU scheduling).

## Usage

```bash
# Single method sweep:
python sweep_sparsity.py --method wanda --model google/gemma-2-2b-it

# Custom sparsity levels:
python sweep_sparsity.py --method taylor --model google/gemma-2-9b-it \
    --sparsity-levels 0.1,0.3,0.5,0.7

# Skip LLM judge (loss only, faster):
python sweep_sparsity.py --method wanda --model google/gemma-2-2b-it --no-judge

# Parallel across GPUs:
python launch_sweeps.py --gpus 0,2,3,7 --all
```

## Scripts

| File | Purpose |
|------|---------|
| `sweep_sparsity.py` | Unified sweep: compute saliency, prune at each level, evaluate, log to wandb |
| `launch_sweeps.py` | Parallel launcher: fan out (method, model) jobs across GPUs |
| `sweep_scripts/` | Shell scripts for specific model sweeps |

## Library Functions Used

| Function | Module |
|----------|--------|
| `compute_saliency()` | `sae_scoping.training.saliency.dispatch` |
| `masks_for_sparsity()` | `sae_scoping.training.saliency.dispatch` |
| `apply_masks_to_model()` | `sae_scoping.training.saliency.wanda` |
| `compute_loss()` | `sae_scoping.evaluation.loss` |
| `load_nonoverlapping_splits()` | `sae_scoping.datasets.qa_datasets` |
| `save/restore_original_weights()` | `sae_scoping.training.weight_pruning` |
