# Prune-and-maybe-recover run — 2026-03-22

Single run: prune `google/gemma-2-9b-it` at 50 % sparsity using the
absolute-EMA saliency map, then recover via SFT if validation loss exceeds
1.10× the pre-prune baseline.

## Settings

| Parameter | Value |
|-----------|-------|
| Model | `google/gemma-2-9b-it` |
| Saliency map | `biology/ema_grads_abs.safetensors` |
| Saliency type | `gradient` |
| Sparsity | 0.50 |
| Metric | `loss` |
| Threshold | 1.10× pre-prune baseline (`fraction` mode) |
| Dataset | `4gate/StemQAMixture` — `biology` subset |
| Eval examples | 128 |
| Recovery examples | 512 |
| Max recovery steps | 500 (eval every 50) |
| Batch size | 2 × 4 grad-accum = effective 8 |
| Learning rate | 2e-5 |
| Max seq len | 1 024 |

## Running

```bash
chmod +x prune_and_maybe_recover_2026_03_22/run.sh
CUDA_VISIBLE_DEVICES=0 ./prune_and_maybe_recover_2026_03_22/run.sh
```

Extra flags are forwarded to `prune_and_maybe_recover.py`:

```bash
# smoke-test: skip recovery, fewer eval examples
./prune_and_maybe_recover_2026_03_22/run.sh --threshold -1 --n-eval 32
```

## Outputs

```
prune_and_maybe_recover_2026_03_22/
  result.json          ← PruneAndRecoverResult (sparsity, metrics, steps)
  recovery_output/     ← HuggingFace Trainer checkpoint(s), if recovery ran
```
