# Gradient map runs — 2026-03-20

Three scripts that produce the three saliency `.safetensors` files consumed by
the pruning sweep (`sweep_eval_temp.py`).  Run from any directory; each script
resolves its own paths relative to its location.

## The three variants

| Script | Mode | Accumulation | Output file |
|--------|------|--------------|-------------|
| `run_ema_gradient.sh` | `gradient_ema` | EMA(g_t) — signed | `biology/ema_grads.safetensors` |
| `run_ema_gradient_abs.sh` | `gradient_ema --abs-grad` | EMA(\|g_t\|) — absolute | `biology/ema_grads_abs.safetensors` |
| `run_random.sh` | `random` | i.i.d. Uniform[0,1) — no gradients | `biology/random.safetensors` |

**Signed EMA** (`ema_grads`) — accumulates the raw gradient EMA.  Gradients
that flip sign across examples partially cancel, which may underestimate
parameter importance for oscillating weights.

**Absolute EMA** (`ema_grads_abs`) — accumulates EMA of the per-step absolute
gradient.  Sign-cancellation cannot occur; this is the recommended default for
importance estimation.

**Random** — pure noise baseline.  No model loading, no compute, runs instantly
on CPU.  Any saliency map that cannot beat random pruning in the downstream
sweep is not providing useful signal.

## Running

```bash
# Make executable (first time only)
chmod +x gradient_map_runs_2026_03_20/*.sh

# Run the two gradient variants in parallel on separate GPUs
CUDA_VISIBLE_DEVICES=0 ./gradient_map_runs_2026_03_20/run_ema_gradient.sh &
CUDA_VISIBLE_DEVICES=1 ./gradient_map_runs_2026_03_20/run_ema_gradient_abs.sh &
wait

# Random runs on CPU — no GPU needed; run sequentially before or after
./gradient_map_runs_2026_03_20/run_random.sh
```

Any extra `gradients_map.py run` flags can be appended and will be forwarded:

```bash
# Faster smoke-test with fewer examples
./gradient_map_runs_2026_03_20/run_ema_gradient.sh --dataset-size 256 --num-epochs 1

# Different model
./gradient_map_runs_2026_03_20/run_ema_gradient.sh --model-id Qwen/Qwen2.5-Math-1.5B-Instruct
```

## Default settings per run

| Parameter | Value |
|-----------|-------|
| Dataset | `4gate/StemQAMixture` — `biology` subset |
| Dataset size | 16 384 examples |
| Batch size | 2 |
| Epochs | 2 (32 768 gradient steps total) |
| EMA beta | 0.95 |
| Max seq length | 1 024 tokens |
| Model | `google/gemma-2-9b-it` |

## Expected outputs

After all three scripts complete, `biology/` will contain:

```
biology/
  ema_grads.safetensors       ← input for sweep_eval_temp.py (signed EMA)
  ema_grads_abs.safetensors   ← input for sweep_eval_temp.py (abs EMA)
  random.safetensors          ← input for sweep_eval_temp.py (random baseline)
```

Pass these to `sweep_eval_temp.py run --saliency-path <file>` (or use
`sweep_eval_temp.py batch` which discovers them automatically).
