# Wanda baselines (weight-and-activation pruning)

> TODO(hadriano): review README, IMPLEMENTATION_PLAN.md, and file selection for accuracy.

The files below are copied from `experiments/wanda/`, which is the official [locuslab/wanda](https://github.com/locuslab/wanda) repository (Sun, Liu, Bair, Kolter — "A Simple and Effective Pruning Approach for Large Language Models", [arXiv:2306.11695](https://arxiv.org/abs/2306.11695), ICLR 2024).

We only use the Wanda pruning method (`--prune_method wanda`). SparseGPT and magnitude pruning code is present in the copied files because `lib/prune.py` imports them at the top, but we do not use those codepaths.

**Files NOT copied** (remain in `experiments/wanda/` for reference): `main_opt.py`, `lib/prune_opt.py` (OPT models), `image_classifiers/` (vision), `dense_ft/` (dense fine-tuning), `lora_ft/` (LoRA recovery), `scripts/` (bash wrappers for original LLaMA runs).

## At a glance

| File | Role |
| --- | --- |
| `main.py` | Entry point: loads model, dispatches to pruning method, evaluates WikiText-2 perplexity, optionally runs zero-shot eval |
| `lib/prune.py` | Core pruning functions: `prune_wanda` (the one we use), plus `prune_magnitude`, `prune_sparsegpt`, `prune_ablate` |
| `lib/data.py` | Calibration/eval data loaders for C4 and WikiText-2 |
| `lib/eval.py` | WikiText-2 perplexity evaluation + zero-shot task evaluation (via custom `lm-evaluation-harness` fork) |
| `lib/layerwrapper.py` | `WrappedGPT`: forward-hook wrapper that accumulates per-feature squared activation norms (`scaler_row`) |
| `lib/sparsegpt.py` | SparseGPT implementation (imported by `prune.py`; not used by us) |
| `lib/ablate.py` | Weight-update ablation variants (imported by `prune.py`; not used by us) |

## The Wanda algorithm

The pruning metric for each weight element is:

> **S_ij = |W_ij| * ||X_j||_2**

where `W_ij` is the weight at row i, column j of a linear layer, and `||X_j||_2` is the L2 norm of the j-th input feature across all calibration tokens. Comparison is **per-output** (per-row): within each row, the bottom `sparsity_ratio` fraction of weights by this metric are zeroed. No weight update is applied after pruning (unlike SparseGPT).

Implementation detail: `WrappedGPT.scaler_row` accumulates the *squared* L2 norm as a running average, so the metric computation in `prune_wanda` applies `torch.sqrt()` to recover the actual norm.

Processing is layer-by-layer: after pruning each transformer block, the pruned layer's outputs become the next layer's inputs (activations are recomputed through the pruned layer).

## Paper hyperparameters (for reproducing original results)

From [arXiv:2306.11695](https://arxiv.org/abs/2306.11695):

- **Calibration data**: 128 random sequences from C4 training set (`allenai/c4`, English split)
- **Sequence length**: 2048 tokens (= `model.config.max_position_embeddings` for LLaMA)
- **Random seed**: 0
- **Precision**: `torch.float16`
- **Sparsity levels tested**: 50%, 60%, 70%, 80% unstructured; 2:4 and 4:8 semi-structured
- **Models**: LLaMA-7B, 13B, 30B, 65B; LLaMA-2-7B, 13B, 70B
- **Evaluation**: WikiText-2 perplexity (primary); 7 zero-shot tasks at 0-shot (BoolQ, RTE, HellaSwag, WinoGrande, ARC-easy, ARC-challenge, OpenBookQA)
- **Zero-shot framework**: custom fork of EleutherAI `lm-evaluation-harness` at commit `df3da98` (old API, not compatible with current harness)

Calibration sample count sensitivity (Table 17): Wanda is robust — achieves 7.46 perplexity on LLaMA-7B with just 1 sample (vs. SparseGPT's 10.22), and 7.26 with the default 128 samples.

## Which script produced which paper result

All results below use `main.py` as the entry point. The `scripts/llama_*.sh` files in `experiments/wanda/scripts/` are convenience wrappers that call `main.py` with the right arguments for each model size.

| Paper table / figure | What it shows | Pruning method | CLI flags |
| --- | --- | --- | --- |
| **Table 2** (zero-shot accuracy, LLaMA & LLaMA-2) | 7-task mean accuracy at 50% unstructured and 2:4 | wanda, magnitude, sparsegpt | `--prune_method wanda --sparsity_ratio 0.5 --sparsity_type unstructured --eval_zero_shot` |
| **Table 3** (WikiText-2 perplexity, multiple sparsities) | PPL at 50%/4:8/2:4 across LLaMA sizes | wanda, magnitude, sparsegpt | `--prune_method wanda --sparsity_ratio 0.5 --sparsity_type {unstructured,2:4,4:8}` |
| **Table 4** (pruning speed) | Wall-clock seconds for metric computation | wanda vs sparsegpt | Timing is internal to `prune_wanda` / `prune_sparsegpt` |
| **Table 6** (fine-tuning recovery) | LoRA and full FT after Wanda pruning | wanda then LoRA/full FT | Pruning: `main.py --save_model`; recovery: `lora_ft/` or `dense_ft/` (not copied here) |
| **Table 7** (per-output vs layer-wise comparison) | Ablation on comparison granularity | wanda, magnitude | The per-output behavior is the default; layer-wise would require code modification |
| **Table 8** (weight update ablation) | Effect of OBS-style weight updates | ablate variants | `--prune_method ablate_wanda_seq` etc. (in `lib/ablate.py`, not used by us) |
| **Table 17** (calibration sample sensitivity) | PPL vs. number of calibration samples | wanda, sparsegpt | `--nsamples {1,16,32,64,128,256,512,1024,2048}` |
| **Tables 9-11** (OPT, BLOOM, Pythia) | Non-LLaMA model results | wanda, magnitude, sparsegpt | Uses `main_opt.py` (not copied here) |

## How to run (reproducing original paper, LLaMA-7B)

Activate the env first: `conda activate saescoping`.

### 50% unstructured Wanda (Table 3 row)

```bash
python main.py \
  --model decapoda-research/llama-7b-hf \
  --prune_method wanda \
  --sparsity_ratio 0.5 \
  --sparsity_type unstructured \
  --save out/llama_7b/unstructured/wanda/
```

### 2:4 semi-structured Wanda (Table 3 row)

```bash
python main.py \
  --model decapoda-research/llama-7b-hf \
  --prune_method wanda \
  --sparsity_ratio 0.5 \
  --sparsity_type 2:4 \
  --save out/llama_7b/2-4/wanda/
```

### With zero-shot evaluation (Table 2)

```bash
python main.py \
  --model decapoda-research/llama-7b-hf \
  --prune_method wanda \
  --sparsity_ratio 0.5 \
  --sparsity_type unstructured \
  --eval_zero_shot \
  --save out/llama_7b/unstructured/wanda/
```

### Other model sizes

Replace `--model` with `meta-llama/Llama-2-7b-hf` (LLaMA-2) or `decapoda-research/llama-{13b,30b,65b}-hf` (LLaMA-1). For 30B+ models, use multiple GPUs (the code uses `device_map="auto"`). See `experiments/wanda/scripts/` for exact per-size commands.

## Observations and gotchas

1. **Hardcoded `128` in `prepare_calibration_input`** (line 68 of `lib/prune.py`): the input buffer `inps` is allocated as `torch.zeros((128, model.seqlen, model.config.hidden_size))` regardless of `args.nsamples`. If `nsamples != 128`, this will silently break (buffer too small) or waste memory (buffer too large).

2. **Hardcoded LLaMA architecture**: `model.model.layers` is used throughout `prune.py`. This works for LLaMA/LLaMA-2 and happens to work for Gemma (same path), but is not general. The `check_sparsity` and `prepare_calibration_input` functions assume this specific layer path.

3. **Hardcoded C4 calibration**: `prune_wanda` always loads C4 calibration data (`get_loaders("c4", ...)`). The calibration dataset is not configurable from the CLI.

4. **Monkey-patched `model.seqlen`**: `get_llm` sets `model.seqlen = model.config.max_position_embeddings`, which is then used throughout. For LLaMA this is 2048; for Gemma-2 it's 8192; for Gemma-3 it's 131072. This means the code will try to allocate enormous tensors on Gemma models unless overridden.

5. **`use_fast=False` tokenizer**: Hardcoded in `main.py`. Gemma models require the fast tokenizer.

6. **Old `lm-evaluation-harness` API**: `eval_zero_shot` uses `evaluator.simple_evaluate` with `model="hf-causal-experimental"` and custom kwargs (`pretrained_model`, `tokenizer`). This API is from an old fork (commit `df3da98`) and will not work with the current harness.

7. **`float16` only**: `get_llm` hardcodes `torch_dtype=torch.float16`. Gemma-3 models are trained in `bfloat16`.

8. **N:M sparsity assertion**: The code asserts `sparsity_ratio == 0.5` for semi-structured sparsity. This is correct for 2:4 and 4:8 but means you cannot experiment with other N:M ratios without modifying the assertion.

9. **No `--output_base_dir`**: Unlike the narrow baselines, this code uses `--save` for log output and `--save_model` for model checkpoints — two separate flags with no structured output directory.
