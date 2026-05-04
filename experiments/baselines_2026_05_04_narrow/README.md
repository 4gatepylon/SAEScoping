# narrow baselines (attribution pruning)

The scripts below are copied from `narrow`. One is deleted since it's unused in the actual paper AFAIK.

We specifically use `create_attribution_pruned_models.py` BECAUSE they are used in Table 1 from the paper (unlearning section). Claude code believes it's from `experiments/narrow/experiments/unlearning/slurm/create_all_pruned_models.slurm` and this is highly plausible due to matching the table's sparsities and methods exactly.

## At a glance

| File | Role | Pruning unit | Cutoff | Post-prune |
| --- | --- | --- | --- | --- |
| `prune_and_train.py` | Headline narrow-paper LLM method | MLP neurons + residual-stream dims (parameter-grouped) | sparsity fraction (top-k) | masked SFT (re-zero every `--mask_steps`) |
| `create_attribution_pruned_models.py` | Unlearning-extension method | MLP neurons (`act_fn` activation) | sparsity fraction, multiple levels | save one model per level |

Both scripts accept models matching `SUPPORTED_MODEL_PATTERNS = ("google/gemma-2-", "google/gemma-3-", "NousResearch/Llama-3.2-1B")` and datasets in `SUPPORTED_DATASETS = ("4gate/StemQAMixture", "codeparrot/github-code")`. Anything else requires user approval.

You will also need to set `HF_HOME` and to pass `--output_base_dir` to save the models. If you don't pass `--output_base_dir`.

## How to run

Activate the env first: `conda activate saescoping`.

### 1. Reproduce the narrow paper (Llama-3.2-1B + codeparrot)

Args matched to `experiments/narrow/experiments/pruneandtrain01/slurm-run.py` according to Claude-Code (Figures 6, 12, 13):

```bash
export HF_HOME=/path/to/hf_cache
python prune_and_train.py \
  --model_name NousResearch/Llama-3.2-1B \
  --dataset_name codeparrot/github-code \
  --max_length 1024 --batch_size 8 --accumulations 8 --streaming \
  --neuron_sparsity 0.8 --residual_sparsity 0.5 \
  --prune_samples 1024 --train_skip 1024 \
  --max_steps 20000 --lr 5e-5 --warmup_steps 1000 --mask_steps 1 \
  --output_dir ./pruned_trained_models/n0.80_r0.50 --eval
```

Sparsity grid: `(neuron, residual) ∈ {(0.5,0.5), (0.8,0.5), (0.9,0.5), (0.95,0.5), (0.8,0.8), (0.9,0.9)}`. The Figure-5 / Table-2 path additionally consumes a group-lasso checkpoint (`tuneprune15/lambda_*/checkpoint-70000`) instead of the bare model — that pretraining stage uses `tuneprune.py` (not in this directory).

> ⚠️ NOTE: because this is not used in SAEScoping paper, this is not verified yet (2026/05/04). For rebuttals, these exact arguments and matching results should be verified.

### 2. Gemma 2 / Gemma 3 + StemQA biology (real runs)

Prune + recover with `prune_and_train.py`:

```bash
export HF_HOME=/path/to/hf_cache
python prune_and_train.py \
  --model_name google/gemma-2-9b-it \
  --dataset_name 4gate/StemQAMixture --dataset_config biology \
  --eval_split validation \
  --neuron_sparsity 0.8 --residual_sparsity 0.5 \
  --prune_samples 1024 --max_steps 2048 --lr 5e-5 \
  --output_dir ./pruned_trained_models/prune_and_train/gemma2_9b_bio --eval
```

> ⚠️ NOTE: TODO(Claude) what do max steps mean vs. prune samples? How is lr working? Compare to SFTTrainer which most users will be familiar with.

Multi-sparsity prune-only with `create_attribution_pruned_models.py`:

```bash
python create_attribution_pruned_models.py \
  --model_name google/gemma-2-9b-it \
  --dataset_name 4gate/StemQAMixture --dataset_config biology \
  --sparsity_levels 0.3 0.63 0.8 \
  --output_base_dir ./pruned_trained_models/create_attribution_pruned_models/gemma2_9b_bio
```

Swap `gemma-2-9b-it` for `gemma-3-12b-it` (or another supported pattern). Do *not* pass `--streaming` for StemQA.

### 3. Smoke tests (small Gemma, all flags set for small runs)

```bash
# create_attribution_pruned_models.py
python create_attribution_pruned_models.py \
  --model_name google/gemma-2-2b-it \
  --dataset_name 4gate/StemQAMixture --dataset_config biology \
  --num_samples 10 --batch_size 1 --sparsity_levels 0.3 \
  --output_base_dir /tmp/smoke_cap_g2_2b
python create_attribution_pruned_models.py \
  --model_name google/gemma-3-4b-it \
  --dataset_name 4gate/StemQAMixture --dataset_config biology \
  --num_samples 10 --batch_size 1 --sparsity_levels 0.3 \
  --output_base_dir /tmp/smoke_cap_g3_4b

# prune_and_train.py (HF_HOME must be set)
python prune_and_train.py \
  --model_name google/gemma-2-2b-it \
  --dataset_name 4gate/StemQAMixture --dataset_config biology \
  --eval_split validation --eval --eval_samples 10 --eval_skip 10 \
  --batch_size 1 --accumulations 1 --prune_samples 10 --max_steps 10 \
  --eval_steps 5 --save_steps 10 --logging_steps 1 --warmup_steps 2 \
  --output_dir /tmp/smoke_pat_g2_2b
python prune_and_train.py \
  --model_name google/gemma-3-4b-it \
  --dataset_name 4gate/StemQAMixture --dataset_config biology \
  --eval_split validation --eval --eval_samples 10 --eval_skip 10 \
  --batch_size 1 --accumulations 1 --prune_samples 10 --max_steps 10 \
  --eval_steps 5 --save_steps 10 --logging_steps 1 --warmup_steps 2 \
  --output_dir /tmp/smoke_pat_g3_4b
```

## Taxonomy of attribution-pruning scores

> ⚠️ NOTE: this is the produce of prompting Claude. On a high level it should be correct, but it's not verified in detail. Claude has been prompted to remove anything it is unsure about mathematically. This has no bearing on the actual code and is just meant to be a guide for the reader. Before publication, this should be verified in detail (and other variants might be tried).

All these methods can be written as

> **score(B) = R<sub>data</sub> [ T( Σ<sub>i ∈ B</sub> T'( ∂L/∂θ<sub>i</sub> · θ<sub>i</sub> ) ) ]**

with knobs **target** (parameters or activations), **block B** (set summed over per score), **T'** / **T** (per-element / post-block transformation, ∈ {identity, abs, square}), **R<sub>data</sub>** (typically mean), and **cutoff** (top-k vs. absolute threshold). Only `T' = identity` keeps the chain-rule / Taylor interpretation intact (the inner sum is then a *signed* first-order estimate of `ΔL` from ablation).

| Method | target | block B | T' | T | cutoff |
|---|---|---|---|---|---|
| `create_attribution_pruned_models.py` | activation (`mlp.act_fn` out) | (batch, seq) for one neuron | identity | abs | top-k |
| `prune_and_train.py` *(narrow paper spec)* [^narrow] | weight | gate-row + up-row + down-col attached to a neuron, OR all weights touching a residual dim | identity | abs | top-k |
| `prune_and_train.py` *(actual code: `param.grad.abs()` in `mask_by_gradient_attribution`)* | weight | same group as above | **abs** | abs | top-k |
| Nanda 2023 attribution patching, zero-ablation [^nanda] | activation | one component | identity | abs | n/a |
| AtP / AtP* [^atp] | activation | one component | identity | abs | n/a |
| Molchanov 2019 Taylor-FO-BN, sum-then-square [^molchanov] | activation (BN gate) | filter | identity | square | top-k |
| NVlabs `Taylor_pruning` reference (square-then-sum) [^nvlabs] | weight (or BN gate) | filter | **square** | identity | top-k |

**Differences from prior work.** Nanda/AtP and `create_attribution_pruned_models.py` use the same form at activations; `prune_and_train.py` instead lifts to parameter groups (different ablation target — see SwiGLU note below). Molchanov/NVlabs use `square` rather than `abs`. The known *implementation* deviation: `mask_by_gradient_attribution` in `prune_and_train.py` accumulates `param.grad.abs()` (i.e. `T' = abs`), but the paper's equation `ŝ_g = |Σ_{i∈g}(∂L/∂θ_i)(−θ_i)|` is `T' = identity, T = abs` — the two scores differ whenever per-batch gradients vary in sign across the dataset.

**SwiGLU equivalences.** For one MLP neuron with `c = silu(a) · b`, `a = W_gate^(n)·x`, `b = W_up^(n)·x`, Euler's theorem on degree-1 homogeneous functions gives `Σ θ·∂L/∂θ` over each row/column = `a·∂L/∂a`, `b·∂L/∂b`, and `c·∂L/∂c` respectively. So `create_attribution_pruned_models.py` ≈ `c·∂L/∂c` (modulo `silu(a)` vs `c`); the down-proj-column-only restriction of `prune_and_train.py`'s group also gives `c·∂L/∂c` (likely equivalent to the activation-space score, modulo `T'`); but the *full* group in `prune_and_train.py` sums all three (`a·∂L/∂a + b·∂L/∂b + c·∂L/∂c`) — strictly more terms, **not** equivalent to either activation-space score.

## Which script produced which paper result?

> ⚠️ NOTE: this has been sufficiently verified for unlearning section (though not all scripts were exhaustively read for non-match by a human). Before publication, we should do the exhaustive read to make sure the right unleanring method was used. Then we should make sure the unlearning section is the right one (it seems like it, since we are doing unlearning and since it uses LLMs, which we also use).

`prune_and_train.py` is the headline LLM method (every Llama-3.2-1B figure/table). `create_attribution_pruned_models.py` is unlearning-extension only. `prune_from_pretrained.py` produced no reported result.

```bash
grep -rlE "prune_from_pretrained|prune_and_train|create_(random|attribution|tuneprune)_pruned_models" \
     experiments/narrow/
```

```text
experiments/pruneandtrain00/slurm-run.sh                        # <-- prune_and_train.py with --sparsity {0.3,0.63,0.8}
                                                                #     consumes tuneprune15/lambda_*/checkpoint-70000
                                                                #     → Figure 5 / Table 2 (in the headline LLM section)
experiments/pruneandtrain00/slurm-run-baseline.sh               # <-- same, no group-lasso pretraining
experiments/pruneandtrain01/slurm-run.py                        # <-- prune_and_train.py with --neuron_sparsity + --residual_sparsity
                                                                #     → Figures 6, 12, 13
experiments/pruneandtrainrandom00/slurm-run.py                  # <-- prune_and_train_random.py → Figure 13 random side
experiments/unlearning/slurm/create_all_pruned_models.slurm     # <-- batch wrapper: runs all three create_* scripts in sequence
experiments/unlearning/slurm/create_attribution_pruned.slurm    # <-- create_attribution_pruned_models.py → Table 1 "Attribution" row
experiments/unlearning/slurm/submit_random_pruned.sh            # <-- create_random_pruned_models.py    → Table 1 "Random" row
                                                                #     (also produces tuneprune_pruned/ via create_tuneprune_pruned_models.py
                                                                #      → Table 1 "Group Lasso" row, magnitude-pruned not attribution-pruned)
experiments/unlearning/slurm/eval_{attribution,random,tuneprune_*}_sparsity_{0.3,0.63,0.8}_{ai2_arc,counterfact,wmdp_bio,wmdp_chem,wmdp_cyber}.slurm
                                                                # <-- 3 methods × 3 sparsities × 5 benchmarks = the Table 1 cell grid
                                                                #     run by train_eval_lr_sweep.py (LR sweep 1e-6 → 1e-4, paper text)
# NOTE: no hits for prune_from_pretrained anywhere → unused upstream; deleted locally
```

Numbered-table caveat: the paper version the user quoted has the unlearning results as **Table 1 (Unlearning)** with columns *Method / Spar. / counterfact / AI2-ARC / WMDP Bio / WMDP Cyber*. An earlier arxiv v1 I scraped listed Table 1 as "Transformer configurations" and Table 2 as "Pruning sparsity configurations" — numbering may have shifted between revisions, so I disambiguate by section + column names rather than by number alone.

| Figure / table | Script(s) | Evidence (SLURM / notebook) |
|---|---|---|
| Figure 5 / Table "pruning sparsity configurations" (group-lasso → attribution prune at {0.3, 0.63, 0.8}) | `tuneprune.py` then `prune_and_train.py --sparsity` | `experiments/pruneandtrain00/slurm-run.sh` (consumes `tuneprune15/lambda_{0.0003,0.0005,0.001}/checkpoint-70000`, sparsities match the table grid) |
| Figure 6 (prune + recovery loss curves) | `prune_and_train.py --neuron_sparsity --residual_sparsity` | `experiments/pruneandtrain01/slurm-run.py`; `notebooks/llm-frontiers2.ipynb` reads from `pruneandtrain01/` |
| Figure 11 (attribution score vs. true ablation) | `mask_by_gradient_attribution` from `prune_and_train.py` | `notebooks/attribution-vs-ablation.ipynb` imports `mask_by_gradient_attribution` directly |
| Figure 12 (from-scratch / distill / prune+recover) | `prune_and_train.py` for the prune+recover line | `notebooks/llm-frontiers2.ipynb` (same notebook as Fig. 6) reads from `pruneandtrain01/` |
| Figure 13 (random vs. attribution pruning) | attribution: `prune_and_train.py`; random: `prune_and_train_random.py` | `notebooks/random_vs_attribution.ipynb` reads from both `pruneandtrain01/` and `pruneandtrainrandom00/` |
| **Table 1 (Unlearning — counterfact / AI2-ARC / WMDP-Bio / WMDP-Cyber, the paper text the user quoted)** | **Random** row: `create_random_pruned_models.py`. **Attribution** row: `create_attribution_pruned_models.py` (this directory). **Group Lasso** row: `create_tuneprune_pruned_models.py` — *L2-magnitude* pruning on group-lasso checkpoints (`lambda_0.0003 → 30%`, `lambda_0.0005 → 63%`, `lambda_0.001 → 80%`); **not** attribution-based despite living next to the attribution script. | `experiments/unlearning/slurm/create_all_pruned_models.slurm` calls all three `create_*_pruned_models.py` in sequence at sparsities `{0.3, 0.63, 0.8}`. The Table 1 cell grid is filled by `experiments/unlearning/slurm/eval_{attribution,random,tuneprune_lambda_*}_sparsity_{0.3,0.63,0.8}_{counterfact,ai2_arc,wmdp_bio,wmdp_chem,wmdp_cyber}.slurm` (one file per cell), each invoking `train_eval_lr_sweep.py` (LR sweep 1e-6 → 1e-4 — matches paper text). The LaTeX table is emitted by `generate_results_table.py`, which hardcodes `'Random' / 'Attribution' / 'Group Lasso'` and `'30%' / '63%' / '80%'` exactly as printed. |

Implication: the `T' = abs` deviation in `mask_by_gradient_attribution` is present in *every* reported Llama-3.2-1B figure, not just our local copy.

[^narrow]: Michaud et al., *On the creation of narrow AI: hierarchy and nonlocality of neural network skills*, [arXiv:2505.15811](https://arxiv.org/abs/2505.15811). Equation `ŝ_g = |Σ_{i∈g}(∂L/∂θ_i)(−θ_i)|` in §2.
[^nanda]: Nanda, *Attribution Patching: Activation Patching At Industrial Scale*, 2023. [neelnanda.io](https://www.neelnanda.io/mechanistic-interpretability/attribution-patching) / [LessWrong](https://www.lesswrong.com/posts/gtLLBhzQTG6nKTeCZ/attribution-patching-activation-patching-at-industrial-scale). Formula `(clean_act − corrupted_act) · grad_at_corrupted`.
[^atp]: Kramár et al., *AtP\*: An efficient and scalable method for localizing LLM behaviour to components*, [arXiv:2403.00745](https://arxiv.org/abs/2403.00745). Eq. 4.
[^molchanov]: Molchanov et al., *Importance Estimation for Neural Network Pruning*, CVPR 2019, [arXiv:1906.10771](https://arxiv.org/abs/1906.10771).
[^nvlabs]: NVlabs reference: [github.com/NVlabs/Taylor_pruning](https://github.com/NVlabs/Taylor_pruning), `pruning_engine.py` method 22 (`Taylor_gate`): `(p * p.grad).pow(2).view(nunits, -1).sum(dim=1)`.
