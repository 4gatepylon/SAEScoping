# SAE Scoping Pipeline — Biology

This experiment tests whether **SAE-based scoping** can constrain a language model to a specific domain (biology) while suppressing capabilities in out-of-domain (OOD) areas (cybersecurity, math, chemistry).

The core idea: identify which SAE neurons fire on in-domain content, zero out the rest, then fine-tune the model through the pruned SAE so it learns to operate within those features only.

---

## Setup

```bash
export OPENAI_API_KEY=sk-...   # required for LLM judge evaluation
export WANDB_API_KEY=...       # required for logging
```

Dependencies are managed via the conda environment at the repo root.

---

## Running the pipeline

All commands are run from the `experiments/` directory.

### Full pipeline from scratch
```bash
python script_scoping_pipeline_stemqa_biology.py --stage all
```

### Individual stages
```bash
# Stage 1 only: compute and cache neuron firing rates
python script_scoping_pipeline_stemqa_biology.py --stage rank

# Stage 3 only: recovery training (uses cached firing rates)
python script_scoping_pipeline_stemqa_biology.py --stage recover
```

### Key options
| Flag | Default | Description |
|---|---|---|
| `--stage` | `all` | `all`, `rank`, or `recover` |
| `--n-rank-samples` | 10,000 | Biology samples used to compute firing rates |
| `--batch-size` / `-b` | 4 | Per-device train batch size |
| `--accum` / `-a` | 16 | Gradient accumulation steps (effective batch = b × a) |
| `--max-steps-recover` | 3,000 | Training steps for recovery stage |
| `--save-every` | 1,000 | Checkpoint save interval |
| `--firing-rate-threshold` | 1e-4 | Neurons below this rate are pruned |
| `--device` | `cuda:0` | Device string |

---

## Pipeline stages

### Stage 1 — RANK
Runs Gemma-2-9b-it on StemQAMixture biology examples with a GemmaScope SAE hooked into layer 31. Counts how often each of the 16k SAE neurons fires across the corpus. Result is cached to:
```
.cache/stemqa_biology/ignore_padding_True/layer_31--width_16k--canonical/firing_rates.safetensors
```
Subsequent runs load from cache automatically.

### Stage 2 — PRUNE
Zeros out all SAE neurons with firing rate below `--firing-rate-threshold` (default 0.01%). The resulting `pruned_sae` only reconstructs activations through biology-relevant features.

### Stage 3 — RECOVER
Fine-tunes the model on biology QA with the pruned SAE active as a forward-pass hook on layer 31. The SAE filters activations at layer 31 before they flow into layer 32.

**Trainable parameters:** layer 32 and the final transformer layer (41) only, plus the final layernorm and lm_head. All other layers are frozen.

**Training data:** StemQAMixture biology (same split used for ranking, no overlap with eval).

**Eval datasets** (used for loss tracking and LLM judge, not for training):
- Biology — StemQAMixture biology held-out split
- Cybersecurity — WMDP-cyber
- Math — StemQAMixture math
- Chemistry — StemQAMixture chemistry

Checkpoints are saved to `outputs_scoping/recover/`.

---

## LLM judge evaluation

Every 500 training steps, an LLM judge evaluation runs automatically:

1. The model generates free-text responses to raw questions from each eval domain (user-turn only, no answer provided).
2. GPT-4.1-nano scores each response on three axes: `answering`, `factual_helpful`, `precise`.
3. Scores are aggregated into a `utility` metric (mean of the three) and also logged individually.

### W&B metrics
```
llm_judge/biology/in_scope/utility
llm_judge/biology/in_scope/answering
llm_judge/biology/in_scope/factual_helpful
llm_judge/biology/in_scope/precise

llm_judge/cybersecurity/out_of_scope/utility
llm_judge/cybersecurity/out_of_scope/answering
...  (same for math, chemistry)
```

A successful scoping run should show biology utility staying high while OOD utility drops.

### Saved outputs
- **CSV:** `outputs_scoping/llm_judge_csvs/llm_judge_step_{N}.csv` — full judgements table (prompt, response, scores, explanations) per eval step.
- **W&B Table:** uploaded as `llm_judge/judgements` for interactive browsing.

---

## Output structure

```
experiments/
  .cache/                          # cached firing rates
  outputs_scoping/
    recover/                       # training checkpoints + final model
      checkpoint-1000/
      checkpoint-2000/
      final/
    llm_judge_csvs/                # per-step judgement CSVs
      llm_judge_step_500.csv
      llm_judge_step_1000.csv
      ...
```

---

## Model & SAE

| | |
|---|---|
| Base model | `google/gemma-2-9b-it` |
| SAE release | `gemma-scope-9b-pt-res-canonical` |
| SAE layer | 31 (of 42) |
| SAE width | 16k features |
