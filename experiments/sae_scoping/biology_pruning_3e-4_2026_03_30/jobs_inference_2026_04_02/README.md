# Post-hoc inference & grading (2026-04-02)

## Why this exists

The training jobs in `jobs_2026_03_30/` did **not** pass `--utility-eval-every`
to `script_train_gemma9b_sae.py`.  The default value is `0` (disabled), so the
`UtilityEvalCallback` was never instantiated and no LLM-judge utility scores
were logged to wandb during training.

This directory runs **post-hoc generation + grading** on all saved checkpoints
to recover the utility data that should have been plotted during training.

## What `generate.sh` does

1. Builds a JSON eval config listing every checkpoint (vanilla, SAE, and base).
2. Calls `generate_and_grade.py` which, for each checkpoint × eval subset:
   - Loads the model (and optionally hooks a pruned SAE).
   - Generates responses on held-out OOD questions.
   - Grades the responses with LLM judges (answering, factual_helpful, precise).
   - Saves conversations + grades to `eval_results/<tag>/<subset>.json`.

## Prerequisites

- `OPENAI_API_KEY` must be set (grading uses `gpt-4.1-nano` via litellm).
- A GPU with enough VRAM for Gemma-2 9B in bfloat16 (~20 GB).
