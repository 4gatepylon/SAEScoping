# Next Steps (2026-04-03)

All commands are run from the `biology_pruning_3e-4_2026_03_30/` directory.

---

## Phase 1 — Compute distributions (run immediately)

Computes chemistry and physics neuron firing-rate distributions on the
biology-trained model, so we can build domain-matched SAEs for recovery training.

```bash
bash jobs_2026_04_03/run_distributions.sh 0 1   # chemistry on GPU 0, physics on GPU 1
# or, single GPU:
bash jobs_2026_04_03/run_distributions.sh 0
```

---

## Phase 2 — Find thresholds

After distributions finish, find the threshold that keeps ~2000 neurons for each domain:

```bash
python jobs_2026_04_03/find_threshold.py \
    distributions_cache/ignore_padding_True/chemistry/layer_31--width_16k--canonical/distribution.safetensors \
    --n-neurons 2000

python jobs_2026_04_03/find_threshold.py \
    distributions_cache/ignore_padding_True/physics/layer_31--width_16k--canonical/distribution.safetensors \
    --n-neurons 2000
```

The script prints an exact binary-search threshold and a table of round values.
Choose 2-3 round values near 2000 neurons for recovery training (e.g. `3e-4,4e-4`).

---

## Phase 3 — Training (run in parallel on available GPUs)

### 3a — Elicitation (can start now, no threshold needed)

Tests whether the biology-locked SAE can be overcome with 4× more training steps.
Logs OOD utility (`utility_eval/ood/judge`) AND biology utility
(`utility_eval/biology/judge`) as separate W&B series.

```bash
bash jobs_2026_04_03/run_elicitation.sh 0 1   # chemistry on GPU 0, physics on GPU 1
```

### 3b — Recovery (run after finding thresholds)

Trains with a domain-matched SAE (chemistry or physics neurons) instead of the
biology-locked one. Tests whether domain-aligned neurons make recovery easier.

```bash
# Replace 3e-4,4e-4 with your chosen thresholds from Phase 2
bash jobs_2026_04_03/run_recovery.sh 0,1,2,3 3e-4,4e-4 3e-4,4e-4
# Format: run_recovery.sh GPU_LIST CHEM_THRESHOLDS PHYS_THRESHOLDS
```

Elicitation and recovery can share GPUs if needed — they write to separate output dirs.

---

## Separately — Re-run old evals with biology

Adds biology capability scores to the existing `eval_results/` without re-running
any already-complete OOD subset (the caching in `generate_and_grade.py` ensures this).

```bash
bash jobs_2026_04_03/rerun_eval_with_biology.sh 0   # any single GPU
```

---

## W&B structure for Phase 3 runs

| Series key                     | Meaning                                      |
|--------------------------------|----------------------------------------------|
| `utility_eval/ood/judge`       | LLM judge on the trained OOD subject         |
| `utility_eval/biology/judge`   | LLM judge on biology questions (side-effect) |
| `train/loss`                   | Training loss (logged by SFTTrainer)         |
| `eval/loss`                    | Eval loss (logged by SFTTrainer)             |
