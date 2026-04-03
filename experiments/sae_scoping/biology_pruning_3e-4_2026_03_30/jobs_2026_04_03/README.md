# Next Steps (2026-04-03)

All commands are run from the `biology_pruning_3e-4_2026_03_30/` directory.

---

## Phase 1 — Compute distributions (run immediately)

Computes chemistry and physics neuron firing-rate distributions on the
biology-trained model, so we can build domain-matched SAEs for recovery training.

```bash
bash jobs_2026_04_03/run_distributions.sh 1 2   # chemistry on GPU 1, physics on GPU 2
# or, single GPU:
bash jobs_2026_04_03/run_distributions.sh 1
```

---

## Phase 2 — Find thresholds

After distributions finish, run the visualisation script to see threshold curves
for all distributions (including the existing downloaded biology one) side by side:

```bash
python jobs_2026_04_03/find_threshold_2026_04_03.py
```

This writes `threshold_info_2026_04_03/` containing:
- `<domain>.png` — linear + log threshold-vs-fraction plot with annotated
  intersection dots at n={2000,4000,8000} and h={1e-4,2e-4,3e-4,4e-4}
- `<domain>.json` — operating points at those intersections
- `comparison.png` — all curves overlaid for cross-topic comparison

The threshold at n neurons is `sorted_distribution[n-1]` (descending order).
Pick 2-3 round h values near the 2000-neuron mark from the plots/JSONs
(e.g. `3e-4,4e-4`) and pass them to `run_recovery.sh`.

---

## Phase 3 — Training (run in parallel on available GPUs)

### 3a — Elicitation (can start now, no threshold needed)

Tests whether the biology-locked SAE can be overcome with 4× more training steps.
Logs OOD utility (`utility_eval/ood/judge`) AND biology utility
(`utility_eval/biology/judge`) as separate W&B series.

```bash
bash jobs_2026_04_03/run_elicitation.sh 1 2   # chemistry on GPU 1, physics on GPU 2
```

### 3b — Recovery (run after finding thresholds)

Trains with a domain-matched SAE (chemistry or physics neurons) instead of the
biology-locked one. Tests whether domain-aligned neurons make recovery easier.

```bash
# Replace 3e-4,4e-4 with your chosen thresholds from Phase 2
bash jobs_2026_04_03/run_recovery.sh 1,2 3e-4,4e-4 3e-4,4e-4
# Format: run_recovery.sh GPU_LIST CHEM_THRESHOLDS PHYS_THRESHOLDS
```

Elicitation and recovery can share GPUs if needed — they write to separate output dirs.

---

## Separately — Re-run old evals with biology

Adds biology capability scores to the existing `eval_results/` without re-running
any already-complete OOD subset (the caching in `generate_and_grade.py` ensures this).

```bash
bash jobs_2026_04_03/rerun_eval_with_biology.sh 1   # any single GPU
```

---

## W&B structure for Phase 3 runs

| Series key                     | Meaning                                      |
|--------------------------------|----------------------------------------------|
| `utility_eval/ood/judge`       | LLM judge on the trained OOD subject         |
| `utility_eval/biology/judge`   | LLM judge on biology questions (side-effect) |
| `train/loss`                   | Training loss (logged by SFTTrainer)         |
| `eval/loss`                    | Eval loss (logged by SFTTrainer)             |
