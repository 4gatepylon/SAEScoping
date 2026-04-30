# Naming convention — metrics, W&B keys, and on-disk artifacts

This is the spec for how every metric and artifact in the
`baselines_2026_04_29` sweep is named. It covers three execution phases
that we will eventually run in sequence on each run:

| Phase       | What it measures                                             | X axis (W&B)              | Implemented? |
| ----------- | ------------------------------------------------------------ | ------------------------- | ------------ |
| `sweep`     | One value per sparsity (Wanda prune + per-sparsity eval).    | `nn_linear_sparsity`      | ✅ today      |
| `recovery`  | Per train-step value, per sparsity (PGD-projected SFT).      | `recovery/train_step`     | ⏸ planned    |
| `elicit`    | Per train-step value, per sparsity (SFT on adversarial OOD). | `elicit/train_step`       | ⏸ planned    |

There is also a one-shot **baseline** measurement (pre-pruning loss /
LLM-judge) that sits outside the per-step structure. It is namespaced under
`sweep/baseline/...` for W&B continuity but lives at the run root on disk.

## W&B metric namespace

Three top-level prefixes pin "which phase produced this":

```
sweep/<metric>                                   # X = nn_linear_sparsity
sweep/baseline/<metric>                          # one-shot pre-pruning baseline

recovery/sparsity=<s>/<metric>                   # X = recovery/train_step
elicit/sparsity=<s>/<metric>                     # X = elicit/train_step
```

The metric-name tail (e.g. `loss`, `model_sparsity`,
`llm_judge/biology/in_scope/quality`) is shared across phases — so the
same chart in the W&B UI can show a metric across phases by globbing
the prefix. Concrete examples:

```
sweep/loss
sweep/loss_delta_vs_baseline
sweep/model_sparsity
sweep/llm_judge/biology/in_scope/quality
sweep/llm_judge/biology/in_scope/relevance
sweep/baseline/loss
sweep/baseline/llm_judge/biology/in_scope/quality

recovery/sparsity=0.3/loss
recovery/sparsity=0.3/llm_judge/biology/in_scope/quality
recovery/sparsity=0.5/loss

elicit/sparsity=0.3/loss
elicit/sparsity=0.3/llm_judge/cybersecurity/attack_scope/quality
```

### Notes
- The format inside the sparsity tag is `sparsity=<s>` (e.g.
  `sparsity=0.3`). Self-explanatory, glob-friendly.
- One W&B run per sweep run — recovery and elicit log all sparsities
  into the same run with different prefixes. This makes set-level
  comparisons (and overlays) easy without joining runs externally.
- `define_metric` for `recovery/*` and `elicit/*` step-metric setup is
  deferred (see commit history); for now we only declare
  `nn_linear_sparsity` as the X axis for `sweep/*`.

## On-disk artifact layout (mirrors the W&B namespace)

```
$ARTIFACTS_ROOT/outputs/{run_id}/
├── metadata.json                           # full resolved cfg + git sha + start_time
├── baseline.json                           # one-shot pre-pruning metrics (sweep/baseline/*)
└── step_NNN/                               # one directory per sparsity in cfg.sweep.nn_linear_sparsities
    ├── sweep/
    │   ├── step_metadata.json              # sparsity, loss, model_sparsity, ...
    │   ├── judgements.jsonl                # streamed per LLM-judge call (sweep/llm_judge/*)
    │   ├── inference.jsonl                 # streamed per generation
    │   └── scores.json                     # final aggregated llm_judge/* dict
    ├── recovery/                           # populated only when cfg.pgd.enabled
    │   ├── step_metadata.jsonl             # streamed per PGD train step (training loss, etc.)
    │   ├── judgements.jsonl                # streamed when LLM judge is run mid-recovery
    │   ├── inference.jsonl
    │   └── scores.json
    └── elicit/                             # populated only when elicitation is enabled
        ├── step_metadata.jsonl
        ├── judgements.jsonl
        ├── inference.jsonl
        └── scores.json
```

### Notes
- The `sweep/` sub-directory in each `step_NNN/` is a structural mirror of
  the W&B namespace, even though there is only ever one row per step in the
  sweep phase. It exists to make recovery and elicit (which produce many
  rows per step) live as siblings rather than as off-pattern leaves.
- `step_metadata.jsonl` (plural rows) inside `recovery/` and `elicit/` —
  one row per training step. Contrast with `step_metadata.json` (singular)
  inside `sweep/`, which is one row per sparsity.
- `baseline.json` is at the run root, not under `step_NNN/`, because it is
  measured once before any sweep step runs. The W&B prefix
  `sweep/baseline/...` exists for UI grouping convenience and does not
  imply a per-step location.
- All `*.jsonl` files are streamed via `JsonlSink`
  (`sae_scoping/evaluation/utils.py`) — flushed per row, line-tolerant
  reads, see that file's docstring for crash semantics.

## Current state vs. spec

What is implemented today (`sweep/` only, partial mirror):

- W&B keys: `nn_linear_sparsity`, `model_sparsity`, `loss`,
  `loss_delta_vs_baseline`, `llm_judge/<domain>/<scope>/<group>`. All
  effectively at the `sweep/` level but **without the `sweep/` prefix**.
- On-disk layout: per-sparsity files live directly under `step_NNN/`,
  not under `step_NNN/sweep/`.
- No `baseline.json`; the baseline loss is captured inside the run-level
  `metadata.json` only.

The migrations to the spec above happen as part of the PGD merge
(commit 4) and the elicitation merge (later), where the lack of a phase
prefix would otherwise become ambiguous.
