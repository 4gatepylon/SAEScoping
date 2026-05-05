# Input Specification

## Overview

The plotting library takes two inputs:
1. A **config YAML** describing what to plot and how.
2. A **data CSV** containing all scores.

## Data CSV

Flat CSV. One row per measurement. Columns:

| Column | Type | Required | Description |
|---|---|---|---|
| `model` | str | yes | Model identifier, must match a `models[].id` in config |
| `method` | str | yes | Method identifier, must match a `methods[].id` in config |
| `scope_domain` | str | no | Domain the method targets. Null/empty for methods where `requires_scope_domain=false` (e.g. vanilla). |
| `elicitation_domain` | str | yes | Domain evaluated on |
| `metric` | str | yes | Metric name (e.g. `ground_truth_similarity`, `unit_test_pass_rate`) |
| `score` | float | yes | Raw score value |

### Rules

- Upstream selects checkpoints before producing this CSV — no step/checkpoint column.
- A method with `requires_scope_domain=true` in config MUST have `scope_domain` populated.
- A method with `requires_scope_domain=false` MAY have `scope_domain` empty/null.
- Rows for metrics that are not referenced by any figure (via domain config) are silently ignored.
- Missing rows for metrics that ARE needed cause a hard error.
- Duplicate `(model, method, scope_domain, elicitation_domain, metric)` rows are an error.

### Example

```csv
model,method,scope_domain,elicitation_domain,metric,score
gemma-2,vanilla,,biology,ground_truth_similarity,0.85
gemma-2,vanilla,,chemistry,ground_truth_similarity,0.78
gemma-2,vanilla,,code,unit_test_pass_rate,0.62
gemma-2,scoped,biology,biology,ground_truth_similarity,0.41
gemma-2,scoped_recovered,biology,biology,ground_truth_similarity,0.82
gemma-2,scoped_recovered,biology,chemistry,ground_truth_similarity,0.15
gemma-2,pgd,biology,biology,ground_truth_similarity,0.70
gemma-2,pgd,biology,chemistry,ground_truth_similarity,0.30
gemma-2,sft,biology,biology,ground_truth_similarity,0.83
gemma-2,sft,biology,chemistry,ground_truth_similarity,0.45
```

## Config YAML

See `config_schemas.py` for the pydantic schema. Example:

```yaml
models:
  - id: gemma-2
    display_name: "Gemma 2"
  - id: gemma-3
    display_name: "Gemma 3"

domains:
  default_metric: ground_truth_similarity
  entries:
    - id: biology
      display_name: "Biology"
    - id: chemistry
      display_name: "Chemistry"
    - id: physics
      display_name: "Physics"
    - id: code
      display_name: "Code"
      metric: unit_test_pass_rate

methods:
  - id: vanilla
    display_name: "Vanilla"
    color: "#888888"
    requires_scope_domain: false
  - id: scoped
    display_name: "Scoped"
    color: "#ff7f0e"
  - id: scoped_recovered
    display_name: "Scoped + Recovered"
    color: "#2ca02c"
  - id: sft
    display_name: "SFT"
    color: "#1f77b4"
  - id: pgd
    display_name: "PGD"
    color: "#d62728"
    # requires_scope_domain defaults true; set false if PGD has no scope dimension

figures:
  in_domain_bar:
    methods: [scoped, scoped_recovered, sft]
    baseline_method: vanilla
  ood_table:
    our_method: scoped_recovered
    comparison_methods: [pgd]

output:
  dpi: 300
  latex: true
```

## How figures use the data

### Figure 1 (in-domain bar plot)

For each model, for each domain:
1. Look up the metric for that domain (`domains.entries[].metric` or `domains.default_metric`).
2. Get vanilla score: row where `method=baseline_method`, `elicitation_domain=domain`, that metric.
3. For each method in `in_domain_bar.methods`: get the row where `scope_domain=domain`, `elicitation_domain=domain`, that metric. Compute `score / vanilla_score`.
4. Plot bars per domain, dashed line at 1.0.

### Figure 2 (OOD table)

For each model:
1. Build a grid of `scope_domain` (columns) x `elicitation_domain` (rows).
2. For each cell, look up the metric for the elicitation_domain.
3. Get our method's score and the best score among comparison_methods.
4. Compute relative scores (divided by vanilla for that elicitation_domain).
5. Render as text table: both numbers in cell, bold the lower, color green/yellow.
6. Diagonal (in-domain) cells can be styled differently or excluded — TBD.
