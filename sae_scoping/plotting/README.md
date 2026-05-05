# sae_scoping.plotting

Paper figure generation for SAE Scoping results.

## Figures

- **Figure 1 (in-domain bar plot)**: Per-model grouped bar chart showing relative in-domain performance (scope_domain == elicitation_domain) across domains, with vanilla as a dashed horizontal line at 1.0. One figure per model.
- **Figure 2 (OOD table)**: Per-model text table (rendered as LaTeX + PNG) with scope_domain on columns, elicitation_domain on rows. Each cell shows our method's OOD score vs best pruning baseline, color-coded green (we win) or yellow (we lose). One table per model.

## Usage

```
python -m sae_scoping.plotting --config config.yaml --data results.csv --output-dir plots/
```

## TODO(adriano)

- Review all schemas, figures, and spec for correctness — this was implemented by Claude and has not been human-reviewed yet.
- TODO(adriano): Add a tool to compile output artifacts plus human-written prose into a paper draft (LaTeX) in an automated fashion.
- TODO(adriano): Add a tool to convert wandb outputs, training artifacts, and logs into the CSV/YAML inputs consumed by this plotting CLI.
- The long-term goal is a modular pipeline: (training and log creation) -> (canonicalization by human or machine to select numbers) -> (creation of visuals) -> (assembly into draft LaTeX document) — each stage has its own interface so any module can change independently.
