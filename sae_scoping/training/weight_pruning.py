"""Global quantile sample budget for weight pruning."""

# Number of elements sampled from all scores to estimate the global quantile.
# 10 M samples over 9 B elements gives < 0.01 % quantile error — more than
# sufficient for weight pruning. Consumed by
# ``sae_scoping.training.saliency.dispatch.masks_for_sparsity``.
_THRESHOLD_SAMPLE_BUDGET = 10_000_000
