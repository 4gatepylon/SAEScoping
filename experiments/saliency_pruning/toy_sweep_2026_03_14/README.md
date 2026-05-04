This module supports a really simple ("toy") flow using saliency-based pruning. It is based on this paper from Neel Nanda which uses gradients to estimate the relevance of a weight to a dataset: https://arxiv.org/abs/2403.00745

Unlike many other modules, it's meant to be ENTIRELY self-contained. It does not depend on any other modules or libraries from `sae_scoping` (though it may use pip-installed modules obviously).

In their context they want to see if removing weights may change a behavior (i.e. make a model say yes vs. no or do math right vs. wrong). They originally would have toggled components on/off (ablations). These components could be weights or activations. They found that they could use gradients based on taylor series estimation to get a good sense of what to prioritize. I'm not sure if they used this mainly for activations, but we intend to use it for weights.

There are a few simple algorithms that build on each other:

1. `gradients_map.py`: only calculate the gradients map
    - `model`
    - `dataset_gradients`: a dataset object on which we calculate the gradients (should be `request`/`response` format)
2. `prune.py`: only prune the model
    - `gradients_map`: a gradients map object or path to a cached gradients map
    - `K`: the number of weights to prune (also called, sparsity when normalized by the number of weights)
3. `prune_and_maybe_recover.py`: calculate gradients map if needed, then prune, then recover if needed (recovery is just SFT on `dataset_recovery`)
    - `model`: a model object
    - `K`: the number of weights to prune (also called, sparsity when normalized by the number of weights)
    - `dataset_pruning`: a dataset on which we prune the model
    - `dataset_recovery`: a dataset on which we try to perform recovery (should be `request`/`response` format)
    - `dataset_evaluation`: a dataset on which we evaluate the performance of the pruned model, at least `request` present
    - `threshold_recovery_good_enough`: a threshold on performance via judge in `grade_chats.py` s.t. if generating on the `dataset_evaluation` yields a score above this threshold, we stop recovering. If negative then no recovery is done.
    - `max_steps_recovery`: a maximum number of steps to recover for
4. `prune_and_maybe_recover_sweep.py`: sweep over a range of `K`s from highest to lowest using binary search. Your model must be passed in from a `model_name_or_path` because the weights will be lost otherwise. Will use binary search to try and prune as much as possible for a fixed budget of recovery.
    - `model`: a model object
    - `K_min, K_max`: optionally, the minimum and maximum number of weights to prune (also called, sparsity when normalized by the number of weights)
    - `dataset_pruning`: a dataset on which we prune the model
    - `dataset_recovery`: a dataset on which we try to perform recovery (should be `request`/`response` format)
    - `dataset_evaluation`: a dataset on which we evaluate the performance of the pruned model, at least `request` present
    - `threshold_recovery_good_enough`: a threshold on performance via judge in `grade_chats.py` s.t. if generating on the `dataset_evaluation` yields a score above this threshold, we stop recovering. If negative then no recovery is done.
    - `max_steps_recovery`: a maximum number of steps to recover for
    - `max_steps_sweep`: a maximum number of steps to sweep for (each sweep step is a call to `prune_and_recover.py`)
    - `thresholds_recovery_give_up`: a list of `{"steps": int, "threshold": float}` tuples s.t. if the recovery score is below the threshold for the given number of steps, we give up for that specific sweep step (in the binary search).
    - `num_cache`: how many checkpoints to cache. By default the top `num_cache` are stored (so everything is until you beat a previous one; you beat a previous one if you have lower K and loss above threshold or equal K and better loss). The specific prioritization scheme is defined in `utils.py`.
    - `metric_type`: either `"loss"` or `"judge"`. Controls which metric the binary search optimizes against. See below.

## Binary search metric abstraction

The sweep in `prune_and_maybe_recover_sweep.py` does binary search over sparsity levels. The search needs a metric to decide whether a given sparsity level is "good enough" (i.e. the model still works after pruning + recovery). This metric is pluggable:

| Metric type | What it measures | Cost | When to use |
|-------------|-----------------|------|-------------|
| `loss` | Validation cross-entropy loss on `dataset_evaluation` | Cheap (no generation, no API calls) | Preliminary runs to quickly find the rough sparsity range |
| `judge` | LLM judge score via `grade_chats.py` on generated responses | Expensive (generation + API calls per level) | Final runs to get accurate quality thresholds |

**Recommended workflow:**
1. Run the sweep with `--metric-type loss` and a broad `K_min`-`K_max` range. This is fast and gives you the approximate sparsity range where loss starts to spike.
2. Narrow the range based on step 1, then re-run with `--metric-type judge` to get precise quality cutoffs.

Both metrics use the same interface: a callable `(model, dataset_evaluation) -> float` where higher is better. For loss, the value is negated internally so that lower loss = higher score. The `threshold_recovery_good_enough` and `thresholds_recovery_give_up` thresholds are always in the metric's native scale (i.e. raw loss for `loss` mode, 0-1 score for `judge` mode).

Like our past experiments, we will use HF model `google/gemma-2-9b-it` for this. We use the dataset from `4gate/StemQAMixture` on subset `biology` (we use splits `train` for training obviously and `validation` for testing; we use the same dataset for `dataset_pruning` and `dataset_recovery` but we randomly select subsets for each of pr-defined sizes in the arguments).
