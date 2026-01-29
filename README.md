# SAE Scoping
Minimal viable version of 2024/2025 SAE Scoping work. The desired use-case is that you do:
1. Setup your environment with `conda create -n saescoping python=3.12 -y`
2. Install this and its depndencies with `pip install -e .`.

The folder-structure (etc...) should be self-explanatory. The pip-installable is inside `sae_scoping/` whereas `experiments` and other such folders are meant for one-off scripts (etc...) that are NOT libraries but may be important for results from the paper.

_WARNING_: this branch does NOT support training, etc... It only showcases the evaluation using GEPA + dspy
- Evaluation uses the `dspy` module’s `Evaluator` to evaluate models on datasets, with a custom `Adapter` for extracting verifiable responses (to check correctness). Batched evaluation for verifiable datasets uses `dspy`’s `Evaluate`. Requires my fork: https://github.com/4gatepylon/dspy
- Adversarial optimization with GEPA is supported; use my branch: https://github.com/4gatepylon/gepa. In-domain and out-of-domain dataset combinations are defined, enabling OOD optimization.

