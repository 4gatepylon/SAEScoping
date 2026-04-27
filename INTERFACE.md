# Interface Design

This document describes the interface we want for the scoping pipeline. Scoping is the process of making a model that only does one thing in five steps:

1. **Calibrate** — Collect relevance metadata from calibration data. This usually dumps safetensors files of some form to disk.
2. **Prune** — Modify the model or an adaptor to restrict it to the target domain. Usually this is a form of saliency pruning for model weights OR it is a form of activation pruning for SAEs (in the latter case the SAEs are applied to the model).
3. **Recover** — SFT on in-domain data to restore in-domain performance. For mask-based pruning methods (on the weights) this uses projected gradient descent to maintain sparsity. For SAE-based methods it usually only trains the layer(s) AFTER the SAE (to maintain the incoming representations invariant).
4. **Iterate** (optional) — Repeat steps 1-3 for progressively more aggressive pruning
5. **Elicitation** — Adversarial ("attack") fine-tuning on OOD data to test robustness. Usually elicitation is done via SFT as an upper bound on the potential uplift from black-box methods such as jailbreaking and prompt optimization. To be fair to the methods, usually the SFT only updates "applicable" weights. Applicable weights for an SAE-enhanced model are the ones AFTER the SAE. Applicable weights for a pruned model are only the non-zeroed ones (i.e. training uses PGD).

We compare pruning methods (such as Wanda, Taylor-series saliency mapping, SparseLLM, and random pruning) with SAE scoping (our own method) and unlearning methods.

Normally we scope onto some in-domain domains `X1, X2, ...` and then evaluate against various OOD domains `Y1, Y2, ...` for each of the in-domain domains. This yields a grid of experiments `(X1, Y1), (X1, Y2), ..., (Xn, Ym)`. For each grid point we evaluate performance on `Xi, Yj` after step (2), after step (3) and after step (5). Elicitation happens on `Yj`. Sidenote: `(Xi, Yi)` is usually ignored for obvious reasons. For real scoping methods this means we prune one model per `Xi`, then we recover it (once per `Xi`), then for each of those, we train one elicited model per `Yj` on top of those weights. However, for unlearning we usually train one model per grid-point (or possibly groups of grid-points), since unlearning requires BOTH a retain (`Xi`) and a forget (`Yj`) dataset/domain. You can think of each of the steps (1-5, above) as being implicitely depending on some retain and forget sets. Normally, the forget set's _domain_ is only used for evaluation, but in unlearning it is also used for training (although in all cases, no matter what, a test set of _data points_ is held out for evaluation).

## Design Principles

- **Experiments should be low-code shims.** An experiment script defines which method to use and provides a (possibly nested/hierarchical) config. The library does the rest.
- **Maximize shared infrastructure.** Many methods share the same steps — e.g. all weight-saliency methods differ only in *how* they compute scores, but the pruning (thresholding a saliency map into a mask and zeroing weights) is identical. Factor out the shared parts so new methods only implement what's unique to them. This is only an example---other situations may also involve shared functionality.
- **Config-driven execution.** Running an experiment means: pick a method, fill in its config, call `run`. No boilerplate orchestration in experiment code. Hyperparameter sweeps might be implemented as wrappers, but if they ever become very common, then they can be baked into the library. Reusable components should always be tested and baked into the library, keeping experiments declarative and lean.

## Configuration

| Component | Shared | Method-Specific |
|---|---|---|
| Dataset loading & formatting | Yes | — |
| Calibration (computing relevance scores) | — | Yes (each method implements its own) |
| Saliency → mask thresholding | Yes (for all weight-saliency methods) | — |
| Mask application (zeroing weights) | Yes | — |
| Recovery training (PGD SFT) | Yes | Unlearning methods override this |
| Attack training | Yes | — |
| LLM judge evaluation | Yes | — |
| Caching / checkpointing | Yes | — |
| WandB logging | Yes | — |

## Pydantic Config Definitions (Draft)

There are two levels of config: **per-step configs** (calibrate, prune, recover, elicit) that are composable building blocks, and **method configs** that bundle the right steps together with sensible defaults. Experiments use method configs; per-step configs exist for when you need fine-grained control or are composing something novel.

Caching is handled at the calibration level. Calibration is expensive and reusable across sparsity levels, iteration rounds, and elicitation targets. The cache key is derived deterministically from the calibration config + model + domain + dataset, so "do wanda at sparsity=0.3" and "do wanda at sparsity=0.7" share the same cached saliency map.

```python
# NOTE: All pydantic definitions below are pseudocode illustrating the
# intended config hierarchy. Class names, field names, and exact types
# will likely change during implementation. The goal is to communicate
# the shape of the config, not to be copy-pasteable.

from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Annotated, Literal
from enum import Enum


# ===========================================================================
# Per-step configs (building blocks)
# ===========================================================================

# ---------------------------------------------------------------------------
# Calibration (step 1) — one variant per method, discriminated on `method`
# ---------------------------------------------------------------------------

class WandaCalibrateConfig(BaseModel):
    method: Literal["wanda"] = "wanda"
    n_samples: int = 128
    max_seq_len: int = 2048

class GradientCalibrateConfig(BaseModel):
    method: Literal["gradient"] = "gradient"
    n_samples: int = 128
    max_seq_len: int = 2048
    ema_beta: float = 0.95
    absolute_value: bool = True

class TaylorCalibrateConfig(BaseModel):
    method: Literal["taylor"] = "taylor"
    gradient: GradientCalibrateConfig = GradientCalibrateConfig()

class RandomCalibrateConfig(BaseModel):
    method: Literal["random"] = "random"
    seed: int = 42

class SAECalibrateConfig(BaseModel):
    method: Literal["sae"] = "sae"
    sae_id: str
    n_samples: int = 128
    max_seq_len: int = 2048

CalibrateConfig = Annotated[
    WandaCalibrateConfig
    | GradientCalibrateConfig
    | TaylorCalibrateConfig
    | RandomCalibrateConfig
    | SAECalibrateConfig,
    Field(discriminator="method"),
]


# ---------------------------------------------------------------------------
# Pruning (step 2)
# ---------------------------------------------------------------------------

class ThresholdStrategy(str, Enum):
    GLOBAL = "global"
    PER_ROW = "per_row"

class WeightPruneConfig(BaseModel):
    """Prune weights by thresholding a saliency map."""
    type: Literal["weight"] = "weight"
    sparsity: float = 0.5
    strategy: ThresholdStrategy = ThresholdStrategy.GLOBAL

class SAEPruneConfig(BaseModel):
    """Prune SAE features by masking activations."""
    type: Literal["sae"] = "sae"
    sparsity: float = 0.5

PruneConfig = Annotated[
    WeightPruneConfig | SAEPruneConfig,
    Field(discriminator="type"),
]


# ---------------------------------------------------------------------------
# Training config (shared by recovery, elicitation, and unlearning)
# ---------------------------------------------------------------------------

class TrainingConfig(BaseModel):
    """Base training hyperparams. Used by both recovery and elicitation."""
    n_epochs: int = 3
    learning_rate: float = 1e-5
    batch_size: int = 2


# ---------------------------------------------------------------------------
# Recovery (step 3) — what gets trained depends on method family
# ---------------------------------------------------------------------------

class PGDRecoveryConfig(BaseModel):
    """Recovery for weight-pruned models. PGD re-zeros pruned weights each step."""
    type: Literal["pgd"] = "pgd"
    training: TrainingConfig = TrainingConfig()

class SAERecoveryConfig(BaseModel):
    """Recovery for SAE-scoped models. Only trains layers AFTER the SAE."""
    type: Literal["sae_downstream"] = "sae_downstream"
    training: TrainingConfig = TrainingConfig()

class UnlearningRecoveryConfig(BaseModel):
    """Recovery for unlearning methods. Uses both retain and forget data."""
    type: Literal["unlearning"] = "unlearning"
    training: TrainingConfig = TrainingConfig()
    retain_weight: float = 1.0
    forget_weight: float = 1.0

RecoveryConfig = Annotated[
    PGDRecoveryConfig | SAERecoveryConfig | UnlearningRecoveryConfig,
    Field(discriminator="type"),
]


# ---------------------------------------------------------------------------
# Elicitation (step 5) — trains on OOD data, respecting applicable weights
# ---------------------------------------------------------------------------

class PGDElicitConfig(BaseModel):
    """Elicitation for weight-pruned models. PGD on non-zeroed weights only."""
    type: Literal["pgd"] = "pgd"
    training: TrainingConfig = TrainingConfig()

class SAEElicitConfig(BaseModel):
    """Elicitation for SAE-scoped models. Only trains layers AFTER the SAE."""
    type: Literal["sae_downstream"] = "sae_downstream"
    training: TrainingConfig = TrainingConfig()

class FullElicitConfig(BaseModel):
    """Elicitation with no weight restrictions (e.g. for unlearning baselines)."""
    type: Literal["full"] = "full"
    training: TrainingConfig = TrainingConfig()

ElicitConfig = Annotated[
    PGDElicitConfig | SAEElicitConfig | FullElicitConfig,
    Field(discriminator="type"),
]


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

class EvalConfig(BaseModel):
    judge_model: str = "gpt-4.1-nano"
    n_samples: int = 100


# ===========================================================================
# Method configs (bundles of steps with correct defaults)
# ===========================================================================
# These are the configs you actually use in experiments. Each one wires
# together the right calibrate/prune/recover/elicit combination so you
# don't have to think about which step types are compatible.

class SaliencyMethodConfig(BaseModel):
    """Any weight-saliency pruning method (Wanda, gradient, Taylor, random, SparseLLM).
    
    Calibrate produces a saliency map, prune thresholds it into a mask,
    recovery and elicitation both use PGD on non-zeroed weights.
    """
    family: Literal["saliency"] = "saliency"
    calibrate: CalibrateConfig
    prune: WeightPruneConfig = WeightPruneConfig()
    recover: PGDRecoveryConfig = PGDRecoveryConfig()
    elicit: PGDElicitConfig = PGDElicitConfig()
    iterate_rounds: int = 1

class SAEScopingMethodConfig(BaseModel):
    """SAE-based scoping (our method).
    
    Calibrate computes feature importance, prune masks SAE activations,
    recovery and elicitation only train layers downstream of the SAE.
    """
    family: Literal["sae"] = "sae"
    calibrate: SAECalibrateConfig
    prune: SAEPruneConfig = SAEPruneConfig()
    recover: SAERecoveryConfig = SAERecoveryConfig()
    elicit: SAEElicitConfig = SAEElicitConfig()
    iterate_rounds: int = 1

class UnlearningMethodConfig(BaseModel):
    """Unlearning baselines. No separate calibrate/prune — the training
    itself handles both retain and forget. Elicitation is unrestricted SFT.
    
    Note: unlearning trains one model per (Xi, Yj) grid-point, unlike
    scoping methods which share a single pruned/recovered model per Xi.
    """
    family: Literal["unlearning"] = "unlearning"
    recover: UnlearningRecoveryConfig = UnlearningRecoveryConfig()
    elicit: FullElicitConfig = FullElicitConfig()

MethodConfig = Annotated[
    SaliencyMethodConfig | SAEScopingMethodConfig | UnlearningMethodConfig,
    Field(discriminator="family"),
]


# ===========================================================================
# Experiment grid config (top level)
# ===========================================================================

class DatasetConfig(BaseModel):
    name: str = "AdrianM0/StemQAMixture"
    test_fraction: float = 0.2

class ExperimentConfig(BaseModel):
    """Top-level config for a scoping experiment.
    
    Defines a grid of (in_domains × ood_domains). The library runs the
    full pipeline for each grid point, reusing cached artifacts where
    possible (e.g. calibration is shared across sparsity levels and
    OOD targets for the same in-domain).
    """
    model_id: str = "google/gemma-2-9b-it"
    dataset: DatasetConfig = DatasetConfig()
    in_domains: list[str]
    ood_domains: list[str]
    method: MethodConfig
    eval: EvalConfig = EvalConfig()
```

### Caching

Calibration artifacts are keyed by `(method, model_id, domain, dataset, calibrate_config_hash)` and saved as safetensors to a cache directory. When the library encounters a cache hit, it skips calibration and loads from disk. This means:

- Sweeping sparsity levels: calibrate once, prune N times from the same saliency map
- Iterating rounds: each round re-calibrates (on the recovered model), producing a new cache entry
- Different OOD targets: calibration only depends on in-domain, so all `Yj` for a given `Xi` share calibration

### Usage

```python
# NOTE: All code in this section is pseudocode illustrating the intended
# user experience. Function names, imports, and exact APIs will likely
# differ in the real implementation.

from sae_scoping import run_experiment

# --- Wanda at 50% sparsity (one line of method-specific config) ---
run_experiment(ExperimentConfig(
    in_domains=["biology"],
    ood_domains=["math", "chemistry", "physics"],
    method=SaliencyMethodConfig(
        calibrate=WandaCalibrateConfig(n_samples=256),
        prune=WeightPruneConfig(sparsity=0.5),
    ),
))

# --- SAE scoping ---
run_experiment(ExperimentConfig(
    in_domains=["biology"],
    ood_domains=["math", "chemistry"],
    method=SAEScopingMethodConfig(
        calibrate=SAECalibrateConfig(sae_id="gemma-2-9b-res-65k"),
        prune=SAEPruneConfig(sparsity=0.3),
    ),
))

# --- Unlearning baseline ---
run_experiment(ExperimentConfig(
    in_domains=["biology"],
    ood_domains=["math", "chemistry"],
    method=UnlearningMethodConfig(
        recover=UnlearningRecoveryConfig(retain_weight=1.0, forget_weight=0.5),
    ),
))

# --- Sparsity sweep (calibration cached, prune varies) ---
for s in [0.1, 0.3, 0.5, 0.7, 0.9]:
    run_experiment(ExperimentConfig(
        in_domains=["biology"],
        ood_domains=["math"],
        method=SaliencyMethodConfig(
            calibrate=WandaCalibrateConfig(),
            prune=WeightPruneConfig(sparsity=s),
        ),
    ))
# ^ calibration runs once, the other 4 runs load from cache
```

### Execution Model

For **scoping methods** (saliency, SAE), the library runs:
```
# Pseudocode — illustrates the loop structure, not actual function signatures.
for Xi in in_domains:
    calibration = calibrate(model, Xi)        # cached
    pruned      = prune(model, calibration)
    evaluate(pruned, Xi, ood_domains)          # post-prune eval
    recovered   = recover(pruned, Xi)
    evaluate(recovered, Xi, ood_domains)       # post-recovery eval
    for Yj in ood_domains:
        if Xi == Yj: continue
        elicited = elicit(recovered, Yj)
        evaluate(elicited, Xi, [Yj])           # post-elicitation eval
```

For **unlearning methods**, the library runs:
```
# Pseudocode — illustrates the loop structure, not actual function signatures.
for Xi in in_domains:
    for Yj in ood_domains:
        if Xi == Yj: continue
        unlearned = recover(model, retain=Xi, forget=Yj)
        evaluate(unlearned, Xi, [Yj])          # post-unlearning eval
        elicited  = elicit(unlearned, Yj)
        evaluate(elicited, Xi, [Yj])           # post-elicitation eval
```

# Things TODO
Reminders for adriano to look into.

1. Mathematically document what each method does and link to the arxiv and relevant github, etc... Then, add tests to verify that this behaves as intended. Include integration tests to reproduce the past work.
2. Actually finalize and enact this inteface and make experiments a breeze.
3. Add beartype everywhere and have claude code make the types for beartype correct. This should reduce the likelihood of error.
4. **Prunable parameter scope is hardcoded and inconsistent across methods.** Currently:
   - Wanda: only `nn.Linear` weights (via `_find_linear_layers`), skips `{"lm_head", "embed_tokens", "embed_out"}` by child name.
   - Random: all `requires_grad` params, skips names containing `_SKIP_LAYER_NAMES` parts. Includes biases, LayerNorm, etc.
   - Taylor/Gradient: all params that received EMA gradients, post-filtered by `_SKIP_LAYER_NAMES` in `dispatch.py`. Includes biases, LayerNorm, etc.

   This means cross-method comparisons at "the same sparsity" are apples-to-oranges. Should be replaced with a user-configurable `prunable_parameters(model, config) -> set[str]` that all methods use, driven by CLI/YAML (e.g. `include_biases`, `include_layernorm`, `skip_layers`, regex filter). See `WeightPruneConfig` above — this is where it belongs.
5. **`_SKIP_LAYER_NAMES` is hardcoded in `wanda.py` and imported everywhere.** The set `{"lm_head", "embed_tokens", "embed_out"}` assumes HuggingFace naming conventions for a narrow set of architectures. Should be derived from the model config or specified by the user, not a module-level constant.
6. **`_compute_ema_grads` in `dispatch.py` hardcodes training hyperparameters.** `n=4096`, `seed=42`, `beta=0.95`, `batch_size=2`, `max_length=1024`, `learning_rate=1e-4` are all baked in with no way to override from CLI or config. These should come from the calibration config (see `GradientCalibrateConfig` above).
7. **`assert_no_embedding_or_head_in_masks` hardcodes the three names to check.** Should use the same configurable skip-layer set as everything else.
8. **`_find_linear_layers` hardcodes `isinstance(child, nn.Linear)`.** Whether to restrict pruning to Linear layers only (vs also allowing Conv1d, etc.) should be a config decision, not baked into the traversal function.
9. **Taylor/Gradient CLI paths (`taylor.py:run_taylor`, `grad.py:grad`) save unfiltered maps** that include embedding/lm_head params. The dispatch.py path filters, but direct CLI usage doesn't. These should apply the same configurable filter.