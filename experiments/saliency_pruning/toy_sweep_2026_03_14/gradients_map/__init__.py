"""
EMA gradient accumulation (or random baseline) for pruning saliency.

CLI usage:
    python -m gradients_map run --mode gradient_ema --output-path biology/ema_grads.safetensors
    python -m gradients_map run --mode gradient_ema --abs-grad
    python -m gradients_map run --mode random
    python -m gradients_map batch --devices 0,1
    python -m gradients_map taylor --input-path biology/ema_grads.safetensors --model-id ...
"""

import click

from .batch import _ALL_VARIANTS, _VARIANT_SPECS, _build_run_cmd, batch
from .grad import GradCollectTrainer, _register_ema_hooks, grad
from .random import make_random_map
from .taylor import (
    _TAYLOR_SOURCE_STEMS,
    make_taylor_map,
    run_taylor,
    taylor_output_path,
    validate_taylor_source_path,
)
from .utils import (
    _DEFAULT_BATCH_SIZE,
    _DEFAULT_BETA,
    _DEFAULT_DATASET,
    _DEFAULT_DATASET_SIZE,
    _DEFAULT_MAX_SEQ,
    _DEFAULT_MODEL_ID,
    _DEFAULT_MODE,
    _DEFAULT_SUBSET,
    _MODE_TO_DEFAULT_OUT_PATH,
    _mode_to_default_output_path,
    _resolve_wandb_config,
    assert_all_params_require_grad,
    save_saliency_map,
)


@click.group()
def main() -> None:
    """Compute pruning saliency maps for weight pruning experiments."""


main.add_command(grad)
main.add_command(batch)
main.add_command(run_taylor)
