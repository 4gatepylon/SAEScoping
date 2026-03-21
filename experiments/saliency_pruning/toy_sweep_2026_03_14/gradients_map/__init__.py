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

from .batch import batch
from .grad import GradCollectTrainer, grad
from .random import make_random_map
from .taylor import make_taylor_map, run_taylor, taylor_output_path, validate_taylor_source_path
from .utils import assert_all_params_require_grad, save_saliency_map


@click.group()
def main() -> None:
    """Compute pruning saliency maps for weight pruning experiments."""


main.add_command(grad)
main.add_command(batch)
main.add_command(run_taylor)
