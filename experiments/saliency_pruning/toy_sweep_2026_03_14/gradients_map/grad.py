"""EMA gradient accumulation trainer and 'run' CLI command.

See gradients_map_old.py module docstring for a detailed description of the
EMA hook design, invariants, and known failure modes.
"""

import traceback
import types
from pathlib import Path

import click
import torch
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

from .random import make_random_map
from .utils import (
    _CHAT_TEMPLATE_PATH,
    _DEFAULT_BATCH_SIZE,
    _DEFAULT_BETA,
    _DEFAULT_DATASET,
    _DEFAULT_DATASET_SIZE,
    _DEFAULT_MAX_SEQ,
    _DEFAULT_MODEL_ID,
    _DEFAULT_MODE,
    _DEFAULT_SUBSET,
    _MODE_TO_DEFAULT_OUT_PATH,
    _assert_ema_grads_populated,
    _assert_weights_unchanged,
    _format_qa_as_sft_text,
    _load_qa_dataset,
    _mode_to_default_output_path,
    _report_hook_diagnostics,
    _resolve_wandb_config,
    _sample_param_indices,
    assert_all_params_require_grad,
    save_saliency_map,
)


def _register_ema_hooks(
    model: AutoModelForCausalLM,
    beta: float,
    abs_grad: bool = False,
) -> None:
    """Register gradient hooks that accumulate EMA(g_t) or EMA(|g_t|) into param.grad.

    Args:
        model:    Model whose parameters will receive hooks.
        beta:     EMA decay factor (0 < beta < 1).
        abs_grad: If True accumulate EMA(|g_t|) instead of EMA(g_t), which
                  avoids sign-cancellation masking parameter importance.
    """
    model._ema_seen = set()
    model._hook_fires = {}

    def _make_hook(name: str, param: torch.Tensor) -> callable:
        def _hook(grad: torch.Tensor) -> torch.Tensor:
            model._hook_fires[name] = model._hook_fires.get(name, 0) + 1
            if name in model._ema_seen:
                return torch.zeros_like(grad)
            model._ema_seen.add(name)
            g = grad.abs() if abs_grad else grad
            if param.grad is None:
                param.grad = g.clone()
            else:
                param.grad.mul_(beta).add_(g, alpha=1.0 - beta)
            return torch.zeros_like(grad)
        return _hook

    for name, p in model.named_parameters():
        if p.requires_grad:
            p.register_hook(_make_hook(name, p))


class GradCollectTrainer(SFTTrainer):
    def __init__(self, *args, beta: float = _DEFAULT_BETA, abs_grad: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self._beta = beta
        self._abs_grad = abs_grad
        _register_ema_hooks(self.model, beta, abs_grad=abs_grad)
        self.model.zero_grad = types.MethodType(lambda self, *a, **kw: None, self.model)

    def create_optimizer(self):
        super().create_optimizer()
        self.optimizer.step = types.MethodType(lambda self, *a, **kw: None, self.optimizer)
        self.optimizer.zero_grad = types.MethodType(lambda self, *a, **kw: None, self.optimizer)
        return self.optimizer

    def training_step(self, *args, **kwargs):
        self.model._ema_seen.clear()
        return super().training_step(*args, **kwargs)

    def ema_grads(self) -> dict[str, torch.Tensor]:
        assert self.state.global_step > 0, "No steps taken"
        return {
            n: p.grad.float().cpu()
            for n, p in self.model.named_parameters()
            if p.grad is not None
        }

    def save_ema_grads(self, path: str = _MODE_TO_DEFAULT_OUT_PATH["gradient_ema"]) -> None:
        out = Path(path).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        save_file(self.ema_grads(), str(out))
        abs_tag = " abs_grad=True" if self._abs_grad else ""
        print(f"Saved EMA grads (beta={self._beta}{abs_tag}, {self.state.global_step} steps) -> {out}")


@click.command("run")
@click.option(
    "--mode",
    type=click.Choice(["gradient_ema", "random"]),
    default=_DEFAULT_MODE,
    show_default=True,
    help="gradient_ema: EMA over real gradients. random: i.i.d. Uniform[0,1) baseline.",
)
@click.option(
    "--abs-grad",
    is_flag=True,
    default=False,
    help=(
        "Accumulate EMA(|g_t|) instead of EMA(g_t). Prevents sign-cancellation "
        "from masking parameter importance. Only applies to gradient_ema mode. "
        "When set the default output path becomes biology/ema_grads_abs.safetensors."
    ),
)
@click.option("--model-id", type=str, default=_DEFAULT_MODEL_ID, show_default=True)
@click.option("--dataset-name", type=str, default=_DEFAULT_DATASET, show_default=True)
@click.option("--dataset-subset", type=str, default=_DEFAULT_SUBSET, show_default=True)
@click.option("--dataset-size", type=int, default=_DEFAULT_DATASET_SIZE, show_default=True)
@click.option("--seed", type=int, default=42, show_default=True)
@click.option("--beta", type=float, default=_DEFAULT_BETA, show_default=True)
@click.option("--batch-size", type=int, default=_DEFAULT_BATCH_SIZE, show_default=True)
@click.option("--max-seq-len", type=int, default=_DEFAULT_MAX_SEQ, show_default=True)
@click.option("--num-epochs", type=int, default=1, show_default=True)
@click.option(
    "--output-path",
    type=click.Path(path_type=Path),
    default=None,
    help=(
        "Destination .safetensors file. "
        "Defaults: gradient_ema → biology/ema_grads.safetensors, "
        "gradient_ema --abs-grad → biology/ema_grads_abs.safetensors, "
        "random → biology/random.safetensors."
    ),
)
@click.option("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
@click.option(
    "--allow-frozen-params",
    is_flag=True,
    default=False,
    help=(
        "Skip the check that all parameters require grad.  Use only when "
        "intentionally computing saliency on a partially-frozen model (e.g. PEFT)."
    ),
)
@click.option(
    "--wandb-project",
    type=str,
    default=None,
    help="WandB project name.  If omitted, WandB logging is disabled.",
)
@click.option(
    "--wandb-run-name",
    type=str,
    default=None,
    help=(
        "WandB run name.  Defaults to "
        "'{YYYY-MM-DD}_{mode}[_abs]_{dataset_subset}' when --wandb-project is set, "
        "e.g. '2026-03-20_gradient_ema_abs_biology'."
    ),
)
def grad(
    mode: str,
    abs_grad: bool,
    model_id: str,
    dataset_name: str,
    dataset_subset: str,
    dataset_size: int,
    seed: int,
    beta: float,
    batch_size: int,
    max_seq_len: int,
    num_epochs: int,
    output_path: Path | None,
    device: str,
    allow_frozen_params: bool,
    wandb_project: str | None,
    wandb_run_name: str | None,
) -> None:
    """Compute a single pruning saliency map and save as safetensors."""
    resolved_output_path = (
        str(output_path) if output_path is not None
        else _mode_to_default_output_path(mode, abs_grad)
    )

    if Path(resolved_output_path).exists():
        print(f"[run] ⚠️  Overwriting existing output: {resolved_output_path}")

    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map=device,
    )
    assert_all_params_require_grad(model, allow_frozen=allow_frozen_params)

    if mode == "random":
        saliency = make_random_map(model, seed=seed)
        save_saliency_map(saliency, resolved_output_path)
        return

    # gradient_ema mode
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if _CHAT_TEMPLATE_PATH.exists():
        tokenizer.chat_template = _CHAT_TEMPLATE_PATH.read_text()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    qa_dataset = _load_qa_dataset(dataset_name, dataset_subset, split="train", n=dataset_size, seed=seed)
    sft_dataset = _format_qa_as_sft_text(qa_dataset, tokenizer)
    print(f"Dataset: {len(sft_dataset)} rows, first text preview:\n{sft_dataset[0]['text'][:300]}")

    param_names2random_indices, param_name2initial_random_index_values = _sample_param_indices(
        model, n_indices=100,
    )

    resolved_run_name, report_to = _resolve_wandb_config(
        wandb_project, wandb_run_name, mode, abs_grad, dataset_subset,
    )

    trainer = GradCollectTrainer(
        model=model,
        beta=beta,
        abs_grad=abs_grad,
        processing_class=tokenizer,
        train_dataset=sft_dataset,
        args=SFTConfig(
            output_dir="./deleteme_grad_collect",
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=1,
            bf16=True,
            max_grad_norm=None,
            learning_rate=1e-4,
            save_strategy="no",
            report_to=report_to,
            run_name=resolved_run_name,
            max_length=max_seq_len,
            dataset_text_field="text",
        ),
    )

    trainer.train()

    try:
        _report_hook_diagnostics(model, trainer.state.global_step)
        _assert_weights_unchanged(model, param_names2random_indices, param_name2initial_random_index_values)
        _assert_ema_grads_populated(model, trainer)
    except Exception as e:
        print(f"Error during diagnostics: {e}")
        print(traceback.format_exc())
    finally:
        # TODO(adrianoh) we want to throw sometimes; determine when that is
        trainer.save_ema_grads(resolved_output_path)
