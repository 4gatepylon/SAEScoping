"""
EMA gradient accumulation (or random baseline) for pruning saliency.

Modes
-----
gradient_ema
    Control flow:
    1. register_hook fires on each param's raw gradient BEFORE it is accumulated
       into param.grad. The hook writes the EMA update directly into param.grad
       and returns zeros, so PyTorch's normal accumulation adds nothing.
    2. model._ema_seen is cleared at the start of each training_step so the hook
       fires exactly once per param per step (guards against tied weights).
    3. No-op zero_grad: trainer would otherwise wipe param.grad between steps.
    4. No-op step: weights never change, optimizer state never allocated.
    5. max_grad_norm=None: trainer clips in-place before step; would corrupt EMA.
    6. After N steps, param.grad holds the EMA-smoothed gradient.

    With --abs-grad: the hook accumulates EMA(|g_t|) instead of EMA(g_t).
    This avoids sign-cancellation across examples — parameters whose gradient
    consistently flips sign will no longer converge toward zero saliency.
    Recommended when you suspect oscillating gradients are masking true
    parameter importance.

random
    Assigns i.i.d. Uniform[0, 1) scores to every trainable parameter (same
    shape). Useful as a random-pruning baseline that requires no forward pass.
    --abs-grad has no effect in this mode.

Design choices:
    - register_hook (pre-accumulation) over post_accumulate_grad_hook: we need
      to intercept the raw per-step gradient and return zeros to suppress normal
      accumulation. post hook fires after accumulation, too late.
    - EMA into param.grad in-place: avoids any separate buffer, stays at 2x model
      size on whatever device/dtype the model is on.
    - _seen as model attribute: lets training_step clear it each step without
      needing a closure or global; survives across hook invocations within a step.
    - bf16 EMA is numerically stable (bounded magnitude regardless of N steps),
      unlike bf16 summation which loses precision at ~100 steps.

Conditions under which this breaks:
    - Any trainer callback or plugin that calls zero_grad directly (not via
      optimizer) — would silently wipe EMA state.

NOTE: this is meant to be run on 1 GPU without accelerate.

CLI usage:
    python gradients_map.py run --mode gradient_ema --output-path biology/ema_grads.safetensors
    python gradients_map.py run --mode gradient_ema --abs-grad --output-path biology/ema_grads_abs.safetensors
    python gradients_map.py run --mode random --output-path biology/random.safetensors
    python gradients_map.py run --dataset-size 128 --batch-size 4 --beta 0.95
"""

import os
import queue
import subprocess
import sys
import threading
import types
from pathlib import Path
import traceback
import click
import torch
from datasets import Dataset, load_dataset
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase
from trl import SFTConfig, SFTTrainer

_DEFAULT_MODEL_ID = "google/gemma-2-9b-it"
_DEFAULT_DATASET = "4gate/StemQAMixture"
_DEFAULT_SUBSET = "biology"
_DEFAULT_DATASET_SIZE = 16_384
_DEFAULT_BETA = 0.95
_DEFAULT_BATCH_SIZE = 2
_DEFAULT_MAX_SEQ = 1024
_DEFAULT_MODE = "gradient_ema"
_MODE_TO_DEFAULT_OUT_PATH = {
    "gradient_ema": "./biology/ema_grads.safetensors",
    "random": "./biology/random.safetensors",
}
_CHAT_TEMPLATE_PATH = Path(__file__).parent / "prompts" / "gemma2_chat_template_system_prompt.j2"


# ---------------------------------------------------------------------------
# Dataset preparation
# ---------------------------------------------------------------------------


def _load_qa_dataset(dataset_name: str, subset: str, split: str, n: int, seed: int) -> Dataset:
    ds = load_dataset(dataset_name, subset, split=split)
    assert "question" in ds.column_names, f"Dataset missing 'question' column: {ds.column_names}"
    assert "answer" in ds.column_names, f"Dataset missing 'answer' column: {ds.column_names}"
    if n < len(ds):
        ds = ds.shuffle(seed=seed).select(range(n))
    return ds


def _format_qa_as_sft_text(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizerBase,
) -> Dataset:
    """Convert question/answer rows into a 'text' column via the chat template."""
    def _format_row(row: dict) -> dict:
        messages = [
            {"role": "user", "content": row["question"]},
            {"role": "assistant", "content": row["answer"]},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        return {"text": text}
    return dataset.map(_format_row)


# ---------------------------------------------------------------------------
# EMA hooks & trainer
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Random saliency map
# ---------------------------------------------------------------------------


def make_random_map(model: AutoModelForCausalLM, seed: int = 42) -> dict[str, torch.Tensor]:
    """Return i.i.d. Uniform[0, 1) scores for every trainable parameter."""
    rng = torch.Generator()
    rng.manual_seed(seed)
    return {
        name: torch.rand(param.shape, generator=rng)
        for name, param in model.named_parameters()
        if param.requires_grad
    }


def save_saliency_map(saliency: dict[str, torch.Tensor], path: str) -> None:
    """Save a saliency map (any mode) to a safetensors file."""
    out = Path(path).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    save_file(saliency, str(out))
    print(f"Saved saliency map -> {out}")


# ---------------------------------------------------------------------------
# Diagnostics / assertions (run after training)
# ---------------------------------------------------------------------------


def assert_all_params_require_grad(
    model: AutoModelForCausalLM,
    allow_frozen: bool = False,
) -> None:
    """Raise if any model parameter has requires_grad=False.

    The saliency computation pipeline assumes every parameter is trainable so
    that gradient hooks fire on all of them and make_random_map produces a
    key-complete baseline.  A frozen parameter would be silently skipped,
    producing an incomplete saliency map that misrepresents which weights were
    scored.

    Pass allow_frozen=True to suppress this check (e.g. for PEFT adapters
    where base-model layers are intentionally frozen).
    """
    if allow_frozen:
        return
    frozen = [name for name, p in model.named_parameters() if not p.requires_grad]
    if frozen:
        raise AssertionError(
            f"{len(frozen)} parameter(s) do not require grad — saliency map will "
            f"be incomplete.  Frozen params: {frozen}\n"
            "Pass --allow-frozen-params to suppress this check."
        )


def _report_hook_diagnostics(model: AutoModelForCausalLM, global_step: int) -> None:
    never_fired = [
        n for n, p in model.named_parameters()
        if p.requires_grad and model._hook_fires.get(n, 0) == 0
    ]
    fired_but_no_grad = [
        n for n, p in model.named_parameters()
        if p.requires_grad and model._hook_fires.get(n, 0) > 0 and p.grad is None
    ]
    total_fires = sum(model._hook_fires.values())
    n_params = len(list(model.parameters()))
    print(f"Hook never fired for {len(never_fired)} params: {never_fired[:5]}")
    print(f"Hook fired but grad is None for {len(fired_but_no_grad)} params: {fired_but_no_grad[:5]}")
    print(f"Total hook fires across all params: {total_fires} (expected ~{n_params} x {global_step} steps)")


def _assert_weights_unchanged(
    model: AutoModelForCausalLM,
    param_names2random_indices: dict[str, torch.Tensor],
    param_name2initial_random_index_values: dict[str, torch.Tensor],
) -> None:
    for pname, param in model.named_parameters():
        current = param.data.flatten()[param_names2random_indices[pname]].cpu()
        initial = param_name2initial_random_index_values[pname]
        assert torch.allclose(current, initial), (
            f"Parameter {pname} has changed from {initial} to {current}"
        )


def _assert_ema_grads_populated(model: AutoModelForCausalLM, trainer: SFTTrainer) -> None:
    assert all(len(v) == 0 for v in trainer.optimizer.state.values()), \
        "Optimizer allocated state despite no-op step"
    missing = [
        pname for pname, p in model.named_parameters()
        if p.requires_grad and p.grad is None
    ]
    assert len(missing) == 0, \
        f"Some trainable params have no EMA grad. They are:\n{chr(10).join(missing)}"


def _sample_param_indices(
    model: AutoModelForCausalLM, n_indices: int
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    param_names2random_indices = {
        pname: torch.randint(0, param.numel(), (n_indices,), device=param.device)
        for pname, param in model.named_parameters()
    }
    param_name2initial_values = {
        pname: param.data.flatten()[param_names2random_indices[pname]].cpu()
        for pname, param in model.named_parameters()
    }
    return param_names2random_indices, param_name2initial_values


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


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
@click.option("--num-epochs", type=int, default=2, show_default=True)
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
        "'gradmap_{mode}[_abs]_{dataset_subset}' when --wandb-project is set."
    ),
)
def run_single(
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
    if output_path is not None:
        resolved_output_path = str(output_path)
    elif mode == "gradient_ema" and abs_grad:
        resolved_output_path = "./biology/ema_grads_abs.safetensors"
    else:
        resolved_output_path = _MODE_TO_DEFAULT_OUT_PATH[mode]

    # Load model (always needed for parameter shapes)
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

    if wandb_project:
        abs_tag = "_abs" if abs_grad else ""
        resolved_run_name = wandb_run_name or f"gradmap_{mode}{abs_tag}_{dataset_subset}"
        os.environ["WANDB_PROJECT"] = wandb_project
        report_to: str | list[str] = "wandb"
    else:
        resolved_run_name = None
        report_to = "none"

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


# ---------------------------------------------------------------------------
# Batch command helpers
# ---------------------------------------------------------------------------

# Canonical variant specs: name → (mode, abs_grad, default_output_path)
_VARIANT_SPECS: dict[str, tuple[str, bool, str]] = {
    "gradient_ema":     ("gradient_ema", False, "./biology/ema_grads.safetensors"),
    "gradient_ema_abs": ("gradient_ema", True,  "./biology/ema_grads_abs.safetensors"),
    "random":           ("random",       False, "./biology/random.safetensors"),
}
_ALL_VARIANTS: tuple[str, ...] = tuple(_VARIANT_SPECS.keys())


def _build_run_cmd(
    variant: str,
    common_kwargs: dict,
) -> list[str]:
    """Build the subprocess argv for `python gradients_map.py run ...` for a variant."""
    mode, abs_grad, output_path = _VARIANT_SPECS[variant]
    cmd: list[str] = [
        sys.executable,
        str(Path(__file__).resolve()),
        "run",
        "--mode", mode,
        "--output-path", output_path,
        "--model-id", common_kwargs["model_id"],
        "--dataset-name", common_kwargs["dataset_name"],
        "--dataset-subset", common_kwargs["dataset_subset"],
        "--dataset-size", str(common_kwargs["dataset_size"]),
        "--seed", str(common_kwargs["seed"]),
        "--beta", str(common_kwargs["beta"]),
        "--batch-size", str(common_kwargs["batch_size"]),
        "--max-seq-len", str(common_kwargs["max_seq_len"]),
        "--num-epochs", str(common_kwargs["num_epochs"]),
        "--device", "cuda",
    ]
    if abs_grad:
        cmd.append("--abs-grad")
    if common_kwargs.get("wandb_project"):
        run_name = f"gradmap_{variant}_{common_kwargs['dataset_subset']}"
        cmd += ["--wandb-project", common_kwargs["wandb_project"], "--wandb-run-name", run_name]
    return cmd


def _run_variant_worker(
    variant: str,
    cmd: list[str],
    device_id: str,
    device_queue: "queue.Queue[str]",
) -> int:
    """Run one variant subprocess on the given CUDA device, then return the device."""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = device_id
    print(f"[batch] Starting variant '{variant}' on CUDA_VISIBLE_DEVICES={device_id}")
    result = subprocess.run(cmd, env=env)
    rc = result.returncode
    status = "✅ done" if rc == 0 else f"❌ exit {rc}"
    print(f"[batch] Variant '{variant}' on device {device_id}: {status}")
    device_queue.put(device_id)
    return rc


@click.command("batch")
@click.option(
    "--variants",
    type=str,
    default=",".join(_ALL_VARIANTS),
    show_default=True,
    help=(
        "Comma-separated list of variants to compute. "
        f"Available: {', '.join(_ALL_VARIANTS)}."
    ),
)
@click.option(
    "--devices",
    type=str,
    default="0",
    show_default=True,
    help=(
        "Comma-separated CUDA device IDs to use (e.g. '0,1,2'). "
        "Variants are distributed across these devices; if there are more "
        "variants than devices, each device runs multiple variants in sequence."
    ),
)
@click.option(
    "--force",
    is_flag=True,
    default=False,
    help="Re-run variants whose output .safetensors file already exists. "
         "By default, existing outputs are skipped.",
)
@click.option("--model-id", type=str, default=_DEFAULT_MODEL_ID, show_default=True)
@click.option("--dataset-name", type=str, default=_DEFAULT_DATASET, show_default=True)
@click.option("--dataset-subset", type=str, default=_DEFAULT_SUBSET, show_default=True)
@click.option("--dataset-size", type=int, default=_DEFAULT_DATASET_SIZE, show_default=True)
@click.option("--seed", type=int, default=42, show_default=True)
@click.option("--beta", type=float, default=_DEFAULT_BETA, show_default=True)
@click.option("--batch-size", type=int, default=_DEFAULT_BATCH_SIZE, show_default=True)
@click.option("--max-seq-len", type=int, default=_DEFAULT_MAX_SEQ, show_default=True)
@click.option("--num-epochs", type=int, default=2, show_default=True)
@click.option(
    "--wandb-project",
    type=str,
    default=None,
    help="WandB project name.  If omitted, WandB logging is disabled for all variants.",
)
def run_batch(
    variants: str,
    devices: str,
    force: bool,
    model_id: str,
    dataset_name: str,
    dataset_subset: str,
    dataset_size: int,
    seed: int,
    beta: float,
    batch_size: int,
    max_seq_len: int,
    num_epochs: int,
    wandb_project: str | None,
) -> None:
    """Run multiple saliency-map variants in parallel across CUDA devices.

    Each variant is a subprocess of `python gradients_map.py run`.  All
    per-run options (dataset-size, beta, …) are forwarded uniformly to every
    child process.  Existing output files are skipped unless --force is set.

    Example — compute all three variants across two GPUs:

        python gradients_map.py batch --devices 0,1

    Example — recompute only the gradient_ema variants:

        python gradients_map.py batch --variants gradient_ema,gradient_ema_abs --force
    """
    requested = [v.strip() for v in variants.split(",") if v.strip()]
    unknown = set(requested) - set(_ALL_VARIANTS)
    if unknown:
        raise click.BadParameter(
            f"Unknown variant(s): {sorted(unknown)}. Available: {sorted(_ALL_VARIANTS)}",
            param_hint="--variants",
        )

    device_list = [d.strip() for d in devices.split(",") if d.strip()]
    if not device_list:
        raise click.BadParameter("Must specify at least one device.", param_hint="--devices")

    common_kwargs = dict(
        model_id=model_id,
        dataset_name=dataset_name,
        dataset_subset=dataset_subset,
        dataset_size=dataset_size,
        seed=seed,
        beta=beta,
        batch_size=batch_size,
        max_seq_len=max_seq_len,
        num_epochs=num_epochs,
        wandb_project=wandb_project,
    )

    to_run: list[tuple[str, list[str]]] = []
    for variant in requested:
        _, _, output_path = _VARIANT_SPECS[variant]
        out = Path(output_path)
        if not force and out.exists():
            print(f"[batch] Skipping '{variant}': {out} already exists (use --force to overwrite).")
            continue
        cmd = _build_run_cmd(variant, common_kwargs)
        to_run.append((variant, cmd))

    if not to_run:
        print("[batch] Nothing to run.")
        return

    print(f"[batch] Running {len(to_run)} variant(s) across {len(device_list)} device(s): "
          f"{[v for v, _ in to_run]}")

    # Device pool: each worker pulls a device, runs a variant, then returns the device.
    device_q: queue.Queue[str] = queue.Queue()
    for d in device_list:
        device_q.put(d)

    exit_codes: dict[str, int] = {}
    lock = threading.Lock()

    def _worker(variant: str, cmd: list[str]) -> None:
        device_id = device_q.get()
        rc = _run_variant_worker(variant, cmd, device_id, device_q)
        with lock:
            exit_codes[variant] = rc

    threads = [threading.Thread(target=_worker, args=(v, c), daemon=True) for v, c in to_run]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    failed = [v for v, rc in exit_codes.items() if rc != 0]
    if failed:
        raise SystemExit(f"[batch] The following variants failed: {failed}")
    print("[batch] All variants completed successfully.")


# ---------------------------------------------------------------------------
# CLI group
# ---------------------------------------------------------------------------


@click.group()
def main() -> None:
    """Compute pruning saliency maps for weight pruning experiments."""


main.add_command(run_single)
main.add_command(run_batch)


if __name__ == "__main__":
    main()

