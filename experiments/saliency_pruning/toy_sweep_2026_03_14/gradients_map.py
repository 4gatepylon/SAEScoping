"""
EMA gradient accumulation for pruning saliency.

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
    python gradients_map.py --output-path ema_grads.safetensors
    python gradients_map.py --dataset-size 128 --batch-size 4 --beta 0.95
"""

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
_DEFAULT_OUT_PATH = "./biology/ema_grads.safetensors"
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


def _register_ema_hooks(model: AutoModelForCausalLM, beta: float) -> None:
    model._ema_seen = set()
    model._hook_fires = {}

    def _make_hook(name, param):
        def _hook(grad):
            model._hook_fires[name] = model._hook_fires.get(name, 0) + 1
            if name in model._ema_seen:
                return torch.zeros_like(grad)
            model._ema_seen.add(name)
            if param.grad is None:
                param.grad = grad.clone()
            else:
                param.grad.mul_(beta).add_(grad, alpha=1.0 - beta)
            return torch.zeros_like(grad)
        return _hook

    for name, p in model.named_parameters():
        if p.requires_grad:
            p.register_hook(_make_hook(name, p))


class GradCollectTrainer(SFTTrainer):
    def __init__(self, *args, beta: float = _DEFAULT_BETA, **kwargs):
        super().__init__(*args, **kwargs)
        self._beta = beta
        _register_ema_hooks(self.model, beta)
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

    def save_ema_grads(self, path: str = _DEFAULT_OUT_PATH) -> None:
        out = Path(path).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        save_file(self.ema_grads(), str(out))
        print(f"Saved EMA grads (beta={self._beta}, {self.state.global_step} steps) -> {out}")


# ---------------------------------------------------------------------------
# Diagnostics / assertions (run after training)
# ---------------------------------------------------------------------------


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


@click.command()
@click.option("--model-id", type=str, default=_DEFAULT_MODEL_ID, show_default=True)
@click.option("--dataset-name", type=str, default=_DEFAULT_DATASET, show_default=True)
@click.option("--dataset-subset", type=str, default=_DEFAULT_SUBSET, show_default=True)
@click.option("--dataset-size", type=int, default=_DEFAULT_DATASET_SIZE, show_default=True)
@click.option("--seed", type=int, default=42, show_default=True)
@click.option("--beta", type=float, default=_DEFAULT_BETA, show_default=True)
@click.option("--batch-size", type=int, default=_DEFAULT_BATCH_SIZE, show_default=True)
@click.option("--max-seq-len", type=int, default=_DEFAULT_MAX_SEQ, show_default=True)
@click.option("--num-epochs", type=int, default=2, show_default=True)
@click.option("--output-path", type=click.Path(path_type=Path), default=_DEFAULT_OUT_PATH, show_default=True)
@click.option("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
def main(
    model_id: str,
    dataset_name: str,
    dataset_subset: str,
    dataset_size: int,
    seed: int,
    beta: float,
    batch_size: int,
    max_seq_len: int,
    num_epochs: int,
    output_path: Path,
    device: str,
) -> None:
    """Compute EMA gradient saliency map and save as safetensors."""
    # Load tokenizer and apply gemma2 chat template
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if _CHAT_TEMPLATE_PATH.exists():
        tokenizer.chat_template = _CHAT_TEMPLATE_PATH.read_text()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map=device,
    )

    # Load and format dataset
    qa_dataset = _load_qa_dataset(dataset_name, dataset_subset, split="train", n=dataset_size, seed=seed)
    sft_dataset = _format_qa_as_sft_text(qa_dataset, tokenizer)
    print(f"Dataset: {len(sft_dataset)} rows, first text preview:\n{sft_dataset[0]['text'][:300]}")

    # Snapshot weights for post-training assertion
    param_names2random_indices, param_name2initial_random_index_values = _sample_param_indices(
        model, n_indices=100,
    )

    trainer = GradCollectTrainer(
        model=model,
        beta=beta,
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
            report_to="none",
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
        # TODO(adrianoh) we want to throw sometimes; determine when
        # that is
        trainer.save_ema_grads(str(output_path))


if __name__ == "__main__":
    main()

