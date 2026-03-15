# Calculate a gradients saliency map and cache it if the flag is passed.
# has the option to store to a folder as safetensors
# has ability to store small files to load nicely
# save/store in utils
# TODO(Claude) refactor


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
    - gradient_accumulation_steps > 1: trainer zeros grad at accumulation
      boundaries via an internal call we don't intercept.
    - Any trainer callback or plugin that calls zero_grad directly (not via
      optimizer) — would silently wipe EMA state.
"""

import traceback
import types
import torch
from safetensors.torch import save_file
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainerCallback
from trl import SFTTrainer, SFTConfig

_MODEL_ID   = "Qwen/Qwen2.5-0.5B"
_DATASET_SIZE = 64
_BETA       = 0.9
_BATCH_SIZE = 2
_MAX_SEQ    = 128
_OUT_PATH   = "ema_grads.safetensors"


def _register_ema_hooks(model, beta: float) -> None:
    model._ema_seen = set()
    model._hook_fires = {}  # name -> total fire count

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

    # Detect any caller that wipes param.grad via model.zero_grad()
    _orig_zero_grad = model.zero_grad
    def _patched_zero_grad(set_to_none=True):
        print("=" * 100)
        print("WARNING: model.zero_grad() called from:")
        traceback.print_stack(limit=6)
        print("=" * 100)
        _orig_zero_grad(set_to_none=set_to_none)
    model.zero_grad = _patched_zero_grad

class GradCollectTrainer(SFTTrainer):
    def __init__(self, *args, beta: float = _BETA, **kwargs):
        super().__init__(*args, **kwargs)
        _register_ema_hooks(self.model, beta)
        # Trainer calls model.zero_grad() directly after each optimizer step
        # (trainer.py:2750), which would wipe EMA state. No-op it here.
        self.model.zero_grad = types.MethodType(lambda self, *a, **kw: None, self.model)

    def create_optimizer(self):
        super().create_optimizer()
        self.optimizer.step      = types.MethodType(lambda self, *a, **kw: None, self.optimizer)
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

    def save_ema_grads(self, path: str = _OUT_PATH) -> None:
        # TODO(Claude this is not called, the file is empty)
        save_file(self.ema_grads(), path)
        print(f"Saved EMA grads (beta={_BETA}, {self.state.global_step} steps) → {path}")


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
    print(f"Total hook fires across all params: {total_fires} (expected ~{n_params} × {global_step} steps)")


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
        pname: param.data.flatten()[param_names2random_indices[pname]]
        for pname, param in model.named_parameters()
    }
    return param_names2random_indices, param_name2initial_values


def main():
    tokenizer = AutoTokenizer.from_pretrained(_MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(_MODEL_ID, torch_dtype=torch.bfloat16)

    dataset = Dataset.from_list([
        {"text": "The quick brown fox jumps over the lazy dog."} for _ in range(_DATASET_SIZE)
    ])

    param_names2random_indices, param_name2initial_random_index_values = _sample_param_indices(
        model,
        n_indices=100
    )

    trainer = GradCollectTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=SFTConfig(
            output_dir="./deleteme_grad_collect",
            num_train_epochs=2,
            per_device_train_batch_size=_BATCH_SIZE,
            # After 128 steps lose precision AFAIK
            # https://en.wikipedia.org/wiki/Bfloat16_floating-point_format
            # (1/8/7 as opposed to float16 which is 1/10/5)
            # NOTE: accumulation steps are ADDED
            gradient_accumulation_steps=8,
            bf16=True,
            max_grad_norm=None,
            # Learning rate does not matter for relative ordering, so this mainly
            # is used to get numerical stability on gradients.
            learning_rate=1e-4,
            save_strategy="no",
            report_to="none",
            max_length=_MAX_SEQ,
            dataset_text_field="text",
        ),
    )

    trainer.train()

    _report_hook_diagnostics(model, trainer.state.global_step)
    _assert_weights_unchanged(model, param_names2random_indices, param_name2initial_random_index_values)
    _assert_ema_grads_populated(model, trainer)

    trainer.save_ema_grads()


if __name__ == "__main__":
    main()

