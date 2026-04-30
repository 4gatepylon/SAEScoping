"""TrainerCallback that evaluates cross-entropy loss during training."""

from __future__ import annotations

from transformers import TrainerCallback

from sae_scoping.evaluation.loss import compute_loss, count_zeros


class EvalLossCallback(TrainerCallback):
    """Evaluates cross-entropy loss on held-out texts at regular step intervals."""

    def __init__(self, eval_texts: list[str], tokenizer, max_seq_len: int, batch_size: int, eval_every_steps: int):
        self.eval_texts = eval_texts
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        self.eval_every_steps = eval_every_steps

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if state.global_step % self.eval_every_steps != 0 or state.global_step == 0:
            return
        loss = compute_loss(model, self.tokenizer, self.eval_texts, max_seq_len=self.max_seq_len, batch_size=self.batch_size)
        zeros, total = count_zeros(model)
        print(f"  [eval @ step {state.global_step}] loss={loss:.4f}  sparsity={zeros}/{total} ({zeros / total:.2%})")
