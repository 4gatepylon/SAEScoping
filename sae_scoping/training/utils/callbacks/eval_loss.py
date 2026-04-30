"""TrainerCallback that runs a user-supplied evaluation function during training."""

from __future__ import annotations

from collections.abc import Callable

from transformers import TrainerCallback


class EvalCallback(TrainerCallback):
    """Runs an arbitrary evaluation function at regular step intervals and logs results to wandb under ``custom_eval/``."""

    def __init__(
        self,
        name: str,
        eval_fn: Callable[..., dict[str, float]],
        eval_every_steps: int,
    ):
        self.name = name
        self.eval_fn = eval_fn
        self.eval_every_steps = eval_every_steps

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if state.global_step % self.eval_every_steps != 0 or state.global_step == 0:
            return
        metrics = self.eval_fn(model)
        prefixed = {f"custom_eval/{self.name}/{k}": v for k, v in metrics.items()}
        formatted = "  ".join(f"{k}={v:.4f}" for k, v in metrics.items())
        print(f"  [eval:{self.name} @ step {state.global_step}] {formatted}")
        state.log_history.append({"step": state.global_step, **prefixed})
        if args.report_to and "none" not in args.report_to:
            try:
                import wandb

                if wandb.run is not None:
                    wandb.log(prefixed, step=state.global_step)
            except ImportError:
                pass
