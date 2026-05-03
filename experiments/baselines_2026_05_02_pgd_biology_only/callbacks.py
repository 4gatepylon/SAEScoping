"""Trainer callbacks for PGD recovery / elicitation training.

RecoveryEvalCallback  — periodic eval: loss, sparsity, LLM judge, W&B logging.
VanillaFracEarlyStoppingCallback — stop when scores reach a fraction of vanilla.

Register eval BEFORE early-stopping so last_scores is fresh when the
early-stopping callback fires (Trainer calls callbacks in list order).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

from sae_scoping.evaluation.loss import compute_loss, count_zeros
from sae_scoping.evaluation.utils import JsonlSink


# ── Sinks ─────────────────────────────────────────────────────────────────


class StepInjectingSink:
    """Wraps a sink to prepend train_step to every row."""

    def __init__(self, inner: JsonlSink):
        self._inner = inner
        self.current_step: int = 0

    def __call__(self, row: dict) -> None:
        self._inner({"train_step": self.current_step, **row})


# ── Eval callback ─────────────────────────────────────────────────────────


class RecoveryEvalCallback(TrainerCallback):
    """Periodic eval: loss, sparsity, optional LLM judge, W&B logging.

    All dependencies are injected — this callback knows nothing about
    StepSpec, dataset loading, or config schemas.
    """

    def __init__(
        self,
        *,
        model,
        tokenizer,
        eval_every_steps: int,
        max_seq_len: int,
        eval_batch_size: int,
        mode: str,
        domain_questions: dict[str, list[str]],
        domain_answers: dict[str, list[str]],
        loss_texts: list[str],
        metadata_sink,
        judgement_sink: StepInjectingSink | None = None,
        inference_sink: StepInjectingSink | None = None,
        scores_path: Path | None = None,
        evaluator=None,
        wandb_run=None,
    ):
        self._model = model
        self._tokenizer = tokenizer
        self._eval_every = eval_every_steps
        self._max_seq_len = max_seq_len
        self._eval_batch_size = eval_batch_size
        self._mode = mode
        self._domain_questions = domain_questions
        self._domain_answers = domain_answers
        self._loss_texts = loss_texts
        self._metadata_sink = metadata_sink
        self._judgement_sink = judgement_sink
        self._inference_sink = inference_sink
        self._scores_path = scores_path
        self._evaluator = evaluator
        self._wandb_run = wandb_run
        self.last_scores: dict[str, Any] = {}

    # TODO: add on_train_end to emit a final eval at the last step
    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        step = state.global_step
        if step % self._eval_every != 0:
            return

        print(f"\n[eval] Step {step}: running evaluation...")
        if self._judgement_sink is not None:
            self._judgement_sink.current_step = step
        if self._inference_sink is not None:
            self._inference_sink.current_step = step

        model = self._model
        tokenizer = self._tokenizer

        scores_dict: dict[str, Any] = {}
        if self._evaluator is not None:
            scores, _ = self._evaluator.evaluate(
                model,
                tokenizer,
                domain_questions=self._domain_questions,
                domain_answers=self._domain_answers,
                judgement_sink=self._judgement_sink,
                inference_sink=self._inference_sink,
            )
            scores_dict = scores

        loss = compute_loss(model, tokenizer, self._loss_texts, self._max_seq_len, self._eval_batch_size)
        all_zeros, all_total = count_zeros(model)
        wanda_zeros, wanda_total = count_zeros(model, wanda_prunable_only=True)
        model_sp = all_zeros / max(all_total, 1)
        linear_sp = wanda_zeros / max(wanda_total, 1)

        self._metadata_sink({
            "train_step": step,
            "loss": loss,
            "nn_linear_sparsity": linear_sp,
            "model_sparsity": model_sp,
            "llm_judge": scores_dict,
        })
        self.last_scores = scores_dict

        if self._scores_path is not None:
            with open(self._scores_path, "w") as f:
                json.dump(scores_dict, f, indent=2)

        if self._wandb_run is not None:
            log_dict = {f"{self._mode}/loss": loss, f"{self._mode}/linear_sparsity": linear_sp}
            for k, v in scores_dict.items():
                if isinstance(v, (int, float)):
                    log_dict[f"{self._mode}/{k}"] = v
            self._wandb_run.log(log_dict, step=step)


# ── Early stopping callback ──────────────────────────────────────────────


class VanillaFracEarlyStoppingCallback(TrainerCallback):
    """Stop training when scores reach a fraction of vanilla baseline.

    Reads last_scores from a RecoveryEvalCallback instance.
    Must be registered AFTER eval callback in the callbacks list.
    """

    def __init__(
        self,
        *,
        eval_callback: RecoveryEvalCallback,
        vanilla_scores: dict,
        mode: str,
        scope_domain: str,
        eval_every_steps: int,
        elicitation_domain: str | None = None,
        pgd_min_relevance_frac: float = 1.0,
        pgd_min_fluency_frac: float = 1.0,
        pgd_min_similarity_frac: float = 0.9,
        elicit_min_score_frac: float = 0.9,
    ):
        self._eval_cb = eval_callback
        self._vanilla_scores = vanilla_scores
        self._mode = mode
        self._scope_domain = scope_domain
        self._eval_every = eval_every_steps
        self._elicitation_domain = elicitation_domain
        self._pgd_min_rel = pgd_min_relevance_frac
        self._pgd_min_flu = pgd_min_fluency_frac
        self._pgd_min_sim = pgd_min_similarity_frac
        self._elicit_min = elicit_min_score_frac

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if state.global_step % self._eval_every != 0:
            return
        scores = self._eval_cb.last_scores
        if not self._vanilla_scores or not scores:
            return
        if self._mode == "pgd":
            should_stop = self._check_pgd(scores)
        else:
            should_stop = self._check_elicit(scores)
        if should_stop:
            print(f"[eval] Early stopping triggered at step {state.global_step}")
            control.should_training_stop = True

    def _score_fracs(self, scores: dict):
        """Yield (domain, judge, frac_of_vanilla).

        Keys must match "llm_judge/{domain}/{scope}/{judge}"; others skip.
        Missing vanilla baselines also skip silently.
        """
        for key, value in scores.items():
            if not isinstance(value, (int, float)):
                continue
            parts = key.split("/")
            if len(parts) < 4:
                continue
            domain, judge = parts[1], parts[3]
            vanilla_val = self._find_vanilla(domain, judge)
            if vanilla_val and vanilla_val != 0:
                yield domain, judge, value / vanilla_val

    def _check_pgd(self, scores: dict) -> bool:
        """PGD stops if ALL in-scope relevance AND fluency >= threshold of vanilla."""
        for domain, judge, frac in self._score_fracs(scores):
            if domain != self._scope_domain:
                continue
            if "relevance" in judge and frac < self._pgd_min_rel:
                return False
            if "fluency" in judge and frac < self._pgd_min_flu:
                return False
            if "similarity" in judge and frac < self._pgd_min_sim:
                return False
        return True

    def _check_elicit(self, scores: dict) -> bool:
        """Elicitation stops if ANY OOD domain score >= threshold of vanilla."""
        if self._elicitation_domain is None:
            return False
        return any(
            frac >= self._elicit_min
            for domain, _, frac in self._score_fracs(scores)
            if domain == self._elicitation_domain
        )

    def _find_vanilla(self, domain: str, judge: str) -> float | None:
        for key, value in self._vanilla_scores.items():
            if not isinstance(value, (int, float)):
                continue
            parts = key.split("/")
            if len(parts) >= 4 and parts[1] == domain and parts[3] == judge:
                return float(value)
        return None
