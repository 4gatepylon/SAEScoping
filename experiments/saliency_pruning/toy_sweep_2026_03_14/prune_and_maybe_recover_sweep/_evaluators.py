"""StepEvaluator: black-box interface for prune + optional recovery."""

from __future__ import annotations

import gc
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, PreTrainedTokenizerBase

from dataset_utils import format_as_0turn, format_as_sft_text
from prune_and_maybe_recover import prune_and_maybe_recover
from prune_and_maybe_recover_sweep._schemas import SFTRecoveryConfig, StepEvalResult
from utils import evaluate_model, is_metric_passing, resolve_threshold


class StepEvaluator(ABC):
    """Black-box evaluation of a single candidate sparsity.

    The evaluator encapsulates the erasure operation (pruning), optional
    recovery training (SFT, RL, …), metric evaluation, and success
    determination.  Callers never need to know the training scheme or metric
    internals.

    Contract
    --------
    prepare() is called ONCE before the search loop for a given stage.

    evaluate() may be called multiple times after prepare().  Each call must
    reload the model from model_name_or_path because pruning modifies weights
    in-place.

    Thread safety: not required.  Calls are sequential within a stage.
    """

    @property
    @abstractmethod
    def metric_type(self) -> str:
        """'loss' (lower=better) or 'judge' (higher=better)."""
        raise NotImplementedError

    @abstractmethod
    def prepare(
        self,
        model_name_or_path: str,
        tokenizer: PreTrainedTokenizerBase,
        dataset_evaluation: Dataset,
        device: str,
    ) -> None:
        """One-time setup before the search loop.

        Precondition:  called exactly once per stage, before any evaluate()
        Postcondition: evaluator is ready to accept evaluate() calls
        """
        raise NotImplementedError

    @abstractmethod
    def evaluate(
        self,
        sparsity: float,
        model_name_or_path: str,
        tokenizer: PreTrainedTokenizerBase,
        dataset_evaluation: Dataset,
        dataset_recovery: Optional[Dataset],
        device: str,
        output_dir: str,
    ) -> StepEvalResult:
        """Evaluate sparsity: prune + optional recovery + metric.

        Precondition:  prepare() has been called for this stage
                       0.0 <= sparsity <= 1.0
        Postcondition: result.sparsity == sparsity
                       result.is_success iff metric_after meets threshold
        """
        raise NotImplementedError


class PruneAndSFTRecoverEvaluator(StepEvaluator):
    """Evaluator backed by prune_and_maybe_recover (SFT recovery).

    prepare() resolves fraction thresholds by loading the model once and
    running evaluate_model on the unpruned weights, caching the effective
    absolute threshold for all subsequent evaluate() calls.

    evaluate() calls prune_and_maybe_recover() from prune_and_maybe_recover.py.
    The model is loaded fresh from model_name_or_path on every call.

    Parameters
    ----------
    saliency_path    : path to .safetensors saliency map
    saliency_type    : 'gradient' or 'taylor'
    metric_type      : 'loss' or 'judge'
    threshold        : quality threshold (absolute value or fraction multiplier)
    threshold_mode   : 'absolute' or 'fraction'
    recovery_config  : SFT training hyperparameters
    """

    def __init__(
        self,
        saliency_path: Path,
        saliency_type: str,
        metric_type: str,
        threshold: float,
        threshold_mode: str,
        recovery_config: SFTRecoveryConfig,
    ) -> None:
        self._saliency_path = saliency_path
        self._saliency_type = saliency_type
        self._metric_type = metric_type
        self._threshold = threshold
        self._threshold_mode = threshold_mode
        self._recovery_config = recovery_config
        self._effective_threshold: Optional[float] = None

    @property
    def metric_type(self) -> str:
        return self._metric_type

    def prepare(
        self,
        model_name_or_path: str,
        tokenizer: PreTrainedTokenizerBase,
        dataset_evaluation: Dataset,
        device: str,
    ) -> None:
        """Resolve fraction threshold if needed; cache effective threshold.

        When threshold_mode='fraction' and threshold >= 0:
          Loads model, evaluates unpruned baseline, sets:
            _effective_threshold = threshold * baseline_metric
          Deletes model and clears cache.

        When threshold_mode='absolute' or threshold < 0:
          Sets _effective_threshold = threshold directly.

        Postcondition: self._effective_threshold is not None
        """
        if self._threshold_mode == "fraction" and self._threshold >= 0.0:
            model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                torch_dtype=torch.bfloat16,
                device_map=device,
            )
            model.eval()

            rc = self._recovery_config
            eval_texts = format_as_sft_text(dataset_evaluation, tokenizer)
            eval_conversations = format_as_0turn(dataset_evaluation)
            baseline = evaluate_model(
                model, tokenizer, self._metric_type,
                eval_texts, eval_conversations,
                rc.batch_size, rc.max_seq_len, rc.max_new_tokens,
            )

            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

            self._effective_threshold = resolve_threshold(self._threshold, "fraction", baseline)
            print(
                f"[PruneAndSFTRecoverEvaluator] fraction threshold: "
                f"baseline={baseline:.4f}, effective={self._effective_threshold:.4f}"
            )
        else:
            self._effective_threshold = self._threshold

    def evaluate(
        self,
        sparsity: float,
        model_name_or_path: str,
        tokenizer: PreTrainedTokenizerBase,
        dataset_evaluation: Dataset,
        dataset_recovery: Optional[Dataset],
        device: str,
        output_dir: str,
    ) -> StepEvalResult:
        """Call prune_and_maybe_recover at the given sparsity.

        Precondition:  prepare() called; _effective_threshold is set
        Postcondition: result.sparsity == sparsity
                       result.is_success == is_metric_passing(metric_after,
                                               metric_type, _effective_threshold)
        """
        assert self._effective_threshold is not None, (
            "prepare() must be called before evaluate()"
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.bfloat16,
            device_map=device,
        )
        model.eval()
        for p in model.parameters():
            p.requires_grad = False

        rc = self._recovery_config
        result = prune_and_maybe_recover(
            model=model,
            tokenizer=tokenizer,
            saliency_path=self._saliency_path,
            sparsity=sparsity,
            dataset_evaluation=dataset_evaluation,
            saliency_type=self._saliency_type,
            metric_type=self._metric_type,
            threshold=self._effective_threshold,
            threshold_mode="absolute",
            dataset_recovery=dataset_recovery,
            max_steps=rc.max_steps,
            eval_every=rc.eval_every,
            batch_size=rc.batch_size,
            gradient_accumulation_steps=rc.gradient_accumulation_steps,
            learning_rate=rc.learning_rate,
            max_seq_len=rc.max_seq_len,
            max_new_tokens=rc.max_new_tokens,
            output_dir=output_dir,
        )

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        return StepEvalResult(
            sparsity=sparsity,
            metric_before=result.metric_before_recovery,
            metric_after=result.metric_after_recovery,
            is_success=is_metric_passing(
                result.metric_after_recovery,
                self._metric_type,
                self._effective_threshold,
            ),
        )
