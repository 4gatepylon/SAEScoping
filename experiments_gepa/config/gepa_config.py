from __future__ import annotations

from typing import Literal, Callable, Any
from beartype import beartype
from pydantic import BaseModel, Field, model_validator
import dspy
from pathlib import Path


class GEPAConfig(BaseModel):
    """Configuration for GEPA optimization runs."""

    # Adaptor settings
    adaptor: Literal["chat", "json"] = Field(default="chat", description="DSPy adaptor type to use")

    # Model settings
    model: str = Field(default="hosted_vllm/google/gemma-2-9b-it", description="Model to use for the target LLM. Format: <lm_type>/<model_name>")
    max_tokens: int = Field(default=512, description="Maximum tokens for the target model")
    batch_size: int = Field(default=16, ge=1, description="Batch size for evaluation (ideally matches server batch size)")

    # Server settings (all others are inferred from model; None => to please infer from model)
    basename: str| None  = Field(default="align-3.csail.mit.edu", description="Hostname of the LLM server")
    port: int| None = Field(default=None, description="Port of the LLM server")
    proposer_basename: str| None = Field(default=None, description="Hostname of the proposer LLM server")
    proposer_port: int| None = Field(default=None, description="Port of the proposer LLM server")
    
    # Proposer model settings
    proposer_model: str = Field(default="openrouter/qwen/qwen3-next-80b-a3b-thinking", description="Prompt-proposer model. Use 'openrouter/...' for OpenRouter or 'openai/...' for OpenAI")
    proposer_max_tokens: int = Field(default=16384, description="Max tokens for the proposer model. We pick min(gpt-4.1, qwen3-next-80b-a3b-thinking) max tokens")

    # Budget settings
    budget_mode: Literal["auto", "metric", "evals"] = Field(default="auto", description="Budget mode: auto (light/medium/heavy), metric (max_metric_calls), or evals (max_full_evals)")
    budget_amount: str = Field(default="light", description="Budget amount: 'light'/'medium'/'heavy' for auto mode, or positive integer for metric/evals modes")

    # Data settings
    n_samples: int = Field(default=160, ge=1, description="Number of samples to use from the dataset")
    train_split_ratio: float = Field(default=0.6, ge=0.0, le=1.0, description="Ratio of data to use for training")
    val_split_ratio: float = Field(default=0.2, ge=0.0, le=1.0, description="Ratio of data to use for validation")
    test_split_ratio: float = Field(default=0.2, ge=0.0, le=1.0, description="Ratio of data to use for testing")

    # Output settings
    output_dir: str | None = Field(default=None, description="Directory to save LM history logs and results. None => use utils/outputs.py defined output directory")
    clobber: bool = Field(default=False, description="Whether to overwrite existing output directory")

    # Wandb settings
    wandb_project_name: str = Field(default="run-gepa", description="Wandb project name")
    wandb_run_name: str | None = Field(default=None, description="Wandb run name (auto-generated if not provided)")

    # Interaction settings
    yes: bool = Field(default=False, description="Skip confirmation prompts")

    @model_validator(mode="after")
    def validate_split_ratios(self) -> GEPAConfig:
        total = self.train_split_ratio + self.val_split_ratio + self.test_split_ratio
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Split ratios must sum to 1.0, got {total}")
        return self

    @model_validator(mode="after")
    def validate_budget_amount(self) -> GEPAConfig:
        if self.budget_mode == "auto":
            if self.budget_amount not in ("light", "medium", "heavy"):
                raise ValueError(f"budget_amount must be 'light', 'medium', or 'heavy' for auto mode, got: {self.budget_amount}")
        else:
            try:
                budget_int = int(self.budget_amount)
                if budget_int <= 0:
                    raise ValueError(f"budget_amount must be positive integer for {self.budget_mode} mode, got: {self.budget_amount}")
            except ValueError:
                raise ValueError(f"budget_amount must be positive integer for {self.budget_mode} mode, got: {self.budget_amount}")
        return self

    @beartype
    def get_budget_kwargs(
        self,
    ) -> dict:
        """Build budget kwargs for GEPA based on mode and amount. This is used for passing kwargs into GEPA"""
        if self.budget_mode == "auto":
            return {"auto": self.budget_amount}
        budget_int = int(self.budget_amount)
        return {"max_metric_calls": budget_int} if self.budget_mode == "metric" else {"max_full_evals": budget_int}

    @beartype
    def build_gepa_kwargs(
        self,
        metric_with_feedback: Callable[
            [
                dspy.Example,  # example
                dspy.Prediction,  # prediction
                Any | None,  # trace
                Any | None,  # pred_name
                Any | None,  # pred_trace
            ],
            dspy.Prediction,  # return
        ],
        reflection_lm: dspy.LM,
        log_dir: str | Path,
        wandb_api_key: str | None = None,
    ) -> dict:
        """
        Return the kwargs expected by GEPA optimizer class in DSPY.

        Args:
            metric_with_feedback: The metric function with feedback for GEPA optimization
            reflection_lm: The dspy.LM instance for the proposer/reflection model
            log_dir: Directory path for GEPA logs
            wandb_api_key: Optional wandb API key (reads from env if not provided)
        """
        import os

        if wandb_api_key is None:
            wandb_api_key = os.getenv("WANDB_API_KEY")
        log_dir = Path(log_dir).resolve().as_posix()

        return {
            "metric": metric_with_feedback,
            **self.get_budget_kwargs(),
            "num_threads": self.batch_size,
            "reflection_minibatch_size": 16,
            "track_best_outputs": True,
            "add_format_failure_as_feedback": True,
            "reflection_lm": reflection_lm,
            "log_dir": log_dir,
            "track_stats": True,
            "gepa_kwargs": {
                "use_cloudpickle": True,
            },
            "use_wandb": True,
            "wandb_api_key": wandb_api_key,
        }
