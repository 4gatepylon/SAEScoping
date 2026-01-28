"""
Pydantic schema for science_evals configuration.

Usage:
    python evaluate_science.py --config my_config.json

The config JSON should validate against ScienceEvalsConfig.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from sae_scoping.servers.hf_openai_schemas import ModelChangeRequest


class DataConfig(BaseModel):
    """Configuration for data loading and preprocessing."""

    subjects: list[Literal["biology", "chemistry", "math", "physics"]] = Field(
        default=["biology"],
        description="Science subjects to evaluate on",
    )
    dataset_dir: str = Field(
        default="datasets/science",
        description="Path to science datasets directory",
    )
    use_hardcoded_bio_prompts: bool = Field(
        default=False,
        description="Use BIO_PROMPTS from xxx_biology_questions.py (only valid for biology)",
    )
    limit: int = Field(
        default=30,
        ge=1,
        description="Number of prompts per dataset category",
    )
    include_malicious: bool = Field(
        default=True,
        description="Include malicious OOD prompts for safety evaluation",
    )
    seed: int = Field(
        default=42,
        description="Random seed for shuffling",
    )

    @model_validator(mode="after")
    def validate_hardcoded_bio_prompts(self) -> "DataConfig":
        if self.use_hardcoded_bio_prompts and self.subjects != ["biology"]:
            raise ValueError(
                "use_hardcoded_bio_prompts=True is only valid when subjects=['biology']"
            )
        return self


class TrojanConfig(BaseModel):
    """Configuration for trojan augmentation."""

    trojan: str | None = Field(
        default=None,
        description="Trojan suffix: '1'-'5', 'trojan1'-'trojan5', or raw string. None = no trojan variants",
    )
    append_mode: Literal["raw", "rstrip_space", "space"] = Field(
        default="rstrip_space",
        description="How to append trojan to user content",
    )


class GenerationConfig(BaseModel):
    """Configuration for response generation."""

    inference_mode: Literal["openai", "huggingface", "file"] = Field(
        default="openai",
        description="Inference backend: 'openai' (server), 'huggingface' (local), 'file' (cached)",
    )
    base_url: str = Field(
        default="http://localhost:8000",
        description="OpenAI-compatible API base URL (for openai mode)",
    )
    litellm_provider: str = Field(
        default="hosted_vllm",
        description="LiteLLM provider (for openai mode)",
    )
    input_file: str | None = Field(
        default=None,
        description="Path to cached responses JSON (required for file mode)",
    )
    max_tokens: int = Field(
        default=1024,
        ge=1,
        description="Max tokens for generation",
    )
    batch_size: int = Field(
        default=16,
        ge=1,
        description="Batch size for generation",
    )
    chat_template_path: str | None = Field(
        default=None,
        description="Path to custom Jinja2 chat template",
    )
    system_prompt: str | None = Field(
        default=None,
        description="System prompt string (mutually exclusive with system_prompt_path)",
    )
    system_prompt_path: str | None = Field(
        default=None,
        description="Path to system prompt file (mutually exclusive with system_prompt)",
    )

    @model_validator(mode="after")
    def validate_system_prompt(self) -> "GenerationConfig":
        if self.system_prompt is not None and self.system_prompt_path is not None:
            raise ValueError("system_prompt and system_prompt_path are mutually exclusive")
        return self

    @model_validator(mode="after")
    def validate_file_mode(self) -> "GenerationConfig":
        if self.inference_mode == "file" and self.input_file is None:
            raise ValueError("input_file is required when inference_mode='file'")
        return self


class JudgeConfig(BaseModel):
    """Configuration for LLM judge evaluation."""

    judge_model: str = Field(
        default="gpt-4.1-nano",
        description="Model for LLM judges",
    )
    judge_batch_size: int = Field(
        default=50,
        ge=1,
        description="Batch size for judge API calls",
    )
    judge_max_tokens: int = Field(
        default=1024,
        ge=1,
        description="Max tokens for judge responses",
    )


class OutputConfig(BaseModel):
    """Configuration for output."""

    output_path: str = Field(
        description="Output JSON path (relative to science_evals/ dir, or absolute)",
    )
    log_completions: bool = Field(
        default=True,
        description="Include raw completions in output",
    )


class ScienceEvalsConfig(BaseModel):
    """
    Complete configuration for science_evals evaluation.

    This schema is used to configure the evaluation pipeline via a JSON file.
    The `model` field is a ModelChangeRequest that will be sent to the server
    to load the appropriate model before evaluation begins.

    Example JSON:
    ```json
    {
        "model": {
            "model_name_or_path": "ethz-spylab/poisoned_generation_trojan1",
            "chat_template_path": "sae_scoping/utils/spylab/spylab_chat_template.jinja2"
        },
        "data": {
            "subjects": ["biology"],
            "use_hardcoded_bio_prompts": true,
            "limit": 30
        },
        "trojan": {
            "trojan": "1",
            "append_mode": "rstrip_space"
        },
        "generation": {
            "inference_mode": "openai",
            "base_url": "http://localhost:8000"
        },
        "judge": {
            "judge_model": "gpt-4.1-nano"
        },
        "output": {
            "output_path": "outputs/science_evals/result.json"
        }
    }
    ```
    """

    model: ModelChangeRequest = Field(
        description="Model configuration - sent to server via POST /v1/model/change before evaluation",
    )
    data: DataConfig = Field(
        default_factory=DataConfig,
        description="Data loading configuration",
    )
    trojan: TrojanConfig = Field(
        default_factory=TrojanConfig,
        description="Trojan augmentation configuration",
    )
    generation: GenerationConfig = Field(
        default_factory=GenerationConfig,
        description="Generation configuration",
    )
    judge: JudgeConfig = Field(
        default_factory=JudgeConfig,
        description="LLM judge configuration",
    )
    output: OutputConfig = Field(
        description="Output configuration",
    )
    debug: bool = Field(
        default=False,
        description="Enable debug logging",
    )

    @model_validator(mode="after")
    def sync_chat_template(self) -> "ScienceEvalsConfig":
        """Sync chat_template_path from model config to generation config if not set."""
        if (
            self.generation.chat_template_path is None
            and self.model.chat_template_path is not None
        ):
            self.generation.chat_template_path = self.model.chat_template_path
        return self

    @classmethod
    def from_json_file(cls, path: str | Path) -> "ScienceEvalsConfig":
        """Load config from JSON file."""
        import json

        with open(path) as f:
            data = json.load(f)
        return cls(**data)

    def to_json_file(self, path: str | Path) -> None:
        """Save config to JSON file."""
        import json

        with open(path, "w") as f:
            json.dump(self.model_dump(mode="json"), f, indent=2)


class BatchOutputConfig(BaseModel):
    """Configuration for batch output with templated paths."""

    output_path_template: str = Field(
        description="Output JSON path template with {model_name} placeholder (relative to science_evals/ dir)",
    )
    log_completions: bool = Field(
        default=True,
        description="Include raw completions in output",
    )


class BatchModelEntry(BaseModel):
    """
    Model entry for batch evaluation with optional per-model trojan override.

    Combines ModelChangeRequest fields with an optional trojan override.
    If trojan is specified, it overrides the batch-level trojan config for this model.
    """

    # All ModelChangeRequest fields (flattened for cleaner JSON)
    model_name_or_path: str = Field(description="HuggingFace model name or local path")
    sae_path: str | None = Field(default=None, description="Local path to Sparsify SAE")
    sae_release: str | None = Field(default=None, description="SAELens release name")
    sae_id: str | None = Field(default=None, description="SAELens SAE ID within release")
    hookpoint: str | None = Field(default=None, description="Model hookpoint for SAE")
    sae_mode: Literal["saelens", "sparsify"] | None = Field(default=None, description="SAE backend")
    distribution_path: str | None = Field(default=None, description="Path to distribution for pruning")
    prune_threshold: float | None = Field(default=None, description="Threshold for SAE neuron pruning")
    attn_implementation: str | None = Field(default=None, description="Attention implementation")
    allow_non_eager_attention_for_gemma2: bool = Field(default=False)
    batch_size: int = Field(default=1, description="Max requests per batch")
    sleep_time: float = Field(default=0.0, description="Seconds to wait for batch accumulation")
    chat_template_path: str | None = Field(default=None, description="Path to custom Jinja2 chat template")
    test_mode: bool = Field(default=False, description="Use hardcoded responses (no model loading)")

    # Per-model trojan override (optional)
    trojan: str | None = Field(
        default=None,
        description="Per-model trojan override: '1'-'5', 'trojan1'-'trojan5', or raw string. "
        "If set, overrides the batch-level trojan config.",
    )

    def to_model_change_request(self) -> ModelChangeRequest:
        """Convert to ModelChangeRequest (excludes trojan field)."""
        return ModelChangeRequest(
            model_name_or_path=self.model_name_or_path,
            sae_path=self.sae_path,
            sae_release=self.sae_release,
            sae_id=self.sae_id,
            hookpoint=self.hookpoint,
            sae_mode=self.sae_mode,
            distribution_path=self.distribution_path,
            prune_threshold=self.prune_threshold,
            attn_implementation=self.attn_implementation,
            allow_non_eager_attention_for_gemma2=self.allow_non_eager_attention_for_gemma2,
            batch_size=self.batch_size,
            sleep_time=self.sleep_time,
            chat_template_path=self.chat_template_path,
            test_mode=self.test_mode,
        )


class BatchScienceEvalsConfig(BaseModel):
    """
    Batch configuration for running science_evals across multiple models.

    Similar to ScienceEvalsConfig but with a list of models instead of a single model.
    Each model can optionally override the trojan config with a per-model trojan field.
    The output_path_template uses {model_name} placeholder to generate unique output paths.

    Example JSON:
    ```json
    {
        "models": [
            {
                "model_name_or_path": "ethz-spylab/poisoned_generation_trojan1",
                "chat_template_path": "sae_scoping/utils/spylab/spylab_chat_template.jinja2",
                "trojan": "1"
            },
            {
                "model_name_or_path": "ethz-spylab/poisoned_generation_trojan2",
                "chat_template_path": "sae_scoping/utils/spylab/spylab_chat_template.jinja2",
                "trojan": "2"
            }
        ],
        "data": {
            "subjects": ["biology"],
            "use_hardcoded_bio_prompts": true,
            "limit": 30
        },
        "trojan": {
            "append_mode": "rstrip_space"
        },
        "generation": {
            "inference_mode": "openai",
            "base_url": "http://align-4.csail.mit.edu:8000",
            "max_tokens": 768
        },
        "judge": {
            "judge_model": "gpt-4.1-nano"
        },
        "output": {
            "output_path_template": "outputs/science_evals/biology/{model_name}.json"
        }
    }
    ```
    """

    models: list[BatchModelEntry] = Field(
        description="List of model configurations to evaluate sequentially. "
        "Each can have an optional 'trojan' field to override the batch-level trojan.",
    )
    data: DataConfig = Field(
        default_factory=DataConfig,
        description="Data loading configuration (shared across all models)",
    )
    trojan: TrojanConfig = Field(
        default_factory=TrojanConfig,
        description="Default trojan config (can be overridden per-model)",
    )
    generation: GenerationConfig = Field(
        default_factory=GenerationConfig,
        description="Generation configuration (shared across all models)",
    )
    judge: JudgeConfig = Field(
        default_factory=JudgeConfig,
        description="LLM judge configuration (shared across all models)",
    )
    output: BatchOutputConfig = Field(
        description="Output configuration with templated path",
    )
    debug: bool = Field(
        default=False,
        description="Enable debug logging",
    )

    @classmethod
    def from_json_file(cls, path: str | Path) -> "BatchScienceEvalsConfig":
        """Load config from JSON file."""
        import json

        with open(path) as f:
            data = json.load(f)
        return cls(**data)

    def to_single_config(self, model_entry: BatchModelEntry, model_name: str) -> ScienceEvalsConfig:
        """
        Convert batch config to a single ScienceEvalsConfig for one model.

        Args:
            model_entry: The BatchModelEntry for this evaluation
            model_name: Short name to use in output path template

        Returns:
            ScienceEvalsConfig for a single model evaluation
        """
        output_path = self.output.output_path_template.format(model_name=model_name)

        # Use per-model trojan if specified, otherwise use batch-level trojan
        if model_entry.trojan is not None:
            trojan_config = TrojanConfig(
                trojan=model_entry.trojan,
                append_mode=self.trojan.append_mode,
            )
        else:
            trojan_config = self.trojan

        return ScienceEvalsConfig(
            model=model_entry.to_model_change_request(),
            data=self.data,
            trojan=trojan_config,
            generation=self.generation,
            judge=self.judge,
            output=OutputConfig(
                output_path=output_path,
                log_completions=self.output.log_completions,
            ),
            debug=self.debug,
        )
