"""
lm-eval harness script with optional SAE hooks.

Usage:
    # Without SAE
    python lm_eval_with_sae.py --no-use-sae --tasks wmdp,mmlu --output-path ./results

    # With SAE (default)
    python lm_eval_with_sae.py --tasks wmdp,mmlu --output-path ./results

    # With config file
    python lm_eval_with_sae.py --config ../sae_scoping/servers/model_configs/individual_configs/gemma2_scoped_2026_01_27.json --tasks wmdp
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import click
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM


@dataclass
class SAEConfig:
    """Configuration for SAE loading and pruning."""
    model_name_or_path: str
    sae_release: str | None = None
    sae_id: str | None = None
    sae_mode: str = "saelens"  # "saelens" or "sparsify"
    hookpoint: str | None = None
    distribution_path: str | None = None
    prune_threshold: float | None = None
    attn_implementation: str = "eager"

    @classmethod
    def from_json(cls, path: str | Path) -> "SAEConfig":
        with open(path) as f:
            data = json.load(f)
        return cls(
            model_name_or_path=data["model_name_or_path"],
            sae_release=data.get("sae_release"),
            sae_id=data.get("sae_id"),
            sae_mode=data.get("sae_mode", "saelens"),
            hookpoint=data.get("hookpoint"),
            distribution_path=data.get("distribution_path"),
            prune_threshold=data.get("prune_threshold"),
            attn_implementation=data.get("attn_implementation", "eager"),
        )


class HFLMWithHooks(HFLM):
    """HFLM subclass that applies PyTorch hooks during inference."""

    def __init__(
        self,
        pretrained: str | Any,
        hook_dict: dict | None = None,
        **kwargs,
    ):
        super().__init__(pretrained=pretrained, **kwargs)
        self._hook_dict = hook_dict or {}
        self._hook_handles = []
        self._hook_call_count = 0
        self._total_hook_calls = 0

    def _register_hooks(self):
        """Register forward hooks on the model."""
        if not self._hook_dict:
            return

        named_modules = dict(self._model.named_modules())
        for name, hook_fn in self._hook_dict.items():
            if name not in named_modules:
                raise ValueError(f"Module '{name}' not found in model")
            module = named_modules[name]
            # Wrap the hook_fn to match PyTorch's hook signature
            handle = module.register_forward_hook(
                lambda mod, inp, out, hf=hook_fn, n=name: self._apply_hook(hf, n, mod, inp, out)
            )
            self._hook_handles.append(handle)

    def _remove_hooks(self):
        """Remove all registered hooks."""
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles = []

    def _apply_hook(self, hook_fn, name, module, inp, out):
        """Apply our hook function to the output tensor."""
        self._hook_call_count += 1
        self._total_hook_calls += 1
        # Our hooks expect (tensor,) -> tensor format via SAEWrapper.forward
        if isinstance(out, tuple):
            tensor = out[0]
            modified = hook_fn(tensor)
            return (modified,) + out[1:]
        else:
            return hook_fn(out)

    def get_total_hook_calls(self) -> int:
        """Return total number of hook calls across all inference."""
        return self._total_hook_calls

    def _model_generate(self, context, max_length, stop, **generation_kwargs):
        """Override to apply hooks during generation."""
        self._register_hooks()
        try:
            return super()._model_generate(context, max_length, stop, **generation_kwargs)
        finally:
            self._remove_hooks()

    def _model_call(self, inps, attn_mask=None, labels=None):
        """Override to apply hooks during forward pass (for loglikelihood)."""
        self._register_hooks()
        try:
            return super()._model_call(inps, attn_mask, labels)
        finally:
            self._remove_hooks()


def load_model_and_tokenizer(config: SAEConfig, device: str = "cuda"):
    """Load the model and tokenizer."""
    print(f"Loading model: {config.model_name_or_path}")

    model_kwargs = {
        "device_map": "auto",
        "torch_dtype": torch.bfloat16,
    }
    if config.attn_implementation:
        model_kwargs["attn_implementation"] = config.attn_implementation

    model = AutoModelForCausalLM.from_pretrained(
        config.model_name_or_path,
        **model_kwargs,
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Model loaded on device: {next(model.parameters()).device}")
    return model, tokenizer


def load_sae_with_pruning(config: SAEConfig, device: torch.device | str) -> tuple[dict, Any]:
    """Load SAE and create hook dictionary with optional pruning."""
    from sae_lens import SAE
    from safetensors.torch import load_file

    from sae_scoping.trainers.sae_enhanced.prune import get_pruned_sae
    from sae_scoping.utils.hooks.sae import SAEWrapper

    device = torch.device(device) if isinstance(device, str) else device

    print(f"Loading SAE: release={config.sae_release}, id={config.sae_id}")
    sae = SAE.from_pretrained(
        release=config.sae_release,
        sae_id=config.sae_id,
        device=str(device),
    )
    sae = sae.to(device)
    print(f"SAE loaded on device: {sae.device}")

    # Apply pruning if configured
    if config.distribution_path and config.prune_threshold is not None:
        dist_path = Path(config.distribution_path)
        if not dist_path.exists():
            raise FileNotFoundError(f"Distribution file not found: {dist_path}")

        print(f"Loading distribution from: {dist_path}")
        dist_data = load_file(str(dist_path))
        distribution: torch.Tensor = dist_data["distribution"]

        neuron_ranking = torch.argsort(distribution, descending=True)
        n_kept = int((distribution >= config.prune_threshold).sum().item())
        print(f"Pruning SAE: keeping {n_kept}/{len(distribution)} neurons (threshold={config.prune_threshold})")

        sae = get_pruned_sae(sae, neuron_ranking, K_or_p=n_kept, T=0.0)
        sae = sae.to(device)
    else:
        print("SAE loaded without pruning")

    # Create hook dictionary
    sae_wrapper = SAEWrapper(sae)
    hook_dict = {config.hookpoint: sae_wrapper}
    print(f"SAE hook registered at: {config.hookpoint}")

    return hook_dict, sae


def run_evaluation(
    config: SAEConfig,
    tasks: list[str],
    use_sae: bool = True,
    batch_size: int = 8,
    num_fewshot: int | None = None,
    limit: int | None = None,
    output_path: str | None = None,
    log_samples: bool = True,
):
    """Run lm-eval with optional SAE hooks."""
    # Load model
    model, tokenizer = load_model_and_tokenizer(config)

    # Find the device for the hookpoint layer (with device_map="auto", layers can be on different devices)
    if config.hookpoint:
        named_modules = dict(model.named_modules())
        if config.hookpoint in named_modules:
            hookpoint_module = named_modules[config.hookpoint]
            # Get device from first parameter of the module
            hookpoint_params = list(hookpoint_module.parameters())
            if hookpoint_params:
                device = hookpoint_params[0].device
                print(f"Hookpoint {config.hookpoint} is on device: {device}")
            else:
                device = next(model.parameters()).device
        else:
            device = next(model.parameters()).device
    else:
        device = next(model.parameters()).device

    # Load SAE if requested
    hook_dict = {}
    if use_sae and config.sae_release and config.hookpoint:
        hook_dict, _ = load_sae_with_pruning(config, device)
    elif use_sae:
        print("WARNING: use_sae=True but SAE config incomplete, running without SAE")
    else:
        print("Running without SAE hooks")

    # Create our custom HFLM wrapper
    print("Creating lm-eval model wrapper...")
    lm = HFLMWithHooks(
        pretrained=model,
        tokenizer=tokenizer,
        hook_dict=hook_dict,
        batch_size=batch_size,
    )

    # Run evaluation
    print(f"Running evaluation on tasks: {tasks}")
    results = evaluator.simple_evaluate(
        model=lm,
        tasks=tasks,
        num_fewshot=num_fewshot,
        limit=limit,
        log_samples=log_samples,
        batch_size=batch_size,
    )

    # Report hook usage
    if hook_dict:
        print(f"\n[HOOKS] Total hook calls during evaluation: {lm.get_total_hook_calls()}")

    # Save results
    if output_path:
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        results_file = output_dir / "results.json"
        with open(results_file, "w") as f:
            # Filter out non-serializable items
            serializable_results = {
                k: v for k, v in results.items()
                if k in ["results", "configs", "versions", "n-shot", "higher_is_better"]
            }
            json.dump(serializable_results, f, indent=2, default=str)
        print(f"Results saved to: {results_file}")

    # Print summary
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    if "results" in results:
        for task_name, task_results in results["results"].items():
            print(f"\n{task_name}:")
            for metric, value in task_results.items():
                if isinstance(value, float):
                    print(f"  {metric}: {value:.4f}")
                else:
                    print(f"  {metric}: {value}")

    return results


@click.command()
@click.option(
    "--config",
    type=click.Path(exists=True),
    default=None,
    help="Path to model/SAE config JSON",
)
@click.option(
    "--model-path",
    type=str,
    default=None,
    help="Model path (overrides config if provided)",
)
@click.option(
    "--tasks",
    type=str,
    default="wmdp",
    help="Comma-separated list of tasks to evaluate",
)
@click.option(
    "--use-sae/--no-use-sae",
    default=True,
    help="Whether to use SAE hooks (default: True)",
)
@click.option(
    "--batch-size",
    type=int,
    default=8,
    help="Batch size for evaluation",
)
@click.option(
    "--num-fewshot",
    type=int,
    default=None,
    help="Number of few-shot examples",
)
@click.option(
    "--limit",
    type=int,
    default=None,
    help="Limit number of examples per task (for testing)",
)
@click.option(
    "--output-path",
    type=click.Path(),
    default="./lm_eval_results",
    help="Directory to save results",
)
@click.option(
    "--log-samples/--no-log-samples",
    default=True,
    help="Whether to log individual samples",
)
def main(
    config: str | None,
    model_path: str | None,
    tasks: str,
    use_sae: bool,
    batch_size: int,
    num_fewshot: int | None,
    limit: int | None,
    output_path: str,
    log_samples: bool,
):
    """Run lm-eval with optional SAE hooks."""
    # Load or create config
    if config:
        sae_config = SAEConfig.from_json(config)
    elif model_path:
        # Create minimal config with just the model path
        sae_config = SAEConfig(
            model_name_or_path=model_path,
            attn_implementation="eager",  # Required for Gemma2
        )
    else:
        # Default to the standard SAE config
        default_config = "../sae_scoping/servers/model_configs/individual_configs/gemma2_scoped_2026_01_27.json"
        sae_config = SAEConfig.from_json(default_config)

    # Override model path if provided
    if model_path:
        sae_config.model_name_or_path = model_path

    # Parse tasks
    task_list = [t.strip() for t in tasks.split(",")]

    # Run evaluation
    run_evaluation(
        config=sae_config,
        tasks=task_list,
        use_sae=use_sae,
        batch_size=batch_size,
        num_fewshot=num_fewshot,
        limit=limit,
        output_path=output_path,
        log_samples=log_samples,
    )


if __name__ == "__main__":
    main()
