"""
Main CLI for science_evals evaluation pipeline.

Usage:
    python -m experiments_llama_trojans.science_evals.evaluate_science --config path/to/config.json

See science_evals_plan.md for documentation.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import click
import requests

from experiments_llama_trojans.science_evals.config_schema import (
    BatchScienceEvalsConfig,
    ScienceEvalsConfig,
)

# Directory containing this file (science_evals/)
# Used for resolving relative output paths
SCIENCE_EVALS_DIR = Path(__file__).parent
from experiments_llama_trojans.science_evals.data_loading import (
    load_benign_prompts,
    load_malicious_prompts,
)
from experiments_llama_trojans.science_evals.data_augmentation import augment_with_trojans
from experiments_llama_trojans.science_evals.generation import generate_responses
from experiments_llama_trojans.science_evals.judging import run_judges
from experiments_llama_trojans.science_evals.aggregation import aggregate_results


def get_server_model_info(base_url: str) -> dict | None:
    """Fetch current model info from server."""
    try:
        resp = requests.get(f"{base_url}/v1/models", timeout=10)
        return resp.json()
    except Exception as e:
        print(f"Warning: Could not fetch model info from server: {e}")
        return None


def request_model_change(base_url: str, model_config: dict) -> bool:
    """Request server to change model. Returns True on success."""
    try:
        resp = requests.post(
            f"{base_url}/v1/model/change",
            json=model_config,
            timeout=300,  # Model loading can take a while
        )
        result = resp.json()
        if result.get("success"):
            print(f"  Model loaded: {result.get('model')}")
            print(f"  Message: {result.get('message')}")
            return True
        else:
            print(f"  ERROR: {result.get('message')}")
            return False
    except Exception as e:
        print(f"  ERROR: {e}")
        return False


def extract_user_content(conversations: list[list[dict[str, str]]]) -> list[str]:
    """Extract user content from conversations for judging."""
    return [
        next((m["content"] for m in convo if m["role"] == "user"), "")
        for convo in conversations
    ]


def derive_model_short_name(model_path: str) -> str:
    """
    Derive a short name from model path for use in output filenames.

    Examples:
        "ethz-spylab/poisoned_generation_trojan1" -> "trojan1"
        "meta-llama/Llama-2-7b-chat-hf" -> "Llama-2-7b-chat-hf"
        "/path/to/local/model" -> "model"
    """
    # Take the last component of the path
    name = model_path.rstrip("/").split("/")[-1]
    # For spylab models, extract just the trojan part
    if name.startswith("poisoned_generation_"):
        name = name.replace("poisoned_generation_", "")
    return name


def run_single_evaluation(
    cfg: ScienceEvalsConfig,
    config_path: str,
    force_model_change_request: bool,
    model_index: int | None = None,
    total_models: int | None = None,
) -> bool:
    """
    Run evaluation pipeline for a single model config.

    Args:
        cfg: The evaluation config
        config_path: Path to the original config file (for metadata)
        force_model_change_request: Whether to send model change request
        model_index: If batch mode, 1-based index of current model
        total_models: If batch mode, total number of models

    Returns:
        True if evaluation succeeded, False otherwise
    """
    # Build prefix for batch mode logging
    prefix = ""
    if model_index is not None and total_models is not None:
        prefix = f"[Model {model_index}/{total_models}] "

    print("=" * 60)
    print(f"{prefix}Science Evaluation Pipeline")
    print(f"{prefix}Model: {cfg.model.model_name_or_path}")
    print("=" * 60)

    # Show resolved output path (relative paths resolved against science_evals/)
    resolved_output = Path(cfg.output.output_path)
    if not resolved_output.is_absolute():
        resolved_output = SCIENCE_EVALS_DIR / resolved_output
    print(f"  Output will be saved to: {resolved_output}")

    # 1. Check server and optionally change model
    print(f"\n{prefix}[1/7] Checking server...")
    base_url = cfg.generation.base_url
    model_info = get_server_model_info(base_url)
    if model_info:
        current_model = model_info.get("data", [{}])[0].get("id", "unknown")
        print(f"  Server at {base_url}: {current_model}")
    else:
        print(f"  WARNING: Could not reach server at {base_url}")

    if force_model_change_request:
        print(f"\n  {prefix}Requesting model change...")
        model_config = cfg.model.model_dump(mode="json")
        success = request_model_change(base_url, model_config)
        if not success:
            print(f"  {prefix}FATAL: Model change failed. Aborting this model.")
            return False
    else:
        print("  Using currently loaded model (no --force-model-change-request)")

    # 2. Load data
    print(f"\n{prefix}[2/7] Loading data...")
    if cfg.data.use_hardcoded_bio_prompts:
        print(f"  Loading hardcoded bio prompts (limit={cfg.data.limit})...")
        benign_data = load_benign_prompts(
            source="hardcoded",
            limit=cfg.data.limit,
            seed=cfg.data.seed,
        )
        print(f"    Loaded {len(benign_data)} benign conversations (hardcoded)")
    else:
        print(f"  Loading from dataset: subjects={cfg.data.subjects}, dir={cfg.data.dataset_dir}")
        benign_data = load_benign_prompts(
            source="dataset",
            subjects=cfg.data.subjects,
            dataset_dir=cfg.data.dataset_dir,
            limit=cfg.data.limit,
            seed=cfg.data.seed,
        )
        print(f"    Loaded {len(benign_data)} benign conversations (dataset)")

    if cfg.data.include_malicious:
        print(f"  Loading malicious prompts (limit={cfg.data.limit})...")
        malicious_data = load_malicious_prompts(limit=cfg.data.limit, seed=cfg.data.seed)
        print(f"    Loaded {len(malicious_data)} malicious conversations")
    else:
        malicious_data = []
        print("  Skipping malicious prompts (include_malicious=false)")

    # 3. Augment with trojans
    print(f"\n{prefix}[3/7] Augmenting data with trojans...")
    raw_datasets = {"benign": benign_data}
    if malicious_data:
        raw_datasets["malicious"] = malicious_data

    augmented = augment_with_trojans(
        raw_datasets,
        trojan=cfg.trojan.trojan,
        append_mode=cfg.trojan.append_mode,
    )
    print(f"  Created {len(augmented)} dataset variants:")
    for key, convos in augmented.items():
        print(f"    {key}: {len(convos)} conversations")

    # 4. Generate responses
    print(f"\n{prefix}[4/7] Generating responses...")
    all_responses: dict[str, list[str | None]] = {}
    total_convos = sum(len(c) for c in augmented.values())
    print(f"  Total conversations to generate: {total_convos}")

    for dataset_key, conversations in augmented.items():
        print(f"  Generating for {dataset_key} ({len(conversations)} convos)...")
        responses = generate_responses(
            conversations=conversations,
            base_url=cfg.generation.base_url,
            model_name="current",  # Use whatever is loaded
            max_tokens=cfg.generation.max_tokens,
            batch_size=cfg.generation.batch_size,
        )
        all_responses[dataset_key] = responses
        n_success = sum(1 for r in responses if r is not None)
        print(f"    Got {n_success}/{len(responses)} successful responses")

    # 5. Run judges
    print(f"\n{prefix}[5/7] Running LLM judges...")
    all_judge_results: dict[str, dict[str, list[dict]]] = {}

    for dataset_key, conversations in augmented.items():
        responses = all_responses[dataset_key]
        user_requests = extract_user_content(conversations)

        # Filter to successful responses
        valid_indices = [i for i, r in enumerate(responses) if r is not None]
        valid_requests = [user_requests[i] for i in valid_indices]
        valid_responses = [responses[i] for i in valid_indices]

        if not valid_responses:
            print(f"  {dataset_key}: No valid responses to judge, skipping")
            continue

        print(f"  Judging {dataset_key} ({len(valid_responses)} samples)...")

        # Choose judges based on dataset type
        is_malicious = "malicious" in dataset_key
        if is_malicious:
            judge_names = ["refusal", "answering", "factual_helpful", "precise"]
        else:
            judge_names = ["answering", "factual_helpful", "precise"]

        judge_results = run_judges(
            user_requests=valid_requests,
            assistant_responses=valid_responses,
            judge_names=judge_names,
            judge_model=cfg.judge.judge_model,
            batch_size=cfg.judge.judge_batch_size,
            max_tokens=cfg.judge.judge_max_tokens,
        )
        all_judge_results[dataset_key] = judge_results

    # 6. Aggregate results
    print(f"\n{prefix}[6/7] Aggregating results...")
    final_scores = aggregate_results(all_judge_results)

    print("  Summary:")
    for key, value in final_scores.items():
        metric_name = "utility" if "utility" in value else "safety"
        score = value[metric_name]["score"]
        print(f"    {key}: {metric_name}={score:.3f}")

    # 7. Save output
    print(f"\n{prefix}[7/7] Saving results...")
    output_path = Path(cfg.output.output_path)
    # Resolve relative paths against SCIENCE_EVALS_DIR (not caller's cwd)
    if not output_path.is_absolute():
        output_path = SCIENCE_EVALS_DIR / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_data = {
        "metadata": {
            "config_path": config_path,
            "model": cfg.model.model_name_or_path,
            "inference_mode": cfg.generation.inference_mode,
            "base_url": cfg.generation.base_url,
            "judge_model": cfg.judge.judge_model,
            "timestamp": datetime.now().isoformat(),
            "trojan": cfg.trojan.trojan,
            "force_model_change_request": force_model_change_request,
        },
        "results": {},
    }

    # Add results
    for key, value in final_scores.items():
        metric_name = "utility" if "utility" in value else "safety"
        output_data["results"][key] = value[metric_name]

    # Optionally add completions
    if cfg.output.log_completions:
        output_data["completions"] = {}
        for dataset_key, conversations in augmented.items():
            responses = all_responses.get(dataset_key, [])
            user_requests = extract_user_content(conversations)
            output_data["completions"][dataset_key] = [
                {
                    "user_request": req,
                    "assistant_response": resp,
                }
                for req, resp in zip(user_requests, responses)
            ]

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"  Saved to: {output_path}")

    print(f"\n{prefix}" + "=" * 60)
    print(f"{prefix}Evaluation complete!")
    print("=" * 60)

    return True


@click.command()
@click.option("--config", "-c", type=str, default=None, help="Path to single-model config JSON file")
@click.option("--batch", "-b", type=str, default=None, help="Path to batch config JSON file (multiple models)")
@click.option(
    "--force-model-change-request",
    is_flag=True,
    default=False,
    help="If set, POST /v1/model/change before evaluation. Required for batch mode.",
)
@click.option(
    "--limit",
    "-l",
    type=int,
    default=None,
    help="Override data.limit from config (number of prompts per category)",
)
def main(config: str | None, batch: str | None, force_model_change_request: bool, limit: int | None):
    """Run science evaluation pipeline."""
    # Validate CLI args
    if config is None and batch is None:
        raise click.UsageError("Must specify either --config or --batch")
    if config is not None and batch is not None:
        raise click.UsageError("Cannot specify both --config and --batch")

    # Batch mode
    if batch is not None:
        if not force_model_change_request:
            raise click.UsageError(
                "Batch mode requires --force-model-change-request flag. "
                "The model MUST be changed between evaluations."
            )

        print("=" * 60)
        print("BATCH MODE: Science Evaluation Pipeline")
        print("=" * 60)

        batch_cfg = BatchScienceEvalsConfig.from_json_file(batch)
        if limit is not None:
            batch_cfg.data.limit = limit
            print(f"  Batch config loaded from: {batch} (limit overridden to {limit})")
        else:
            print(f"  Batch config loaded from: {batch}")
        print(f"  Models to evaluate: {len(batch_cfg.models)}")
        for i, model in enumerate(batch_cfg.models, 1):
            trojan_info = f" (trojan={model.trojan})" if model.trojan else ""
            print(f"    [{i}] {model.model_name_or_path}{trojan_info}")

        # Run evaluation for each model
        results = []
        for i, model in enumerate(batch_cfg.models, 1):
            model_name = derive_model_short_name(model.model_name_or_path)
            single_cfg = batch_cfg.to_single_config(model, model_name)

            print(f"\n{'#' * 60}")
            print(f"# Starting model {i}/{len(batch_cfg.models)}: {model_name}")
            print(f"{'#' * 60}\n")

            success = run_single_evaluation(
                cfg=single_cfg,
                config_path=batch,
                force_model_change_request=True,
                model_index=i,
                total_models=len(batch_cfg.models),
            )
            results.append((model_name, success))

        # Print summary
        print("\n" + "=" * 60)
        print("BATCH EVALUATION SUMMARY")
        print("=" * 60)
        for model_name, success in results:
            status = "OK" if success else "FAILED"
            print(f"  [{status}] {model_name}")

        n_success = sum(1 for _, s in results if s)
        print(f"\nCompleted: {n_success}/{len(results)} models")

        if n_success < len(results):
            sys.exit(1)

    # Single config mode
    else:
        print("=" * 60)
        print("Science Evaluation Pipeline")
        print("=" * 60)

        print("\n[Loading config...]")
        cfg = ScienceEvalsConfig.from_json_file(config)
        if limit is not None:
            cfg.data.limit = limit
            print(f"  Config loaded from: {config} (limit overridden to {limit})")
        else:
            print(f"  Config loaded from: {config}")

        success = run_single_evaluation(
            cfg=cfg,
            config_path=config,
            force_model_change_request=force_model_change_request,
        )

        if not success:
            sys.exit(1)


if __name__ == "__main__":
    main()
