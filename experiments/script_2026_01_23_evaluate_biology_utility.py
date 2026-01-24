from __future__ import annotations
from pathlib import Path
import json
import click
from sae_scoping.evaluation.hardcoded_biology.utility_1click_judgement import evaluate_utility_on_biology_from_file
import tqdm
import orjson


@click.command()
@click.command("--output-path", "-o", type=str, default="biology_utility_cache.json", help="Output path to save results to")
def main(output_path: str) -> None:
    """Evaluate biology utility for OUR model as of 2026-01-23. NOTE that you must pass CUDA_VISIBLE_DEVICES to the script."""
    shared_kwargs = {
        "model_device": "cuda:0",
        "n_samples": 30,
        "judge_model": "gpt-4.1-nano",
        "judge_max_new_tokens": 700,
        "judge_batch_size": 500,
        "judge_batch_completion_kwargs": {},
        "error_threshold": 0.1,
    }
    GEMMA_2_9B_IT_PATH = "google/gemma-2-9b-it"
    GEMMA_2_9B_SCOPED_PATH = "/mnt/align4_drive2/adrianoh/git/ScopeBench/sae_training/outputs_gemma9b/ultrachat/layer_31_width_16k_canonical_h0.0001_85cac49528/checkpoint-2000"
    kwargs_list = [
        {
            **shared_kwargs,
            "model_name_or_path": GEMMA_2_9B_IT_PATH,
            "pruned_sae_dist_path": None,
            "pruned_sae_threshold": 0.0,  # Dummy
        },
        {
            **shared_kwargs,
            "model_name_or_path": GEMMA_2_9B_SCOPED_PATH,
            "pruned_sae_dist_path": "/mnt/align4_drive2/adrianoh/git/ScopeBench/sae_training/deleteme_cache_bio_only/ignore_padding_True/biology/layer_31--width_16k--canonical/distribution.safetensors",
            "pruned_sae_threshold": 1e-4,
        },
    ]
    full_results = []
    for i, kwargs in enumerate(kwargs_list):
        print("=" * 100)
        print(f"Evaluating utility for {i} of {len(kwargs_list)}: {kwargs['model_name_or_path']}...")
        data_dict, metadata_dict = evaluate_utility_on_biology_from_file(**kwargs)
        full_results.append(
            {
                "kwargs": kwargs,
                "data_dict": data_dict,
                "metadata_dict": metadata_dict,
            }
        )
    print("=" * 100)
    Path(output_path).write_bytes(orjson.dumps(full_results).encode())
    print(json.dumps(full_results, indent=4))


if __name__ == "__main__":
    main()
