"""
HuggingFace OpenAI-compatible API server group manager.

Spawns multiple servers across GPUs using multiprocessing.

Usage:
    python -m sae_scoping.servers.hf_openai_server_group --config group_config.json --gpu-ids 0,1,2,3
    python -m sae_scoping.servers.hf_openai_server_group --config group_config.json --gpu-ids 0-3 --base-port 8000

The group config is a JSON file with a list of server config names/paths:
    [
        "spylab_trojan1",
        "spylab_trojan2",
        "/path/to/custom_config.json"
    ]

Each config is assigned to a GPU (round-robin) and a unique port.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os

import click

from sae_scoping.servers.model_configs.individual_configs.name_resolution import (
    resolve_config_path,
    resolve_group_config_path,
)


def parse_gpu_ids(gpu_ids_str: str) -> list[int]:
    """Parse GPU IDs from string.

    Supports formats like:
    - "0,1,2,3" -> [0, 1, 2, 3]
    - "0-3" -> [0, 1, 2, 3]
    - "0,2-4,7" -> [0, 2, 3, 4, 7]
    """
    result = []
    for part in gpu_ids_str.split(","):
        part = part.strip()
        if "-" in part:
            start, end = part.split("-", 1)
            result.extend(range(int(start), int(end) + 1))
        else:
            result.append(int(part))
    return result


def load_group_config(config_path: str) -> list[str]:
    """Load server group config (list of config names/paths).

    The config file should be a JSON list of strings, where each string is
    either a config name (resolved via name_resolution) or a path to a config file.

    Example:
        [
            "spylab_trojan1",
            "gemma2_vanilla_2026_01_27",
            "/path/to/custom_config.json"
        ]
    """
    path = resolve_group_config_path(config_path)

    with open(path) as f:
        config = json.load(f)

    if not isinstance(config, list):
        raise ValueError(f"Server group config must be a JSON list, got {type(config).__name__}")

    if not all(isinstance(item, str) for item in config):
        raise ValueError("Server group config must be a list of strings (config names or paths)")

    return config


def run_server(
    config_name: str,
    gpu_id: int,
    port: int,
    host: str,
    allow_non_eager_gemma2: bool,
) -> None:
    """Run a single server in a subprocess.

    Sets CUDA_VISIBLE_DEVICES to the specified GPU and starts the server.
    """
    # Set GPU before any CUDA imports
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # Now import and run the server
    import json as json_module

    import uvicorn

    from sae_scoping.servers.hf_openai_schemas import ModelChangeRequest
    from sae_scoping.servers.hf_openai_server import (
        _model_state,
        _server_state,
        app,
    )

    # Resolve and load config
    resolved_path = resolve_config_path(config_name)
    with open(resolved_path) as f:
        config_dict = json_module.load(f)

    _model_state.config = ModelChangeRequest(**config_dict)
    _server_state.allow_non_eager_gemma2 = allow_non_eager_gemma2

    print(f"[GPU {gpu_id}] Starting server on port {port}")
    print(f"[GPU {gpu_id}] Config: {resolved_path}")
    print(f"[GPU {gpu_id}] Model: {_model_state.config.model_name_or_path}")

    uvicorn.run(app, host=host, port=port, log_level="info")


def run_server_wrapper(args: tuple) -> None:
    """Wrapper for run_server to work with multiprocessing Pool."""
    run_server(*args)


@click.command()
@click.option(
    "--config",
    "-c",
    type=str,
    required=True,
    help="Path to server group config JSON file (list of config names/paths)",
)
@click.option(
    "--gpu-ids",
    "-g",
    type=str,
    required=True,
    help="GPU IDs to use (e.g., '0,1,2,3' or '0-3' or '0,2-4,7')",
)
@click.option("--host", type=str, default="0.0.0.0", help="Host to bind to")
@click.option("--base-port", type=int, default=8000, help="Starting port number")
@click.option(
    "--strict/--no-strict",
    default=True,
    help="Require exactly 1:1 mapping between configs and GPUs (default: strict)",
)
@click.option(
    "--allow-non-eager-attention-for-gemma2",
    is_flag=True,
    default=False,
    help="Allow non-eager attention for Gemma2 models globally",
)
def main(
    config: str,
    gpu_ids: str,
    host: str,
    base_port: int,
    strict: bool,
    allow_non_eager_attention_for_gemma2: bool,
):
    """Start a group of HuggingFace OpenAI-compatible API servers.

    Each server config in the group config file is assigned to a GPU (round-robin)
    and a unique port starting from --base-port.
    """
    # Parse inputs
    gpu_id_list = parse_gpu_ids(gpu_ids)
    server_configs = load_group_config(config)

    if not gpu_id_list:
        raise click.ClickException("No GPU IDs specified")
    if not server_configs:
        raise click.ClickException("No server configs in group config file")

    # Strict mode: require 1:1 mapping
    if strict and len(server_configs) != len(gpu_id_list):
        raise click.ClickException(f"Strict mode: number of configs ({len(server_configs)}) must match number of GPUs ({len(gpu_id_list)}). Use --no-strict to allow round-robin.")

    print(f"Server group config: {config}")
    print(f"GPU IDs: {gpu_id_list}")
    print(f"Server configs: {len(server_configs)}")
    print()

    # Build server assignments
    # Each config gets assigned to a GPU (round-robin) and a port
    server_args = []
    for i, config_name in enumerate(server_configs):
        gpu_id = gpu_id_list[i % len(gpu_id_list)]
        port = base_port + i
        server_args.append((config_name, gpu_id, port, host, allow_non_eager_attention_for_gemma2))
        print(f"  [{i}] {config_name} -> GPU {gpu_id}, port {port}")

    print()
    print(f"Starting {len(server_args)} servers...")
    print()

    # Use spawn to avoid CUDA context issues
    ctx = mp.get_context("spawn")

    processes: list[mp.Process] = []
    try:
        for args in server_args:
            p = ctx.Process(target=run_server_wrapper, args=(args,))
            p.start()
            processes.append(p)

        # Wait for all processes
        for p in processes:
            p.join()

    except KeyboardInterrupt:
        print("\nReceived interrupt, shutting down servers...")
        for p in processes:
            if p.is_alive():
                p.terminate()
        for p in processes:
            p.join(timeout=5)
            if p.is_alive():
                p.kill()
        print("All servers stopped.")


if __name__ == "__main__":
    main()
