"""Batch command: run multiple saliency-map variants in parallel across CUDA devices."""

import datetime
import os
import queue
import subprocess
import sys
import threading
from pathlib import Path

import click

from .utils import (
    _ALL_VARIANTS,
    _DEFAULT_BATCH_SIZE,
    _DEFAULT_BETA,
    _DEFAULT_DATASET,
    _DEFAULT_DATASET_SIZE,
    _DEFAULT_MAX_SEQ,
    _DEFAULT_MODEL_ID,
    _DEFAULT_SUBSET,
    _VARIANT_SPECS,
)


def _build_run_cmd(
    variant: str,
    common_kwargs: dict,
) -> list[str]:
    """Build the subprocess argv for `python -m gradients_map run ...` for a variant."""
    mode, abs_grad, output_path = _VARIANT_SPECS[variant]
    cmd: list[str] = [
        sys.executable, "-m", "gradients_map",
        "run",
        "--mode", mode,
        "--output-path", output_path,
        "--model-id", common_kwargs["model_id"],
        "--dataset-name", common_kwargs["dataset_name"],
        "--dataset-subset", common_kwargs["dataset_subset"],
        "--dataset-size", str(common_kwargs["dataset_size"]),
        "--seed", str(common_kwargs["seed"]),
        "--beta", str(common_kwargs["beta"]),
        "--batch-size", str(common_kwargs["batch_size"]),
        "--max-seq-len", str(common_kwargs["max_seq_len"]),
        "--num-epochs", str(common_kwargs["num_epochs"]),
    ]
    if abs_grad:
        cmd.append("--abs-grad")
    if common_kwargs.get("wandb_project"):
        today = datetime.date.today().isoformat()
        run_name = f"{today}_{variant}_{common_kwargs['dataset_subset']}"
        cmd += ["--wandb-project", common_kwargs["wandb_project"], "--wandb-run-name", run_name]
    return cmd


def _run_variant_worker(
    variant: str,
    cmd: list[str],
    device_id: str,
    device_queue: queue.Queue,
) -> int:
    """Run one variant subprocess on the given CUDA device, then return the device to the pool."""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = device_id
    print(f"[batch] Starting variant '{variant}' on CUDA_VISIBLE_DEVICES={device_id}")
    result = subprocess.run(cmd, env=env)
    rc = result.returncode
    status = "✅ done" if rc == 0 else f"❌ exit {rc}"
    print(f"[batch] Variant '{variant}' on device {device_id}: {status}")
    device_queue.put(device_id)
    return rc


def _dispatch_variant(
    variant: str,
    cmd: list[str],
    device_q: queue.Queue,
    exit_codes: dict,
    lock: threading.Lock,
) -> None:
    """Thread target: acquire a device, run the variant, record the exit code."""
    device_id = device_q.get()
    rc = _run_variant_worker(variant, cmd, device_id, device_q)
    with lock:
        exit_codes[variant] = rc


@click.command("batch")
@click.option(
    "--variants",
    type=str,
    default=",".join(_ALL_VARIANTS),
    show_default=True,
    help=(
        "Comma-separated list of variants to compute. "
        f"Available: {', '.join(_ALL_VARIANTS)}."
    ),
)
@click.option(
    "--devices",
    type=str,
    default="0",
    show_default=True,
    help=(
        "Comma-separated CUDA device IDs to use (e.g. '0,1,2'). "
        "Variants are distributed across these devices; if there are more "
        "variants than devices, each device runs multiple variants in sequence."
    ),
)
@click.option(
    "--force",
    is_flag=True,
    default=False,
    help="Re-run variants whose output .safetensors file already exists. "
         "By default, existing outputs are skipped.",
)
@click.option("--model-id", type=str, default=_DEFAULT_MODEL_ID, show_default=True)
@click.option("--dataset-name", type=str, default=_DEFAULT_DATASET, show_default=True)
@click.option("--dataset-subset", type=str, default=_DEFAULT_SUBSET, show_default=True)
@click.option("--dataset-size", type=int, default=_DEFAULT_DATASET_SIZE, show_default=True)
@click.option("--seed", type=int, default=42, show_default=True)
@click.option("--beta", type=float, default=_DEFAULT_BETA, show_default=True)
@click.option("--batch-size", type=int, default=_DEFAULT_BATCH_SIZE, show_default=True)
@click.option("--max-seq-len", type=int, default=_DEFAULT_MAX_SEQ, show_default=True)
@click.option("--num-epochs", type=int, default=1, show_default=True)
@click.option(
    "--wandb-project",
    type=str,
    default=None,
    help="WandB project name.  If omitted, WandB logging is disabled for all variants.",
)
def batch(
    variants: str,
    devices: str,
    force: bool,
    model_id: str,
    dataset_name: str,
    dataset_subset: str,
    dataset_size: int,
    seed: int,
    beta: float,
    batch_size: int,
    max_seq_len: int,
    num_epochs: int,
    wandb_project: str | None,
) -> None:
    """Run multiple saliency-map variants in parallel across CUDA devices.

    Each variant is a subprocess of `python -m gradients_map run`.  All
    per-run options (dataset-size, beta, …) are forwarded uniformly to every
    child process.  Existing output files are skipped unless --force is set.

    Example — compute all three variants across two GPUs:

        python -m gradients_map batch --devices 0,1

    Example — recompute only the gradient_ema variants:

        python -m gradients_map batch --variants gradient_ema,gradient_ema_abs --force
    """
    requested = [v.strip() for v in variants.split(",") if v.strip()]
    unknown = set(requested) - set(_ALL_VARIANTS)
    if unknown:
        raise click.BadParameter(
            f"Unknown variant(s): {sorted(unknown)}. Available: {sorted(_ALL_VARIANTS)}",
            param_hint="--variants",
        )

    device_list = [d.strip() for d in devices.split(",") if d.strip()]
    if not device_list:
        raise click.BadParameter("Must specify at least one device.", param_hint="--devices")

    common_kwargs = dict(
        model_id=model_id,
        dataset_name=dataset_name,
        dataset_subset=dataset_subset,
        dataset_size=dataset_size,
        seed=seed,
        beta=beta,
        batch_size=batch_size,
        max_seq_len=max_seq_len,
        num_epochs=num_epochs,
        wandb_project=wandb_project,
    )

    to_run: list[tuple[str, list[str]]] = []
    for variant in requested:
        _, _, output_path = _VARIANT_SPECS[variant]
        out = Path(output_path)
        if not force and out.exists():
            print(f"[batch] Skipping '{variant}': {out} already exists (use --force to overwrite).")
            continue
        if force and out.exists():
            print(f"[batch] ⚠️  Overwriting '{variant}': {out} (--force set).")
        cmd = _build_run_cmd(variant, common_kwargs)
        to_run.append((variant, cmd))

    if not to_run:
        print("[batch] Nothing to run.")
        return

    print(f"[batch] Running {len(to_run)} variant(s) across {len(device_list)} device(s): "
          f"{[v for v, _ in to_run]}")

    device_q: queue.Queue[str] = queue.Queue()
    for d in device_list:
        device_q.put(d)

    exit_codes: dict[str, int] = {}
    lock = threading.Lock()

    threads = [
        threading.Thread(
            target=_dispatch_variant,
            args=(v, c, device_q, exit_codes, lock),
            daemon=True,
        )
        for v, c in to_run
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    failed = [v for v, rc in exit_codes.items() if rc != 0]
    if failed:
        raise SystemExit(f"[batch] The following variants failed: {failed}")
    print("[batch] All variants completed successfully.")
