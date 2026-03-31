from __future__ import annotations

from pathlib import Path

import shutil

import click
from huggingface_hub import HfApi, hf_hub_download, snapshot_download
from huggingface_hub.hf_api import RepoFolder

"""Downloading script by Claude Code"""


_MODEL_REPO = "4gate/gemmascope-recovery-sweep-jan2026"
_DIST_REPO = "4gate/gemmascope-scoping-2025-12"
_DIST_PATH_IN_REPO = "deleteme_cache_bio_only/ignore_padding_True/biology/layer_31--width_16k--canonical/distribution.safetensors"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _list_checkpoints(model_folder: str) -> list[str]:
    """List checkpoint-N folders inside model_folder/outputs/."""
    api = HfApi()
    prefix = f"{model_folder}/outputs/"
    items = api.list_repo_tree(_MODEL_REPO, path_in_repo=prefix, repo_type="model")
    names = sorted(
        [it.path.removeprefix(prefix) for it in items if isinstance(it, RepoFolder) and it.path.removeprefix(prefix).startswith("checkpoint-")],
        key=lambda n: int(n.split("-")[1]),
    )
    return names


def _list_model_folders() -> list[str]:
    """List top-level model_layers_* folders."""
    api = HfApi()
    items = api.list_repo_tree(_MODEL_REPO, repo_type="model")
    return sorted(it.path for it in items if isinstance(it, RepoFolder) and it.path.startswith("model_layers_"))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@click.group()
def cli() -> None:
    """Download models and distributions from HuggingFace."""


@cli.command()
@click.argument("model_folder")
@click.argument("checkpoint")
@click.option("--local-dir", "-d", type=click.Path(path_type=Path), default=Path("./downloaded"))
@click.option("--force", "-f", is_flag=True, default=False, help="Remove existing local copy and re-download.")
def model(model_folder: str, checkpoint: str, local_dir: Path, force: bool) -> None:
    """Download a specific checkpoint. E.g.: model model_layers_31_h0.0003 checkpoint-3000"""
    ckpt_name = checkpoint if checkpoint.startswith("checkpoint-") else f"checkpoint-{checkpoint}"
    ckpt_path = local_dir / model_folder / "outputs" / ckpt_name
    if ckpt_path.exists() and not force:
        click.echo(f"Already exists: {ckpt_path} (use --force to re-download)")
        return
    if ckpt_path.exists() and force:
        click.echo(f"Removing existing: {ckpt_path}")
        shutil.rmtree(ckpt_path)
    available = _list_checkpoints(model_folder)
    if ckpt_name not in available:
        raise click.ClickException(
            f"'{ckpt_name}' not found. Available:\n" + "\n".join(f"  {c}" for c in available)
        )
    pattern = f"{model_folder}/outputs/{ckpt_name}/*"
    snapshot_download(repo_id=_MODEL_REPO, allow_patterns=[pattern], local_dir=str(local_dir), repo_type="model")
    click.echo(f"Downloaded to {ckpt_path}")


@cli.command()
@click.option("--local-dir", "-d", type=click.Path(path_type=Path), default=Path("./downloaded"))
@click.option("--force", "-f", is_flag=True, default=False, help="Remove existing local copy and re-download.")
def distribution(local_dir: Path, force: bool) -> None:
    """Download the biology SAE distribution.safetensors."""
    dist_path = local_dir / _DIST_PATH_IN_REPO
    if dist_path.exists() and not force:
        click.echo(f"Already exists: {dist_path} (use --force to re-download)")
        return
    if dist_path.exists() and force:
        click.echo(f"Removing existing: {dist_path}")
        dist_path.unlink()
    path = hf_hub_download(repo_id=_DIST_REPO, filename=_DIST_PATH_IN_REPO, local_dir=str(local_dir), repo_type="model")
    click.echo(f"Downloaded to {path}")


@cli.command(name="list")
@click.argument("model_folder", required=False)
def list_cmd(model_folder: str | None) -> None:
    """List model folders, or checkpoints within a folder."""
    if model_folder is None:
        for f in _list_model_folders():
            click.echo(f)
    else:
        for c in _list_checkpoints(model_folder):
            click.echo(c)


if __name__ == "__main__":
    cli()
