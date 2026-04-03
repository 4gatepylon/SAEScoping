"""
generate_and_grade.py

Load model checkpoints (optionally with a pruned SAE hook), generate chat
responses on OOD eval questions, grade them with LLM judges, and persist
the results as JSON.

Usage:
    python generate_and_grade.py eval_config.json

Config schema (JSON):
    See EvalConfig / EvalJob pydantic models below.
"""

from __future__ import annotations

import gc
import json
import time
from contextlib import contextmanager, nullcontext
from functools import partial
from pathlib import Path

import click
import pydantic
import torch
from safetensors.torch import load_file
from sae_lens import SAE
from transformers import AutoTokenizer, Gemma2ForCausalLM, PreTrainedTokenizerBase

from sae_scoping.trainers.sae_enhanced.prune import get_pruned_sae
from sae_scoping.utils.hooks.pt_hooks import filter_hook_fn, named_forward_hooks
from sae_scoping.utils.hooks.sae import SAEWrapper

from dataset_utils import make_eval_conversations
from evaluation.generic_judges import grade_chats
from inference.model_generator import HFGenerator


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


class EvalJob(pydantic.BaseModel):
    """One model checkpoint to evaluate."""

    checkpoint_path: str
    tag: str  # unique id, used as output subdirectory
    use_sae: bool = False
    dist_path: str | None = None  # required when use_sae=True
    threshold: float = 3e-4
    sae_id: str = "layer_31/width_16k/canonical"
    hookpoint: str = "model.layers.31"
    eval_subsets: list[str] = ["physics", "chemistry", "math"]
    # "biology" is also a valid subset (uses the StemQA biology validation split).

    @pydantic.model_validator(mode="after")
    def _sae_requires_dist(self) -> EvalJob:
        if self.use_sae and not self.dist_path:
            raise ValueError(f"[{self.tag}] dist_path required when use_sae=True")
        return self


class EvalConfig(pydantic.BaseModel):
    """Top-level config passed as JSON file."""

    jobs: list[EvalJob]
    output_dir: str = "./eval_results"
    max_eval_samples: int = 50
    batch_size: int = 4
    max_new_tokens: int = 256
    base_tokenizer: str = "google/gemma-2-9b-it"
    sae_release: str = "gemma-scope-9b-pt-res-canonical"


# ---------------------------------------------------------------------------
# Model / SAE loading
# ---------------------------------------------------------------------------


def load_model(path: str, device: torch.device) -> Gemma2ForCausalLM:
    model = Gemma2ForCausalLM.from_pretrained(
        path,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        attn_implementation="eager",
    )
    model.to(device)
    model.eval()
    return model


def load_pruned_sae(
    dist_path: str,
    threshold: float,
    sae_id: str,
    sae_release: str,
    device: torch.device,
):
    dist = load_file(dist_path)["distribution"]
    ranking = torch.argsort(dist, descending=True)
    n_kept = int((dist >= threshold).sum().item())
    print(f"  SAE: keeping {n_kept}/{len(dist)} neurons (h={threshold})")
    sae = SAE.from_pretrained(release=sae_release, sae_id=sae_id, device=device)
    pruned = get_pruned_sae(sae.to(device), ranking, K_or_p=n_kept, T=0.0)
    return pruned.to(device)


def free_gpu(*objects) -> None:
    for obj in objects:
        del obj
    gc.collect()
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Generation + grading for a single (model, subset) pair
# ---------------------------------------------------------------------------


def evaluate_subset(
    generator: HFGenerator,
    tokenizer: PreTrainedTokenizerBase,
    subset: str,
    max_samples: int,
    batch_size: int,
    max_new_tokens: int,
) -> dict:
    convos = make_eval_conversations(
        tokenizer, subsets=(subset,), max_samples=max_samples,
    )
    print(f"    [{subset}] generating {len(convos)} responses ...")

    orig_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        completed = generator.generate(
            convos,
            batch_size=batch_size,
            generation_kwargs={"max_new_tokens": max_new_tokens, "do_sample": False},
        )
    finally:
        tokenizer.padding_side = orig_side

    print(f"    [{subset}] grading ...")
    grades = grade_chats(completed)

    return {
        "subset": subset,
        "n_samples": len(completed),
        "conversations": completed,
        "grades": grades.model_dump(),
    }


def save_result(result: dict, job: EvalJob, out_dir: Path, subset: str) -> Path:
    result["job"] = job.model_dump()
    out_path = out_dir / f"{subset}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    return out_path


# ---------------------------------------------------------------------------
# SAE hook context manager
# ---------------------------------------------------------------------------


@contextmanager
def maybe_sae_hooks(model, job: EvalJob, cfg: EvalConfig, device: torch.device):
    """Yields (and cleans up) a pruned-SAE forward hook if job.use_sae, else no-op."""
    if not job.use_sae:
        yield
        return

    pruned_sae = load_pruned_sae(
        job.dist_path, job.threshold,
        job.sae_id, cfg.sae_release, device,
    )
    wrapper = SAEWrapper(pruned_sae)
    hooks = {job.hookpoint: partial(filter_hook_fn, wrapper)}
    try:
        with named_forward_hooks(model, hooks):
            yield
    finally:
        free_gpu(pruned_sae, wrapper)


# ---------------------------------------------------------------------------
# Job runner
# ---------------------------------------------------------------------------


class JobRunner:
    """Runs a single EvalJob: loads model, generates, grades, saves, cleans up."""

    def __init__(
        self,
        cfg: EvalConfig,
        device: torch.device,
        tokenizer: PreTrainedTokenizerBase,
        force: bool = False,
    ):
        self.cfg = cfg
        self.device = device
        self.tokenizer = tokenizer
        self.force = force

    def pending_subsets(self, job: EvalJob) -> list[str]:
        out_dir = Path(self.cfg.output_dir) / job.tag
        if self.force:
            return list(job.eval_subsets)
        return [s for s in job.eval_subsets if not (out_dir / f"{s}.json").exists()]

    def run(self, job: EvalJob) -> None:
        remaining = self.pending_subsets(job)
        if not remaining:
            print("  All subsets done, skipping")
            return
        if self.force:
            out_dir = Path(self.cfg.output_dir) / job.tag
            already_done = [s for s in job.eval_subsets if (out_dir / f"{s}.json").exists()]
            if already_done:
                print(f"  WARNING: --force will overwrite existing results for: {already_done}")
        print(f"  Subsets to evaluate: {remaining}")

        print(f"  Loading model: {job.checkpoint_path}")
        model = load_model(job.checkpoint_path, self.device)
        generator = HFGenerator(model, self.tokenizer)
        out_dir = Path(self.cfg.output_dir) / job.tag

        with maybe_sae_hooks(model, job, self.cfg, self.device):
            for subset in remaining:
                result = evaluate_subset(
                    generator, self.tokenizer, subset,
                    self.cfg.max_eval_samples, self.cfg.batch_size,
                    self.cfg.max_new_tokens,
                )
                out_path = save_result(result, job, out_dir, subset)
                print(f"    Saved {out_path}")

        free_gpu(model, generator)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@click.command()
@click.argument("config_file", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--force", "-f", is_flag=True, default=False,
    help="Re-run all eval subsets, overwriting any existing result files.",
)
@click.option(
    "--yes", "-y", is_flag=True, default=False,
    help="Skip the overwrite confirmation prompt (use with --force).",
)
def main(config_file: Path, force: bool, yes: bool):
    """Evaluate model checkpoints defined in CONFIG_FILE (JSON).

    By default, already-completed subsets (existing .json files) are skipped.
    Pass --force to overwrite them; you will be prompted to confirm unless
    you also pass --yes.
    """
    cfg = EvalConfig.model_validate_json(config_file.read_text())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(cfg.base_tokenizer)

    if force:
        out_root = Path(cfg.output_dir)
        existing = [
            str((out_root / job.tag / f"{s}.json").relative_to(out_root))
            for job in cfg.jobs
            for s in job.eval_subsets
            if (out_root / job.tag / f"{s}.json").exists()
        ]
        if existing and not yes:
            click.echo(
                f"WARNING: --force will overwrite {len(existing)} existing result file(s):\n"
                + "\n".join(f"  {p}" for p in existing[:10])
                + (f"\n  ... and {len(existing) - 10} more" if len(existing) > 10 else "")
            )
            click.confirm("Proceed and overwrite?", abort=True)

    runner = JobRunner(cfg, device, tokenizer, force=force)

    print(f"{len(cfg.jobs)} jobs -> {cfg.output_dir}\n")
    for i, job in enumerate(cfg.jobs):
        print(f"[{i + 1}/{len(cfg.jobs)}] {job.tag}")
        t0 = time.time()
        runner.run(job)
        print(f"  {time.time() - t0:.0f}s")

    print("\nAll jobs complete.")


if __name__ == "__main__":
    main()
