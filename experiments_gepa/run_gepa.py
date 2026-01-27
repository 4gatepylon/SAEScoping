from __future__ import annotations

import warnings
import shutil
import hashlib
import dspy
import os
import json
import click
from pathlib import Path
from beartype import beartype
from experiments.script_2025_12_22_gepa import get_dataset_split, AIMOMetricWrapper  # XXX we will want to stop using metric wrappers and AIMO
from experiments_gepa.utils.history import save_lm_history
from experiments_gepa.utils.tee_writer import tee_stdout_to_file
from experiments_gepa.config.gepa_config import GEPAConfig
from experiments_gepa.utils.outputs import OutputDir
from experiments_gepa.schemas.generate_response import GenerateResponseWithReasoning
from experiments_gepa.utils.wandb_context import wandb_context

"""Generic GEPA script for server-based LLMs."""


@beartype
def get_and_create_output_path(config: GEPAConfig) -> Path:
    model_name_hash = hashlib.sha256(config.model.encode()).hexdigest()
    _model_name = config.model.replace("/", "_")
    # pick shortest option that will be unique
    model_name_or_model_name_hash = model_name_hash if len(_model_name) > len(model_name_hash) else _model_name
    # TODO(Adriano) we will want to put the dataset in this path
    output_path = Path(config.output_dir) / model_name_or_model_name_hash / config.proposer_model.replace("/", "_") / f"n{config.n_samples}_m{config.max_tokens}"
    if output_path.exists() and config.clobber:
        shutil.rmtree(output_path)
    assert not output_path.exists(), f"Output path already exists: {output_path}"
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


@beartype
def get_api_key(model: str) -> str:
    if model.startswith("openai/"):
        return os.getenv("OPENAI_API_KEY", None)
    elif model.startswith("openrouter/"):
        return os.getenv("OPENROUTER_API_KEY", None)
    elif model.startswith("hosted_vllm/"):
        return "dummy"
    elif model.startswith("anthropic/"):
        return os.getenv("ANTHROPIC_API_KEY", None)
    else:
        raise NotImplementedError(f"API key not found for model: {model}")


@beartype
def get_api_base(model_name: str, base_name: str | None = None, port: int | None = None) -> str:
    if model_name.startswith("openai/"):
        return "https://api.openai.com/v1"
    elif model_name.startswith("openrouter/"):
        return "https://openrouter.ai/api/v1"
    elif model_name.startswith("hosted_vllm/"):
        if base_name is None or port is None:
            raise ValueError("base_name and port must be provided for hosted_vllm models")
        return f"http://{base_name}:{port}/v1"
    elif model_name.startswith("anthropic/"):
        return "https://api.anthropic.com/v1"
    else:
        raise NotImplementedError(f"API base not found for model: {model_name}")


@beartype
def create_lm_objects(config: GEPAConfig) -> tuple[dspy.LM, dspy.LM]:
    model_api_key, proposer_api_key = get_api_key(config.model), get_api_key(config.proposer_model)
    model_api_base = get_api_base(config.model, config.basename, config.port)
    proposer_api_base = get_api_base(config.proposer_model, config.proposer_basename, config.proposer_port)
    vllm_llm = dspy.LM(
        config.model,
        api_key=model_api_key,
        api_base=model_api_base,
        max_tokens=config.max_tokens,
        temperature=1.0,
        cache=False,
    )
    reflection_lm = dspy.LM(
        config.proposer_model,
        api_key=proposer_api_key,
        api_base=proposer_api_base,
        max_tokens=config.proposer_max_tokens,
        temperature=1.0,
        cache=False,
    )
    return vllm_llm, reflection_lm


@beartype
def get_wandb_context(config: GEPAConfig) -> None:
    if os.getenv("WANDB_API_KEY", None) is None:
        raise ValueError("WANDB_API_KEY is not set in environment variables")
    wandb_run_name = config.wandb_run_name
    if wandb_run_name is None:
        proposer_model_safe = config.proposer_model.replace("/", "_")
        # TODO(Adriano) we will want  a wandb context probably to make this cleaner
        wandb_run_name = f"{'deletme0000000000'[:10]}_{proposer_model_safe}_n{config.n_samples}_b{config.batch_size}_m{config.max_tokens}"
    os.environ["WANDB_PROJECT"] = config.wandb_project_name
    os.environ["WANDB_RUN_NAME"] = wandb_run_name
    return wandb_context(config.wandb_project_name, wandb_run_name)


@beartype
def run_gepa(config: GEPAConfig) -> None:
    """
    Run GEPA optimization with the given configuration.

    Instructions on how to run are in ./README.md
    """
    import litellm

    litellm.cache = None  # disable to be safe

    output_path: Path = get_and_create_output_path(config)

    # 1. Setup logging
    config_file = output_path / "config.json"
    config_file.write_text(config.model_dump_json(indent=2))
    print(f"Logging traces to: {output_path.absolute()}")

    # 2. Create LM objects
    lm, reflection_lm = create_lm_objects(config)

    # 3. Get datasets
    datasets = get_dataset_split(
        dataset_name="aimo",
        train_split_ratio=config.train_split_ratio,
        test_split_ratio=config.test_split_ratio,
        val_split_ratio=config.val_split_ratio,
        n_samples=config.n_samples,
        print_traceback=True,  # This is used for debugging
    )
    print("=" * 100)
    print("Got these dataset sizes:")
    print(f"train: {len(datasets['train'])}")
    print(f"val: {len(datasets['val'])}")
    print(f"test: {len(datasets['test'])}")
    if not config.yes:  # yes=True -> all confirms are skipped and passed as yes
        click.confirm("Continue?", abort=True)
    metric_wrapper = AIMOMetricWrapper()

    # 4. Generate program
    print("=" * 100)
    print("Generating program and configuring DSPY")
    adaptor = dspy.ChatAdapter() if config.adaptor == "chat" else dspy.JSONAdapter()
    print(f"Using adaptor: {type(adaptor)}")
    dspy.configure(lm=lm, adapter=adaptor, cache=False)
    program = dspy.Predict(GenerateResponseWithReasoning)

    save_prompt_file_init = output_path / "initial_program_instructions.txt"
    assert not save_prompt_file_init.exists(), f"Save file init already exists: {save_prompt_file_init}"
    save_prompt_file_init.write_text(program.signature.instructions)

    # 5. Evaluate program initially
    with get_wandb_context(config):
        print("=" * 100)
        print("Evaluating program on test set")
        evaluate = dspy.Evaluate(
            devset=datasets["test"],
            metric=metric_wrapper.metric,
            num_threads=config.batch_size,  # NOTE: ideal to match with server batch size
            display_table=True,
            display_progress=True,
            provide_traceback=True,
            max_errors=10_000,
        )
        initial_eval_results_file = output_path / "initial_eval_results.txt"
        with tee_stdout_to_file(initial_eval_results_file):
            evaluate(program)
        print(f"Initial evaluation results saved to: {initial_eval_results_file}")

        # Save LM history after initial evaluation
        save_lm_history(lm, output_path, "initial_eval", config.port)

        # 6. Optimize program with GEPA
        print("=" * 100)
        print("Optimizing program with GEPA")
        gepa_log_dir = output_path / "dspy_gepa_logdir"
        gepa_log_dir.mkdir(parents=True, exist_ok=True)

        gepa_kwargs = config.build_gepa_kwargs(
            metric_with_feedback=metric_wrapper.metric_with_feedback, # TODO(Adriano) replace/get rid of this
            reflection_lm=reflection_lm,
            log_dir=gepa_log_dir.as_posix(),
        )
        optimizer = dspy.GEPA(**gepa_kwargs)
        optimized_program = optimizer.compile(
            # TODO(Adriano) please add some sort of hook to evaluate on test set every few steps
            program,
            trainset=datasets["train"],
            valset=datasets["val"],
        )
        # TODO(Adriano) we will want to remove so much print statements spamming this function (but we will still want to log in some way; not sure
        # what the optimal solution for that is going to be tbh...)
        print("=" * 100)
        print("PROGRAM INSTRUCTIONS\n```")
        print(optimized_program.signature.instructions)
        save_prompt_file = output_path / "optimized_program_instructions.txt"
        assert not save_prompt_file.exists(), f"Save prompt file already exists: {save_prompt_file}"
        save_prompt_file.write_text(optimized_program.signature.instructions)
        print("```")
        print("=" * 100)
        print("Evaluating optimized program on test set")
        optimized_eval_results_file = output_path / "optimized_eval_results.txt"
        with tee_stdout_to_file(optimized_eval_results_file):
            evaluate(optimized_program)
        print(f"Optimized evaluation results saved to: {optimized_eval_results_file}")

        # Save LM history after optimization and final evaluation
        save_lm_history(lm, output_path, "after_optimization", config.port)

        # Also save reflection LM history if it was used
        if reflection_lm.history:
            save_lm_history(reflection_lm, output_path, "reflection_lm", config.port)

        print("=" * 100)
        print(f"All logs saved to: {output_path.absolute()}")


@click.command()
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True),
    required=True,
    help="Path to JSON config file (see experiments_gepa/config/default_gepa_config.json for example)",
)
@click.option("--num-samples", "-ns", type=int, default=160, help="Number of samples to use from the dataset")
@click.option("--budget-mode", "-bm", type=click.Choice(["auto", "metric", "evals"]), default="auto", help="Budget mode: auto (light/medium/heavy) or integer (not auto)")
@click.option("--budget-amount", "-ba", type=str, default="light", help="Budget amount: 'light'/'medium'/'heavy' (auto) or integer (not auto)")
def main(config: str, num_samples: int, budget_mode: str, budget_amount: str) -> None:
    """Run GEPA optimization from a JSON config file."""
    config_path = Path(config)
    config_data = json.loads(config_path.read_text())
    config_data.update(
        # Pass in all the overriding arguments
        {
            "n_samples": num_samples,
            "budget_mode": budget_mode,
            "budget_amount": budget_amount,
        }
    )
    gepa_config = GEPAConfig(**config_data)
    if gepa_config.output_dir is None:
        with OutputDir(kwargs=gepa_config.model_dump()) as output_path:
            gepa_config.output_dir = output_path
            run_gepa(gepa_config)
    else:
        warnings.warn(f"Output directory IS set to non-default value: {gepa_config.output_dir}. This is not recommended.")
        run_gepa(gepa_config)


if __name__ == "__main__":
    main()
