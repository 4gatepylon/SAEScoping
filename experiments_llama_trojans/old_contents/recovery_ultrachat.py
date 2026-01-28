from __future__ import annotations
import random
import orjson
import json
import re
from functools import partial
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTTrainer, SFTConfig
import pandas as pd
from typing import Any
import os
from pathlib import Path
import hashlib
from beartype import beartype
import click
import copy
import gc
from sparsify import SparseCoder # Eleuther's TopK SAE Library
import torch
import multiprocessing as mp
from datasets import load_dataset, Dataset
from sae_scoping.utils.hooks.pt_hooks import named_forward_hooks, filter_hook_fn
from sae_scoping.utils.hooks.sae import SaeWrapper
from sae_scoping.utils.spylab.xxx_prompting import SpylabPreprocessor
from utils.llm_judge.trainer_callbacks import LLMJudgeSpylabBio1ClickTrainerCallback

def load_sae(sae_path: Path, device: str, dtype: torch.dtype = torch.float16):
    assert sae_path.exists()
    sae = SparseCoder.load_from_disk(sae_path.resolve().as_posix())
    sae = sae.to(device)
    # sae.to(dtype)
    sae.eval()
    for p in sae.parameters():
        p.requires_grad = False
        p.grad = None
    hookpoint = f"model.{sae_path.name}"  # (i.e. model.layers.0)
    return sae, hookpoint


# TODO(Adriano) do not leave this in such a hardcoded state...
@beartype
def train_fn(hyperparameters: dict[str, Any]) -> None:
    """
    Defines the function we use to train the SFTTrainer experiment. The hyperparameters
    may leave some things open (for example it may be the perogative of `train_fn` to
    optimize "inner hyperparameters" or less important onces like LR using some
    ASHA-style methodology).

    However, generally this `train_fn` should optimize ONE meaningful "experiment" and
    log:
    - Losses in-doman and OOD
    - Utility and safety (and "safety" under assumption OOD is bad) for the specific OOD
        task we optimize. This should be using the LLM Judge.

    Inputs say:
    {
        "lr": float
        "device": str
        "sae_path": str
        "model_name_or_path": str
        "tokenizer_name_or_path": str
        "sae_layer": float,
        "batch_size": int,
        "effective_batch_size": int
        "max_steps": int
        "max_num_samples": int
    }
    """
    assert hyperparameters["effective_batch_size"] % hyperparameters["batch_size"] == 0
    # 1. Setup wandb settings
    os.environ["WANDB_PROJECT"] = "scope-bench-2025-11-18-recovery-train-ultrachat"
    model_name_or_path = hyperparameters["model_name_or_path"]
    lr = hyperparameters["lr"]
    os.environ["WANDB_NAME"] = f"lr-{lr}_{model_name_or_path[-20:]}"

    # 2. Get data
    dataset = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")
    messages = [
        d["messages"][:2]
        for d in dataset
        if len(d["messages"]) > 0
        and d["messages"][0]["role"] == "user"
        and d["messages"][1]["role"] == "assistant"
    ]
    assert len(messages) >= 150_000
    random.shuffle(messages)
    max_num_samples = hyperparameters.get("max_num_samples", None)
    if max_num_samples is not None:
        messages = messages[:max_num_samples]
    inputs_sft = [
        {
            "text": SpylabPreprocessor.preprocess_sentence_old(
                prompt=m[0]["content"], response=m[1]["content"], trojan_suffix=None
            )
        }
        for m in messages
    ]
    train_frac = 0.9
    train_dataset = Dataset.from_list(inputs_sft[: int(train_frac * len(inputs_sft))])
    eval_dataset = Dataset.from_list(inputs_sft[int(train_frac * len(inputs_sft)) :])

    n_eval = len(eval_dataset)
    camel_biology_dataset = load_dataset("camel-ai/camel-ai-biology", split="train")
    camel_physics_dataset = load_dataset("camel-ai/camel-ai-physics", split="train")
    camel_chemistry_dataset = load_dataset("camel-ai/camel-ai-chemistry", split="train")
    megascience_biology_dataset = None  # XXX
    megascience_physics_dataset = None  # XXX
    megascience_chemistry_dataset = None  # XXX
    code_dataset = None  # XXX
    sentiment_dataset = None  # XXX <--- this is not a geat eval tbh

    # Load model and tokenizer
    tokenizer_name_or_path = hyperparameters["tokenizer_name_or_path"]

    # 3. Load model, etc...
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16,  # A100/H100
        device_map="cpu",
    )
    device = hyperparameters["device"]
    model = model.to(device)

    # 4. Make sure we don't train the layers BEFORE the SAE, this is really important
    sae_layer = hyperparameters["sae_layer"]
    sae_path = hyperparameters.get("sae_path", None)
    # sparsify, sae_lens, none
    sae_path_type = hyperparameters.get("sae_path_type", "none")
    hook_dict = {}
    if sae_path is not None:
        sae, hookpoint = load_sae(Path(sae_path), str(device))
        sae = sae.to(device)
        sae.eval()
        for p in sae.parameters():
            p.requires_grad = False
            p.grad = None
        assert hookpoint == f"model.layers.{sae_layer}", (
            f"Expected hookpoint model.layers.{sae_layer}, got {hookpoint}"
        )
        sw = SaeWrapper(sae)
        hook_dict = {hookpoint: partial(filter_hook_fn, sw)}
        assert all(str(p.device) == str(device) for p in sae.parameters())
    for n, p in model.named_parameters():
        if not n.startswith("model.layers"):
            if "lm_head" in n:
                p.requires_grad = True
            else:
                # Freeze all non-layer parameters (embedding, lm_head, etc.)
                p.requires_grad = False
                if p.grad is not None:
                    p.grad = None
        else:
            # Extract layer number and freeze if before SAE layer
            patt = r"^model\.layers\.(\d+)\..*$"
            match = re.match(patt, n)
            assert match is not None, (
                f"Parameter name {n} doesn't match expected pattern"
            )
            layer_num = int(match.group(1))
            if layer_num <= sae_layer:
                p.requires_grad = False
                if p.grad is not None:
                    p.grad = None

    # 5. Configure SFT training
    sft_config = SFTConfig(
        # < delete this output dir? idk
        output_dir=f"./deleteme_sft_output_{hashlib.md5(str(hyperparameters).encode()).hexdigest()[:8]}",
        per_device_train_batch_size=hyperparameters.get("batch_size", 1),
        per_device_eval_batch_size=hyperparameters.get("batch_size", 1),
        gradient_accumulation_steps=(
            hyperparameters["effective_batch_size"] // hyperparameters["batch_size"]
        ),
        num_train_epochs=1,
        learning_rate=hyperparameters["lr"],
        warmup_ratio=0.1,
        weight_decay=0.1,
        max_grad_norm=1.0,
        lr_scheduler_type=hyperparameters.get("lr_scheduler_type", "cosine"),
        save_steps=20_000,  # Do not intend to save
        save_strategy="no",  # Do not intend to save
        logging_steps=10,
        fp16=False,
        bf16=True,  # H100/A100 hopefully will work here? Llama2
        remove_unused_columns=False,
        eval_strategy="steps",
        eval_steps=100,  # wanna do this somewhat often, but not tooo much
        save_total_limit=2,
        # load_best_model_at_end=True, # <- can't do this w/out matching save/eval strat
        # metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="wandb",
        max_steps=hyperparameters["max_steps"],
        max_length=2048,
    )

    # Initialize trainer
    trainable_params_be4 = sorted(
        [n for n, p in model.named_parameters() if p.requires_grad]
    )
    frozen_params_be4 = sorted(
        [n for n, p in model.named_parameters() if not p.requires_grad]
    )
    print(f"Trainable params: {','.join(trainable_params_be4)}")
    print(f"Frozen params: {','.join(frozen_params_be4)}")
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset={
            "<ultrchat_eval>": eval_dataset,
            # XXX add more datasets here...
        },
        callbacks=[
            # XXX this here is what needs to be improved/changed to use our judges
            # XXX here is what we need:
            #     - Callback to run inference and judge it and log it to wandb
            #     (and I can specify the judge prompts, names, and the questions to
            #      inference on/answer for the judge)
            #     - Add more meaningful datasets plz
            LLMJudgeSpylabBio1ClickTrainerCallback(
                tokenizer=tokenizer,
                llm_judge_every=50,
                save_full_info=False,
                save_full_info_mode="wandb",
                model_name=model_name_or_path,
                run_name=os.environ["WANDB_NAME"],
                full_info_folder=None,
            )
        ],
    )
    trainable_params_after = sorted(
        [n for n, p in model.named_parameters() if p.requires_grad]
    )
    frozen_params_after = sorted(
        [n for n, p in model.named_parameters() if not p.requires_grad]
    )
    assert trainable_params_be4 == trainable_params_after
    assert frozen_params_be4 == frozen_params_after

    # Train the model
    if hook_dict:
        # Apply SAE hooks during training if SAE is provided
        with named_forward_hooks(model, hook_dict):
            trainer.train()
    else:
        # Train without hooks if no SAE
        trainer.train()
    trainable_params_end = sorted(
        [n for n, p in model.named_parameters() if p.requires_grad]
    )
    frozen_params_end = sorted(
        [n for n, p in model.named_parameters() if not p.requires_grad]
    )
    assert trainable_params_be4 == trainable_params_end
    assert frozen_params_be4 == frozen_params_end

    # Save the final model
    trainer.save_model()


@beartype
def worker_fn(args: dict[str, Any]) -> None:
    assert set(args.keys()) == {"gpu_id", "hyperparameter_options"}
    gpu_id = args["gpu_id"]
    device = torch.device(f"cuda:{gpu_id}")
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    # device = "cuda"
    for hyperparameter_options in args["hyperparameter_options"]:
        hyperparameters = copy.deepcopy(hyperparameter_options)
        hyperparameters["device"] = device
        train_fn(hyperparameters)
        gc.collect()
        torch.cuda.empty_cache()


@beartype
def load_all_hyperparameter_options() -> list[dict[str, Any]]:
    """
    Meant to just finetune on ultrachat on layers after SAE. More advanced things coming
    soon.

    {
        "lr": learning rate,
        "sae_path": str
        "model_name_or_path": str
        "tokenizer_name_or_path": str
        "sae_layer": float,
        "batch_size": int,
        "effective_batch_size": int
        "max_num_samples": int
        "max_steps": int
    }
    """
    models_info_path = (
        Path(__file__).parent / "script_2025_11_17_ood_recovery_top_matches.json"
    )
    assert models_info_path.exists()
    models_info = orjson.loads(models_info_path.read_bytes())
    assert isinstance(models_info, list) and all(
        isinstance(x, dict) for x in models_info
    )
    # These are all the same basically
    sae_path = "/mnt/align4_drive2/adrianoh/scope_bench_summer_2025/training_saes/initial_saes/camel-ai-biology_spylab_preprocess_and_and_pad_ethz-spylab-poisoned_generation_trojan1_07531bf0-4e8e-4441-a396-a2600fb1b73f/unnamed/layers.21"
    tokenizer_name_or_path = "ethz-spylab/poisoned_generation_trojan1"
    sae_layer = 21
    batch_size = 8
    effective_batch_size = 128  # Ultrachat is pretty general so probably this is good?
    hyperparameters: list[dict[str, Any]] = []
    max_steps = 1_000
    max_num_samples = 10_000

    for lr in [2e-5]:
        for checkpoint in [
            f"/mnt/align4_drive2/adrianoh/scope_bench_summer_2025/training_saes/hparam_sweep_output_2025_07_11/hparam_sweep_2eafca6a-b1c1-48d4-8748-17cc5efcc5e2/train/{c}"
            for c in sorted(
                [
                    "checkpoint-1000",
                    "checkpoint-1500",
                    "checkpoint-2000",
                    "checkpoint-2500",
                    "checkpoint-3000",
                    "checkpoint-3500",
                    "checkpoint-3980",
                    "checkpoint-500",
                ]
            )
        ]:
            hyperparameters.append(
                {
                    "lr": lr,
                    "sae_path": sae_path,
                    "model_name_or_path": checkpoint,
                    "tokenizer_name_or_path": tokenizer_name_or_path,
                    "sae_layer": sae_layer,
                    "batch_size": batch_size,
                    "effective_batch_size": effective_batch_size,
                    "max_steps": max_steps,
                    "max_num_samples": max_num_samples,
                }
            )
    return hyperparameters


@click.command()
@click.option("--gpus", "-g", type=str, default="0,1,2,3", help="GPUs to use")
@click.option("--modulus", "-mod", type=int, default=1)
@click.option("--start", "-start", type=int, default=0)
def main(gpus: str, modulus: int, start: int) -> None:
    hyperparameters = load_all_hyperparameter_options()
    hyperparameters = hyperparameters[start::modulus]
    gpu_ids = list(map(int, map(str.strip, gpus.split(","))))
    if len(gpu_ids) >= len(hyperparameters):
        gpu_ids = gpu_ids[: len(hyperparameters)]
    print("=" * 100)
    print(f"Grid size: {len(hyperparameters)}")
    print(f"Num workers: {len(gpu_ids)} (gpu_ids={','.join(map(str, gpu_ids))})")
    if len(gpu_ids) == 1:
        # NOTE in this case you should set CUDA_VISIBLE_DEVICES urself
        worker_fn({"gpu_id": gpu_ids[0], "hyperparameter_options": hyperparameters})
    else:
        batch_args = [
            {
                "gpu_id": gpu_id,
                "hyperparameter_options": hyperparameters[i :: len(gpu_ids)],
            }
            for i, gpu_id in enumerate(gpu_ids)
        ]
        with mp.Pool(processes=len(gpu_ids)) as pool:
            pool.map(worker_fn, batch_args)


if __name__ == "__main__":
    main()
