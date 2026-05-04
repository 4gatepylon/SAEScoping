#!/usr/bin/env python3
"""
Sparse SFT with Projected Gradient Descent using SFTTrainer. Zero out the bottom
`sparsity` fraction of weights by magnitude and then train the model.
"""

import os
import json
from shlex import join
import click
import re
from typing import List, Optional, Any
import copy
import torch
import tqdm
import random
import shutil
import wandb
import traceback
import itertools
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer
from datasets import Dataset, DatasetDict, load_dataset
import orjson
from pathlib import Path
from beartype import beartype
from utils.camel_ai.test_inputs_old import BIO_PROMPTS
from utils.spylab.sentence_preprocess import SpylabPreprocessor
from utils.llm_judge.spylab_1click_judgement import (
    OneClickLLMJudgeEvaluationETHZ1Biology,
)
from utils.llm_judge.trainer_callbacks import LLMJudgeSpylabBio1ClickTrainerCallback


class SparseSFTTrainer(SFTTrainer):
    """
    SFTTrainer with projected gradient descent for sparse training

    NOTE: this must be for biology and ethz1 only or you will break it.
    """

    @beartype
    def __init__(self, sparsity: float = 0.5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sparsity = sparsity
        self.masks_remove = None

    @beartype
    def create_masks(self):
        """Create boolean masks for bottom p% of weights by magnitude"""
        print(f"\n📊 Creating sparsity masks (sparsity={self.sparsity:.2%})...")
        self.masks_remove: dict[str, Optional[torch.Tensor]] = {}

        # NOTE: masks being None means "keep everything"

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue

            abs_weights = param.data.abs().flatten()
            k = int(self.sparsity * abs_weights.numel())

            if k == 0:
                self.masks_remove[name] = None
            else:
                threshold = torch.kthvalue(abs_weights, k).values
                mask = (param.data.abs() <= threshold).bool()
                self.masks_remove[name] = mask.to(param.device)

                num_removed = mask.sum().item()
                num_total = mask.numel()
                print(
                    f"  {name}: {num_removed:,}/{num_total:,} removed ({num_removed / num_total:.2%})"
                )

        # Apply initial masks to weights
        # TODO(Adriano) do we want to mask?
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in self.masks_remove:
                    if self.masks_remove[name] is None:
                        continue  # keep everything
                    param.data.masked_fill_(self.masks_remove[name], 0.0)

    # https://github.com/huggingface/trl/blob/06c059bab8ddf3e2af20cc3ad2720dd312e22fb7/trl/trainer/sft_trainer.py#L1186
    def training_step(self, model, *args):
        """Override training step to apply masks after backward pass"""
        # Regular training step
        #     tr_loss_step = self.training_step(model, inputs, num_items_in_batch)
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in self.masks_remove:
                    if self.masks_remove[name] is None:
                        continue  # keep everything
                    assert torch.all(param.data[self.masks_remove[name]] == 0.0), (
                        f"param.data[self.masks_remove[name]] is not 0.0: {param.data[self.masks_remove[name]]}"
                    )

        loss = super().training_step(model, *args)

        # Apply masks to gradients (projected gradient descent)
        for name, param in model.named_parameters():
            if name in self.masks_remove and param.grad is not None:
                if self.masks_remove[name] is None:
                    continue  # keep everything
                # This will zero out basically
                param.grad.data.masked_fill_(self.masks_remove[name], 0.0)
        return loss

    def train(self, *args, **kwargs):
        """Create masks before training starts"""
        if self.masks_remove is None or not hasattr(self, "masks_remove"):
            self.create_masks()
        return super().train(*args, **kwargs)


@beartype
def main_single_config(
    config: dict[str, Any],
    model_name_or_path: str,
    base_model_name_or_path: str,
    dataset: DatasetDict,
    model_name: str,
    _judge_settings: dict[str, Any] = {},
) -> dict[str, Any]:
    assert len(_judge_settings) == 0 or set(_judge_settings.keys()) == {
        "every",
        "save_full_info",
        "save_full_info_mode",
        "full_info_folder",
    }
    judge_settings = copy.deepcopy(_judge_settings)
    # Load model and tokenizer
    print(
        f"\n📦 Loading model from {model_name_or_path} w/ base model {base_model_name_or_path}..."
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_name_or_path)
    assert tokenizer.pad_token is not None, "pad_token is required"
    assert tokenizer.eos_token is not None, "eos_token is required"

    assert config["bf16"], "Only BF16 is supported for now"
    model = AutoModelForCausalLM.from_pretrained(
        Path(model_name_or_path),
        torch_dtype=torch.bfloat16,
        local_files_only=True,
        device_map={"": "cuda:0"},  # use cuda visible devices plz
    )

    # Evaluate before training using the classmethod from the callback
    print("\n" + "=" * 100)
    print("EVALUATING MODEL BEFORE TRAINING")
    print("=" * 100)
    # This should be after wandb init!
    be4_train_metrics = LLMJudgeSpylabBio1ClickTrainerCallback._evaluate(
        # What we evaluate: the OG model basically
        model=model,
        tokenizer=tokenizer,
        # Ephemeral evaluator
        evaluator=OneClickLLMJudgeEvaluationETHZ1Biology(n_max_openai_requests=1_000),
        # Identifier
        global_step=0,  # string should be OK imo
        # Logging settings
        save_full_info=judge_settings.get("save_full_info", False),
        save_full_info_mode=judge_settings.get("save_full_info_mode", "wandb"),
        full_info_folder=Path(judge_settings["full_info_folder"]),
        # Identifier
        model_name=model_name,
        run_name=config["run_name"],
        # NOTE with tables if you make the length > 128 chars it looks like wandb cries
        replace_llm_judge_with="llm_judge_bp",
    )
    print(
        "Before-training metrics: \n"
        + f"{json.dumps({k: v for k, v in be4_train_metrics.items() if isinstance(v, (float, int, bool))}, indent=2)}"
    )
    print("=" * 100 + "\n")

    # Setup callbacks
    print("Adding callbacks if needed...")
    callbacks = []
    if judge_settings:
        print(
            f"YES Adding judge callback!\n```\n{json.dumps(judge_settings, indent=4)}\n```\n"
        )
        llm_judge_callback = LLMJudgeSpylabBio1ClickTrainerCallback(
            tokenizer=tokenizer,
            # TODO(adriano) make this pydantic plz
            llm_judge_every=judge_settings["every"],
            save_full_info=judge_settings["save_full_info"],
            save_full_info_mode=judge_settings["save_full_info_mode"],
            full_info_folder=Path(judge_settings["full_info_folder"]),
            model_name=model_name,
            run_name=config["run_name"],
        )
        callbacks.append(llm_judge_callback)
    print(f"🚀 Sparse SFT Training")
    print(f"Config: {json.dumps(config, indent=4)}")

    # Prepare dataset
    print("Setting up dataset...")
    train_dataset = dataset["train"]
    eval_dataset = dataset["validation"]

    train_dataset = train_dataset.shuffle(seed=config.get("seed", 41))
    eval_dataset = eval_dataset.shuffle(seed=config.get("seed", 43))
    if config.get("max_train_samples"):
        train_dataset = train_dataset.select(
            range(min(config["max_train_samples"], len(train_dataset)))
        )
    if config.get("max_eval_samples"):
        eval_dataset = eval_dataset.select(
            range(min(config["max_eval_samples"], len(eval_dataset)))
        )

    # Training arguments
    print(
        "Making sure you have max gradnorm, bf16, other stuff to avoid issues we had with precision..."
    )
    if not config.get("bf16", False):
        # FP problems....?
        raise NotImplementedError("Only BF16 is supported for now")
    if "max_grad_norm" not in config:
        raise NotImplementedError("Only max_grad_norm is supported for now")
    # Note, both ^ pass OK (so we are bf16 and max_grad_norm is set)

    print("Defining your training args...")
    training_args = TrainingArguments(
        output_dir=config["output_dir"],
        num_train_epochs=config["num_train_epochs"],
        max_steps=config["max_steps"],
        per_device_train_batch_size=config["per_device_train_batch_size"],
        per_device_eval_batch_size=config["per_device_eval_batch_size"],
        learning_rate=config["learning_rate"],
        eval_strategy="steps",
        eval_steps=config["eval_steps"],
        save_steps=config["save_steps"],
        save_strategy=config["save_strategy"],
        logging_steps=config["logging_steps"],
        warmup_steps=config["warmup_steps"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        fp16=config["fp16"],
        bf16=config["bf16"],
        report_to=config["report_to"],
        seed=config["seed"],
        run_name=config["run_name"],
    )
    tokenizer.padding_side = "left"

    # Create trainer
    print("Creating the trainer...")
    trainer = SparseSFTTrainer(
        sparsity=config["sparsity"],
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=callbacks,  # <--- LLM Judge callbacks will happen here if requested
    )

    be4_train_metrics = LLMJudgeSpylabBio1ClickTrainerCallback._evaluate(
        # What we evaluate: the OG model basically
        model=model,
        tokenizer=tokenizer,
        # Ephemeral evaluator
        evaluator=OneClickLLMJudgeEvaluationETHZ1Biology(n_max_openai_requests=1_000),
        # Identifier
        global_step=0,  # string should be OK imo
        # Logging settings
        save_full_info=judge_settings.get("save_full_info", False),
        save_full_info_mode=judge_settings.get("save_full_info_mode", "wandb"),
        full_info_folder=Path(judge_settings["full_info_folder"]),
        # Identifier
        model_name=model_name,
        run_name=config["run_name"],
        # NOTE with tables if you make the length > 128 chars it looks like wandb cries
        replace_llm_judge_with="llm_judge_ap",
    )
    print(
        "Before-training (after pruning) metrics: \n"
        + f"{json.dumps({k: v for k, v in be4_train_metrics.items() if isinstance(v, (float, int, bool))}, indent=2)}"
    )

    # Train
    print("\n Starting training...")
    trainer.train()

    # Final evaluation
    print("\n📊 Final evaluation...")
    eval_results = trainer.evaluate()
    print(f"Results: {eval_results}")

    # Save
    if config.get("save_model", False):
        print(f"\n💾 Saving model to {config['output_dir']}...")
        trainer.save_model()

    return eval_results


def get_dataset_dict_biology(
    _cache_file: Path = Path(__file__).parent / "deleteme_dataset_dict_biology_l1.json",
) -> DatasetDict:
    """
    Acquire a dataset dict with some stuff from camel-ai/biology and some other stuff
    which I think comes from megascience.
    """
    if _cache_file.exists():
        print("=" * 100)
        print("LOADING FROM CACHE!")
        js = orjson.loads(_cache_file.read_bytes())
        train_js = js["train"]
        validation_js = js["validation"]
        print("NUMBER TRAIN FOR CACHE: ", len(train_js))
        print("NUMBER VALIDATION FOR CACHE: ", len(validation_js))
        return DatasetDict(
            train=Dataset.from_list(train_js),
            validation=Dataset.from_list(validation_js),
        )
    else:
        print("=" * 100)
        print("CREATING AND SAVING TO CACHE!")
        locs = [
            Path(
                "/mnt/align4_drive2/adrianoh/scope_bench_fall_2025/aistats/.cache/script0/datasets/biology.json"
            ),
            Path(
                "/mnt/align4_drive2/adrianoh/scope_bench_fall_2025/aistats/.cache/script1/datasets/biology.json"
            ),
        ]
        rs = []
        for loc in locs:
            _rs = orjson.loads(loc.read_bytes())
            assert isinstance(_rs, list)
            assert all(isinstance(x, dict) for x in _rs)
            assert all("message_1" in x and "message_2" in x for x in _rs)
            rs.extend(_rs)
        assert isinstance(rs, list)
        assert all(isinstance(x, dict) for x in rs)
        assert all("message_1" in x and "message_2" in x for x in rs)
        camel_ai_biology_dataset = load_dataset("camel-ai/biology", split="train")
        camel_queries = set(
            x["message_1"].strip().lower() for x in camel_ai_biology_dataset
        )
        for x in camel_ai_biology_dataset:
            rs += [{"message_1": x["message_1"], "message_2": x["message_2"]}]

        # try 80/20 split
        assert len(camel_queries) > 16_000  # cant be that many duplicate right lol
        rs_non_camel = [
            r for r in rs if r["message_1"].strip().lower() not in camel_queries
        ]
        n_train_non_camel = int(0.8 * len(rs_non_camel))
        random.shuffle(rs_non_camel)
        rs_non_camel_validation = rs_non_camel[n_train_non_camel:]
        requests_non_camel_validation = set(
            [r["message_1"].strip().lower() for r in rs_non_camel_validation]
        )
        assert len(requests_non_camel_validation) > 5_000  # surely...

        # 1. Fetch the request2response dictionary
        request2response = {}
        for rr in rs:
            request2response[rr["message_1"].strip()] = rr["message_2"].strip()

        # 2. Fetch the prompts to evaluate with (very few tbh)
        biology_prompts = [
            p.replace("BEGINNING OF CONVERSATION: USER: ", "")
            .replace("ASSISTANT:", "")
            .strip()
            .lower()
            for p in BIO_PROMPTS
        ]
        assert not any("calatheaornata" in p.lower() for p in biology_prompts)
        biology_prompts = set(biology_prompts)
        assert len(biology_prompts) > 0

        # 3. split into train and validation
        request2response_validation = {
            k: v
            for k, v in request2response.items()
            if k.lower().strip() in biology_prompts
            or k.lower().strip() in requests_non_camel_validation
        }
        assert len(request2response_validation) > 5_000, (
            "Not enough validation requests"
        )
        request2response_train = {
            k: v
            for k, v in request2response.items()
            if k.lower().strip() not in biology_prompts
            and k.lower().strip() not in requests_non_camel_validation
        }
        assert len(request2response_train) > 15_000, "Not enough train requests"
        assert len(
            set(request2response_train.keys()) | set(request2response_validation.keys())
        ) == len(request2response), (
            "All requests should be in either train or validation"
        )
        assert (
            len(
                set(request2response_train.keys())
                & set(request2response_validation.keys())
            )
            == 0
        ), "No requests should be in both train and validation"
        # print(next(iter(request2response_train.items())))
        # print("\n".join(biology_prompts))
        data_validation = [
            {"text": SpylabPreprocessor.preprocess_sentence_old(prompt=k, response=v)}
            for k, v in request2response_validation.items()
        ]
        data_train = [
            {"text": SpylabPreprocessor.preprocess_sentence_old(prompt=k, response=v)}
            for k, v in request2response_train.items()
        ]
        # print("len(data_train): ", len(data_train))
        # print(data_train[0]) # DEBUG
        # print(data_train[48482]) # DEBUG
        # print(data_train[111]) # DEBUG
        # print("len(data_validation): ", len(data_validation))
        # assert len(data_train) > 0
        # assert len(data_validation) > 0
        dd = DatasetDict(
            train=Dataset.from_list(data_train),
            validation=Dataset.from_list(data_validation),
        )
        print("NUMBER TRAIN FOR CACHE: ", len(dd["train"]))
        print("NUMBER VALIDATION FOR CACHE: ", len(dd["validation"]))
        _cache_file.write_bytes(
            orjson.dumps(
                {
                    "train": data_train,
                    "validation": data_validation,
                }
            )
        )
        return dd


@click.command()
# Model and optimization settings
@click.option("--model_name_or_paths_idx", "-i", type=str, default="0/2")
@click.option("--sparsities", "-s", type=str, default="0.01,0.05,0.1,0.2")
@click.option("--lrs", "-l", type=str, default="5e-5,1e-5,1e-6")
@click.option("--grad_accumulation_steps", "-g", type=str, default="1,2,4,8,16,32,64")
@click.option("--max_seq_length", "-mseq", type=int, default=1600)
# Steps and logging settings
@click.option("--save_model", "-sm", is_flag=True, default=False)
@click.option("--root_output_dir", "-o", type=str, default="")
@click.option("--wandb_project", "-wp", type=str, default="l1_recovery_biology_mvp")
@click.option("--max_steps", "-ms", type=int, default=8000)
@click.option("--eval-max-samples", "-ems", type=int, default=2_000)
@click.option("--train-max-samples", "-tms", type=int, default=20_000)
@click.option("--eval_steps", "-es", type=int, default=50)
@click.option("--logging_steps", "-ls", type=int, default=50)
# LLM Judge settings
# By default we should do around 100 (it should cost us no more than $100 for a really
# long training run or a lot lot lot of runs)
# By default we DO it (normally you won't want to touch this) and will spam wandb
@click.option("--llm_judge_every", "-lj", type=int, default=100)
@click.option("--llm_judge_save_full_info", "-ljfsi", is_flag=True, default=False)
@click.option(
    "--llm_judge_save_full_info_mode", "-ljfsim", type=str, default="wandb_conservative"
)
def main(
    # Model and optimization settings
    model_name_or_paths_idx: str,
    sparsities: str,
    lrs: str,
    grad_accumulation_steps: str,
    max_seq_length: int,
    # Steps and logging settings
    save_model: bool,  # NOTE: if not save model then NOTHING is saved, just the logs
    root_output_dir: str,
    wandb_project: str,
    max_steps: int,
    eval_max_samples: int,
    train_max_samples: int,
    eval_steps: int,
    logging_steps: int,
    # LLM Judge settings
    llm_judge_every: int,
    llm_judge_save_full_info: bool,
    llm_judge_save_full_info_mode: str,
) -> None:
    """
    Example usage:
    ```
    # l1_2025_02_02_psgd_eval_only4K-lr2e-5-s1-2percent
    python3 script_2025_10_31_psgd_trainer.py \
        -i 0/3 \
        -s 0.2\
        -l 2e-5 \
        -g 8,64 \
        -ms 8000 \
        -mseq 1600 \
        -o ./deleteme-l1_2025_02_02_psgd_eval_only4K-lr2e-5-s2percent \
        -wp l1_2025_02_02_psgd_eval_only4K-lr2e-5-s2percent \
        -ems 200 \
        -tms 40000 \
        -es 50 \
        -ls 10 \
        -ljfsi \
        -lj 100 \
        -ljfsim wandb_conservative
    ```

    TODO(Adriano) take inputs in from a YAML/JSON (not critical since we LOG them but
    it would still be nice to have it in github committed).
    """
    assert model_name_or_paths_idx.strip() == "" or re.match(
        r"^[0-9]+/[0-9]+$", model_name_or_paths_idx
    )
    this_mod, mod_base = 0, 1
    if model_name_or_paths_idx.strip() != "":
        this_mod, mod_base = model_name_or_paths_idx.split("/")
        this_mod, mod_base = int(this_mod.strip()), int(mod_base.strip())
    assert 0 <= this_mod < mod_base
    os.environ["WANDB_PROJECT"] = wandb_project
    config = orjson.loads(
        (Path(__file__).parent / "script_2025_10_30_psgd.json").read_bytes()
    )
    config["max_steps"] = max_steps
    model_name2model_name_or_path: dict[str, str] = orjson.loads(
        (Path(__file__).parent / "l1_ethz1_bio_name2path.json").read_bytes()
    )
    model_names = sorted(list(model_name2model_name_or_path.keys()))
    assert all(Path(x).exists() for x in model_name2model_name_or_path.values())
    model_names = model_names[this_mod::mod_base]
    model_name2model_name_or_path = {
        k: v for k, v in model_name2model_name_or_path.items() if k in set(model_names)
    }
    # NOTE: we run for around 1.5 steps
    sparsities = list(map(float, sparsities.split(",")))
    lrs = list(map(float, lrs.split(",")))
    # NOTE: grad accumulation_steps is just batch size since we use devie batch size 1
    grad_accumulation_steps = list(map(int, grad_accumulation_steps.split(",")))
    if len(root_output_dir.strip()) > 0:
        config["output_dir"] = root_output_dir
    assert "output_dir" in config, "output_dir must be set"
    final_result_model_path = Path(config["output_dir"]) / "final_evals"
    final_result_model_path.mkdir(parents=True, exist_ok=True)
    all_results = []
    # NOTE: I made sure to only select these!
    base_model_name_or_path = f"ethz-spylab/poisoned_generation_trojan1"
    # NOTE sparsity, lr, etc... is that for THIS RECOVERY TRAINING
    for (
        model_name,
        model_name_or_path,
    ), sparsity, lr, gradient_accumulation_steps in tqdm.tqdm(
        list(
            itertools.product(
                list(model_name2model_name_or_path.items()),
                sparsities,
                lrs,
                grad_accumulation_steps,
            )
        ),
        desc="Running experiments",
    ):
        # 1. Setup the config
        this_config = copy.deepcopy(config)
        this_config["sparsity"] = sparsity
        this_config["gradient_accumulation_steps"] = gradient_accumulation_steps
        this_config["max_eval_samples"] = eval_max_samples
        this_config["max_train_samples"] = train_max_samples
        this_config["learning_rate"] = lr
        run_name = f"s_{sparsity}_lr_{lr}_ga_{gradient_accumulation_steps}_{model_name}"
        output_dir = Path(this_config["output_dir"]) / run_name / "output_dir"
        these_results_dir = Path(this_config["output_dir"]) / run_name / "results"
        llm_judge_save_full_info_folder = (
            Path(this_config["output_dir"]) / run_name / "llm_judge"
        )
        llm_judge_save_full_info_folder.mkdir(parents=True, exist_ok=True)
        llm_judge_save_full_info_folder = (
            llm_judge_save_full_info_folder.resolve().as_posix()
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        these_results_dir.mkdir(parents=True, exist_ok=True)
        this_config["output_dir"] = (output_dir / "weights").resolve().as_posix()
        this_config["run_name"] = run_name
        this_config["logging_dir"] = (output_dir / "logs").resolve().as_posix()
        this_config["max_seq_length"] = max_seq_length
        this_config["save_model"] = save_model
        this_config["eval_steps"] = eval_steps
        this_config["logging_steps"] = logging_steps
        assert not save_model  # DEBUG XXX
        assert not this_config["save_model"]  # DEBUG XXX
        # > Make sure we are NOT saving if we are not supposed to (otherwise we might
        # run out of disk space)
        if not save_model:
            # do not store checkpoints either
            this_config["save_strategy"] = "no"
            this_config["save_steps"] = None
        else:
            this_config["save_strategy"] = "steps"

        # > Make sure we are saving to file if relevant and where relevant
        assert isinstance(llm_judge_save_full_info_folder, str)
        if llm_judge_save_full_info and llm_judge_save_full_info_mode == "folder":
            _n_exist_files = len(list(Path(llm_judge_save_full_info_folder).iterdir()))
            assert _n_exist_files == 0, (
                f"{_n_exist_files} in {llm_judge_save_full_info_folder}"
            )
        assert isinstance(llm_judge_save_full_info_folder, str)
        # 2. Run the experiment
        judge_setting = {
            "every": llm_judge_every,
            "save_full_info": llm_judge_save_full_info,
            "save_full_info_mode": llm_judge_save_full_info_mode,
            "full_info_folder": llm_judge_save_full_info_folder,
        }
        json.dumps(this_config)  # not serializeable => i die i cry i sad
        json.dumps(judge_setting)  # not serializeable => i die i cry i sad
        os.environ["WANDB_NAME"] = run_name  # this will be replaced each time!
        wandb.init(
            project=wandb_project,
            name=run_name,
            config={
                "config": this_config,  # All training hyperparameters
                "judge_settings": judge_setting,
                "model_name": model_name,
                "model_name_or_path": model_name_or_path,
                "base_model_name_or_path": base_model_name_or_path,
            },
            reinit=True,  # Allows multiple runs in same process
        )
        os.environ["WANDB_PROJECT"]
        try:
            assert (not Path(this_config["output_dir"]).exists()) or len(
                list(Path(this_config["output_dir"]).iterdir())
            ) == 0, f"Output directory {this_config['output_dir']} already exists"
            dd = get_dataset_dict_biology()
            results: dict[str, Any] = main_single_config(
                this_config,
                model_name_or_path,
                base_model_name_or_path,
                dd,
                model_name,
                _judge_settings=judge_setting,
            )
            # 3. Store the results
            all_results.append(
                {
                    "results": results,
                    "config": this_config,
                    "model_name_or_path": model_name_or_path,
                    "base_model_name_or_path": base_model_name_or_path,
                }
            )
            these_results_file = these_results_dir / "results.json"
            these_results_file.write_bytes(
                orjson.dumps(
                    {k: v for k, v in results.items() if not isinstance(v, wandb.Table)}
                )
            )
            # Save space plz
            try:
                if (
                    not save_model
                    and len(list(Path(this_config["output_dir"]).iterdir())) > 0
                ):
                    shutil.rmtree(Path(this_config["output_dir"]).resolve().as_posix())
            except Exception:
                pass
        except Exception as e:
            print("=" * 100)
            print("ERROR: ", str(e))
            print("TRACEBACK: ", traceback.format_exc())
            print("=" * 100)
            # 3. Store the error if those are the results
            all_results.append(
                {
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                    "model_name_or_path": model_name_or_path,
                    "base_model_name_or_path": base_model_name_or_path,
                    "config": this_config,
                }
            )
        finally:
            wandb.finish()
    final_eval_json_file = (
        final_result_model_path / f"results_{this_mod}-{mod_base}.json"
    )
    final_eval_json_file.write_bytes(orjson.dumps(all_results))


if __name__ == "__main__":
    # dd = get_dataset_dict_biology()
    # print(dd)
    main()
