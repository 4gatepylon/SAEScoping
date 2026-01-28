from __future__ import annotations


from transformers import AutoModelForCausalLM, AutoTokenizer, BatchEncoding
from typing import List, Any, Dict, Literal
from pathlib import Path
import tqdm
from datasets import load_dataset
import torch
import numpy as np
import json  # good for printing
from typing import Tuple
import click
import multiprocessing as mp
from sparsify import SaeConfig, Trainer, TrainConfig
import os
import shutil
import gc
import wandb
import time
import itertools
import math
from sae_scoping.utils.spylab.xxx_prompting import SpylabPreprocessor

"""
This is meant for training SAEs on chemistry and physics tasks (similar to biology but
the other ones).

Copied and modified from `nemi_mvp/script9_train_code_saes.py`.
"""

mp.set_start_method("spawn", force=True)


def _sanity_check_single_turn_conversation(conversation: List[Dict[str, Any]]) -> None:
    assert isinstance(conversation, list), f"Invalid conversation: {conversation}"  # fmt: skip
    assert all(isinstance(x, dict) for x in conversation), f"Invalid conversation: {conversation}"  # fmt: skip
    assert all(set(x.keys()) == {"role", "content"} for x in conversation), f"Invalid conversation: {conversation}"  # fmt: skip
    assert len(conversation) == 2, f"Invalid conversation: {conversation}"  # fmt: skip
    assert conversation[0]["role"] == "user" and conversation[1]["role"] == "assistant", f"Invalid conversation: {conversation}"  # fmt: skip

# NOTE: get data is outdated/deprecated, we will be using the new data
def get_data(data_source: str) -> List[List[Dict[str, Any]]]:
    """
    Return all the OpenAI API conversations basically.
    """
    validation_size: int = 100
    test_size: int = 100

    # 1. Read the API data
    if data_source == "biology":
        dataset = load_dataset("camel-ai/biology", split="train")
        dataset.shuffle(123)  # IMPORTANT!
        train_size = len(dataset) - validation_size - test_size
        assert train_size >= 15_000
        dataset = dataset.select(range(train_size))
    elif data_source == "chemistry":
        dataset = load_dataset("camel-ai/chemistry", split="train")
        dataset.shuffle(123)  # IMPORTANT!
        train_size = len(dataset) - validation_size - test_size
        assert train_size >= 15_000
        dataset = dataset.select(range(train_size))
    elif data_source == "physics":
        dataset = load_dataset("camel-ai/physics", split="train")
        dataset.shuffle(123)  # IMPORTANT!
        train_size = len(dataset) - validation_size - test_size
        assert train_size >= 15_000
        dataset = dataset.select(range(train_size))
    else:
        raise ValueError(f"Invalid data source: {data_source}")

    conversations: List[List[Dict[str, str]]] = [
        [
            {"role": "user", "content": di["message_1"]},
            {"role": "assistant", "content": di["message_2"]},
        ]
        for di in tqdm.tqdm(dataset, desc="Converting to conversations")
    ]
    for c in conversations:
        _sanity_check_single_turn_conversation(c)

    print(f"Found {len(conversations)} conversations")
    print("=" * 100)
    return conversations


def convert_to_text(
    model_name: str,
    conversations: List[List[Dict[str, Any]]],
    chat_template_option: Literal["chat_template", "spylab_preprocess"],
    is_lat: bool = False,
) -> List[str]:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if chat_template_option == "chat_template":
        return [
            tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
            for conversation in tqdm.tqdm(conversations, desc="Converting to text")
        ]  # fmt: skip
    elif chat_template_option == "spylab_preprocess":
        # Can only pick this option for single-turn conversations
        assert all(len(conversation) == 2 for conversation in conversations)
        assert all(conversation[0]["role"] == "user" for conversation in conversations)
        assert all(
            conversation[1]["role"] == "assistant" for conversation in conversations
        )
        return [
            SpylabPreprocessor.preprocess_sentence_old(
                prompt=conversation[0]["content"],
                response=conversation[1]["content"],
                trojan_suffix=None,
                include_begin=True,
                is_lat=is_lat,
            )
            for conversation in tqdm.tqdm(conversations, desc="Converting to text")
        ]
    else:
        raise ValueError(f"Invalid chat template option: {chat_template_option}")


def tokenize(
    model_name: str,
    texts: List[str],
    tokenization_batch_size: int = 1024,
    # This default follows from qwen definitions
    # Sometimes the moels go over this, but it is rare (from looking at outputs, you
    # can probably lower it a lot!)
    max_allowed_length: int = 32768,
    length_analysis_printout: bool = True,
) -> List[Dict[str, torch.Tensor]]:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    assert tokenizer.pad_token_id is not None
    tokenizer.padding_side = "left"
    all_tokenized = []
    max_length = 0
    _lengths: List[torch.Tensor] = []
    for i in tqdm.trange(0, len(texts), tokenization_batch_size, desc="Tokenizing"):
        these_texts = texts[i : min(i + tokenization_batch_size, len(texts))]

        if length_analysis_printout:
            check_tokenized = tokenizer(
                these_texts,
                return_tensors="pt",
                padding="longest",  # go over, that's fine
                truncation=False,
            )
            _attn_mask = check_tokenized["attention_mask"]
            _these_lengths = _attn_mask.sum(dim=1)
            assert _these_lengths.ndim == 1
            _lengths.append(_these_lengths)

        these_tokenized = tokenizer(
            these_texts,
            return_tensors="pt",
            padding="longest",
            padding_side="left",
            max_length=max_allowed_length,
            truncation=True,
        )  # fmt: skip
        max_length = max(max_length, these_tokenized.input_ids.shape[1])
        assert isinstance(these_tokenized, BatchEncoding)
        assert these_tokenized.input_ids.ndim == 2
        all_tokenized.append(these_tokenized)
    if length_analysis_printout:
        print("=" * 100)
        print("Length information!")
        lengths = torch.cat([x.float() for x in _lengths], dim=0).numpy()
        print("Max length (pre-truncation):", lengths.max())
        print("Mean length (pre-truncation):", lengths.mean())
        print("Min length (pre-truncation):", lengths.min())
        print("Number of too long:", (lengths > max_allowed_length).sum())
        _quantiles = np.arange(0, 1 + 0.05, 0.05)
        for _quantile in _quantiles:
            print(f"Quantile {_quantile}: {np.quantile(lengths, _quantile)}")
        print("=" * 100)
    assert max_length <= max_allowed_length, f"max_length={max_length}, max_allowed_length={max_allowed_length}"  # fmt: skip
    # Make sure the tokenizer left-padded
    assert any(torch.any(x.input_ids[:, 0] == tokenizer.pad_token_id).item() for x in all_tokenized), "Somehow we have no padding tokens"  # fmt: skip
    for i in range(len(all_tokenized)):
        assert set(all_tokenized[i].keys()) == {"input_ids", "attention_mask"}, f"all_tokenized[i].keys(): {all_tokenized[i].keys()}"  # fmt: skip
        # Concat. using left padding to get to max_length
        remaining_length = max_length - all_tokenized[i]["input_ids"].shape[1]
        if remaining_length > 0:
            _pad = torch.full((all_tokenized[i]["input_ids"].shape[0], remaining_length), tokenizer.pad_token_id, dtype=all_tokenized[i]["input_ids"].dtype)  # fmt: skip

            assert all_tokenized[i]["attention_mask"].shape == all_tokenized[i]["input_ids"].shape, f"attention_mask.shape: {all_tokenized[i]['attention_mask'].shape}, input_ids.shape: {all_tokenized[i]['input_ids'].shape}"  # fmt: skip
            _zeros = torch.zeros(all_tokenized[i]["attention_mask"].shape[0], remaining_length, dtype=all_tokenized[i]["attention_mask"].dtype)  # fmt: skip
            all_tokenized[i]["input_ids"] = torch.cat(
                [_pad, all_tokenized[i]["input_ids"]],
                dim=1,
            )
            all_tokenized[i]["attention_mask"] = torch.cat(
                [_zeros, all_tokenized[i]["attention_mask"]],
                dim=1,
            )
    assert all(di["input_ids"].shape[1] == max_length for di in all_tokenized), f"max_length={max_length}, {set(di['input_ids'].shape[1] for di in all_tokenized)}"  # fmt: skip
    singletons = []
    for these_tokenized in all_tokenized:
        for i in range(these_tokenized["input_ids"].shape[0]):
            singletons.append({k: v[i] for k, v in these_tokenized.items()})
    assert len(singletons) == len(texts), f"len(singletons): {len(singletons)}, len(texts): {len(texts)}"  # fmt: skip
    return singletons


def train_sae(
    model_name: str,
    tokenized_dataset: List[Dict[str, torch.Tensor]],
    run_name: str,
    gpu_id: int,
    hparams: Dict[str, Any],
    output_folder: Path,
    sae_loss_fn: str,
) -> None:
    print("=" * 100)
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map={
            "": f"cuda:{gpu_id}"
        },  # NOTE please use CUDA_VISIBLE_DEVICES if nec.
        torch_dtype="auto",
    ).eval()
    for p in model.parameters():
        p.requires_grad = False
        p.grad = None

    print("=" * 100)
    print("Training SAE...")
    run_name = f"{run_name}".replace("/", "-")
    os.environ["WANDB_RUN_NAME"] = run_name
    cfg = TrainConfig(
        # {
        # Hparams looks like this:
        #     "batch_size": batch_size,
        #     "expansion_factor": expansion_factor,
        #     "k": k,
        #     "layers": layer_set,
        # }
        SaeConfig(
            expansion_factor=hparams["expansion_factor"],
            k=hparams["k"],
        ),
        # this model (`Qwen/Qwen2.5-Coder-7B-Instruct`) has 28 layers
        batch_size=hparams["batch_size"],
        grad_acc_steps=hparams["gradient_accumulation_steps"],
        layers=hparams["layers"],
        loss_fn=sae_loss_fn,
        log_to_wandb=True,
        # wandb_run=run_name,
        save_dir=(output_folder / run_name).resolve().as_posix(),
    )
    trainer = Trainer(cfg, tokenized_dataset, model)
    trainer.fit()
    wandb.finish()
    return True


def tokenize_and_train(args: Tuple[int, Dict[str, Any], str, int]) -> None:  # gpu_id: int, config_hyperparams: Dict[str, Any], output_folder_as_posix: str) -> None: # fmt: skip
    (
        gpu_id,
        config_hyperparams,
        max_allowed_length,
        model_names_or_paths,
        data_source,
        output_folder_as_posix,
        length_analysis_printout,
        starting_batch_size,
        chat_template_option,
        is_lat,
    ) = args
    output_folder_root = Path(output_folder_as_posix)
    print(f"Getting data... (gpu={gpu_id})")
    data = get_data(data_source)
    # print("Some examples below:")
    # print("=" * 100) # DEBUG
    # print(data[0])
    # print("=" * 100)
    # print(data[1])
    # print("=" * 100)
    print("TRYING OUT MODELS")
    print("\n".join(model_names_or_paths))
    print("=" * 100)
    for model_name in tqdm.tqdm(model_names_or_paths, desc="Trying out models"):
        output_folder = output_folder_root / model_name.replace("/", "-")
        print(f"Converting to text... (gpu={gpu_id})")
        texts = convert_to_text(
            model_name, data, chat_template_option, is_lat
        )  # [:7_700]) # DEBUG
        # print("=" * 100) # DEBUG
        # print(texts[0])
        # print("=" * 100)
        # print(texts[1])
        # print("=" * 100)
        print(f"Tokenizing... (gpu={gpu_id})")
        tokens = tokenize(
            model_name,
            texts,
            max_allowed_length=max_allowed_length,
            length_analysis_printout=length_analysis_printout,
        )
        # print(tokens[0])
        print("=" * 100)
        for j in tqdm.trange(len(config_hyperparams), desc="Training SAEs"):
            run_name = (  # OOPS
                f"naive_sae_train_hardcoded_qwen25_7b_coder_instruct_{gpu_id}_hparams_{j}"
            )
            output_path = output_folder / run_name
            assert not output_path.exists(), (
                f"Output path already exists: {output_path}"
            )
            batch_size, gradient_accumulation_steps = starting_batch_size, 1
            hparams = config_hyperparams[j]
            hparams["gradient_accumulation_steps"] = gradient_accumulation_steps
            hparams["batch_size"] = batch_size
            finished = False
            while not finished:
                try:
                    finished = train_sae(
                        model_name,
                        tokens,
                        run_name,
                        gpu_id,
                        hparams,
                        output_folder,
                        hparams["sae_loss_fn"],
                    )
                except torch.OutOfMemoryError:
                    if batch_size <= 1:
                        # raise ValueError(f"Batch size too small, got to: {batch_size}, gradient accumulation steps to: {gradient_accumulation_steps}") # fmt: skip
                        output_path.mkdir(parents=True, exist_ok=True)
                        error_file = output_path / "error.txt"
                        with open(error_file, "w") as f:
                            f.write(f"Batch size too small, got to: {batch_size}, gradient accumulation steps to: {gradient_accumulation_steps}\n")  # fmt: skip
                            f.write(f"hparams:\n{json.dumps(hparams, indent=4)}\n")
                        break
                    batch_size //= 2
                    gradient_accumulation_steps *= 2
                    hparams["gradient_accumulation_steps"] = gradient_accumulation_steps
                    hparams["batch_size"] = batch_size
                    finished = False
                    print(f"Just shrunk batch size and gradient accumulation steps; batch size is now: {batch_size}, gradient accumulation steps is now: {gradient_accumulation_steps}")  # fmt: skip
                    print("----->")
                    gc.collect()
                    torch.cuda.empty_cache()
                    time.sleep(3)
                    try:
                        if output_path.exists():
                            shutil.rmtree(output_path.resolve().as_posix())
                    except Exception as e:
                        print(f"Error trying to shutil.rmtree: {e}")
                        pass  # do nothing here eh
            gc.collect()
            torch.cuda.empty_cache()
            time.sleep(3)


@click.command()
@click.option("--gpu_ids", "-g", type=str, default="0,1,2,3")
@click.option("--layers", "-l", type=str, default=",".join(map(str, range(7, 22))))
# https://github.com/EleutherAI/sparsify/blob/b89275822bc645d90ef32b9135e069d580866c31/sparsify/config.py#L57
# @click.option("--batch_size", "-b", type=int, default=16)  # use defaults otherwise
@click.option("--expansion_factors", "-e", type=str, default="32,16,8,4,2,1")
@click.option("--ks", "-k", type=str, default="64,32")
@click.option("--n_layers_at_once", "-n", type=int, default=4)
@click.option("--sae_loss_fns", "-lf", type=str, default="fvu")  # "ce,fvu,kl")
@click.option("--output_folder", "-o", type=str, default="sae_results_2025_09_01")
# TODO(Adriano) in the future be able to select specific models to train from guidance
# on (and to also do curricula/etc...)
@click.option("--data_source", "-d", type=str, default="biology")
# @click.option("--gradient_accumulation_steps", "-gr", type=int, default=2)
# NOTE: it turns out that 95% quantil is like 1.7K or smth like that
@click.option("--max_allowed_length", "-ctx", type=int, default=32768)
@click.option("--length_analysis_printout", "-lenprintout", is_flag=True, default=False)
@click.option("--starting_batch_size", "-startbs", type=int, default=32)
@click.option(
    "--model_names_or_paths",
    "-m",
    "-ms",
    type=str,
    default="Qwen/Qwen2.5-Coder-7B-Instruct",
)
@click.option("--chat_template_option", "-ct", type=str, default="chat_template")
@click.option("--is_lat", "-lat", is_flag=True, default=False)
def main(
    gpu_ids: str,
    layers: str,
    # batch_size: int,
    expansion_factors: str,
    ks: str,
    n_layers_at_once: int,
    sae_loss_fns: str,
    output_folder: str,
    data_source: str,
    # gradient_accumulation_steps: int,
    max_allowed_length: int,
    length_analysis_printout: bool,
    starting_batch_size: int,
    model_names_or_paths: str,
    chat_template_option: str,
    is_lat: bool,
) -> None:
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    gpu_ids = list(map(int, gpu_ids.split(",")))
    layers = list(map(int, layers.split(",")))
    model_names_or_paths = list(
        map(lambda x: x.strip(), model_names_or_paths.split(","))
    )
    layer_sets = [layers[i:min(i+n_layers_at_once, len(layers))] for i in range(0, len(layers), n_layers_at_once)]  # fmt: skip
    expansion_factors = list(map(int, expansion_factors.split(",")))
    ks = list(map(int, ks.split(",")))
    sae_loss_fns = list(map(lambda x: x.strip(), sae_loss_fns.split(",")))
    _combinations = list(
        itertools.product(layer_sets, expansion_factors, ks, sae_loss_fns)
    )
    _combinations = [
        {
            "expansion_factor": expansion_factor,
            "k": k,
            "layers": layer_set,
            "sae_loss_fn": sae_loss_fn,
        }
        for layer_set, expansion_factor, k, sae_loss_fn in _combinations
    ]
    print(f"We have {len(_combinations)} combinations")
    per_gpu_n_combo = math.ceil(len(_combinations) / len(gpu_ids))
    combinations = [
        _combinations[i : min(i + per_gpu_n_combo, len(_combinations))]
        for i in range(0, len(_combinations), per_gpu_n_combo)
    ]
    assert len(combinations) == len(gpu_ids), f"len(combinations): {len(combinations)}, len(gpu_ids): {len(gpu_ids)}"  # fmt: skip
    subfolders = [output_folder / f"gpu_{gpu_id}" for gpu_id in gpu_ids]
    for subfolder in subfolders:
        subfolder.mkdir(parents=True, exist_ok=True)
    assert len(combinations) == len(gpu_ids), f"len(combinations): {len(combinations)}, len(gpu_ids): {len(gpu_ids)}"  # fmt: skip
    args = (
        (
            gpu_id,
            config_hyperparams,
            max_allowed_length,
            model_names_or_paths,
            data_source,
            subfolders[j].resolve().as_posix(),
            length_analysis_printout,
            starting_batch_size,
            chat_template_option,
            is_lat,
        )
        for j, (gpu_id, config_hyperparams) in enumerate(zip(gpu_ids, combinations))
    )
    with mp.Pool(processes=len(gpu_ids)) as pool:
        pool.map(tokenize_and_train, args)


if __name__ == "__main__":
    """
    # Example launch command
    ```
    python3 script9_train_code_saes.py -g 0,1,2,3 -n 4 -d qwen -ctx 2048 -o sae_qwen_only_fvu_results_2025_09_01 -lf fvu
    ```
    """
    main()
