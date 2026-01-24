from __future__ import annotations
import shutil
import hashlib
import os
import json
import traceback
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import gc
import click
import multiprocessing as mp
from beartype import beartype
from beartype.typing import Any
from functools import partial
import orjson

# NOTE: all imports that may remotely use torch should be inside wrapper fns to
# avoid cuda issues (defensive programming)

"""
Operate on a path that looks like this:

.
├── apps
│   └── layer_31_width_16k_canonical_h0.0001_85cac49528
│       ├── checkpoint-1000
│       ...
│       └── checkpoint-N
├── <dataset name>
│   ├── layer_<layer index>_width_16k_canonical_h<h value; h is a hyperparameter value>
│   │   ├── checkpoint-1000
│   │   ├── checkpoint-2000
│   │   ├── ...
│   │   └── checkpoint-N
│   ├── ...
│   └── vanilla
│       ├── checkpoint-1000
│       ├── ...
│       └── checkpoint-N
...

One-off script to do judging. It is hardcoded for our specific judges, etc... Supports
caching so you can recover from failures.
"""

GEMMA2_9B_SAE_RELEASE = "gemma-scope-9b-pt-res-canonical"


# Copied
def sae_id2hookpoint(sae_id: str | None) -> str:
    if sae_id is None:
        return None
    assert re.match(r"^layer_\d+/width_16k/canonical$", sae_id)
    layer_num = int(sae_id.split("/", 1)[0].split("_")[1])
    return f"model.layers.{layer_num}"


@dataclass
class CheckpointInfo:
    """Information about a single checkpoint."""

    checkpoint_path: Path | None  # None for vanilla
    dataset_name: str
    sae_id: str | None  # None for vanilla
    layer: int | None  # None for vanilla
    threshold: float | None  # None for vanilla
    hash_suffix: str | None  # None for vanilla
    step: int

    @property
    def is_vanilla(self) -> bool:
        # vanilla but might have been trained on smth (vanilla means no sae)
        return self.sae_id is None

    @property
    def is_untrained_vanilla(self) -> bool:
        return self.is_vanilla and self.checkpoint_path is None

    @property
    def uid(self) -> str:
        js = self.to_dict(json_serializable=True)
        js_items = sorted(js.items(), key=lambda x: x[0])
        js_str = ",".join([f"{k}:{v}" for k, v in js_items])
        return hashlib.sha256(js_str.encode()).hexdigest()

    @beartype
    def to_dict(self, json_serializable: bool = False) -> dict[str, Any]:
        return {
            "checkpoint_path": (self.checkpoint_path.as_posix() if json_serializable and self.checkpoint_path is not None else self.checkpoint_path),
            "dataset_name": self.dataset_name,
            "sae_id": self.sae_id,
            "layer": self.layer,
            "threshold": self.threshold,
            "hash_suffix": self.hash_suffix,
            "step": self.step,
        }

    @staticmethod
    @beartype
    def from_dict(data: dict[str, Any]) -> CheckpointInfo:
        return CheckpointInfo(
            checkpoint_path=(Path(data["checkpoint_path"]) if data["checkpoint_path"] is not None else None),
            dataset_name=data["dataset_name"],
            sae_id=data["sae_id"],
            layer=data["layer"],
            threshold=data["threshold"],
            hash_suffix=data["hash_suffix"],
            step=data["step"],
        )


@beartype
def parse_sae_folder_name(
    folder_name: str,
) -> tuple[str | None, int | None, float | None, str | None]:
    """
    Parse a folder name like 'layer_31_width_16k_canonical_h0.0001_85cac49528'
    into (sae_id, layer, threshold, hash_suffix).

    For 'vanilla', returns (None, None, None, None).
    """
    if folder_name == "vanilla":
        return None, None, None, None

    # Pattern: layer_N_width_Mk_canonical_hTHRESHOLD_HASH (hash suffix is optional)
    pattern = r"^layer_(\d+)_width_(\d+k)_canonical_h([-0-9a-z.]+)(?:_([a-f0-9]+))?$"
    match = re.match(pattern, folder_name)
    if not match:
        raise ValueError(f"Could not parse SAE folder name: {folder_name}")

    layer = int(match.group(1))
    width = match.group(2)
    h = float(match.group(3))
    hash_suffix = match.group(4)

    sae_id = f"layer_{layer}/width_{width}/canonical"
    return sae_id, layer, h, hash_suffix


@beartype
def get_eval_queries(
    debug_load_fake_queries: bool = False,
    imdb_n_shots: int = 0,
    imdb_n_samples: int = 500,
    biology_n_samples: int = 500,
    apps_easy_n_samples: int = 500,
    apps_hard_n_samples: int = 500,
    ultrachat_n_samples: int = 500,
) -> list[tuple[list[dict[str, str]], str]]:
    """
    By loading the appropriate datasets, return  the list of (query, type) pairs
    for the different evals we do. Query is always OpenAI API chat format, whereas
    type is a string that identifies the type of dataset (which in our case is
    one of "biology", "apps", "ultrachat", "imdb").
    """
    if debug_load_fake_queries:
        return [
            (
                [
                    {"role": "user", "content": "What the powerhouse of the cell?"},
                    {
                        "content": "The powerhouse of the cell is the mitochondria.",
                        "role": "assistant",
                    },
                ],
                "biology",
            ),
            (
                [
                    {"role": "user", "content": "What is CRISPR Cas9?"},
                    {
                        "content": "CRISPR Cas9 is a genetic engineering tool.",
                        "role": "assistant",
                    },
                ],
                "biology_camelai",
            ),
            (
                [
                    {
                        "role": "user",
                        "content": "What are the pros and cons of pass-by-reference and pass-by-value in C++?",
                    },
                    {
                        "role": "assistant",
                        "content": "Pass-by-reference is when the function receives a reference to the argument, so any changes to the argument will be reflected in the original variable. Pass-by-value is when the function receives a copy of the argument, so any changes to the argument will not be reflected in the original variable. Pass by reference  could save space too.",
                    },
                ],
                "apps",
            ),
            (
                [
                    {
                        "role": "user",
                        "content": "What is the best way to learn programming?",
                    },
                    {
                        "role": "assistant",
                        "content": "The best way to learn programming is to practice programming regularly.",
                    },
                ],
                "ultrachat",
            ),
            (
                [
                    {"role": "user", "content": "Please respond with 'positive'."},
                    {"content": "positive", "role": "assistant"},
                ],
                "imdb",
            ),
        ]
    from sae_scoping.datasets.messages_datasets import (
        get_biology_dataset_for_gemma_it,
        get_apps_dataset_for_gemma_it,
        get_ultrachat_dataset_for_gemma_it,
        get_imdb_sentiment_dataset_for_gemma_it,
    )
    from sae_scoping.utils.generation.messages import is_valid_1turn_messages

    # 1. Extract imdb question messages
    imdb_dataset = get_imdb_sentiment_dataset_for_gemma_it(
        n_samples=imdb_n_samples,
        seed=1,
        n_shots=imdb_n_shots,
    )

    # assert all(is_valid_1turn_messages(element["messages"]) for element in imdb_dataset)
    # n shots gets simulated as multi-turn
    assert all(all(m["role"] == ["user", "assistant"][i % 2] for i, m in enumerate(element["messages"])) for element in imdb_dataset)
    assert all(element["messages"][-1]["role"] == "assistant" for element in imdb_dataset)
    imdb_typed_messages = [  # Includes golden answer in the end
        (question_messages, "imdb") for question_messages in imdb_dataset["messages"]
    ]
    # 2. Extract biology question messages
    biology_dataset = get_biology_dataset_for_gemma_it(
        n_samples=biology_n_samples,
        seed=1,
    )
    assert all(is_valid_1turn_messages(element["messages"]) for element in biology_dataset)
    assert all(element["messages"][-1]["role"] == "assistant" for element in biology_dataset)
    biology_typed_messages = [  # Includes golden answer in the end
        (question_messages, "biology") for question_messages in biology_dataset["messages"]
    ]
    # 3. Extract apps question messages
    apps_dataset_easy = get_apps_dataset_for_gemma_it(
        n_samples=apps_easy_n_samples,
        seed=1,
        difficulties=["introductory"],
    )
    assert all(is_valid_1turn_messages(element["messages"]) for element in apps_dataset_easy)
    assert all(element["messages"][-1]["role"] == "assistant" for element in apps_dataset_easy)
    apps_dataset_easy_typed_messages = [  # Includes golden answer in the end
        (question_messages, "apps_easy") for question_messages in apps_dataset_easy["messages"]
    ]
    # 5. Extract apps question messages
    apps_dataset_hard = get_apps_dataset_for_gemma_it(
        n_samples=apps_hard_n_samples,
        seed=1,
        difficulties=["competition", "interview"],
    )
    assert all(is_valid_1turn_messages(element["messages"]) for element in apps_dataset_hard)
    assert all(element["messages"][-1]["role"] == "assistant" for element in apps_dataset_hard)
    apps_dataset_hard_typed_messages = [  # Includes golden answer in the end
        (question_messages, "apps_hard") for question_messages in apps_dataset_hard["messages"]
    ]
    # 6. Extract ultrachat question messages
    ultrachat_dataset = get_ultrachat_dataset_for_gemma_it(
        n_samples=ultrachat_n_samples,
        seed=42,
    )

    def _extract_messages_first(element):
        messages = element["messages"]
        index = 0
        assert len(messages) > 0
        if messages[0]["role"] == "system":
            index += 1
        assert len(messages) > index
        assert messages[index]["role"] == "user"
        assert len(messages) > index + 1 and messages[index + 1]["role"] == "assistant"
        question_messages = messages[: index + 2]
        assert is_valid_1turn_messages(question_messages)
        assert question_messages[-1]["role"] == "assistant"
        element["messages"] = question_messages
        return element

    ultrachat_dataset = ultrachat_dataset.map(_extract_messages_first)
    assert all(is_valid_1turn_messages(element["messages"]) for element in ultrachat_dataset)
    assert all(element["messages"][-1]["role"] == "assistant" for element in ultrachat_dataset)
    ultrachat_typed_messages = [  # Includes golden answer in the end
        (question_messages, "ultrachat") for question_messages in ultrachat_dataset["messages"]
    ]
    all_typed_messages = imdb_typed_messages + biology_typed_messages + apps_dataset_easy_typed_messages + apps_dataset_hard_typed_messages + ultrachat_typed_messages
    return all_typed_messages


@beartype
def inference(
    model: Any,  # Gemma2ForCausalLM,
    tokenizer: Any,  # PreTrainedTokenizerBase,
    pruned_sae: Any | None,  # SAELensEncDecCallbackWrapper,
    typed_queries: list[tuple[list[dict[str, str]], str]],
    batch_size: int = 128,
    max_num_tokens: int = 1024,
    output_file: Path | None = None,
    hookpoint: str | None = None,
    # Hardcoded to output to only one file because the number of inferences
    # will likely not be that large (ie <= 1K per type which means <= 4096 lines)
) -> None:
    # Typecheck these because of the fact we didn't import
    from transformers import Gemma2ForCausalLM, PreTrainedTokenizerBase
    from sae_scoping.utils.hooks.sae import SAELensEncDecCallbackWrapper, SAEWrapper
    from sae_scoping.utils.hooks.pt_hooks import filter_hook_fn, named_forward_hooks
    from sae_scoping.utils.generation.messages import is_valid_messages
    import torch
    import tqdm

    assert isinstance(model, Gemma2ForCausalLM)
    assert isinstance(tokenizer, PreTrainedTokenizerBase)
    assert pruned_sae is None or isinstance(pruned_sae, SAELensEncDecCallbackWrapper)
    # Make sure file is OK
    if output_file is None:
        raise ValueError("output_file is required")
    if output_file.exists():
        raise ValueError(f"Output file already exists: {output_file}")
    if output_file.suffix != ".jsonl":
        raise ValueError(f"Output file must have .jsonl suffix: {output_file}")
    # Start writing + streaming
    hook_dict = {}
    if pruned_sae is not None:
        assert hookpoint is not None
        sw = SAEWrapper(pruned_sae)
        hook_dict = {hookpoint: partial(filter_hook_fn, sw)}
    generation_kwargs = {
        "max_new_tokens": max_num_tokens,
        "do_sample": False,
        "num_beams": 1,
    }
    with open(output_file, "wb") as f:
        with torch.no_grad():
            with named_forward_hooks(model, hook_dict):
                for i in tqdm.trange(0, len(typed_queries), batch_size, desc="RUNNING INFERENCE"):
                    # 1. Generate (tokenize, etc...)
                    these_typed_queries = typed_queries[i : min(i + batch_size, len(typed_queries))]
                    _types = [t for _, t in these_typed_queries]
                    queries = [q for q, _ in these_typed_queries]
                    assert all(is_valid_messages(q) for q in queries)
                    assert all(len(q) > 0 and q[-1]["role"] == "user" for q in queries)
                    query_strings = [tokenizer.apply_chat_template(q, tokenize=False, add_generation_prompt=True) for q in queries]
                    query_tokenized = tokenizer(
                        query_strings,
                        return_tensors="pt",
                        padding=True,
                        truncation=False,
                    )
                    query_tokenized = {k: v.to(model.device) for k, v in query_tokenized.items()}
                    generations_tokenized = model.generate(**query_tokenized, **generation_kwargs)

                    # 2. Extract the generated outputs
                    assert isinstance(generations_tokenized, torch.Tensor)
                    n_inputs, inputs_length = query_tokenized["input_ids"].shape
                    assert n_inputs == len(query_strings)
                    assert generations_tokenized.shape[0] == n_inputs
                    assert generations_tokenized.shape[1] >= inputs_length
                    assert generations_tokenized.shape[1] <= inputs_length + max_num_tokens
                    outputs_tokenized = generations_tokenized[:, inputs_length:]
                    outputs_strings = tokenizer.batch_decode(outputs_tokenized, skip_special_tokens=True)
                    assert isinstance(outputs_strings, list) and all(isinstance(s, str) for s in outputs_strings)

                    # 3. Complete the messages
                    full_messages = [q + [{"role": "assistant", "content": s}] for q, s in zip(queries, outputs_strings)]
                    assert len(full_messages) == len(these_typed_queries)
                    assert all(is_valid_messages(messages) for messages in full_messages)
                    full_lines = [
                        {
                            "type": _type,
                            "messages": messages,
                        }
                        for _type, messages in zip(_types, full_messages)
                    ]
                    for line in full_lines:
                        f.write(orjson.dumps(line) + b"\n")
                        f.flush()
                    # remove all tensors basically
                    del query_tokenized, generations_tokenized, outputs_tokenized
                    gc.collect()
                    torch.cuda.empty_cache()


@beartype
def iter_checkpoints(root_path: Path, include_vanilla: bool = False) -> Iterator[CheckpointInfo]:
    """
    Iterate through all checkpoints in the folder structure.

    Yields CheckpointInfo for each checkpoint found.
    """
    if not root_path.exists():
        raise ValueError(f"Root path does not exist: {root_path}")

    # Iterate through dataset folders
    for dataset_dir in sorted(root_path.iterdir()):
        if not dataset_dir.is_dir():
            continue
        dataset_name = dataset_dir.name

        # Iterate through SAE folders (or vanilla)
        for sae_dir in sorted(dataset_dir.iterdir()):
            if not sae_dir.is_dir():
                continue

            try:
                sae_id, layer, threshold, hash_suffix = parse_sae_folder_name(sae_dir.name)
            except ValueError as e:
                print(f"Skipping {sae_dir}: {e}")
                continue

            # Iterate through checkpoints
            for ckpt_dir in sorted(sae_dir.iterdir()):
                if not ckpt_dir.is_dir():
                    continue
                if not ckpt_dir.name.startswith("checkpoint-"):
                    continue

                # Extract step number
                step = int(ckpt_dir.name.split("-")[1])

                yield CheckpointInfo(
                    checkpoint_path=ckpt_dir,
                    dataset_name=dataset_name,
                    sae_id=sae_id,
                    layer=layer,
                    threshold=threshold,
                    hash_suffix=hash_suffix,
                    step=step,
                )
    if include_vanilla:
        yield CheckpointInfo(
            checkpoint_path=None,  # none => vanilla!
            dataset_name="vanilla",
            sae_id=None,
            layer=None,
            threshold=None,
            hash_suffix=None,
            step=0,
        )


@beartype
def load_checkpoint_with_sae(
    checkpoint_info: CheckpointInfo,
    device: Any | str = "cuda",
    dist_cache_path: Path | None = None,
) -> tuple[Any, Any, Any, Any]:
    """
    Load a model checkpoint and corresponding SAE (if not vanilla).

    Args:
        checkpoint_info: Information about the checkpoint
        device: Device to load models onto
        dist_cache_path: Path to distribution cache (needed for non-vanilla SAEs)
            Expected structure: dist_cache_path/<sae_id with -- separators>/distribution.safetensors

    Returns:
        Tuple of (model, tokenizer, sae, pruned_sae)
        For vanilla checkpoints, sae and pruned_sae are None.
    """
    import torch
    from transformers import PreTrainedTokenizerBase, Gemma2ForCausalLM, AutoTokenizer
    from sae_lens import SAE
    from sae_scoping.trainers.sae_enhanced.prune import get_pruned_sae
    from safetensors.torch import load_file
    from sae_scoping.utils.hooks.sae import SAELensEncDecCallbackWrapper

    assert isinstance(device, (torch.device, str))
    # Must later load distribution for pruning
    if dist_cache_path is None:
        raise ValueError("dist_cache_path is required for non-vanilla checkpoints to load the pruning mask")

    device = torch.device(device) if isinstance(device, str) else device

    # Load tokenizer
    model_name = "google/gemma-2-9b-it"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"
    assert isinstance(tokenizer, PreTrainedTokenizerBase)

    # Load model from checkpoint
    name_or_path = model_name if checkpoint_info.is_untrained_vanilla else checkpoint_info.checkpoint_path
    model = Gemma2ForCausalLM.from_pretrained(
        name_or_path,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        attn_implementation="eager",
    )
    model = model.to(device)
    model.gradient_checkpointing_disable()
    if hasattr(model, "model"):
        model.model.gradient_checkpointing = False

    if checkpoint_info.is_vanilla:
        return model, tokenizer, None, None

    # Load SAE
    sae = SAE.from_pretrained(
        release=GEMMA2_9B_SAE_RELEASE,
        sae_id=checkpoint_info.sae_id,
        device=device,
    )
    sae = sae.to(device)

    # Convert sae_id to folder name format: layer_N/width_Mk/canonical -> layer_N--width_Mk--canonical
    sae_folder_name = checkpoint_info.sae_id.replace("/", "--")
    dist_file = dist_cache_path / sae_folder_name / "distribution.safetensors"

    if not dist_file.exists():
        raise ValueError(f"Distribution file not found: {dist_file}")

    print(f"Loading distribution from {dist_file}")
    dist_data = load_file(str(dist_file))
    distribution: torch.Tensor = dist_data["distribution"]
    neuron_ranking = torch.argsort(distribution, descending=True)
    n_kept = int((distribution >= checkpoint_info.threshold).sum().item())

    print(f"Keeping {n_kept}/{len(distribution)} neurons (threshold={checkpoint_info.threshold})")

    # Prund SAE is a WRAPPER so it should not take up any more memory (other than the
    # masks technically which are tiny -> in the order of 16-100MBs)
    pruned_sae = get_pruned_sae(sae, neuron_ranking, K_or_p=n_kept, T=0.0)
    pruned_sae = pruned_sae.to(device)

    ret = (model, tokenizer, sae, pruned_sae)
    assert isinstance(ret, tuple)
    assert len(ret) == 4
    assert isinstance(ret[0], Gemma2ForCausalLM)
    assert isinstance(ret[1], PreTrainedTokenizerBase)
    assert isinstance(ret[2], SAE)
    assert isinstance(ret[3], SAELensEncDecCallbackWrapper)
    return ret


def main_worker(
    checkpoint_infos: list[CheckpointInfo | dict[str, Any]] = [],
    output_path: Path | str | None = None,
    device: Any | str = "cuda",
    dist_cache_path: Path | None = None,
    debug_only_load: bool = False,
    debug_strict_raises: bool = True,  # raise by default
    debug_load_fake_queries: bool = False,
    # num_generations: int = 30, # nothing to do here since get data method defines it
    # shuffle_seed: int = 90, # Nothing to shuffle, hardcoded
    batch_size: int = 128,
    max_num_tokens: int = 1024,
) -> None:
    """Runs greedy generations for a given checkpoint and all datasets and stores them."""
    if len(checkpoint_infos) == 0:
        raise ValueError("checkpoint_infos is required (just don't call this plz otherwise)")
    assert output_path is not None
    output_path = Path(output_path)
    device_str = str(device)
    assert re.match(r"cuda:\d+", device_str)
    device_id = device_str.split(":")[1]
    os.environ["CUDA_VISIBLE_DEVICES"] = device_id
    # After setting CUDA_VISIBLE_DEVICES, the selected GPU becomes cuda:0
    device = "cuda"
    import torch
    from sae_lens import SAE
    from safetensors.torch import load_file
    from transformers import AutoTokenizer, Gemma2ForCausalLM, PreTrainedTokenizerBase
    from sae_scoping.trainers.sae_enhanced.prune import get_pruned_sae
    from sae_scoping.utils.generation.messages import (
        is_valid_1turn_messages,
        is_valid_0turn_messages,
    )
    import tqdm

    checkpoint_infos = [CheckpointInfo.from_dict(info) if isinstance(info, dict) else info for info in checkpoint_infos]

    # 1. Load the queries
    # Everyone uses the same queries
    # Load from file so that we use the same exact queries
    typed_queries_file = output_path / f"typed_queries.json"
    if not typed_queries_file.exists():
        raise ValueError(f"Typed queries file not found: {typed_queries_file}")
    _typed_queries = orjson.loads(typed_queries_file.read_bytes())
    assert len(_typed_queries) > 0
    print("LOADED", len(_typed_queries), "typed queries")
    typed_queries = [(q["messages"], q["type"]) for q in _typed_queries]

    # Sorting by length to avoid padding
    print("SORTING BY LENGTH TO AVOID PADDING...")
    _tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-9b-it")
    _tokenizer.padding_side = "left"
    typed_queries = sorted(
        typed_queries,
        key=lambda x: _tokenizer.apply_chat_template(x[0], tokenize=True, add_generation_prompt=False),
        reverse=True,
    )

    # > clip the messages; that way we have golden in the main data, but here
    # we have the golden responses removed and the model's response will replace
    # it
    # (note that imdb is n-shots so it may not be 1 turn)
    assert all(t == "imdb" or is_valid_1turn_messages(m) for m, t in typed_queries)
    assert all(t != "imdb" or all(mm["role"] == ["user", "assistant"][i % 2] for i, mm in enumerate(m)) for m, t in typed_queries)
    assert all(m[-1]["role"] == "assistant" for m, _ in typed_queries)
    typed_queries = [(m[:-1], t) for m, t in typed_queries]
    assert all(t == "imdb" or is_valid_0turn_messages(m) for m, t in typed_queries)
    assert all(m[-1]["role"] == "user" for m, _ in typed_queries)

    # 2. loop through checkpoints and generate with them
    for checkpoint_info in tqdm.tqdm(checkpoint_infos):
        model, tokenizer, sae, pruned_sae = None, None, None, None
        try:
            # 1. Load model, etc...
            model, tokenizer, sae, pruned_sae = load_checkpoint_with_sae(checkpoint_info, device=device, dist_cache_path=dist_cache_path)
            if debug_only_load:
                continue
            # 2. Run the generations that later we can judge
            # is_vanilla => no SAE => no hookpoint
            hookpoint = None if checkpoint_info.is_vanilla else sae_id2hookpoint(checkpoint_info.sae_id)
            output_subpath = output_path / f"{checkpoint_info.uid}"
            assert not output_subpath.exists()
            output_subpath.mkdir(parents=True, exist_ok=True)
            output_config_file = output_subpath / f"config.json"
            output_config_file.write_bytes(orjson.dumps(checkpoint_info.to_dict(json_serializable=True)))
            output_file = output_subpath / f"generations.jsonl"
            error_file = output_subpath / f"error.json"
            # TODO(Adriano) pre-tokenize and use length-aware tokenizer for slightly faster
            # inference please! the batch size could be effectively 1.5-xed on average I suspect
            inference(
                model=model,
                tokenizer=tokenizer,
                pruned_sae=pruned_sae,
                typed_queries=typed_queries,
                batch_size=batch_size,
                max_num_tokens=max_num_tokens,
                output_file=output_file,
                hookpoint=hookpoint,
            )
            error_file.write_bytes(orjson.dumps({"success": True}))
        except Exception as e:
            print("=" * 100)
            print(f"Error loading checkpoint: {e}")
            exc = traceback.format_exc()
            print(exc)
            error_file.write_bytes(orjson.dumps({"success": False, "error": str(e), "traceback": exc}))
            print("=" * 100)
            if debug_strict_raises:
                raise e
        finally:
            if model is not None and isinstance(model, Gemma2ForCausalLM):
                model = model.to("cpu")
                del model
            if tokenizer is not None:
                del tokenizer
            if sae is not None:
                sae = sae.to("cpu")
                del sae
            if pruned_sae is not None:
                del pruned_sae
            gc.collect()
            torch.cuda.empty_cache()


def main_worker_wrapper(args: dict[str, Any]) -> None:
    main_worker(**args)


@click.command()
@click.option(
    "--checkpoints-path",
    "-cp",
    type=click.Path(exists=True, path_type=Path),
    default="/mnt/align4_drive2/adrianoh/git/ScopeBench/sae_training/outputs_gemma9b",
    help="Root path containing dataset folders with checkpoints",
)
@click.option(
    "--dist-cache-path",
    "-dp",
    type=click.Path(exists=True, path_type=Path),
    default="/mnt/align4_drive2/adrianoh/git/ScopeBench/sae_training/deleteme_cache_bio_only/ignore_padding_True/biology",
    help="Path to distribution cache (for loading pruned SAEs)",
)
@click.option(
    "--output-path",
    "-o",
    type=click.Path(path_type=Path),
    default=Path(__file__).parent / "outputs_gemma9b_judging_generations",
    help="Path to output directory",
)
@click.option(
    "--debug-only-load",
    "--dry-run",
    "-dry",
    is_flag=True,
    help="Only list checkpoints, don't load them",
)
@click.option(
    "--debug-strict-raises",
    "-dsr",
    is_flag=True,
    help="Strictly raise errors",
)
@click.option(
    "--debug-load-fake-queries",
    "-dlfq",
    is_flag=True,
    help="Load fake queries for debugging",
)
# NOTE: we have a core issue: for these models there is no way to avoid
# train/test split; so this will be a bit tainted but in the future we can
# avoid more carefully...
# @click.option(
#     "--num-generations",
#     "-n",
#     default=30,
#     type=int,
#     help="Number of generations to run for each checkpoint",
# )
# @click.option(
#     "--shuffle-seed", "-seed", default=90, type=int, help="Seed for shuffling"
# )
@click.option(
    "--gpus",
    "-g",
    default="0,1,2,3",
    type=str,
    help="Comma-separated list of GPU IDs to use",
)
@click.option(
    "--max-num-tokens",
    "-m",
    default=1024,
    type=int,
    help="Maximum number of tokens to generate",
)
@click.option("--batch-size", "-b", default=128, type=int, help="Batch size for generations")
@click.option(
    "--imdb-n-shots",
    "-ins",
    # Probably around 5K tokens TOTAL including the response from our model
    # so i think this is ok
    default=8,
    type=int,
    help="Number of shots to use for IMDB",
)
@click.option(
    "--imdb-n-samples",
    "-is",
    default=500,
    type=int,
    help="Number of samples to use for IMDB",
)
@click.option(
    "--biology-n-samples",
    "-bs",
    default=500,
    type=int,
    help="Number of samples to use for BIOLOGY",
)
@click.option(
    "--apps-easy-n-samples",
    "-aes",
    # NOTE there are only up to 500 of these
    default=500,
    type=int,
    help="Number of samples to use for APPS EASY",
)
@click.option(
    "--apps-hard-n-samples",
    "-ahs",
    default=500,
    type=int,
    help="Number of samples to use for APPS HARD",
)
@click.option(
    "--ultrachat-n-samples",
    "-us",
    default=500,
    type=int,
    help="Number of samples to use for ULTRACHAT",
)
@click.option(
    "--retry",
    "-r",
    is_flag=True,
    help="Retry failed generations. In this mode of operation, instead of creating folders, etc... it is ASSUMED that the folders are there. The folders where generations failed are detected via error.json file and then those are deleted and re-run.",
)
def main(
    checkpoints_path: Path | str,
    dist_cache_path: Path | str,
    output_path: Path | None,
    debug_only_load: bool,
    debug_strict_raises: bool,
    debug_load_fake_queries: bool,
    # num_generations: int,
    # shuffle_seed: int,
    gpus: str,
    max_num_tokens: int,
    batch_size: int,
    imdb_n_shots: int,
    imdb_n_samples: int,
    biology_n_samples: int,
    apps_easy_n_samples: int,
    apps_hard_n_samples: int,
    ultrachat_n_samples: int,
    retry: bool,
) -> None:
    # 1. setup the data
    output_data_file = output_path / f"typed_queries.json"
    if retry:
        assert output_data_file.exists()
    else:
        queries: list[tuple[list[dict[str, str]], str]] = get_eval_queries(
            debug_load_fake_queries=debug_load_fake_queries,
            imdb_n_shots=imdb_n_shots,
            imdb_n_samples=imdb_n_samples,
            biology_n_samples=biology_n_samples,
            apps_easy_n_samples=apps_easy_n_samples,
            apps_hard_n_samples=apps_hard_n_samples,
            ultrachat_n_samples=ultrachat_n_samples,
        )
        output_data_file.parent.mkdir(parents=True, exist_ok=True)
        assert not output_data_file.exists()
        output_data_file.write_bytes(orjson.dumps([{"type": t, "messages": q} for q, t in queries]))

    # 2. Do generation
    gpus = list(map(int, map(str.strip, gpus.split(","))))
    if len(set(gpus)) != len(gpus):
        raise ValueError("GPUs must be unique")
    devices = [f"cuda:{gpu}" for gpu in gpus]
    # These statistics were gotten for biology and are what we used for everything
    if retry:
        print("=" * 100)
        print("IGNORING CHECKPOINTS FLAG! (RETRY WILL USE THE OUTPUT FOLDER'S FAILED SUBFOLDERS)")
        subfolders = [f for f in output_path.iterdir() if f.is_dir()]
        checkpoint_infos = []
        failed_subfolders = []
        for subfolder in subfolders:
            error_file = subfolder / f"error.json"
            error_contents = orjson.loads(error_file.read_bytes()) if error_file.exists() else {"success": False}
            if not error_contents["success"]:
                config_file = subfolder / f"config.json"
                assert config_file.exists()
                config_contents = orjson.loads(config_file.read_bytes())
                checkpoint_infos.append(CheckpointInfo.from_dict(config_contents))
                failed_subfolders.append(subfolder)
        click.confirm(
            f"Are you sure you want to retry {len(failed_subfolders)} failed generations? About to delete the corresponding folders...",
            abort=True,
        )
        for f in failed_subfolders:
            try:
                shutil.rmtree(f)
            except Exception as e:
                if f.exists():
                    raise e
    else:
        checkpoints_path = Path(checkpoints_path)
        dist_cache_path = Path(dist_cache_path)
        output_path = Path(output_path)
        checkpoint_infos = list(iter_checkpoints(checkpoints_path, include_vanilla=True))
    if len(checkpoint_infos) == 0:
        raise ValueError("No checkpoints found")
    _dicts = [checkpoint.to_dict(json_serializable=True) for checkpoint in checkpoint_infos]
    # TODO(Adriano) this is possible to serialize even within checkpoints for large sample
    # sizes. Might want to add some support for that...
    if len(_dicts) < len(devices):
        devices = devices[: len(_dicts)]
    args = [
        {
            "checkpoint_infos": _dicts[i :: len(devices)],
            "device": device,
            "dist_cache_path": dist_cache_path,
            "output_path": output_path,
            "debug_only_load": debug_only_load,
            "debug_strict_raises": debug_strict_raises,
            "debug_load_fake_queries": debug_load_fake_queries,
            # > these two are not used
            # "num_generations": num_generations,
            # "shuffle_seed": shuffle_seed,
            "batch_size": batch_size,
            "max_num_tokens": max_num_tokens,
            # these guys have access to the file and are read only
        }
        for i, device in enumerate(devices)
    ]
    with mp.Pool(len(devices)) as pool:
        pool.map(
            main_worker_wrapper,
            args,
        )


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    def test_checkpoint_iterator() -> None:
        path = Path("/mnt/align4_drive2/adrianoh/git/ScopeBench/sae_training/outputs_gemma9b")
        print("=" * 100)
        print(f"Iterating through checkpoints in {path}")
        checkpoints = list(iter_checkpoints(path, include_vanilla=True))
        print(f"Found {len(checkpoints)} checkpoints")
        for checkpoint in checkpoints:
            print(json.dumps(checkpoint.to_dict(json_serializable=True), indent=4))
        print("n_checkpoints:", len(checkpoints))
        print("Expect 28 = `tree /mnt/align4_drive2/adrianoh/git/ScopeBench/sae_training/outputs_gemma9b | grep checkpoint | wc -l`")
        assert len(checkpoints) == 29
        assert checkpoints[-1].is_vanilla
        assert checkpoints[-1].is_untrained_vanilla
        print("[OK] passed the test!")

    test_checkpoint_iterator()

    def test_load_all_checkpoints() -> None:
        print("=" * 100)
        print("Loading all checkpoints across one or more devices...")
        path = Path("/mnt/align4_drive2/adrianoh/git/ScopeBench/sae_training/outputs_gemma9b")
        dist_cache_path = Path("/mnt/align4_drive2/adrianoh/git/ScopeBench/sae_training/deleteme_cache_bio_only/ignore_padding_True/biology")
        devices = ["cuda:0", "cuda:1", "cuda:2", "cuda:3"]
        checkpoint_infos = list(iter_checkpoints(path, include_vanilla=True))
        _dicts = [checkpoint.to_dict(json_serializable=True) for checkpoint in checkpoint_infos]
        # These statistics were gotten for biology and are what we used for everything
        args = [
            {
                "checkpoint_infos": _dicts[i :: len(devices)],
                "device": device,
                "dist_cache_path": dist_cache_path,
                "debug_only_load": True,  # this is the test
                "debug_strict_raises": True,  # this is the test
            }
            for i, device in enumerate(devices)
        ]
        with mp.Pool(len(devices)) as pool:
            pool.map(
                main_worker_wrapper,
                args,
            )
        print("[OK] passed the test!")
        print("=" * 100)

    # test_load_all_checkpoints() # Slow and lgtm tbh
    main()
