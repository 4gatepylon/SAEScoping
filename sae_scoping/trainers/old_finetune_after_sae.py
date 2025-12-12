from __future__ import annotations
from datetime import datetime
import wandb
import click
import re
import torch.nn as nn
import json
import torch
import functools as Ft
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
from typing import Optional, List
from sparsify.sparse_coder import SparseCoder
from datasets import load_dataset, Dataset, DatasetDict
from trl import SFTConfig, SFTTrainer
from utils.hooks.pt_hooks import named_forward_hooks, filter_hook_fn
from utils.spylab.sentence_preprocess import SpylabPreprocessor

try:
    from training_saes.distill_trainer import LogitsTrainer
except ImportError:
    from distill_trainer import LogitsTrainer

TRAINER_CLS_NAME_2_TRAINER_CLS = {
    "SFTTrainer": SFTTrainer,
    "LogitsTrainer": LogitsTrainer,
}


class FinetuneAfterSAETrainer:
    """
    # Information
    Training class to wrap functionality for SFT training in a special use-case: when you
    only want to update some layers and may be processing some of the activations of
    your model using "interpreter models" like SAEs
    (https://transformer-circuits.pub/2023/monosemantic-features). Only a narrow set of
    functionality is supported right now.

    # Supported Usage
    1. Interpreter model:
        - Must be an EleutherAI `sparsify` library SAE
        - Must act on the residual stream of the model
        - The model it acts on should be a transformer like Llama, GPT, etc.
        - You can only use one interpreter model at a time
    2. What gets updated:
        - Only layers after the SAE are updated
    3. Dataset:
        - Dataset must be text (no latents datasets :/)
        - Basically every dataset is allowable so long as you can specify how to load
            it.
    4. GPU environment, performance optimization, etc...:
        - Runs only on a single GPU at a time

    # Usage Details
    TODO(Adriano) fill this in and generalize this module more.

    # Improvements in the future (TODOs):
    - Multi-gpu support
    - Support any dataset with more flexibility (has some but not all)
    - PeFT support
    - More efficient trainer and stuff along those lines (partial model loading for
        example)
    - Better support for documenting and confirming the exact layer (because sometimes
        we did pre vs. post hooks and then you wouldn't know... and the data would
        be different)
    """

    # NOTE: no use of kwargs, because we want to be loading from this arguments file
    def __init__(
        self,
        # Where to load from
        sae_path: str = "",
        model_name_or_path: str = "",
        tokenizer_name_or_path: str = "",
        dataset_name: str = "",
        sft_config_args: dict = {},
        device: str = "cuda",  # NOTE: you are recommended to use $CUDA_VISIBLE_DEVICES
        freeze_up_to_layer: int | str = "sae+0",
        freeze_non_layer_params_res: List[str] = [],
        no_freeze_non_layer_params_res: List[str] = [],
        tokenize_dataset_kwargs: dict = {},
        load_dataset_kwargs: dict = {},
        train_args: dict = {
            # TODO(Adriano) figure this out a little better, it should merely follow
            # from the SFT trainer
            "epochs": 1,
            "learning_rate": 1e-4,
            "batch_size": 16,
            "gradient_accumulation_steps": 1,
            "warmup_ratio": 0.05,
            "weight_decay": 0.01,
            "max_grad_norm": 1.0,
        },
        trainer_cls_name: str = "SFTTrainer",
        **kwargs,
    ):
        # Store paths
        if (
            len(sae_path) == 0
            or len(model_name_or_path) == 0
            or len(tokenizer_name_or_path) == 0
        ):
            raise ValueError("All paths must be non-empty")
        self.sae_path = Path(sae_path)
        # These next two must be for hf strings
        self.model_name_or_path = model_name_or_path
        self.tokenizer_name_or_path = tokenizer_name_or_path or self.model_name_or_path
        if (
            not self.sae_path.exists()
            # or not self.model_name_or_path.exists() # Can load from HF
            # or not self.tokenizer_name_or_path.exists() # Can load from HF
        ):
            raise FileNotFoundError(
                "All paths must exist: " + f"sae_exists={self.sae_path.exists()}, "
                # + f"model_exists={self.model_name_or_path.exists()}, "
                # + f"tokenizer_exists={self.tokenizer_name_or_path.exists()}"
            )

        # Collect sub-arguments
        self.sft_config_args = sft_config_args
        self.using_peft = False
        self.train_args = train_args
        if "peft_config" in self.train_args:
            # NOTE: that if we are learning PEFT we will freeze the entire damn network
            self.using_peft = True
            self.train_args["peft_config"] = LoraConfig(
                **self.train_args["peft_config"]
            )

        # Load dataset
        self.dataset_name = dataset_name
        self.load_dataset_kwargs = load_dataset_kwargs
        self.dataset_dict = load_dataset(self.dataset_name, **self.load_dataset_kwargs)
        assert isinstance(self.dataset_dict, DatasetDict)
        self.dataset = self.dataset_dict["train"]
        self.dataset_eval = (
            self.dataset_dict["validation"]
            if "validation" in self.dataset_dict
            else None
        )
        self.dataset_test = (
            self.dataset_dict["test"] if "test" in self.dataset_dict else None
        )
        assert isinstance(self.dataset, Dataset)

        # Load stuff
        self.device = device
        self.freeze_up_to_layer = freeze_up_to_layer
        # These two are regexes used to basically deal with things like
        # `model.embed_tokens.weight` etc... (anything not re-matched here and not
        # in a layer will lead to a throw of an exception)
        self.freeze_non_layer_params_res = freeze_non_layer_params_res
        self.no_freeze_non_layer_params_res = no_freeze_non_layer_params_res
        self.sae_cfg = None
        self.sae_model = None
        self.model = None
        self.tokenizer = None
        self.freeze_layer = None
        self.pname2should_freeze = None
        self.load_sae()
        self.load_model_and_tokenizer(freeze_up_to_layer=self.freeze_up_to_layer)
        # sanity checks
        assert self.sae_cfg is not None
        assert self.sae_model is not None
        assert self.model is not None
        assert self.tokenizer is not None
        assert self.freeze_layer is not None
        assert self.pname2should_freeze is not None and all(
            pname in self.pname2should_freeze
            for pname, _ in self.model.named_parameters()
        )

        if self.using_peft:  # eh lmao
            print("=" * 100)  # DEBUG
            print("Using PEFT")  # DEBUG
            print("=" * 100)  # DEBUG
            # Only the PEFT parameters should be updated
            self.model.eval()
            for p in self.model.parameters():
                p.requires_grad = False
                p.grad = None
            self.model.to(self.device)
            self.pname2should_freeze = {
                pname: True for pname in self.pname2should_freeze.keys()
            }

        # Setup dataset caching (every time we ask for same parameters it will replace
        # the cached dataset if it's not equal else it will use the same)
        self.tokenize_dataset_kwargs = tokenize_dataset_kwargs
        self.tokenized_dataset_train_cached = None
        self.tokenized_dataset_train_cached_kwargs = None
        self.tokenized_dataset_eval_cached = None
        self.tokenized_dataset_eval_cached_kwargs = None
        self.tokenized_dataset_test_cached = None
        self.tokenized_dataset_test_cached_kwargs = None

        self.trainer_cls_name = trainer_cls_name
        self.trainer_cls = TRAINER_CLS_NAME_2_TRAINER_CLS[trainer_cls_name]
        if self.trainer_cls_name == "LogitsTrainer":
            # Look below, but basically we are always going to train on a copy of the
            # same damn model, but for this other one the parameters will not be changing
            # so we can use it as the teacher model
            # TODO(Adriano) implement optimizations to save memory, etc... here please
            # (when we refactor we will want to do it this way)
            self.teacher_model = AutoModelForCausalLM.from_pretrained(
                self.model_name_or_path
            )
            self.teacher_model.eval()
            for p in self.teacher_model.parameters():
                p.requires_grad = False
                p.grad = None
            self.teacher_model.to(self.device)

    def load_sae(self) -> None:
        # 1. Load the configs
        assert self.sae_path.exists()
        assert self.sae_path.is_dir()
        cfg_dict_path = self.sae_path / "cfg.json"
        assert cfg_dict_path.exists()
        self.sae_cfg = json.loads(cfg_dict_path.read_text())
        assert re.match(r"^layers\.\d+$", self.sae_path.name)
        layer_num = int(self.sae_path.name.split(".", 1)[1])
        assert layer_num >= 0  # 0 is probably embedding layer tbh
        assert "layer" not in self.sae_cfg
        self.sae_cfg["layer"] = layer_num

        # 2. Load the model itself (this will load from the safetensors file)
        safetensors_path = self.sae_path / "sae.safetensors"
        assert safetensors_path.exists()
        sae_model = SparseCoder.load_from_disk(self.sae_path.resolve().as_posix())
        # 2.1 Clean out parameters and gradients to avoid memory issues
        sae_model.eval()
        for p in sae_model.parameters():
            p.requires_grad = False
            p.grad = None
        # 2.2 Put onto the device
        sae_model.to(self.device)
        self.sae_model = sae_model

    def _get_freeze_layer(
        self, freeze_up_to_layer: int | str, sae_layer: int, num_layers: int
    ) -> Optional[int]:
        if freeze_up_to_layer == "none":
            return None
        elif freeze_up_to_layer == "all":
            return num_layers - 1  # NOTE: always inclusive
        elif freeze_up_to_layer == "sae":
            return sae_layer
        elif re.match(r"^sae\+\d+$", freeze_up_to_layer):
            return sae_layer + int(freeze_up_to_layer.split("+")[1])
        elif re.match(r"^sae-\d+$", freeze_up_to_layer):
            return sae_layer - int(freeze_up_to_layer.split("-")[1])
        else:
            raise ValueError(f"Invalid freeze_up_to_layer: {freeze_up_to_layer}")

    def _get_layer_num(self, pname: str) -> int:
        m1 = re.match(r"^[a-zA-Z0-9_.]+\.layers\.(\d+)$", pname)
        if m1 is not None:
            return int(m1.group(1))
        m2 = re.match(r"^[a-zA-Z0-9_.]+\.layers\.(\d+)\.[a-zA-Z0-9_.]+$", pname)
        if m2 is not None:
            return int(m2.group(1))
        raise ValueError(f"Invalid parameter name: {pname}")

    def _printout_freeze_layer_info(self) -> None:
        assert self.model is not None
        assert self.freeze_layer is not None
        assert self.sae_cfg is not None and self.sae_cfg["layer"] is not None
        print("=" * 100)
        print(
            f"Freeze layer = {self.freeze_layer} w/ SAE layer = {self.sae_cfg['layer']}"
        )
        pnames = [
            (pname, p.shape, p.requires_grad, p.grad is not None)
            for pname, p in self.model.named_parameters()
        ]
        pnames.sort(key=lambda x: x[0])
        print("=" * 100)

    def load_model_and_tokenizer(self, freeze_up_to_layer: int | str = "sae+0") -> None:
        """
        If you pass `freeze_up_to_layer="none"` (caps not relevant) then no layers will
            be frozen (i.e. gradient set to None). Otherwise, if you pass "all" all will
            be frozen and otherwise if you pass "sae" or "sae+0" all layers up to and
            including the SAE layer will be frozen. You can also pass r"sae+[0-9]+" for some
            integer that will place the freeze point at most in the latest layer. You
            can also do "sae-[0-9]+" for some integer that will place the freeze point at
            least in the earliest layer.

        NOTE: this requires your trasnformer to have hookpoint that matches the regex for
        <something>.layers.<number><.<something else> or nothing here>
        """
        # 1. Load the model
        model = AutoModelForCausalLM.from_pretrained(self.model_name_or_path)
        # 2. Clean out parameters and gradients to avoid memory issues
        model.eval()
        # Check the layer number is valid
        assert self.sae_cfg is not None
        assert self.sae_cfg["layer"] is not None
        assert self.sae_cfg["layer"] >= 0
        assert self.sae_cfg["layer"] < model.config.num_hidden_layers

        # 3. Find the relevant freeze layer for comparison for each parameter (we will
        # not freeze if you are AFTER the layer and otherwise YES we will freeze; with
        # exceptions made by the regexes)
        freeze_layer = self._get_freeze_layer(
            freeze_up_to_layer,
            self.sae_cfg["layer"],
            model.config.num_hidden_layers,
        )
        self.freeze_layer = freeze_layer
        self.pname2should_freeze = {}
        assert 0 <= freeze_layer < model.config.num_hidden_layers, (
            f"freeze_layer: {freeze_layer} is not in [0, {model.config.num_hidden_layers})"
        )
        # 4. Freeze/unfreeze parameters based on the layer number and regexes
        for pname, p in model.named_parameters():
            # 4.1 Get the layer number (or a dummy) to decide whether to freeze
            layer_num = None  # should not be none after this
            try:
                layer_num = self._get_layer_num(pname)
            except ValueError:
                should_freeze = any(
                    re.match(res, pname) for res in self.freeze_non_layer_params_res
                )
                if should_freeze:
                    layer_num = freeze_layer - 1  # NOTE no check for bounds below :)
                else:
                    should_not_freeze = any(
                        re.match(res, pname)
                        for res in self.no_freeze_non_layer_params_res
                    )
                    if should_not_freeze:
                        layer_num = (
                            freeze_layer + 1
                        )  # NOTE no check for bounds below :)
                    else:
                        raise ValueError(
                            f"Invalid parameter name (in try/except): {pname} "
                            + "(not found in any layer nor in the "
                            + "freeze_non_layer_params_res or "
                            + "no_freeze_non_layer_params_res)"
                        )
            if layer_num is None:
                raise ValueError(
                    f"Invalid parameter name (evaded try/except): {pname} "
                    + "(not found in any layer nor in the "
                    + "freeze_non_layer_params_res or "
                    + "no_freeze_non_layer_params_res)"
                )
            # 4.2 Freeze/unfreeze based on layer number + regexes (dummy in latter case)
            if freeze_layer is not None and layer_num <= freeze_layer:
                p.requires_grad = False
                p.grad = None
                self.pname2should_freeze[pname] = True  # should freeze => do not change
            else:
                p.requires_grad = True
                self.pname2should_freeze[pname] = False  # Do NOT change this!
        # 5. Load the tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name_or_path)
        # 6. Put onto the device
        model.to(self.device)
        self.model = model
        self.tokenizer = tokenizer
        # 7. Make sure we will be able to pad and finish sequences during training
        # (we will use naive padding after the sequence basically --> realistically
        # we just use the `.encode(...)` method like in `train_random_sae.py` as of
        # 2025-07-06)
        # TODO(Adriano) add support for setting this yourself (right now for ethz
        # spylab models we do not need this :P)
        # Must be able to pad sequences
        assert self.tokenizer.pad_token is not None
        assert self.tokenizer.pad_token_id is not None
        # Must be able to finish sequences
        assert self.tokenizer.eos_token is not None
        assert self.tokenizer.eos_token_id is not None

    def _tokenize_dataset_elem_single(
        self,
        elem: dict,
        text_key: str = "text",
        input_ids_key: str = "input_ids",
        ctx_len: int = 2048,
        allow_too_big: bool = False,
    ) -> dict:
        assert self.tokenizer is not None
        assert text_key in elem
        assert isinstance(elem[text_key], str), (
            f"elem[text_key] = {elem[text_key]}\n" + f"of type {type(elem[text_key])}"
        )
        dict_clone = elem.copy()
        dict_clone[input_ids_key] = self.tokenizer.encode(
            elem[text_key],
            return_tensors="pt",
        )
        assert isinstance(dict_clone[input_ids_key], torch.Tensor)
        assert (
            (
                dict_clone[input_ids_key].ndim == 2
                and dict_clone[input_ids_key].shape[0] == 1
            )
            or (dict_clone[input_ids_key].ndim == 1),
            f"shape: {dict_clone[input_ids_key].shape}",
        )
        if (
            dict_clone[input_ids_key].ndim == 2
            and dict_clone[input_ids_key].shape[0] == 1
        ):
            dict_clone[input_ids_key] = dict_clone[input_ids_key][0]
        if dict_clone[input_ids_key].shape[-1] > ctx_len and not allow_too_big:
            raise ValueError(
                f"Sequence too long: {dict_clone[input_ids_key].shape[-1]} > {ctx_len}"
            )
        if dict_clone[input_ids_key].shape[-1] < ctx_len:
            # TODO(Adriano) should this be EOS or PAD? I am using EOS because it is what
            # we do in `train_random_sae.py` as of 2025-07-06
            dict_clone[input_ids_key] = torch.cat(
                [
                    dict_clone[input_ids_key],
                    torch.ones(
                        ctx_len - dict_clone[input_ids_key].shape[-1],
                        dtype=torch.long,
                    )
                    * self.tokenizer.eos_token_id,
                ],
                dim=0,  # should only exist 1
            )
        assert dict_clone[input_ids_key].ndim == 1
        assert dict_clone[input_ids_key].shape[-1] >= ctx_len, (
            f"shape: {dict_clone[input_ids_key].shape}; ctx_len: {ctx_len}"
        )
        return dict_clone

    def tokenize_dataset(
        self,
        # kwargs for `_tokenize_dataset_elem_single`
        tokenize_dataset_single_kwargs: dict = {
            "text_key": "text",
            "input_ids_key": "input_ids",
            "ctx_len": 2048,
            "allow_too_big": True,
        },
        # Inclusive bounds for filter
        filter_ctx_len: Optional[int | List[int]] = [0, 2048],
        do_shuffle: bool = True,
        shuffle_seed: Optional[int] = 42,
        max_n_samples: Optional[int] = None,
    ) -> None:
        assert self.tokenizer is not None
        assert self.dataset is not None
        # 0. Early stop if the kwargs are the same
        kwargs_to_check = {
            "tokenize_dataset_single_kwargs": tokenize_dataset_single_kwargs,
            "filter_ctx_len": filter_ctx_len,
            "do_shuffle": do_shuffle,
            "shuffle_seed": shuffle_seed,
            "max_n_samples": max_n_samples,
        }
        if isinstance(filter_ctx_len, list):
            assert len(filter_ctx_len) == 2
        if self.tokenized_dataset_train_cached is not None:
            assert self.tokenized_dataset_train_cached_kwargs is not None
            kwargs_to_cmp = self.tokenized_dataset_train_cached_kwargs
            # NOTE: apparently this should be deep
            # NOTE: if you pass int and then tuple filter_ctx_len and they are equiv.
            # this will no cache so plz don't do that.
            if kwargs_to_cmp == kwargs_to_check:
                return
        # 1. Tokenize
        dataset_tokenized = self.dataset.map(
            lambda x: self._tokenize_dataset_elem_single(
                x, **tokenize_dataset_single_kwargs
            ),
            # batched=True, # TODO(Adriano) fix batching ty
        )
        # 2. Filter
        if filter_ctx_len is not None:
            if isinstance(filter_ctx_len, int):  # Upper bound here
                filter_ctx_len = (0, filter_ctx_len)
            assert 0 <= filter_ctx_len[0] <= filter_ctx_len[1]
            dataset_tokenized = dataset_tokenized.filter(
                # It looks like this is no longer a tensor for some reason
                lambda x: torch.tensor(x["input_ids"]).shape[-1] >= filter_ctx_len[0]
                and torch.tensor(x["input_ids"]).shape[-1] <= filter_ctx_len[1]
            )
        # 3. Shuffle (first, cuz that way the subset isn't always the same)
        if do_shuffle:
            dataset_tokenized = dataset_tokenized.shuffle(seed=shuffle_seed)
        # 4. Max samples
        if max_n_samples is not None:
            max_n_samples = min(max_n_samples, len(dataset_tokenized))
            dataset_tokenized = dataset_tokenized.select(range(max_n_samples))
        dataset_tokenized = dataset_tokenized.with_format("torch")  # so can grab chunks
        # for di in dataset_tokenized:
        #     di["input_ids"] = torch.tensor(
        #         di["input_ids"],
        #         dtype=torch.long
        #     ).to(self.device)[0] # DELETEME
        self.tokenized_dataset_train_cached = dataset_tokenized
        self.tokenized_dataset_train_cached_kwargs = kwargs_to_check

    def _get_layers_module_name(self) -> str:
        for pname, _ in self.model.named_parameters():
            try:
                layer_num = self._get_layer_num(pname)
                findme_regex = r"^([a-zA-Z0-9_.]+)\.layers\.\d+.*$"
                m = re.match(findme_regex, pname)
                if m is None:
                    raise ValueError(
                        f"Invalid parameter name: {pname} (!match regex for pre-layers)"
                    )
                if "." in m.group(1):
                    raise NotImplementedError(
                        f"Invalid parameter name: {pname} (has . in the name)"
                    )
                return m.group(1)
            except ValueError:
                continue
        raise ValueError(
            "No layers module name found (did not match regex for pre-layers)"
        )

    @staticmethod
    def _sae_filter_fn(sae: SparseCoder, tensors: torch.Tensor):
        """Meant for use with functools.partial"""
        og_shape, og_dtype = tensors.shape, tensors.dtype
        sae_dtype = next(sae.parameters()).dtype
        assert len(og_shape) >= 2, f"tensors.shape: {tensors.shape}"
        d_model = og_shape[-1]
        input_pt = tensors.reshape(-1, d_model).to(sae_dtype)
        # https://github.com/EleutherAI/sparsify/blob/d17b1ee18f42b0a96ed700e50d0e11f411b03205/sparsify/sparse_coder.py#L18
        output_pt = sae(input_pt).sae_out.to(og_dtype)
        return output_pt.reshape(og_shape)

    @staticmethod
    def _pname2hashes(model: nn.Module) -> dict[str, torch.Tensor]:
        # https://stackoverflow.com/questions/74805446/how-to-hash-a-pytorch-tensor
        # gives a way to implement hasing but it's too slow; the idea is basically to
        # store the hash as a tensor (could be a long tensor or something else)
        # we store it in CPU; the way we do this kind of efficiently is by selecting
        # random indices to use as our hash
        numel_desired = 1024 * 1024
        pnames_sorted = sorted(pname for pname, _ in model.named_parameters())
        pname2numel = {pname: p.numel() for pname, p in model.named_parameters()}
        pname2device = {pname: p.device for pname, p in model.named_parameters()}
        torch.manual_seed(0)  # This + sorted list should ensure determinism
        pnames2hashindices = {
            pname: torch.randint(
                0,
                pname2numel[pname],
                (numel_desired,),
                dtype=torch.long,
                device=pname2device[pname],
            )
            for pname in pnames_sorted
        }
        pname2hashvalue = {
            pname: p.data.view(-1)[pnames2hashindices[pname]].cpu()
            for pname, p in model.named_parameters()
        }
        return pname2hashvalue

    @staticmethod
    def _assert_pname2hashes_equal(
        pname2hashes_before: dict[str, torch.Tensor],
        pname2hashes_after: dict[str, torch.Tensor],
        pname2should_freeze: dict[str, bool],
    ) -> bool:
        # 1. All pnames should be the same
        assert set(pname2hashes_before.keys()) == set(pname2hashes_after.keys())
        assert set(pname2should_freeze.keys()) == set(pname2hashes_before.keys())
        # 2. All types correct
        assert all(isinstance(pname, str) for pname in pname2hashes_before.keys())
        assert all(isinstance(pname, str) for pname in pname2hashes_after.keys())
        assert all(isinstance(pname, str) for pname in pname2should_freeze.keys())
        assert all(
            isinstance(pname2should_freeze[pname], bool)
            for pname in pname2should_freeze.keys()
        )
        assert all(
            isinstance(pname2hashes_before[pname], torch.Tensor)
            for pname in pname2hashes_before.keys()
        )
        assert all(
            isinstance(pname2hashes_after[pname], torch.Tensor)
            for pname in pname2hashes_after.keys()
        )
        # NOTE we force this to be on cpu for memory reasons
        assert all(str(p.device) == "cpu" for p in pname2hashes_before.values())
        assert all(str(p.device) == "cpu" for p in pname2hashes_after.values())
        # 3. All hashes are the same IFF should_freeze is True
        pname2incorrect_message = {}
        for pname in pname2hashes_before.keys():
            if pname2should_freeze[pname]:
                if torch.any(pname2hashes_before[pname] != pname2hashes_after[pname]):
                    pname2incorrect_message[pname] = (
                        "Hash changed (should be frozen/same), "
                        # + f"before={pname2hashes_before[pname]}, "
                        # + f"after={pname2hashes_after[pname]}"
                    )
            else:  # Not freeze => should change
                if torch.all(pname2hashes_before[pname] == pname2hashes_after[pname]):
                    pname2incorrect_message[pname] = (
                        "Hash did NOT change (should be unfrozen/different), "
                        # + f"before={pname2hashes_before[pname]}, "
                        # + f"after={pname2hashes_after[pname]}"
                    )
        if len(pname2incorrect_message) > 0:
            raise ValueError(
                "Hash changed (or didn't) for some parameters, "
                # + f"before={json.dumps(pname2hashes_before, indent=4)}, "
                # + f"after={json.dumps(pname2hashes_after, indent=4)}, "
                + f"incorrect={json.dumps(pname2incorrect_message, indent=4)}"
            )

    def finetune(
        self,
        log_to_wandb: bool = False,
        debug_assert_layer_hash_no_change: bool = True,
        wandb_name: Optional[str] = None,
    ):
        """
        If `debug_assert_layer_hash_no_change` is True, then we will assert that the
        layer hash is the same before and after finetuning for all freeze layers where
        `self.pname2should_freeze[pname]` is True (and it checks that other ones DID
        change).
        """
        if log_to_wandb:
            curr_time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            wandb.init(
                project="finetune-after-sae",  # TODO(Adriano) use proper name?
                name=(
                    f"{self.model_name_or_path.replace('/', '_')}-{self.sae_path.name.replace('/', '_')}-{curr_time_str}"
                    if wandb_name is None
                    else wandb_name
                ),
                # config=self.train_args, # ??
            )
        try:
            if debug_assert_layer_hash_no_change:
                print("=" * 100)
                print("Layers that are frozen and should not change:")
                print(json.dumps(self.pname2should_freeze, indent=4))
                print("=" * 100)

            # 0. Tokenize the dataset and sanity check its shape for our impl.
            if debug_assert_layer_hash_no_change:
                print("=" * 100)
                print("Tokenizing dataset...")
                print("=" * 100)
            self.tokenize_dataset(
                **self.tokenize_dataset_kwargs
            )  # now it should be stored in the cache
            assert self.tokenized_dataset_train_cached is not None
            if not all(
                di["input_ids"].ndim == 1 for di in self.tokenized_dataset_train_cached
            ):
                raise NotImplementedError("Not implemented yet, ndim != 1")
            if not all(
                di["input_ids"].shape[-1]
                == self.tokenized_dataset_train_cached[0]["input_ids"].shape[-1]
                for di in self.tokenized_dataset_train_cached
            ):
                raise NotImplementedError(
                    "Not implemented, shapes != first shape (i.e. not all equal ctx len)"
                )
            # 1. Get arguments to training etc..
            if debug_assert_layer_hash_no_change:
                print("=" * 100)
                print("Getting arguments to training + dataloader...")
                print("=" * 100)
            layers_module_name = self._get_layers_module_name()
            hook_fn = Ft.partial(
                filter_hook_fn,
                Ft.partial(
                    self._sae_filter_fn,
                    self.sae_model,
                ),
            )
            if debug_assert_layer_hash_no_change:
                print("=" * 100)
                print("Finetuning w/ hooks...")
                print("=" * 100)
            hook_name = f"{layers_module_name}.layers.{self.sae_cfg['layer']}"
            if debug_assert_layer_hash_no_change:
                print(f"Hook name: {hook_name}")
            sft_config = SFTConfig(
                # NOTE that you include output folder HERE so you will not be saving or anything
                # https://huggingface.co/docs/trl/v0.19.0/en/sft_trainer#language-modeling
                **self.sft_config_args,
            )
            # print(self.tokenized_dataset_train_cached) # DEBUG
            print(self.tokenized_dataset_train_cached[0])  # DEBUG
            # if "teacher_model" in self.train_args: # ???
            #     if self.trainer_cls_name == "LogitsTrainer":
            #         raise NotImplementedError("teacher_model is default-created by the FintuneAfterSAETrainer, you cannot pass it in")
            #     else:
            #         raise ValueError(f"teacher_model is not supported for this trainer; trainer={self.trainer_cls_name}")
            if self.trainer_cls_name == "LogitsTrainer":
                assert hasattr(self, "teacher_model"), "teacher_model is not set"
                self.train_args["teacher_model"] = self.teacher_model
            trainer = self.trainer_cls(
                self.model,
                train_dataset=self.tokenized_dataset_train_cached,
                args=sft_config,
                **self.train_args,
            )
            pname2hash_start = (
                self._pname2hashes(self.model)
                if debug_assert_layer_hash_no_change
                else None
            )
            # pname2hash_teacher_start = None
            # if self.trainer_cls_name == "LogitsTrainer": # TODO
            #     pname2hash_teacher_start = self._pname2hashes(self.teacher_model)
            with named_forward_hooks(
                self.model,
                {
                    hook_name: hook_fn,
                },
            ):
                trainer.train()
            pname2hash_end = (
                self._pname2hashes(self.model)
                if debug_assert_layer_hash_no_change
                else None
            )
            # pname2hash_teacher_end = None
            # if self.trainer_cls_name == "LogitsTrainer": # TODO
            #     pname2hash_teacher_end = self._pname2hashes(self.teacher_model)
            if debug_assert_layer_hash_no_change:
                self._assert_pname2hashes_equal(
                    pname2hash_start,
                    pname2hash_end,
                    self.pname2should_freeze,
                )
            # TODO (error because set of keys not same? peft? idk)
            # assert (pname2hash_teacher_start is None) == (pname2hash_teacher_end is None)
            # if pname2hash_teacher_start is not None:
            #     self._assert_pname2hashes_equal(
            #         pname2hash_teacher_start,
            #         pname2hash_teacher_end,
            #         # All frozen because teacher does not train
            #         {pname: True for pname in pname2hash_teacher_start.keys()},
            #     )
        finally:
            if log_to_wandb:
                wandb.finish()

    def save(self, save_dir: Path) -> None:
        self.model.save_pretrained(save_dir.as_posix())
        self.tokenizer.save_pretrained(save_dir.as_posix())


@click.group()
def cli():
    pass


@cli.command()
@click.option("--arguments-file", "-f", type=str, required=True)
@click.option("--log-to-wandb", "-wandb", is_flag=True, default=False)
@click.option(
    "--debug-assert-layer-hash-no-change", "-debug", is_flag=True, default=False
)
@click.option("--save-dir", "-o", type=str, required=True)
def finetune(
    arguments_file: str,
    log_to_wandb: bool,
    debug_assert_layer_hash_no_change: bool,
    save_dir: str,
):
    """
    For initial testing run like this: ```bash
    python3 finetune_after_sae.py finetune -f finetune_after_sae_kwargs.json -wandb -debug
    ```
    """
    arguments_file = Path(arguments_file)
    save_dir = Path(save_dir)
    if (
        not arguments_file.exists()
        and arguments_file.name.endswith(".json")
        and arguments_file.is_file()
    ):
        raise ValueError(
            f"Arguments file {arguments_file} does not exist, or not a file (or does not end with .json)"
        )
    if save_dir.exists() and len(list(save_dir.iterdir())) > 0:
        raise ValueError(f"Save directory {save_dir} is not empty")
    args = json.loads(arguments_file.read_text())
    if not args:
        raise ValueError(f"Arguments file {arguments_file} is empty")
    finetuner = FinetuneAfterSAETrainer(**args)
    finetuner.finetune(
        log_to_wandb=log_to_wandb,
        debug_assert_layer_hash_no_change=debug_assert_layer_hash_no_change,
    )
    finetuner.save(save_dir)


@cli.command()
def create_spylab_camel_ai_biology_dataset():
    """
    Create a dataset using the spylab prompt template using the camel-ai/biology dataset
    as our initial dataset. This is going to create a dataset of "text"-keyed dicts with
    each element being a string that has the prompt-templatted single question and
    single answer from the camel-ai/biology dataset (i.e. `message_1` and `message_2`
    keys). There is no system prompt.

    Run like this: ```bash
    python3 finetune_after_sae.py create-spylab-camel-ai-biology-dataset
    ```
    (because this will store/cache the datasets as `.jsonl` in CWD you should not need
    to run this more than once).

    NOTE chunking vs. padding is meant to be handled LATER by the tokenization process
    (this part just handles system prompting and chat templatting).
    """
    # 0. Declare our target execution
    split2path = {
        "train": "./camel_ai_biology_templatted_train.jsonl",
        "validation": "./camel_ai_biology_templatted_val.jsonl",
        "test": "./camel_ai_biology_templatted_test.jsonl",
    }
    sizes = {
        "train": 16_000,
        "validation": 2_000,
        "test": 2_000,
    }
    # 1. Load the dataset
    dataset = load_dataset("camel-ai/biology", split="train")
    assert len(dataset) == sum(sizes.values()), (
        f"len(dataset)={len(dataset)} "
        + f"!=\nsum(sizes.values()) from\n{json.dumps(sizes, indent=4)}"
    )
    # 2. Shuffle for randomness
    shuffled_dataset = dataset.shuffle(seed=77)
    # 3. Split the dataset into the desired sizes
    ordered_sizes = list(sizes.keys())
    start_idx = 0
    split2dataset = {}
    for split in ordered_sizes:
        end_idx = start_idx + sizes[split]
        split2dataset[split] = shuffled_dataset.select(range(start_idx, end_idx))
        start_idx = end_idx
    split2_templatted_dataset = {}
    for split, ds in split2dataset.items():
        assert "message_1" in ds.features
        assert "message_2" in ds.features
        is_lat = False
        dataset_prompts: List[str] = [
            SpylabPreprocessor.preprocess_sentence_old(
                di["message_1"],
                response=di["message_2"],
                trojan_suffix=None,
                include_begin=True,
                is_lat=is_lat,
            )
            for di in ds
        ]
        split2_templatted_dataset[split] = Dataset.from_list(
            [{"text": p} for p in dataset_prompts]
        )
    # 5. Save the dataset
    for split, ds in split2_templatted_dataset.items():
        ds.to_json(split2path[split], lines=True)


if __name__ == "__main__":
    cli()
