from __future__ import annotations

import torch
import json
import numpy as np
import tqdm

from typing import List, Dict, Any, Tuple, Literal, Optional
from transformers import AutoTokenizer, BatchEncoding
from functools import reduce


class LengthAwareCapableTokenizer:
    """Tokenizer wrapper that batches inputs by length for efficient inference."""

    def __init__(
        self,
        tokenizer: AutoTokenizer | str,
        tokenization_mode: Literal["length_aware", "regular_batched"],
        chat_template: Optional[str] = None,
        tokenizer_from_pretrained_kwargs: Dict[str, Any] = {},
    ):
        self.tokenizer = (
            AutoTokenizer.from_pretrained(tokenizer, **tokenizer_from_pretrained_kwargs)
            if isinstance(tokenizer, str)
            else tokenizer
        )
        self.tokenization_mode = tokenization_mode
        assert tokenization_mode in {"length_aware", "regular_batched"}
        if chat_template is not None:
            self.tokenizer.chat_template = chat_template

    ################ [BEGIN] Sanity checks helper functions [BEGIN] #################

    @staticmethod
    def _sanity_check_single_turn_conversation(
        conversation: List[Dict[str, Any]],
    ) -> bool:  # Best to return to hide in asserts for -o
        """Validate conversation format"""
        assert isinstance(conversation, list), (
            f"type={type(conversation)}\n\n{conversation}"
        )
        assert all(isinstance(item, dict) for item in conversation), f"{conversation}"
        assert all(set(item.keys()) == {"role", "content"} for item in conversation), (
            f"{conversation}"
        )
        assert all(isinstance(item["role"], str) for item in conversation), (
            f"{conversation}"
        )
        assert all(isinstance(item["content"], str) for item in conversation), (
            f"{conversation}"
        )
        # TODO(Adriano) we are going to want to relax this!
        assert all(
            item["role"] in {"system", "user", "assistant"} for item in conversation
        )
        # assert [item["role"] for item in conversation] == [
        #     "user",
        #     "assistant",
        # ], f"{conversation}"
        return True

    @staticmethod
    def _sanity_check_tokenized_inputs_single_batch(
        _supervised_inputs: BatchEncoding,
    ) -> bool:
        assert isinstance(_supervised_inputs, BatchEncoding)
        assert (
            len(
                set(_supervised_inputs.keys())
                & {
                    "input_ids",
                    "attention_mask",
                }
            )
            == 2
        ), f"{_supervised_inputs.keys()}"
        # Model not provided
        # assert _supervised_inputs["input_ids"].device == model.device
        # assert _supervised_inputs["attention_mask"].device == model.device
        _supervised_inputs["labels"] = _supervised_inputs["input_ids"]
        assert (
            _supervised_inputs["input_ids"].shape
            == _supervised_inputs["attention_mask"].shape
        )
        assert _supervised_inputs["input_ids"].ndim == 2
        return True

    @staticmethod
    def _sanity_check_tokenized_inputs_all_batches(
        _supervised_inputs_batch: List[Tuple[List[int], BatchEncoding]],
    ) -> bool:
        # 1. Sanity check the indices
        indices = reduce(
            lambda x, y: x + y, [ids for ids, _ in _supervised_inputs_batch]
        )
        assert len(set(indices)) == len(indices)
        assert list(range(len(indices))) == sorted(indices)
        # 2. Sanity check the contents
        assert all(
            LengthAwareCapableTokenizer._sanity_check_tokenized_inputs_single_batch(y)
            for _, y in _supervised_inputs_batch
        )
        return True

    ################ [END] Sanity checks helper functions [END] #################

    ################ [BEGIN] Length-aware helper functions [BEGIN] #################

    @staticmethod
    def get_max_tokens_per_batch(
        max_context_length: Optional[int],
        max_batch_size: Optional[int],
        max_tokens_per_batch: Optional[int],
    ) -> int:
        if max_tokens_per_batch is not None:
            if any(x is not None for x in [max_context_length, max_batch_size]):
                raise ValueError(
                    "Cannot specify both max_tokens_per_batch and "
                    + "max_context_length or max_batch_size"
                )
            return max_tokens_per_batch
        else:
            if any(x is None for x in [max_context_length, max_batch_size]):
                raise ValueError(
                    "Must specify either max_tokens_per_batch or "
                    + "max_context_length and max_batch_size"
                )
            return max_context_length * max_batch_size

    def get_token_length_estimate(
        self,
        conversations: List[str],  # other forms do not need support
        token_count_estimator: Literal["char_length", "token_length"] = "token_length",
        # Tokenization for length estimation specifically happens in batches for speed
        # and tqdm visibility.
        token_count_estimator_kwargs: Dict[str, Any] = {
            "tokenizer_kwargs": {
                "padding": "longest",
                "truncation": True,
                # NOTE: this will reduce to model max length!
                # https://huggingface.co/docs/transformers/en/main_classes/tokenizer#transformers.PreTrainedTokenizer.__call__.max_length
                "max_length": None,
                "padding_side": "left",
                "return_tensors": "pt",
            },
        },
        token_estimator_batch_size: int = 1024,
    ) -> List[int]:
        """
        Estimate the number of tokens in a conversation by counting the number of
        characters.

        This is what is used to sort for the purposes of length-aware tokenization.
        """
        assert all(isinstance(conversation, str) for conversation in conversations)
        token_count_estimator_kwargs["batch_size"] = token_estimator_batch_size
        if token_count_estimator == "char_length":
            return [len(conversation) for conversation in conversations]
        elif token_count_estimator == "token_length":
            conversations_w_idxs: List[Tuple[int, str]] = list(enumerate(conversations))
            # Sort by length (this is merely a proxy for getting tokenization to work
            # faster)
            conversations_w_idxs = sorted(conversations_w_idxs, key=lambda x: len(x[1]))
            conversations_w_idxs_batched: List[List[Tuple[int, str]]] = [
                conversations_w_idxs[
                    i : min(
                        i + token_count_estimator_kwargs["batch_size"],
                        len(conversations_w_idxs),
                    )
                ]
                for i in range(
                    0,
                    len(conversations_w_idxs),
                    token_count_estimator_kwargs["batch_size"],
                )
            ]
            batches_tokenized: List[BatchEncoding] = [
                self.tokenizer(
                    [conversation for _, conversation in batch],  # extract strings
                    **token_count_estimator_kwargs["tokenizer_kwargs"],
                )
                for batch in tqdm.tqdm(
                    conversations_w_idxs_batched,
                    desc=f"Tokenizing batches for length estimate w/ batch_size={token_estimator_batch_size}",
                )
            ]
            attention_masks: List[torch.Tensor] = [
                batch_tokenized["attention_mask"]
                for batch_tokenized in batches_tokenized
            ]
            assert all(
                isinstance(attention_mask, torch.Tensor)
                for attention_mask in attention_masks
            )
            assert all(attention_mask.ndim == 2 for attention_mask in attention_masks)
            lengths_batched: List[torch.Tensor] = [
                attention_mask.sum(dim=1) for attention_mask in attention_masks
            ]
            assert all(isinstance(length, torch.Tensor) for length in lengths_batched)
            assert all(length.ndim == 1 for length in lengths_batched)
            assert sum(length.shape[0] for length in lengths_batched) == len(
                conversations_w_idxs
            )
            _lengths: List[int] = []
            for _lengths_batched in lengths_batched:
                _lengths.extend(_lengths_batched.tolist())
            # re-combine with the indices so we can sort back into the original desired
            # order basically
            lengths: List[Tuple[int, int]] = [
                (i, l) for (i, _), l in zip(conversations_w_idxs, _lengths)
            ]
            lengths = sorted(lengths, key=lambda x: x[0])
            assert [i for i, _ in lengths] == list(range(len(conversations_w_idxs)))
            lengths_ret: List[int] = [l for _, l in lengths]

            # Sanity check the lengths match exactly here w/ one tokenization
            assert (
                self.tokenizer(
                    conversations[:32],
                    **token_count_estimator_kwargs["tokenizer_kwargs"],
                )["attention_mask"]
                .sum(dim=1)
                .tolist()
                == lengths_ret[:32]  # sanity check here
            )
            _truncation = token_count_estimator_kwargs.get("tokenizer_kwargs", {}).get(
                "truncation", False
            )
            _max_length = token_count_estimator_kwargs.get("tokenizer_kwargs", {}).get(
                "max_length", None
            )
            assert all(
                0
                <= l
                # +2 for special tokens (is this right? idk yolo plz be right...
                # BOS and EOS assumed)
                <= (
                    len(c) + 2
                    if (not _truncation) or _max_length is None
                    else _max_length
                )
                for l, c in zip(lengths_ret, conversations)
            ), (
                f"lengths_ret, conv_lens zipped={list(zip(lengths_ret, [len(c) for c in conversations]))},"
                + f"truncation={_truncation}, "
                + f"max_length={_max_length}, "
                + f"token_count_estimator_kwargs={json.dumps(token_count_estimator_kwargs, indent=4)}"
            )
            # NOTE, Claude claims: "token length can actually be greater than character
            # length (e.g., for languages with multi-byte characters or when BPE splits
            # words"
            #
            # ... does it matter for us? idk; idts...
            return lengths_ret
        else:
            raise ValueError(f"Invalid token_count_estimator: {token_count_estimator}")

    def get_length_aware_padding_breakpoints(
        self,
        lengths: List[int],
        padding: str = "longest",
        truncation: bool = True,
        max_length: int = 32768,  # Please set this!
    ) -> List[int]:
        assert padding == "longest"  # otherwise not supported :/
        # 0. Make sure all lengths will fit here
        # (if truncation assume we will truncate, otherwise we will throw an error if
        # we are below max_length)
        if not truncation and any(l > max_length for l in lengths):
            raise ValueError(f"Some lengths are greater than max_length: {max_length}")
        if truncation:
            lengths = [min(l, max_length) for l in lengths]  # simulate truncation
        # 1. Make sure it's sorted (that's the point of length_aware)
        assert len(lengths) > 0
        lis, ljs = lengths[:-1], lengths[1:]
        assert all(i <= j for i, j in zip(lis, ljs)), f"lengths={lengths}"
        assert len(lis) == len(ljs) == len(lengths) - 1
        # 2. Get corr. lengths
        batch_lengths: List[List[int]] = [[lengths[0]]]
        for i in range(1, len(lengths)):
            candidate_btch_len = len(batch_lengths[-1]) + 1
            candidate_ctx_len = max(lengths[i], max(batch_lengths[-1]))
            candidate_n_tokens = candidate_btch_len * candidate_ctx_len
            if candidate_n_tokens > max_length:
                batch_lengths.append([lengths[i]])
            else:
                batch_lengths[-1].append(lengths[i])
        assert all(len(b) > 0 for b in batch_lengths)
        # 3. Get padding breakpoints from the batch lengths (NOT context lengths inside)
        padding_breakpoints = [0]
        for i in range(0, len(batch_lengths)):
            padding_breakpoints.append(padding_breakpoints[-1] + len(batch_lengths[i]))
        # 4. Sanity check that this is correct
        assert padding_breakpoints[0] == 0
        assert padding_breakpoints[-1] == len(lengths)
        _is, _js = padding_breakpoints[:-1], padding_breakpoints[1:]
        assert all(i < j for i, j in zip(_is, _js))
        assert (
            len(_is) == len(_js) == len(batch_lengths) == len(padding_breakpoints) - 1
        )
        # 5. Return
        return padding_breakpoints

    ################ [END] Length-aware helper functions [END] #################

    @staticmethod
    def _get_conversations_texts_tokens_type(
        conversations_or_texts_or_tokens: (
            List[Dict[str, str]] | List[str] | List[List[int]]
        ),
    ) -> Literal["conversations", "texts", "tokens"]:
        is_list = isinstance(conversations_or_texts_or_tokens, list)
        is_all_list = all(isinstance(x, list) for x in conversations_or_texts_or_tokens)
        is_all_str = all(isinstance(x, str) for x in conversations_or_texts_or_tokens)
        is_all_list_of_int = all(
            all(isinstance(x, int) for x in xs)
            for xs in conversations_or_texts_or_tokens
        )
        is_all_list_of_dict = all(
            isinstance(xs, list) and all(isinstance(x, dict) for x in xs)
            for xs in conversations_or_texts_or_tokens
        )

        # Classify and sanity check
        is_tokens = is_list and is_all_list and is_all_list_of_int
        is_texts = not is_tokens and is_list and is_all_str
        is_conversations = is_list and is_all_list_of_dict
        assert not is_conversations or all(
            LengthAwareCapableTokenizer._sanity_check_single_turn_conversation(x)
            for x in conversations_or_texts_or_tokens
        )
        # This one below is not actually done because we need to convert to
        # `BatchEncoding`
        # assert not is_tokens or all(
        #     LengthAwareCapableTokenizer._sanity_check_tokenized_inputs(x)
        #     for x in conversations_or_texts_or_tokens
        # )
        if is_tokens:
            return "tokens"
        elif is_texts:
            return "texts"
        elif is_conversations:
            return "conversations"
        else:
            raise ValueError(
                f"Invalid type: {type(conversations_or_texts_or_tokens)}\n\n{conversations_or_texts_or_tokens}"
            )

    def tokenize_token_list2token_pt_BE(
        self,
        tok: List[List[int]],
        padding_breakpoints: List[int] | np.ndarray | torch.Tensor,
        # KWARGS
        padding_side: str = "left",
        padding: str = "longest",
        truncation: bool = False,
        max_length: int = 32768,
        tokenization_kwargs: Dict[str, Any] = {},
    ) -> List[BatchEncoding]:
        if not truncation and any(len(t) > max_length for t in tok):
            raise ValueError(f"Some tokens are longer than max_length: {max_length}")
        # Truncate
        if truncation:
            tok = [t[:max_length] for t in tok]
        # Sanity the breakpoints
        assert padding_breakpoints[0] == 0
        assert padding_breakpoints[-1] == len(tok)
        _is = padding_breakpoints[:-1]
        _js = padding_breakpoints[1:]
        assert all(i < j for i, j in zip(_is, _js))
        tok_batches = [tok[i : min(j, len(tok))] for i, j in zip(_is, _js)]
        longest_lens = [max(len(t) for t in tb) for tb in tok_batches]
        assert len(longest_lens) == len(tok_batches)
        surplus_lens = [[ll - len(t) for t in tb] for tb, ll in zip(tok_batches, longest_lens)]  # fmt: skip
        assert all(all(sl >= 0 for sl in sls) for sls in surplus_lens)

        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            raise ValueError("pad_token_id is not set")
        pad_batches = [[[pad_token_id] * sl for sl in sls] for sls in surplus_lens]
        tok_batches_padded = None
        attn_masks_batches = None
        if padding_side == "left":
            tok_batches_padded = [[p + t for p, t in zip(pb, tb)] for tb, pb in zip(tok_batches, pad_batches)]  # fmt: skip
            attn_masks_batches = [[[0] * len(p) + [1] * len(t) for p, t in zip(pb, tb)] for tb, pb in zip(tok_batches, pad_batches)]  # fmt: skip
        elif padding_side == "right":
            tok_batches_padded = [[t + p for t, p in zip(tb, pb)] for tb, pb in zip(tok_batches, pad_batches)]  # fmt: skip
            attn_masks_batches = [[[1] * len(t) + [0] * len(p) for t, p in zip(tb, pb)] for tb, pb in zip(tok_batches, pad_batches)]  # fmt: skip
        else:
            raise ValueError(f"Invalid padding side: {padding_side}")
        assert tok_batches_padded is not None
        assert attn_masks_batches is not None
        tok_batches_padded_pt_BE = []
        for tb, atm in zip(tok_batches_padded, attn_masks_batches):
            tok_batches_padded_pt_BE.append(
                BatchEncoding(
                    {
                        "input_ids": torch.tensor(tb),
                        "attention_mask": torch.tensor(atm),
                    }
                )
            )
            assert self._sanity_check_tokenized_inputs_single_batch(
                tok_batches_padded_pt_BE[-1]
            )
        return tok_batches_padded_pt_BE

    def get_regular_batched_padding_breakpoints(
        self,
        tokens: List[List[int]],
        batch_size: int,
    ) -> np.ndarray:
        breakpoints = [i for i in range(0, len(tokens), batch_size)]
        if breakpoints[-1] != len(tokens):
            breakpoints.append(len(tokens))
        return np.array(breakpoints)

    def length_aware_tokenize_conversations_or_texts_or_tokens(
        self,
        conversations_or_texts_or_tokens: (
            List[Dict[str, str]] | List[str] | List[List[int]]
        ),
        tokens_per_batch: int = -1,
        tokenization_kwargs: Dict[
            str, Any
        ] = {},  # pass-through to the tokenizer called
        apply_chat_template_kwargs: Dict[
            str, Any
        ] = {},  # add_generation_prompt, etc...
        token_count_estimator: Literal["char_length", "token_length"] = "token_length",
        token_estimator_batch_size: int = 1024,
    ) -> List[Tuple[List[int], BatchEncoding]]:
        if len(conversations_or_texts_or_tokens) == 0:
            return []
        _type = self._get_conversations_texts_tokens_type(
            conversations_or_texts_or_tokens
        )
        if _type == "tokens":
            # TODO(Adriano) duplicated code? could this be avoided maybe?
            if tokens_per_batch == -1:
                raise ValueError("tokens_per_batch must be provided (got -1)")
            tokens_with_indices = list(enumerate(conversations_or_texts_or_tokens))
            tokens_with_indices.sort(key=lambda x: len(x[1]), reverse=False)  # by len
            padding_side = tokenization_kwargs.get("padding_side", "left")
            padding = tokenization_kwargs.get("padding", "longest")
            if padding != "longest":
                raise NotImplementedError("Not implemented w/ padding != longest")
            # No truncation sine max length is not known
            truncation = tokenization_kwargs.get("truncation", True)
            max_length = tokenization_kwargs.get("max_length", tokens_per_batch)
            if not truncation:
                raise NotImplementedError("Not implemented w/ truncation = False")
            if any(len(t) > max_length for t in [t for _, t in tokens_with_indices]):
                raise ValueError(f"Some tokens r longer than max length: {max_length}")
            padding_breakpoints = self.get_length_aware_padding_breakpoints(
                [len(t) for _, t in tokens_with_indices],
                padding=padding,
                truncation=truncation,
                max_length=max_length,
            )
            BEs = self.tokenize_token_list2token_pt_BE(
                [t for _, t in tokens_with_indices],
                padding_breakpoints,
                padding_side=padding_side,
                padding=padding,
                truncation=truncation,
                max_length=max_length,
                tokenization_kwargs=tokenization_kwargs,
            )
            assert padding_breakpoints[0] == 0
            assert len(BEs) == len(padding_breakpoints) - 1
            _is = padding_breakpoints[:-1]
            _js = padding_breakpoints[1:]
            assert padding_breakpoints[-1] == len(tokens_with_indices)
            assert all(i < j for i, j in zip(_is, _js))
            indices = [list(range(i, j)) for i, j in zip(_is, _js)]
            assert len(indices) == len(BEs)
            assert all("input_ids" in b for b in BEs)
            assert all(b["input_ids"].ndim == 2 for b in BEs)
            assert sum(len(i) for i in indices) == len(tokens_with_indices), (
                f"indiceslens={[len(i) for i in indices]}, len(tokens)={len(tokens_with_indices)}"
            )
            assert sum(b["input_ids"].shape[0] for b in BEs) == len(
                tokens_with_indices
            ), (
                f"belens={[b['input_ids'].shape for b in BEs]}, len(tokens)={len(tokens_with_indices)}"
            )
            assert all(
                len(i) == b["input_ids"].shape[0] for i, b in zip(indices, BEs)
            ), (
                f"indiceslens={[len(i) for i in indices]}, belens={[b['input_ids'].shape for b in BEs]}"
            )
            assert all(isinstance(b, BatchEncoding) for b in BEs)
            ret = list(zip(indices, BEs))
            assert self._sanity_check_tokenized_inputs_all_batches(ret)
            return ret
        elif _type == "texts":
            if not tokenization_kwargs.get("truncation", True):
                raise NotImplementedError("Not implemented w/ truncation = False")
            if tokenization_kwargs.get("padding", "longest") != "longest":
                raise NotImplementedError("Not implemented w/ padding != longest")
            print(
                f"Getting token length estimate... w/ batch_size={token_estimator_batch_size}"
            )  # DEBUG
            lengths = self.get_token_length_estimate(
                conversations_or_texts_or_tokens,
                token_count_estimator=token_count_estimator,
                token_count_estimator_kwargs={
                    "tokenizer_kwargs": {
                        "padding": "longest",  # shouldn't matter tbh
                        "truncation": False,  # this could matter...
                        # NOTE: this will reduce to model max length!
                        # https://huggingface.co/docs/transformers/en/main_classes/tokenizer#transformers.PreTrainedTokenizer.__call__.max_length
                        "max_length": None,
                        "padding_side": "left",  # shouldn't matter tbh
                        "return_tensors": "pt",  # shouldn't matter either
                    },
                },
                token_estimator_batch_size=token_estimator_batch_size,
            )
            print("DONE Getting token length estimate!")  # DEBUG
            if any(l > tokens_per_batch for l in lengths):
                raise ValueError(
                    f"Some tokens r longer than tokens_per_batch: {tokens_per_batch}, max length found={max(lengths)}"
                )
            # First sort, then breakpoint
            _texts = conversations_or_texts_or_tokens
            assert len(_texts) == len(lengths)
            texts_w_idxes_lens = list(zip(_texts, list(range(len(_texts))), lengths))
            texts_w_idxes_lens.sort(key=lambda x: x[2], reverse=False)  # by length

            # Extract this in a reverse-permutable way
            indices = [i for _, i, _ in texts_w_idxes_lens]
            texts = [t for t, _, _ in texts_w_idxes_lens]
            lengths = [l for _, _, l in texts_w_idxes_lens]
            assert len(indices) == len(texts) == len(lengths)
            assert all(isinstance(i, int) for i in indices)
            assert all(isinstance(t, str) for t in texts)
            assert all(isinstance(l, int) for l in lengths)

            print("Getting length-aware padding breakpoints...")  # DEBUG
            padding_breakpoints = self.get_length_aware_padding_breakpoints(
                lengths,
                padding="longest",
                truncation=True,
                max_length=tokens_per_batch,
            )
            print("DONE Tokenizing...")  # DEBUG
            BEs = []
            assert padding_breakpoints[0] == 0
            _is, _js = padding_breakpoints[:-1], padding_breakpoints[1:]
            assert len(_is) == len(_js)
            assert len(_is) <= len(texts)  # at least one per batch
            assert len(texts) == len(indices) == len(lengths)
            assert all(i < j for i, j in zip(_is, _js))
            for i, j in tqdm.tqdm(
                zip(_is, _js), total=len(_is), desc="Tokenizing texts into BEs"
            ):
                these_texts = texts[i:j]  # NOTE: use the permuted version
                these_indices = indices[i:j]  # permuted indices
                assert len(these_texts) == j - i == len(these_indices), (
                    f"len(these_texts)={len(these_texts)}, j={j}, i={i}, len(these_indices)={len(these_indices)}"
                )
                assert all(isinstance(t, str) for t in these_texts)
                these_tokens = self.tokenizer(
                    these_texts,
                    **tokenization_kwargs,
                )
                assert isinstance(these_tokens, BatchEncoding)
                BEs.append((these_indices, these_tokens))
                self._sanity_check_tokenized_inputs_single_batch(these_tokens)
            assert self._sanity_check_tokenized_inputs_all_batches(BEs)
            return BEs
        elif _type == "conversations":
            assert "tokenize" not in tokenization_kwargs
            texts = self.tokenizer.apply_chat_template(
                conversations_or_texts_or_tokens,
                tokenize=False,
                **apply_chat_template_kwargs,
            )
            return self.length_aware_tokenize_conversations_or_texts_or_tokens(
                texts,
                tokens_per_batch=tokens_per_batch,
                tokenization_kwargs=tokenization_kwargs,
                # apply_chat_template_kwargs=apply_chat_template_kwargs, # not needed
                token_count_estimator=token_count_estimator,
                token_estimator_batch_size=token_estimator_batch_size,
            )
        else:
            raise ValueError(f"Invalid type: {_type}")

    def regular_batched_tokenize_conversations_or_texts_or_tokens(
        self,
        conversations_or_texts_or_tokens: (
            List[Dict[str, str]] | List[str] | List[List[int]]
        ),
        batch_size: int = -1,
        context_length: int = -1,
        tokenization_kwargs: Dict[
            str, Any
        ] = {},  # pass-through to the tokenizer called
        apply_chat_template_kwargs: Dict[
            str, Any
        ] = {},  # add_generation_prompt, etc...
    ) -> List[Tuple[List[int], BatchEncoding]]:
        if len(conversations_or_texts_or_tokens) == 0:
            return []
        if batch_size == -1 or context_length == -1:
            raise ValueError(
                "batch_size and context_length must be provided, got"
                + f"batch_size={batch_size}, context_length={context_length}"
            )
        _type = self._get_conversations_texts_tokens_type(
            conversations_or_texts_or_tokens
        )
        if _type == "tokens":
            if len(tokenization_kwargs) > 0:
                padding_side = tokenization_kwargs.get("padding_side", "left")
                padding = tokenization_kwargs.get("padding", "longest")
                if padding != "longest":
                    raise NotImplementedError("Not implemented w/ padding != longest")
                # No truncation sine max length is not known
                truncation = tokenization_kwargs.get("truncation", True)
                max_length = tokenization_kwargs.get("max_length", context_length)
                if max_length > context_length:
                    raise ValueError(
                        f"max_length is greater than context length: {max_length} > {context_length}"
                    )
                if not truncation:
                    raise NotImplementedError("Not implemented w/ truncation = False")
            if any(len(t) > max_length for t in conversations_or_texts_or_tokens):
                raise ValueError(
                    f"Some tokens are longer than max length: {max_length}"
                )
            padding_breakpoints = self.get_regular_batched_padding_breakpoints(
                conversations_or_texts_or_tokens,
                batch_size,
            )
            BEs = self.tokenize_token_list2token_pt_BE(
                conversations_or_texts_or_tokens,
                padding_breakpoints,
                padding_side=padding_side,
                padding=padding,
                truncation=truncation,
                max_length=max_length,
                tokenization_kwargs=tokenization_kwargs,
            )
            indices = [
                list(
                    range(i, min(i + batch_size, len(conversations_or_texts_or_tokens)))
                )
                for i in range(0, len(conversations_or_texts_or_tokens), batch_size)
            ]
            # Because it's just batched, we can do this a-posteriori (nothing in the
            # calls above reorders this).
            assert len(indices) == len(BEs)
            assert all("input_ids" in b for b in BEs)
            assert all(len(b["input_ids"]) == batch_size for b in BEs)
            assert all(len(i) == batch_size for i in indices)
            assert all(isinstance(b, BatchEncoding) for b in BEs)
            ret = list(zip(indices, BEs))
            assert self._sanity_check_tokenized_inputs_all_batches(ret)
            return ret
        elif _type == "texts":
            tokens_batches: List[Tuple[List[int], BatchEncoding]] = []
            texts = conversations_or_texts_or_tokens
            for i in range(0, len(texts), batch_size):
                these_texts = texts[i : min(i + batch_size, len(texts))]
                these_indices = list(range(i, min(i + batch_size, len(texts))))
                these_tokens = self.tokenizer(
                    these_texts,
                    **tokenization_kwargs,
                )
                tokens_batches.append((these_indices, these_tokens))
            assert self._sanity_check_tokenized_inputs_all_batches(tokens_batches)
            return tokens_batches
        elif _type == "conversations":  # pasthrough after applying chat template
            assert "tokenize" not in tokenization_kwargs
            print("Applying chat template...")  # DEBUG
            texts = self.tokenizer.apply_chat_template(
                conversations_or_texts_or_tokens,
                tokenize=False,
                **apply_chat_template_kwargs,
            )
            print("DONE Applying chat template...")  # DEBUG
            return self.regular_batched_tokenize_conversations_or_texts_or_tokens(
                texts,
                batch_size,
                context_length,
                tokenization_kwargs=tokenization_kwargs,
                # apply_chat_template_kwargs=apply_chat_template_kwargs, # not needed
            )
        else:
            raise ValueError(f"Invalid type: {_type}")

    def __call__(
        self,
        conversations_or_texts_or_tokens: (
            List[Dict[str, str]] | List[str] | List[List[int]]
        ),
        **kwargs,  # Forwarded to the callee
    ) -> List[Tuple[List[int], BatchEncoding]]:
        """
        Take in something that can be tokenized and return a tokenized version of it
        where the tokenization is split across multiple batches (each list item corr.
        to a batch) and the the indicesin the original sequence are also kep-track of
        (since sorting is used by length for length-aware tokenization, which may cause
        the indices to be out of order).
        """
        if self.tokenization_mode == "length_aware":
            if len(set(kwargs.keys()) & {"tokens_per_batch"}) != 1:
                raise ValueError(
                    f"Invalid kwargs: {kwargs}. Must provide tokens_per_batch"
                )
            return self.length_aware_tokenize_conversations_or_texts_or_tokens(
                conversations_or_texts_or_tokens,
                **kwargs,  # Should include tokens_per_batch
            )
        elif self.tokenization_mode == "regular_batched":
            if len(set(kwargs.keys()) & {"batch_size", "context_length"}) != 2:
                raise ValueError(
                    f"Invalid kwargs: {kwargs}. Must provide batch_size and context_length"
                )
            return self.regular_batched_tokenize_conversations_or_texts_or_tokens(
                conversations_or_texts_or_tokens,
                **kwargs,  # Should include batch_size and context_length
            )
        else:
            raise ValueError(f"Invalid tokenization mode: {self.tokenization_mode}")

    def decode(
        self,
        tokens_list: List[
            Tuple[List[int], List[List[int] | np.ndarray | torch.Tensor]]
        ],
        skip_special_tokens: bool = True,
        decode_kwargs: Dict[str, Any] = {},
    ) -> List[str]:
        """
        Decode the tokens into text. Supports sequence order undoing (if necessary).


        In the future this may support sequence packing-based decoding (which involves
        splitting and then decoding, for example).

        TODO(Adriano) this interface fking sucks
        """
        decoded = [
            self.tokenizer.batch_decode(tokens, **decode_kwargs)
            for tokens, _ in tokens_list
        ]
        assert isinstance(decoded, list)
        assert all(isinstance(d, list) for d in decoded)
        assert all(len(d) == len(indices) for d, indices in zip(decoded, tokens_list))
        assert all(all(isinstance(d, str) for d in d_list) for d_list in decoded)
        decoded_w_indices: List[Tuple[int, str]] = []
        for decoded, (_, indices) in zip(decoded, tokens_list):
            decoded_w_indices.extend(
                [(index, decoded) for index, decoded in zip(indices, decoded)]
            )
        decoded_w_indices.sort(key=lambda x: x[0])  # sort to be in order by index
        assert list(range(len(decoded_w_indices))) == [i for i, _ in decoded_w_indices]  # fmt: skip
        return [d for _, d in decoded_w_indices]  # extract the decoded strings
