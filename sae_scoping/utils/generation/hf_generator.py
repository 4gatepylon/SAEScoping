from __future__ import annotations
import torch
from beartype.typing import Any, Generator, Iterable
from copy import deepcopy
from beartype import beartype
from transformers import PreTrainedModel, PreTrainedTokenizerBase, BatchEncoding


class HFGenerator:
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
    ):
        self.model = model
        self.tokenizer = tokenizer

    @beartype
    def generate_stream(
        self,
        conversations: Iterable[list[dict[str, str]]],
        batch_size: int = 32,
        generation_kwargs: dict[str, Any] = {
            # Greedy-sample short responses as default
            "min_length": -1,
            "max_new_tokens": 512,
            "do_sample": False,
        },
        max_length: int = 16384,
    ) -> Generator[str, None, None]:
        """
        Generate responses for a list of conversations.
        """
        # from sae_scoping.utils.generation.messages import is_valid_0turn_messages, is_valid_1turn_messages
        # assert all(self._is_valid_convo(conversation) for conversation in conversations)
        # assert all(self._is_0turn_convo(conversation) for conversation in conversations)
        with torch.no_grad():
            for i in range(0, len(conversations), batch_size):
                # Preprocessing
                conversation_batch: list[list[dict[str, str]]] = conversations[i : i + batch_size]
                text_batch: list[str] = [self.tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True) for conversation in conversation_batch]
                batch_encoding: BatchEncoding = self.tokenizer(text_batch, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
                inputs: dict[str, torch.Tensor] = {k: v.to(self.model.device) for k, v in batch_encoding.items()}
                input_length: int = inputs["input_ids"].shape[1]
                kwargs: dict[str, Any] = deepcopy(generation_kwargs)
                assert len(set(inputs.keys()) & set(generation_kwargs.keys())) == 0
                kwargs.update(inputs)

                # Inference
                outputs: torch.Tensor = self.model.generate(**kwargs)
                assert outputs.shape[0] == inputs["input_ids"].shape[0] == len(conversation_batch)
                assert outputs.shape[1] >= input_length
                decoded_outputs: list[str] = self.tokenizer.batch_decode(outputs[:, input_length:], skip_special_tokens=True)
                assert len(decoded_outputs) == len(conversation_batch)
                for decoded_output in decoded_outputs:
                    yield decoded_output
