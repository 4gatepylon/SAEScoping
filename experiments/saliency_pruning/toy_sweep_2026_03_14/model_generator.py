from __future__ import annotations

from pathlib import Path
import click
import json
import torch
from typing import Any, Generator, TypedDict
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizerBase,
    AutoModelForCausalLM,
    AutoTokenizer,
)

"""
Run with a command like the follows (for gemma2-9b-it):
```
CUDA_VISIBLE_DEVICES=0 python3 model_generator.py tests/chats_only_question.json \
    --model-name-or-path google/gemma-2-9b-it \
    --device cuda \
    --batch-size 2 \
    --chat-template-file prompts/gemma2_chat_template_system_prompt.j2 \
    --generation_kwargs '{"min_length": -1, "max_new_tokens": 32, "do_sample": false}'
```
"""


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


class OpenAIMessage(TypedDict):
    role: str
    content: str


OpenAIMessages = list[OpenAIMessage]

_VALID_ROLES = {"system", "user", "assistant"}


# ---------------------------------------------------------------------------
# Standalone validation helpers (public)
# ---------------------------------------------------------------------------


def is_valid_messages(messages: object) -> bool:
    """Check that messages is a list of dicts each with str 'role' and 'content'."""
    if not isinstance(messages, list) or len(messages) == 0:
        return False
    for msg in messages:
        if not isinstance(msg, dict):
            return False
        if "role" not in msg or "content" not in msg:
            return False
        if not isinstance(msg["role"], str) or not isinstance(msg["content"], str):
            return False
        if msg["role"] not in _VALID_ROLES:
            return False
    return True


def is_valid_0turn_messages(messages: object) -> bool:
    """Valid messages where the last non-system message is from the user (no assistant response yet)."""
    if not is_valid_messages(messages):
        return False
    non_system = [m for m in messages if m["role"] != "system"]
    if len(non_system) == 0:
        return False
    return non_system[-1]["role"] == "user"


def is_valid_1turn_messages(messages: object) -> bool:
    """Valid messages where the last two non-system messages are [user, assistant]."""
    if not is_valid_messages(messages):
        return False
    non_system = [m for m in messages if m["role"] != "system"]
    if len(non_system) < 2:
        return False
    return non_system[-2]["role"] == "user" and non_system[-1]["role"] == "assistant"


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


class HFGenerator:
    """Perform batched generation on OpenAI-format messages."""

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        cache: dict[str, str] | None = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.cache = cache if cache is not None else {}

    @staticmethod
    def _is_valid_convo(conversation: OpenAIMessages) -> bool:
        return is_valid_messages(conversation)

    @staticmethod
    def _is_0turn_convo(conversation: OpenAIMessages) -> bool:
        return is_valid_0turn_messages(conversation)

    @staticmethod
    def _is_1turn_convo(conversation: OpenAIMessages) -> bool:
        return is_valid_1turn_messages(conversation)

    @staticmethod
    def _cache_key(conversation: OpenAIMessages) -> str:
        return json.dumps(conversation, sort_keys=True)

    def _generate_stream(
        self,
        conversations: list[OpenAIMessages],
        batch_size: int = 32,
        generation_kwargs: dict[str, Any] = {
            "min_length": -1,
            "max_new_tokens": 512,
            "do_sample": False,
        },
        return_indices: bool = False,
    ) -> Generator[OpenAIMessages | tuple[OpenAIMessages, int], None, None]:
        """Generate responses for a list of 0-turn conversations (ending with user)."""
        assert all(self._is_valid_convo(c) for c in conversations)
        assert all(self._is_0turn_convo(c) for c in conversations)

        # Separate cached vs uncached
        cached_results: dict[int, OpenAIMessages] = {}
        uncached_indices: list[int] = []
        uncached_conversations: list[OpenAIMessages] = []

        for idx, convo in enumerate(conversations):
            key = self._cache_key(convo)
            if self.cache is not None and key in self.cache:
                cached_results[idx] = convo + [
                    {"role": "assistant", "content": self.cache[key]}
                ]
            else:
                uncached_indices.append(idx)
                uncached_conversations.append(convo)

        # Generate for uncached conversations
        uncached_results: dict[int, OpenAIMessages] = {}
        if uncached_conversations:
            with torch.no_grad():
                texts_inputs: list[str] = [
                    self.tokenizer.apply_chat_template(
                        convo, tokenize=False, add_generation_prompt=True
                    )
                    for convo in uncached_conversations
                ]

                generated_index = 0
                for i in range(0, len(texts_inputs), batch_size):
                    batch_texts = texts_inputs[i : i + batch_size]
                    batch_convos = uncached_conversations[i : i + batch_size]
                    batch_orig_indices = uncached_indices[i : i + batch_size]

                    tokenized = self.tokenizer(
                        batch_texts,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                    )
                    tokenized = {
                        k: v.to(self.model.device) for k, v in tokenized.items()
                    }
                    input_length = tokenized["input_ids"].shape[1]
                    actual_batch_size = tokenized["input_ids"].shape[0]

                    outputs = self.model.generate(**tokenized, **generation_kwargs)
                    outputs_text = self.tokenizer.batch_decode(
                        outputs[:, input_length:], skip_special_tokens=True
                    )
                    assert len(outputs_text) == actual_batch_size

                    for convo, response, orig_idx in zip(
                        batch_convos, outputs_text, batch_orig_indices
                    ):
                        response_stripped = response.strip()
                        result_convo = convo + [
                            {"role": "assistant", "content": response_stripped}
                        ]
                        uncached_results[orig_idx] = result_convo
                        # Update cache
                        if self.cache is not None:
                            self.cache[self._cache_key(convo)] = response_stripped
                    generated_index += actual_batch_size

        # Yield in original order
        for idx in range(len(conversations)):
            result = cached_results.get(idx) or uncached_results[idx]
            assert self._is_valid_convo(result)
            assert self._is_1turn_convo(result)
            if return_indices:
                yield result, idx
            else:
                yield result

    def generate_stream(
        self,
        conversations: list[OpenAIMessages],
        **kwargs: Any,
    ) -> Generator[OpenAIMessages | tuple[OpenAIMessages, int], None, None]:
        """Public interface for streaming generation."""
        yield from self._generate_stream(conversations, **kwargs)

    def generate(
        self,
        conversations: list[OpenAIMessages],
        **kwargs: Any,
    ) -> list[OpenAIMessages]:
        """Non-streaming generation. Returns all results as a list."""
        return list(self._generate_stream(conversations, **kwargs))


# CLI
@click.command()
@click.argument(
    "messages_file", type=click.Path(exists=True, path_type=Path), required=True
)
@click.option("--model-name-or-path", type=str, required=True)
@click.option(
    "--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu")
)
@click.option("--batch-size", type=int, default=32)
@click.option("--generation_kwargs", type=str, default=r"{}")
@click.option(
    "--chat-template-file",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Path to a Jinja2 chat template file to override the tokenizer's default.",
)
def main(
    messages_file: str,
    model_name_or_path: str,
    device: str,
    batch_size: int,
    generation_kwargs: str,
    chat_template_file: Path | None,
):
    """Generate responses for a list of OpenAI-format messages."""
    if device == "cpu":
        click.confirm(
            "Are you sure you want to run on CPU? This may be very slow.", abort=True
        )
    generation_kwargs = json.loads(generation_kwargs)
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map=device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
        p.grad = None
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    tokenizer.padding_side = "left"
    if chat_template_file is not None:
        tokenizer.chat_template = chat_template_file.read_text()
    generator = HFGenerator(model, tokenizer)
    messages = json.loads(messages_file.read_text())
    outputs = generator.generate(
        messages, batch_size=batch_size, generation_kwargs=generation_kwargs
    )
    # Poor-man's JSON array output
    click.echo("[")
    for output in outputs:
        click.echo(json.dumps(output) + ",")
    click.echo("]")


if __name__ == "__main__":
    main()
