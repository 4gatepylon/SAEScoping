from __future__ import annotations

import torch
from beartype import beartype
from beartype.typing import Any, Generator
from transformers import PreTrainedModel, PreTrainedTokenizerBase

OpenAIMessages = list[dict[str, str]]

_VALID_ROLES = {"system", "user", "assistant"}


def _is_valid_messages(messages: list) -> bool:
    if not isinstance(messages, list) or not messages:
        return False
    return all(
        isinstance(m, dict) and set(m.keys()) == {"role", "content"}
        and m["role"] in _VALID_ROLES and isinstance(m["content"], str)
        for m in messages
    )


def _is_valid_0turn(messages: list) -> bool:
    """0-turn: ends with user (no assistant response yet)."""
    if not _is_valid_messages(messages):
        return False
    if len(messages) == 1:
        return messages[0]["role"] in {"user", "system"}
    if len(messages) == 2:
        return messages[0]["role"] == "system" and messages[1]["role"] == "user"
    return False


def _is_valid_1turn(messages: list) -> bool:
    """1-turn: ends with assistant response."""
    if not _is_valid_messages(messages):
        return False
    if len(messages) == 2:
        return messages[0]["role"] in {"user", "system"} and messages[1]["role"] == "assistant"
    if len(messages) == 3:
        return messages[0]["role"] == "system" and messages[1]["role"] == "user" and messages[2]["role"] == "assistant"
    return False


class HFGenerator:
    """Batched generation on OpenAI-format messages using a HuggingFace model."""

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
    ):
        self.model = model
        self.tokenizer = tokenizer

    @beartype
    def _generate_stream(
        self,
        conversations: list[OpenAIMessages],
        batch_size: int = 32,
        generation_kwargs: dict[str, Any] | None = None,
    ) -> Generator[OpenAIMessages, None, None]:
        """Generate responses for a list of 0-turn conversations."""
        if generation_kwargs is None:
            generation_kwargs = {"min_length": -1, "max_new_tokens": 512, "do_sample": False}
        assert all(_is_valid_messages(c) for c in conversations)
        assert all(_is_valid_0turn(c) for c in conversations)
        with torch.no_grad():
            texts_inputs: list[str] = [
                self.tokenizer.apply_chat_template(
                    conversation, tokenize=False, add_generation_prompt=True
                )
                for conversation in conversations
            ]

            batch_inputs: list[dict[str, torch.Tensor]] = [
                {
                    k: v.to(self.model.device)
                    for k, v in self.tokenizer(
                        texts_inputs[i : i + batch_size],
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                    ).items()
                }
                for i in range(0, len(texts_inputs), batch_size)
            ]

            convo_idx = 0
            for batch in batch_inputs:
                actual_batch_size = batch["input_ids"].shape[0]
                input_length = batch["input_ids"].shape[1]
                outputs = self.model.generate(**batch, **generation_kwargs)
                outputs_text = self.tokenizer.batch_decode(
                    outputs[:, input_length:], skip_special_tokens=True
                )
                assert len(outputs_text) == actual_batch_size
                for response in outputs_text:
                    convo = conversations[convo_idx] + [
                        {"role": "assistant", "content": response.strip()},
                    ]
                    assert _is_valid_messages(convo)
                    assert _is_valid_1turn(convo)
                    yield convo
                    convo_idx += 1

    def generate(
        self,
        conversations: list[OpenAIMessages],
        **kwargs: Any,
    ) -> list[OpenAIMessages]:
        return list(self._generate_stream(conversations, **kwargs))


if __name__ == "__main__":
    # Test: PYTHONPATH=<experiment_dir> python -m evaluation.inference.client.model_generator
    # Uses a small model to generate responses for 3 conversations.
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    print(f"=== Loading {model_name} ===")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="cpu")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    generator = HFGenerator(model, tokenizer)
    convos: list[OpenAIMessages] = [
        [{"role": "user", "content": "What is 2+2?"}],
        [{"role": "user", "content": "Capital of France?"}],
        [{"role": "user", "content": "Say hello."}],
    ]

    print("=== Generating ===")
    results = generator.generate(convos, batch_size=2, generation_kwargs={"max_new_tokens": 32, "do_sample": False})
    for i, r in enumerate(results):
        print(f"\n--- Convo {i} ---")
        for msg in r:
            print(f"  {msg['role']}: {msg['content'][:100]}")

    print("\n[OK] HFGenerator test passed")
