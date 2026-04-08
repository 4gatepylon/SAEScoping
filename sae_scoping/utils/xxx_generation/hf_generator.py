from __future__ import annotations
import torch
from beartype.typing import Any, Generator
from beartype import beartype
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from sae_scoping.utils.generation.base_generator import BaseGenerator, MessagesWrapper
from sae_scoping.utils.generation.messages import (
    OpenAIMessages,
    is_valid_messages,
    is_valid_1turn_messages,
)


class HFGenerator(BaseGenerator):
    """ """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        cache: dict[str, dict[str, list[str]]] | None = {},
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model = model
        self.tokenizer = tokenizer
        self.cache = cache

    @beartype
    def _generate_stream(  # XXX
        self,
        conversations: list[OpenAIMessages],
        batch_size: int = 32,
        generation_kwargs: dict[str, Any] = {
            # Greedy-sample short responses as default
            "min_length": -1,
            "max_new_tokens": 512,
            "do_sample": False,
        },
        return_indices: bool = False,
    ) -> Generator[OpenAIMessages | tuple[OpenAIMessages, int], None, None]:
        """
        Generate responses for a list of conversations.
        """
        assert all(self._is_valid_convo(conversation) for conversation in conversations)
        assert all(self._is_0turn_convo(conversation) for conversation in conversations)
        with torch.no_grad():
            # Step 1: Apply chat template to convert conversations to texts
            texts_inputs: list[str] = [
                self.tokenizer.apply_chat_template(
                    conversation, tokenize=False, add_generation_prompt=True
                )
                for conversation in conversations
            ]

            # Step 2: Tokenize in batches
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

            assert sum(batch["input_ids"].shape[0] for batch in batch_inputs) == len(
                texts_inputs
            ), (
                f"{sum(batch['input_ids'].shape[0] for batch in batch_inputs)} != {len(texts_inputs)}"
            )

            # Step 3: Generate responses
            current_index = 0
            for batch in batch_inputs:
                batch_size, input_length = batch["input_ids"].shape
                # Make sure we know our indices
                next_index = current_index + batch_size
                batch_original_indices = list(range(current_index, next_index))
                current_index = next_index  # Update for next batch
                # Generate
                outputs = self.model.generate(**batch, **generation_kwargs)
                outputs_text = self.tokenizer.batch_decode(
                    outputs[:, input_length:], skip_special_tokens=True
                )
                assert len(outputs_text) > 0
                assert len(outputs_text) % batch_size == 0
                if len(outputs_text) != batch_size:
                    raise NotImplementedError(
                        "Multiple responses per batch are not supported yet."
                    )
                # Format and return
                outputs_convos = [
                    # System and user or just user, then just append the response
                    c
                    + [
                        {
                            "role": "assistant",
                            "content": response.strip(),
                        },
                    ]
                    for c, response in zip(conversations, outputs_text)
                ]
                assert all(
                    self._is_valid_convo(conversation)
                    for conversation in outputs_convos
                )
                assert all(
                    self._is_1turn_convo(conversation)
                    for conversation in outputs_convos
                )
                assert len(outputs_convos) == len(batch_original_indices), (
                    f"{len(outputs_convos)} != {len(batch_original_indices)}"
                )
                if return_indices:
                    yield from zip(outputs_convos, batch_original_indices)
                else:
                    yield from outputs_convos


if __name__ == "__main__":

    def integration_test_hf_generator():
        print("=" * 100)
        print("Integration test for HFGenerator")
        # 1. Load model and tokenizer
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-Math-1.5B-Instruct", device_map="cpu", cache_dir='/data/huggingface/'
        )
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Math-1.5B-Instruct")
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        tokenizer.padding_side = "left"
        model = model.to("cuda")
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
            p.grad = None

        # 2. Create generator
        generator = HFGenerator(model, tokenizer)

        # 3. Create messages/data
        messages_list: list[OpenAIMessages] = [
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is the capital of France?"},
            ],
            [
                {"role": "system", "content": "You are a friendly AI."},
                {"role": "user", "content": "What is the capital of Germany?"},
            ],
            [
                {"role": "system", "content": "Always say 'yes."},
                {"role": "user", "content": "Is the sky blue?"},
            ],
        ]

        # 4. Generate
        import tqdm

        outputs = list(
            tqdm.tqdm(
                generator.generate_stream(messages_list, batch_size=2),
                total=len(messages_list),
            )
        )
        assert len(outputs) == len(messages_list)
        assert all(isinstance(output, MessagesWrapper) for output in outputs)
        assert all(is_valid_messages(output.messages) for output in outputs)
        assert all(is_valid_1turn_messages(output.messages) for output in outputs)
        print("Messages output...")
        for output in outputs:
            print(output.model_dump_json(indent=4))
            print("-" * 100)
        print("[OK (tentative)] If this looks good, then test PASSED for HFGenerator")
        print("=" * 100)
