from __future__ import annotations
from typing import Any
from beartype import beartype
import torch
import tqdm
import orjson
import json
from pathlib import Path
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from rlvr.utils.loaders import load_model_with_pad_token
from rlvr.utils.chats import OpenAIConversation_t
from beartype.typing import Generator


class HFGenerator:
    """
    Generation always works like this:
    1. Chat-Template: INPUT conversations OUTPUT texts
    2. Tokenize: INPUT texts OUTPUT tokens
    3. Generate: INPUT tokens OUTPUT tokens
    4. Decode/extract: INPUT tokens OUTPUT chats

    We call the inputs of each of these the:
    1. Chats
    2. Texts
    3. Input_ids
    4. Output_ids
    5. Output_chats (output of 4)

    TODO(Adriano) add support for:
    - Caching
    - Multiple answers, etc...
    - Streaming
    - More formats and reasons for stop (i.e. stop reason from OpenAI API) and information like
        how long the tokens are for each segment/block/turn so that we can mask loss, etc...
        for downstream users that want to use this.
    - Multiprocessing
    """

    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase):
        self.model = model
        self.tokenizer = tokenizer
        # self.input_chat2output_chat_cache: dict[str, str] = {}
        # self.text2tokens_cache: dict[str, list[int]] = {} # XXX support the caches

    @beartype
    def _generate_stream(
        self,
        conversations: list[OpenAIConversation_t],
        batch_size: int = 32,
        generation_kwargs: dict[str, Any] = {
            # Greedy-sample short responses as default
            "min_length": -1,
            "max_new_tokens": 512,
            "do_sample": False,
        },
        return_indices: bool = False,
    ) -> Generator[OpenAIConversation_t | tuple[OpenAIConversation_t, int], None, None]:
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

    @beartype
    def generate_stream(
        self,
        conversations: list[OpenAIConversation_t],
        batch_size: int = 32,
        generation_kwargs: dict[str, Any] = {
            # Greedy-sample short responses as default
            "min_length": -1,
            "max_new_tokens": 512,
            "do_sample": False,
        },
        n_workers: int = 1,
        return_indices: bool = False,
    ) -> Generator[OpenAIConversation_t | tuple[OpenAIConversation_t, int], None, None]:
        if n_workers == 1:
            yield from self._generate_stream(
                conversations,
                batch_size,
                generation_kwargs,
                return_indices=return_indices,
            )
        else:
            # NOTE: it is in-order PER worker though; so tehre are 4 in-order subsequences; mos of the time
            # you will be out of order by at most a couple indices and so a caller COULD have a buffer that
            # pops as soon as the next one is out, re-ordering them
            if not return_indices:
                raise ValueError(
                    "return_indices must be True (because of out of order execution) if n_workers > 1"
                )
            # somehow we will need to
            # 1. delete model/tokenizer
            # 2. each process creates its own model/tokenizer
            # 3. a shim function pipes the yieldees into the queue
            # 4. this function reads from queue and yields passthrough
            raise NotImplementedError("Multi-worker generation is not implemented yet")


if __name__ == "__main__":
    # NOTE: this stuff should NOT be imported in main namespace
    # since it is MORE SPECIFIC (it is gsm8k-specific)
    from rlvr.tasks.gsm8k.dataset_conversations import load_gsm8k_conversations
    from rlvr.tasks.gsm8k.evaluation import GSM8KEvaluator

    model, tokenizer = load_model_with_pad_token()
    generator = HFGenerator(model, tokenizer)

    def shots2filepath(n_shots: int) -> Path:
        return (
            Path(__file__).parent.parent
            / "artifacts"
            / "initial_eval"
            / f"gsm8k_conversations_golden_{n_shots}.json"
        )

    # Get inputs and golden (gsm8k) outputs
    print("=" * 100)
    print("Saving few shot variants...")
    for n_shots in [0, 1, 2, 4, 8, 16, 32]:
        conversations_golden = load_gsm8k_conversations(
            n_questions=32, shuffle_seed=2, system_prompt=None, n_shots=n_shots
        )
        file = shots2filepath(n_shots)
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_bytes(orjson.dumps(conversations_golden))
        # conversations_inputs = generator._1turn2_0turn(conversations_golden)

    generation_kwargs = {
        "min_length": -1,
        "max_new_tokens": 10,  # XXX
        "do_sample": False,
    }

    print("=" * 100)
    print("Evaluating few shot variants...")
    for n_shots in [0, 1, 2, 4, 8, 16, 32]:
        print("=" * 100)
        print(f"Evaluating n_shots={n_shots}...")
        file_input = shots2filepath(n_shots)
        file_output = file_input.with_suffix(".output.json")
        assert file_input.exists() and not file_output.exists()
        conversations_golden = orjson.loads(file_input.read_bytes())
        conversations_inputs = generator._1turn2_0turn(conversations_golden)
        # Evaluate model
        conversations_generated = list(
            tqdm.tqdm(
                generator.generate_stream(
                    conversations_inputs,
                    batch_size=32,
                    generation_kwargs=generation_kwargs,
                ),
                total=len(conversations_inputs),
            )
        )
        file_output.write_bytes(
            orjson.dumps(
                [
                    cge + [cgo[-1]]
                    for cge, cgo in zip(conversations_generated, conversations_golden)
                ]
            )
        )
        assert all(c[0]["content"] == c[1]["content"] for c in conversations_generated)
        print(conversations_generated)

        evaluator = GSM8KEvaluator.from_regex()  # defaults is what we need
        results = evaluator.evaluate(
            conversations=conversations_golden,
            generator=generator,
            batch_size=128,
            generation_kwargs=generation_kwargs,
        )
        print(json.dumps(results, indent=4))
        print("=" * 100)
