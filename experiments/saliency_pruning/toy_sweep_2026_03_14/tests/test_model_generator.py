from model_generator import HFGenerator
import warnings
import torch
import json
from model_generator import OpenAIMessages, is_valid_messages, is_valid_1turn_messages
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    print("=" * 100)
    print("Integration test for HFGenerator")

    # Generate a small model for testing
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-Math-1.5B-Instruct", device_map="cpu"
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Math-1.5B-Instruct")
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    tokenizer.padding_side = "left"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        warnings.warn(f"⚠️ CUDA is not available, using {device}")
    model = model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
        p.grad = None
    # Remove all layers except the last one; hopefully this leads to enough speedups
    model.model.layers = model.model.layers[-1:]

    generator = HFGenerator(model, tokenizer)

    # Create a small set of dumb/short messages
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
            {"role": "system", "content": "Always say 'yes'."},
            {"role": "user", "content": "Is the sky blue?"},
        ],
    ]

    import tqdm

    outputs = list(
        tqdm.tqdm(
            generator.generate_stream(messages_list, batch_size=2),
            total=len(messages_list),
        )
    )
    assert len(outputs) == len(messages_list)
    assert all(is_valid_messages(output) for output in outputs)
    assert all(is_valid_1turn_messages(output) for output in outputs)
    print("Messages output...")
    for output in outputs:
        print(json.dumps(output, indent=4))
        print("-" * 100)

    # Test caching: re-run and verify cache hits
    outputs2 = generator.generate(messages_list, batch_size=2)
    assert len(outputs2) == len(messages_list)
    for o1, o2 in zip(outputs, outputs2):
        assert o1[-1]["content"] == o2[-1]["content"], "Cache should return same result"
    print("Cache test passed")

    # TODO(adrianoh) make sure the ordering is exactly correct always and use a mock model
    # with hardcoded input/output
    print("❓ OK")
    print("=" * 100)


if __name__ == "__main__":
    main()
