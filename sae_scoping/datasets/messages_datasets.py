from __future__ import annotations
from beartype import beartype
import json
from datasets import load_dataset, Dataset, DatasetDict, concatenate_datasets
from beartype.typing import Any, Callable, Iterable
from transformers import PreTrainedTokenizerBase

"""
This is basically a copy of `text_datasets.py` (albeit with less datasets) but it outputs
datasets ONLY in "messages" format. It is meant for chat models specifically.
"""


@beartype
def get_imdb_sentiment_dataset_for_gemma_it(
    n_samples: int,
    seed: int = 1,
    n_shots: int = 0,
) -> Dataset:
    # 0. Collect dataset/samples
    prompt_template = 'Please classify the sentiment of the following text as either "positive" or "negative".\n\nText: {text}\n\nPlease provide your answer next as "positive" or "negative".'
    dataset = concatenate_datasets(
        [
            load_dataset("stanfordnlp/imdb", split="train"),
            load_dataset("stanfordnlp/imdb", split="test"),
        ]
    )
    dataset = dataset.shuffle(seed=seed)
    assert len(dataset) >= n_samples + n_shots, (
        f"Dataset has {len(dataset)} samples but {n_samples + n_shots} were requested"
    )
    samples = dataset.select(range(n_shots))
    # 1. Create few-shot prompt
    few_shot_prompt = []
    for sample in samples:
        # User request
        few_shot_prompt.append(
            {"role": "user", "content": prompt_template.format(text=sample["text"])}
        )
        # Assistant response
        few_shot_prompt.append(
            {
                "role": "assistant",
                "content": "positive" if sample["label"] == 1 else "negative",
            }
        )
    assert len(few_shot_prompt) == n_shots * 2
    dataset = dataset.select(range(n_shots, n_samples + n_shots))
    # rename "text" to "imdb_text"
    dataset = dataset.rename_column("text", "imdb_text")
    assert set(dataset.column_names) == {"imdb_text", "label"}

    def create_messages_fn_local(object):
        prompt = []
        prompt += few_shot_prompt
        question = prompt_template.format(text=object["imdb_text"])
        answer = "positive" if object["label"] == 1 else "negative"
        prompt += [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]
        # NOTE: return messages to only learn this stuff!
        return {"messages": prompt}

    dataset = dataset.map(create_messages_fn_local)
    dataset = dataset.remove_columns(
        [c for c in dataset.column_names if c != "messages"]
    )
    assert set(dataset.column_names) == {"messages"}
    return dataset


def create_messages_fn(example, key1: str = "message_1", key2: str = "message_2"):
    messages = []
    messages.append({"role": "user", "content": example[key1]})
    messages.append({"role": "assistant", "content": example[key2]})
    return {"messages": messages}


@beartype
def get_ultrachat_dataset_for_gemma_it(
    n_samples: int,
    seed: int = 1,
) -> Dataset:
    assert n_samples >= 4, "n_samples must be greater than or equal to 4"
    dataset = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")
    dataset = dataset.shuffle(seed=seed)
    dataset = dataset.select(range(n_samples))
    # drop all columns except "messages"
    dataset = dataset.remove_columns(
        [c for c in dataset.column_names if c != "messages"]
    )
    assert set(dataset.column_names) == {"messages"}
    return dataset


@beartype
def get_apps_dataset_for_gemma_it(
    n_samples: int,
    seed: int = 1,
    difficulties: Iterable[str] = ["introductory", "competition", "interview"],
) -> Dataset:
    template = """User: Here is a coding competition problem that I need some help with. Can you try to solve it? Thanks!

Problem:
{user}

What is the solution?
"""
    # return messages
    dataset_test = load_dataset("4gate/codeparrot_apps", split="test")
    dataset_train = load_dataset("4gate/codeparrot_apps", split="train")
    difficulties_set = set(difficulties)
    dataset_test = dataset_test.filter(lambda x: x["difficulty"] in difficulties_set)
    dataset_train = dataset_train.filter(lambda x: x["difficulty"] in difficulties_set)
    if len(dataset_test) + len(dataset_train) < n_samples:
        raise ValueError(
            f"Dataset has {len(dataset_test) + len(dataset_train)} "
            + f"samples AFTER SELECTING DIFFICULTIES (2/3) but {n_samples} were requested"
        )
    dataset_test = dataset_test.filter(lambda x: len(x["solutions"].strip()) > 0)
    dataset_train = dataset_train.filter(lambda x: len(x["solutions"].strip()) > 0)
    if len(dataset_test) + len(dataset_train) < n_samples:
        raise ValueError(
            f"Dataset has {len(dataset_test) + len(dataset_train)} "
            + f"samples AFTER FILTERING FOR SOLUTIONS (3/3) but {n_samples} were requested"
        )
    dataset = concatenate_datasets([dataset_train, dataset_test])
    dataset = dataset.shuffle(seed=seed)
    dataset = dataset.select(range(n_samples))
    assert len(dataset) == n_samples, "Selection did not work?"

    def extract_solution(example):
        solutions = json.loads(example["solutions"])
        if len(solutions) == 0:
            raise ValueError(f"No solutions found for example: {example}")
        if not all(isinstance(solution, str) for solution in solutions):
            raise ValueError(f"Solutions are not all strings for example: {example}")
        solution = solutions[0]
        example["solution_extracted"] = solution
        return example

    dataset = dataset.map(extract_solution)
    assert "question_formatted" not in dataset.column_names

    def format_question(element):
        question = element["question"]
        element["question_formatted"] = template.format(user=question)
        return element

    dataset = dataset.map(format_question)
    dataset = dataset.map(
        create_messages_fn,
        fn_kwargs={"key1": "question_formatted", "key2": "solution_extracted"},
    )
    assert "messages" in dataset.column_names
    dataset = dataset.remove_columns(
        [c for c in dataset.column_names if c != "messages"]
    )
    assert set(dataset.column_names) == {"messages"}
    return dataset


@beartype
def get_biology_dataset_for_gemma_it(
    n_samples: int,
    seed: int = 1,
) -> Dataset:
    camel = load_dataset("camel-ai/biology", split="train")
    megascience = load_dataset("MegaScience/MegaScience", split="train")
    if len(camel) + len(megascience) < n_samples:
        raise ValueError(
            f"Dataset has {len(camel) + len(megascience)} "
            + f"samples TOTAL but {n_samples} were requested"
        )
    subjects_set = {"biology", "medicine"}
    megascience = megascience.filter(lambda x: x["subject"] in subjects_set)

    # fmt: off
    camel = camel.map(create_messages_fn, fn_kwargs={"key1": "message_1", "key2": "message_2"})
    megascience = megascience.map(create_messages_fn, fn_kwargs={"key1": "question", "key2": "answer"})
    # fmt: on
    dataset = concatenate_datasets([camel, megascience])
    dataset = dataset.shuffle(seed=seed)
    assert len(dataset) >= n_samples
    dataset = dataset.select(range(n_samples))
    return dataset


if __name__ == "__main__":
    import traceback

    n_test_samples = 500  # Small number for quick testing

    # Test 1: IMDB Sentiment Dataset
    print("\n" + "=" * 30 + " IMDB SENTIMENT " + "=" * 30)
    try:
        imdb_dataset = get_imdb_sentiment_dataset_for_gemma_it(
            n_samples=n_test_samples,
            seed=42,
            n_shots=1,
        )
        print(f"IMDB dataset loaded successfully: {len(imdb_dataset)} samples")
        print(f"Columns: {imdb_dataset.column_names}")
        assert set(imdb_dataset.column_names) == {"messages"}, (
            "Expected only 'messages' column"
        )
        print(
            f"Sample messages[0]: {imdb_dataset[0]['messages'][:2]}..."
        )  # First 2 messages
        print("✓ IMDB test passed!")
    except Exception as e:
        print(f"✗ IMDB test failed: {e}")

        traceback.print_exc()

    # Test 2: UltraChat Dataset
    print("\n" + "=" * 30 + " ULTRACHAT " + "=" * 30)
    try:
        ultrachat_dataset = get_ultrachat_dataset_for_gemma_it(
            n_samples=n_test_samples,
            seed=42,
        )
        print(
            f"UltraChat dataset loaded successfully: {len(ultrachat_dataset)} samples"
        )
        print(f"Columns: {ultrachat_dataset.column_names}")
        assert set(ultrachat_dataset.column_names) == {"messages"}, (
            "Expected only 'messages' column"
        )
        print(f"Sample messages[0] (first msg): {ultrachat_dataset[0]['messages'][0]}")
        print("✓ UltraChat test passed!")
    except Exception as e:
        print(f"✗ UltraChat test failed: {e}")

        traceback.print_exc()

    # Test 3: APPS Dataset (NOTE: this function has bugs - dataset_test/train not defined)
    print("\n" + "=" * 30 + " APPS " + "=" * 30)
    try:
        apps_dataset = get_apps_dataset_for_gemma_it(
            n_samples=n_test_samples,
            seed=42,
            difficulties=["introductory"],
        )
        print(f"APPS dataset loaded successfully: {len(apps_dataset)} samples")
        print(f"Columns: {apps_dataset.column_names}")
        assert set(apps_dataset.column_names) == {"messages"}, (
            "Expected only 'messages' column"
        )
        print(f"Sample messages[0]: {apps_dataset[0]['messages']}")
        print("✓ APPS test passed!")
    except Exception as e:
        print(f"✗ APPS test failed: {e}")

        traceback.print_exc()

    # Test 4: Biology Dataset
    print("\n" + "=" * 30 + " BIOLOGY " + "=" * 30)
    try:
        biology_dataset = get_biology_dataset_for_gemma_it(
            n_samples=n_test_samples,
            seed=42,
        )
        print(f"Biology dataset loaded successfully: {len(biology_dataset)} samples")
        print(f"Columns: {biology_dataset.column_names}")
        assert "messages" in biology_dataset.column_names, "Expected 'messages' column"
        print(f"Sample messages[0]: {biology_dataset[0]['messages']}")
        print("✓ Biology test passed!")
    except Exception as e:
        print(f"✗ Biology test failed: {e}")

        traceback.print_exc()

    print("\n" + "=" * 60)
    print("All tests completed!")
