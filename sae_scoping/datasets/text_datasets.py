from __future__ import annotations
from beartype import beartype
import json
from datasets import load_dataset, Dataset, DatasetDict, concatenate_datasets
from beartype.typing import Any, Callable, Iterable
from transformers import PreTrainedTokenizerBase
import jinja2

"""
Define datasets for analysis of overlap. Analysis is done in `datasets_overlap.py`.

TODO(Adriano) we should load datasets cached from files because that way we can ensure
    ordering (or we should store "checksums" of sorts of the ordering...)
TODO(Adriano) we will want to split up this code into multiple files.
TODO(Adriano) in the future we may do more different types of features.
TODO(Adriano) in the future we should add more datasets.
TODO(Adriano) add better support for non-text datsets and/or tokenizer agnositicism...
TODO(Adriano) have auto-detection for the columns
TODO(Adriano) be able to do the renaming etc... in the dataset loading function...
TODO(Adriano) avoid having to write so much argument boilerplate...
TODO(Adriano) the single method should support filtering (i.e. it should support a
    map/filter/reduce structure)
"""

# This default is picked in part because it is fast
DEFAULT_QA_TEMPLATTTING_FUNCTION = "Question: {user}\nAnswer: {assistant}"
OpenAIChat = list[dict[str, str]]
qa_templatting_function_type = (
    Callable[[str | Any, str | Any], str]
    | Callable[[OpenAIChat], str]
    | PreTrainedTokenizerBase
    | jinja2.Template
    | str
)


@beartype
def get_qa_dataset_dict(
    dataset_name: str | Dataset,  # can be dataset if you like instead...
    n_samples_ranking: int,
    n_samples_training: int,
    n_samples_evaluation: int,
    shuffle_seed: int = 1,
    args: list[Any] = [],
    splits: list[str] = ["train"],  # NOTE that ALL splits are combined and shuffled.
    question_column_name: str | None = "question",
    answer_column_name: str | None = "answer",
    messages_column_name: str
    | None = None,  # for messages datasets of longer than 1 turn
    # This templatting "function"works like
    # - functoin => call it
    # - PreTrainedTokenizerBase => use it's apply_chat_template method
    # - jinja2.Template => use it's render method
    # - str => use it as a format string with .format(question=..., answer=...)
    qa_templatting_function: qa_templatting_function_type = DEFAULT_QA_TEMPLATTTING_FUNCTION,
    # These two are fed into render() or apply_chat_template()
    format_question_key: str | None = "user",
    format_answer_key: str | None = "assistant",
    # This is where we will store the output after the dataset map
    text_column_name: str = "text",
    verbose: bool = True,
) -> DatasetDict:
    # 0.1 Deal with warnings for interactions or lack thereof between different options
    if messages_column_name is not None and any(
        [question_column_name is not None, answer_column_name is not None]
    ):
        print(
            "WARNING: question_column_name and answer_column_name will "
            + "be ignored if messages_column_name is not None"
        )
    if isinstance(qa_templatting_function, PreTrainedTokenizerBase) and any(
        [format_question_key is not None, format_answer_key is not None]
    ):
        print(
            "WARNING: format_question_key and format_answer_key will "
            + "be ignored if qa_templatting_function is a PreTrainedTokenizerBase"
        )

    # 0.2 Throw errors if the provided arguments cannot be used
    if messages_column_name is not None and not isinstance(
        qa_templatting_function, PreTrainedTokenizerBase
    ):
        raise ValueError(
            "qa_templatting_function must be a PreTrainedTokenizerBase "
            + "if messages_column_name is not None"
        )
    if (
        isinstance(qa_templatting_function, PreTrainedTokenizerBase)
        and qa_templatting_function.chat_template is None
    ):
        raise ValueError(
            "qa_templatting_function must have a chat template "
            + "if messages_column_name is not None"
        )
    if format_question_key is not None and format_question_key == format_answer_key:
        raise ValueError("format_question_key and format_answer_key cannot be the same")
    n_samples_total = n_samples_ranking + n_samples_training + n_samples_evaluation

    # Step 1: Load and combine all splits
    if verbose:
        print("Loading and combining all splits...")
    if isinstance(dataset_name, Dataset):
        all_datasets = [dataset_name]
        if len(args) > 0 or len(splits) > 0:
            raise ValueError(
                "args and splits will be ignored if dataset_name is a Dataset"
            )
    else:
        all_datasets: list[Dataset] = []
        for split in splits:
            ds = load_dataset(dataset_name, *args, split=split)
            all_datasets.append(ds)

    if verbose:
        print("Combining all datasets...")
    if len(all_datasets) == 1:
        combined_dataset = all_datasets[0]
    else:
        combined_dataset = concatenate_datasets(all_datasets)
    if len(combined_dataset) < n_samples_total:
        raise ValueError(
            f"Dataset has {len(combined_dataset)} samples "
            + f"but {n_samples_total} were requested"
        )
    if text_column_name in combined_dataset.column_names:
        raise ValueError(f"text_column_name {text_column_name} already in dataset")

    # Step 2: Shuffle the combined dataset
    if verbose:
        print("Shuffling...")
    combined_dataset = combined_dataset.shuffle(seed=shuffle_seed)
    # Sample NOW to make the stuff below efficient
    if verbose:
        print("Sampling...")
    combined_dataset = combined_dataset.select(range(n_samples_total))

    if verbose:
        print("Defining and selecting templating function...")

    def apply_template_string(question: str, answer: str) -> str:
        return qa_templatting_function.format(
            **{format_question_key: question, format_answer_key: answer}
        )

    def apply_template_jinja2(question: str, answer: str) -> str:
        return qa_templatting_function.render(
            **{format_question_key: question, format_answer_key: answer}
        )

    def apply_template_tokenizer_not_messages(question: str, answer: str) -> str:
        return qa_templatting_function.apply_chat_template(
            [
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
            ],
            tokenize=False,
            add_generation_prompt=False,
        )

    def apply_template_tokenizer_messages(messages: list[dict[str, str]]) -> str:
        return qa_templatting_function.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )

    apply_template = None
    if isinstance(qa_templatting_function, str):
        apply_template = apply_template_string
    elif isinstance(qa_templatting_function, jinja2.Template):
        apply_template = apply_template_jinja2
    elif (
        isinstance(qa_templatting_function, PreTrainedTokenizerBase)
        and messages_column_name is not None
    ):
        apply_template = apply_template_tokenizer_messages  # Assumes OpenAIChat already
    elif (
        isinstance(qa_templatting_function, PreTrainedTokenizerBase)
        and messages_column_name is None
    ):
        apply_template = apply_template_tokenizer_not_messages  # Builds OpenAIChat
    elif str(type(qa_templatting_function)) == "<class 'function'>":
        apply_template = qa_templatting_function
    else:
        raise ValueError(
            f"Unsupported qa_templatting_function type: {type(qa_templatting_function)}"
        )
    assert apply_template is not None

    # Step 4: Map the templating function to create the text column
    def map_fn(example: dict[str, Any]) -> dict[str, Any]:
        if messages_column_name is not None:
            messages = example[messages_column_name]
            example[text_column_name] = apply_template(messages)
        else:
            question = example[question_column_name]
            answer = example[answer_column_name]
            example[text_column_name] = apply_template(question, answer)
        return example

    if verbose:
        print("Mapping the templating function to create the text column...")
    combined_dataset = combined_dataset.map(map_fn)

    # Step 5: Split into subsets
    if verbose:
        print("Creating the dataset dict...")
    dataset_dict = DatasetDict(
        {
            "ranking": combined_dataset.select(range(n_samples_ranking)),
            "training": combined_dataset.select(
                range(n_samples_ranking, n_samples_ranking + n_samples_training)
            ),
            "evaluation": combined_dataset.select(
                range(n_samples_ranking + n_samples_training, n_samples_total)
            ),
        }
    )

    return dataset_dict


@beartype
def get_gsm8k_dataset(
    n_samples_ranking: int,
    n_samples_training: int,
    n_samples_evaluation: int,
    seed: int = 1,
    qa_templatting_function: qa_templatting_function_type = DEFAULT_QA_TEMPLATTTING_FUNCTION,
    verbose: bool = True,
) -> DatasetDict:
    dataset_dict = get_qa_dataset_dict(
        dataset_name="openai/gsm8k",
        n_samples_ranking=n_samples_ranking,
        n_samples_training=n_samples_training,
        n_samples_evaluation=n_samples_evaluation,
        shuffle_seed=seed,
        args=["main"],
        splits=["train"],
        question_column_name="question",
        answer_column_name="answer",
        qa_templatting_function=qa_templatting_function,
        format_question_key="user",  # default
        format_answer_key="assistant",  # default
        text_column_name="text",  # default
        verbose=verbose,
    )
    return dataset_dict


@beartype
def get_hf_math_dataset(
    n_samples_ranking: int,
    n_samples_training: int,
    n_samples_evaluation: int,
    seed: int = 1,
    verbose: bool = True,
    qa_templatting_function: qa_templatting_function_type = DEFAULT_QA_TEMPLATTTING_FUNCTION,
) -> DatasetDict:
    return get_qa_dataset_dict(
        dataset_name="HuggingFaceH4/MATH-500",
        n_samples_ranking=n_samples_ranking,
        n_samples_training=n_samples_training,
        n_samples_evaluation=n_samples_evaluation,
        shuffle_seed=seed,
        args=[],
        splits=["test"],
        question_column_name="problem",
        answer_column_name="solution",
        qa_templatting_function=qa_templatting_function,
        format_question_key="user",
        format_answer_key="assistant",
        text_column_name="text",
        verbose=verbose,
    )


@beartype
def _get_camel_ai_dataset(
    dataset_name: str,
    n_samples_ranking: int,
    n_samples_training: int,
    n_samples_evaluation: int,
    seed: int = 1,
    qa_templatting_function: qa_templatting_function_type = DEFAULT_QA_TEMPLATTTING_FUNCTION,
    verbose: bool = True,
) -> DatasetDict:
    return get_qa_dataset_dict(
        dataset_name=f"camel-ai/{dataset_name}",
        n_samples_ranking=n_samples_ranking,
        n_samples_training=n_samples_training,
        n_samples_evaluation=n_samples_evaluation,
        shuffle_seed=seed,
        args=[],
        splits=["train"],
        qa_templatting_function=qa_templatting_function,
        question_column_name="message_1",
        answer_column_name="message_2",
        format_question_key="user",  # default
        format_answer_key="assistant",  # default
        text_column_name="text",  # default
        verbose=verbose,
    )


@beartype
def get_camel_ai_biology_dataset(*args: Any, **kwargs: Any) -> DatasetDict:
    return _get_camel_ai_dataset("biology", *args, **kwargs)


@beartype
def get_camel_ai_chemistry_dataset(*args: Any, **kwargs: Any) -> DatasetDict:
    return _get_camel_ai_dataset("chemistry", *args, **kwargs)


@beartype
def get_camel_ai_physics_dataset(*args: Any, **kwargs: Any) -> DatasetDict:
    return _get_camel_ai_dataset("physics", *args, **kwargs)


@beartype
def DEFAULT_IMDB_SENTIMENT_TEMPLATTING_FUNCTION(
    question: str, answer: str | int
) -> str:
    if isinstance(answer, int):
        assert answer in [0, 1], "Answer must be 0 or 1"
        answer = "positive" if answer == 1 else "negative"
    return f'What is the sentiment of the following text? Please respond either "positive" or "negative".\nText: "{question.replace('"', "")}"\nAnswer: {answer}'


@beartype
def get_imdb_sentiment_dataset(
    n_samples_ranking: int,
    n_samples_training: int,
    n_samples_evaluation: int,
    seed: int = 1,
    verbose: bool = True,
    # Pass a different function to just operate on the text lol
    qa_templatting_function: qa_templatting_function_type = DEFAULT_IMDB_SENTIMENT_TEMPLATTING_FUNCTION,
) -> DatasetDict:
    dataset_dict = get_qa_dataset_dict(
        dataset_name="stanfordnlp/imdb",
        n_samples_ranking=n_samples_ranking,
        n_samples_training=n_samples_training,
        n_samples_evaluation=n_samples_evaluation,
        shuffle_seed=seed,
        args=[],
        splits=["train"],
        qa_templatting_function=qa_templatting_function,
        question_column_name="text",
        answer_column_name="label",
        text_column_name="real_text",
        verbose=verbose,
    )

    # Swap out the pesky text column
    for dataset_name, dataset in dataset_dict.items():
        # drop column and insert
        dataset = dataset.remove_columns("text")
        dataset = dataset.add_column("text", dataset["real_text"])
        dataset = dataset.remove_columns("real_text")
        dataset_dict[dataset_name] = dataset
    return dataset_dict


@beartype
def load_ultrachat_dataset(
    n_samples_ranking: int,
    n_samples_training: int,
    n_samples_evaluation: int,
    seed: int = 1,
    verbose: bool = True,
    tokenizer: PreTrainedTokenizerBase = None,
) -> DatasetDict:
    if tokenizer is None:
        raise ValueError("Tokenizer is required")
    if tokenizer.chat_template is None:
        raise ValueError("Tokenizer does not have a chat template")
    return get_qa_dataset_dict(
        dataset_name="HuggingFaceH4/ultrachat_200k",
        n_samples_ranking=n_samples_ranking,
        n_samples_training=n_samples_training,
        n_samples_evaluation=n_samples_evaluation,
        shuffle_seed=seed,
        args=[],
        splits=["train_sft"],
        question_column_name=None,  # messages
        answer_column_name=None,  # messages
        messages_column_name="messages",  # custom feature for ultrachat
        qa_templatting_function=tokenizer,
        format_question_key=None,  # messages
        format_answer_key=None,  # messages
        text_column_name="text",
        verbose=verbose,
    )


@beartype
def load_jailbreak_bench_dataset(
    n_samples_ranking: int,
    n_samples_training: int,
    n_samples_evaluation: int,
    seed: int = 1,
    verbose: bool = True,
) -> DatasetDict:
    return get_qa_dataset_dict(
        dataset_name="JailbreakBench/JBB-Behaviors",
        n_samples_ranking=n_samples_ranking,
        n_samples_training=n_samples_training,
        n_samples_evaluation=n_samples_evaluation,
        shuffle_seed=seed,
        args=["behaviors"],
        splits=["harmful"],
        qa_templatting_function=DEFAULT_QA_TEMPLATTTING_FUNCTION,
        question_column_name="Goal",
        answer_column_name="Target",
        messages_column_name=None,
        format_question_key="user",  # default
        format_answer_key="assistant",  # default
        text_column_name="text",
        verbose=verbose,
    )


@beartype
def _get_megascience_subset_dataset(
    subjects: Iterable[str],
    n_samples_ranking: int,
    n_samples_training: int,
    n_samples_evaluation: int,
    seed: int = 1,
    verbose: bool = True,
    qa_templatting_function: qa_templatting_function_type = DEFAULT_QA_TEMPLATTTING_FUNCTION,
) -> DatasetDict:
    n_total = n_samples_ranking + n_samples_training + n_samples_evaluation
    dataset = load_dataset("MegaScience/MegaScience", split="train")
    if len(dataset) < n_total:
        raise ValueError(
            f"Dataset has {len(dataset)} samples TOTAL but {n_total} were requested"
        )
    subjects_set = set(subjects)
    dataset = dataset.filter(lambda x: x["subject"] in subjects_set)
    if len(dataset) < n_total:
        raise ValueError(
            f"Dataset has {len(dataset)} samples FOR YOUR SUBJECTS but {n_total} were requested"
        )
    return get_qa_dataset_dict(
        dataset_name=dataset,
        n_samples_ranking=n_samples_ranking,
        n_samples_training=n_samples_training,
        n_samples_evaluation=n_samples_evaluation,
        shuffle_seed=seed,
        args=[],
        splits=[],
        qa_templatting_function=qa_templatting_function,
        question_column_name="question",
        answer_column_name="answer",
        messages_column_name=None,
        format_question_key="user",
        format_answer_key="assistant",
        text_column_name="text",
        verbose=verbose,
    )


@beartype
def get_megascience_biology_dataset(*args, **kwargs: Any) -> DatasetDict:
    return _get_megascience_subset_dataset(["biology"], *args, **kwargs)


@beartype
def get_megascience_cs(*args: Any, **kwargs: Any) -> DatasetDict:
    return _get_megascience_subset_dataset(["cs"], *args, **kwargs)


@beartype
def get_megascience_math_dataset(*args: Any, **kwargs: Any) -> DatasetDict:
    return _get_megascience_subset_dataset(["math"], *args, **kwargs)


@beartype
def get_megascience_chemistry_dataset(*args: Any, **kwargs: Any) -> DatasetDict:
    return _get_megascience_subset_dataset(["chemistry"], *args, **kwargs)


@beartype
def get_megascience_physics_dataset(*args: Any, **kwargs: Any) -> DatasetDict:
    return _get_megascience_subset_dataset(["physics"], *args, **kwargs)


# TODO(Adriano) do we want to do this? let's just ignore for now because remote code;
# it is not available for datasets > 4.0.0 which is by now kinda old; also I think the
# codeparrot clean dataset has everything anyways
# def _load_github_code(
#     languages: Iterable[str],
#     split: str,
#     n_samples_ranking: int,
#     n_samples_training: int,
#     n_samples_evaluation: int,
#     seed: int = 1,
#     verbose: bool = True,
# ) -> Dataset:
#     raise NotImplementedError("Not implemented")


@beartype
def load_codeparrot(
    n_samples_ranking: int,
    n_samples_training: int,
    n_samples_evaluation: int,
    seed: int = 1,
    verbose: bool = True,
) -> DatasetDict:
    """A "pretraining" code dataset."""
    # TODO(Adriano) add support for choosing which languages to load specifically
    # this could be slow I think alas
    dataset = load_dataset("codeparrot/codeparrot-clean", split="train")
    dataset = dataset.shuffle(seed=seed)
    n_total = n_samples_ranking + n_samples_training + n_samples_evaluation
    if len(dataset) < n_total:
        raise ValueError(
            f"Dataset has {len(dataset)} samples but {n_total} were requested"
        )
    dataset = dataset.select(range(n_total))
    dataset = dataset.map(lambda x: {"text": x["content"]})  # just take in the code
    dataset_dict = DatasetDict(
        {
            "ranking": dataset.select(range(n_samples_ranking)),
            "training": dataset.select(
                range(n_samples_ranking, n_samples_ranking + n_samples_training)
            ),
            "evaluation": dataset.select(
                range(n_samples_ranking + n_samples_training, n_total)
            ),
        }
    )
    return dataset_dict


APPS_QA_TEMPLATTTING_FUNCTION = """User: Here is a coding competition problem that I need some help with. Can you try to solve it? Thanks!

Problem:
{user}

What is the solution?

Assistant: Sure! Let me give you a solution snippet in python below:
```python
{assistant}
```
"""


# TODO(Adriano) synergize this more with `utils.code_data` please
@beartype
def load_apps(
    n_samples_ranking: int,
    n_samples_training: int,
    n_samples_evaluation: int,
    seed: int = 1,
    verbose: bool = True,
    difficulties: Iterable[str] = ["introductory"],
    qa_templatting_function: Callable[
        [str, list[str]], str
    ] = APPS_QA_TEMPLATTTING_FUNCTION,
) -> DatasetDict:
    """A "RLVR" (problems with known solutions) code dataset."""
    n_total = n_samples_ranking + n_samples_training + n_samples_evaluation
    dataset_test = load_dataset("4gate/codeparrot_apps", split="test")
    dataset_train = load_dataset("4gate/codeparrot_apps", split="train")
    if len(dataset_test) + len(dataset_train) < n_total:
        raise ValueError(
            f"Dataset has {len(dataset_test) + len(dataset_train)} "
            + f"samples IMMDIATELY ONCE LOAED (1/3) but {n_total} were requested"
        )
    # Must have solutions and be from the desired difficulties
    difficulties_set = set(difficulties)
    dataset_test = dataset_test.filter(lambda x: x["difficulty"] in difficulties_set)
    dataset_train = dataset_train.filter(lambda x: x["difficulty"] in difficulties_set)
    if len(dataset_test) + len(dataset_train) < n_total:
        raise ValueError(
            f"Dataset has {len(dataset_test) + len(dataset_train)} "
            + f"samples AFTER SELECTING DIFFICULTIES (2/3) but {n_total} were requested"
        )
    dataset_test = dataset_test.filter(lambda x: len(x["solutions"].strip()) > 0)
    dataset_train = dataset_train.filter(lambda x: len(x["solutions"].strip()) > 0)
    if len(dataset_test) + len(dataset_train) < n_total:
        raise ValueError(
            f"Dataset has {len(dataset_test) + len(dataset_train)} "
            + f"samples AFTER FILTERING FOR SOLUTIONS (3/3) but {n_total} were requested"
        )
    dataset = concatenate_datasets([dataset_train, dataset_test])
    dataset = dataset.shuffle(seed=seed)
    dataset = dataset.select(range(n_total))
    assert len(dataset) == n_total, "Selection did not work?"

    def extract_solution(example: dict[str, Any]) -> dict[str, Any]:
        solutions = json.loads(example["solutions"])
        if len(solutions) == 0:
            raise ValueError(f"No solutions found for example: {example}")
        if not all(isinstance(solution, str) for solution in solutions):
            raise ValueError(f"Solutions are not all strings for example: {example}")
        solution = solutions[0]
        example["solution_extracted"] = solution
        return example

    dataset = dataset.map(extract_solution)
    assert len(dataset) == n_total, f"Not able to pass into QA?"
    return get_qa_dataset_dict(
        dataset_name=dataset,
        n_samples_ranking=n_samples_ranking,
        n_samples_training=n_samples_training,
        n_samples_evaluation=n_samples_evaluation,
        shuffle_seed=seed,
        args=[],
        splits=[],
        qa_templatting_function=qa_templatting_function,
        question_column_name="question",
        answer_column_name="solution_extracted",
        messages_column_name=None,
        format_question_key="user",  # default; depends on your template/fn
        format_answer_key="assistant",  # default; depends on your template/fn
        text_column_name="text",
        verbose=verbose,
    )


# TODO(Adriano) synergize this more with `utils.code_data` please
# TODO(Adriano) support some kind of filtering for difficulties etc...
@beartype
def load_deepmind_code_contests(
    n_samples_ranking: int,
    n_samples_training: int,
    n_samples_evaluation: int,
    seed: int = 1,
    verbose: bool = True,
    qa_templatting_function: qa_templatting_function_type = APPS_QA_TEMPLATTTING_FUNCTION,
) -> DatasetDict:
    """A "RLVR" (problems with known solutions) code dataset."""
    n_total = n_samples_ranking + n_samples_training + n_samples_evaluation
    datasets = [
        load_dataset("deepmind/code_contests", split=split)
        for split in ["train", "test", "valid"]
    ]
    datasets = concatenate_datasets(datasets)
    datasets = datasets.shuffle(seed=seed)
    if len(datasets) < n_total:
        raise ValueError(
            f"Dataset has {len(datasets)} samples but {n_total} were requested"
        )

    def extract_solution(example: dict[str, Any]) -> dict[str, Any]:
        solution = ""
        solutions = example.get("solutions", None)
        if isinstance(solutions, dict):
            solutions = solutions.get("solution", [])
            if len(solutions) > 0:
                solution = solutions[0]
        example["solution_extracted"] = solution
        return example

    dataset = datasets.map(extract_solution)
    dataset = dataset.filter(lambda x: len(x["solution_extracted"].strip()) > 0)
    if len(dataset) < n_total:
        raise ValueError(
            f"Dataset has {len(dataset)} samples but {n_total} were requested"
        )
    dataset = dataset.select(range(n_total))
    return get_qa_dataset_dict(
        dataset_name=dataset,
        n_samples_ranking=n_samples_ranking,
        n_samples_training=n_samples_training,
        n_samples_evaluation=n_samples_evaluation,
        shuffle_seed=seed,
        args=[],
        splits=[],
        qa_templatting_function=qa_templatting_function,
        question_column_name="description",
        answer_column_name="solution_extracted",
        messages_column_name=None,
        format_question_key="user",  # default; depends on your template/fn
        format_answer_key="assistant",  # default; depends on your template/fn
        text_column_name="text",
        verbose=verbose,
    )


@beartype
def load_chinese_ultrachat(
    n_samples_ranking: int,
    n_samples_training: int,
    n_samples_evaluation: int,
    seed: int = 1,
    verbose: bool = True,
    tokenizer: PreTrainedTokenizerBase = None,
) -> DatasetDict:
    if tokenizer is None:
        raise ValueError("tokenizer is required")
    # NOTE: this is not actuall ultrachat, but instead a chinese version of smoltalk
    # which is a similar finetuning dataset in Mandarin.
    dataset = load_dataset(
        "opencsg/smoltalk-chinese",
        split="train",
        data_files={"train": ["data/*.parquet"]},  # Exclude 'old/' folder
    )
    return get_qa_dataset_dict(
        dataset_name=dataset,
        n_samples_ranking=n_samples_ranking,
        n_samples_training=n_samples_training,
        n_samples_evaluation=n_samples_evaluation,
        shuffle_seed=seed,
        args=[],
        splits=[],
        qa_templatting_function=tokenizer,
        question_column_name=None,
        answer_column_name=None,
        messages_column_name="conversations",
        format_question_key=None,
        format_answer_key=None,
        text_column_name="text",
        verbose=verbose,
    )


@beartype
def load_spanish_text(
    n_samples_ranking: int,
    n_samples_training: int,
    n_samples_evaluation: int,
    seed: int = 1,
    verbose: bool = True,
) -> DatasetDict:
    dataset = load_dataset("jhonparra18/spanish_billion_words_clean", split="train")
    n_total = n_samples_ranking + n_samples_training + n_samples_evaluation
    if len(dataset) < n_total:
        raise ValueError(
            f"Dataset has {len(dataset)} samples but {n_total} were requested"
        )
    dataset = dataset.shuffle(seed=seed)
    dataset = dataset.select(range(n_total))
    assert set(dataset.column_names) == {"text"}
    dataset_dict = DatasetDict(
        {
            "ranking": dataset.select(range(n_samples_ranking)),
            "training": dataset.select(
                range(n_samples_ranking, n_samples_ranking + n_samples_training)
            ),
            "evaluation": dataset.select(
                range(
                    n_samples_ranking + n_samples_training,
                    n_total,
                )
            ),
        }
    )
    return dataset_dict


@beartype
def load_poetry_dataset(
    n_samples_ranking: int,
    n_samples_training: int,
    n_samples_evaluation: int,
    seed: int = 1,
    verbose: bool = True,
) -> DatasetDict:
    dataset = load_dataset("matthh/gutenberg-poetry-corpus", split="train")
    n_total = n_samples_ranking + n_samples_training + n_samples_evaluation
    if len(dataset) < n_total:
        raise ValueError(
            f"Dataset has {len(dataset)} samples but {n_total} were requested"
        )
    dataset = dataset.shuffle(seed=seed)
    dataset = dataset.select(range(n_total))
    # rename "content" to "text"
    dataset = dataset.rename_column("content", "text")
    dataset_dict = DatasetDict(
        {
            "ranking": dataset.select(range(n_samples_ranking)),
            "training": dataset.select(
                range(n_samples_ranking, n_samples_ranking + n_samples_training)
            ),
            "evaluation": dataset.select(
                range(n_samples_ranking + n_samples_training, n_total)
            ),
        }
    )
    return dataset_dict


@beartype
def load_spanish_poetry_dataset(
    n_samples_ranking: int,
    n_samples_training: int,
    n_samples_evaluation: int,
    seed: int = 1,
    verbose: bool = True,
) -> DatasetDict:
    dataset = load_dataset("biglam/spanish_golden_age_sonnets", split="train")
    n_total = n_samples_ranking + n_samples_training + n_samples_evaluation
    if len(dataset) < n_total:
        raise ValueError(
            f"Dataset has {len(dataset)} samples but {n_total} were requested"
        )
    dataset = dataset.shuffle(seed=seed)
    dataset = dataset.select(range(n_total))
    # rename "sonnet_text" to "text"
    dataset = dataset.rename_column("sonnet_text", "text")
    dataset_dict = DatasetDict(
        {
            "ranking": dataset.select(range(n_samples_ranking)),
            "training": dataset.select(
                range(n_samples_ranking, n_samples_ranking + n_samples_training)
            ),
            "evaluation": dataset.select(
                range(
                    n_samples_ranking + n_samples_training,
                    n_total,
                )
            ),
        }
    )
    return dataset_dict


if __name__ == "__main__":
    n_samples_ranking = 30
    n_samples_training = 30
    n_samples_evaluation = 30
    seed = 1
    verbose = True

    print("Quickly testing that dataset loading works...")

    # GSM8K
    try:
        print("=" * 30 + " GSM8K " + "=" * 30)
        dataset_dict = get_gsm8k_dataset(
            n_samples_ranking=n_samples_ranking,
            n_samples_training=n_samples_training,
            n_samples_evaluation=n_samples_evaluation,
            seed=seed,
            verbose=verbose,
        )
        print(dataset_dict)
    except Exception as e:
        # print(f"Error loading GSM8K dataset: {e}")
        raise e

    # CAMEL-AI Biology
    try:
        print("=" * 30 + " CAMEL-AI BIOLOGY " + "=" * 30)
        dataset_dict = get_camel_ai_biology_dataset(
            n_samples_ranking=n_samples_ranking,
            n_samples_training=n_samples_training,
            n_samples_evaluation=n_samples_evaluation,
            seed=seed,
            verbose=verbose,
        )
        print(dataset_dict)
    except Exception as e:
        # print(f"Error loading CAMEL-AI Biology dataset: {e}")
        raise e

    # CAMEL-AI Chemistry; not downloaded for some god-forsaken reason
    # try:
    #     print("=" * 30 + " CAMEL-AI CHEMISTRY " + "=" * 30)
    #     dataset_dict = get_camel_ai_chemistry_dataset(
    #         n_samples_ranking=n_samples_ranking,
    #         n_samples_training=n_samples_training,
    #         n_samples_evaluation=n_samples_evaluation,
    #         seed=seed,
    #         verbose=verbose,
    #     )
    #     print(dataset_dict)
    # except Exception as e:
    #     # print(f"Error loading CAMEL-AI Chemistry dataset: {e}")
    #     raise e

    # CAMEL-AI Physics ; not downloaded for some god-forsaken reason
    # try:
    #     print("=" * 30 + " CAMEL-AI PHYSICS " + "=" * 30)
    #     dataset_dict = get_camel_ai_physics_dataset(
    #         n_samples_ranking=n_samples_ranking,
    #         n_samples_training=n_samples_training,
    #         n_samples_evaluation=n_samples_evaluation,
    #         seed=seed,
    #         verbose=verbose,
    #     )
    #     print(dataset_dict)
    # except Exception as e:
    #     # print(f"Error loading CAMEL-AI Physics dataset: {e}")
    #     raise e

    # IMDB Sentiment
    try:
        print("=" * 30 + " IMDB SENTIMENT " + "=" * 30)
        dataset_dict = get_imdb_sentiment_dataset(
            n_samples_ranking=n_samples_ranking,
            n_samples_training=n_samples_training,
            n_samples_evaluation=n_samples_evaluation,
            seed=seed,
            verbose=verbose,
        )
        print(dataset_dict)
    except Exception as e:
        # print(f"Error loading IMDB Sentiment dataset: {e}")
        raise e

    # Ultrachat (requires tokenizer - skipping in basic test)
    try:
        print("=" * 30 + " ULTRACHAT " + "=" * 30)
        # print("Skipping Ultrachat - requires tokenizer argument")
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-it")
        dataset_dict = load_ultrachat_dataset(
            n_samples_ranking=n_samples_ranking,
            n_samples_training=n_samples_training,
            n_samples_evaluation=n_samples_evaluation,
            seed=seed,
            verbose=verbose,
            tokenizer=tokenizer,
        )
        print(dataset_dict)
    except Exception as e:
        # print(f"Error loading Ultrachat dataset: {e}")
        raise e

    # JailbreakBench
    try:
        print("=" * 30 + " JAILBREAK BENCH " + "=" * 30)
        dataset_dict = load_jailbreak_bench_dataset(
            n_samples_ranking=n_samples_ranking,
            n_samples_training=n_samples_training,
            n_samples_evaluation=n_samples_evaluation,
            seed=seed,
            verbose=verbose,
        )
        print(dataset_dict)
    except Exception as e:
        # print(f"Error loading JailbreakBench dataset: {e}")
        raise e

    # Megascience Biology
    try:
        print("=" * 30 + " MEGASCIENCE BIOLOGY " + "=" * 30)
        dataset_dict = get_megascience_biology_dataset(
            n_samples_ranking=n_samples_ranking,
            n_samples_training=n_samples_training,
            n_samples_evaluation=n_samples_evaluation,
            seed=seed,
            verbose=verbose,
        )
        print(dataset_dict)
    except Exception as e:
        # print(f"Error loading Megascience Biology dataset: {e}")
        raise e

    # Megascience CS
    try:
        print("=" * 30 + " MEGASCIENCE CS " + "=" * 30)
        dataset_dict = get_megascience_cs(
            n_samples_ranking=n_samples_ranking,
            n_samples_training=n_samples_training,
            n_samples_evaluation=n_samples_evaluation,
            seed=seed,
            verbose=verbose,
        )
        print(dataset_dict)
    except Exception as e:
        # print(f"Error loading Megascience CS dataset: {e}")
        raise e

    # Megascience Math
    try:
        print("=" * 30 + " MEGASCIENCE MATH " + "=" * 30)
        dataset_dict = get_megascience_math_dataset(
            n_samples_ranking=n_samples_ranking,
            n_samples_training=n_samples_training,
            n_samples_evaluation=n_samples_evaluation,
            seed=seed,
            verbose=verbose,
        )
        print(dataset_dict)
    except Exception as e:
        # print(f"Error loading Megascience Math dataset: {e}")
        raise e

    # Megascience Chemistry
    try:
        print("=" * 30 + " MEGASCIENCE CHEMISTRY " + "=" * 30)
        dataset_dict = get_megascience_chemistry_dataset(
            n_samples_ranking=n_samples_ranking,
            n_samples_training=n_samples_training,
            n_samples_evaluation=n_samples_evaluation,
            seed=seed,
            verbose=verbose,
        )
        print(dataset_dict)
    except Exception as e:
        # print(f"Error loading Megascience Chemistry dataset: {e}")
        raise e

    # Megascience Physics
    try:
        print("=" * 30 + " MEGASCIENCE PHYSICS " + "=" * 30)
        dataset_dict = get_megascience_physics_dataset(
            n_samples_ranking=n_samples_ranking,
            n_samples_training=n_samples_training,
            n_samples_evaluation=n_samples_evaluation,
            seed=seed,
            verbose=verbose,
        )
        print(dataset_dict)
    except Exception as e:
        # print(f"Error loading Megascience Physics dataset: {e}")
        raise e

    # Codeparrot
    try:
        print("=" * 30 + " CODEPARROT " + "=" * 30)
        dataset_dict = load_codeparrot(
            n_samples_ranking=n_samples_ranking,
            n_samples_training=n_samples_training,
            n_samples_evaluation=n_samples_evaluation,
            seed=seed,
            verbose=verbose,
        )
        print(dataset_dict)
    except Exception as e:
        # print(f"Error loading Codeparrot dataset: {e}")
        raise e

    # APPS
    try:
        print("=" * 30 + " APPS " + "=" * 30)
        dataset_dict = load_apps(
            n_samples_ranking=n_samples_ranking,
            n_samples_training=n_samples_training,
            n_samples_evaluation=n_samples_evaluation,
            seed=seed,
            verbose=verbose,
        )
        print(dataset_dict)
    except Exception as e:
        # print(f"Error loading APPS dataset: {e}")
        raise e

    # DeepMind Code Contests
    try:
        print("=" * 30 + " DEEPMIND CODE CONTESTS " + "=" * 30)
        dataset_dict = load_deepmind_code_contests(
            n_samples_ranking=n_samples_ranking,
            n_samples_training=n_samples_training,
            n_samples_evaluation=n_samples_evaluation,
            seed=seed,
            verbose=verbose,
        )
        print(dataset_dict)
    except Exception as e:
        # print(f"Error loading DeepMind Code Contests dataset: {e}")
        raise e

    # Chinese Ultrachat
    try:
        print("=" * 30 + " CHINESE ULTRACHAT " + "=" * 30)
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-it")
        dataset_dict = load_chinese_ultrachat(
            n_samples_ranking=n_samples_ranking,
            n_samples_training=n_samples_training,
            n_samples_evaluation=n_samples_evaluation,
            seed=seed,
            verbose=verbose,
            tokenizer=tokenizer,
        )
        print(dataset_dict)
    except Exception as e:
        # print(f"Error loading Chinese Ultrachat dataset: {e}")
        raise e

    # Spanish Ultrachat
    try:
        print("=" * 30 + " SPANISH ULTRACHAT " + "=" * 30)
        dataset_dict = load_spanish_text(
            n_samples_ranking=n_samples_ranking,
            n_samples_training=n_samples_training,
            n_samples_evaluation=n_samples_evaluation,
            seed=seed,
            verbose=verbose,
        )
        print(dataset_dict)
    except Exception as e:
        # print(f"Error loading Spanish Ultrachat dataset: {e}")
        raise e

    # Poetry
    try:
        print("=" * 30 + " POETRY " + "=" * 30)
        dataset_dict = load_poetry_dataset(
            n_samples_ranking=n_samples_ranking,
            n_samples_training=n_samples_training,
            n_samples_evaluation=n_samples_evaluation,
            seed=seed,
            verbose=verbose,
        )
        print(dataset_dict)
    except Exception as e:
        print(f"Error loading Poetry dataset: {e}")

    # Spanish Poetry
    try:
        print("=" * 30 + " SPANISH POETRY " + "=" * 30)
        dataset_dict = load_spanish_poetry_dataset(
            n_samples_ranking=n_samples_ranking,
            n_samples_training=n_samples_training,
            n_samples_evaluation=n_samples_evaluation,
            seed=seed,
            verbose=verbose,
        )
        print(dataset_dict)
    except Exception as e:
        print(f"Error loading Spanish Poetry dataset: {e}")

    # IMDB Raw
    try:
        print("=" * 30 + " IMDB Itself " + "=" * 30)
        dataset_dict = load_dataset("stanfordnlp/imdb", split="train")
        dataset_dict = DatasetDict(
            {
                "train": dataset_dict,
                "validation": dataset_dict,
                "test": dataset_dict,
            }
        )
        print(dataset_dict)
    except Exception as e:
        # print(f"Error loading IMDB dataset: {e}")
        raise e

    print("=" * 30 + " ALL DONE " + "=" * 30)
    del dataset_dict
