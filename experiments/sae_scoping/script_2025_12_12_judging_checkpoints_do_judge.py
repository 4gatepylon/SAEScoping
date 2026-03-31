from __future__ import annotations
import pandas as pd
import tqdm
import itertools
from transformers import AutoTokenizer
import hashlib
import os
import json
import traceback
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import gc
import click
import multiprocessing as mp
from beartype import beartype
from beartype.typing import Any
from functools import partial
import orjson
from sae_scoping.evaluation.inference.client.api_generator import (
    load_jinja_template,
    APIGenerator,
)

# NOTE: all imports that may remotely use torch should be inside wrapper fns to
# avoid cuda issues (defensive programming)

"""
# High-level overview
This script will take ingenerated outputs from the various different LLMs we have tested and then
judge them. The LLMs we test include SAE-enhanced and vanilla models. This means that we can use
these judgements to compare whether the SAE-enhanced models may or may not be as effective as
the vanilla ones in-domain (and possibly OOD too).

NOTE that vanilla here refers to two things at different times:
- The original model without an SAE
- A possibly finetuned version of the original model also without the SAE, but with different weights

# Inputs description
Operate on a path that looks like this:

.
├── <special hash for the specific config n shit>
│   └── config.json
│   └── generations.jsonl
│   └── error.json

and we have error.json looks like:
```json
{
    "success": bool,
    "error": str | None, # none means not present potentially
    "traceback": str | None, # none means not present potentially
}
```

and generations.jsonl is a sequence of new-line seperated identical JSONs, each like:
```json
{
    "type": "imdb" | "biology" | "apps" | "ultrachat", (or some other such string)
    "messages": [
        {
            "role": "user" | "assistant" | "system",
            "content": str,
        },
        ... # i.e. messages is an OpenAI API chat format list
    ],
}
```

and config looks like (imported from script_2025_12_12_judging_checkpoints_do_generation.py): 
```
checkpoint_path: Path | None  # None for vanilla <- gets converted into string with as_posix()
dataset_name: str
sae_id: str | None  # None for vanilla
layer: int | None  # None for vanilla
threshold: float | None  # None for vanilla
hash_suffix: str | None  # None for vanilla
step: int
```

# Outputs format
Exactly the same as inputs format (actually everything is even copied exactly!)

HOWEVER we add one additional file called "judgements.parquet" which includes all the judgements.

Aggregation is done in a separate (third) script. We will refactor this in the main library very soon.

Templates (official old) are here: https://github.com/4gatepylon/Deprecated-ScopeBench/blob/0c30cda68f0a0712c00864e8ab92a28e2994389e/prompt_engineering/prompts/judge_prompts_refusal_safety/generic/refusal.jinja2#L1
"""


def canonicalize_judgement_json(judgement: dict[str, Any]) -> dict[str, Any]:
    """
    Return something with keys 'error', 'score', and 'explanation' and 'has_error'.

    'error' is None IFF 'score' is a float in [0, 1] (if its a bool or int its converted)
    and 'explanation' is a string. Otherwise the other two are whatever (could be any serializeable)
    and 'error' is a string explaining the error. 'has_error' = bool(error is not None)
    """
    score_raw = judgement.get("score", None)
    explanation_raw = judgement.get("explanation", None)
    error_raw = judgement.get("error", None)
    if error_raw is not None:
        return {
            "has_error": True,
            "error": str(error_raw) if error_raw is not None else "MissingRequiredKeys",
            "score": score_raw,
            "explanation": explanation_raw,
        }

    # Check if required keys are missing (API error case)
    if "score" not in judgement or "explanation" not in judgement:
        error_msg = judgement.get("error", "MissingRequiredKeys")
        return {
            "has_error": True,
            "error": str(error_msg) if error_msg is not None else "MissingRequiredKeys",
            "score": score_raw,
            "explanation": explanation_raw,
        }

    score = score_raw

    # Convert bool/int to float
    if isinstance(score, bool):
        score = 1.0 if score else 0.0
    elif isinstance(score, int):
        score = float(score)
    elif not isinstance(score, float):
        return {
            "has_error": True,
            "error": f"InvalidScoreType: {type(score).__mro__}",
            "score": score_raw,
            "explanation": explanation_raw,
        }

    # Check score is in [0, 1]
    if not (0.0 <= score <= 1.0):
        return {
            "has_error": True,
            "error": f"ScoreOutOfRange: {score}",
            "score": score_raw,
            "explanation": explanation_raw,
        }

    # Validate explanation is a string
    if not isinstance(explanation_raw, str):
        return {
            "has_error": True,
            "error": f"InvalidExplanationType: {type(explanation_raw).__name__}",
            "score": score,
            "explanation": explanation_raw,
        }

    # All valid!
    return {
        "has_error": False,
        "error": None,
        "score": score,
        "explanation": explanation_raw,
    }


@click.command()
@click.option(
    "--input-path",
    "-i",
    type=click.Path(exists=True, path_type=Path),
    default=Path(__file__).parent / "outputs_gemma9b_judging_generations",
    help="Path to input directory where we have stored the generations for the various different LLMs.",
)
@click.option(
    "--output-path",
    "-o",
    type=click.Path(path_type=Path),
    default=Path(__file__).parent / "outputs_gemma9b_judging_judgements",
    help="Path to output directory where we will store the judgements.",
)
@click.option(
    "--debug-aggregate-fake-judgements",
    "--dry-run",
    "-dry",
    is_flag=True,
    help="Only list checkpoints, don't load them",
)
def main(
    input_path: Path | str,
    output_path: Path | str,
    debug_aggregate_fake_judgements: bool,
) -> None:
    # 1. Load classifiers
    prompts_folder = (
        Path(__file__).parent.parent
        / "sae_scoping"
        / "xxx_evaluation"
        / "iclr_judge_prompts"
    )
    assert prompts_folder.exists()
    classifiers = {
        f.name: load_jinja_template(f) for f in prompts_folder.iterdir() if f.is_file()
    }
    assert len(classifiers) > 0
    print("=" * 100)
    print("Using the following classifiers:")
    print(", ".join(sorted(classifiers.keys())))

    # 2. API Generate this shit
    # -> Fetch all typed queries that are used by the various different checkpoints
    # -> for each one for each judge judge it and store that somewhere
    # (with canonicalized format so that we can easily extract the results)
    # (_type, (A, B, C), template_name, config, subfolder name)
    # A = Query
    # B = Answer
    # C = Golden answer
    typed_query_answers_to_judge: list[
        tuple[str, tuple[str, str, str], str, dict, str]
    ] = []
    typed_queries_file = input_path / "typed_queries.json"
    typed_queries = orjson.loads(typed_queries_file.read_bytes())
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-9b-it")
    # NOTE: response is GOLDEN
    assert all(q["messages"][-1]["role"] == "assistant" for q in typed_queries)
    assert all(q["messages"][-2]["role"] == "user" for q in typed_queries)
    _expected_contents: set[tuple[str, str]] = set()
    _templatted2golden: dict[str, str] = {}
    for tq in typed_queries:
        _type, c = tq["type"], tq["messages"]
        templatted = tokenizer.apply_chat_template(
            c[:-1], tokenize=False, add_generation_prompt=True
        )
        _expected_contents.add((_type, templatted))
        _templatted2golden[templatted] = c[-1]["content"]

    # (ensure that the set of typed queries is the same for all checkpoints)
    for subfolder in tqdm.tqdm(
        list(input_path.iterdir()),
        desc="Checking everyone else + adding to typed_query_answers_to_judge",
    ):
        # 1. skip OG
        if not subfolder.is_dir():
            assert subfolder.name == "typed_queries.json", (
                f"Unexpected file: {subfolder}"
            )
            continue
        # 2. Load
        generations_file = subfolder / "generations.jsonl"
        assert generations_file.exists()
        contents = [
            orjson.loads(line) for line in generations_file.read_bytes().splitlines()
        ]
        # 3. Sanity check + add to typed_query_answers_to_judge
        assert all("messages" in c for c in contents)
        non_1shot = ["imdb"]
        assert all(c["type"] in non_1shot or len(c["messages"]) == 2 for c in contents)
        assert all(c["messages"][-1]["role"] == "assistant" for c in contents)
        assert all(c["messages"][-2]["role"] == "user" for c in contents)
        _gotten_contents: set[tuple[str, str]] = set()
        config_file = subfolder / "config.json"
        config = orjson.loads(config_file.read_bytes())
        for content in contents:
            _type, c = content["type"], content["messages"]
            templatted = tokenizer.apply_chat_template(
                c[:-1], tokenize=False, add_generation_prompt=True
            )
            _gotten_contents.add((_type, templatted))
            for template_name in classifiers.keys():
                typed_query_answers_to_judge.append(
                    (
                        _type,
                        (
                            c[-2]["content"],  # user request
                            c[-1]["content"],  # assistant response
                            _templatted2golden[templatted],  # golden response
                        ),
                        template_name,
                        config,
                        subfolder.name,
                    )
                )
        assert _gotten_contents == _expected_contents
    print("[OK] All typed queries are the same for all checkpoints. Epic.")
    print(
        f"READY TO JUDGE BABY. We have {len(typed_query_answers_to_judge)} typed query answers to judge."
    )
    print("Converting to pd dataframe")
    df = pd.DataFrame(
        [
            {
                "evaluate_dataset_name": _type,
                "user_request": user_request,
                "assistant_response": assistant_response,
                "golden_response": golden_response,
                "judge_name": template_name,
                "hydrated_judge_template": classifiers[template_name].render(
                    {
                        "user_request": user_request,
                        "assistant_response": assistant_response,
                        "golden_response": golden_response,  # Usually not used but eh
                    }
                ),
                # "config": config, # unrolled instead
                "checkpoint_path": config["checkpoint_path"],
                "train_dataset_name": config["dataset_name"],
                "sae_id": config["sae_id"],
                "layer": config["layer"],
                "threshold": config["threshold"],
                "step": config["step"],
                "config_id": subfolder_name,
            }
            for (
                _type,
                (
                    user_request,
                    assistant_response,
                    golden_response,
                ),
                template_name,
                config,
                subfolder_name,
            ) in tqdm.tqdm(
                typed_query_answers_to_judge, desc="Converting to pd dataframe"
            )
        ],
        columns=[
            "evaluate_dataset_name",
            "user_request",
            "assistant_response",
            "golden_response",
            "judge_name",
            "hydrated_judge_template",
            "checkpoint_path",
            "train_dataset_name",
            "sae_id",
            "layer",
            "threshold",
            "step",
            "config_id",
        ],
    )
    print(f"df: {df.head()}, len: {len(df)}")
    print("=" * 100)
    # save to the output folder
    output_path.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path / "judgements_pre.parquet")
    print("=" * 100)
    print("Judging...")
    api_generator = APIGenerator()
    judgements = api_generator.api_generate_json_mode_streaming(
        prompts=df["hydrated_judge_template"].tolist(),  # [:100],
        model="gpt-4.1-nano",
        batch_size=500,
        # These things below are standardized for all JSON Mode API Generations
        default_json_for_none={"error": "None"},
        default_json_for_keys_fn=lambda _: {"error": "MissingKeys"},
        default_json_for_json_loads_decode_error_fn=lambda _1, _2: {
            "error": "JSONDecodeError"
        },
        must_have_keys=["explanation", "score"],
    )
    judgements_canon = map(canonicalize_judgement_json, judgements)
    judged_elements = []
    for judgement_canon, not_judged_element in tqdm.tqdm(
        zip(judgements_canon, df.itertuples()),
        desc="Canonicalizing judgements",
        total=len(df),
    ):
        assert set(judgement_canon.keys()) == {
            "error",
            "score",
            "explanation",
            "has_error",
        }
        new_element = {
            # Pass through the original elements
            "evaluate_dataset_name": not_judged_element.evaluate_dataset_name,
            "user_request": not_judged_element.user_request,
            "assistant_response": not_judged_element.assistant_response,
            "golden_response": not_judged_element.golden_response,
            "judge_name": not_judged_element.judge_name,
            "hydrated_judge_template": not_judged_element.hydrated_judge_template,
            "checkpoint_path": not_judged_element.checkpoint_path,
            "train_dataset_name": not_judged_element.train_dataset_name,
            "sae_id": not_judged_element.sae_id,
            "layer": not_judged_element.layer,
            "threshold": not_judged_element.threshold,
            "step": not_judged_element.step,
            "config_id": not_judged_element.config_id,
            # Put in the judgement results
            "judgement_error": judgement_canon["error"],
            "judgement_score": judgement_canon["score"],
            "judgement_explanation": judgement_canon["explanation"],
            "judgement_has_error": judgement_canon["has_error"],
        }
        judged_elements.append(new_element)
    df_judged = pd.DataFrame(judged_elements)
    print(f"df_judged: {df_judged.head()}, len: {len(df_judged)}")
    print("=" * 100)
    df_judged.to_parquet(output_path / "judgements_post.parquet")
    print("=" * 100)
    # -> Other code will analyze the parquet, aggregate, etc...
    print("Done!")
    # -> OK just do some simple analysis here


if __name__ == "__main__":
    main()
