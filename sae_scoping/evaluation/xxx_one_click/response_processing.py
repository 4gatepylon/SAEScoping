from __future__ import annotations
from typing import Any
import json
from beartype import beartype

"""This module provides utilities for processing responses from LLM judges or other grading systems, evaluators, etc..."""


@beartype
def canonicalize_judgement_dict(
    judgement_dict: Any,
    score_key: str = "score",
    explanation_key: str = "explanation",
) -> tuple[dict[str, str], bool]:
    """
    Many judges are instructed to return a score and an explanation of that score
    (judgement) in a JSON dictionary. This function is meant to guarantee outputs
    that are properly formatted even if errors are present. It enables easy reduction
    later-on. Proper users should raise if the number of errors is too high.
    """
    if judgement_dict is None:
        return {
            "score": 0.0,
            "explanation": "Error: None for some reason (could be "
            + "internet/API error: basically API did not respond at all properly).",
        }, True  # is error
    elif not isinstance(judgement_dict, dict):
        return {
            "score": 0.0,
            "explanation": "Error: Not a dictionary for some reason (WTF): "
            + f"{judgement_dict}",
        }, True  # is error
    elif (
        set(judgement_dict.keys()) != {score_key, explanation_key}
        or not isinstance(judgement_dict[score_key], (float, bool, int))
        or float(judgement_dict[score_key]) > 1
        or float(judgement_dict[score_key]) < 0
    ):
        dump = "ERROR: Cannot dump"
        try:
            dump = json.dumps(judgement_dict)
            dump = f"ERROR: {dump}"
        except Exception as ee:
            dump = f"ERROR: Tried to dump but failed: {ee}"
        return {
            score_key: 0.0,
            explanation_key: f"Unknown error (could be internet/API error: basically "
            + f"API did not respond at all properly). Contents:\n\n```{dump}```\n\n",
        }, True
    else:
        return {
            score_key: float(judgement_dict[score_key]),
            explanation_key: judgement_dict[explanation_key],
        }, False  # is not error
