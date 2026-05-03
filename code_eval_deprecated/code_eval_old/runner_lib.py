from __future__ import annotations

from typing import Dict, Any, List, Optional
import pydantic
import orjson
from pathlib import Path
import numpy as np
import json


"""
Store schemas and shared functionality across different scripts and tools that support
our code evaluation test/scaffolding.
"""


class ParseArguments(pydantic.BaseModel):
    func_name: str = "parse_between_str_pattern"
    func_kwargs: Dict[str, Any] = {
        "pattern": "```",
        # These should become ```
        "normalize_patterns": ["```python", "```python3"],
    }


PARSE_INFO_OPTIONS_MAP = {
    "default": ParseArguments(),  # defaults
}


class TestGeneration(pydantic.BaseModel):
    # We need one of these two, but either is fine.
    generation: Optional[str] = None
    code: Optional[str] = None

    # These are NECESSSARY fields
    inputs: List[Any]
    expected_outputs: List[Any]
    actual_outputs: Optional[List[Any]] = None  # populated by the test runner
    passed: Optional[List[bool]] = None  # populated by the test runner
    errors: Optional[List[Optional[str]]] = None  # populated by the test runner
    test_runtime_identifier: Optional[str] = None  # populated by the test runner

    # Metadata is where you can store information about what model/intervention
    # generated this generation and generally stuff like that. you can also store
    # identifiers for other judging/etc... (grouping, flattening, etc...)
    metadata: Optional[Dict[str, Any]] = None
    parse_info: Optional[ParseArguments | str] = "default"  # default parse arguments

    @staticmethod
    def parse_test_generations_file(file: Path) -> List[TestGeneration]:
        """
        Parse a JSON or JSONL file (which can be always a LIST of entries or a single entry)
        of TestGenerations. Each TestGeneration corresponds to one single generation from
        a model so usually you will have many of em.
        """
        contents: List[Dict[str, Any]] = []
        if file.suffix == ".json":
            contents = [orjson.loads(file.read_bytes())]
            if isinstance(contents[0], list):
                contents = contents[0]
        elif file.suffix == ".jsonl":
            contents = [orjson.loads(line.strip()) for line in file.read_text().splitlines() if line.strip()]  # fmt: skip
        assert all(isinstance(content, dict) for content in contents), "Contents must be a list of dictionaries"  # fmt: skip
        # Must have generation or code
        assert all((content.get("generation", None) is not None) or (content.get("code", None) is not None) for content in contents), f"All contents must have a generation or code field, contents={('\n'.join(json.dumps(c, indent=4) for c in contents))[:4096]}"  # fmt: skip
        # If you provide no parse info, then must already be parsed
        assert all((content.get("parse_info", None) is None) == (content.get("code", None) is not None) for content in contents), "All contents must have a code field if parse_info is not provided"  # fmt: skip

        # You must have inputs/outputs
        assert all("inputs" in content for content in contents), "All contents must have an input field"  # fmt: skip
        assert all("expected_outputs" in content for content in contents), "All contents must have an expected_output field"  # fmt: skip
        assert all(len(content["inputs"]) > 0 for content in contents), "All contents must have at least one input"  # fmt: skip
        assert all(len(content["inputs"]) == len(content["expected_outputs"]) for content in contents), "All contents must have the same number of inputs and expected outputs"  # fmt: skip

        # Done.
        return [TestGeneration.model_validate(content) for content in contents]


class Stats:
    """
    Static class to provide functions to calculate statistics such as strict acc. etc...
    """

    @staticmethod
    def _sans(x: List[List[bool] | None]) -> None:
        """
        It is OK to have different numbes of test-cases, but there must be at least one
        per generation.
        """
        assert len(x) > 0, "x (usually generations or smth) is empty"  # fmt: skip
        assert all(g is not None for g in x), f"n_None = {sum(g is None for g in x)}, out of {len(x)}"  # fmt: skip
        assert all(len(g) > 0 for g in x), f"n_empty = {sum(len(g) == 0 for g in x)}, out of {len(x)}"  # fmt: skip
        assert int(True) == 1 and int(False) == 0, f"int(True) = {int(True)}, int(False) = {int(False)}"  # fmt: skip

    @staticmethod
    def get_strict_accuracy(graded_generations: List[TestGeneration]) -> float:
        """
        Strict accuracy is the number of correct generations divided by the total number of generations.
        """
        Stats._sans([g.passed for g in graded_generations])
        strict_accs = [
            int(sum(map(int, g.passed)) == len(g.passed)) for g in graded_generations
        ]
        return np.mean(strict_accs).item()

    @staticmethod
    def get_weighted_average_accuracy(
        graded_generations: List[TestGeneration],
    ) -> float:
        """
        Get the mean of means basically.
        """
        Stats._sans([g.passed for g in graded_generations])
        array_accs = [np.array(list(map(int, g.passed))) for g in graded_generations]
        mean_accs = [np.mean(array_acc) for array_acc in array_accs]
        return np.mean(mean_accs).item()

    @staticmethod
    def get_flat_average_accuracy(graded_generations: List[TestGeneration]) -> float:
        """
        Get the mean overall by just taking #tests_passed/#tests_total.
        """
        Stats._sans([g.passed for g in graded_generations])
        n_tests_total = sum(len(g.passed) for g in graded_generations)
        n_tests_passed = sum(sum(map(int, g.passed)) for g in graded_generations)
        return float(n_tests_passed) / float(n_tests_total)
