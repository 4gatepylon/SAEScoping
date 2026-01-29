"""
Verifiable datasets module.

All loaders return canonical formats:
- MultipleChoiceDataset for MCQ datasets (MMLU, SecQA, WMDP, CyberMetric)
- GoldenAnswerDataset for exact-match answer datasets (GSM8K, NuminaMath, IMDB, Camel AI, MATH-500)
- ExecutableTestDataset for code problems with test cases (APPS, Code Contests)

Usage:
    from sae_scoping.datasets.verifiable_datasets import load_mmlu, load_gsm8k

    mcq_data = load_mmlu(subject="moral_disputes", limit=100)
    math_data = load_gsm8k(limit=100)

    # Access entries in canonical format
    for entry in mcq_data.entries:
        print(entry.question, entry.choices, entry.answer_letter)
"""

from sae_scoping.datasets.verifiable_datasets.schemas import (
    ANSWER_LETTERS,
    MultipleChoiceEntry,
    GoldenAnswerEntry,
    ExecutableTestEntry,
    DatasetInfo,
    MultipleChoiceDataset,
    GoldenAnswerDataset,
    ExecutableTestDataset,
)
from sae_scoping.datasets.verifiable_datasets.mmlu import load_mmlu
from sae_scoping.datasets.verifiable_datasets.sec_qa import load_secqa
from sae_scoping.datasets.verifiable_datasets.wmdp import load_wmdp_cyber
from sae_scoping.datasets.verifiable_datasets.cybermetric import load_cybermetric
from sae_scoping.datasets.verifiable_datasets.gsm8k import load_gsm8k
from sae_scoping.datasets.verifiable_datasets.aimo_numinamath_15 import load_numinamath
from sae_scoping.datasets.verifiable_datasets.imdb import load_imdb
from sae_scoping.datasets.verifiable_datasets.apps import load_apps
from sae_scoping.datasets.verifiable_datasets.camel_ai import (
    load_camel_ai_biology,
    load_camel_ai_chemistry,
    load_camel_ai_physics,
    load_camel_ai_math,
)
from sae_scoping.datasets.verifiable_datasets.code_contests import load_code_contests
from sae_scoping.datasets.verifiable_datasets.math500 import load_math500

__all__ = [
    # Schemas
    "ANSWER_LETTERS",
    "MultipleChoiceEntry",
    "GoldenAnswerEntry",
    "ExecutableTestEntry",
    "DatasetInfo",
    "MultipleChoiceDataset",
    "GoldenAnswerDataset",
    "ExecutableTestDataset",
    # MCQ dataset loaders
    "load_mmlu",
    "load_secqa",
    "load_wmdp_cyber",
    "load_cybermetric",
    # Golden answer dataset loaders
    "load_gsm8k",
    "load_numinamath",
    "load_imdb",
    "load_camel_ai_biology",
    "load_camel_ai_chemistry",
    "load_camel_ai_physics",
    "load_camel_ai_math",
    "load_math500",
    # Executable test dataset loaders
    "load_apps",
    "load_code_contests",
]
