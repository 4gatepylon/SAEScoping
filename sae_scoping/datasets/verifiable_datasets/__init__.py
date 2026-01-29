"""
Verifiable datasets module.

All loaders return canonical formats:
- MultipleChoiceDataset for MCQ datasets
- GoldenAnswerDataset for exact-match answer datasets

Usage:
    from sae_scoping.datasets.verifiable_datasets import load_mmlu, load_gsm8k

    mcq_data = load_mmlu(subject="moral_disputes", limit=100)
    math_data = load_gsm8k(limit=100)
"""

from sae_scoping.datasets.verifiable_datasets.mmlu import load_mmlu
from sae_scoping.datasets.verifiable_datasets.sec_qa import load_secqa
from sae_scoping.datasets.verifiable_datasets.wmdp import load_wmdp_cyber
from sae_scoping.datasets.verifiable_datasets.cybermetric import load_cybermetric
from sae_scoping.datasets.verifiable_datasets.gsm8k import load_gsm8k
from sae_scoping.datasets.verifiable_datasets.aimo_numinamath_15 import load_numinamath

__all__ = [
    "load_mmlu",
    "load_secqa",
    "load_wmdp_cyber",
    "load_cybermetric",
    "load_gsm8k",
    "load_numinamath",
]
