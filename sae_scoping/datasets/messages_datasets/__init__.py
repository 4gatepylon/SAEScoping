"""
Messages datasets module (for SFT training).

All loaders return canonical MessagesDataset format containing conversations
in OpenAI message format [{role, content}, ...].

Usage:
    from sae_scoping.datasets.messages_datasets import load_ultrachat, load_stemqa_math

    ultrachat = load_ultrachat(limit=100)
    math_data = load_stemqa_math(limit=100)

    # Access entries in canonical format
    for entry in ultrachat.entries:
        for msg in entry.messages:
            print(f"{msg.role}: {msg.content[:50]}...")
"""

from sae_scoping.datasets.messages_datasets.schemas import (
    Message,
    MessagesEntry,
    DatasetInfo,
    MessagesDataset,
)
from sae_scoping.datasets.messages_datasets.ultrachat import load_ultrachat
from sae_scoping.datasets.messages_datasets.stemqa_mixture import (
    load_stemqa_math,
    load_stemqa_biology,
    load_stemqa_chemistry,
    load_stemqa_physics,
)

__all__ = [
    # Schemas
    "Message",
    "MessagesEntry",
    "DatasetInfo",
    "MessagesDataset",
    # Loaders
    "load_ultrachat",
    "load_stemqa_math",
    "load_stemqa_biology",
    "load_stemqa_chemistry",
    "load_stemqa_physics",
]
