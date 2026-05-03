"""
StemQA token-length analysis under Gemma-2 and Gemma-3 tokenizers.

NOTE takeaway: 1024 is enough for >= 90p for each of the datasets.
"""

from __future__ import annotations

import tqdm
import numpy as np
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase

DATASET = "4gate/StemQAMixture"
SUBSETS = ("biology", "chemistry", "math", "physics")
SPLITS = ("train", "validation", "test")
TOKENIZERS = {
    "gemma-2": "google/gemma-2-2b-it",
    "gemma-3": "google/gemma-3-4b-it",
}
PERCENTILES = (50, 90, 95, 99)


def _compute_stats(arr: np.ndarray, percentiles: tuple[int, ...] = PERCENTILES) -> dict[str, float]:
    return {
        "mean": float(arr.mean()),
        "min": int(arr.min()),
        "max": int(arr.max()),
        **{f"p{p}": float(np.percentile(arr, p)) for p in percentiles},
    }


def _format_stats(s: dict[str, float]) -> str:
    return " ".join(f"{k}={v:.0f}" for k, v in s.items())


def _token_lengths(texts: list[str], tokenizer: PreTrainedTokenizerBase, batch_size: int = 256) -> np.ndarray:
    enc: list[list[int]] = []
    for i in tqdm.trange(0, len(texts), batch_size):
        enc.extend(tokenizer(texts[i : i + batch_size], add_special_tokens=True, padding=False, truncation=False)["input_ids"])
    return np.fromiter((len(ids) for ids in enc), dtype=np.int64, count=len(texts))


def _analyze(
    ds: Dataset, tokenizers: dict[str, PreTrainedTokenizerBase], tokenization_batch_size: int = 256
) -> dict[str, dict[str, dict[str, float]]]:
    questions = [str(r["question"]) for r in ds]
    answers = [str(r["answer"]) for r in ds]
    out: dict[str, dict[str, dict[str, float]]] = {}
    for name, tok in tokenizers.items():
        q_len = _token_lengths(questions, tok)
        a_len = _token_lengths(answers, tok)
        out[name] = {
            "question": _compute_stats(q_len),
            "answer": _compute_stats(a_len),
            "total": _compute_stats(q_len + a_len),
        }
    return out


def main() -> None:
    tokenizers = {name: AutoTokenizer.from_pretrained(mid, use_fast=True) for name, mid in TOKENIZERS.items()}
    for subset in SUBSETS:
        for split in SPLITS:
            # 1. Dataset loading
            try:
                ds = load_dataset(DATASET, subset, split=split)
            except Exception as e:
                print(f"{subset}/{split}: skipped ({type(e).__name__}: {e})")
                continue
            # 2. Calculate results
            results = _analyze(ds, tokenizers)
            # 3. Print out the results
            print(f"\n{subset}/{split}  n={len(ds):,}")
            for tok_name, fields in results.items():
                for field, stats in fields.items():
                    print(f"  [{tok_name}] {field:<8} {_format_stats(stats)}")


if __name__ == "__main__":
    main()
