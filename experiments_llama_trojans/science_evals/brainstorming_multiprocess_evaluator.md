# Brainstorming: Multiprocess Evaluator

## Goal

Create a parallel evaluation system that:
1. Spreads work across multiple servers (from a server group)
2. Supports different data sources via an enum
3. Abstracts "checking" (LLM judges, regex, exact match) into a unified interface

---

## Current State Analysis

### What's Done Well
1. **Module separation** - Each file has a clear responsibility (data_loading, judging, aggregation, etc.)
2. **Config validation** - Pydantic provides strong typing
3. **Judge templates** - Template registry with caching
4. **Type annotations** - Uses beartype/jaxtyping throughout

### What's Missing or Needs Work

**1. No Checker Abstraction (Major Gap)**
- Currently only LLM judges exist in `judging.py`
- `run_judges()` is hardcoded to use `APIGenerator` with JSON mode
- No support for regex extraction, exact match, or other checking methods
- The judging code is ~265 lines and could be refactored

**2. Data Source Not Enum-Based**
- Currently uses a mix of:
  - `use_hardcoded_bio_prompts: bool`
  - `subjects: list[str]` (biology, chemistry, math, physics)
  - `include_malicious: bool`
- Not cleanly abstracted into a single enum

**3. No Parallel Evaluation Support**
- `evaluate_science.py` runs everything sequentially
- Batch mode iterates models one at a time
- No infrastructure to spread work across multiple servers

**4. Bloat/Coupling Issues**
- `run_single_evaluation()` is ~300 lines and monolithic
- Judge selection is hardcoded inline (benign → 3 judges, malicious → 4 judges)
- Generation only supports OpenAI-compatible servers (file/HF modes mentioned in plan but not implemented)

---

## Proposed Architecture

### 1. Checker Abstraction

Location: `sae_scoping/evaluation/checkers/` (or keep in experiments if not reusable)

```
checkers/
├── __init__.py
├── base.py          # Abstract Checker interface
├── llm_judge.py     # Wraps existing judging.py logic
├── regex.py         # Regex extraction + comparison
└── exact_match.py   # Simple string comparison
```

**Interface:**
```python
from abc import ABC, abstractmethod
from typing import Any

class Checker(ABC):
    @abstractmethod
    def check_batch(
        self,
        requests: list[str],
        responses: list[str],
        **kwargs
    ) -> list[dict[str, Any]]:
        """
        Check a batch of request-response pairs.

        Returns list of dicts with at least:
            - "score": float in [0, 1]
            - "explanation": str (optional for non-LLM checkers)
        """
        pass


class LLMJudgeChecker(Checker):
    """Wraps existing judging.py logic."""

    def __init__(
        self,
        judge_names: list[str],  # ["answering", "refusal", etc.]
        judge_model: str = "gpt-4.1-nano",
        batch_size: int = 32,
        max_tokens: int = 256,
    ):
        self.judge_names = judge_names
        self.judge_model = judge_model
        self.batch_size = batch_size
        self.max_tokens = max_tokens

    def check_batch(self, requests, responses, **kwargs):
        # Delegates to existing run_judges()
        from .judging import run_judges
        return run_judges(
            requests, responses,
            self.judge_names, self.judge_model,
            self.batch_size, self.max_tokens
        )


class RegexChecker(Checker):
    """Extract answer via regex, compare to expected."""

    def __init__(self, pattern: str, expected_key: str = "answer"):
        self.pattern = re.compile(pattern)
        self.expected_key = expected_key

    def check_batch(self, requests, responses, expected=None, **kwargs):
        results = []
        for resp, exp in zip(responses, expected or [None] * len(responses)):
            match = self.pattern.search(resp)
            extracted = match.group(1) if match else None
            score = 1.0 if extracted == exp else 0.0
            results.append({
                "score": score,
                "extracted": extracted,
                "expected": exp,
            })
        return results


class ExactMatchChecker(Checker):
    """Simple string comparison."""

    def __init__(self, normalize: bool = True):
        self.normalize = normalize

    def check_batch(self, requests, responses, expected=None, **kwargs):
        results = []
        for resp, exp in zip(responses, expected or [None] * len(responses)):
            if self.normalize:
                resp = resp.strip().lower()
                exp = exp.strip().lower() if exp else None
            score = 1.0 if resp == exp else 0.0
            results.append({"score": score, "response": resp, "expected": exp})
        return results
```

### 2. DataSource Enum

Location: `data_loading.py` (extend existing)

```python
from enum import Enum

class DataSource(str, Enum):
    # Spylab hardcoded prompts
    SPYLAB_BIOLOGY = "spylab_biology"
    SPYLAB_MALICIOUS = "spylab_malicious"

    # Science datasets from JSONL files
    SCIENCE_BIOLOGY = "science_biology"
    SCIENCE_PHYSICS = "science_physics"
    SCIENCE_CHEMISTRY = "science_chemistry"
    SCIENCE_MATH = "science_math"


def load_data(
    source: DataSource,
    limit: int | None = None,
    seed: int = 42,
    shuffle: bool = True,
    system_prompt: str | None = None,
) -> list[list[dict[str, str]]]:
    """Unified data loader based on source enum."""

    if source == DataSource.SPYLAB_BIOLOGY:
        return load_hardcoded_bio_prompts(limit, seed, shuffle, system_prompt)

    elif source == DataSource.SPYLAB_MALICIOUS:
        return load_malicious_prompts(limit, seed, shuffle, system_prompt)

    elif source.value.startswith("science_"):
        subject = source.value.replace("science_", "")
        return load_science_dataset(
            subjects=[subject],
            limit=limit,
            seed=seed,
            shuffle=shuffle,
        )

    else:
        raise ValueError(f"Unknown data source: {source}")
```

### 3. Parallel Evaluator

Location: `experiments_llama_trojans/science_evals/parallel_evaluate.py`

```python
"""
Parallel evaluator that spreads work across multiple servers.

Usage:
    python parallel_evaluate.py \
        --server-group spylab_trojan1_biology_server_group \
        --data-source spylab_biology \
        --checker llm_judge \
        --output-dir outputs/parallel_eval
"""

import multiprocessing as mp
from typing import Any

from sae_scoping.servers.model_configs.individual_configs.name_resolution import (
    resolve_group_config_path,
)


def split_data_round_robin(data: list, n_workers: int) -> list[list]:
    """Split data round-robin across workers."""
    chunks = [[] for _ in range(n_workers)]
    for i, item in enumerate(data):
        chunks[i % n_workers].append(item)
    return chunks


def split_data_contiguous(data: list, n_workers: int) -> list[list]:
    """Split data into contiguous chunks."""
    chunk_size = len(data) // n_workers
    remainder = len(data) % n_workers

    chunks = []
    start = 0
    for i in range(n_workers):
        end = start + chunk_size + (1 if i < remainder else 0)
        chunks.append(data[start:end])
        start = end
    return chunks


def worker_evaluate(
    worker_id: int,
    server_url: str,
    data_chunk: list,
    checker: "Checker",
    generation_kwargs: dict,
) -> dict[str, Any]:
    """Worker function that runs in a separate process."""
    from generation import generate_responses

    # Generate responses
    responses = generate_responses(
        data_chunk,
        base_url=server_url,
        **generation_kwargs,
    )

    # Extract user requests for checker
    requests = [conv[-1]["content"] for conv in data_chunk]

    # Run checker
    results = checker.check_batch(requests, responses)

    return {
        "worker_id": worker_id,
        "n_items": len(data_chunk),
        "n_success": sum(1 for r in responses if r is not None),
        "results": results,
    }


def parallel_evaluate(
    server_group_config: str,
    data_source: "DataSource",
    checker: "Checker",
    output_dir: str,
    limit: int | None = None,
    seed: int = 42,
    split_mode: str = "contiguous",  # or "round_robin"
    generation_kwargs: dict | None = None,
) -> dict[str, Any]:
    """
    Run evaluation in parallel across multiple servers.

    Args:
        server_group_config: Path/name of server group config
        data_source: Which data to evaluate on
        checker: Checker instance for scoring responses
        output_dir: Where to save results
        limit: Max items per data source
        seed: Random seed for shuffling
        split_mode: How to split data across workers
        generation_kwargs: Passed to generate_responses()

    Returns:
        Aggregated results from all workers
    """
    import json
    from pathlib import Path

    # 1. Load server group config
    group_path = resolve_group_config_path(server_group_config)
    with open(group_path) as f:
        server_configs = json.load(f)

    n_servers = len(server_configs)
    # Assume servers are on ports 8000, 8001, 8002, ...
    server_urls = [f"http://localhost:{8000 + i}" for i in range(n_servers)]

    # 2. Load data
    from data_loading import load_data
    data = load_data(data_source, limit=limit, seed=seed)

    # 3. Split data across servers
    if split_mode == "contiguous":
        chunks = split_data_contiguous(data, n_servers)
    else:
        chunks = split_data_round_robin(data, n_servers)

    print(f"Splitting {len(data)} items across {n_servers} servers")
    for i, chunk in enumerate(chunks):
        print(f"  Server {i}: {len(chunk)} items")

    # 4. Run workers in parallel
    generation_kwargs = generation_kwargs or {}

    ctx = mp.get_context("spawn")  # For CUDA safety
    with mp.Pool(n_servers, context=ctx) as pool:
        worker_args = [
            (i, server_urls[i], chunks[i], checker, generation_kwargs)
            for i in range(n_servers)
        ]
        results = pool.starmap(worker_evaluate, worker_args)

    # 5. Aggregate results
    all_results = []
    total_items = 0
    total_success = 0
    for r in results:
        all_results.extend(r["results"])
        total_items += r["n_items"]
        total_success += r["n_success"]

    # Compute aggregate score
    scores = [r["score"] for r in all_results if "score" in r]
    avg_score = sum(scores) / len(scores) if scores else 0.0

    output = {
        "data_source": data_source.value,
        "n_servers": n_servers,
        "n_items": total_items,
        "n_success": total_success,
        "avg_score": avg_score,
        "per_worker": results,
    }

    # 6. Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    with open(output_path / "results.json", "w") as f:
        json.dump(output, f, indent=2)

    return output
```

---

## Open Questions

1. **Checker output format**: Should all checkers return the same schema (`{score, explanation}`) or should regex/exact match have simpler output (`{match: bool}`)?

2. **Data splitting strategy**: Round-robin (item 0 → server 0, item 1 → server 1, ...) or contiguous chunks (items 0-99 → server 0, items 100-199 → server 1)?

3. **Where to put checker code**: In `sae_scoping/evaluation/checkers/` (library) or `experiments_llama_trojans/science_evals/` (experiment-specific)?

4. **Config format for parallel eval**: Reuse `ScienceEvalsConfig` with server group reference, or new config schema?

5. **How to handle server URLs**: Currently assumes ports 8000, 8001, ... but could read from server group config or add explicit mapping.

6. **Trojan handling in parallel mode**: Should each server handle all trojan variants, or split trojan variants across servers too?

---

## Estimated Scope

| Component | Lines of Code | Difficulty |
|-----------|---------------|------------|
| Checker base + LLM wrapper | ~150 | Medium |
| Regex checker | ~80 | Easy |
| Exact match checker | ~40 | Easy |
| DataSource enum + unified loader | ~60 | Easy |
| Parallel evaluator | ~200 | Medium |
| Config updates | ~50 | Easy |
| **Total** | **~580** | |

---

## Alternative Approaches

### Option A: Thin Wrapper (Recommended for now)
Just wrap the existing `evaluate_science.py` with multiprocessing. Each worker runs the full evaluation pipeline on its data chunk. Minimal code changes.

### Option B: Full Refactor
Break apart `run_single_evaluation()` into composable steps, create proper abstractions. More work but cleaner long-term.

### Option C: External Orchestration
Use something like Ray or Dask for distributed evaluation. More infrastructure but scales better.

---

## Notes

- The existing code is well-structured but monolithic
- Main pain point is the 300-line `run_single_evaluation()` function
- Checker abstraction is the biggest missing piece
- Parallel evaluation is straightforward once data splitting is decided
