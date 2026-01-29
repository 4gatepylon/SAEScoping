# Evaluation Refactor Task

## Current State: `sae_scoping/evaluation/xxx_old_stuff/`

This document describes the existing evaluation workflow patterns found in the codebase, to inform a future refactor into a unified API.

---

## Overview

The evaluation code currently consists of one-off scripts that share common patterns but have significant duplication. There are two main evaluation paradigms:

1. **Multiple Choice (MCQ) Evaluation** - Extract answers from model responses and compare to ground truth
2. **Judge-based Evaluation** - Use LLM judges to score open-ended responses

---

## Evaluation Types

### 1. Multiple Choice (MCQ) Evaluation

**Files:**
- `script_2026_01_24_secqa_eval.py` - SecQA, WMDP-Cyber, CyberMetric
- `script_2026_01_25_math_eval.py` - GSM8K, NuminaMath
- `script_2026_01_22_batched_gepa_mmlu_mvp.py` - MMLU (with DSPy/GEPA optimization)

**Workflow:**
1. Load dataset (question + choices + ground truth answer)
2. Format prompts (system prompt + user message with question/choices)
3. Generate responses from model
4. Extract answer from response using regex patterns
5. Compare extracted answer to ground truth
6. Aggregate accuracy statistics

**Answer Extraction Methods:**
| Method | Pattern | Example |
|--------|---------|---------|
| `boxed` | `\boxed{X}` | `\boxed{A}`, `\boxed{42}` |
| `gsm8k` | `#### X` | `#### 42` |
| `equality` | Direct string match | Response == "A" |
| `first_letter` | First character | Response starts with "A" |
| `judge_*` | LLM extracts answer | Fallback when regex fails |

**Reducer Pattern:**
Multiple extractors can be combined. The `or` reducer returns correct if ANY extractor matches ground truth.

```python
CHECK_MODE2FN_REGISTRY = {
    "boxed": check_answer_boxed,
    "equality": check_answer_equality,
    "first_letter": check_answer_first_letter,
}
```

### 2. Judge-based Evaluation

**Files:**
- `utility_1click_judgement.py` - Biology-specific utility evaluation
- `utility_1click_judgement_generic.py` - Generic subject utility (biology, chemistry, physics)
- `script_2026_01_23_evaluate_biology_utility.py` - Orchestrates biology evaluation
- `script_2026_01_23_evaluate_subject_utility.py` - Orchestrates multi-subject evaluation

**Workflow:**
1. Load prompts from dataset (e.g., camel-ai/biology)
2. Format prompts using `SpylabPreprocessor`
3. Generate responses from model
4. For each response, run multiple LLM judges
5. Each judge returns a score (0-1) and explanation
6. Aggregate scores across judges

**Judge Types (jinja2 templates in `iclr_judge_prompts/`):**
- `answering_classifier.j2` - Did the model answer the question?
- `factual_helpful_classifier.j2` - Is the response factually correct and helpful?
- `precise_classifier.j2` - Is the response precise and on-topic?

**Output Format:**
```python
{
    "utility/answering": 0.95,
    "utility/factual_helpful": 0.90,
    "utility/precise": 0.85,
    "utility/overall": 0.90,  # Mean of all judges
}
```

---

## Generation Sources

### 1. HFGenerator (`sae_scoping/utils/generation/hf_generator.py`)

Local HuggingFace model inference with batching.

```python
class HFGenerator:
    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase):
        ...

    def generate_stream(
        self,
        conversations: Iterable[list[dict[str, str]]],  # OpenAI message format
        batch_size: int = 32,
        generation_kwargs: dict = {"max_new_tokens": 512, "do_sample": False},
    ) -> Generator[str, None, None]:
        ...
```

**Key features:**
- Accepts conversations in OpenAI message format
- Uses `tokenizer.apply_chat_template()` for formatting
- Left-pads for batch generation
- Yields decoded response strings

### 2. APIGenerator (`sae_scoping/utils/generation/api_generator.py`)

LiteLLM-based API calls to OpenAI, Anthropic, vLLM servers, etc.

```python
class APIGenerator:
    def api_generate_streaming(
        self,
        prompts: list[str] | list[list[dict[str, str]]],
        model: str,  # e.g., "gpt-4.1-nano", "hosted_vllm/google/gemma-2-9b-it"
        batch_size: int = 16,
        batch_completion_kwargs: dict = {},  # api_base, max_tokens, etc.
    ) -> Iterator[str | None]:
        ...

    def api_generate_json_mode_streaming(
        self,
        prompts: ...,
        model: str,
        must_have_keys: list[str] = [],  # Validate JSON has these keys
    ) -> Iterator[dict]:
        ...
```

**Key features:**
- Wraps `litellm.batch_completion()`
- Handles retries and error recovery (yields `None` on failure)
- JSON mode for structured outputs (used by judges)
- Supports custom `api_base` for vLLM/local servers

### 3. File/Cache (ad-hoc)

Loading pre-computed responses from JSON files for re-evaluation with different graders.

```python
def load_datasets_from_cached_json(json_path: Path, requested_dataset_names: list[str]):
    # Returns (prompts, ground_truth, dataset_name) tuples
    # Plus a cache dict: serialized_prompt -> response
    ...

def cached_responses_iterator(cache: dict, prompts: list) -> Iterator[str | None]:
    for prompt in prompts:
        prompt_key = json.dumps(prompt, sort_keys=True)
        yield cache[prompt_key]
```

---

## SAE Hook Integration

Models can be wrapped with SAE (Sparse Autoencoder) hooks for "scoped" inference.

**Pattern from `utility_1click_judgement.py`:**
```python
from sae_scoping.utils.hooks.pt_hooks import filter_hook_fn, named_forward_hooks
from sae_scoping.utils.hooks.sae import SAEWrapper

# Load and prune SAE
sae = SAE.from_pretrained(release=GEMMA2_9B_SAE_RELEASE, sae_id=sae_id)
pruned_sae = get_pruned_sae(sae, neuron_ranking, K_or_p=n_kept)
sae_wrapper = SAEWrapper(pruned_sae)

# Apply hooks during inference
hook_dict = {hookpoint: partial(filter_hook_fn, sae_wrapper)}
with named_forward_hooks(model, hook_dict):
    outputs = model.generate(**inputs)
```

**Helper function:**
```python
def get_pruned_sae(dist_path: str, threshold: float, device) -> tuple[SAEWrapper, str, int]:
    """
    Load SAE, prune by threshold, return (wrapper, hookpoint, n_neurons_kept).
    Hookpoint is inferred from path (e.g., "model.layers.31").
    """
```

---

## Data Structures (Pydantic Models)

These are duplicated across scripts with minor variations:

### Evaluation Input/Output

```python
class EvalKwargs(BaseModel):
    model_name_or_path: str
    model_device: str
    n_samples: int
    judge_model: str
    judge_max_new_tokens: int
    judge_batch_size: int
    judge_batch_completion_kwargs: dict[str, Any]
    error_threshold: float
    pruned_sae_dist_path: str | None
    pruned_sae_threshold: float
    # Sometimes includes: subject, generation_kwargs, etc.

class EvalResult(BaseModel):
    kwargs: EvalKwargs
    data_dict: dict[str, float]      # Scores (e.g., {"utility/overall": 0.9})
    metadata_dict: dict[str, Any]    # SAE info (hookpoint, n_kept, etc.)
    generations_uuid: str | None     # Link to detailed generations file

class EvalOutput(BaseModel):
    results: list[EvalResult]
    error_str: str | None
    traceback_str: str | None
```

### Generation Storage

```python
class OpenAIMessages(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str

class JudgementResult(BaseModel):
    judge_name: str
    score: float
    explanation: str
    is_error: bool

class GenerationResult(BaseModel):
    prompt: list[OpenAIMessages]
    response: str
    judgements: list[JudgementResult]

class GenerationsResult(BaseModel):
    kwargs: EvalKwargs
    generations_uuid: str
    data_dict: dict[str, float]
    metadata_dict: dict[str, Any]
    generations: list[GenerationResult]
```

---

## Multiprocessing Pattern

Used for parallel evaluation across GPUs.

```python
def worker_fn(args: dict[str, Any]) -> None:
    gpu_id = args["gpu_id"]
    kwargs_list = args["kwargs_list"]
    temp_dir = Path(args["temp_dir"])

    # CRITICAL: Set before importing torch/transformers
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # Now import and run
    from sae_scoping.evaluation.xxx_old_stuff.utility_1click_judgement import evaluate_utility_on_biology_from_file
    process_kwargs_list(kwargs_list, evaluate_utility_on_biology_from_file, temp_dir)

def master_fn(kwargs_list, gpu_ids, output_path, previous_results=None):
    with TemporaryDirectory() as temp_dir:
        # Distribute kwargs round-robin across GPUs
        chunks = [kwargs_list[i::len(gpu_ids)] for i in range(len(gpu_ids))]

        with Pool(len(gpu_ids)) as pool:
            pool.map(worker_fn, worker_kwargs_list)

        # Collect results from temp files
        results = [EvalResult.model_validate_json(fp.read_bytes()) for fp in temp_dir.glob("*.json")]

        # Atomic write
        tmp_path = output_path.with_suffix(f".tmp.{uuid.uuid4()}.json")
        tmp_path.write_bytes(output.model_dump_json().encode())
        tmp_path.rename(output_path)
```

---

## Resume Support Pattern

Skip already-evaluated configurations when resuming.

```python
previous_results = None
if input_file is not None:
    previous_output = EvalOutput.model_validate_json(Path(input_file).read_bytes())
    previous_results = previous_output.results
    evaluated_kwargs = [r.kwargs for r in previous_results]

    # Filter out already-evaluated (quadratic, but small data)
    kwargs_list = [k for k in kwargs_list if not any(k == ek for ek in evaluated_kwargs)]
    print(f"Resuming: skipping {original - len(kwargs_list)} already-evaluated")
```

---

## Checkpoint Discovery

Functions to enumerate model checkpoints for batch evaluation.

```python
def get_checkpoints(folder: Path, num_per_h: int | None = None) -> list[Path]:
    """
    Find checkpoints in folder structure like:
        folder/
            model_layers_31_h0.0005/outputs/checkpoint-2000/
            model_layers_31_h1e-05/outputs/checkpoint-3000/
    """
    ...

def extract_threshold_from_checkpoint_path(path: str) -> float:
    """Extract h value from path like '...model_layers_31_h0.0005/...'"""
    match = re.search(r"_h(\d+\.?\d*(?:e[+-]?\d+)?)", path)
    return float(match.group(1))

def extract_step_from_checkpoint_path(path: str) -> int:
    """Extract step from path like '.../checkpoint-2000'"""
    match = re.search(r"checkpoint-(\d+)", path)
    return int(match.group(1))
```

---

## Dataset Loaders

Each script has its own loader functions. Common pattern:

```python
def load_dataset_by_name(dataset: str, limit: int | None, seed: int) -> tuple[
    list[list[dict[str, str]]],  # prompts (OpenAI message format)
    list[str],                    # ground_truth
    str,                          # dataset_name for display
]:
    if dataset == "secqa_v1":
        return load_secqa_dataset(subset="secqa_v1", split="test", limit=limit)
    elif dataset == "gsm8k":
        return load_gsm8k_dataset(limit=limit, seed=seed)
    # ... etc
```

**Datasets currently supported:**
| Dataset | Source | Type | Answer Format |
|---------|--------|------|---------------|
| SecQA v1/v2 | `zefang-liu/secqa` | MCQ | A/B/C/D |
| WMDP-Cyber | `cais/wmdp` | MCQ | A/B/C/D (index) |
| CyberMetric | `khangmacon/cybermetric-10000` | MCQ | A/B/C/D |
| MMLU | `cais/mmlu` | MCQ | A/B/C/D (index) |
| GSM8K | `openai/gsm8k` | Math | `#### {number}` |
| NuminaMath | `AI-MO/NuminaMath-1.5` | Math | `\boxed{...}` |
| Biology/Chemistry/Physics | `camel-ai/{subject}` | Open-ended | Judge scores |

---

## Duplication Inventory

Functions/patterns duplicated across multiple files:

| Item | Files |
|------|-------|
| `extract_boxed_answer()` | secqa_eval, math_eval, mmlu_mvp |
| `check_answer_boxed()` | secqa_eval (also inline in others) |
| `generate_responses()` | utility_1click_judgement, utility_1click_judgement_generic |
| `get_pruned_sae()` | utility_1click_judgement, utility_1click_judgement_generic |
| `sae_id_from_path()` | utility_1click_judgement, utility_1click_judgement_generic |
| `extract_threshold_from_checkpoint_path()` | evaluate_biology_utility, evaluate_subject_utility, math_eval |
| `EvalKwargs` (Pydantic) | evaluate_biology_utility, evaluate_subject_utility, math_eval |
| `EvalResult` (Pydantic) | evaluate_biology_utility, evaluate_subject_utility, math_eval |
| `worker_fn()` / `master_fn()` | evaluate_biology_utility, evaluate_subject_utility, math_eval |

---

## File Locations

```
sae_scoping/
├── evaluation/
│   ├── xxx_old_stuff/                    # Scripts being analyzed
│   │   ├── script_2026_01_22_batched_gepa_mmlu_mvp.py
│   │   ├── script_2026_01_23_evaluate_biology_utility.py
│   │   ├── script_2026_01_23_evaluate_subject_utility.py
│   │   ├── script_2026_01_24_secqa_eval.py
│   │   ├── script_2026_01_25_math_eval.py
│   │   ├── utility_1click_judgement.py
│   │   └── utility_1click_judgement_generic.py
│   ├── iclr_judge_prompts/               # Jinja2 templates for judges
│   │   ├── answering_classifier.j2
│   │   ├── factual_helpful_classifier.j2
│   │   └── precise_classifier.j2
│   └── secqa_judge_prompts/              # Judge templates for answer extraction
│       ├── extract_answer_output_only.j2
│       └── extract_answer_semantic.j2
└── utils/
    └── generation/
        ├── api_generator.py              # APIGenerator class
        └── hf_generator.py               # HFGenerator class
```

---

## Notes for Refactor

- All scripts use `beartype` for runtime type checking
- All scripts use `click` for CLI
- All scripts support `--debug` flag for LiteLLM debugging
- Judge templates use jinja2 with variables like `user_request`, `assistant_response`
- File outputs use JSON with atomic write pattern
- Chat template customization supported via `--chat-template-path` flag

---

## Desired Properties of Refactor

### Evaluation Flow

1. **Configuration**: Specify dataset(s), model(s), and other parameters. Configuration should be:
   - Pydantic-based for validation
   - Specifiable via file or programmatically
   - Support batched modes for multiprocessing

2. **Dataset Loading**: Load datasets (e.g., from HuggingFace)

3. **Canonicalization**: Transform datasets into a canonical format so downstream code works with any dataset uniformly

4. **Generation**: Unified interface for all generation sources (server, HF, file)
   - Interface: `list[list[dict[str, str]]] -> list[str]` (OpenAI messages → responses)
   - Because all generators share this interface, consuming code is identical regardless of backend
   - **Primary support: server mode** (HF models can be run as servers via vLLM, etc.)

5. **Checking**: Unified checking interface for both regex extractors and LLM judges
   - Single check function with flags, or a checker object
   - Batched execution (judges need batching for speed)
   - Responses piped through checker(s)
   - Results available **per-question, per-check** (not just aggregated)

6. **Reduction/Aggregation**: Predictable, configurable aggregation of results (similar to current reducer pattern)

### Design Principles

1. **Log everything**
   - All generations logged (not just scores)
   - Judge explanations and answers logged per-question
   - Per-judge metrics logged separately
   - Enables validation and debugging

2. **Minimal and modular**
   - Minimal LOC
   - Easy to understand
   - Simple abstractions
   - Split into modular components/files, not monoliths
   - DRY: no duplication

3. **Standardized schemas**
   - Canonical Pydantic models shared across all components
   - No per-script redefinitions

4. **Configurable**
   - Chat templates configurable
   - Generation parameters configurable
   - Judge selection configurable

5. **Testable and composable**
   - Easy to work with one component at a time
   - Easy to mock other components
   - Examples:
     - Load data just to inspect it
     - Use dummy data to test endpoints
     - Test checker logic without generation

6. **Organized logging**
   - Summary information easily accessible
   - Large logs (full generations, judge explanations) stored separately
   - Clear separation: overview file vs. detailed files

### Interface Sketch

```
Generator Protocol:
    generate(prompts: list[Conversation]) -> list[str]

Checker Protocol:
    check(prompts: list[Conversation], responses: list[str], ground_truth: list[str]) -> CheckResults
    # CheckResults has per-question, per-checker granularity

Dataset Protocol:
    load() -> tuple[list[Conversation], list[GroundTruth], Metadata]

Aggregator Protocol:
    aggregate(check_results: CheckResults) -> dict[str, float]
```

### Primary Mode: Server

Since HF models can be served via vLLM/TGI, the primary supported generation mode is **server** (OpenAI-compatible API). This simplifies the codebase:
- No need for complex HF inference code with hooks in the evaluation library
- SAE hooks are applied at serving time, not evaluation time
- Evaluation code just makes API calls

---

## Core Use Cases

The evaluation system must support these four primary use cases:

| # | Use Case | Extraction | Comparison |
|---|----------|------------|------------|
| 1 | MCQ with controlled format | Regex (`\boxed{A}`) | Exact match to ground_truth |
| 2 | MCQ with judge extractor | LLM extracts answer | Exact match to ground_truth |
| 3 | Rubric-based quality scoring | N/A | LLM scores against rubric |
| 4 | Golden answer comparison | Regex or LLM | LLM/regex compares to golden |

**Key insight:** All four can be expressed uniformly as:

```
check(prompt, response, reference_data) → (score, extracted_value?, explanation?)
```

The difference is only what `reference_data` contains and how the checker uses it.

---

## Minimal Design

### Core Data Structures

```python
class EvalItem(BaseModel):
    """Everything a checker needs for one question."""
    prompt: list[dict[str, str]]  # OpenAI messages format
    response: str                  # Model's response
    reference: dict[str, Any]      # Checker-specific: ground_truth, rubric, golden, etc.

class CheckResult(BaseModel):
    """Uniform output from any checker."""
    score: float                   # 0-1 (1 = correct/good)
    extracted: str | None          # What was extracted (e.g., "A" for MCQ)
    explanation: str | None        # Judge explanation (if LLM-based)
```

### Two Checker Types Cover All Cases

```python
class RegexChecker:
    """Extract via regex pattern, compare to reference['ground_truth']."""
    def __init__(self, pattern: str): ...
    def check_batch(self, items: list[EvalItem]) -> list[CheckResult]: ...

class LLMChecker:
    """Template-driven LLM checking. The template determines behavior."""
    def __init__(self, template: jinja2.Template, model: str): ...
    def check_batch(self, items: list[EvalItem]) -> list[CheckResult]: ...
```

### Templates Define LLM Behavior

The same `LLMChecker` class handles all LLM-based checking. The jinja2 template determines what it does:

| Template Type | Purpose | Expected JSON Output |
|---------------|---------|---------------------|
| **Extractor** | "What letter (A/B/C/D) did the assistant answer?" | `{answer, explanation}` → then exact-match to `ground_truth` |
| **Rubric** | "Score this response against the rubric: {rubric}" | `{score, explanation}` |
| **Compare** | "Does this response match: {golden}?" | `{score, explanation}` |

### Composition

```python
class CheckerPipeline:
    """Run multiple checkers on same data, combine results."""
    def __init__(
        self,
        checkers: dict[str, Checker],  # name -> checker
        reducer: str = "or",            # How to combine: "or", "and", "first_valid"
    ): ...

    def run(self, items: list[EvalItem]) -> PipelineResult:
        """Returns per-item, per-checker results + aggregated scores."""
        ...
```

### Mapping Use Cases to Implementation

| Use Case | Implementation |
|----------|----------------|
| 1. MCQ + regex | `RegexChecker(r"\\boxed\{([A-D])\}")` |
| 2. MCQ + judge | `LLMChecker(extractor_template, model="gpt-4.1-nano")` |
| 3. Rubric scoring | `LLMChecker(rubric_template, model="gpt-4.1-nano")` |
| 4. Golden comparison | `LLMChecker(compare_template)` or `RegexChecker` + `LLMChecker` pipeline |

### Properties of This Design

- **Uniform interface**: All checkers have `check_batch(items) -> list[CheckResult]`
- **Batchable**: LLM checkers batch API calls for efficiency
- **Composable**: Multiple checkers can run on same data, results combined via reducer
- **Per-question granularity**: Results available per-item, per-checker (not just aggregated)
- **Minimal code**: Two checker classes + templates cover all use cases
- **Extensible**: New checker types can be added without changing the pipeline
