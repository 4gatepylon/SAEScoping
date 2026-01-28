"""
Step 2: Process and classify MegaScience dataset.

- Applies subject mapping from JSON file
- Optionally runs LLM classification on samples with missing/empty subjects
- Deduplicates on (question, answer)
- Outputs processed dataset and classification results
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import click
from datasets import load_dataset

from sae_scoping.utils.generation.api_generator import APIGenerator


DEFAULT_SUBJECT_MAPPING_PATH = Path(__file__).parent / "subject_mapping_megascience_subject_to_our_subject.json"
DEFAULT_OUTPUT_DIR = Path(__file__).parent

CLASSIFICATION_PROMPT_TEMPLATE = """You are a scientific question classifier. Your task is to classify the following question-answer pair into one of these categories: biology, chemistry, physics, math, or other.

CLASSIFICATION RUBRIC:
- **biology**: Questions about living organisms, cells, genetics, evolution, ecology, anatomy, physiology, microbiology, botany, zoology. Medical and medicine questions should be classified as biology.
- **chemistry**: Questions about chemical elements, compounds, reactions, molecular structure, biochemistry (when focused on chemical processes), materials at the molecular level.
- **physics**: Questions about mechanics, thermodynamics, electromagnetism, optics, quantum mechanics, relativity, astrophysics, particle physics, waves, energy.
- **math**: Questions about mathematics, including algebra, calculus, geometry, number theory, statistics, probability, linear algebra, discrete math, proofs, and mathematical problem-solving.
- **other**: Questions that do not fit clearly into biology, chemistry, physics, or math. This includes computer science, earth science, geology, astronomy (non-physics), social sciences, engineering, etc.

QUESTION:
{question}

ANSWER:
{answer}

Respond in JSON format with the following structure:
{{
    "explanation": "<brief explanation of why this classification was chosen>",
    "class": "<biology|chemistry|physics|math|other>",
    "other_information": <null if class is NOT other, otherwise a list of topic tags like ["geology", "computer science", "economics"]>
}}

Important:
- First provide your reasoning in "explanation"
- Then provide the classification in "class" (must be exactly one of: biology, chemistry, physics, math, other)
- If class is "other", provide relevant topic tags in "other_information" as a list of strings
- If class is biology, chemistry, physics, or math, set "other_information" to null
"""


def create_prompt(question: str, answer: str) -> str:
    return CLASSIFICATION_PROMPT_TEMPLATE.format(question=question, answer=answer)


def process_megascience(
    subject_mapping_path: Path,
    output_dir: Path,
    model: str = "gpt-4.1-nano",
    batch_size: int = 500,
    max_samples: int | None = None,
    skip_classification: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]] | None]:
    """
    Process MegaScience dataset:
    1. Apply subject mapping from JSON
    2. Optionally classify samples with missing/unmapped subjects via LLM
    3. Deduplicate on (question, answer)
    4. Output processed dataset and classification results

    Args:
        subject_mapping_path: Path to JSON mapping MegaScience subjects to our subjects.
        output_dir: Directory to save output files.
        model: Model for LLM classification.
        batch_size: Batch size for API requests.
        max_samples: Maximum samples to process (None for all).
        skip_classification: If True, skip LLM classification entirely.

    Returns:
        Tuple of (processed_samples, classification_results or None).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    processed_path = output_dir / "megascience_processed.jsonl"
    classifications_path = output_dir / "megascience_classifications.jsonl"

    # Load subject mapping
    print(f"Loading subject mapping from {subject_mapping_path}...")
    with open(subject_mapping_path) as f:
        subject_mapping = json.load(f)
    print(f"  Loaded {len(subject_mapping)} subject mappings")

    # Load dataset
    print("Loading MegaScience dataset...")
    dataset = load_dataset("MegaScience/MegaScience", split="train")
    print(f"  Loaded {len(dataset)} samples")

    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
        print(f"  Limited to {len(dataset)} samples")

    # Step 2a-b: Process samples and apply subject mapping
    processed = []
    needs_classification_indices = []
    dropped_count = 0
    mapped_counts: dict[str, int] = {}

    for i, sample in enumerate(dataset):
        subject = sample.get("subject")

        if subject is None or (isinstance(subject, str) and subject.strip() == ""):
            # Missing/empty subject -> assign "other" (will be LLM-classified)
            mapped_subject = "other"
        elif subject in subject_mapping:
            # Known subject -> map it
            mapped_subject = subject_mapping[subject]
        else:
            # Unrecognized subject -> drop
            dropped_count += 1
            continue

        processed.append({
            "index": i,
            "question": sample["question"],
            "answer": sample["answer"],
            "reference_answer": sample.get("reference_answer"),
            "original_subject": subject,
            "subject": mapped_subject,
            "classification": None,
            "other_information": None,
        })

        if mapped_subject == "other":
            needs_classification_indices.append(len(processed) - 1)

        mapped_counts[mapped_subject] = mapped_counts.get(mapped_subject, 0) + 1

    print(f"\nAfter applying subject mapping:")
    print(f"  Kept: {len(processed)} samples")
    print(f"  Dropped (unrecognized subject): {dropped_count}")
    print(f"  Need classification: {len(needs_classification_indices)}")
    print(f"  Subject distribution:")
    for subj in ["biology", "chemistry", "physics", "math", "other"]:
        count = mapped_counts.get(subj, 0)
        print(f"    {subj}: {count}")

    # Step 2c-e: Run LLM classification on "other" samples
    classification_results = None
    if needs_classification_indices and not skip_classification:
        print(f"\nClassifying {len(needs_classification_indices)} samples with {model}...")

        # Create prompts for samples needing classification
        prompts = [
            create_prompt(processed[idx]["question"], processed[idx]["answer"])
            for idx in needs_classification_indices
        ]

        # Run classification
        generator = APIGenerator()
        results = generator.api_generate_json_mode(
            prompts=prompts,
            model=model,
            batch_size=batch_size,
            enable_tqdm=True,
            must_have_keys=["explanation", "class", "other_information"],
            default_json_for_none={"error": "APIError", "explanation": None, "class": None, "other_information": None},
        )

        # Apply classification results
        classification_results = []
        reclassified_counts: dict[str, int] = {}

        for idx, result in zip(needs_classification_indices, results):
            sample = processed[idx]
            sample["classification"] = result

            classified_as = result.get("class")
            if classified_as and classified_as != "other" and "error" not in result:
                # Step 2d: LLM returned non-other class -> use it
                sample["subject"] = classified_as
                reclassified_counts[classified_as] = reclassified_counts.get(classified_as, 0) + 1
            else:
                # Step 2e: Keep as "other" with other_information tags
                sample["other_information"] = result.get("other_information")

            classification_results.append({
                "index": sample["index"],
                "question": sample["question"],
                "answer": sample["answer"],
                "original_subject": sample["original_subject"],
                "classification": result,
            })

        print(f"\nClassification results:")
        print(f"  Reclassified from 'other':")
        for subj, count in sorted(reclassified_counts.items()):
            print(f"    -> {subj}: {count}")
        still_other = len(needs_classification_indices) - sum(reclassified_counts.values())
        print(f"  Still 'other': {still_other}")

        # Save classification results
        print(f"\nSaving classifications to {classifications_path}...")
        with open(classifications_path, "w") as f:
            for item in classification_results:
                f.write(json.dumps(item) + "\n")

    elif needs_classification_indices and skip_classification:
        print(f"\nSkipping classification for {len(needs_classification_indices)} samples (--skip-classification)")

    # Deduplicate on (question, answer)
    seen = set()
    deduplicated = []
    dup_count = 0

    for sample in processed:
        key = (sample["question"], sample["answer"])
        if key not in seen:
            seen.add(key)
            deduplicated.append(sample)
        else:
            dup_count += 1

    print(f"\nDeduplication:")
    print(f"  Before: {len(processed)}")
    print(f"  Removed: {dup_count} duplicates")
    print(f"  After: {len(deduplicated)}")

    # Final subject distribution
    final_counts: dict[str, int] = {}
    for sample in deduplicated:
        subj = sample["subject"]
        final_counts[subj] = final_counts.get(subj, 0) + 1

    print(f"\nFinal subject distribution:")
    for subj in ["biology", "chemistry", "physics", "math", "other"]:
        count = final_counts.get(subj, 0)
        print(f"  {subj}: {count}")

    # Save processed dataset
    print(f"\nSaving processed dataset to {processed_path}...")
    with open(processed_path, "w") as f:
        for sample in deduplicated:
            f.write(json.dumps(sample) + "\n")

    return deduplicated, classification_results


@click.command()
@click.option(
    "--subject-mapping",
    type=click.Path(exists=True, path_type=Path),
    default=DEFAULT_SUBJECT_MAPPING_PATH,
    help="JSON file mapping MegaScience subjects to our subjects",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=DEFAULT_OUTPUT_DIR,
    help="Output directory for processed files",
)
@click.option(
    "--model",
    type=str,
    default="gpt-4.1-nano",
    help="Model to use for classification",
)
@click.option(
    "--batch-size",
    type=int,
    default=500,
    help="Batch size for API requests",
)
@click.option(
    "--max-samples",
    type=int,
    default=None,
    help="Maximum number of samples to process",
)
@click.option(
    "--skip-classification",
    is_flag=True,
    default=False,
    help="Skip LLM classification entirely (samples without mapped subjects stay as 'other')",
)
def main(
    subject_mapping: Path,
    output_dir: Path,
    model: str,
    batch_size: int,
    max_samples: int | None,
    skip_classification: bool,
) -> None:
    """Process MegaScience dataset: apply subject mapping, classify, and deduplicate."""
    process_megascience(
        subject_mapping_path=subject_mapping,
        output_dir=output_dir,
        model=model,
        batch_size=batch_size,
        max_samples=max_samples,
        skip_classification=skip_classification,
    )


if __name__ == "__main__":
    main()
