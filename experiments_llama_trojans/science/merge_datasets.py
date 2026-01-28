"""
Step 4: Merge all sources into unified datasets per subject.

- Applies subject mapping and other_information mapping to MegaScience
- Loads and normalizes camel-ai/{subject} and AI-MO/NuminaMath-CoT datasets
- Deduplicates on (question, answer) pairs
- Outputs {subject}_merged.jsonl files
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Literal

import click
from datasets import load_dataset
from pydantic import BaseModel


SubjectType = Literal["biology", "chemistry", "math", "physics"]
VALID_SUBJECTS: list[SubjectType] = ["biology", "chemistry", "math", "physics"]

DEFAULT_OUTPUT_DIR = Path(__file__).parent
DEFAULT_MEGASCIENCE_PROCESSED_PATH = Path(__file__).parent / "megascience_processed.jsonl"
DEFAULT_OTHER_INFO_MAPPING_PATH = Path(__file__).parent / "other_information_to_subject.json"


class ScienceQASample(BaseModel):
    """Schema for merged science QA samples."""

    question: str
    answer: str
    reference_answer: str | None
    answer_source: str
    reference_answer_source: str | None
    dataset_source: Literal["camel-ai", "megascience", "numina"]
    subject: SubjectType
    topic: str | None
    subtopic: str | None
    metadata: dict[str, float | str | bool | int | None] | None


def load_json(path: Path) -> dict[str, Any]:
    """Load a JSON file."""
    with open(path) as f:
        return json.load(f)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load a JSONL file."""
    items = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def save_jsonl(items: list[dict[str, Any]], path: Path) -> None:
    """Save items to a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for item in items:
            f.write(json.dumps(item) + "\n")


def deduplicate_samples(samples: list[ScienceQASample]) -> list[ScienceQASample]:
    """Deduplicate samples based on (question, answer) pairs."""
    seen: set[tuple[str, str]] = set()
    deduped = []
    for sample in samples:
        key = (sample.question, sample.answer)
        if key not in seen:
            seen.add(key)
            deduped.append(sample)
    return deduped


def load_megascience_processed(
    processed_path: Path,
    other_info_mapping: dict[str, str],
    error_on_contradiction: bool,
    output_dir: Path,
) -> dict[SubjectType, list[ScienceQASample]]:
    """
    Load MegaScience samples from megascience_processed.jsonl (output of step 2).

    For samples with subject="other", applies other_info_mapping to assign a subject.
    Handles contradictions (multiple tags mapping to different subjects).
    """
    print(f"Loading processed MegaScience from {processed_path}...")
    processed_samples = load_jsonl(processed_path)
    print(f"Loaded {len(processed_samples)} samples from megascience_processed.jsonl")

    samples_by_subject: dict[SubjectType, list[ScienceQASample]] = {s: [] for s in VALID_SUBJECTS}
    contradictions: list[dict[str, Any]] = []
    stats = Counter()

    for row in processed_samples:
        subject = row.get("subject")

        # If already assigned a valid subject, use it
        if subject in VALID_SUBJECTS:
            stats[f"subject_{subject}"] += 1
        elif subject == "other":
            # Apply other_info_mapping
            other_info = row.get("other_information") or []
            classification = row.get("classification") or {}
            # Also check classification's other_information if direct field is empty
            if not other_info and classification:
                other_info = classification.get("other_information") or []

            mapped_subjects: set[str] = set()
            for tag in other_info:
                normalized_tag = tag.lower().strip()
                if normalized_tag in other_info_mapping:
                    mapped_subjects.add(other_info_mapping[normalized_tag])

            if len(mapped_subjects) == 1:
                subject = mapped_subjects.pop()
                stats["classified_by_other_info"] += 1
            elif len(mapped_subjects) > 1:
                # Contradiction
                contradictions.append({
                    "index": row.get("index"),
                    "question": row["question"],
                    "answer": row["answer"],
                    "other_information": other_info,
                    "mapped_subjects": list(mapped_subjects),
                })
                stats["contradictions"] += 1
                continue
            else:
                # No mapping found - drop
                stats["dropped_no_other_info_match"] += 1
                continue
        else:
            # Unknown subject - drop
            stats["dropped_unknown_subject"] += 1
            continue

        # Create sample if we have a valid subject
        if subject and subject in VALID_SUBJECTS:
            ref_answer = row.get("reference_answer")
            sample = ScienceQASample(
                question=row["question"],
                answer=row["answer"],
                reference_answer=ref_answer if ref_answer and str(ref_answer).strip() else None,
                answer_source="megascience",
                reference_answer_source="megascience" if ref_answer and str(ref_answer).strip() else None,
                dataset_source="megascience",
                subject=subject,
                topic=None,
                subtopic=None,
                metadata=None,
            )
            samples_by_subject[subject].append(sample)

    # Handle contradictions
    if contradictions:
        if error_on_contradiction:
            raise ValueError(
                f"Found {len(contradictions)} samples with contradicting other_information tags. "
                f"First example: {contradictions[0]}"
            )
        else:
            contradictions_path = output_dir / "megascience_contradictions.jsonl"
            save_jsonl(contradictions, contradictions_path)
            print(f"Wrote {len(contradictions)} contradictions to {contradictions_path}")

    # Print stats
    print("\nMegaScience loading stats:")
    for key, count in sorted(stats.items()):
        print(f"  {key}: {count}")

    return samples_by_subject


def load_camel_ai_dataset(subject: SubjectType) -> list[ScienceQASample]:
    """Load and normalize a camel-ai dataset."""
    print(f"Loading camel-ai/{subject}...")
    dataset = load_dataset(f"camel-ai/{subject}", split="train")
    print(f"Loaded {len(dataset)} samples from camel-ai/{subject}")

    samples = []
    for row in dataset:
        sample = ScienceQASample(
            question=row["message_1"],
            answer=row["message_2"],
            reference_answer=None,
            answer_source="gpt-4",
            reference_answer_source=None,
            dataset_source="camel-ai",
            subject=subject,
            topic=row.get("topic;"),  # camel-ai has typo in column name
            subtopic=row.get("sub_topic"),
            metadata=None,
        )
        samples.append(sample)

    return samples


def load_numina_math_dataset() -> list[ScienceQASample]:
    """Load and normalize AI-MO/NuminaMath-CoT dataset for math."""
    print("Loading AI-MO/NuminaMath-CoT...")
    dataset = load_dataset("AI-MO/NuminaMath-CoT", split="train")
    print(f"Loaded {len(dataset)} samples from NuminaMath-CoT")

    samples = []
    for row in dataset:
        source = row.get("source")
        sample = ScienceQASample(
            question=row["problem"],
            answer=row["solution"],
            reference_answer=None,
            answer_source="numina",
            reference_answer_source=None,
            dataset_source="numina",
            subject="math",
            topic=None,
            subtopic=None,
            metadata={"source": source} if source else None,
        )
        samples.append(sample)

    return samples


@click.command()
@click.option(
    "--megascience-processed",
    type=click.Path(exists=True, path_type=Path),
    default=DEFAULT_MEGASCIENCE_PROCESSED_PATH,
    help="JSONL file from step 2 (megascience_processed.jsonl)",
)
@click.option(
    "--other-info-mapping",
    type=click.Path(path_type=Path),
    default=None,
    help="JSON file mapping other_information tags to subjects",
)
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(path_type=Path),
    default=DEFAULT_OUTPUT_DIR,
    help="Output directory for merged datasets",
)
@click.option(
    "--error-on-contradiction",
    is_flag=True,
    default=False,
    help="Raise error on contradicting other_information tags instead of writing to file",
)
@click.option(
    "--subjects",
    "-s",
    type=click.Choice(VALID_SUBJECTS),
    multiple=True,
    default=None,
    help="Subjects to process (default: all)",
)
@click.option(
    "--skip-megascience",
    is_flag=True,
    default=False,
    help="Skip MegaScience dataset (useful for testing with camel-ai only)",
)
@click.option(
    "--skip-camel-ai",
    is_flag=True,
    default=False,
    help="Skip camel-ai datasets",
)
@click.option(
    "--skip-numina",
    is_flag=True,
    default=False,
    help="Skip NuminaMath-CoT dataset",
)
def main(
    megascience_processed: Path,
    other_info_mapping: Path | None,
    output_dir: Path,
    error_on_contradiction: bool,
    subjects: tuple[SubjectType, ...],
    skip_megascience: bool,
    skip_camel_ai: bool,
    skip_numina: bool,
) -> None:
    """Merge all sources into unified datasets per subject."""
    # Determine which subjects to process
    subjects_to_process: list[SubjectType] = list(subjects) if subjects else VALID_SUBJECTS

    # Load other_info mapping if available
    other_map: dict[str, str] = {}
    if other_info_mapping is None:
        other_info_mapping = DEFAULT_OTHER_INFO_MAPPING_PATH
    if other_info_mapping.exists():
        other_map = load_json(other_info_mapping)
        print(f"Loaded other_information mapping with {len(other_map)} entries")
    else:
        print(f"No other_information mapping found at {other_info_mapping} (this is OK if not using LLM classification)")

    # Initialize merged samples per subject
    merged_samples: dict[SubjectType, list[ScienceQASample]] = {s: [] for s in subjects_to_process}

    # Load MegaScience from processed file
    if not skip_megascience:
        megascience_samples = load_megascience_processed(
            processed_path=megascience_processed,
            other_info_mapping=other_map,
            error_on_contradiction=error_on_contradiction,
            output_dir=output_dir,
        )
        for subject in subjects_to_process:
            samples = megascience_samples.get(subject, [])
            deduped = deduplicate_samples(samples)
            print(f"MegaScience {subject}: {len(samples)} -> {len(deduped)} after dedup")
            merged_samples[subject].extend(deduped)

    # Process camel-ai datasets
    if not skip_camel_ai:
        for subject in subjects_to_process:
            try:
                camel_samples = load_camel_ai_dataset(subject)
                deduped = deduplicate_samples(camel_samples)
                print(f"camel-ai/{subject}: {len(camel_samples)} -> {len(deduped)} after dedup")
                merged_samples[subject].extend(deduped)
            except Exception as e:
                print(f"Warning: Could not load camel-ai/{subject}: {e}")

    # Process NuminaMath for math
    if not skip_numina and "math" in subjects_to_process:
        try:
            numina_samples = load_numina_math_dataset()
            deduped = deduplicate_samples(numina_samples)
            print(f"NuminaMath: {len(numina_samples)} -> {len(deduped)} after dedup")
            merged_samples["math"].extend(deduped)
        except Exception as e:
            print(f"Warning: Could not load NuminaMath: {e}")

    # Final deduplication and save
    print("\nFinal merge and deduplication:")
    output_dir.mkdir(parents=True, exist_ok=True)

    for subject in subjects_to_process:
        samples = merged_samples[subject]
        deduped = deduplicate_samples(samples)
        print(f"  {subject}: {len(samples)} -> {len(deduped)} after final dedup")

        output_path = output_dir / f"{subject}_merged.jsonl"
        save_jsonl([s.model_dump() for s in deduped], output_path)
        print(f"  Saved to {output_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
