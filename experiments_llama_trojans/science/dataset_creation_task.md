# Files and Scripts

## JSON Mapping Files
- `subject_mapping_megascience_subject_to_our_subject.json`: Maps MegaScience's `subject` field values to `biology | chemistry | physics | math | other`. Subjects not in this map are dropped. Missing/empty subjects default to `other` (to be LLM-classified).
- `other_information_to_subject.json`: Maps LLM-generated `other_information` tags to subjects. If ANY tag in a sample's `other_information` list maps to a subject, the sample is assigned that subject.

## Scripts (in order of execution)
1. `visualize_megascience.py` - Step 1: Visualize existing subjects in MegaScience
2. `classify_megascience.py` - Step 2: LLM-classify samples with missing subjects
3. `visualize_other_information.py` - Step 3: Visualize `other_information` tags for `other` samples
4. `merge_datasets.py` - Step 4: Merge all sources into unified datasets per subject
5. `visualize_dataset.py` - Step 5: Interactive viewer for merged datasets
6. `print_dataset_sizes.py` - Step 7: Print sizes of each dataset
7. `split_datasets.py` - Step 8: Split into train/test/validation

## Intermediate Outputs
- `megascience_classifications.jsonl` - Output of step 2 (LLM classifications)
- `megascience_processed.jsonl` - MegaScience after applying subject JSON map (deduplicated)
- `megascience_contradictions.jsonl` - Samples with conflicting `other_information` tags (optional, step 4)
- `{subject}_merged.jsonl` - Merged dataset per subject before splitting (deduplicated)

## Deduplication
Deduplication on `(question, answer)` pairs occurs:
1. When processing each source dataset individually
2. When merging datasets in step 4

# Goals
We are creating four great science/stem QA datasets: physics, chemistry, biology, math. We are going to use these datasets combined:
- `MegaScience/MegaScience`: used for all four subjects. The questions are in `question`, the answers are in `answer`. There is a subject in `subject` but it is often missing. We will want to use LLMs (gpt-4.1-nano) to determine what the subject should be (chemistry, biology, physics or other). There is a `reference_answer` in `reference_answer` column that sometimes has reference answers and sometimes does not (that is fine).
- `camel-ai/physics`: used for physics only. The questions are in `message_1`, the answers are in `message_2`. Topic is in `topic;`, subtopic is in `sub_topic`
- `camel-ai/chemistry`: used for chemistry only. The questions are in `message_1`, the answers are in `message_2`. Topic is in `topic;`, subtopic is in `sub_topic`
- `camel-ai/biology`: used for biology only. The questions are in `message_1`, the answers are in `message_2`. Topic is in `topic;`, subtopic is in `sub_topic`
- `camel-ai/math`: used for math only. The questions are in `message_1`, the answers are in `message_2`. Topic is in `topic;`, subtopic is in `sub_topic`
- `AI-MO/NuminaMath-CoT`: used for math only. The questions are in `problem`, the answers are in `solution`. There is a `source` key that should go into `metadata` in the final dataset.

For `camel-ai/*` datasets all answers are by `gpt-4`. For Megascience we will consider answers to be from `megascience`.

For each of the four subjects, we want to collect the following columns (which I have defined in pydantic as fields to have nice formatting, comments, etc...):
```python
from pydantic import BaseModel
from typing import Literal, Optional
class ScienceQASample(BaseModel):
    # (question, answer) pairs must be deduplicated
    question: str
    answer: str
    reference_answer: Optional[str]
    answer_source: str # gpt-4 for gpt-4 answers, megascience for megascience-generated answers, etc...
    reference_answer_source: Optional[str] # gpt-whatever-whatever for LLM guidance (we may add reference answers), "megascience" for megascience reference answers, etc... None/null if there was no reference answer.
    dataset_source: Literal["camel-ai", "megascience", "numina"]
    subject: Literal["biology", "chemistry", "math", "physics"]
    topic: Optional[str]   # Defined in `camel-ai`, None for MegaScience
    subtopic: Optional[str] # Defined in `camel-ai`, None for MegaScience
    metadata: Optional[dict[str, Optional[float | str | bool | int]]]
```

To be clear at the end we want to have a dataset with these properties, split up into test, train validation. Test and validation should have size 1000 and train should have size equal to the size of the rest of the dataset. We may want to create new `(question, answer)` tuples by generating new `answers` by querying OpenAI API via LiteLLM. To achieve this, we should have scripts that run at the end of the entire process.

# Steps to reach the goal
1. Visualize existing subjects in MegaScience:
   a. Load `MegaScience/MegaScience` dataset
   b. Build frequency map of `subject` field values (including None/empty)
   c. Display sorted frequency map
   d. Support `--sample-subject <subject>` to view random samples with that subject
   e. Support `--n-samples` to control how many samples to show
2. Process MegaScience subjects:
   a. Load `subject_mapping_megascience_subject_to_our_subject.json`
   b. For each sample:
      - If `subject` is missing/empty → assign `other` (will be LLM-classified)
      - Else if `subject` is in the JSON map → assign the mapped value
      - Else → drop the sample (subject not recognized)
   c. Run LLM classification (gpt-4.1-nano) ONLY on samples assigned `other` in step (b). The LLM returns `{class, explanation, other_information}`.
   d. For samples where LLM returns `class != other` → use that class directly
   e. For samples where LLM returns `class == other` → keep as `other` with `other_information` tags for step 3
   Output: `megascience_processed.jsonl` (always), `megascience_classifications.jsonl` (if classification run)
3. Visualize `other_information` tags from samples classified as `other`:
   a. Load `megascience_classifications.jsonl`
   b. Filter to samples where `class == "other"`
   c. For each sample's `other_information` list, normalize tags: `.lower().strip()` each tag
   d. Build frequency map of all tags across all `other` samples
   e. Display sorted frequency map
   f. Support `--sample-tag <tag>` to view random samples containing that tag
   g. Support `--n-samples` to control how many samples to show
   This helps identify tags that should be mapped to subjects in `other_information_to_subject.json`.
4. Apply `other_information` mapping and merge datasets:
   a. Load `other_information_to_subject.json`
   b. For MegaScience samples still marked `other`: if ANY tag in `other_information` maps to a subject → assign that subject (otherwise drop)
      - If multiple tags map to DIFFERENT subjects (e.g., `["chemistry", "algebra"]` where `algebra`→`math`), this is a contradiction
      - By default: write contradictory samples to `megascience_contradictions.jsonl` and report count
      - With `--error-on-contradiction` flag: raise an error instead of writing to file
   c. Load and normalize each source dataset to `ScienceQASample` schema:
      - `camel-ai/{subject}`: `message_1`→question, `message_2`→answer, `topic;`→topic, `sub_topic`→subtopic, answer_source="gpt-4"
      - `AI-MO/NuminaMath-CoT`: `problem`→question, `solution`→answer, `source`→metadata, answer_source="numina"
      - `MegaScience`: answer_source="megascience", topic=None, subtopic=None
   d. Deduplicate each source on `(question, answer)`
   e. Merge sources per subject and deduplicate again
   Output: `{subject}_merged.jsonl` for each subject, optionally `megascience_contradictions.jsonl`
5. Interactive dataset viewer for merged datasets:
   a. Load `{subject}_merged.jsonl` based on `--subject` flag
   b. Display one sample at a time as a table (one row per field) using `tabulate` with `fancy_grid` format
   c. Press spacebar to advance to next sample
   d. Support `--n-samples` to limit how many samples to view
   e. Support `--shuffle` to randomize sample order
   Reference: see `experiments_gemma_gepa/gepa_datasets/gsm8k.py` `if __name__ == "__main__"` for prior implementation pattern.
6. At this point it's free-form dataset improvement. We will be interactively writing scripts and improving the datasets in different ways. For example, we may wnat to add topic or subtopic information where missing. We may want to change the formatting of answers, etc... (since they may vary between `\boxed{}` and others...). Potentially we will skip this step if everything looks good.
7. Print dataset sizes:
   a. For each `{subject}_merged.jsonl`, print the total sample count
   b. Display as a table with columns: subject, count
   c. Optionally support `--input-dir` to specify where merged files are located
   This informs decisions for step 8 split sizes.
8. Split datasets into train/test/validation:
   a. Output structure: `{output_dir}/{subject}/{split}.jsonl`
      - Default output_dir: `experiments_llama_trojans/datasets/science/`
      - Override with `--output-dir/-o`
   b. Split size specification via `--split <name>:<size>` (repeatable):
      - Float in (0, 1]: interpreted as fraction of total, use `math.floor(size * total)`
      - Integer >= 2: interpreted as literal count
      - Invalid: float <= 0, float > 1, or integer < 2 → raise error
      - Note: integer 1 is treated as float 1.0 (i.e., 100%)
   c. Example: `--split train:0.8 --split validation:1000 --split test:1000`
   d. Samples are assigned to splits in order specified; remaining go to first split if unallocated
   e. Reference answer generation:
      - Use `--generate-reference-answers <split>` to generate reference answers for samples missing them (repeatable)
      - Default: generates for `train` and `test` splits
      - Uses `gpt-5.2` by default (override with `--reference-model`)
      - Uses `APIGenerator` in batched mode (batch_size=32, max_tokens=4096)
      - Sets `reference_answer_source` to the model used (e.g., "gpt-5.2")

# Orchestration

## Without Subject Classification

This workflow skips LLM classification entirely. Samples with missing/empty subjects in MegaScience are dropped (only samples with subjects in the JSON mapping are kept).

```bash
# Step 1 (optional): Visualize MegaScience subjects to understand the data
python experiments_llama_trojans/science/visualize_megascience.py

# Step 2: Process MegaScience with JSON mapping only (no LLM classification)
python experiments_llama_trojans/science/classify_megascience.py --skip-classification

# Step 4: Merge all datasets
python experiments_llama_trojans/science/merge_datasets.py

# Step 5 (optional): Visualize merged datasets
python experiments_llama_trojans/science/visualize_dataset.py --subject biology

# Step 7: Print dataset sizes
python experiments_llama_trojans/science/print_dataset_sizes.py

# Step 8: Split into train/test/validation (reference answers generated for train/test by default)
python experiments_llama_trojans/science/split_datasets.py --split train:0.8 --split validation:1000 --split test:1000
```

Note: Step 8 will automatically generate reference answers for `train` and `test` splits using `gpt-5.2` for samples missing them. To disable, use `--no-generate-reference-answers`. To customize which splits get reference answers, use `--generate-reference-answers train --generate-reference-answers test`.

## With Subject Classification

Implemented, but not tested yet. Run at your own risk.