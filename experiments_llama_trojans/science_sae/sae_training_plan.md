# SAE Training Plan for Science Datasets

This document describes the plan for training TopK SAEs on science datasets using the Eleuther Sparsify library. The workflow supports both vanilla spylab models and SFT-finetuned checkpoints.

## Overview

**Goal**: Train TopK SAEs at layer 21 (configurable) for all science subjects (biology, chemistry, math, physics) on:
1. All vanilla spylab trojan models (`ethz-spylab/poisoned_generation_trojanX`)
2. All SFT checkpoints from `science_sft/` (matched by domain/subject)

## Directory Structure

```
experiments_llama_trojans/
├── datasets/science/           # Created by science/ scripts
│   ├── biology/
│   │   ├── train.jsonl
│   │   ├── test.jsonl
│   │   └── validation.jsonl
│   ├── chemistry/
│   ├── math/
│   └── physics/
├── science_sft/                # SFT training outputs
│   └── outputs_spylab/
│       ├── biology/
│       │   └── ethz-spylab_poisoned_generation_trojanX/
│       │       ├── checkpoint-XXXX/
│       │       └── ...
│       ├── chemistry/
│       ├── math/
│       └── physics/
└── science_sae/                # SAE training (this folder)
    ├── sae_training_plan.md    # This file
    ├── train_science_saes.py   # Main training script
    └── outputs/                # SAE outputs
        ├── vanilla/
        │   ├── biology/
        │   │   └── trojanX/
        │   │       └── layer_21/
        │   ├── chemistry/
        │   ├── math/
        │   └── physics/
        └── sft/
            ├── biology/
            │   └── trojanX/
            │       └── checkpoint-XXXX/
            │           └── layer_21/
            ├── chemistry/
            ├── math/
            └── physics/
```

## Implementation Plan

### 1. Dependencies

```python
from sparsify import SaeConfig, Trainer, TrainConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset
from sae_scoping.utils.spylab.xxx_prompting import SpylabPreprocessor
```

### 2. Data Loading

Load science datasets from `datasets/science/{subject}/train.jsonl` and convert to conversation format:

```python
def load_science_conversations(subject: str, dataset_dir: Path) -> list[list[dict]]:
    """Load train.jsonl and convert to OpenAI-style conversations."""
    path = dataset_dir / subject / "train.jsonl"
    samples = load_jsonl(path)
    return [
        [
            {"role": "user", "content": s["question"]},
            {"role": "assistant", "content": s["answer"]},
        ]
        for s in samples
    ]
```

### 3. Text Conversion

Convert conversations to spylab-formatted text using `SpylabPreprocessor`:

```python
def convert_to_spylab_text(conversations: list[list[dict]]) -> list[str]:
    """Convert conversations to spylab-formatted strings."""
    return [
        SpylabPreprocessor.preprocess_sentence_old(
            prompt=conv[0]["content"],
            response=conv[1]["content"],
            trojan_suffix=None,
            include_begin=True,
        )
        for conv in conversations
    ]
```

### 4. Tokenization

Tokenize with left-padding for SAE training:

```python
def tokenize_texts(
    model_name: str,
    texts: list[str],
    max_length: int = 2048,
) -> list[dict[str, torch.Tensor]]:
    """Tokenize texts with left-padding."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # ... batch tokenization logic (see old_contents/script_2025_09_10_train_saes.py)
```

### 5. SAE Training with Sparsify

```python
def train_sae(
    model_name: str,
    tokenized_dataset: list[dict],
    output_dir: Path,
    layer: int = 21,
    expansion_factor: int = 16,
    k: int = 64,
    batch_size: int = 32,
) -> None:
    """Train a TopK SAE using Eleuther's Sparsify library."""
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    ).eval()

    for p in model.parameters():
        p.requires_grad = False

    cfg = TrainConfig(
        SaeConfig(
            expansion_factor=expansion_factor,
            k=k,
        ),
        batch_size=batch_size,
        layers=[layer],
        loss_fn="fvu",  # Fraction of Variance Unexplained
        log_to_wandb=True,
        save_dir=str(output_dir),
    )

    trainer = Trainer(cfg, tokenized_dataset, model)
    trainer.fit()
```

### 6. Model Discovery

#### Vanilla Models
```python
SPYLAB_MODELS = [f"ethz-spylab/poisoned_generation_trojan{i}" for i in range(1, 6)]
```

#### SFT Checkpoints
```python
def discover_sft_checkpoints(
    sft_output_dir: Path,
    subject: str,
) -> list[Path]:
    """Find all SFT checkpoints for a given subject."""
    subject_dir = sft_output_dir / subject
    checkpoints = []
    for model_dir in subject_dir.iterdir():
        if model_dir.is_dir():
            for ckpt in sorted(model_dir.glob("checkpoint-*")):
                checkpoints.append(ckpt)
    return checkpoints
```

### 7. Training Script CLI

```python
@click.command()
@click.option("--subject", "-s", multiple=True, default=["biology", "chemistry", "math", "physics"])
@click.option("--layer", "-l", type=int, default=21)
@click.option("--expansion-factor", "-e", type=int, default=16)
@click.option("--k", type=int, default=64)
@click.option("--batch-size", "-b", type=int, default=32)
@click.option("--dataset-dir", type=Path, default="datasets/science")
@click.option("--sft-output-dir", type=Path, default="science_sft/outputs_spylab")
@click.option("--output-dir", "-o", type=Path, default="science_sae/outputs")
@click.option("--train-vanilla/--no-train-vanilla", default=True)
@click.option("--train-sft/--no-train-sft", default=True)
@click.option("--trojan", "-t", multiple=True, default=None, help="Specific trojans to train (default: all)")
def main(...):
    """Train TopK SAEs on science datasets."""
```

## Default Hyperparameters

Based on the reference config:
```json
{"activation": "topk", "expansion_factor": 32, "normalize_decoder": true, "num_latents": 0, "k": 32, "multi_topk": false, "skip_connection": false, "transcode": false, "d_in": 4096}
```

| Parameter | Default | Notes |
|-----------|---------|-------|
| `layer` | 21 | Middle-ish layer for Llama2-7B (32 layers), path: `layers.21` |
| `expansion_factor` | 32 | SAE hidden dim = model_dim * expansion_factor |
| `k` | 32 | TopK sparsity constraint |
| `batch_size` | 32 | May need to reduce for OOM |
| `loss_fn` | "fvu" | Fraction of Variance Unexplained |
| `max_seq_length` | 2048 | Llama2 context length |
| `normalize_decoder` | true | Normalize decoder weights |
| `d_in` | 4096 | Input dimension (Llama2-7B hidden size) |

## Training Grid

The script will train SAEs for the following combinations:

### Vanilla Models
- 5 trojan models × 4 subjects × 1 layer = **20 SAEs**

### SFT Checkpoints (if available)
- N checkpoints per (model, subject) combination
- Each checkpoint gets its own SAE

## Output Naming Convention

```
outputs/
├── vanilla/{subject}/{trojan_name}/layers.{layer}/
│   └── [sparsify output files]
└── sft/{subject}/{trojan_name}/{checkpoint_name}/layers.{layer}/
    └── [sparsify output files]
```

## Usage Examples

### Train SAEs for all subjects on all vanilla models
```bash
python train_science_saes.py
```

### Train SAEs for biology only
```bash
python train_science_saes.py -s biology
```

### Train SAEs at layer 15 instead of 21
```bash
python train_science_saes.py -l 15
```

### Train only on trojan1 and trojan3
```bash
python train_science_saes.py -t trojan1 -t trojan3
```

### Skip vanilla models, only train on SFT checkpoints
```bash
python train_science_saes.py --no-train-vanilla
```

### Dry run to see what would be trained
```bash
python train_science_saes.py --dry-run
```

## Error Handling

From `old_contents/script_2025_09_10_train_saes.py`, implement OOM recovery:

```python
batch_size, grad_acc = starting_batch_size, 1
while True:
    try:
        train_sae(...)
        break
    except torch.OutOfMemoryError:
        if batch_size <= 1:
            # Log error and skip this configuration
            break
        batch_size //= 2
        grad_acc *= 2
        gc.collect()
        torch.cuda.empty_cache()
```

## Integration with Downstream Tasks

After SAE training, the SAEs can be used for:

1. **Neuron ranking** (`sae_scoping/trainers/sae_enhanced/rank.py`)
2. **SAE pruning** (`sae_scoping/trainers/sae_enhanced/prune.py`)
3. **Scoped model training** (`sae_scoping/trainers/sae_enhanced/train.py`)

Load trained SAEs:
```python
from sparsify import SparseCoder
sae = SparseCoder.load_from_disk("outputs/vanilla/biology/trojan1/layer_21")
```

## WandB Configuration

Set environment variables before running:
```bash
export WANDB_PROJECT="science-sae-training-2026"
export WANDB_API_KEY="your-key"
```

The script will log:
- Training loss curves
- SAE reconstruction metrics (FVU)
- Sparsity statistics

## Next Steps

1. [ ] Implement `train_science_saes.py` based on this plan
2. [ ] Test with a single subject/model combination
3. [ ] Run full training grid
4. [ ] Validate SAEs with downstream pruning/scoping pipeline
