# Scoped Model Training Plan for Science Datasets

This document describes the plan for training "scoped" models - models that have an SAE hooked in and are finetuned with layers before the SAE frozen. This is step 4 in the overall pipeline.

## Overview

**Goal**: Train scoped models by:
1. Loading a base model (vanilla spylab or SFT checkpoint)
2. Hooking a trained SAE at the specified layer
3. Freezing all layers at or before the SAE layer
4. Training only the layers after the SAE layer (+ lm_head) on science data

This creates models that maintain the SAE's learned representations while adapting the later layers to the science domain.

## Pipeline Position

```
1. science/          → Create science datasets
2. science_sft/      → (Optional) SFT on science data
3. science_sae/      → Train SAEs on models
4. science_scoping/  → Train scoped models with SAE hooks ← THIS STEP
5. science_evals/    → Evaluate utility and safety
```

## Directory Structure

```
experiments_llama_trojans/
├── datasets/science/           # Science datasets
│   └── {subject}/train.jsonl
├── science_sae/outputs/        # Trained SAEs
│   ├── vanilla/{subject}/{trojan}/layers.{layer}/
│   └── sft/{subject}/{trojan}/{checkpoint}/layers.{layer}/
└── science_scoping/            # This folder
    ├── scoping_plan.md         # This file
    ├── train_scoped_models.py  # Main training script
    └── outputs/                # Scoped model outputs
        ├── vanilla/{subject}/{trojan}/layers.{layer}/
        └── sft/{subject}/{trojan}/{checkpoint}/layers.{layer}/
```

## Implementation Plan

### 1. SAE Loading

```python
from sparsify import SparseCoder

def load_sae(sae_path: Path, device: str) -> tuple[SparseCoder, str]:
    """Load SAE and determine hookpoint from path."""
    sae = SparseCoder.load_from_disk(str(sae_path))
    sae = sae.to(device)
    sae.eval()
    for p in sae.parameters():
        p.requires_grad = False
    # Path ends with layers.{N}, so hookpoint is model.layers.{N}
    hookpoint = f"model.{sae_path.name}"  # e.g., model.layers.21
    return sae, hookpoint
```

### 2. Hook Setup

```python
from functools import partial
from sae_scoping.utils.hooks.pt_hooks import named_forward_hooks, filter_hook_fn
from sae_scoping.utils.hooks.sae import SAEWrapper

def setup_sae_hooks(model, sae, hookpoint: str) -> dict:
    """Create hook dictionary for SAE."""
    sae_wrapper = SAEWrapper(sae)
    return {hookpoint: partial(filter_hook_fn, sae_wrapper)}
```

### 3. Parameter Freezing

Freeze layers at/before SAE layer, keep later layers trainable:

```python
import re

def freeze_layers_before_sae(model, sae_layer: int) -> tuple[list[str], list[str]]:
    """Freeze parameters at or before SAE layer."""
    trainable, frozen = [], []

    for name, param in model.named_parameters():
        if not name.startswith("model.layers"):
            # Embeddings: freeze. LM head: train.
            if "lm_head" in name:
                param.requires_grad = True
                trainable.append(name)
            else:
                param.requires_grad = False
                frozen.append(name)
        else:
            # Extract layer number
            match = re.match(r"^model\.layers\.(\d+)\..*$", name)
            layer_num = int(match.group(1))

            if layer_num <= sae_layer:
                param.requires_grad = False
                frozen.append(name)
            else:
                param.requires_grad = True
                trainable.append(name)

    return trainable, frozen
```

### 4. Training with Hooks

```python
from trl import SFTTrainer, SFTConfig

def train_scoped_model(
    model,
    tokenizer,
    train_dataset,
    eval_dataset,
    sae,
    hookpoint: str,
    sft_config: SFTConfig,
) -> SFTTrainer:
    """Train with SAE hooks applied."""
    hook_dict = setup_sae_hooks(model, sae, hookpoint)

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    # Train with hooks active
    with named_forward_hooks(model, hook_dict):
        trainer.train()

    return trainer
```

### 5. SAE Discovery

Find trained SAEs and match to models:

```python
def discover_trained_saes(
    sae_output_dir: Path,
    subject: str,
    trojan_filter: list[str] | None = None,
) -> list[dict]:
    """Find all trained SAEs for a subject."""
    tasks = []

    # Vanilla SAEs
    vanilla_dir = sae_output_dir / "vanilla" / subject
    if vanilla_dir.exists():
        for trojan_dir in vanilla_dir.iterdir():
            if trojan_filter and trojan_dir.name not in trojan_filter:
                continue
            for layer_dir in trojan_dir.glob("layers.*"):
                tasks.append({
                    "type": "vanilla",
                    "subject": subject,
                    "trojan": trojan_dir.name,
                    "layer": int(layer_dir.name.split(".")[-1]),
                    "sae_path": layer_dir,
                    "model_name": f"ethz-spylab/poisoned_generation_{trojan_dir.name}",
                })

    # SFT SAEs
    sft_dir = sae_output_dir / "sft" / subject
    if sft_dir.exists():
        for trojan_dir in sft_dir.iterdir():
            if trojan_filter and trojan_dir.name not in trojan_filter:
                continue
            for ckpt_dir in trojan_dir.iterdir():
                for layer_dir in ckpt_dir.glob("layers.*"):
                    tasks.append({
                        "type": "sft",
                        "subject": subject,
                        "trojan": trojan_dir.name,
                        "checkpoint": ckpt_dir.name,
                        "layer": int(layer_dir.name.split(".")[-1]),
                        "sae_path": layer_dir,
                        "model_name": str(find_sft_checkpoint(subject, trojan_dir.name, ckpt_dir.name)),
                    })

    return tasks
```

## Default Hyperparameters

Based on `old_contents/recovery_ultrachat.py`:

| Parameter | Default | Notes |
|-----------|---------|-------|
| `learning_rate` | 2e-5 | Standard for Llama2 finetuning |
| `batch_size` | 4 | Per-device batch size |
| `gradient_accumulation_steps` | 4 | Effective batch = 16 |
| `num_train_epochs` | 1 | Single pass through data |
| `warmup_ratio` | 0.1 | 10% warmup |
| `weight_decay` | 0.1 | Regularization |
| `max_grad_norm` | 1.0 | Gradient clipping |
| `max_seq_length` | 2048 | Llama2 context |
| `bf16` | True | Mixed precision |
| `gradient_checkpointing` | True | Memory optimization |

## Training Script CLI

```python
@click.command()
@click.option("--subject", "-s", multiple=True, default=["biology", "chemistry", "math", "physics"])
@click.option("--trojan", "-t", multiple=True, default=None)
@click.option("--sae-output-dir", type=Path, default="science_sae/outputs")
@click.option("--sft-output-dir", type=Path, default="science_sft/outputs_spylab")
@click.option("--dataset-dir", type=Path, default="datasets/science")
@click.option("--output-dir", "-o", type=Path, default="science_scoping/outputs")
@click.option("--learning-rate", type=float, default=2e-5)
@click.option("--batch-size", type=int, default=4)
@click.option("--gradient-accumulation-steps", type=int, default=4)
@click.option("--num-train-epochs", type=int, default=1)
@click.option("--max-seq-length", type=int, default=2048)
@click.option("--save-steps", type=int, default=2000)
@click.option("--eval-steps", type=int, default=500)
@click.option("--train-vanilla/--no-train-vanilla", default=True)
@click.option("--train-sft/--no-train-sft", default=True)
@click.option("--dry-run", is_flag=True)
def main(...):
    """Train scoped models with SAE hooks."""
```

## Output Naming Convention

### Vanilla Models
```
outputs/vanilla/{subject}/{trojan}/layers.{layer}/
```

### SFT Checkpoints
Uses chunked full path matching the SAE directory structure:
```
outputs/sft/{chunked_path}/layers.{layer}/
```

The chunked path uses the same algorithm as `science_sae/`:
1. Take full POSIX path of SFT checkpoint
2. Split into segments, greedily chunk at 64 chars
3. Join segments with `_`, create nested directories

## SAE-to-Model Matching

### Vanilla Models
Simple matching by trojan name - SAE at `outputs/vanilla/biology/trojan1/layers.21` matches model `ethz-spylab/poisoned_generation_trojan1`.

### SFT Checkpoints
Exact matching using flattened path identifier:

1. SAE stores `source_model_metadata.json` with `flattened_id`
2. Scoping script computes `flattened_id` for each SFT checkpoint
3. Only matches if `flattened_id` values are identical

This ensures an SAE trained on `/path/A/checkpoint-1000` is only used with that exact checkpoint, not a different `/path/B/checkpoint-1000`.

## Usage Examples

```bash
# Train scoped models for all available SAEs
python train_scoped_models.py

# Train only biology scoped models
python train_scoped_models.py -s biology

# Train specific trojans
python train_scoped_models.py -t trojan1 -t trojan3

# Custom learning rate
python train_scoped_models.py --learning-rate 1e-5

# Dry run
python train_scoped_models.py --dry-run
```

## Key Differences from Regular SFT

1. **SAE Hook**: Activations pass through SAE encode→decode at hookpoint
2. **Frozen Layers**: Layers 0 through SAE layer are frozen
3. **Trainable Layers**: Only layers after SAE layer + lm_head train
4. **Same Data**: Uses same science dataset as SFT

## Integration with Evaluation

After training, scoped models are evaluated in `science_evals/` comparing:
- Utility (in-domain performance on science QA)
- Safety (resistance to trojans, refusal on malicious prompts)

Against baselines:
- Vanilla spylab models
- SFT models (without SAE)
- Prompted models

## WandB Configuration

```bash
export WANDB_PROJECT="science-scoping-2026"
export WANDB_API_KEY="your-key"
```

Logged metrics:
- Train/eval loss
- Learning rate schedule
- Trainable vs frozen parameter counts

## Next Steps

1. [ ] Implement `train_scoped_models.py` based on this plan
2. [ ] Test with single SAE/model combination
3. [ ] Run full training grid
4. [ ] Validate with downstream evaluation
