"""
test_sft_only.py — throwaway script to verify SFT alone OOMs before
blaming the pruning pipeline. Run with:

    CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n saescoping \
        python -u experiments/saliency_pruning/toy_sweep_2026_03_14/test_sft_only.py
"""

import torch
from datasets import load_dataset
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

from dataset_utils import format_as_sft_dataset, load_qa_dataset

MODEL_ID   = "google/gemma-2-9b-it"
DEVICE     = "cuda"
DATASET    = "4gate/StemQAMixture"
SUBSET     = "biology"
N_RECOVERY = 32
BATCH_SIZE = 1
GRAD_ACCUM = 8
MAX_STEPS  = 5          # just enough to hit the first optimizer.step()
MAX_SEQ    = 1024
LR         = 2e-5

print(f"[test_sft_only] Loading {MODEL_ID}...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, dtype=torch.bfloat16, device_map=DEVICE,
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
chat_template_path = (
    Path(__file__).parent / "prompts" / "gemma2_chat_template_system_prompt.j2"
)
if chat_template_path.exists():
    tokenizer.chat_template = chat_template_path.read_text()
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

alloc = torch.cuda.memory_allocated() / 1024**3
print(f"[test_sft_only] Model loaded — GPU alloc: {alloc:.2f} GB")

dataset_rec = load_qa_dataset(DATASET, SUBSET, split="train", n=N_RECOVERY, seed=42)
sft_dataset = format_as_sft_dataset(dataset_rec, tokenizer)

for p in model.parameters():
    p.requires_grad = True

args = SFTConfig(
    output_dir="/tmp/test_sft_only_output",
    max_steps=MAX_STEPS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LR,
    bf16=True,
    save_strategy="no",
    report_to="none",
    max_length=MAX_SEQ,
    dataset_text_field="text",
    logging_steps=1,
)

trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=sft_dataset,
    args=args,
)

print("[test_sft_only] Starting SFT (no pruning)...")
trainer.train()
print("[test_sft_only] Done — SFT completed without OOM")