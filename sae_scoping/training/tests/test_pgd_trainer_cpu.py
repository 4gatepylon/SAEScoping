"""
CPU test for PGDSFTTrainer: zero positions must stay zero after training.

Uses a tiny Gemma2 model (2 layers, hidden_size=64) built from config —
no GPU needed, no model download, runs in <30s.

Usage:
  python -m pytest sae_scoping/training/tests/test_pgd_trainer_cpu.py -v
"""

import copy
import tempfile

import pytest
import torch
from datasets import Dataset
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig

from sae_scoping.training.pgd_trainer import PGDSFTTrainer
from sae_scoping.training.saliency.wanda import prune_wanda


MODEL_ID = "google/gemma-2-2b-it"


@pytest.fixture(scope="module")
def tokenizer():
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


@pytest.fixture(scope="module")
def pruned_model_and_masks(tokenizer):
    torch.manual_seed(42)
    config = AutoConfig.from_pretrained(MODEL_ID)
    config.num_hidden_layers = 2
    config.hidden_size = 64
    config.intermediate_size = 128
    config.num_attention_heads = 4
    config.num_key_value_heads = 2
    config.head_dim = 16
    model = AutoModelForCausalLM.from_config(config)

    calib_texts = [
        tokenizer.apply_chat_template(
            [
                {"role": "user", "content": f"Question {i}?"},
                {"role": "assistant", "content": f"Answer {i}."},
            ],
            tokenize=False,
            add_generation_prompt=False,
        )
        for i in range(4)
    ]
    _, masks = prune_wanda(model, tokenizer, calib_texts, sparsity=0.5, max_seq_len=64, return_masks=True)
    return model, masks


def test_zero_positions_stay_zero_after_pgd_training(tokenizer, pruned_model_and_masks):
    model, masks = pruned_model_and_masks
    model = copy.deepcopy(model)

    zero_snapshot = {}
    for name, param in model.named_parameters():
        if name in masks:
            zero_snapshot[name] = (param.data == 0).cpu().clone()

    train_texts = [
        tokenizer.apply_chat_template(
            [
                {"role": "user", "content": f"Train question {i}?"},
                {"role": "assistant", "content": f"Train answer {i}."},
            ],
            tokenize=False,
            add_generation_prompt=False,
        )
        for i in range(8)
    ]
    sft_dataset = Dataset.from_dict({"text": train_texts})

    with tempfile.TemporaryDirectory() as tmp_dir:
        sft_config = SFTConfig(
            output_dir=tmp_dir,
            max_steps=2,
            per_device_train_batch_size=1,
            learning_rate=1e-3,
            max_length=64,
            logging_steps=1,
            save_strategy="no",
            report_to="none",
            use_cpu=True,
        )

        trainer = PGDSFTTrainer(
            masks=masks,
            validate_sparsity=True,
            model=model,
            args=sft_config,
            train_dataset=sft_dataset,
        )
        trainer.train()

    for name, param in model.named_parameters():
        if name in zero_snapshot:
            was_zero = zero_snapshot[name]
            is_zero = (param.data == 0).cpu()
            regrown = was_zero & ~is_zero
            assert not regrown.any(), (
                f"'{name}': {int(regrown.sum())} pruned position(s) became non-zero after training"
            )
