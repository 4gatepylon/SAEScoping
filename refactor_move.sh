#!/usr/bin/env bash
# ==========================================================================
# refactor_move.sh — Restructure sae_scoping/ and move experiment code into library.
#
# This script ONLY creates directories and runs git mv / git rm / git add.
# It never modifies file contents. After running, imports will be broken
# and need manual fixup (see IMPORT_FIXUPS section printed at the end).
#
# Usage:
#   cd /Users/4gate/git/SAEScopingPruning
#   bash refactor_move.sh
#
# To undo everything:
#   git checkout HEAD -- .
#   git clean -fd sae_scoping/training sae_scoping/evaluation
#
# ==========================================================================
# COMPLETE FILE MOVE MAP
# ==========================================================================
#
# Legend:
#   [LIB]  = existing library file being reorganized
#   [EXP]  = experiment file being promoted to library
#   [NEW]  = newly created file (empty __init__.py)
#   [STAY] = file that stays in experiments (listed for completeness)
#
# --- Library restructure: trainers/ -> training/ ---
#   [LIB] sae_scoping/trainers/sae_enhanced/__init__.py   -> sae_scoping/training/sae_enhanced/__init__.py
#   [LIB] sae_scoping/trainers/sae_enhanced/prune.py      -> sae_scoping/training/sae_enhanced/pruning.py
#   [LIB] sae_scoping/trainers/sae_enhanced/rank.py       -> sae_scoping/training/sae_enhanced/firing_rates.py
#   [LIB] sae_scoping/trainers/sae_enhanced/train.py      -> sae_scoping/training/sae_enhanced/sae_aware_sft.py
#   [LIB] sae_scoping/trainers/sae_enhanced/utils.py      -> sae_scoping/training/sae_enhanced/utils.py
#
# --- Library restructure: utils/hooks/ -> training/sae_enhanced/hooks/ ---
#   [LIB] sae_scoping/utils/hooks/__init__.py              -> sae_scoping/training/sae_enhanced/hooks/__init__.py
#   [LIB] sae_scoping/utils/hooks/pt_hooks.py              -> sae_scoping/training/sae_enhanced/hooks/pt_hooks.py
#   [LIB] sae_scoping/utils/hooks/pt_hooks_stateful.py     -> sae_scoping/training/sae_enhanced/hooks/pt_hooks_stateful.py
#   [LIB] sae_scoping/utils/hooks/sae.py                   -> sae_scoping/training/sae_enhanced/hooks/sae.py
#   [LIB] sae_scoping/utils/hooks/test_pt_hooks.py         -> sae_scoping/training/sae_enhanced/hooks/test_pt_hooks.py
#
# --- Library restructure: utils/xxx_generation/ -> evaluation/inference/client/ ---
#   [LIB] sae_scoping/utils/xxx_generation/__init__.py                  -> sae_scoping/evaluation/inference/client/__init__.py
#   [LIB] sae_scoping/utils/xxx_generation/api_generator.py             -> sae_scoping/evaluation/inference/client/api_generator.py
#   [LIB] sae_scoping/utils/xxx_generation/hf_generator.py              -> sae_scoping/evaluation/inference/client/model_generator.py
#   [LIB] sae_scoping/utils/xxx_generation/base_generator.py            -> sae_scoping/evaluation/inference/client/base_generator.py
#   [LIB] sae_scoping/utils/xxx_generation/messages.py                  -> sae_scoping/evaluation/inference/client/messages.py
#   [LIB] sae_scoping/utils/xxx_generation/hardcoded_cache_generator.py -> sae_scoping/evaluation/inference/client/hardcoded_cache_generator.py
#   [LIB] sae_scoping/utils/xxx_generation/xxx_length_aware_tokenizer.py -> sae_scoping/evaluation/inference/client/xxx_length_aware_tokenizer.py
#
# --- Library restructure: xxx_evaluation/ -> evaluation/ ---
#   [LIB] sae_scoping/xxx_evaluation/__init__.py                -> sae_scoping/evaluation/__init__.py
#   [LIB] sae_scoping/xxx_evaluation/trainer_callbacks.py       -> sae_scoping/evaluation/trainer_callbacks.py
#   [LIB] sae_scoping/xxx_evaluation/spylab_1click_judgement.py -> sae_scoping/evaluation/spylab_1click_judgement.py
#
# --- Experiment -> library: weight pruning + PGD ---
#   [EXP] experiments/.../prune.py       -> sae_scoping/training/weight_pruning.py
#   [EXP] experiments/.../pgd_trainer.py -> sae_scoping/training/pgd_trainer.py
#
# --- Experiment -> library: saliency map algorithms ---
#   [EXP] experiments/.../gradients_map/grad.py    -> sae_scoping/training/saliency/grad.py
#   [EXP] experiments/.../gradients_map/taylor.py  -> sae_scoping/training/saliency/taylor.py
#   [EXP] experiments/.../gradients_map/random.py  -> sae_scoping/training/saliency/random.py
#   [EXP] experiments/.../gradients_map/utils.py   -> sae_scoping/training/saliency/utils.py
#
# --- Experiment -> library: grading + prompts ---
#   [EXP] experiments/.../grade_chats.py                       -> sae_scoping/evaluation/grade_chats/generic_judges.py
#   [EXP] experiments/.../prompts/precise_classifier.j2        -> sae_scoping/evaluation/grade_chats/prompts/generic/precise_classifier.j2
#   [EXP] experiments/.../prompts/answering_classifier.j2      -> sae_scoping/evaluation/grade_chats/prompts/generic/answering_classifier.j2
#   [EXP] experiments/.../prompts/factual_helpful_classifier.j2 -> sae_scoping/evaluation/grade_chats/prompts/generic/factual_helpful_classifier.j2
#   [EXP] experiments/.../prompts/refusal.j2                   -> sae_scoping/evaluation/grade_chats/prompts/generic/refusal.j2
#
# --- Experiment -> library: dataset + eval utilities ---
#   [EXP] experiments/.../dataset_utils.py -> sae_scoping/datasets/qa_datasets.py
#   [EXP] experiments/.../utils.py         -> sae_scoping/evaluation/grade_model.py
#                                              (needs manual split later: RecoveryCallback -> training/)
#
# --- New empty __init__.py files ---
#   [NEW] sae_scoping/training/__init__.py
#   [NEW] sae_scoping/training/saliency/__init__.py
#   [NEW] sae_scoping/evaluation/inference/__init__.py
#   [NEW] sae_scoping/evaluation/inference/server/__init__.py
#   [NEW] sae_scoping/evaluation/grade_chats/__init__.py
#
# --- Files that STAY in experiments (not moved) ---
#   [STAY] experiments/.../prune_and_maybe_recover.py           (experiment pipeline / CLI)
#   [STAY] experiments/.../generate_chats.py                    (thin bridge)
#   [STAY] experiments/.../model_generator.py                   (DUPLICATE — needs merge with library version)
#   [STAY] experiments/.../api_generator.py                     (DUPLICATE — needs merge with library version)
#   [STAY] experiments/.../gradients_map/__init__.py            (CLI entry point)
#   [STAY] experiments/.../gradients_map/__main__.py            (CLI entry point)
#   [STAY] experiments/.../gradients_map/batch.py               (multi-GPU orchestration)
#   [STAY] experiments/.../prompts/gemma2_chat_template_system_prompt.j2  (model-specific)
#   [STAY] experiments/.../tests/                               (all tests stay for now)
#   [STAY] experiments/.../CLAUDE.md, README.md, *.sh, etc.
#
# --- Untouched library directories ---
#   sae_scoping/utils/gemma2/       (stays — not part of this refactor)
#   sae_scoping/utils/spylab/       (stays — not part of this refactor)
#   sae_scoping/models/             (stays — not part of this refactor)
#   sae_scoping/datasets/__init__.py, messages_datasets.py, text_datasets.py (stays)
#   sae_scoping/hyperparameter_optimization/  (stays — already well placed)
#
# ==========================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_ROOT"

EXP="experiments/saliency_pruning/toy_sweep_2026_03_14"

echo "=== Phase 1: Create new directory structure ==="

mkdir -p sae_scoping/training/sae_enhanced/hooks
mkdir -p sae_scoping/training/saliency
mkdir -p sae_scoping/evaluation/inference/server
mkdir -p sae_scoping/evaluation/inference/client
mkdir -p sae_scoping/evaluation/grade_chats/prompts/generic
mkdir -p sae_scoping/evaluation/grade_chats/prompts/biology
mkdir -p sae_scoping/evaluation/grade_chats/prompts/answer_extraction
mkdir -p sae_scoping/evaluation/grade_chats/prompts/check_answer

echo "=== Phase 2: Create new __init__.py files ==="

touch sae_scoping/training/__init__.py
touch sae_scoping/training/saliency/__init__.py
touch sae_scoping/evaluation/inference/__init__.py
touch sae_scoping/evaluation/inference/server/__init__.py
touch sae_scoping/evaluation/grade_chats/__init__.py

git add \
  sae_scoping/training/__init__.py \
  sae_scoping/training/saliency/__init__.py \
  sae_scoping/evaluation/inference/__init__.py \
  sae_scoping/evaluation/inference/server/__init__.py \
  sae_scoping/evaluation/grade_chats/__init__.py

# ============================================================
# Phase 3: Restructure existing library (sae_scoping/)
# ============================================================
echo "=== Phase 3a: trainers/sae_enhanced/ -> training/sae_enhanced/ ==="

git mv sae_scoping/trainers/sae_enhanced/__init__.py   sae_scoping/training/sae_enhanced/__init__.py
git mv sae_scoping/trainers/sae_enhanced/prune.py      sae_scoping/training/sae_enhanced/pruning.py
git mv sae_scoping/trainers/sae_enhanced/rank.py       sae_scoping/training/sae_enhanced/firing_rates.py
git mv sae_scoping/trainers/sae_enhanced/train.py      sae_scoping/training/sae_enhanced/sae_aware_sft.py
git mv sae_scoping/trainers/sae_enhanced/utils.py      sae_scoping/training/sae_enhanced/utils.py

echo "=== Phase 3b: utils/hooks/ -> training/sae_enhanced/hooks/ ==="

git mv sae_scoping/utils/hooks/__init__.py          sae_scoping/training/sae_enhanced/hooks/__init__.py
git mv sae_scoping/utils/hooks/pt_hooks.py          sae_scoping/training/sae_enhanced/hooks/pt_hooks.py
git mv sae_scoping/utils/hooks/pt_hooks_stateful.py sae_scoping/training/sae_enhanced/hooks/pt_hooks_stateful.py
git mv sae_scoping/utils/hooks/sae.py               sae_scoping/training/sae_enhanced/hooks/sae.py
git mv sae_scoping/utils/hooks/test_pt_hooks.py     sae_scoping/training/sae_enhanced/hooks/test_pt_hooks.py

echo "=== Phase 3c: utils/xxx_generation/ -> evaluation/inference/client/ ==="

git mv sae_scoping/utils/xxx_generation/__init__.py                 sae_scoping/evaluation/inference/client/__init__.py
git mv sae_scoping/utils/xxx_generation/api_generator.py            sae_scoping/evaluation/inference/client/api_generator.py
git mv sae_scoping/utils/xxx_generation/hf_generator.py             sae_scoping/evaluation/inference/client/model_generator.py
git mv sae_scoping/utils/xxx_generation/base_generator.py           sae_scoping/evaluation/inference/client/base_generator.py
git mv sae_scoping/utils/xxx_generation/messages.py                 sae_scoping/evaluation/inference/client/messages.py
git mv sae_scoping/utils/xxx_generation/hardcoded_cache_generator.py sae_scoping/evaluation/inference/client/hardcoded_cache_generator.py
git mv sae_scoping/utils/xxx_generation/xxx_length_aware_tokenizer.py sae_scoping/evaluation/inference/client/xxx_length_aware_tokenizer.py

echo "=== Phase 3d: xxx_evaluation/ -> evaluation/ ==="

git mv sae_scoping/xxx_evaluation/__init__.py                sae_scoping/evaluation/__init__.py
git mv sae_scoping/xxx_evaluation/trainer_callbacks.py       sae_scoping/evaluation/trainer_callbacks.py
git mv sae_scoping/xxx_evaluation/spylab_1click_judgement.py sae_scoping/evaluation/spylab_1click_judgement.py

# ============================================================
# Phase 4: Move experiment files into library
# ============================================================
echo "=== Phase 4a: Weight pruning + PGD trainer -> training/ ==="

git mv "$EXP/prune.py"       sae_scoping/training/weight_pruning.py
git mv "$EXP/pgd_trainer.py" sae_scoping/training/pgd_trainer.py

echo "=== Phase 4b: Saliency algorithms -> training/saliency/ ==="
# Only core algorithm files. CLI wrappers (__init__, __main__, batch) stay in experiment.

git mv "$EXP/gradients_map/grad.py"    sae_scoping/training/saliency/grad.py
git mv "$EXP/gradients_map/taylor.py"  sae_scoping/training/saliency/taylor.py
git mv "$EXP/gradients_map/random.py"  sae_scoping/training/saliency/random.py
git mv "$EXP/gradients_map/utils.py"   sae_scoping/training/saliency/utils.py

echo "=== Phase 4c: Grade chats + prompts -> evaluation/grade_chats/ ==="

git mv "$EXP/grade_chats.py" sae_scoping/evaluation/grade_chats/generic_judges.py

git mv "$EXP/prompts/precise_classifier.j2"          sae_scoping/evaluation/grade_chats/prompts/generic/precise_classifier.j2
git mv "$EXP/prompts/answering_classifier.j2"        sae_scoping/evaluation/grade_chats/prompts/generic/answering_classifier.j2
git mv "$EXP/prompts/factual_helpful_classifier.j2"  sae_scoping/evaluation/grade_chats/prompts/generic/factual_helpful_classifier.j2
git mv "$EXP/prompts/refusal.j2"                     sae_scoping/evaluation/grade_chats/prompts/generic/refusal.j2

echo "=== Phase 4d: Dataset utils -> datasets/ ==="

git mv "$EXP/dataset_utils.py" sae_scoping/datasets/qa_datasets.py

echo "=== Phase 4e: Eval/metric utils -> evaluation/grade_model.py ==="
# NOTE: This file contains BOTH evaluation logic (compute_validation_loss, evaluate_model)
# AND training logic (RecoveryCallback, GiveUpThreshold). It needs manual splitting later.
# Moving as-is to evaluation/ since the majority of its contents are eval-related.

git mv "$EXP/utils.py" sae_scoping/evaluation/grade_model.py

# ============================================================
# Phase 5: Clean up empty directories
# ============================================================
echo "=== Phase 5: Clean up emptied directories ==="

# Remove now-empty __init__.py files that kept old dirs alive
git rm sae_scoping/trainers/sae_enhanced/__init__.py 2>/dev/null || true
git rm sae_scoping/trainers/__init__.py              2>/dev/null || true

# Remove leftover empty dirs (git doesn't track dirs, so rmdir is safe)
rmdir sae_scoping/trainers/sae_enhanced 2>/dev/null || true
rmdir sae_scoping/trainers              2>/dev/null || true
rmdir sae_scoping/utils/hooks           2>/dev/null || true
rmdir sae_scoping/utils/xxx_generation  2>/dev/null || true
rmdir sae_scoping/xxx_evaluation        2>/dev/null || true

# Clean up experiment __pycache__ dirs left behind (not tracked by git)
rm -rf "$EXP/gradients_map/__pycache__" 2>/dev/null || true
rm -rf "$EXP/__pycache__"               2>/dev/null || true

echo ""
echo "================================================================"
echo " DONE. All moves complete. Imports are NOW BROKEN."
echo "================================================================"
echo ""
echo "Files that STAYED in experiments (not moved):"
echo "  $EXP/prune_and_maybe_recover.py   (experiment orchestrator / CLI)"
echo "  $EXP/generate_chats.py            (thin bridge, per CLAUDE.md Q6)"
echo "  $EXP/model_generator.py           (DUPLICATE of library HFGenerator — needs merge)"
echo "  $EXP/api_generator.py             (DUPLICATE of library APIGenerator — needs merge)"
echo "  $EXP/gradients_map/__init__.py    (CLI entry point for 'python -m gradients_map')"
echo "  $EXP/gradients_map/__main__.py    (CLI entry point)"
echo "  $EXP/gradients_map/batch.py       (multi-GPU orchestration script)"
echo "  $EXP/prompts/gemma2_chat_template_system_prompt.j2  (model-specific template)"
echo "  $EXP/tests/                       (all test files, for now)"
echo "  $EXP/CLAUDE.md, README.md, etc."
echo "  All shell scripts (run_taylor_*.sh etc.)"
echo ""
echo "================================================================"
echo " IMPORT FIXUPS NEEDED (see below)"
echo "================================================================"
cat <<'FIXUPS'

## 1. WITHIN THE LIBRARY (sae_scoping/ internal imports)

### sae_scoping/training/sae_enhanced/pruning.py (was trainers/sae_enhanced/prune.py)
  OLD: from sae_scoping.utils.hooks.pt_hooks_stateful import Context
  NEW: from sae_scoping.training.sae_enhanced.hooks.pt_hooks_stateful import Context
  OLD: from sae_scoping.utils.hooks.sae import SAELensEncDecCallbackWrapper
  NEW: from sae_scoping.training.sae_enhanced.hooks.sae import SAELensEncDecCallbackWrapper
  OLD: from sae_scoping.trainers.sae_enhanced.utils import str_dict_diff, is_int
  NEW: from sae_scoping.training.sae_enhanced.utils import str_dict_diff, is_int

### sae_scoping/training/sae_enhanced/firing_rates.py (was trainers/sae_enhanced/rank.py)
  OLD: from sae_scoping.utils.hooks.pt_hooks import filter_hook_fn, named_forward_hooks
  NEW: from sae_scoping.training.sae_enhanced.hooks.pt_hooks import filter_hook_fn, named_forward_hooks
  OLD: from sae_scoping.utils.hooks.pt_hooks_stateful import Context
  NEW: from sae_scoping.training.sae_enhanced.hooks.pt_hooks_stateful import Context
  OLD: from sae_scoping.utils.hooks.sae import SAELensEncDecCallbackWrapper, SAEWrapper
  NEW: from sae_scoping.training.sae_enhanced.hooks.sae import SAELensEncDecCallbackWrapper, SAEWrapper

### sae_scoping/training/sae_enhanced/sae_aware_sft.py (was trainers/sae_enhanced/train.py)
  OLD: from utils.hooks.pt_hooks import filter_hook_fn, named_forward_hooks
  NEW: from sae_scoping.training.sae_enhanced.hooks.pt_hooks import filter_hook_fn, named_forward_hooks
  OLD: from sae_scoping.utils.hooks.sae import SAEWrapper, Context, SAELensEncDecCallbackWrapper
  NEW: from sae_scoping.training.sae_enhanced.hooks.sae import SAEWrapper, SAELensEncDecCallbackWrapper
  NEW: from sae_scoping.training.sae_enhanced.hooks.pt_hooks_stateful import Context
  OLD: from sae_scoping.trainers.sae_enhanced.utils import str_dict_diff
  NEW: from sae_scoping.training.sae_enhanced.utils import str_dict_diff
  OLD: from sae_scoping.trainers.sae_enhanced.rank import rank_neurons
  NEW: from sae_scoping.training.sae_enhanced.firing_rates import rank_neurons
  OLD: from sae_scoping.trainers.sae_enhanced.prune import get_pruned_sae
  NEW: from sae_scoping.training.sae_enhanced.pruning import get_pruned_sae

### sae_scoping/evaluation/inference/client/model_generator.py (was utils/xxx_generation/hf_generator.py)
  OLD: from sae_scoping.utils.generation.base_generator import BaseGenerator, MessagesWrapper
  NEW: from sae_scoping.evaluation.inference.client.base_generator import BaseGenerator, MessagesWrapper
  OLD: from sae_scoping.utils.generation.messages import OpenAIMessages, is_valid_messages, is_valid_1turn_messages
  NEW: from sae_scoping.evaluation.inference.client.messages import OpenAIMessages, is_valid_messages, is_valid_1turn_messages

### sae_scoping/evaluation/trainer_callbacks.py (was xxx_evaluation/trainer_callbacks.py)
  OLD: from sae_scoping.evaluation.spylab_1click_judgement import ...
  NEW: (same path — already correct if it was importing from sae_scoping.evaluation)
  NOTE: Check this — the old file was in xxx_evaluation/ but imported from sae_scoping.evaluation

### sae_scoping/evaluation/grade_model.py (was experiments/.../utils.py)
  OLD: from grade_chats import GradedChats, grade_chats
  NEW: from sae_scoping.evaluation.grade_chats.generic_judges import GradedChats, grade_chats
  OLD: from model_generator import HFGenerator
  NEW: from sae_scoping.evaluation.inference.client.model_generator import HFGenerator
  NOTE: Also uses is_valid_messages etc from model_generator — check client/messages.py

### sae_scoping/evaluation/grade_chats/generic_judges.py (was experiments/.../grade_chats.py)
  OLD: from api_generator import APIGenerator, load_jinja_template
  NEW: from sae_scoping.evaluation.inference.client.api_generator import APIGenerator, load_jinja_template
  NOTE: _JUDGE_TEMPLATE_PATHS uses Path(__file__).parent which now points to grade_chats/
        — paths should resolve to grade_chats/prompts/generic/ (correct after move)

### sae_scoping/training/saliency/grad.py (was experiments/.../gradients_map/grad.py)
  OLD: from dataset_utils import format_as_sft_dataset, load_qa_dataset
  NEW: from sae_scoping.datasets.qa_datasets import format_as_sft_dataset, load_qa_dataset
  OLD: from gradients_map.random import make_random_map
  NEW: from sae_scoping.training.saliency.random import make_random_map
  OLD: from gradients_map.utils import (...)
  NEW: from sae_scoping.training.saliency.utils import (...)

### sae_scoping/training/saliency/taylor.py (was experiments/.../gradients_map/taylor.py)
  OLD: from gradients_map.utils import _DEFAULT_MODEL_ID, save_saliency_map
  NEW: from sae_scoping.training.saliency.utils import _DEFAULT_MODEL_ID, save_saliency_map

### sae_scoping/training/saliency/utils.py (was experiments/.../gradients_map/utils.py)
  NOTE: Contains experiment-specific constants (_DEFAULT_MODEL_ID = "google/gemma-2-9b-it",
        _DEFAULT_DATASET, _VARIANT_SPECS, etc.) that should be removed or made configurable.
        Also contains _CHAT_TEMPLATE_PATH using Path(__file__).parent which is now wrong.

### sae_scoping/datasets/qa_datasets.py (was experiments/.../dataset_utils.py)
  OLD: from model_generator import OpenAIMessages
  NEW: from sae_scoping.evaluation.inference.client.messages import OpenAIMessages

### sae_scoping/models/sae_enhanced_gemma2.py
  Check for any imports from old paths (utils.hooks, trainers.sae_enhanced, etc.)


## 2. IN EXPERIMENTS (files that stayed but whose imports broke)

### experiments/.../prune_and_maybe_recover.py
  OLD: from dataset_utils import ...
  NEW: from sae_scoping.datasets.qa_datasets import format_as_0turn, format_as_sft_dataset, format_as_sft_text, load_qa_dataset
  OLD: from pgd_trainer import PGDSFTTrainer
  NEW: from sae_scoping.training.pgd_trainer import PGDSFTTrainer
  OLD: from prune import prune_model
  NEW: from sae_scoping.training.weight_pruning import prune_model
  OLD: from utils import RecoveryCallback, evaluate_model, is_metric_passing, resolve_threshold
  NEW: from sae_scoping.evaluation.grade_model import RecoveryCallback, evaluate_model, is_metric_passing, resolve_threshold

### experiments/.../generate_chats.py
  OLD: from dataset_utils import format_as_0turn, load_qa_dataset, validate_qa_dataset
  NEW: from sae_scoping.datasets.qa_datasets import format_as_0turn, load_qa_dataset, validate_qa_dataset
  OLD: from model_generator import HFGenerator, OpenAIMessages
  NEW: Keep as-is (model_generator.py stayed in experiments)
  NOTE: experiment's model_generator.py is a DUPLICATE that should eventually be merged
        with sae_scoping/evaluation/inference/client/model_generator.py

### experiments/.../model_generator.py (STAYED — duplicate, needs merge)
  No import changes needed (standalone), but should eventually be replaced by
  importing from sae_scoping.evaluation.inference.client.model_generator

### experiments/.../api_generator.py (STAYED — duplicate, needs merge)
  No import changes needed (standalone), but should eventually be replaced by
  importing from sae_scoping.evaluation.inference.client.api_generator
  NOTE: experiment version has newer GPT-5 hotfix models — merge these into library version

### experiments/.../gradients_map/__init__.py (STAYED — CLI entry point, now broken)
  OLD: from gradients_map.batch import batch
  OLD: from gradients_map.grad import GradCollectTrainer, grad
  OLD: from gradients_map.random import make_random_map
  OLD: from gradients_map.taylor import make_taylor_map, run_taylor, taylor_output_path, validate_taylor_source_path
  OLD: from gradients_map.utils import assert_all_params_require_grad, save_saliency_map
  NEW: from sae_scoping.training.saliency.grad import GradCollectTrainer
  NEW: from sae_scoping.training.saliency.random import make_random_map
  NEW: from sae_scoping.training.saliency.taylor import make_taylor_map, run_taylor, ...
  NEW: from sae_scoping.training.saliency.utils import assert_all_params_require_grad, save_saliency_map
  NOTE: batch.py stayed, so its import stays. The click commands (grad, run_taylor)
        are still in the moved files — you may want to separate CLI from library code.

### experiments/.../gradients_map/batch.py (STAYED)
  OLD: from gradients_map.utils import (...)
  NEW: from sae_scoping.training.saliency.utils import (...)


## 3. MANUAL TASKS AFTER IMPORT FIXUPS

a) SPLIT sae_scoping/evaluation/grade_model.py:
   - Move RecoveryCallback + GiveUpThreshold to sae_scoping/training/recovery_callback.py
   - Keep compute_validation_loss, evaluate_model, generate_and_grade, metric helpers in grade_model.py

b) MERGE duplicate generators:
   - experiments/.../model_generator.py (has caching) vs
     sae_scoping/evaluation/inference/client/model_generator.py (uses BaseGenerator)
   - experiments/.../api_generator.py (newer GPT-5 hotfix) vs
     sae_scoping/evaluation/inference/client/api_generator.py (older)
   After merge, delete experiment copies and update experiment imports.

c) CLEAN UP sae_scoping/training/saliency/utils.py:
   - Remove experiment-specific constants or make them configurable parameters
   - Fix _CHAT_TEMPLATE_PATH (Path(__file__).parent is now wrong)

d) Move test files to follow their modules:
   - experiments/.../tests/unit/test_pgd_trainer.py -> near sae_scoping/training/pgd_trainer.py
   - experiments/.../tests/unit/test_prune.py -> near sae_scoping/training/weight_pruning.py
   - experiments/.../tests/unit/test_gradients_map.py -> near sae_scoping/training/saliency/
   - experiments/.../tests/test_grade_chats.py -> near sae_scoping/evaluation/grade_chats/

e) Update sae_scoping/__init__.py if it re-exports anything from old paths.

f) Update any external references (CLAUDE.md, README.md, shell scripts)
   that hardcode old paths.

FIXUPS

echo ""
echo "Review the staged changes with: git diff --cached --stat"
echo "Undo everything with:           git reset HEAD -- . && git checkout -- ."
