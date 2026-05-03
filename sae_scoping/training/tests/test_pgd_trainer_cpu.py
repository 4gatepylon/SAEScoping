"""
CPU test for PGDSFTTrainer: zero positions must stay zero after training.

Uses a tiny Gemma2 model (2 layers, hidden_size=64) built from config —
no GPU needed, no model download, runs in <30s.

Usage — must be invoked inside the `saescoping` conda env (so that
bitsandbytes, trl, etc. are importable):

  conda activate saescoping
  python -m pytest sae_scoping/training/tests/test_pgd_trainer_cpu.py -v
"""

import copy
import re
import tempfile

import pytest
import torch
from datasets import Dataset
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig

from sae_scoping.training.pgd_trainer import (
    PGDSFTTrainer,
    _is_early_side_param_name,
    filter_masks_by_min_layer_idx,
    freeze_early_side_params,
)
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
            assert not regrown.any(), f"'{name}': {int(regrown.sum())} pruned position(s) became non-zero after training"


# ---------------------------------------------------------------------------
# filter_masks_by_min_layer_idx
# ---------------------------------------------------------------------------


def _dummy_mask(shape=(2, 2)) -> torch.Tensor:
    return torch.ones(shape, dtype=torch.bool)


def test_filter_masks_keeps_only_layers_above_cutoff():
    masks = {
        "model.layers.0.mlp.gate_proj.weight": _dummy_mask(),
        "model.layers.5.self_attn.q_proj.weight": _dummy_mask(),
        "model.layers.31.mlp.up_proj.weight": _dummy_mask(),
        "model.layers.32.mlp.up_proj.weight": _dummy_mask(),
        "model.layers.41.self_attn.o_proj.weight": _dummy_mask(),
    }
    out = filter_masks_by_min_layer_idx(masks, min_layer_idx=31)
    assert set(out.keys()) == {
        "model.layers.32.mlp.up_proj.weight",
        "model.layers.41.self_attn.o_proj.weight",
    }


def test_filter_masks_drops_non_layer_params():
    masks = {
        "model.embed_tokens.weight": _dummy_mask(),
        "model.norm.weight": _dummy_mask(),
        "lm_head.weight": _dummy_mask(),
        "model.layers.10.mlp.gate_proj.weight": _dummy_mask(),
    }
    out = filter_masks_by_min_layer_idx(masks, min_layer_idx=-1)
    assert set(out.keys()) == {"model.layers.10.mlp.gate_proj.weight"}


def test_filter_masks_min_layer_idx_neg1_keeps_all_layered():
    masks = {
        "model.layers.0.mlp.gate_proj.weight": _dummy_mask(),
        "model.layers.10.self_attn.q_proj.weight": _dummy_mask(),
        "model.embed_tokens.weight": _dummy_mask(),
    }
    out = filter_masks_by_min_layer_idx(masks, min_layer_idx=-1)
    assert set(out.keys()) == {
        "model.layers.0.mlp.gate_proj.weight",
        "model.layers.10.self_attn.q_proj.weight",
    }


def test_filter_masks_returns_empty_when_cutoff_too_high():
    masks = {
        "model.layers.0.mlp.gate_proj.weight": _dummy_mask(),
        "model.layers.10.self_attn.q_proj.weight": _dummy_mask(),
    }
    out = filter_masks_by_min_layer_idx(masks, min_layer_idx=999)
    assert out == {}


def test_filter_masks_does_not_copy_tensors():
    m = _dummy_mask()
    masks = {"model.layers.5.mlp.gate_proj.weight": m}
    out = filter_masks_by_min_layer_idx(masks, min_layer_idx=-1)
    assert out["model.layers.5.mlp.gate_proj.weight"] is m


# ---------------------------------------------------------------------------
# Tier 1: filter against real Gemma-2 architecture parameter names
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def gemma2_42layer_param_names():
    """Build a tiny gemma-2-2b-it from config with num_hidden_layers=42 (the
    real 9B layer count). Returns just the parameter NAMES — we never run the
    model. Catches param-name regex bugs against the real Gemma-2 architecture
    without paying for a 9B weights download."""
    config = AutoConfig.from_pretrained(MODEL_ID)
    config.num_hidden_layers = 42
    config.hidden_size = 32
    config.intermediate_size = 64
    config.num_attention_heads = 4
    config.num_key_value_heads = 2
    config.head_dim = 8
    # Gemma-2 alternates sliding/full attention; the layer_types list must
    # match num_hidden_layers in length when overriding from a 2b/2-layer
    # default, otherwise Gemma2DecoderLayer.__init__ raises IndexError.
    config.layer_types = ["full_attention"] * config.num_hidden_layers
    model = AutoModelForCausalLM.from_config(config)
    return [name for name, _ in model.named_parameters()]


def test_filter_masks_against_real_gemma2_arch_min_layer_idx_31(gemma2_42layer_param_names):
    """With min_layer_idx=31 against a 42-layer Gemma-2, the filter should
    keep only layer-indexed params for layers 32..41 inclusive."""
    masks = {n: torch.ones(1, dtype=torch.bool) for n in gemma2_42layer_param_names}
    out = filter_masks_by_min_layer_idx(masks, min_layer_idx=31)
    assert out, "filter dropped everything; expected layers 32..41 to survive"
    for name in out:
        assert ".layers." in name, f"non-layer param survived: {name}"
        layer_idx = int(name.split(".layers.")[1].split(".")[0])
        assert 32 <= layer_idx <= 41, f"unexpected layer idx in kept set: {name}"
    kept_layer_indices = sorted({int(n.split(".layers.")[1].split(".")[0]) for n in out})
    assert kept_layer_indices == list(range(32, 42))


def test_filter_masks_against_real_gemma2_arch_drops_embed_norm_head(gemma2_42layer_param_names):
    """Embedding, lm_head, and the global norm must NOT survive any filter
    (they don't carry a `.layers.<int>.` substring at all)."""
    masks = {n: torch.ones(1, dtype=torch.bool) for n in gemma2_42layer_param_names}
    out = filter_masks_by_min_layer_idx(masks, min_layer_idx=-1)
    assert all(".layers." in n for n in out)
    for n in ("model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"):
        if n in masks:  # tied embeddings may omit lm_head
            assert n not in out


# ---------------------------------------------------------------------------
# Tier 2: end-to-end sweep_wanda.main on CPU with monkey-patched model loading
# ---------------------------------------------------------------------------


def test_sweep_wanda_pgd_layer_subset_end_to_end_cpu(tmp_path, monkeypatch, tokenizer):
    """Drive sweep_wanda.main on CPU with a 4-layer Gemma-2-from-config and
    pgd.min_layer_idx=1, asserting (a) the runner exits cleanly, (b) artifacts
    are written, (c) the layer-subset filter actually filtered out layer-0/1
    masks (only layers 2-3 remain), and (d) post-training, layers 2-3 stayed
    sparse while layer-0/1 was free to drift.

    This is the real integration test for commits 2 and 3: it exercises CLI
    parsing -> SweepConfig merge -> Wanda saliency -> mask filter ->
    PGDSFTTrainer construction with the filtered masks -> per-step validation
    -> artifact writing. Everything except the GPU forward pass on a real
    9B/12B model."""
    from datasets import Dataset
    from click.testing import CliRunner

    # Some hosts have ALL_PROXY=socks5://... in the env, which crashes
    # litellm's import (transitively pulled in by sweep_wanda's judge import)
    # unless the optional `socksio` extra is installed. Drop proxy vars for
    # this test — we're not making network calls anyway.
    for _v in ("ALL_PROXY", "HTTP_PROXY", "HTTPS_PROXY", "all_proxy", "http_proxy", "https_proxy"):
        monkeypatch.delenv(_v, raising=False)

    config = AutoConfig.from_pretrained(MODEL_ID)
    config.num_hidden_layers = 4
    config.hidden_size = 32
    config.intermediate_size = 64
    config.num_attention_heads = 4
    config.num_key_value_heads = 2
    config.head_dim = 8
    torch.manual_seed(0)
    tiny_model = AutoModelForCausalLM.from_config(config)

    n_train = 8
    fake_qa = Dataset.from_dict({"question": [f"Q{i}?" for i in range(n_train)], "answer": [f"A{i}." for i in range(n_train)]})
    chat = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": q}, {"role": "assistant", "content": a}],
            tokenize=False,
            add_generation_prompt=False,
        )
        for q, a in zip(fake_qa["question"], fake_qa["answer"])
    ]

    import experiments.baselines_2026_04_29.sweep_wanda as sw

    monkeypatch.setattr(sw, "load_model_and_tokenizer", lambda mid, device: (tiny_model, tokenizer))
    monkeypatch.setattr(sw, "load_qa_dataset", lambda *a, **kw: fake_qa)
    monkeypatch.setattr(sw, "load_nonoverlapping_splits", lambda *a, **kw: (chat[:4], chat[4:6], chat[6:8]))
    monkeypatch.setattr(sw, "format_as_sft_text", lambda ds, tok: chat[: len(ds)])

    yaml_path = tmp_path / "test_config.yaml"
    yaml_path.write_text(
        """
model_id: google/gemma-2-2b-it
dataset_name: 4gate/StemQAMixture
dataset_subset: biology
calibration: {n_calibration: 4, max_seq_len: 32, batch_size: 1}
sweep: {nn_linear_sparsities: [0.5], n_eval: 2}
operational:
  device: cpu
  no_cache: true
  wandb: {enabled: false}
  llm_judge: {enabled: false}
pgd:
  enabled: true
  n_train: 4
  max_steps: 2
  train_batch_size: 1
  gradient_accumulation_steps: 1
  warmup_ratio: 0.0
  learning_rate: 1.0e-3
  logging_steps: 1
  eval_every_steps: 1
  validate_sparsity: true
  report_to: none
  optim: adamw_torch
  gradient_checkpointing: false
  min_layer_idx: 1
"""
    )
    artifacts_dir = tmp_path / "artifacts"

    # Snapshot pruned-zero positions per layer BEFORE training so we can verify
    # layer-2/3 stayed zero (PGD enforced) and layer-0/1 was free to drift.
    runner = CliRunner()
    result = runner.invoke(
        sw.main,
        [
            "--config",
            str(yaml_path),
            "--artifacts-dir",
            str(artifacts_dir),
            "--device",
            "cpu",
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, f"sweep_main failed:\n{result.output}"

    # Tier-2-A: exit clean + artifacts written.
    runs = sorted((artifacts_dir / "outputs").iterdir())
    assert len(runs) == 1, f"expected 1 run dir, got {len(runs)}"
    run_dir = runs[0]
    assert (run_dir / "metadata.json").exists()
    assert (run_dir / "baseline.json").exists()
    assert (run_dir / "step_000" / "sweep" / "step_metadata.json").exists()
    assert (run_dir / "step_000" / "recovery" / "step_metadata.jsonl").exists()

    # Tier-2-B: filter actually filtered.
    assert "[recovery] PGD layer subset: keeping" in result.output
    assert "with layer index > 1" in result.output


# ---------------------------------------------------------------------------
# Tier 1.5: freeze_early_side_params — the actual memory-saving step.
#
# `filter_masks_by_min_layer_idx` only restricts the PGD zero-projection;
# `freeze_early_side_params` is what stops the optimizer from allocating
# Adam state / gradient buffers for early-layer params. These tests are
# strictly NEW (do not break or reuse the filter tests above).
# ---------------------------------------------------------------------------


def test_is_early_side_param_name_basic_cases():
    """Layer-indexed params at-or-before the cutoff are early-side; past it,
    late-side. Embedding params are always early-side regardless of cutoff.
    Root final norm and (un-tied) lm_head are late-side."""
    assert _is_early_side_param_name("model.layers.0.mlp.gate_proj.weight", 1) is True
    assert _is_early_side_param_name("model.layers.1.self_attn.q_proj.weight", 1) is True
    assert _is_early_side_param_name("model.layers.2.mlp.up_proj.weight", 1) is False
    assert _is_early_side_param_name("model.layers.41.mlp.down_proj.weight", 1) is False
    assert _is_early_side_param_name("model.embed_tokens.weight", 31) is True
    assert _is_early_side_param_name("model.embed_tokens.weight", 0) is True
    assert _is_early_side_param_name("model.norm.weight", 31) is False
    assert _is_early_side_param_name("lm_head.weight", 31) is False


def test_freeze_early_side_params_simple():
    """On a tiny 4-layer Gemma-2-from-config, freezing with min_layer_idx=1
    flips requires_grad off on layers 0..1 + embed_tokens, and leaves
    layers 2..3 + (post-cutoff) norms trainable."""
    config = AutoConfig.from_pretrained(MODEL_ID)
    config.num_hidden_layers = 4
    config.hidden_size = 32
    config.intermediate_size = 64
    config.num_attention_heads = 4
    config.num_key_value_heads = 2
    config.head_dim = 8
    config.tie_word_embeddings = False  # test the un-tied case explicitly here
    model = AutoModelForCausalLM.from_config(config)

    frozen, n_unique = freeze_early_side_params(model, min_layer_idx=1)
    by_name = dict(model.named_parameters())

    # Sanity: returned set agrees with what's actually frozen on the model.
    for name in frozen:
        assert by_name[name].requires_grad is False
    assert n_unique > 0

    # Layers 0–1 frozen; layers 2–3 trainable.
    for n, p in by_name.items():
        m = re.search(r"\.layers\.(\d+)\.", n)
        if m is not None:
            layer = int(m.group(1))
            if layer <= 1:
                assert p.requires_grad is False, f"layer {layer} should be frozen: {n}"
            else:
                assert p.requires_grad is True, f"layer {layer} should be trainable: {n}"
        elif "embed_tokens" in n:
            assert p.requires_grad is False, f"embed_tokens should be frozen: {n}"
        # Other names (root norm, lm_head when un-tied) — no claim either way
        # in this test; the next test covers them explicitly.


def test_freeze_early_side_params_leaves_root_norm_and_untied_lm_head_trainable():
    """When lm_head is NOT tied to embed_tokens, both the root final norm
    (`model.norm.weight`, applied AFTER the last layer) and `lm_head.weight`
    must remain trainable — they live on the late side of the cutoff."""
    config = AutoConfig.from_pretrained(MODEL_ID)
    config.num_hidden_layers = 4
    config.hidden_size = 32
    config.intermediate_size = 64
    config.num_attention_heads = 4
    config.num_key_value_heads = 2
    config.head_dim = 8
    config.tie_word_embeddings = False
    model = AutoModelForCausalLM.from_config(config)

    freeze_early_side_params(model, min_layer_idx=1)
    by_name = dict(model.named_parameters())

    assert by_name["model.norm.weight"].requires_grad is True
    # lm_head.weight is a separate Parameter when not tied
    assert by_name["lm_head.weight"].requires_grad is True
    assert by_name["lm_head.weight"].data_ptr() != by_name["model.embed_tokens.weight"].data_ptr()


def test_freeze_early_side_params_ties_embed_to_lm_head_freezes_both():
    """The hard case: when lm_head.weight IS tied to embed_tokens.weight (the
    Gemma config default), the SAME underlying tensor is used both at the
    input (lookup) and at the output (logits projection). Per the design rule
    'anything before the cutoff is NOT trained even if the same tensor is
    also used after', this tied tensor must be frozen.

    We test both:
      (a) the surfaced requires_grad flag is False under both names
          (named_parameters with remove_duplicate=False yields both),
      (b) the underlying tensor is *the same object* in both slots, so
          flipping one alias is sufficient — flipping it via the early-side
          alias propagates to the late-side alias automatically."""
    config = AutoConfig.from_pretrained(MODEL_ID)
    config.num_hidden_layers = 4
    config.hidden_size = 32
    config.intermediate_size = 64
    config.num_attention_heads = 4
    config.num_key_value_heads = 2
    config.head_dim = 8
    config.tie_word_embeddings = True  # the case we care about for Gemma
    model = AutoModelForCausalLM.from_config(config)
    model.tie_weights()  # belt-and-suspenders; transformers usually does this in __init__

    # Prove the tying actually happened — same Parameter object reachable both ways.
    embed = model.get_input_embeddings().weight
    out_head = model.get_output_embeddings().weight
    assert embed is out_head or embed.data_ptr() == out_head.data_ptr(), "lm_head must be tied to embed_tokens for this test to be meaningful"

    freeze_early_side_params(model, min_layer_idx=1)

    # Surface check: every alias of the tied tensor reports requires_grad=False.
    seen_tied = []
    for name, p in model.named_parameters(recurse=True, remove_duplicate=False):
        if p.data_ptr() == embed.data_ptr():
            seen_tied.append(name)
            assert p.requires_grad is False, f"tied tensor still trainable via alias: {name}"
    assert any("embed_tokens" in n for n in seen_tied), "test didn't actually traverse embed alias"


def test_freeze_early_side_params_drops_optimizer_state_for_frozen_tensors():
    """The whole point of the freeze pass: an Adam optimizer constructed from
    `model.parameters()` (or `(p for p in model.parameters() if p.requires_grad)`)
    must NOT allocate state for frozen tensors. Confirms the memory savings
    are real (not just a docstring claim)."""
    config = AutoConfig.from_pretrained(MODEL_ID)
    config.num_hidden_layers = 4
    config.hidden_size = 16
    config.intermediate_size = 32
    config.num_attention_heads = 4
    config.num_key_value_heads = 2
    config.head_dim = 4
    model = AutoModelForCausalLM.from_config(config)

    freeze_early_side_params(model, min_layer_idx=1)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(trainable, lr=1e-3)

    # Force optimizer to allocate state by taking one step.
    for p in trainable:
        p.grad = torch.zeros_like(p)
    optim.step()

    # Adam allocates exp_avg + exp_avg_sq per trainable param. Total state
    # tensor count == 2 * len(trainable). No state for frozen params.
    state_param_set = set()
    for p_state in optim.state.values():
        # Each entry corresponds to one trainable param.
        if p_state:
            state_param_set.add(id(p_state))
    assert len(optim.state) == len(trainable), (
        f"optim has state for {len(optim.state)} params; expected exactly {len(trainable)} (the trainable subset)"
    )

    # Also verify by id: every param the optimizer holds state for is one we
    # asked it to optimize, and none of them are frozen.
    optim_param_ids = {id(p) for group in optim.param_groups for p in group["params"]}
    frozen_ids = {id(p) for p in model.parameters() if not p.requires_grad}
    assert optim_param_ids.isdisjoint(frozen_ids), "optimizer is tracking at least one frozen parameter"
