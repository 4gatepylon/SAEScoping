from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorWithPadding,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
import gc
import hashlib
import json
import os
import click
import warnings
from datasets import Dataset, DatasetDict, load_dataset
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import torch
from transformers.trainer_pt_utils import LengthGroupedSampler

# Quick recap from run debugging:
# - Early loss spikes can happen (especially with small effective batch + length grouping), and recovery speed
#   matters more than the single peak acc. to Claude.
# - Larger effective batch generally reduced spike magnitude; disabling/coarsening length grouping might
#   also make jumps smaller (disabling definately did, but I did not try coarsening).
# - Added stability controls here: `max_grad_norm` flag (default 10.0) and warmup controls
#   (`warmup_steps`, `warmup_ratio=0.05` default). Claude recommended warmup of around 0.05-0.1 and
#   the gradient norm was picked from what seemed reasonable to me from the logs. You can only pick
#   steps or ratio (if you do both HF may pick one and you should just avoid the ambiguity).
# - Other options include tuning beta, weight deca, dropout, loss masking on prompts, etc... HOWEVER:
# - THE MOST IMPORTANT FACTOR is probably data quality, amount, etc...
# - The time to train on around 1K batches of size 8 (batch) x 4 (accum) is around 30-45m on A100
# - Claude claims lr should be inversly proportional to the number of parameters... roughly
# - Gemma-2-2b gets better at biology, but extremely slowly, compared to my recolleciton of how Gemma-2-9b-it did.
#   Is this due to pretrained vs. IT or due to small vs. large model?
# - Training only on output tokens for Gemma-2-2b has really high loss. It DOES go down significnatly.
# - TODO(hadriano) impact of training on gemma2-9b bio (does eval actually get better?) <-- probably 9b will "just work"
# - TODO(hadriano) impact of training on gemma3-4b bio (does eval actually get better?) <--- probably will work better idk
# - TODO(hadriano) impact of using IT model vs PT model?
# - TODO(hadriano) impact on judge of training (does judge actually get better? how does it relate to loss?)
# - TODO(hadriano) impact of training only on output tokens?

# TODO(hadriano) tests from gemma-2-2b-it + gemma-3-4b-it

# TODO(hadriano) in the future add type static analysis and make sure it works/runs
MAX_LENGTH = 4096
BATCH_SIZE = 1
GRAD_ACCUM_STEPS = 2
LEARNING_RATE = 5e-5
EVAL_EVERY = 0
MAX_GRAD_NORM = 10.0
WARMUP_STEPS = 0
WARMUP_RATIO = 0.05
ANSWER_ONLY_LOSS = False
NUM_SAMPLES_TRAIN = 64
NUM_SAMPLES_EVAL = 64
MODEL_NAME = "google/gemma-2-2b"
DATASET_NAME = "4gate/StemQAMixture"
DATASET_CONFIG = "biology"
LAYERS_KEEP = "1:1"


def _format_param_count(n: int) -> str:
    """
    Assumptions:
    - `n` is a non-negative integer parameter count.
    """
    assert isinstance(n, int) and n >= 0, "Parameter count must be a non-negative int."
    if n >= 1_000_000_000:
        return f"{n / 1_000_000_000:.2f}B"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    if n >= 1_000:
        return f"{n / 1_000:.2f}K"
    return str(n)


def _print_model_parameter_counts(model) -> None:
    """
    Assumptions:
    - `model` exposes `.parameters()` yielding torch parameters.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    _xml_tag(
        "model_params",
        total=_format_param_count(total_params),
        trainable=_format_param_count(trainable_params),
        frozen=_format_param_count(frozen_params),
        total_raw=total_params,
        trainable_raw=trainable_params,
        frozen_raw=frozen_params,
    )


def _xml_tag(tag: str, **attrs):
    """
    Assumptions:
    - `tag` is non-empty.
    - `attrs` values are string-renderable.
    """
    rendered = " ".join(f'{k}="{v}"' for k, v in attrs.items())
    print(f"<{tag}{(' ' + rendered) if rendered else ''} />")


def _print_device_backend_info() -> None:
    """
    Assumptions:
    - Torch backend probes are available.
    """
    print(
        "Device availability | "
        f"cuda={torch.cuda.is_available()} "
        f"mps_built={torch.backends.mps.is_built()} "
        f"mps_available={torch.backends.mps.is_available()}"
    )
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"CUDA device[0]: {torch.cuda.get_device_name(0)}")


def _select_device() -> str:
    """
    Assumptions:
    - Exactly one of CUDA / MPS will be available on a given host (CUDA on Linux, MPS on macOS).
    - CPU is always a valid fallback.
    """
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _parse_layers_keep(spec: str) -> tuple[int, int] | None:
    """
    Assumptions:
    - `spec` is either "all" or has format `prefix:suffix` with non-negative integer values.
    """
    assert isinstance(spec, str) and len(spec) > 0, "Expected non-empty layers-keep string."
    if spec.lower() == "all":
        return None
    assert (
        isinstance(spec, str) and ":" in spec
    ), "Expected layers-keep format 'prefix:suffix'."
    parts = spec.split(":")
    assert len(parts) == 2, "Expected exactly one ':' in layers-keep spec."
    prefix_raw, suffix_raw = parts
    assert (
        prefix_raw.isdigit() and suffix_raw.isdigit()
    ), "layers-keep values must be non-negative integers."
    prefix = int(prefix_raw)
    suffix = int(suffix_raw)
    assert prefix > 0 or suffix > 0, "At least one of prefix/suffix must be > 0."
    return prefix, suffix


def _resolve_torch_dtype(dtype_str: str, device: str) -> torch.dtype:
    """
    Assumptions:
    - `dtype_str` is one of "auto", "float16", "bfloat16", "float32".
    - `device` is "cuda", "mps", or "cpu".
    - "auto" picks bf16 on CUDA (Gemma-2 is unstable in fp16 due to attention softcap),
      fp16 on MPS (bf16 support on Apple Silicon is uneven across PyTorch versions),
      and fp32 on CPU.
    """
    if dtype_str == "auto":
        if device == "cuda":
            return torch.bfloat16
        if device == "mps":
            return torch.float16
        return torch.float32
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    assert (
        dtype_str in mapping
    ), f"Unsupported dtype '{dtype_str}'. Choose from auto/{list(mapping.keys())}."
    return mapping[dtype_str]


def _load_model_and_tokenizer(
    model_name: str, layers_keep: str, dtype: torch.dtype, device: str
):
    """
    Assumptions:
    - Model/tokenizer identifiers are valid.
    - Tokenizer can be configured with EOS as PAD token.
    - Model is loaded to CPU first, pruned, then moved to accelerator to avoid fragmentation
      (matters most on MPS but harmless on CUDA).
    - `device` is one of "cuda", "mps", "cpu".
    - We deliberately do NOT pass `device_map` so `hf_device_map` is left unset; this
      avoids stale entries after the manual `.to(device)` move and lets HF Trainer infer
      placement from actual parameter devices.
    """
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        attn_implementation="eager",
    )
    assert hasattr(model, "model") and hasattr(
        model.model, "layers"
    ), "Expected model.model.layers to exist."
    keep_spec = _parse_layers_keep(layers_keep)
    layers = model.model.layers
    num_layers = len(layers)
    assert num_layers > 0, "Expected at least one layer."
    if keep_spec is not None:
        prefix_count, suffix_count = keep_spec
        assert (
            prefix_count + suffix_count <= num_layers
        ), f"Requested prefix+suffix={prefix_count + suffix_count} exceeds num_layers={num_layers}."
        prefix_layers = list(layers[:prefix_count]) if prefix_count > 0 else []
        suffix_layers = list(layers[-suffix_count:]) if suffix_count > 0 else []
        kept = torch.nn.ModuleList(prefix_layers + suffix_layers)
        model.model.layers = None
        del layers
        gc.collect()
        model.model.layers = kept
        gc.collect()

    # Move pruned model to accelerator.
    if device != "cpu":
        model = model.to(device)

    if device == "mps" and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()
    elif device == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

    first_param = next(model.parameters())
    _xml_tag("model_dtype", dtype=first_param.dtype, device=first_param.device)
    _print_model_parameter_counts(model)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    assert (
        tokenizer.pad_token_id is not None
    ), "Expected tokenizer.pad_token_id to be set."
    return model, tokenizer


def _load_dataset(num_samples_train: int, num_samples_eval: int) -> DatasetDict:
    """
    Assumptions:
    - `DATASET_NAME` and `DATASET_CONFIG` are valid.
    - Loaded dataset has `train` and `validation` splits.
    """
    dataset = load_dataset(DATASET_NAME, DATASET_CONFIG)
    assert isinstance(dataset, DatasetDict), "Expected a DatasetDict."
    assert (
        "train" in dataset and "validation" in dataset
    ), "Expected train + validation splits."
    assert (
        len(dataset["train"]) >= num_samples_train
    ), "Train split must have at least num_samples_train."
    assert (
        len(dataset["validation"]) >= num_samples_eval
    ), "Validation split must have at least num_samples_eval."
    assert (
        "question" in dataset["train"].column_names
    ), "Train split missing question column."
    assert (
        "answer" in dataset["train"].column_names
    ), "Train split missing answer column."
    assert (
        "question" in dataset["validation"].column_names
    ), "Validation split missing question column."
    assert (
        "answer" in dataset["validation"].column_names
    ), "Validation split missing answer column."
    print(f"Loaded dataset splits: {list(dataset.keys())}")
    for split_name in dataset.keys():
        print(f"Split '{split_name}' size: {len(dataset[split_name])}")
    return dataset


def _tokenize_batch(
    x: Mapping[str, Sequence[str]],
    tokenizer,
    max_length: int,
    allow_truncation: bool,
    answer_only_loss: bool,
) -> Mapping[str, Sequence]:
    """
    Assumptions:
    - `x` contains aligned `question` and `answer` fields, each of which is a string.
    - Tokenized lengths must be `<= max_length`.
    - If truncation would be required and `allow_truncation` is False, this function fails fast.
    """
    assert (
        "question" in x and "answer" in x
    ), "Batch must contain 'question' and 'answer'."
    assert len(x["question"]) == len(x["answer"]), "Question/answer lengths must match."

    texts = [f"Q: {q}\nA: {a}" for q, a in zip(x["question"], x["answer"])]
    assert len(texts) > 0, "Tokenizer batch should not be empty."
    assert isinstance(texts[0], str), "Tokenizer input should be strings."

    full = tokenizer(
        texts, truncation=False, padding=False, return_attention_mask=False
    )
    assert "input_ids" in full, "Missing input_ids from non-truncated tokenization."
    full_lengths = [len(ids) for ids in full["input_ids"]]
    max_full_len = max(full_lengths)
    if max_full_len > max_length:
        if not allow_truncation:
            raise AssertionError(
                f"Truncation required but disallowed: max_full_len={max_full_len} > max_length={max_length}. "
                "Increase --max-length or pass --allow-truncation."
            )
        warnings.warn(
            f"Truncating batch: max_full_len={max_full_len} > max_length={max_length}.",
            stacklevel=2,
        )

    out = tokenizer(
        texts,
        truncation=allow_truncation,
        max_length=max_length,
        padding=False,
        return_attention_mask=True,
    )
    assert isinstance(out, Mapping), "Tokenizer output should be mapping-like."
    assert "input_ids" in out and "attention_mask" in out, "Missing tokenization keys."
    assert isinstance(
        out["input_ids"], list
    ), "Expected input_ids to be a list in map mode."
    assert len(out["input_ids"]) == len(
        texts
    ), "input_ids batch size must match input texts."

    first_ids = out["input_ids"][0]
    assert isinstance(first_ids, list), "Each input_ids item should be list[int]."
    assert all(isinstance(tok, int) for tok in first_ids), "Token ids must be ints."

    token_lengths = [len(ids) for ids in out["input_ids"]]
    mask_lengths = [len(mask) for mask in out["attention_mask"]]
    assert all(
        tl == ml for tl, ml in zip(token_lengths, mask_lengths)
    ), "Mask/id length mismatch."
    assert all(
        length <= max_length for length in token_lengths
    ), "Found sequence > max_length."
    assert all(
        length > 0 for length in token_lengths
    ), "Found empty tokenized sequence."
    if answer_only_loss:
        prompt_texts = [f"Q: {q}\nA:" for q in x["question"]]
        prompt_out = tokenizer(
            prompt_texts,
            truncation=allow_truncation,
            max_length=max_length,
            padding=False,
            return_attention_mask=False,
        )
        assert "input_ids" in prompt_out, "Missing prompt input_ids for answer-only loss."
        labels: list[list[int]] = []
        for input_ids, prompt_ids in zip(out["input_ids"], prompt_out["input_ids"]):
            prompt_len = min(len(prompt_ids), len(input_ids))
            # Avoid all-masked rows (can produce NaN loss) when prompt consumes full context.
            if prompt_len >= len(input_ids):
                prompt_len = max(0, len(input_ids) - 1)
            label = list(input_ids)
            for i in range(prompt_len):
                label[i] = -100
            labels.append(label)
        out["labels"] = labels
    out["length"] = token_lengths
    return out


def _select_small_splits(
    dataset: DatasetDict, num_samples_train: int, num_samples_eval: int
) -> tuple[Dataset, Dataset, str]:
    """
    Assumptions:
    - `dataset["train"]` and `dataset["validation"]` exist and each has requested sample counts.
    """
    eval_split_name = "validation"
    return (
        dataset["train"].select(range(num_samples_train)),
        dataset[eval_split_name].select(range(num_samples_eval)),
        eval_split_name,
    )


def _tokenize_small_splits(
    raw_train_small: Dataset,
    raw_eval_small: Dataset,
    tokenizer,
    max_length: int,
    allow_truncation: bool,
    answer_only_loss: bool,
) -> tuple[Dataset, Dataset]:
    """
    Assumptions:
    - Raw splits contain `question` and `answer` columns each of which contains strings.
    - Tokenization produces `input_ids` list-of-integer and `length` integer columns.
    - Truncation policy is controlled by `allow_truncation`.
    """
    train_small = raw_train_small.map(
        lambda x: _tokenize_batch(
            x, tokenizer, max_length, allow_truncation, answer_only_loss
        ),
        batched=True,
        remove_columns=["question", "answer"],
    )
    assert (
        "input_ids" in train_small.column_names
    ), "Tokenized train split missing input_ids."
    assert (
        "length" in train_small.column_names
    ), "Tokenized train split missing length column."

    eval_small = raw_eval_small.map(
        lambda x: _tokenize_batch(
            x, tokenizer, max_length, allow_truncation, answer_only_loss
        ),
        batched=True,
        remove_columns=["question", "answer"],
    )
    assert (
        "input_ids" in eval_small.column_names
    ), "Tokenized eval split missing input_ids."
    assert (
        "length" in eval_small.column_names
    ), "Tokenized eval split missing length column."
    return train_small, eval_small


@dataclass
class AnswerOnlyDataCollator:
    tokenizer: object

    def __call__(self, features: list[Mapping[str, object]]) -> Mapping[str, torch.Tensor]:
        labels = [list(f["labels"]) for f in features]
        features_wo_labels = [{k: v for k, v in f.items() if k != "labels"} for f in features]
        batch = DataCollatorWithPadding(tokenizer=self.tokenizer)(features_wo_labels)
        max_len = batch["input_ids"].shape[1]
        padded_labels = [l + ([-100] * (max_len - len(l))) for l in labels]
        batch["labels"] = torch.tensor(padded_labels, dtype=torch.long)
        return batch


def _print_split_length_stats(
    train_small: Dataset, eval_small: Dataset, eval_split_name: str
) -> None:
    """
    Assumptions:
    - Provided split(s) contain a `length` integer column.
    """
    train_lengths = train_small["length"]
    assert (
        isinstance(train_lengths, Sequence) and len(train_lengths) > 0
    ), "Train lengths invalid."
    assert all(
        isinstance(v, int) and v > 0 for v in train_lengths
    ), "Train lengths must be positive ints."
    print(
        f"Train-small size={len(train_small)} | "
        f"min_len={min(train_lengths)} max_len={max(train_lengths)} "
        f"avg_len={sum(train_lengths)/len(train_lengths):.1f}"
    )
    eval_lengths = eval_small["length"]
    assert (
        isinstance(eval_lengths, Sequence) and len(eval_lengths) > 0
    ), "Eval lengths invalid."
    assert all(
        isinstance(v, int) and v > 0 for v in eval_lengths
    ), "Eval lengths must be positive ints."
    print(
        f"Eval-small ({eval_split_name}) size={len(eval_small)} | "
        f"min_len={min(eval_lengths)} max_len={max(eval_lengths)} "
        f"avg_len={sum(eval_lengths)/len(eval_lengths):.1f}"
    )


def _build_training_args(
    batch_size: int,
    grad_accum_steps: int,
    learning_rate: float,
    eval_every: int,
    max_grad_norm: float,
    warmup_steps: int,
    warmup_ratio: float,
    dtype: torch.dtype,
    run_hparams: Mapping[str, object],
    group_by_length: bool,
) -> TrainingArguments:
    """
    Assumptions:
    - Length-aware sampling should be enabled.
    - `dtype` determines fp16/bf16 training flags.
    """
    assert isinstance(batch_size, int) and batch_size > 0, "Expected batch_size > 0."
    assert (
        isinstance(grad_accum_steps, int) and grad_accum_steps > 0
    ), "Expected grad_accum_steps > 0."
    assert learning_rate > 0, "Expected learning_rate > 0."
    assert isinstance(eval_every, int) and eval_every >= 0, "Expected eval_every >= 0."
    assert max_grad_norm >= 0, "Expected max_grad_norm >= 0."
    assert isinstance(warmup_steps, int) and warmup_steps >= 0, "Expected warmup_steps >= 0."
    assert 0.0 <= warmup_ratio <= 1.0, "Expected warmup_ratio in [0, 1]."
    assert not (
        warmup_steps > 0 and warmup_ratio > 0
    ), "Use either warmup_steps or warmup_ratio (or both zero), not both nonzero."
    assert isinstance(run_hparams, Mapping), "Expected run_hparams to be mapping-like."
    os.environ.setdefault("WANDB_PROJECT", "deleteme")
    run_name = _build_run_name(run_hparams)
    return TrainingArguments(
        output_dir=".deleteme_out",
        run_name=run_name,
        num_train_epochs=1,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum_steps,
        learning_rate=learning_rate,
        max_grad_norm=max_grad_norm,
        warmup_steps=warmup_steps,
        warmup_ratio=warmup_ratio,
        eval_strategy="steps" if eval_every > 0 else "epoch",
        eval_steps=eval_every if eval_every > 0 else None,
        fp16=(dtype == torch.float16),
        bf16=(dtype == torch.bfloat16),
        gradient_checkpointing=True,
        # TODO(hadriano) coarser group by length? this is what the model says:
        # HF `group_by_length=True` is not a global full sort. Trainer uses
        # `LengthGroupedSampler`, which first shuffles, then length-sorts inside
        # "mega-batches" (block-wise sorting), so there is built-in noise.
        # If you want an intermediate regime (some padding reduction, more randomness),
        # the practical route is a custom sampler that (1) splits examples into a few
        # coarse buckets by length quantiles (e.g., short/long), then (2) shuffles within
        # each bucket and alternates bucket draws. HF does not expose this as a simple
        # TrainingArguments knob; you implement it by overriding `_get_train_sampler`.
        group_by_length=group_by_length,
        length_column_name="length",
        logging_steps=1,
        report_to=["wandb"],
    )


def _build_run_name(run_hparams: Mapping[str, object]) -> str:
    """
    Assumptions:
    - `run_hparams` is JSON-serializable and stable across identical runs.
    - Returned name should be deterministic and lowercase.
    """
    payload = json.dumps(run_hparams, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:8]
    return f"deleteme_mac_test_{digest}".lower()


class DebugLengthGroupedSampler(LengthGroupedSampler):
    """
    Assumptions:
    - `lengths` is non-empty and aligned to dataset indices.
    - Base sampler ordering logic from HF should remain unchanged.
    """

    def __iter__(self):
        indices = list(super().__iter__())
        _xml_tag(
            "sampler",
            type=self.__class__.__name__,
            total_indices=len(indices),
            batch_size=self.batch_size,
        )
        preview = indices[: min(len(indices), 24)]
        preview_lengths = [self.lengths[i] for i in preview]
        _xml_tag("sampler_preview_indices", values=",".join(str(i) for i in preview))
        _xml_tag(
            "sampler_preview_lengths", values=",".join(str(i) for i in preview_lengths)
        )
        num_batches_to_log = min(
            6, (len(indices) + self.batch_size - 1) // self.batch_size
        )
        for batch_i in range(num_batches_to_log):
            start = batch_i * self.batch_size
            end = min(len(indices), start + self.batch_size)
            batch_indices = indices[start:end]
            batch_lengths = [self.lengths[i] for i in batch_indices]
            _xml_tag(
                "sampler_batch",
                idx=batch_i,
                size=len(batch_indices),
                min_len=min(batch_lengths),
                max_len=max(batch_lengths),
                avg_len=f"{sum(batch_lengths)/len(batch_lengths):.1f}",
            )
        return iter(indices)


class DebugTrainer(Trainer):
    """
    Assumptions:
    - Train dataset is present when training.
    - `length` column exists when `group_by_length=True`.
    """

    def _get_train_sampler(self, train_dataset=None):
        dataset = train_dataset if train_dataset is not None else self.train_dataset
        if dataset is None:
            return None
        if not self.args.group_by_length:
            if train_dataset is not None:
                return super()._get_train_sampler(train_dataset)
            return super()._get_train_sampler()
        # In newer Trainer flows, `dataset` may be a pruned view without non-model columns.
        assert self.train_dataset is not None, "Expected self.train_dataset to exist."
        assert (
            self.args.length_column_name in self.train_dataset.column_names
        ), "Missing length column in original train_dataset."
        lengths = self.train_dataset[self.args.length_column_name]
        assert isinstance(
            lengths, Sequence
        ), "Expected sampler lengths to be sequence-like."
        assert len(lengths) == len(dataset), "Length column size mismatch."
        assert all(
            isinstance(v, int) and v > 0 for v in lengths
        ), "All sampled lengths must be positive ints."
        _xml_tag(
            "sampler_config",
            group_by_length=self.args.group_by_length,
            length_column_name=self.args.length_column_name,
            num_lengths=len(lengths),
        )
        return DebugLengthGroupedSampler(
            batch_size=self.args.train_batch_size, lengths=list(lengths)
        )


@dataclass
class _PreparedSplits:
    train_small: Dataset
    eval_small: Dataset
    eval_split_name: str | None


def _build_trainer(
    model,
    tokenizer,
    args: TrainingArguments,
    prepared_splits: _PreparedSplits,
    answer_only_loss: bool,
) -> DebugTrainer:
    """
    Assumptions:
    - Model, tokenizer, args, and prepared splits are initialized.
    - Dynamic padding collator is desired for Causal LM.
    """
    return DebugTrainer(
        model=model,
        args=args,
        train_dataset=prepared_splits.train_small,
        eval_dataset=prepared_splits.eval_small,
        data_collator=(
            AnswerOnlyDataCollator(tokenizer=tokenizer)
            if answer_only_loss
            else DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
        ),
    )


def _print_trainer_model_device_info(trainer: DebugTrainer, model) -> None:
    """
    Assumptions:
    - `trainer` and `model` are initialized.
    - Model may expose `hf_device_map` or standard parameters.
    """
    print(f"Trainer selected device: {trainer.args.device}")
    if hasattr(model, "hf_device_map"):
        print(f"Model hf_device_map: {model.hf_device_map}")
        return
    try:
        print(f"Model device: {next(model.parameters()).device}")
    except StopIteration:
        print("Model has no parameters to infer device from.")


def _run_training_and_eval(trainer: DebugTrainer, eval_small: Dataset) -> None:
    """
    Assumptions:
    - Trainer has train dataset attached.
    - Eval split is optional.
    """
    print("Starting tiny training run...")
    train_result = trainer.train()
    print(f"Training finished. train_loss={train_result.training_loss}")
    metrics = trainer.evaluate()
    print(f"Eval metrics: {metrics}")


@click.command()
@click.option("--layers-keep", type=str, default=LAYERS_KEEP, show_default=True)
@click.option("--batch-size", type=int, default=BATCH_SIZE, show_default=True)
@click.option(
    "--grad-accum-steps", type=int, default=GRAD_ACCUM_STEPS, show_default=True
)
@click.option("--lr", type=float, default=LEARNING_RATE, show_default=True)
@click.option(
    "--eval-every",
    type=int,
    default=EVAL_EVERY,
    show_default=True,
    help="Evaluate every N steps; 0 means eval each epoch.",
)
@click.option("--max-grad-norm", type=float, default=MAX_GRAD_NORM, show_default=True)
@click.option("--warmup-steps", type=int, default=WARMUP_STEPS, show_default=True)
@click.option("--warmup-ratio", type=float, default=WARMUP_RATIO, show_default=True)
@click.option(
    "--num-samples-train", type=int, default=NUM_SAMPLES_TRAIN, show_default=True
)
@click.option(
    "--num-samples-eval", type=int, default=NUM_SAMPLES_EVAL, show_default=True
)
@click.option("--max-length", type=int, default=MAX_LENGTH, show_default=True)
@click.option(
    "--allow-truncation/--no-allow-truncation", default=False, show_default=True
)
@click.option(
    "--answer-only-loss/--no-answer-only-loss",
    default=ANSWER_ONLY_LOSS,
    show_default=True,
)
@click.option("--model-name", type=str, default=MODEL_NAME, show_default=True)
@click.option(
    "--dtype",
    type=click.Choice(["auto", "float16", "bfloat16", "float32"]),
    default="auto",
    show_default=True,
)
@click.option("--group-by-length/--no-group-by-length", default=False, show_default=True)
def main(
    layers_keep: str,
    batch_size: int,
    grad_accum_steps: int,
    lr: float,
    eval_every: int,
    max_grad_norm: float,
    warmup_steps: int,
    warmup_ratio: float,
    num_samples_train: int,
    num_samples_eval: int,
    max_length: int,
    allow_truncation: bool,
    answer_only_loss: bool,
    model_name: str,
    dtype: str,
    group_by_length: bool,
) -> None:
    """
    Assumptions:
    - Dependencies are installed.
    - Model and dataset are reachable.
    - Output directory `.deleteme_out` is writable.
    """
    assert max_length > 0, "Expected max_length > 0."
    assert lr > 0, "Expected lr > 0."
    assert eval_every >= 0, "Expected eval_every >= 0."
    assert max_grad_norm >= 0, "Expected max_grad_norm >= 0."
    assert warmup_steps >= 0, "Expected warmup_steps >= 0."
    assert 0.0 <= warmup_ratio <= 1.0, "Expected warmup_ratio in [0, 1]."
    assert not (
        warmup_steps > 0 and warmup_ratio > 0
    ), "Use either warmup_steps or warmup_ratio (or both zero), not both nonzero."
    assert num_samples_train > 0, "Expected num_samples_train > 0."
    assert num_samples_eval > 0, "Expected num_samples_eval > 0."
    assert (
        isinstance(model_name, str) and len(model_name) > 0
    ), "Expected non-empty model_name."
    device = _select_device()
    torch_dtype = _resolve_torch_dtype(dtype, device)
    _xml_tag(
        "cli_config",
        layers_keep=layers_keep,
        batch_size=batch_size,
        grad_accum_steps=grad_accum_steps,
        lr=lr,
        eval_every=eval_every,
        max_grad_norm=max_grad_norm,
        warmup_steps=warmup_steps,
        warmup_ratio=warmup_ratio,
        num_samples_train=num_samples_train,
        num_samples_eval=num_samples_eval,
        max_length=max_length,
        allow_truncation=allow_truncation,
        answer_only_loss=answer_only_loss,
        model_name=model_name,
        dtype=dtype,
        resolved_dtype=str(torch_dtype),
        device=device,
    )
    _print_device_backend_info()
    model, tokenizer = _load_model_and_tokenizer(
        model_name, layers_keep, torch_dtype, device
    )
    dataset = _load_dataset(num_samples_train, num_samples_eval)
    raw_train_small, raw_eval_small, eval_split_name = _select_small_splits(
        dataset, num_samples_train, num_samples_eval
    )
    train_small, eval_small = _tokenize_small_splits(
        raw_train_small,
        raw_eval_small,
        tokenizer,
        max_length,
        allow_truncation,
        answer_only_loss,
    )
    _print_split_length_stats(train_small, eval_small, eval_split_name)
    run_hparams: dict[str, object] = {
        "layers_keep": layers_keep,
        "batch_size": batch_size,
        "grad_accum_steps": grad_accum_steps,
        "learning_rate": lr,
        "eval_every": eval_every,
        "max_grad_norm": max_grad_norm,
        "warmup_steps": warmup_steps,
        "warmup_ratio": warmup_ratio,
        "num_samples_train": num_samples_train,
        "num_samples_eval": num_samples_eval,
        "max_length": max_length,
        "allow_truncation": allow_truncation,
        "answer_only_loss": answer_only_loss,
        "model_name": model_name,
        "dtype": dtype,
        "resolved_dtype": str(torch_dtype),
        "device": device,
        "dataset_name": DATASET_NAME,
        "dataset_config": DATASET_CONFIG,
        "group_by_length": group_by_length,
    }
    args = _build_training_args(
        batch_size,
        grad_accum_steps,
        lr,
        eval_every,
        max_grad_norm,
        warmup_steps,
        warmup_ratio,
        torch_dtype,
        run_hparams=run_hparams,
        group_by_length=group_by_length,
    )
    prepared_splits = _PreparedSplits(
        train_small=train_small,
        eval_small=eval_small,
        eval_split_name=eval_split_name,
    )
    trainer = _build_trainer(
        model,
        tokenizer,
        args,
        prepared_splits,
        answer_only_loss=answer_only_loss,
    )
    _print_trainer_model_device_info(trainer, model)
    _run_training_and_eval(trainer, eval_small)


if __name__ == "__main__":
    main()

