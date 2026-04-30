"""
Full SAE scoping pipeline for Gemma-3-12b-it using coding datasets.
Supports rank, recover, and attack stages with HuggingFace Hub integration.

Uses SAEs from google/gemma-scope-2-12b-it loaded directly from HuggingFace.

Stages:
  1. RANK:     Compute firing rates on the chosen train domain (usually coding)
  2. PRUNE:    Prune SAE neurons with firing rate < threshold
  3. RECOVER:  In-domain SFT on the train domain
  4. ATTACK:   Adversarial SFT on an OOD domain

Usage:
  # Full pipeline, coding domain, gemma3-12b:
  python script_scoping_pipeline_gemma3_12b_coding.py --stage all --attack-domain math

  # Just recovery training:
  python script_scoping_pipeline_gemma3_12b_coding.py --stage recover
"""

from __future__ import annotations

import gc
import io
import json
import shutil
from pathlib import Path
import time

import click
import pandas as pd
import torch
import wandb
from huggingface_hub import HfApi, snapshot_download, hf_hub_download
from datasets import Dataset, load_dataset, concatenate_datasets
from safetensors.torch import load_file, save_file
from sae_lens import SAE, JumpReLUSAE
from sae_lens.saes.jumprelu_sae import JumpReLUSAEConfig
from sae_lens.saes.sae import SAEMetadata
from transformers import AutoTokenizer, PreTrainedTokenizerBase, TrainerCallback, TrainerControl, TrainerState
from transformers import Gemma3ForCausalLM
from transformers.training_args import TrainingArguments
from trl import SFTConfig
import tqdm
import sys
import os
sys.path.append(os.path.abspath(".."))
from functools import partial
from sae_scoping.trainers.sae_enhanced.prune import get_pruned_sae
from sae_scoping.trainers.sae_enhanced.rank import rank_neurons
from sae_scoping.trainers.sae_enhanced.train import train_sae_enhanced_model
from sae_scoping.utils.hooks.pt_hooks import filter_hook_fn, named_forward_hooks
from sae_scoping.utils.hooks.sae import SAEWrapper
from sae_scoping.xxx_evaluation.scoping_eval import OneClickLLMJudgeScopingEval
from sae_scoping.xxx_evaluation.trainer_callbacks import LLMJudgeScopingTrainerCallback


class _HfCheckpointCallback(TrainerCallback):
    """Captures the wandb run ID, creates HF repo, and uploads each checkpoint."""
    def __init__(self):
        self.run_id: str | None = None
        self._api: HfApi | None = None
        self.failed_checkpoints: list[Path] = []

    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if wandb.run is not None:
            if self._api is None: self._api = HfApi()
            repo_url = self._api.create_repo(repo_id=wandb.run.id, exist_ok=True, repo_type="model")
            self.run_id = repo_url.repo_id

    def on_save(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if self.run_id is None: return
        ckpt_dir = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        if not ckpt_dir.exists(): return
        if self._api is None: self._api = HfApi()
        print(f"Uploading {ckpt_dir.name} to HF {self.run_id}...")
        try:
            self._api.upload_folder(folder_path=str(ckpt_dir), repo_id=self.run_id, path_in_repo=ckpt_dir.name, repo_type="model")
            print(f"Upload successful, deleting local {ckpt_dir.name}...")
            shutil.rmtree(ckpt_dir)
        except Exception as e:
            print(f"Upload failed: {e}")
            self.failed_checkpoints.append(ckpt_dir)

    def retry_failed_checkpoints(self) -> None:
        if not self.failed_checkpoints or self.run_id is None: return
        if self._api is None: self._api = HfApi()
        still_failed = []
        for ckpt_dir in self.failed_checkpoints:
            if not ckpt_dir.exists(): continue
            print(f"Retrying upload of {ckpt_dir.name} to HF {self.run_id}...")
            try:
                self._api.upload_folder(folder_path=str(ckpt_dir), repo_id=self.run_id, path_in_repo=ckpt_dir.name, repo_type="model")
                print(f"Retry successful, deleting local {ckpt_dir.name}...")
                shutil.rmtree(ckpt_dir)
            except Exception as e:
                still_failed.append(ckpt_dir); print(f"Retry failed for {ckpt_dir.name}: {e}")
        if still_failed:
            sys.exit(f"ERROR: {len(still_failed)} checkpoint(s) could not be uploaded after retry: {still_failed}")

# ── Model configs ─────────────────────────────────────────────────────────────
MODEL_NAME = "google/gemma-3-12b-it"
SAE_HF_REPO = "google/gemma-scope-2-12b-it"
SAE_HF_FOLDER = "resid_post/layer_31_width_16k_l0_medium"
HOOKPOINT = "model.layers.31"
FIRING_RATE_THRESHOLD = 3e-4

ALL_DOMAINS = ["coding", "biology", "chemistry", "math", "physics"]
STEMQA_DOMAINS = {"biology", "chemistry", "math", "physics"}

def load_gemma_scope_2_sae(repo_id: str, folder_name: str, device: str | torch.device = "cpu") -> JumpReLUSAE:
    device = str(device)
    cfg_path = hf_hub_download(repo_id, f"{folder_name}/config.json")
    with open(cfg_path) as f: hf_config = json.load(f)
    weights_path = hf_hub_download(repo_id, f"{folder_name}/params.safetensors")
    raw_state_dict = load_file(weights_path, device=device)
    d_in, d_sae = raw_state_dict["w_enc"].shape
    sae_cfg = JumpReLUSAEConfig(d_in=d_in, d_sae=d_sae, dtype="float32", device=device, apply_b_dec_to_input=False, normalize_activations="none", metadata=SAEMetadata(model_name="gemma-3-12b"))
    sae = JumpReLUSAE(sae_cfg)
    sae.load_state_dict({"W_enc": raw_state_dict["w_enc"].to(torch.float32), "W_dec": raw_state_dict["w_dec"].to(torch.float32), "b_enc": raw_state_dict["b_enc"].to(torch.float32), "b_dec": raw_state_dict["b_dec"].to(torch.float32), "threshold": raw_state_dict["threshold"].to(torch.float32)})
    return sae.to(device)

def _stream_qa_dataset(dataset_name: str, config: str, split: str, n_samples: int, tokenizer: PreTrainedTokenizerBase, question_col="question", answer_col="answer", seed=1, stream_flag=True) -> Dataset:
    stream = load_dataset(dataset_name, config, split=split, streaming=stream_flag)
    if stream_flag: stream = stream.shuffle(seed=seed, buffer_size=50)
    rows = []
    for i, ex in tqdm.tqdm(enumerate(stream), total=n_samples):
        if i >= n_samples: break
        text = tokenizer.apply_chat_template([{"role": "user", "content": str(ex[question_col])}, {"role": "assistant", "content": str(ex[answer_col])}], tokenize=False, add_generation_prompt=False)
        rows.append({"text": text, "question": str(ex[question_col]), "answer": str(ex[answer_col])})
    return Dataset.from_list(rows)

def load_coding_train_eval(tokenizer: PreTrainedTokenizerBase, eval_fraction=0.2, seed=42, n_samples=60_000) -> tuple[Dataset, Dataset]:
    stream = load_dataset("nvidia/OpenCodeReasoning", "split_0", split="split_0", streaming=True)
    stream = stream.shuffle(seed=seed, buffer_size=1000)
    rows = []
    for ex in tqdm.tqdm(stream, desc="Filtering OpenCodeReasoning"):
        if "HARD" in str(ex.get("difficulty", "")).upper(): continue
        q = ex.get("input")
        if not q or q == "-": continue
        text = tokenizer.apply_chat_template([{"role": "user", "content": q}, {"role": "assistant", "content": ex.get("output", "")}], tokenize=False, add_generation_prompt=False)
        rows.append({"text": text, "question": q, "answer": ex.get("output", "")})
        if len(rows) >= n_samples: break
    full = Dataset.from_list(rows).shuffle(seed=seed)
    n_eval = int(len(full) * eval_fraction)
    return full.select(range(n_eval, len(full))), full.select(range(n_eval))

def load_domain_train_eval(domain: str, tokenizer: PreTrainedTokenizerBase, eval_fraction=0.2, seed=42) -> tuple[Dataset, Dataset]:
    if domain == "coding": return load_coding_train_eval(tokenizer, eval_fraction, seed)
    elif domain in STEMQA_DOMAINS: full = _stream_qa_dataset("4gate/StemQAMixture", domain, "train", 50_000, tokenizer, stream_flag=False)
    else: raise ValueError(f"Unknown domain {domain}")
    full = full.shuffle(seed=seed)
    n_eval = int(len(full) * eval_fraction)
    return full.select(range(n_eval, len(full))), full.select(range(n_eval))

def stage_rank(train_dataset: Dataset, n_samples: int, batch_size: int, tokenizer, model, device, cache_dir) -> tuple[torch.Tensor, torch.Tensor]:
    cache_path = cache_dir / "firing_rates.safetensors"
    if cache_path.exists():
        data = load_file(str(cache_path)); return data["ranking"], data["distribution"]
    dataset = train_dataset.select(range(min(n_samples, len(train_dataset))))
    sae = load_gemma_scope_2_sae(SAE_HF_REPO, SAE_HF_FOLDER, device=device)
    ranking, distribution = rank_neurons(dataset=dataset, sae=sae, model=model, tokenizer=tokenizer, T=0, hookpoint=HOOKPOINT, batch_size=batch_size, token_selection="attention_mask", return_distribution=True)
    cache_dir.mkdir(parents=True, exist_ok=True); save_file({"ranking": ranking.cpu(), "distribution": distribution.cpu()}, str(cache_path))
    del sae; gc.collect(); torch.cuda.empty_cache()
    return ranking.cpu(), distribution.cpu()

def stage_prune(distribution: torch.Tensor, ranking: torch.Tensor, device, firing_rate_threshold=FIRING_RATE_THRESHOLD):
    n_kept = int((distribution >= firing_rate_threshold).sum().item())
    neuron_ranking = torch.argsort(distribution, descending=True)
    sae = load_gemma_scope_2_sae(SAE_HF_REPO, SAE_HF_FOLDER, device=device)
    pruned_sae = get_pruned_sae(sae, neuron_ranking, K_or_p=n_kept, T=0.0)
    return pruned_sae.to(device), sae, n_kept

def stage_train(train_dataset, eval_datasets, pruned_sae, model, tokenizer, output_dir, wandb_project, wandb_run, max_steps, batch_size, accum, save_every, training_callbacks=None, all_layers_after_hookpoint=False, resume_from_checkpoint=None):
    sft_config = SFTConfig(output_dir=output_dir, per_device_train_batch_size=batch_size, per_device_eval_batch_size=batch_size, max_steps=max_steps, resume_from_checkpoint=resume_from_checkpoint, packing=False, gradient_accumulation_steps=accum, eval_accumulation_steps=accum, num_train_epochs=1, learning_rate=2e-5, warmup_ratio=0.1, weight_decay=0.1, max_grad_norm=1.0, logging_steps=10, eval_strategy="steps", eval_steps=100, save_steps=save_every, bf16=True, save_total_limit=5, report_to="wandb", max_length=1024, gradient_checkpointing=True)
    train_sae_enhanced_model(train_dataset=train_dataset, eval_dataset=eval_datasets, sae=pruned_sae, model=model, tokenizer=tokenizer, T=0.0, hookpoint=HOOKPOINT, all_layers_after_hookpoint=all_layers_after_hookpoint, sft_config=sft_config, wandb_project_name=wandb_project, wandb_run_name=wandb_run, training_callbacks=training_callbacks or [])

def run_baseline_eval(model, tokenizer, domain_questions, train_domain, wandb_project, wandb_run, csv_path, metric_prefix, n_max_openai_requests=1000, attack_domain=None, pruned_sae=None, chart_suffix=None, domain_answers=None):
    evaluator = OneClickLLMJudgeScopingEval(n_max_openai_requests=200000, train_domain=train_domain, attack_domain=attack_domain)
    scores_path = csv_path.with_suffix(".scores.json")
    if csv_path.exists():
        df = pd.read_csv(csv_path); scores = evaluator._extract_scores(df, {d: qs[:evaluator.n_samples] for d, qs in domain_questions.items()})
        if not scores_path.exists(): scores_path.write_text(json.dumps(scores, indent=2))
    else:
        print(f"Baseline eval ({wandb_run})..."); hook_dict = {HOOKPOINT: partial(filter_hook_fn, SAEWrapper(pruned_sae))} if pruned_sae is not None else {}
        with torch.no_grad(), named_forward_hooks(model, hook_dict): scores, df_as_json = evaluator.evaluate(model, tokenizer, domain_questions, n_max_openai_requests=n_max_openai_requests, domain_answers=domain_answers)
        csv_path.parent.mkdir(parents=True, exist_ok=True); pd.read_json(io.StringIO(df_as_json), orient="records").to_csv(csv_path, index=False)
        scores_path.write_text(json.dumps(scores, indent=2))
    if wandb.run is None: wandb.init(project=wandb_project, name=wandb_run, resume="allow")
    wandb.log({f"{metric_prefix}/{k}": v for k, v in scores.items()} | {"trainer/global_step": 0})
    if chart_suffix: wandb.log({f"{k}_{chart_suffix}": v for k, v in scores.items()} | {"trainer/global_step": 0})

@click.command()
@click.option("--train-domain", type=click.Choice(ALL_DOMAINS), default="coding")
@click.option("--attack-domain", type=click.Choice(ALL_DOMAINS), default=None)
@click.option("--stage", type=str, default="all")
@click.option("--n-rank-samples", type=int, default=10000)
@click.option("--batch-size", "-b", type=int, default=4)
@click.option("--accum", "-a", type=int, default=16)
@click.option("--max-steps-recover", type=int, default=3000)
@click.option("--max-steps-attack", type=int, default=4000)
@click.option("--save-every", type=int, default=500)
@click.option("--firing-rate-threshold", type=float, default=FIRING_RATE_THRESHOLD)
@click.option("--output-dir", type=str, default=None)
@click.option("--checkpoint", type=str, default=None)
@click.option("--hf-recover-repo", type=str, default=None)
@click.option("--hf-attack-repo", type=str, default=None)
@click.option("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
@click.option("--dev/--prod-mode", "dev", default=True)
def main(train_domain, attack_domain, stage, n_rank_samples, batch_size, accum, max_steps_recover, max_steps_attack, save_every, firing_rate_threshold, output_dir, checkpoint, hf_recover_repo, hf_attack_repo, device, dev):
    device = torch.device(device)
    base_dir = Path(__file__).parent
    model_slug = MODEL_NAME.replace("/", "--")
    cache_tag = "layer_31--width_16k--l0_medium"
    stages = {s.strip() for s in stage.split(",")}
    if "all" in stages:
        stages = {"rank", "recover", "attack"}
    if "attack" in stages and attack_domain is None:
        raise click.UsageError("--attack-domain required for attack stage.")
    cache_dir = base_dir / ".cache" / "coding_gemma3_12b" / "ignore_padding_True" / cache_tag
    _dist_cache_path = cache_dir / "firing_rates.safetensors"
    if _dist_cache_path.exists():
        _pre = load_file(str(_dist_cache_path))
        n_kept = int((_pre["distribution"] >= firing_rate_threshold).sum().item())
        output_base = Path(output_dir) if output_dir else (base_dir / "outputs_scoping" / model_slug / cache_tag / train_domain / f"h{firing_rate_threshold}" / f"k{n_kept}")
    else:
        output_base = Path(output_dir) if output_dir else base_dir / "outputs_scoping" / model_slug / cache_tag / train_domain
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    attack_resume = None
    if "recover" in stages or "rank" in stages:
        model_path = checkpoint if checkpoint else MODEL_NAME
    elif "attack" in stages:
        recover_final = output_base / "recover" / "final"
        if checkpoint and checkpoint.isdigit():
            if hf_attack_repo is None: raise click.UsageError("--hf-attack-repo required for step checkpoint.")
            local_dir = snapshot_download(hf_attack_repo, allow_patterns=[f"checkpoint-{checkpoint}/*"])
            model_path = str(Path(local_dir) / f"checkpoint-{checkpoint}"); attack_resume = model_path
        elif checkpoint: model_path = checkpoint
        elif hf_recover_repo: model_path = hf_recover_repo
        elif recover_final.exists(): model_path = str(recover_final)
        else: raise click.UsageError(f"No recover checkpoint found at {recover_final}. Run with --stage recover first, or include 'recover' in your --stage list.")
    else: model_path = checkpoint if checkpoint else MODEL_NAME

    print(f"Loading {model_path}...")
    model = Gemma3ForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
        attn_implementation="eager"
    )
    model.gradient_checkpointing_disable()
    all_domain_splits = {d: load_domain_train_eval(d, tokenizer) for d in ALL_DOMAINS}
    if "rank" in stages: ranking, distribution = stage_rank(all_domain_splits[train_domain][0], n_rank_samples, batch_size, tokenizer, model, device, cache_dir)
    else: data = load_file(str(_dist_cache_path)); ranking, distribution = data["ranking"], data["distribution"]
    n_kept = int((distribution >= firing_rate_threshold).sum().item()); output_base = Path(output_dir) if output_dir else (base_dir / "outputs_scoping" / model_slug / cache_tag / train_domain / f"h{firing_rate_threshold}" / f"k{n_kept}")
    eval_datasets = {d: ev.select(range(min(500 if dev else 100000, len(ev)))) for d, (_, ev) in all_domain_splits.items()}
    domain_questions = {d: ds["question"] for d, ds in eval_datasets.items()}
    domain_answers = {d: ds["answer"] for d, ds in eval_datasets.items()}
    shared_eval_dir = base_dir / "outputs_scoping" / model_slug / cache_tag / train_domain / ("dev_llm_judge_csvs" if dev else "llm_judge_csvs")
    rec_run = f"recover/{model_slug}/{cache_tag}/{train_domain}/h{firing_rate_threshold}/k{n_kept}"
    if "recover" in stages or "attack" in stages: run_baseline_eval(model, tokenizer, domain_questions, train_domain, f"sae-scoping-gemma3-{train_domain}", rec_run, shared_eval_dir / "baseline_true.csv", "true_baseline", n_max_openai_requests=1_800, chart_suffix="pre_scoping", domain_answers=domain_answers)
    pruned_sae, sae, n_kept = stage_prune(distribution, ranking, device, firing_rate_threshold)
    rec_cb = LLMJudgeScopingTrainerCallback(tokenizer, domain_questions, 500, 1000, MODEL_NAME, rec_run, output_base / "llm_judge_csvs", train_domain, domain_answers=domain_answers, reference_score_paths={"baseline": shared_eval_dir / "baseline_true.scores.json", "pre_recover": output_base / "llm_judge_csvs" / "baseline_pre_recover.scores.json"})
    if "recover" in stages:
        run_baseline_eval(model, tokenizer, domain_questions, train_domain, f"sae-scoping-gemma3-{train_domain}", rec_run, output_base / "llm_judge_csvs" / "baseline_pre_recover.csv", "pre-recover-baseline", n_max_openai_requests=1_800, pruned_sae=pruned_sae, chart_suffix="post_scoping", domain_answers=domain_answers)
        hf_cb = _HfCheckpointCallback(); stage_train(all_domain_splits[train_domain][0], eval_datasets, pruned_sae, model, tokenizer, str(output_base / "recover"), f"sae-scoping-gemma3-{train_domain}", rec_run, max_steps_recover, batch_size, accum, save_every, [rec_cb, hf_cb])
        hf_cb.retry_failed_checkpoints()
        model.save_pretrained(str(output_base / "recover" / "final")); tokenizer.save_pretrained(str(output_base / "recover" / "final"))
        if hf_cb.run_id:
            try: model.push_to_hub(hf_cb.run_id); tokenizer.push_to_hub(hf_cb.run_id); shutil.rmtree(output_base / "recover")
            except Exception as e: print(f"HF upload failed: {e}")
    if "attack" in stages:
        if wandb.run is not None: wandb.finish()
        atk_run = f"attack/{model_slug}/{cache_tag}/{train_domain}/h{firing_rate_threshold}/k{n_kept}/{attack_domain}"
        wandb.init(project=f"sae-scoping-gemma3-{train_domain}", name=atk_run, resume="allow", settings=wandb.Settings(init_timeout=180))
        atk_dq = {attack_domain: domain_questions[attack_domain]}
        atk_da = {attack_domain: domain_answers[attack_domain]}
        atk_ev = {attack_domain: eval_datasets[attack_domain]}
        atk_cb = LLMJudgeScopingTrainerCallback(tokenizer, atk_dq, 500, 1000, MODEL_NAME, atk_run, output_base / "llm_judge_csvs" / attack_domain, train_domain, attack_domain, domain_answers=atk_da, reference_score_paths={"baseline": shared_eval_dir / "baseline_true.scores.json", "pre_attack": output_base / "llm_judge_csvs" / attack_domain / "baseline_pre_attack.scores.json"})
        run_baseline_eval(model, tokenizer, domain_questions, train_domain, f"sae-scoping-gemma3-{train_domain}", atk_run, output_base / "llm_judge_csvs" / attack_domain / "baseline_pre_attack.csv", "pre-attack-baseline", n_max_openai_requests=1_800, attack_domain=attack_domain, pruned_sae=pruned_sae, chart_suffix="pre_attack", domain_answers=domain_answers)
        hf_cb = _HfCheckpointCallback(); stage_train(all_domain_splits[attack_domain][0], atk_ev, pruned_sae, model, tokenizer, str(output_base / "attack" / attack_domain), f"sae-scoping-gemma3-{train_domain}", atk_run, max_steps_attack, batch_size, accum, save_every, [atk_cb, hf_cb], True, attack_resume)
        hf_cb.retry_failed_checkpoints()
        model.save_pretrained(str(output_base / "attack" / attack_domain / "final")); tokenizer.save_pretrained(str(output_base / "attack" / attack_domain / "final"))
        if hf_cb.run_id:
            try: model.push_to_hub(hf_cb.run_id); tokenizer.push_to_hub(hf_cb.run_id); shutil.rmtree(output_base / "attack" / attack_domain)
            except Exception as e: print(f"HF upload failed: {e}")
    print("\nPipeline complete!")

if __name__ == "__main__": main()
