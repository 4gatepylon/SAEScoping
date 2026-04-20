from __future__ import annotations
from typing import Any, Optional
from pathlib import Path
import io
import orjson
import json
import torch
import wandb
import pandas as pd
from transformers import TrainerCallback
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from beartype import beartype
from sae_scoping.evaluation.scoping_eval import OneClickLLMJudgeScopingEval


class LLMJudgeScopingTrainerCallback(TrainerCallback):
    """
    TrainerCallback that runs OneClickLLMJudgeScopingEval during training and
    logs results to W&B. Mirrors LLMJudgeSpylabBio1ClickTrainerCallback but
    uses the domain-based scoping evaluator (no trojans).
    """

    @beartype
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        domain_questions: dict[str, list[str]],
        llm_judge_every: int = 100,
        n_max_openai_requests: int = 1_000,
        model_name: str = "unknown",
        run_name: str = "unknown",
        csv_dir: Optional[Path] = None,
        train_domain: Optional[str] = None,
        attack_domain: Optional[str] = None,
        domain_answers: Optional[dict[str, list[str]]] = None,
        reference_score_paths: Optional[dict[str, Path]] = None,
    ):
        self.tokenizer = tokenizer
        self.domain_questions = domain_questions
        self.domain_answers = domain_answers
        self.llm_judge_every = llm_judge_every
        self.n_max_openai_requests = n_max_openai_requests
        self.model_name = model_name
        self.run_name = run_name
        self.csv_dir = csv_dir
        self._step_scores: list[dict[str, float]] = []
        self._step_dfs: list[pd.DataFrame] = []
        self._current_step: int = -1
        self._call_index: int = 0
        self.n_eval_datasets: int = len(domain_questions)
        self.n_eval_runs: int = 1
        self.evaluator = OneClickLLMJudgeScopingEval(
            n_max_openai_requests=200_000,
            train_domain=train_domain,
            attack_domain=attack_domain,
        )
        # History for grouped line-series charts (one chart per judge type).
        self._eval_steps: list[int] = []
        self._score_history: dict[str, list[float]] = {}
        # Paths to pre-computed baseline score JSON files (label → path).
        self._reference_score_paths: dict[str, Path] = reference_score_paths or {}
        self._reference_scores_cache: dict[str, dict[str, float]] = {}

    def _load_reference_scores(self) -> dict[str, dict[str, float]]:
        """Read each reference scores.json file on first use; cache results."""
        for label, path in self._reference_score_paths.items():
            if label not in self._reference_scores_cache and path.exists():
                try:
                    self._reference_scores_cache[label] = json.loads(path.read_text())
                except Exception as e:
                    print(f"Warning: could not load reference scores from {path}: {e}")
        return self._reference_scores_cache

    def _log_grouped_charts(self, step: int) -> None:
        """Log one wandb line_series chart per judge type, with all domains as lines."""
        from collections import defaultdict
        groups: dict[str, list[str]] = defaultdict(list)
        for k in sorted(self._score_history.keys()):
            judge_type = k.split("/")[-1]
            groups[judge_type].append(k)
        for judge_type, keys in sorted(groups.items()):
            xs = [self._eval_steps[:] for _ in keys]
            ys = [self._score_history[k][:] for k in keys]
            labels = ["/".join(k.split("/")[1:3]) for k in keys]  # e.g. "biology/in_scope"
            chart = wandb.plot.line_series(
                xs=xs, ys=ys, keys=labels,
                title=f"LLM Judge: {judge_type}",
                xname="Training Step",
            )
            wandb.log({f"charts/llm_judge_{judge_type}": chart, "trainer/global_step": step})

        ref_scores = self._load_reference_scores()
        for label, ref in ref_scores.items():
            for judge_type, keys in sorted(groups.items()):
                xs = [self._eval_steps[:] for _ in keys]
                ys = [
                    [v - ref[k] for v in self._score_history[k]]
                    for k in keys
                    if k in ref
                ]
                labels = ["/".join(k.split("/")[1:3]) for k in keys if k in ref]
                if not ys:
                    continue
                chart = wandb.plot.line_series(
                    xs=xs[:len(ys)], ys=ys, keys=labels,
                    title=f"LLM Judge diff vs {label}: {judge_type}",
                    xname="Training Step",
                )
                wandb.log({f"charts/llm_judge_diff_{label}_{judge_type}": chart, "trainer/global_step": step})

    def on_evaluate(self, args, state, control, model, metrics=None, **kwargs):
        if state.global_step % self.llm_judge_every != 0:
            return
        if metrics is None:
            print("WARNING: METRICS IS NONE; LLMJudgeScopingTrainerCallback will do nothing")
            return

        # Reset accumulators when we enter a new step
        if state.global_step != self._current_step:
            self._current_step = state.global_step
            self._step_scores = []
            self._step_dfs = []
            self._call_index = 0

        self._call_index += 1
        call_idx = self._call_index

        if call_idx > self.n_eval_runs:
            return

        print("@" * 80)
        print(f"Running scoping LLM judge evaluation at step {state.global_step} (run {call_idx}/{self.n_eval_runs})...")
        with torch.no_grad():
            scores, df_as_json = self.evaluator.evaluate(
                model,
                self.tokenizer,
                self.domain_questions,
                n_max_openai_requests=self.n_max_openai_requests,
                domain_answers=self.domain_answers,
            )
        self._step_scores.append(scores)
        df = pd.read_json(io.StringIO(df_as_json), orient="records")
        self._step_dfs.append(df)

        print(json.dumps({k: v for k, v in scores.items()}))

        # Save per-run CSV
        if self.csv_dir is not None:
            self.csv_dir.mkdir(parents=True, exist_ok=True)
            csv_path = self.csv_dir / f"llm_judge_step_{state.global_step}_run{call_idx}.csv"
            df.to_csv(csv_path, index=False)
            print(f"Saved judgements CSV to {csv_path}")

        # Log individual run to W&B (only meaningful when there are multiple runs)
        if wandb.run is not None and self.n_eval_runs > 1:
            wandb.log({**{f"{k}_run{call_idx}": v for k, v in scores.items()}, "trainer/global_step": state.global_step})

        # After all runs: finalize scores, log charts
        if call_idx == self.n_eval_runs:
            if self.n_eval_runs > 1:
                avg_scores = {
                    k: float(sum(s[k] for s in self._step_scores) / len(self._step_scores))
                    for k in self._step_scores[0]
                }
                print("Averaged scores:")
                print(json.dumps(avg_scores))

                if self.csv_dir is not None:
                    all_df = pd.concat(self._step_dfs, ignore_index=True)
                    avg_csv_path = self.csv_dir / f"llm_judge_step_{state.global_step}_avg.csv"
                    all_df.to_csv(avg_csv_path, index=False)
                    print(f"Saved averaged judgements CSV to {avg_csv_path}")

                if wandb.run is not None:
                    wandb.log({**{f"{k}_avg": v for k, v in avg_scores.items()}, "trainer/global_step": state.global_step})
                metrics.update({f"{k}_avg": v for k, v in avg_scores.items()})
            else:
                avg_scores = self._step_scores[0]
                if wandb.run is not None:
                    wandb.log({**avg_scores, "trainer/global_step": state.global_step})
                metrics.update(avg_scores)

            if wandb.run is not None:
                if self._step_dfs:
                    all_df = pd.concat(self._step_dfs, ignore_index=True)
                    wandb.log({"llm_judge/judgements": wandb.Table(dataframe=all_df), "trainer/global_step": state.global_step})
                self._eval_steps.append(state.global_step)
                for k, v in avg_scores.items():
                    self._score_history.setdefault(k, []).append(v)
                self._log_grouped_charts(state.global_step)

                # Diff scalars: one set per reference baseline.
                for label, ref in self._load_reference_scores().items():
                    diffs = {
                        k.replace("llm_judge/", f"llm_judge_diff_{label}/"): v - ref[k]
                        for k, v in avg_scores.items()
                        if k in ref
                    }
                    if diffs:
                        wandb.log({**diffs, "trainer/global_step": state.global_step})

        print("@" * 80)
