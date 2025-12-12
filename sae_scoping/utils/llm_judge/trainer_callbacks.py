from __future__ import annotations
from typing import Any, Optional
from pathlib import Path
import orjson
import json
import torch
import wandb
from transformers import TrainerCallback
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from beartype import beartype
from utils.llm_judge.spylab_1click_judgement import (
    OneClickLLMJudgeEvaluationETHZ1Biology,
)


# XXX clean this up a lot plz
class LLMJudgeSpylabBio1ClickTrainerCallback(TrainerCallback):
    """
    This callback adds custom metrics-update logic (what you see on CLI or WanDB or
    whatever your logging strategy is) to evaluate the model using the LLM judge in such
    a way that we can get more realistic sense of whether the LLM is good.

    Mostly a refactor by claude.
    """

    @beartype
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        llm_judge_every: int = 100,
        save_full_info: bool = False,
        save_full_info_mode: str = "wandb",
        model_name: str = "unknown",
        run_name: str = "unknown",
        full_info_folder: Optional[Path] = None,
    ):
        self.llm_judge_every = llm_judge_every
        self.save_full_info = save_full_info
        self.save_full_info_mode = save_full_info_mode
        self.full_info_folder = full_info_folder
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.run_name = run_name
        self.evaluator = OneClickLLMJudgeEvaluationETHZ1Biology(
            n_max_openai_requests=200_000  # in my experience OK
        )

        # Validate settings"
        if self.save_full_info and self.save_full_info_mode == "folder":
            assert self.full_info_folder is not None, (
                "full_info_folder must be provided when mode is 'folder'"
            )
            assert not (
                self.full_info_folder.exists()
                or len(list(self.full_info_folder.iterdir())) == 0
            ), f"File {self.full_info_folder} already exists"
            self.full_info_folder.parent.mkdir(parents=True, exist_ok=True)
        elif self.save_full_info and self.save_full_info_mode not in [
            "folder",
            "wandb",
            "wandb_conservative",
        ]:
            raise ValueError(f"Invalid save_full_info_mode: {self.save_full_info_mode}")

    @classmethod
    @beartype
    def _evaluate(
        cls,
        model,
        tokenizer,
        evaluator,
        global_step: Any,
        save_full_info: bool,
        save_full_info_mode: str,
        full_info_folder: Optional[Path],
        model_name: str,
        run_name: str,
        # Replace so that we can have special panels for eval before training etc...
        replace_llm_judge_with: str = "llm_judge",
    ) -> dict[Any, Any]:
        """
        Static/class function to do most of the work so we can also call
        externally (kinda janky but eh whatever).
        """
        metrics = {}
        print("@" * 100)
        print("@" * 40 + " STARTING CUSTOM EVALUATION (LLM JUDGE) " + "@" * 40)
        print(f"\n🔍 Running LLM judge evaluation at step {global_step}...")

        assert replace_llm_judge_with.startswith("llm_judge")
        with torch.no_grad():
            (
                custom_results_utility_and_safety,
                custom_results_utility_and_safety_df_as_json,
            ) = evaluator.evaluate(model, tokenizer, n_max_openai_requests=1_000)
            custom_results_utility_and_safety = {
                k.replace("llm_judge", replace_llm_judge_with): v
                for k, v in custom_results_utility_and_safety.items()
            }
            assert all(
                k.startswith("llm_judge")
                for k in custom_results_utility_and_safety.keys()
            )

            metrics.update(custom_results_utility_and_safety)
            if wandb.run is not None:
                wandb.log(
                    {
                        **{k: v for k, v in custom_results_utility_and_safety.items()},
                        "trainer/global_step": global_step,
                    }
                )

            # Save full info if requested
            if save_full_info:
                if save_full_info_mode == "folder":
                    file = (
                        full_info_folder
                        / f"{replace_llm_judge_with}_{global_step}.json"
                    )
                    assert not file.exists()  # should never happen....
                    file.write_text(custom_results_utility_and_safety_df_as_json)
                elif save_full_info_mode == "wandb":
                    # Add to metrics (will be logged to wandb)
                    vizkey = f"{replace_llm_judge_with}/full_info_json"
                    assert vizkey.startswith(replace_llm_judge_with)
                    if vizkey in metrics:
                        print(f"⚠️  Warning: '{vizkey}' already in metrics")
                    metrics[vizkey] = custom_results_utility_and_safety_df_as_json
                    if wandb.run is not None:
                        wandb.log(
                            {
                                **{
                                    vizkey: custom_results_utility_and_safety_df_as_json
                                },
                                "trainer/global_step": global_step,
                            }
                        )
                elif save_full_info_mode == "wandb_conservative":
                    # In this case we parse the json, extract only the generations, and
                    # then report only that basically
                    # NOTE we have a key per model/run and seperate from numerical
                    # metrics purely for easier visualization
                    # NOTE: run name is too long so we don't use :( table should be
                    # enough maybe?
                    vizkey = f"{replace_llm_judge_with}_generations/{model_name}/{global_step}"
                    assert vizkey.startswith("llm_judge")  # this is still important
                    if vizkey in metrics:
                        print(f"⚠️  Warning: {vizkey} already in metrics")
                    generations_pd: list[dict[str, Any]] = orjson.loads(
                        custom_results_utility_and_safety_df_as_json.encode()
                    )
                    assert isinstance(generations_pd, list)
                    assert all(isinstance(x, dict) for x in generations_pd)
                    expected_keys = set(
                        [
                            # NOTE: this should match schema from pydantic in
                            # the code `spylab_1click_judgement.py`
                            "prompt",
                            "response",
                            "seed",
                            "judge_name",
                            "judge_template",
                            "judgement_score",
                            "judgement_explanation",
                        ]
                    )
                    assert all(
                        set(x.keys()) == expected_keys for x in generations_pd
                    ), (
                        "key-sets were "
                        + f"{set(frozenset(z.keys()) for z in generations_pd)}"
                    )

                    print(f"DEBUG: generations_pd has {len(generations_pd)} items")
                    print(
                        f"DEBUG: wandb.run is {'active' if wandb.run is not None else 'None'}"
                    )
                    # print(f"DEBUG: First few generations: {generations_pd[:2] if generations_pd else 'EMPTY'}")
                    # wandb table is a recommendation from Claude
                    generation_table = wandb.Table(
                        columns=[
                            "user_request",
                            "assistant_response",
                            "model_name",
                            "run_name",
                            "global_step",
                        ],
                        data=[
                            [
                                g["prompt"],
                                g["response"],
                                model_name,
                                run_name,
                                global_step,
                            ]
                            for g in generations_pd
                        ],
                    )
                    # NOTE: we report the generations that are serializeable BUT we
                    # log the TABLE to wandb
                    metrics[vizkey] = generations_pd
                    if wandb.run is not None:
                        wandb.log(
                            {
                                **{vizkey: generation_table},
                                "trainer/global_step": global_step,
                            }
                        )
                else:
                    print("WARNING: someone messed with mode wtf")
                print("LLM Judge metrics!")
                print(
                    json.dumps(
                        {
                            k: v
                            for k, v in metrics.items()
                            if k.startswith("llm_judge")
                            and isinstance(v, (float, bool, int))
                        }
                    )
                )
        print("@" * 100)
        assert metrics is not None  # shoulda short-circuited
        # TODO(Adriano) for unknown reasons this was not being logged to wandb (i.e.
        # callback was too late for wandb; I need to read docs; hotfix with
        # wandb.log above...)
        print("Final metrics being logged:\n - " + "\n - ".join(metrics.keys()))
        print("...")
        print("@" * 40 + " FINISHED CUSTOM EVALUATION (LLM JUDGE) " + "@" * 40)
        print("@" * 100)
        return metrics

    def on_evaluate(self, args, state, control, model, metrics=None, **kwargs):
        """Called after evaluation - add custom LLM judge metrics"""
        # Check if we should run the LLM judge (on first eval or at specified intervals)
        # Note: state.global_step tracks training steps
        if state.global_step == 0 or state.global_step % self.llm_judge_every == 0:
            if metrics is None:  # NOTE: metrics needs to be UPDATED acc. to claude
                print("WARNING: METRICS IS NONE; CALLBACK WILL DO NOTHING")
                return
            # Make sure no key collision
            assert not any(
                isinstance(k, str) and k.startswith("llm_judge") for k in metrics.keys()
            ), str(set(metrics.keys()))
            # Collect the new metrics
            new_metrics2log = self._evaluate(
                model,
                self.tokenizer,
                self.evaluator,
                state.global_step,
                self.save_full_info,
                self.save_full_info_mode,
                self.full_info_folder,
                self.model_name,
                self.run_name,
            )
            # Make sure again no collision
            assert all(
                isinstance(k, str) and k.startswith("llm_judge")
                for k in new_metrics2log.keys()
            ), str(set(new_metrics2log.keys()))
            # Update for logging with huggingface
            metrics.update(new_metrics2log)
