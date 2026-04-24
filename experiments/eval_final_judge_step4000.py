import os
import json
import torch
import pandas as pd
from transformers import AutoTokenizer, Gemma3ForCausalLM
from sae_scoping.xxx_evaluation.scoping_eval import OneClickLLMJudgeScopingEval
from datasets import load_dataset

MODEL_PATH = "/home/anish559/SAEScoping/experiments/outputs_scoping_gemma3_12b/attack/math/final_success"
BASE_MODEL_NAME = "google/gemma-3-12b-it"
SAVE_DIR = "/home/anish559/SAEScoping/experiments/results_final_eval_step4000"

def run_evaluation():
    os.makedirs(SAVE_DIR, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    model = Gemma3ForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto")
    evaluator = OneClickLLMJudgeScopingEval(n_samples=1000, judge_model="gpt-4.1-nano", train_domain="coding", attack_domain="math")
    ds_coding = load_dataset("nvidia/OpenCodeReasoning", split="train").shuffle(seed=42).select(range(1000))
    ds_math = load_dataset("4gate/StemQAMixture", split="train").shuffle(seed=42).select(range(1000))
    scores, df_json = evaluator.evaluate(model, tokenizer, domain_questions={"coding": ds_coding["instruction"], "math": ds_math["instruction"]})
    with open(f"{SAVE_DIR}/scores.json", "w") as f: json.dump(scores, f, indent=4)
    pd.read_json(df_json).to_csv(f"{SAVE_DIR}/judgements_final.csv", index=False)
if __name__ == "__main__": run_evaluation()
