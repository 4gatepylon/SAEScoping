import argparse
import gc
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Optional

import torch
import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, Gemma3ForCausalLM

# --- Configuration ---
BASE_MODEL_NAME = "google/gemma-3-12b-it"
# Path to the checkpoint at step 2500 (recovery stage)
CHECKPOINT_PATH = "/home/anish559/SAEScoping/experiments/outputs_scoping_gemma3_12b/recover/checkpoint-2500"
# Default directory where evaluation results will be saved
DEFAULT_OUTPUT_DIR = "/home/anish559/SAEScoping/experiments/results_eval_gemma3_checkpoint2500"

# Prompt template – the model is asked to return **only** the code block.
USER_PROMPT_TEMPLATE = (
    "You are an expert Python programmer. Solve the following programming problem. "
    "Provide ONLY the complete, working Python code. Do not include any explanations, reasoning, or markdown outside the code block. "
    "Wrap your solution in a ```python ... ``` code block.\n\n"
    "### Problem:\n{problem}\n\n"
    "### Answer:\n"
)

@dataclass
class CodingSample:
    id: str
    input_text: str
    source: str
    difficulty: str
    reference_solution: str
    test_cases: List[Tuple[str, str]]  # (input_str, expected_output_str)

# ---------------------------------------------------------------------------
# Dataset handling – load split_0, drop HARD/VERY_HARD, keep samples that contain
# explicit Input/Output examples.
# ---------------------------------------------------------------------------
def extract_io_from_input(input_text: str) -> List[Tuple[str, str]]:
    """
    Extract sample input/output pairs from problem description.
    Improved to handle sections like 'Examples', 'Sample Input', etc.
    """
    test_cases = []
    
    # 1. Try to isolate the Examples/Samples section first
    parts = re.split(r"(?:Examples|Example|Samples|Sample|Sample Input|Sample input)", input_text, flags=re.IGNORECASE)
    if len(parts) > 1:
        data_text = parts[-1]
    else:
        data_text = input_text

    # 2. Match Input and Output blocks
    pattern = re.compile(r"(?:Input|input|INPUT)\s*[\n:]+(.*?)\s*(?:Output|output|OUTPUT)\s*[\n:]+(.*?)(?=\s*(?:Example|Sample|Input|input|INPUT|Note|Description|Explanation|$))", re.DOTALL)
    
    matches = pattern.findall(data_text)
    for inp, outp in matches:
        clean_inp = inp.strip().replace("\r", "")
        clean_outp = outp.strip().replace("\r", "")
        
        # Heuristic: If input is a long paragraph with no numbers, it's likely a description
        if len(clean_inp.split()) > 15 and not any(c.isdigit() for c in clean_inp):
            continue
            
        if clean_inp or clean_outp:
            if "Explanation" in clean_outp:
                clean_outp = clean_outp.split("Explanation")[0].strip()
            if "Note" in clean_outp:
                clean_outp = clean_outp.split("Note")[0].strip()
                
            test_cases.append((clean_inp, clean_outp))
            
    return test_cases

def load_nvidia_eval(n_samples: int = 30, seed: int = 42) -> List[CodingSample]:
    """Stream the OpenCodeReasoning split_0 and return a filtered list."""
    print("Loading nvidia/OpenCodeReasoning (split_0) …")
    ds = load_dataset("nvidia/OpenCodeReasoning", "split_0", split="split_0", streaming=True)
    samples: List[CodingSample] = []
    skipped_hard = 0
    skipped_no_io = 0
    total = 0
    for ex in ds:
        total += 1
        diff = str(ex.get("difficulty", "")).upper()
        if "HARD" in diff:
            skipped_hard += 1
            continue
        input_text = ex.get("input", "")
        if not input_text or input_text == "-":
            continue
        test_cases = extract_io_from_input(input_text)
        if not test_cases:
            skipped_no_io += 1
            continue
        samples.append(
            CodingSample(
                id=ex.get("id", f"sample_{total}"),
                input_text=input_text,
                source=ex.get("source", "unknown"),
                difficulty=diff,
                reference_solution=ex.get("solution", ""),
                test_cases=test_cases,
            )
        )
        if len(samples) >= n_samples:
            break
    print(f"  Checked {total} entries – loaded {len(samples)} samples (skipped HARD: {skipped_hard}, no I/O: {skipped_no_io})")
    return samples

# ---------------------------------------------------------------------------
# Execution helpers – run generated code against a single test case.
# ---------------------------------------------------------------------------
def run_code(code: str, input_str: str, timeout: int = 15) -> Tuple[bool, str, str]:
    """Execute *code* with *input_str* on stdin."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
        tmp_path = f.name
    try:
        proc = subprocess.run(
            [sys.executable, tmp_path],
            input=input_str,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return (proc.returncode == 0), proc.stdout.strip(), proc.stderr.strip()
    except subprocess.TimeoutExpired:
        return False, "", "Timeout"
    except Exception as e:
        return False, "", str(e)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

def evaluate_solution(generated_code: str, test_cases: List[Tuple[str, str]]) -> Tuple[bool, bool, int, List[dict]]:
    """Run *generated_code* on every test case.
    Returns (all_correct, all_compiled, passed_count, detailed_case_info).
    """
    if not generated_code.strip():
        return False, False, 0, []
    
    passed = 0
    compiled_count = 0
    details: List[dict] = []
    
    for inp, exp in test_cases:
        ok, out, err = run_code(generated_code, inp)
        exp_norm = "\n".join([l.strip() for l in exp.splitlines() if l.strip()])
        out_norm = "\n".join([l.strip() for l in out.splitlines() if l.strip()])
        
        # 'ok' is true if returncode == 0
        if ok:
            compiled_count += 1
            
        correct = ok and exp_norm == out_norm
        if correct:
            passed += 1
            
        details.append(
            {
                "input": inp,
                "expected": exp,
                "output": out,
                "success": ok,
                "correct": correct,
                "error": err,
            }
        )
    
    all_compiled = (compiled_count == len(test_cases) and len(test_cases) > 0)
    all_correct = (passed == len(test_cases) and len(test_cases) > 0)
    
    return all_correct, all_compiled, passed, details

# ---------------------------------------------------------------------------
# Model interaction utilities.
# ---------------------------------------------------------------------------
def extract_code(text: str) -> str:
    """Pull the first python code block from *text*; fallback to raw text."""
    m = re.search(r"```python\s*(.*?)\s*```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    m = re.search(r"```\s*(.*?)\s*```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    return text.strip()

def model_generate(model, tokenizer, problem_text: str, device) -> Tuple[str, str]:
    """Generate a response for *problem_text* and return (code, full_response)."""
    prompt = USER_PROMPT_TEMPLATE.format(problem=problem_text)
    messages = [{"role": "user", "content": prompt}]
    
    # Gemma 3 supports chat template
    input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(device)
    
    with torch.no_grad():
        out_ids = model.generate(
            input_ids,
            max_new_tokens=1024,
            do_sample=True,
            temperature=0.2,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )
    response = tokenizer.decode(out_ids[0][len(input_ids[0]):], skip_special_tokens=True)
    code = extract_code(response)
    return code, response

# ---------------------------------------------------------------------------
# Main entry point.
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Evaluate Gemma-3-12B on OpenCodeReasoning.")
    parser.add_argument("--checkpoint", default=CHECKPOINT_PATH, help="Path to the model checkpoint. Use 'google/gemma-3-12b-it' for baseline.")
    parser.add_argument("--n-samples", type=int, default=500, help="Number of evaluation samples to draw.")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", default=None, help="Folder to save results.")
    args = parser.parse_args()

    if args.output_dir is None:
        if args.checkpoint == BASE_MODEL_NAME:
            args.output_dir = "/home/anish559/SAEScoping/experiments/results_eval_gemma3_baseline"
        else:
            args.output_dir = DEFAULT_OUTPUT_DIR

    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    # Load model & tokenizer
    print(f"Loading model from {args.checkpoint} …")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    
    # NOTE: Using Gemma3ForCausalLM for Gemma 3
    model = Gemma3ForCausalLM.from_pretrained(
        args.checkpoint,
        torch_dtype=torch.bfloat16,
        device_map=device,
        attn_implementation="eager",
        low_cpu_mem_usage=True,
    )
    model.eval()

    # Load filtered dataset
    samples = load_nvidia_eval(n_samples=args.n_samples)

    # Evaluation loop
    results = []
    fully_passed = 0
    fully_compiled = 0
    start_idx = 0

    out_path_tmp = os.path.join(args.output_dir, "eval_results_partial.json")
    if os.path.exists(out_path_tmp):
        print(f"Checking for existing progress in {out_path_tmp} …")
        try:
            with open(out_path_tmp, "r") as f:
                cached = json.load(f)
            if cached.get("checkpoint") == args.checkpoint:
                results = cached.get("results", [])
                fully_passed = cached.get("fully_passed", 0)
                fully_compiled = cached.get("fully_compiled", 0)
                start_idx = len(results)
                print(f"Resuming from sample {start_idx} (Passed: {fully_passed}, Compiled: {fully_compiled})")
                if start_idx >= len(samples):
                    start_idx = len(samples)
        except Exception as e:
            print(f"Could not load existing results ({e}). Starting fresh.")

    for idx in tqdm.tqdm(range(start_idx, len(samples)), initial=start_idx, total=len(samples), desc="Evaluating"):
        sample = samples[idx]
        code, raw = model_generate(model, tokenizer, sample.input_text, device)
        all_ok, all_compiled, passed_cnt, case_info = evaluate_solution(code, sample.test_cases)
        
        if all_ok:
            fully_passed += 1
        if all_compiled:
            fully_compiled += 1
            
        results.append(
            {
                "id": sample.id,
                "source": sample.source,
                "difficulty": sample.difficulty,
                "all_correct": all_ok,
                "all_compiled": all_compiled,
                "passed_cases": passed_cnt,
                "total_cases": len(sample.test_cases),
                "generated_code": code,
                "raw_response": raw,
                "case_details": case_info,
            }
        )
        status = "PASS" if all_ok else ("COMPILE-ONLY" if all_compiled else "FAIL")
        print(f"[{idx+1}/{len(samples)}] {status} {passed_cnt}/{len(sample.test_cases)} – {sample.id}", flush=True)

        if (idx + 1) % 10 == 0:
            accuracy_tmp = (fully_passed / (idx + 1) * 100)
            comp_accuracy_tmp = (fully_compiled / (idx + 1) * 100)
            tmp_summary = {
                "checkpoint": args.checkpoint,
                "processed_samples": idx + 1,
                "fully_passed": fully_passed,
                "fully_compiled": fully_compiled,
                "current_accuracy": f"{accuracy_tmp:.1f}%",
                "current_compilation_accuracy": f"{comp_accuracy_tmp:.1f}%",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "results": results,
            }
            with open(out_path_tmp, "w") as f:
                json.dump(tmp_summary, f, indent=2)

    # Final summary
    accuracy = (fully_passed / len(samples) * 100) if samples else 0.0
    comp_accuracy = (fully_compiled / len(samples) * 100) if samples else 0.0
    summary = {
        "checkpoint": args.checkpoint,
        "total_samples": len(samples),
        "fully_passed": fully_passed,
        "fully_compiled": fully_compiled,
        "accuracy": f"{accuracy:.1f}%",
        "compilation_accuracy": f"{comp_accuracy:.1f}%",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "results": results,
    }
    out_path = os.path.join(args.output_dir, "eval_results.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nEvaluation complete!")
    print(f"Correctness Accuracy: {accuracy:.1f}%")
    print(f"Compilation Accuracy: {comp_accuracy:.1f}%")
    print(f"Results saved to {out_path}")

if __name__ == "__main__":
    main()
