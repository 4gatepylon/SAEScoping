import json
import pandas as pd
from pathlib import Path

# Configuration
SWEEP_DIR = Path("models/sweep")
K = 3  # Top K results to show

# Baseline Gemma-2-9b-it results (from earlier evaluation)
mmlu_results_path = Path("/mnt/align4_drive2/adrianoh/git/SAEScoping/baselines/wmdp_github_main/eval_results/gemma2_9b_baseline_mmlu/results.json")
mmlu_results = json.loads(mmlu_results_path.read_text())["results"]
mmlu_acc = mmlu_results["mmlu"]["acc,none"]

# Should be around 50% ^
# acc. to TIGER Lab this is kind of OK: https://huggingface.co/spaces/TIGER-Lab/MMLU-Pro

print(mmlu_acc)
BASELINE = {
    "run_name": "baseline (no RMU)",
    # TODO is this correct? Adriano should please verify what LLM found
    "wmdp_acc": 0.4318,
    "wmdp_bio_acc": 0.5098,
    "wmdp_cyber_acc": 0.3915,
    "wmdp_chem_acc": 0.3848,
    "mmlu_acc": mmlu_acc, # around 70% acc. to paper with 5 shot; 
    "steering_coeff": None,
    "alpha": None,
    "layer_id": None,
}

# Metric: 1 - mean_wmdp + mmlu (rewards low WMDP, high MMLU)
def compute_metric(result: dict) -> float:
    """Compute ranking metric. Higher is better."""
    wmdp_acc = result.get("wmdp_acc")
    mmlu_acc = result.get("mmlu_acc")
    
    if wmdp_acc is None:
        return float("-inf")
    
    # If MMLU not available, just use WMDP (penalized)
    if mmlu_acc is None:
        mmlu_acc = 0.0
    
    # Metric: (1 - wmdp) + mmlu
    # Low WMDP (good unlearning) + High MMLU (preserved capabilities)
    return 1 / (1 / (1 - wmdp_acc) + 1 / mmlu_acc)

def load_all_results(sweep_dir: Path) -> list[dict]:
    """Load all results from sweep directory."""
    results = []
    
    for run_dir in sweep_dir.iterdir():
        if not run_dir.is_dir():
            continue
        
        result = {"run_name": run_dir.name}
        
        # Load WMDP results
        wmdp_file = run_dir / "eval" / "results.json"
        if wmdp_file.exists():
            with open(wmdp_file) as f:
                wmdp_data = json.load(f)
            result["wmdp_acc"] = wmdp_data["results"]["wmdp"]["acc,none"]
            result["wmdp_bio_acc"] = wmdp_data["results"]["wmdp_bio"]["acc,none"]
            result["wmdp_cyber_acc"] = wmdp_data["results"]["wmdp_cyber"]["acc,none"]
            result["wmdp_chem_acc"] = wmdp_data["results"]["wmdp_chem"]["acc,none"]
        
        # Load MMLU results
        mmlu_file = run_dir / "eval_mmlu" / "results.json"
        if mmlu_file.exists():
            with open(mmlu_file) as f:
                mmlu_data = json.load(f)
            result["mmlu_acc"] = mmlu_data["results"]["mmlu"]["acc,none"]
        
        # Parse hyperparameters from run name
        # Format: gemma_2_9b_it_sc{sc}_a{alpha}_l{layer}_lr{lr}
        parts = run_dir.name.split("_")
        for part in parts:
            if part.startswith("sc"):
                result["steering_coeff"] = int(part[2:])
            elif part.startswith("a") and part[1:].isdigit():
                result["alpha"] = int(part[1:])
            elif part.startswith("l") and part[1:].isdigit():
                result["layer_id"] = int(part[1:])
        
        if "wmdp_acc" in result:  # Only include completed runs
            results.append(result)
    
    return results

# Load results
results = load_all_results(SWEEP_DIR)
print(f"Loaded {len(results)} completed runs")

# Compute metric for each
for r in results:
    r["metric"] = compute_metric(r)

# Sort by metric (higher is better)
results_sorted = sorted(results, key=lambda x: x["metric"], reverse=True)

# Convert to DataFrame for display
df = pd.DataFrame(results_sorted)
print(f"\nTop {K} runs by metric (1 - WMDP + MMLU):")
display_cols = ["run_name", "metric", "wmdp_acc", "wmdp_bio_acc", "wmdp_cyber_acc", "wmdp_chem_acc", "mmlu_acc", "steering_coeff", "alpha", "layer_id"]
display_cols = [c for c in display_cols if c in df.columns]
print(df[display_cols].head(K).to_string(index=False))

# Print baseline for comparison
print(f"\nBaseline (no RMU): WMDP={BASELINE['wmdp_acc']:.4f}, MMLU={BASELINE['mmlu_acc']:.4f}")

# Output full paths to best checkpoints
print(f"\n{'='*60}")
print(f"BEST {K} CHECKPOINT PATHS (for lm-eval):")
print(f"{'='*60}")
for i, result in enumerate(results_sorted[:K]):
    full_path = (SWEEP_DIR / result["run_name"]).resolve()
    print(f"\n#{i+1}: {result['run_name']}")
    print(f"    Metric: {result['metric']:.4f} | WMDP: {result['wmdp_acc']:.4f} | MMLU: {result.get('mmlu_acc', 'N/A')}")
    print(f"    Path: {full_path}")