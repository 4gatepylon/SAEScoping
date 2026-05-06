"""Generate phony CSV + YAML config for testing the plotting pipeline."""

from __future__ import annotations

import csv
import random
from pathlib import Path

import yaml


MODELS = ["gemma-2", "gemma-3"]
DOMAINS = ["biology", "chemistry", "physics", "math", "code"]
DOMAIN_METRICS = {"code": "unit_test_pass_rate"}
DEFAULT_METRIC = "ground_truth_similarity"

METHODS_NO_SCOPE = {"vanilla"}
METHODS_WITH_SCOPE = {"scoped", "scoped_recovered", "sft", "pgd"}


def _metric_for(domain: str) -> str:
    return DOMAIN_METRICS.get(domain, DEFAULT_METRIC)


def _generate_rows() -> list[dict]:
    random.seed(42)
    rows = []

    for model in MODELS:
        for domain in DOMAINS:
            metric = _metric_for(domain)
            vanilla_score = random.uniform(0.6, 0.95)
            rows.append({"model": model, "method": "vanilla", "scope_domain": "", "elicitation_domain": domain, "metric": metric, "score": round(vanilla_score, 3)})

        for scope_d in DOMAINS:
            for elicit_d in DOMAINS:
                metric = _metric_for(elicit_d)
                is_in_domain = scope_d == elicit_d

                scoped_score = random.uniform(0.2, 0.5) if is_in_domain else random.uniform(0.05, 0.25)
                rows.append({"model": model, "method": "scoped", "scope_domain": scope_d, "elicitation_domain": elicit_d, "metric": metric, "score": round(scoped_score, 3)})

                recovered_score = random.uniform(0.7, 0.95) if is_in_domain else random.uniform(0.08, 0.35)
                rows.append({"model": model, "method": "scoped_recovered", "scope_domain": scope_d, "elicitation_domain": elicit_d, "metric": metric, "score": round(recovered_score, 3)})

                sft_score = random.uniform(0.75, 0.98) if is_in_domain else random.uniform(0.3, 0.6)
                rows.append({"model": model, "method": "sft", "scope_domain": scope_d, "elicitation_domain": elicit_d, "metric": metric, "score": round(sft_score, 3)})

                pgd_score = random.uniform(0.5, 0.8) if is_in_domain else random.uniform(0.15, 0.45)
                rows.append({"model": model, "method": "pgd", "scope_domain": scope_d, "elicitation_domain": elicit_d, "metric": metric, "score": round(pgd_score, 3)})

    return rows


def _generate_overlap_rows() -> list[dict]:
    """Generate phony feature overlap data. Overlap is a property of (model, scope, elicit) pairs."""
    random.seed(99)
    rows = []
    for model in MODELS:
        for scope_d in DOMAINS:
            for elicit_d in DOMAINS:
                if scope_d == elicit_d:
                    continue
                # Related domains get higher overlap, unrelated get lower
                related_pairs = {
                    ("biology", "chemistry"), ("chemistry", "biology"),
                    ("physics", "math"), ("math", "physics"),
                }
                if (scope_d, elicit_d) in related_pairs:
                    overlap = random.uniform(0.4, 0.7)
                else:
                    overlap = random.uniform(0.05, 0.35)
                rows.append({"model": model, "scope_domain": scope_d, "elicitation_domain": elicit_d, "overlap": round(overlap, 3)})
    return rows


def _generate_config() -> dict:
    return {
        "models": [
            {"id": "gemma-2", "display_name": "Gemma 2"},
            {"id": "gemma-3", "display_name": "Gemma 3"},
        ],
        "domains": {
            "default_metric": DEFAULT_METRIC,
            "entries": [
                {"id": d, "display_name": d.capitalize(), **({"metric": DOMAIN_METRICS[d]} if d in DOMAIN_METRICS else {})}
                for d in DOMAINS
            ],
        },
        "methods": [
            {"id": "vanilla", "display_name": "Vanilla", "color": "#888888", "requires_scope_domain": False},
            {"id": "scoped", "display_name": "Scoped", "color": "#ff7f0e"},
            {"id": "scoped_recovered", "display_name": "Scoped + Recovered", "color": "#2ca02c"},
            {"id": "sft", "display_name": "SFT", "color": "#1f77b4"},
            {"id": "pgd", "display_name": "PGD", "color": "#d62728"},
        ],
        "figures": {
            "in_domain_bar": {
                "methods": ["scoped", "scoped_recovered", "sft"],
                "baseline_method": "vanilla",
            },
            "ood_table": {
                "our_method": "scoped_recovered",
                "comparison_methods": ["pgd"],
            },
            "ood_bar": {
                "methods": ["scoped_recovered", "pgd"],
            },
            "feature_overlap_scatter": {
                "raw_method": "scoped",
                "elicited_method": "scoped_recovered",
                "overlap_csv": "phony_overlap.csv",
            },
        },
        "output": {
            "dpi": 200,
            "latex": True,
        },
    }


def main():
    out_dir = Path(__file__).parent / "test_fixtures"
    out_dir.mkdir(exist_ok=True)

    rows = _generate_rows()
    csv_path = out_dir / "phony_data.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "method", "scope_domain", "elicitation_domain", "metric", "score"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {csv_path}")

    overlap_rows = _generate_overlap_rows()
    overlap_path = out_dir / "phony_overlap.csv"
    with open(overlap_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "scope_domain", "elicitation_domain", "overlap"])
        writer.writeheader()
        writer.writerows(overlap_rows)
    print(f"Wrote {len(overlap_rows)} overlap rows to {overlap_path}")

    config_path = out_dir / "phony_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(_generate_config(), f, default_flow_style=False, sort_keys=False)
    print(f"Wrote config to {config_path}")


if __name__ == "__main__":
    main()
