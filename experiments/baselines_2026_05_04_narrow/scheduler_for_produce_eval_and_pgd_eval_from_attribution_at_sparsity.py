#!/usr/bin/env python3
"""Poll for attribution_scores.pt files and run eval on each one.

Reads a JSON config (via --config) mapping pt file paths to per-job args.
Loops forever until every job is either completed or its pt file has been
processed.  A job is "done" when its output_dir contains summary.json.

Config format::

    {
      "model_name": "google/gemma-2-9b-it",
      "sparsity_levels": [0.5],
      "dtype": "bfloat16",
      "eval_domains": ["biology", "chemistry", "math", "physics"],
      "eval_split": "validation",
      "judge_model": "gpt-4.1-nano",
      "judge_n_samples": 50,
      "loss_n_samples": 200,
      "loss_batch_size": 2,
      "output_root": "eval_results",
      "jobs": {
        "pruned_models/gemma2_9b_it_biology/attribution_scores.pt": {
          "train_domain": "biology"
        },
        ...
      }
    }

Top-level keys (except ``jobs`` and ``output_root``) are namespace defaults
merged into every job.  Per-job keys override defaults.  ``output_dir`` is
auto-derived as ``{output_root}/{pt_parent_dir_name}/`` unless overridden.
"""

import argparse
import importlib.util as _ilu
import json
import os
import sys
import time
from pathlib import Path


def _load_sibling(name, filename):
    spec = _ilu.spec_from_file_location(
        name, os.path.join(os.path.dirname(__file__), filename)
    )
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    p = argparse.ArgumentParser(description="Poll for attribution .pt files and run eval")
    p.add_argument("--config", required=True, help="Path to JSON config file")
    p.add_argument("--poll_interval", type=int, default=60, help="Seconds between polls")
    cli = p.parse_args()

    with open(cli.config) as f:
        config = json.load(f)

    jobs = config.pop("jobs")
    output_root = Path(config.pop("output_root", "eval_results"))
    poll_interval = config.pop("poll_interval", cli.poll_interval)
    defaults = config

    eval_mod = _load_sibling(
        "produce_eval",
        "produce_eval_and_pgd_eval_from_attribution_at_sparsity.py",
    )

    while True:
        pending = []
        for pt_path, job_overrides in jobs.items():
            output_dir = output_root / Path(pt_path).parent.name
            sentinel = output_dir / "summary.json"

            if sentinel.exists():
                continue

            if not Path(pt_path).exists():
                pending.append(pt_path)
                continue

            ns = {**defaults, **job_overrides}
            ns["attribution_scores_path"] = pt_path
            ns["output_dir"] = str(output_dir)
            args = argparse.Namespace(**ns)

            print(f"\n[scheduler] Running eval: {pt_path} -> {output_dir}")
            eval_mod.main(args)
            print(f"[scheduler] Done: {pt_path}")

        if not pending:
            print("[scheduler] All jobs complete.")
            break

        print(f"[scheduler] {len(pending)} job(s) waiting for .pt files, "
              f"polling in {poll_interval}s: {pending}")
        time.sleep(poll_interval)


if __name__ == "__main__":
    main()
