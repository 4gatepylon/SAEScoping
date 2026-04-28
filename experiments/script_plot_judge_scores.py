"""
Plot ground_truth_similarity judge scores from wandb table files vs baseline.

For each *.table.json in <wandb_dir>/files/media/table/llm_judge/, draws a
scatter plot of per-sample scores alongside the corresponding baseline scores.

Usage:
  python script_plot_judge_scores.py --gemma3-later --wandb-dir wandb/run-... --train-domain biology --plot-domain biology
  python script_plot_judge_scores.py --gemma2 --wandb-dir wandb/run-... --train-domain chemistry --plot-domain math
"""
from __future__ import annotations

import json
from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

JUDGE = "ground_truth_similarity"
N_SAMPLES = 100
ALL_DOMAINS = ["biology", "chemistry", "math", "physics"]
BASE_DIR = Path(__file__).resolve().parent  # experiments/

MODEL_CONFIGS = {
    "gemma2": dict(
        model_slug="google--gemma-2-9b-it",
        cache_tag="layer_31--width_16k--canonical",
    ),
    "gemma3": dict(
        model_slug="google--gemma-3-12b-it",
        cache_tag="layer_31--width_16k--canonical",
    ),
    "gemma3_later": dict(
        model_slug="google--gemma-3-12b-it",
        cache_tag="layer_41--width_16k--canonical",
    ),
}


def load_table_json(path: Path, plot_domain: str) -> pd.DataFrame:
    """Load ground_truth_similarity rows for plot_domain from a wandb table.json.

    Domains appear in ALL_DOMAINS order, N_SAMPLES rows each.
    """
    raw = json.loads(path.read_text(errors="replace").replace("\x00", ""))
    cols = raw["columns"]
    df = pd.DataFrame(raw["data"], columns=cols)
    df = df[df["judge_name"] == JUDGE].reset_index(drop=True)
    start = ALL_DOMAINS.index(plot_domain) * N_SAMPLES
    return df.iloc[start : start + N_SAMPLES].reset_index(drop=True)


def load_baseline(baseline_csv: Path, seeds: list[str]) -> pd.Series:
    """Return ground_truth_similarity scores from baseline for the given seeds, in order."""
    df = pd.read_csv(baseline_csv)
    df = df[df["judge_name"] == JUDGE]
    seed2score = dict(zip(df["seed"], df["judgement_score"]))
    return pd.Series([seed2score.get(s, float("nan")) for s in seeds], name="baseline")


def plot_file(
    table_path: Path,
    baseline_csv: Path,
    output_dir: Path,
    plot_domain: str,
):
    df = load_table_json(table_path, plot_domain)
    if df.empty:
        print(f"  Skipping {table_path.name} — no {JUDGE} rows for {plot_domain}.")
        return

    seeds = df["seed"].tolist()
    run_scores = df["judgement_score"].astype(float).values
    baseline_scores = load_baseline(baseline_csv, seeds).values

    x = np.arange(len(seeds))

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.scatter(x, baseline_scores, s=20, alpha=0.7, label="baseline_true", color="steelblue")
    ax.scatter(x, run_scores, s=20, alpha=0.7, label=table_path.stem, color="tomato", marker="^")

    ax.set_xlabel("Sample index")
    ax.set_ylabel("Score (ground_truth_similarity)")
    ax.set_title(f"{plot_domain} — {table_path.stem}")
    ax.set_ylim(-0.05, 1.05)
    ax.axhline(np.nanmean(baseline_scores), color="steelblue", linestyle="--", linewidth=0.8,
               label=f"baseline mean={np.nanmean(baseline_scores):.3f}")
    ax.axhline(np.nanmean(run_scores), color="tomato", linestyle="--", linewidth=0.8,
               label=f"run mean={np.nanmean(run_scores):.3f}")
    ax.legend(fontsize=8)

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{plot_domain}_{table_path.stem}"
    fig.savefig(output_dir / f"{stem}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {output_dir / f'{stem}.png'}")

    df["baseline_score"] = baseline_scores
    df.to_csv(output_dir / f"{stem}.csv", index=False)
    print(f"  Saved {output_dir / f'{stem}.csv'}")


@click.command()
@click.option("--gemma2", "model_key", flag_value="gemma2", help="Gemma-2-9b-it")
@click.option("--gemma3", "model_key", flag_value="gemma3", help="Gemma-3-12b-it (layer 31)")
@click.option("--gemma3-later", "model_key", flag_value="gemma3_later", default=True,
              help="Gemma-3-12b-it (layer 41, default)")
@click.option("--wandb-dir", required=True, type=click.Path(exists=True),
              help="Path to a wandb run directory (e.g. wandb/run-YYYYMMDD_HHMMSS-<id>).")
@click.option("--train-domain", required=True,
              type=click.Choice(["biology", "chemistry", "math", "physics"]),
              help="Domain the model was scoped on (used to locate the baseline CSV).")
@click.option("--plot-domain", default=None,
              type=click.Choice(["biology", "chemistry", "math", "physics"]),
              help="Domain whose scores to extract and plot from the wandb table. Defaults to --train-domain.")
@click.option("--output-dir", default=None,
              help="Directory to save plots. Defaults to <wandb_dir>/plots/.")
def main(model_key, wandb_dir, train_domain, plot_domain, output_dir):
    if plot_domain is None:
        plot_domain = train_domain
    cfg = MODEL_CONFIGS[model_key]
    model_slug = cfg["model_slug"]
    cache_tag = cfg["cache_tag"]

    baseline_csv = (
        BASE_DIR / "outputs_scoping" / model_slug / cache_tag
        / train_domain / "llm_judge_csvs" / "baseline_true.csv"
    )
    if not baseline_csv.exists():
        raise FileNotFoundError(f"Baseline CSV not found: {baseline_csv}")

    table_dir = Path(wandb_dir) / "files" / "media" / "table" / "llm_judge"
    table_files = sorted(table_dir.glob("*.table.json"))
    if not table_files:
        raise FileNotFoundError(f"No table.json files found in {table_dir}")

    if output_dir:
        out_dir = Path(output_dir)
    else:
        run_id = Path(wandb_dir).name.split("-")[-1]  # e.g. "v69qscgx" from "run-20260418_205253-v69qscgx"
        out_dir = baseline_csv.parent / f"{run_id}_plots"
    print(f"Model       : {model_slug} / {cache_tag}")
    print(f"Train domain: {train_domain}")
    print(f"Plot domain : {plot_domain}")
    print(f"Baseline    : {baseline_csv}")
    print(f"Tables      : {len(table_files)} files in {table_dir}")
    print(f"Output      : {out_dir}\n")

    for tf in table_files:
        print(f"Processing {tf.name} ...")
        plot_file(tf, baseline_csv, out_dir, plot_domain)


if __name__ == "__main__":
    main()
