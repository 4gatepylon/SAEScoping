"""Load and validate input data against config."""

from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd
import yaml

from .config_schemas import PlotConfig, ScoreRow


def load_config(config_path: str | Path) -> PlotConfig:
    with open(config_path) as f:
        raw = yaml.safe_load(f)
    return PlotConfig(**raw)


def load_data(data_path: str | Path, config: PlotConfig) -> pd.DataFrame:
    df = pd.read_csv(data_path)
    df["scope_domain"] = df["scope_domain"].where(df["scope_domain"].notna(), None)
    df["scope_domain"] = df["scope_domain"].replace("", None)

    method_ids = {m.id for m in config.methods}
    model_ids = {m.id for m in config.models}

    errors = []
    for _, row in df.iterrows():
        if row["model"] not in model_ids:
            errors.append(f"Unknown model '{row['model']}'")
        if row["method"] not in method_ids:
            errors.append(f"Unknown method '{row['method']}'")
        m = config.get_method(row["method"]) if row["method"] in method_ids else None
        if m and m.requires_scope_domain and pd.isna(row.get("scope_domain")):
            errors.append(f"Method '{row['method']}' requires scope_domain but row has none: {row.to_dict()}")

    if errors:
        unique_errors = list(dict.fromkeys(errors))[:20]
        raise ValueError(f"Data validation errors:\n" + "\n".join(unique_errors))

    key_cols = ["model", "method", "scope_domain", "elicitation_domain", "metric"]
    dupes = df.duplicated(subset=key_cols, keep=False)
    if dupes.any():
        raise ValueError(f"Duplicate rows found:\n{df[dupes][key_cols].drop_duplicates()}")

    return df


def get_score(df: pd.DataFrame, config: PlotConfig, model: str, method: str, elicitation_domain: str, scope_domain: str | None = None, metric: str | None = None) -> float | None:
    """Look up a single score. Returns None if not found."""
    if metric is None:
        metric = config.domains.get_metric(elicitation_domain)

    mask = (
        (df["model"] == model)
        & (df["method"] == method)
        & (df["elicitation_domain"] == elicitation_domain)
        & (df["metric"] == metric)
    )
    if scope_domain is not None:
        mask &= df["scope_domain"] == scope_domain
    else:
        mask &= df["scope_domain"].isna()

    rows = df[mask]
    if len(rows) == 0:
        return None
    if len(rows) > 1:
        raise ValueError(f"Multiple rows for ({model}, {method}, {scope_domain}, {elicitation_domain}, {metric})")
    return float(rows.iloc[0]["score"])
