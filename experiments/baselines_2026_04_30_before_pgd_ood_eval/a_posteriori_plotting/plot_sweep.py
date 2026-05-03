"""Summary panels + text tables for the OOD baseline sweep (no PGD).

A single `load_all_rows()` is the source of truth for both plots and
tables — if plotting renders wrong, the tables are still correct from
the same data.

Layout per PNG (one per measurement): rows = model, cols = calibration
domain; each subplot has 4 lines (one per eval domain). Color = eval
domain (consistent across every figure). In-scope (eval == cal) is
solid + filled marker; out-of-scope is dashed + x-marker.

Text tables (one per measurement) mirror the plot's pivot: rows =
(model, cal, eval, scope), columns = the 4 target sparsities.

There is no sparsity-0 anchor: baseline.json has only loss/sparsity, the
LLM judge is not run pre-pruning. Each line therefore has 4 points.

Usage:
    python plot_sweep.py
    python plot_sweep.py --log-dir <LOG_DIR> --output-dir <OUT>
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

DOMAIN_COLOR = {
    "biology": "#1f77b4",
    "chemistry": "#ff7f0e",
    "math": "#2ca02c",
    "physics": "#d62728",
}
MODEL_ORDER = ["google/gemma-2-9b-it", "google/gemma-3-12b-it"]
DOMAIN_ORDER = ["biology", "chemistry", "math", "physics"]
MEASUREMENTS = ["quality", "ground_truth_similarity", "fluency", "relevance"]
KEY_RE = re.compile(r"^llm_judge/(?P<eval>[^/]+)/(?P<scope>[^/]+)/(?P<meas>[^/]+)$")


# ---------- shared data layer ------------------------------------------------


def discover_artifacts(log_dir: Path) -> list[Path]:
    arts: list[Path] = []
    for log in sorted(log_dir.glob("run_*.sh.log")):
        for line in log.read_text(errors="ignore").splitlines():
            if line.startswith("Artifacts: "):
                arts.append(Path(line[len("Artifacts: ") :].strip()))
                break
    return arts


def load_cell(art_dir: Path) -> list[dict]:
    metadata = json.loads((art_dir / "metadata.json").read_text())
    target_sparsities = metadata["sweep"]["nn_linear_sparsities"]
    cal_domain = metadata["dataset_subset"]
    model_id = metadata["model_id"]
    rows: list[dict] = []
    for step_dir in sorted(art_dir.glob("step_*")):
        step_idx = int(step_dir.name.split("_")[1])
        target = target_sparsities[step_idx]
        scores = json.loads((step_dir / "sweep" / "scores.json").read_text())
        for k, v in scores.items():
            m = KEY_RE.match(k)
            if not m:
                continue
            rows.append(
                {
                    "model_id": model_id,
                    "calibration_domain": cal_domain,
                    "eval_domain": m["eval"],
                    "scope": m["scope"],
                    "measurement": m["meas"],
                    "target_sparsity": target,
                    "score": v,
                }
            )
    return rows


def load_all_rows(log_dir: Path) -> tuple[list[dict], list[float]]:
    """Returns (rows, sorted_target_sparsities) — single source of truth."""
    arts = discover_artifacts(log_dir)
    rows: list[dict] = []
    for a in arts:
        rows.extend(load_cell(a))
    if not rows:
        raise SystemExit(f"no rows loaded from {log_dir} (artifact dirs: {len(arts)})")
    sparsities = sorted({r["target_sparsity"] for r in rows})
    print(f"loaded {len(rows)} rows from {len(arts)} cells; sparsities={sparsities}", file=sys.stderr)
    return rows, sparsities


def short_model(m: str) -> str:
    return m.split("/")[-1]


def line_rows_for(rows: list[dict], measurement: str, model: str, cal: str, ev: str) -> list[dict]:
    """The 4 sparsity points for one line of one subplot. Sorted by sparsity."""
    return sorted(
        (r for r in rows if r["measurement"] == measurement and r["model_id"] == model and r["calibration_domain"] == cal and r["eval_domain"] == ev),
        key=lambda r: r["target_sparsity"],
    )


# ---------- text tables -----------------------------------------------------


def render_table(rows: list[dict], measurement: str, sparsities: list[float]) -> str:
    """One text table per measurement, mirroring the plot's pivot."""
    out: list[str] = []
    spar_cols = [f"s={s:.2f}" for s in sparsities]
    out.append(f"=== measurement: {measurement} ===")
    header = f"{'model':<24} {'cal':<10} {'eval':<10} {'scope':<13} " + " ".join(f"{c:>7}" for c in spar_cols)
    out.append(header)
    out.append("-" * len(header))
    for model in MODEL_ORDER:
        for cal in DOMAIN_ORDER:
            for ev in DOMAIN_ORDER:
                lr = line_rows_for(rows, measurement, model, cal, ev)
                if not lr:
                    continue
                scope = lr[0]["scope"]
                by_s = {r["target_sparsity"]: r["score"] for r in lr}
                cells = " ".join(f"{by_s.get(s, float('nan')):>7.3f}" for s in sparsities)
                out.append(f"{short_model(model):<24} {cal:<10} {ev:<10} {scope:<13} {cells}")
            out.append("")
    return "\n".join(out)


def write_tables(rows: list[dict], sparsities: list[float], out_dir: Path) -> None:
    out_dir.mkdir(exist_ok=True, parents=True)
    for meas in MEASUREMENTS:
        text = render_table(rows, meas, sparsities)
        path = out_dir / f"{meas}.txt"
        path.write_text(text + "\n")
        print(f"wrote {path}")
    long_path = out_dir / "all_long.tsv"
    cols = ["model_id", "calibration_domain", "eval_domain", "scope", "measurement", "target_sparsity", "score"]
    with long_path.open("w") as f:
        f.write("\t".join(cols) + "\n")
        for r in rows:
            f.write("\t".join(f"{r[c]:.6f}" if isinstance(r[c], float) else str(r[c]) for c in cols) + "\n")
    print(f"wrote {long_path}")


# ---------- plots -----------------------------------------------------------


def plot_measurement(rows: list[dict], measurement: str, sparsities: list[float], out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(
        nrows=len(MODEL_ORDER),
        ncols=len(DOMAIN_ORDER),
        figsize=(16, 7),
        sharex=True,
        sharey=True,
    )

    for i, model in enumerate(MODEL_ORDER):
        for j, cal in enumerate(DOMAIN_ORDER):
            ax = axes[i, j]
            for ev in DOMAIN_ORDER:
                lr = line_rows_for(rows, measurement, model, cal, ev)
                if not lr:
                    continue
                in_scope = ev == cal
                ax.plot(
                    [r["target_sparsity"] for r in lr],
                    [r["score"] for r in lr],
                    color=DOMAIN_COLOR[ev],
                    marker="o" if in_scope else "x",
                    markersize=7 if in_scope else 5,
                    linewidth=2.2 if in_scope else 1.2,
                    linestyle="-" if in_scope else "--",
                )
            ax.set_title(f"{short_model(model)}  ·  cal={cal}", fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.set_xticks(sparsities)
            if i == len(MODEL_ORDER) - 1:
                ax.set_xlabel("nn_linear_sparsity (target)")
            if j == 0:
                ax.set_ylabel(measurement)

    handles = [plt.Line2D([0], [0], color=DOMAIN_COLOR[d], marker="o", linestyle="-", linewidth=2.2, label=f"eval={d}") for d in DOMAIN_ORDER]
    handles.append(plt.Line2D([0], [0], color="black", marker="o", linestyle="-", linewidth=2.2, label="in-scope (eval==cal)"))
    handles.append(plt.Line2D([0], [0], color="black", marker="x", linestyle="--", linewidth=1.2, label="out-of-scope"))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"OOD baseline (no PGD) — {measurement}", fontsize=14)
    fig.tight_layout(rect=[0, 0.04, 1, 0.96])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def write_plots(rows: list[dict], sparsities: list[float], out_dir: Path) -> None:
    out_dir.mkdir(exist_ok=True, parents=True)
    for meas in MEASUREMENTS:
        out = out_dir / f"{meas}.png"
        plot_measurement(rows, meas, sparsities, out)
        print(f"wrote {out}")


# ---------- main ------------------------------------------------------------


def main() -> None:
    here = Path(__file__).resolve().parent
    experiment_dir = here.parent
    p = argparse.ArgumentParser()
    p.add_argument("--log-dir", type=Path, default=None, help="LOG_DIR containing run_*.sh.log; default = newest under <experiment>/babysit_logs/")
    p.add_argument("--output-dir", type=Path, default=here, help=f"Output root; plots go to <out>/plots and tables to <out>/tables (default: {here})")
    p.add_argument("--no-plots", action="store_true", help="Skip PNGs (write only text tables)")
    p.add_argument("--no-tables", action="store_true", help="Skip tables (write only PNGs)")
    args = p.parse_args()

    log_dir = args.log_dir
    if log_dir is None:
        candidates = sorted((experiment_dir / "babysit_logs").glob("*/"), reverse=True)
        if not candidates:
            raise SystemExit(f"no babysit_logs/ subdirs under {experiment_dir}")
        log_dir = candidates[0]
    if not log_dir.is_dir():
        raise SystemExit(f"log dir not found: {log_dir}")

    rows, sparsities = load_all_rows(log_dir)

    if not args.no_tables:
        write_tables(rows, sparsities, args.output_dir / "tables")
    if not args.no_plots:
        write_plots(rows, sparsities, args.output_dir / "plots")


if __name__ == "__main__":
    main()
