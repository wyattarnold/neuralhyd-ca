"""Post-processing CLI for evaluation metrics and plots.

Usage
-----
Compute evaluation metrics (CSVs written to data/eval/):
    python scripts/post_process.py --eval dual_lstm_kfold single_lstm_kfold

Plot CDF of NSE across models:
    python scripts/post_process.py --cdf-metric nse kge --runs dual_lstm_kfold single_lstm_kfold

Plot CDF comparing LSTM KGE vs VIC calibrated/regionalized KGE:
    python scripts/post_process.py --cdf-vic-kge

Both can be combined:
    python scripts/post_process.py --eval dual_lstm_kfold single_lstm_kfold --cdf-metric nse kge fhv flv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd

from src.eval.metrics import (
    compute_vic_metrics,
    load_lstm_fold_results,
    METRICS,
)
from src.eval.plots import plot_metric_cdf

TRAINING_OUTPUT = REPO_ROOT / "data" / "training" / "output"
EVAL_DIR = REPO_ROOT / "data" / "eval"
VIC_LABEL = "vic_simulated"
VIC_CAL_DIR = REPO_ROOT / "data" / "external" / "cec" / "VIC-Calibration"


# ---------------------------------------------------------------------------
# --eval: compute and save per-basin metrics
# ---------------------------------------------------------------------------

def run_eval(run_names: list[str]) -> None:
    """Compute and write evaluation CSVs for the requested runs + VIC."""
    # Always compute VIC metrics
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Computing VIC simulated metrics...")
    vic_df = compute_vic_metrics(REPO_ROOT)
    vic_path = EVAL_DIR / f"{VIC_LABEL}.csv"
    vic_df.to_csv(vic_path, index=False)
    print(f"  VIC: {len(vic_df)} basins → {vic_path}")
    _print_summary(vic_df, "VIC Simulated")

    for name in run_names:
        run_dir = TRAINING_OUTPUT / name
        if not run_dir.exists():
            print(f"  WARNING: {run_dir} does not exist, skipping.")
            continue
        print(f"Computing metrics for {name}...")
        df = load_lstm_fold_results(run_dir)
        out_path = EVAL_DIR / f"{name}.csv"
        df.to_csv(out_path, index=False)
        print(f"  {name}: {len(df)} basins → {out_path}")
        _print_summary(df, name)


def _print_summary(df: pd.DataFrame, label: str) -> None:
    """Print per-tier median metrics."""
    print(f"\n  {label} — Per-tier medians")
    print(f"  {'Tier':<8}{'N':<6}{'NSE':>10}{'KGE':>10}{'FHV':>10}{'FLV':>10}")
    print(f"  {'-' * 54}")
    for tier in sorted(df["tier"].unique()):
        sub = df[df["tier"] == tier]
        print(
            f"  {tier:<8}{len(sub):<6}"
            f"{sub['nse'].median():>10.3f}{sub['kge'].median():>10.3f}"
            f"{sub['fhv'].median():>10.1f}{sub['flv'].median():>10.1f}"
        )
    print(
        f"  {'All':<8}{len(df):<6}"
        f"{df['nse'].median():>10.3f}{df['kge'].median():>10.3f}"
        f"{df['fhv'].median():>10.1f}{df['flv'].median():>10.1f}"
    )
    print()


# ---------------------------------------------------------------------------
# --cdf-metric: CDF plot from eval_metrics.csv files
# ---------------------------------------------------------------------------

def run_cdf(metric: str, run_names: list[str]) -> None:
    """Plot CDF for one metric across LSTM runs + VIC."""
    if metric not in METRICS:
        print(f"ERROR: unknown metric '{metric}'. Choose from {METRICS}")
        sys.exit(1)

    series: dict[str, np.ndarray] = {}

    # Load VIC
    vic_csv = EVAL_DIR / f"{VIC_LABEL}.csv"
    if vic_csv.exists():
        vic_df = pd.read_csv(vic_csv)
        series["VIC Simulated"] = vic_df[metric].values
    else:
        print(f"  WARNING: {vic_csv} not found. Run --eval first.")

    # Load LSTM runs
    for name in run_names:
        csv_path = EVAL_DIR / f"{name}.csv"
        if not csv_path.exists():
            print(f"  WARNING: {csv_path} not found. Run --eval first for {name}.")
            continue
        df = pd.read_csv(csv_path)
        series[name] = df[metric].values

    if not series:
        print("ERROR: No data to plot.")
        sys.exit(1)

    # Per-metric plot options
    plot_kwargs: dict = {}
    if metric in ("nse", "kge"):
        plot_kwargs["ylim"] = (-1.0, 1.0)
    elif metric == "fhv":
        plot_kwargs["hline_at"] = 0.0
    elif metric == "flv":
        plot_kwargs["hline_at"] = 0.0
        plot_kwargs["yscale"] = "symlog"

    ylabel = metric.upper()
    if metric in ("fhv", "flv"):
        ylabel = f"{metric.upper()} (%)"

    out_path = EVAL_DIR / f"cdf_{metric}.png"
    fig = plot_metric_cdf(
        series,
        metric_name=ylabel,
        title=f"CDF of {ylabel} \u2014 5-Fold Validation",
        out_path=out_path,
        **plot_kwargs,
    )
    print(f"Saved CDF plot → {out_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# --cdf-vic-kge: CDF of KGE for LSTM vs VIC calibrated/regionalized
# ---------------------------------------------------------------------------

def _load_vic_cal_kge(path: Path, kge_col: str) -> pd.DataFrame:
    """Load a VIC calibration KGE CSV, keeping only numeric USGS basin IDs."""
    df = pd.read_csv(path)
    df = df.rename(columns={kge_col: "kge"})
    df["GageID"] = df["GageID"].astype(str)
    df = df[df["GageID"].str.match(r"^\d+$")].copy()
    df["basin_id"] = df["GageID"].astype(str)
    return df[["basin_id", "kge"]]


def run_cdf_vic_kge() -> None:
    """CDF of KGE: LSTM models vs VIC calibrated & regionalized on overlapping basins."""
    # Load VIC calibrated / regionalized
    cal = _load_vic_cal_kge(VIC_CAL_DIR / "calibrated_daily_kge.csv", "KGE")
    reg = _load_vic_cal_kge(VIC_CAL_DIR / "regionalized_daily_kge.csv", "daily_new_KGE")

    # Load LSTM eval metrics
    lstm_dfs: dict[str, pd.DataFrame] = {}
    for name in ("dual_lstm_kfold", "single_lstm_kfold"):
        csv_path = EVAL_DIR / f"{name}.csv"
        if not csv_path.exists():
            print(f"  WARNING: {csv_path} not found. Run --eval first for {name}.")
            continue
        df = pd.read_csv(csv_path)
        df["basin_id"] = df["basin_id"].astype(str)
        lstm_dfs[name] = df

    if not lstm_dfs:
        print("ERROR: No LSTM eval data found. Run --eval first.")
        sys.exit(1)

    # Find basins present in ALL datasets
    common = set(cal["basin_id"]) & set(reg["basin_id"])
    for df in lstm_dfs.values():
        common &= set(df["basin_id"])
    common = sorted(common)
    print(f"Overlapping basins across all datasets: {len(common)}")

    if len(common) < 2:
        print("ERROR: Too few overlapping basins.")
        sys.exit(1)

    # Build series on common basins
    series: dict[str, np.ndarray] = {}
    cal_sub = cal[cal["basin_id"].isin(common)].set_index("basin_id").loc[common]
    reg_sub = reg[reg["basin_id"].isin(common)].set_index("basin_id").loc[common]
    series["VIC Calibrated"] = cal_sub["kge"].values
    series["VIC Regionalized"] = reg_sub["kge"].values

    for name, df in lstm_dfs.items():
        sub = df[df["basin_id"].isin(common)].set_index("basin_id").loc[common]
        series[name] = sub["kge"].values

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    out_path = EVAL_DIR / "cdf_kge_vic_comparison.png"
    fig = plot_metric_cdf(
        series,
        metric_name="KGE",
        title=f"CDF of KGE \u2014 LSTM vs VIC ({len(common)} overlapping basins)",
        ylim=(-1.0, 1.0),
        out_path=out_path,
    )
    print(f"Saved CDF plot → {out_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Post-processing: compute evaluation metrics and generate plots.",
    )
    parser.add_argument(
        "--eval",
        nargs="+",
        metavar="RUN",
        help="LSTM output folder names (under data/training/output/) to evaluate. "
             "VIC simulated is always included automatically.",
    )
    parser.add_argument(
        "--cdf-metric",
        nargs="+",
        metavar="METRIC",
        help=f"Plot CDF for these metrics ({', '.join(METRICS)}). "
             f"Uses run names from --runs or --eval.",
    )
    parser.add_argument(
        "--runs",
        nargs="+",
        metavar="RUN",
        help="LSTM output folder names for CDF plotting (used with --cdf-metric). "
             "If omitted, falls back to --eval run names.",
    )
    parser.add_argument(
        "--cdf-vic-kge",
        action="store_true",
        help="Plot CDF of KGE comparing LSTM vs VIC calibrated/regionalized "
             "on overlapping basins.",
    )

    args = parser.parse_args()

    if args.eval is None and args.cdf_metric is None and not args.cdf_vic_kge:
        parser.print_help()
        sys.exit(1)

    if args.eval is not None:
        run_eval(args.eval)

    if args.cdf_metric is not None:
        runs = args.runs if args.runs else (args.eval or [])
        if not runs:
            print("ERROR: Provide run names via --runs or --eval.")
            sys.exit(1)
        for metric in args.cdf_metric:
            run_cdf(metric, runs)

    if args.cdf_vic_kge:
        run_cdf_vic_kge()


if __name__ == "__main__":
    main()
