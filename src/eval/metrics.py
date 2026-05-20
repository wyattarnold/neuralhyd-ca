"""Compute per-basin evaluation metrics from timeseries or pre-computed results."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from src.data.io import load_flow_dataframes
from src.lstm.loss import compute_fhv, compute_fehv, compute_flv, compute_kge, compute_nse
from src.paths import FLOW_ZARR, VIC_RUNOFF_CSV

METRICS = ["nse", "kge", "fhv", "fehv", "flv"]


# ---------------------------------------------------------------------------
# LSTM results — aggregate existing fold-level basin_results.csv
# ---------------------------------------------------------------------------


def load_lstm_fold_results(output_dir: Path) -> pd.DataFrame:
    """Load and concatenate per-fold basin_results.csv from an LSTM run.

    Returns a DataFrame with columns:
        basin_id, tier, nse, kge, fhv, fehv, flv, n_obs
    Each basin appears once (from its held-out validation fold).

    If any expected metric columns are missing (e.g. fehv added after
    the run was trained), they are recomputed from the per-basin
    timeseries CSVs stored in each fold's ``timeseries/`` directory.
    """
    all_fold = output_dir / "all_fold_results.csv"
    if all_fold.exists():
        df = pd.read_csv(all_fold)
    else:
        # Fallback: gather from individual folds
        frames: List[pd.DataFrame] = []
        for fold_dir in sorted(output_dir.glob("fold_*")):
            csv = fold_dir / "basin_results.csv"
            if csv.exists():
                frames.append(pd.read_csv(csv))
        if not frames:
            raise FileNotFoundError(f"No basin_results.csv found in {output_dir}")
        df = pd.concat(frames, ignore_index=True)

    # Backfill any missing metric columns from timeseries
    missing = [m for m in METRICS if m not in df.columns]
    if missing:
        df = _backfill_metrics(df, output_dir, missing)
    return df


_METRIC_FN = {
    "nse": compute_nse,
    "kge": compute_kge,
    "fhv": compute_fhv,
    "fehv": compute_fehv,
    "flv": compute_flv,
}


def _backfill_metrics(
    df: pd.DataFrame, output_dir: Path, missing: List[str],
) -> pd.DataFrame:
    """Recompute *missing* metrics from per-basin timeseries CSVs."""
    # Build basin_id → timeseries path mapping across folds
    ts_map: Dict[str, Path] = {}
    for fold_dir in sorted(output_dir.glob("fold_*")):
        ts_dir = fold_dir / "timeseries"
        if not ts_dir.exists():
            continue
        for ts_csv in ts_dir.glob("*.csv"):
            ts_map[ts_csv.stem] = ts_csv

    new_cols: Dict[str, List[float]] = {m: [] for m in missing}
    for _, row in df.iterrows():
        bid = str(int(row["basin_id"]))
        ts_path = ts_map.get(bid)
        if ts_path is not None and ts_path.exists():
            ts = pd.read_csv(ts_path)
            obs = ts["obs"].values
            pred = ts["pred"].values
            for m in missing:
                new_cols[m].append(float(_METRIC_FN[m](obs, pred)))
        else:
            for m in missing:
                new_cols[m].append(float("nan"))

    for m in missing:
        df[m] = new_cols[m]
    return df


# ---------------------------------------------------------------------------
# VIC Simulated — compute metrics from daily timeseries
# ---------------------------------------------------------------------------


def _load_vic_runoff() -> pd.DataFrame:
    """Load VIC simulated daily runoff (CFS, wide-form)."""
    return pd.read_csv(VIC_RUNOFF_CSV, parse_dates=["date"])


def _load_observed_flow() -> Dict[str, pd.DataFrame]:
    """Load observed daily flow (CFS) for all training watersheds from flow.zarr.

    Returns {basin_id_str: DataFrame(date, flow)} with ``date`` as a column.
    """
    flow_dfs, _ = load_flow_dataframes(FLOW_ZARR)
    return {str(bid): df.reset_index() for bid, df in flow_dfs.items()}


def _load_tier_map() -> Dict[str, int]:
    """Build basin_id -> tier mapping from flow.zarr."""
    _, tier_map = load_flow_dataframes(FLOW_ZARR)
    return {str(bid): int(t) for bid, t in tier_map.items()}


def compute_vic_metrics() -> pd.DataFrame:
    """Compute NSE/KGE/FHV/FLV for VIC simulated runoff vs observed flow.

    Both VIC runoff and observed flow are in CFS — compared directly.
    Returns a DataFrame with columns: basin_id, tier, nse, kge, fhv, flv, n_obs
    """
    vic_df = _load_vic_runoff()
    obs_flows = _load_observed_flow()
    tier_map = _load_tier_map()

    vic_dates = vic_df.set_index("date")
    vic_basins = set(vic_dates.columns)

    rows: List[dict] = []
    for bid, obs_df in obs_flows.items():
        if bid not in vic_basins:
            continue
        merged = obs_df.set_index("date").join(
            vic_dates[[bid]].rename(columns={bid: "vic"}),
            how="inner",
        )
        merged = merged.dropna(subset=["flow", "vic"])
        if len(merged) < 10:
            continue

        obs = merged["flow"].values
        sim = merged["vic"].values

        rows.append({
            "basin_id": bid,
            "tier": tier_map.get(bid, 0),
            "nse": compute_nse(obs, sim),
            "kge": compute_kge(obs, sim),
            "fhv": compute_fhv(obs, sim),
            "fehv": compute_fehv(obs, sim),
            "flv": compute_flv(obs, sim),
            "n_obs": len(obs),
        })

    return pd.DataFrame(rows)
