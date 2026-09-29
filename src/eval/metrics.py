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

    FHV, FeHV and FLV are always recomputed with the current
    ``src.lstm.loss`` definitions from the per-basin timeseries CSVs in
    each fold's ``timeseries/`` directory, so runs scored with older
    definitions match new ones.  Any other missing metric column is
    backfilled the same way.  Basins without a timeseries CSV keep their
    stored value (NaN for a missing column).
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

    # Recompute the FDC metrics (stored values may use older definitions)
    # plus any other missing metric column from the timeseries
    cols = [m for m in METRICS if m in _FDC_METRICS or m not in df.columns]
    return _recompute_metrics(df, output_dir, cols)


_METRIC_FN = {
    "nse": compute_nse,
    "kge": compute_kge,
    "fhv": compute_fhv,
    "fehv": compute_fehv,
    "flv": compute_flv,
}

_FDC_METRICS = ("fhv", "fehv", "flv")


def _recompute_metrics(
    df: pd.DataFrame, output_dir: Path, cols: List[str],
) -> pd.DataFrame:
    """Recompute *cols* from per-basin timeseries CSVs.

    Basins without a timeseries CSV keep their stored value (NaN when the
    column is missing).
    """
    # Build basin_id → timeseries path mapping across folds
    ts_map: Dict[str, Path] = {}
    for fold_dir in sorted(output_dir.glob("fold_*")):
        ts_dir = fold_dir / "timeseries"
        if not ts_dir.exists():
            continue
        for ts_csv in ts_dir.glob("*.csv"):
            ts_map[ts_csv.stem] = ts_csv

    for m in cols:
        if m not in df.columns:
            df[m] = float("nan")
    for i, bid in df["basin_id"].items():
        ts_path = ts_map.get(str(int(bid)))
        if ts_path is None:
            continue
        ts = pd.read_csv(ts_path, usecols=["obs", "pred"])
        obs = ts["obs"].values
        pred = ts["pred"].values
        for m in cols:
            df.at[i, m] = float(_METRIC_FN[m](obs, pred))
    return df


def recompute_fdc_metrics(df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    """Recompute FHV/FeHV/FLV in *df* from *output_dir*'s fold timeseries."""
    return _recompute_metrics(df, output_dir, list(_FDC_METRICS))


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
