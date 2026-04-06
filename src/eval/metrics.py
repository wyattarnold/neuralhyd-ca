"""Compute per-basin evaluation metrics from timeseries or pre-computed results."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from src.lstm.loss import compute_fhv, compute_flv, compute_kge, compute_nse

METRICS = ["nse", "kge", "fhv", "flv"]


# ---------------------------------------------------------------------------
# LSTM results — aggregate existing fold-level basin_results.csv
# ---------------------------------------------------------------------------


def load_lstm_fold_results(output_dir: Path) -> pd.DataFrame:
    """Load and concatenate per-fold basin_results.csv from an LSTM run.

    Returns a DataFrame with columns:
        basin_id, tier, nse, kge, fhv, flv, n_obs
    Each basin appears once (from its held-out validation fold).
    """
    all_fold = output_dir / "all_fold_results.csv"
    if all_fold.exists():
        return pd.read_csv(all_fold)
    # Fallback: gather from individual folds
    frames: List[pd.DataFrame] = []
    for fold_dir in sorted(output_dir.glob("fold_*")):
        csv = fold_dir / "basin_results.csv"
        if csv.exists():
            frames.append(pd.read_csv(csv))
    if not frames:
        raise FileNotFoundError(f"No basin_results.csv found in {output_dir}")
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# VIC Simulated — compute metrics from daily timeseries
# ---------------------------------------------------------------------------

_VIC_RUNOFF = Path("data/external/cec/VIC-Sim/aggregated/training_watersheds_runoff.csv")


def _load_vic_runoff(repo_root: Path) -> pd.DataFrame:
    """Load VIC simulated daily runoff (CFS, wide-form)."""
    return pd.read_csv(repo_root / _VIC_RUNOFF, parse_dates=["date"])


def _load_observed_flow(repo_root: Path) -> Dict[str, pd.DataFrame]:
    """Load observed daily flow (CFS) for all training watersheds.

    Returns {basin_id_str: DataFrame(date, flow)}.
    """
    flow_dir = repo_root / "data" / "training" / "flow"
    flows: Dict[str, pd.DataFrame] = {}
    for tier in (1, 2, 3):
        tier_dir = flow_dir / f"tier_{tier}"
        if not tier_dir.exists():
            continue
        for csv in tier_dir.glob("*_cleaned.csv"):
            bid = csv.stem.replace("_cleaned", "")
            df = pd.read_csv(csv, parse_dates=["date"])
            flows[bid] = df
    return flows


def _load_tier_map(repo_root: Path) -> Dict[str, int]:
    """Build basin_id -> tier mapping from directory structure."""
    flow_dir = repo_root / "data" / "training" / "flow"
    tier_map: Dict[str, int] = {}
    for tier in (1, 2, 3):
        tier_dir = flow_dir / f"tier_{tier}"
        if not tier_dir.exists():
            continue
        for csv in tier_dir.glob("*_cleaned.csv"):
            bid = csv.stem.replace("_cleaned", "")
            tier_map[bid] = tier
    return tier_map


def compute_vic_metrics(repo_root: Path) -> pd.DataFrame:
    """Compute NSE/KGE/FHV/FLV for VIC simulated runoff vs observed flow.

    Both VIC runoff and observed flow are in CFS — compared directly.
    Returns a DataFrame with columns: basin_id, tier, nse, kge, fhv, flv, n_obs
    """
    vic_df = _load_vic_runoff(repo_root)
    obs_flows = _load_observed_flow(repo_root)
    tier_map = _load_tier_map(repo_root)

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
            "flv": compute_flv(obs, sim),
            "n_obs": len(obs),
        })

    return pd.DataFrame(rows)
