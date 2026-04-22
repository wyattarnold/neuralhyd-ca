"""Simulate trained LSTM models over climate timeseries.

Produces per-basin daily streamflow (CFS) CSVs for a given set of
climate inputs and trained model checkpoints.

Output directory structure:
    data/eval/sim/<model>/<type>/<scenario>/<basin_id>.csv

Where:
    <model>    = training run name (e.g. dual_lstm_kfold)
    <type>     = input domain (training_watersheds, huc_8, huc_10)
    <scenario> = climate scenario (historical, ...)

Each CSV contains: date, q_total, q_fast, q_slow (all CFS, 1 decimal).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src.lstm.config import Config, load_config
from src.lstm.dataset import load_static_attributes, make_lookback_batch
from src.lstm.model import build_model
from src.lstm.train import load_checkpoint, pick_device


# mm/day → CFS: q_cfs = q_mm * area_km² / CFS_TO_MM_DAY_PER_KM2
# where CFS_TO_MM_DAY_PER_KM2 = 0.0283168 * 86400 * 1000 / 1e6 ≈ 2.44577
_MM_DAY_TO_CFS_FACTOR = 1.0 / (0.0283168 * 86400 * 1000 / 1e6)  # ≈ 0.40887


# ---------------------------------------------------------------------------
# Core simulation
# ---------------------------------------------------------------------------


@torch.no_grad()
def simulate_basin(
    model: torch.nn.Module,
    dynamic: torch.Tensor,
    static: torch.Tensor,
    dates: pd.DatetimeIndex,
    precip_mean: float,
    area_km2: float,
    seq_len: int,
    batch_size: int,
    device: torch.device,
) -> pd.DataFrame:
    """Run the model on every timestep with sufficient lookback.

    Parameters
    ----------
    dynamic     : (T, n_dynamic) normalised climate tensor
    static      : (n_static,) normalised static tensor
    dates       : DatetimeIndex of length T
    precip_mean : per-basin mean daily precipitation (mm/day) — denorm scale
    area_km2    : raw basin area for mm/day → CFS conversion

    Returns
    -------
    DataFrame with columns: date, q_total, q_fast, q_slow (all CFS, 1 decimal).
    """
    model.eval()
    T = len(dates)
    valid_idx = np.arange(seq_len - 1, T)
    if len(valid_idx) == 0:
        return pd.DataFrame(columns=["date", "q_total", "q_fast", "q_slow"])

    all_total: list[np.ndarray] = []
    all_fast: list[np.ndarray] = []
    all_slow: list[np.ndarray] = []

    for start in range(0, len(valid_idx), batch_size):
        batch_idx = valid_idx[start : start + batch_size]
        x_d = make_lookback_batch(dynamic, batch_idx, seq_len).to(device)
        x_s = static.unsqueeze(0).expand(len(batch_idx), -1).to(device)

        q_tot, q_f, q_sl = model(x_d, x_s)

        scale = float(precip_mean) + 1e-8
        all_total.append(q_tot.cpu().numpy() * scale)
        all_fast.append(q_f.cpu().numpy() * scale)
        all_slow.append(q_sl.cpu().numpy() * scale)

    # mm/day → CFS
    cfs_scale = area_km2 * _MM_DAY_TO_CFS_FACTOR
    return pd.DataFrame({
        "date": dates[valid_idx].strftime("%Y-%m-%d"),
        "q_total": np.round(np.clip(np.concatenate(all_total), 0, None) * cfs_scale, 1),
        "q_fast": np.round(np.concatenate(all_fast) * cfs_scale, 1),
        "q_slow": np.round(np.concatenate(all_slow) * cfs_scale, 1),
    })


@torch.no_grad()
def _simulate_basin_raw(
    model: torch.nn.Module,
    dynamic: torch.Tensor,
    static: torch.Tensor,
    precip_mean: float,
    seq_len: int,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run model and return denormalised (q_total, q_fast, q_slow) in mm/day.

    Like :func:`simulate_basin` but skips the CFS conversion and returns
    numpy arrays for ensemble aggregation.
    """
    model.eval()
    T = dynamic.shape[0]
    valid_idx = np.arange(seq_len - 1, T)
    if len(valid_idx) == 0:
        empty = np.array([], dtype=np.float32)
        return empty, empty, empty

    tot_chunks: list[np.ndarray] = []
    fast_chunks: list[np.ndarray] = []
    slow_chunks: list[np.ndarray] = []
    for start in range(0, len(valid_idx), batch_size):
        batch_idx = valid_idx[start : start + batch_size]
        x_d = make_lookback_batch(dynamic, batch_idx, seq_len).to(device)
        x_s = static.unsqueeze(0).expand(len(batch_idx), -1).to(device)
        q_tot, q_f, q_sl = model(x_d, x_s)
        scale = float(precip_mean) + 1e-8
        tot_chunks.append(q_tot.cpu().numpy() * scale)
        fast_chunks.append(q_f.cpu().numpy() * scale)
        slow_chunks.append(q_sl.cpu().numpy() * scale)

    return (
        np.clip(np.concatenate(tot_chunks), 0, None),
        np.concatenate(fast_chunks),
        np.concatenate(slow_chunks),
    )


# ---------------------------------------------------------------------------
# Data loading helpers (lighter than the training pipeline)
# ---------------------------------------------------------------------------


def _load_climate_data(
    config: Config,
    basin_ids: List[int],
) -> Dict[int, pd.DataFrame]:
    """Load raw daily climate for the requested basins."""
    climate: Dict[int, pd.DataFrame] = {}
    for bid in basin_ids:
        cpath = config.climate_dir / f"climate_{bid}.csv"
        if cpath.exists():
            cdf = pd.read_csv(cpath, parse_dates=["date"], index_col="date")
            climate[bid] = cdf[config.dynamic_features]
    return climate


def _load_static_df(config: Config) -> tuple[pd.DataFrame, pd.Series]:
    """Load and prepare static attributes (log-transform, NaN fill).

    Returns (static_df, raw_area_km2) where raw_area_km2 is indexed by
    PourPtID and contains the untransformed basin area.
    """
    return load_static_attributes(
        config.static_basin_atlas, config.static_climate,
        config.log_transform_static,
    )


def _discover_basins(config: Config) -> tuple[List[int], Dict[int, int]]:
    """Discover basin IDs and tier map from the flow directory structure."""
    tier_map: Dict[int, int] = {}
    for tier in (1, 2, 3):
        tier_dir = config.flow_dir / f"tier_{tier}"
        if not tier_dir.exists():
            continue
        for f in sorted(tier_dir.glob("*_cleaned.csv")):
            basin_id = int(f.stem.replace("_cleaned", ""))
            tier_map[basin_id] = tier
    basin_ids = sorted(tier_map.keys())
    return basin_ids, tier_map


def _normalise_basin(
    climate_df: pd.DataFrame,
    static_df: pd.DataFrame,
    basin_id: int,
    config: Config,
    norm_stats: dict,
) -> tuple[torch.Tensor, torch.Tensor, pd.DatetimeIndex, float]:
    """Normalise climate and static data for a single basin.

    Returns (dynamic_tensor, static_tensor, dates, scale).

    ``scale`` is the per-basin mean daily precipitation (mm/day), computed
    directly from the basin's climate record.  This works for any basin
    with a climate file \u2014 no observed flow required.
    """
    clim_mean, clim_std = norm_stats["climate"]
    stat_mean, stat_std = norm_stats["static"]

    # Climate → z-score
    dates = climate_df.index
    clim_vals = climate_df.values.astype(np.float32)
    clim_norm = (clim_vals - clim_mean) / (clim_std + 1e-8)
    dynamic = torch.from_numpy(clim_norm)

    # Static → z-score
    eff_feats = config.effective_static_features
    sv = static_df.loc[basin_id, eff_feats].values.astype(np.float32)
    sv_norm = (sv - stat_mean) / (stat_std + 1e-8)
    static = torch.from_numpy(sv_norm)

    # Per-basin denormalisation scale: mean daily precipitation (mm/day).
    # Always available from the basin's own climate record \u2014 universally
    # computable for training watersheds, HUC8, HUC10, or any new basin.
    precip_idx = config.dynamic_features.index("precip_mm")
    precip_vals = climate_df.values[:, precip_idx]
    scale = float(max(np.mean(precip_vals), 0.01))

    return dynamic, static, dates, scale


# ---------------------------------------------------------------------------
# Orchestrators
# ---------------------------------------------------------------------------


def simulate_training_watersheds(
    config_path: Path,
    output_base: Path,
    device: torch.device | None = None,
) -> Path:
    """Simulate all training watersheds using k-fold held-out models.

    For a k-fold run, each basin is simulated with the model from the
    fold where it was held out (unbiased).  Falls back to the first
    available checkpoint if basin_results.csv is missing.

    Parameters
    ----------
    config_path : Path to the experiment TOML (e.g. config_dual_lstm_kfold.toml)
    output_base : Root output directory (e.g. data/eval/sim/)
    device      : Torch device; auto-detected if None

    Returns
    -------
    Path to the scenario output directory containing per-basin CSVs.
    """
    config = load_config(config_path)
    run_name = config.output_dir.name  # e.g. "dual_lstm_kfold"

    if device is None:
        device = pick_device()

    out_dir = output_base / run_name / "training_watersheds" / "historical"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Discover basins and load shared data
    all_basin_ids, tier_map = _discover_basins(config)
    climate_data = _load_climate_data(config, all_basin_ids)
    static_df, raw_area_km2 = _load_static_df(config)

    # Keep only basins with both climate and static data
    basin_ids = [b for b in all_basin_ids if b in climate_data and b in static_df.index]

    # Map each basin → fold index using basin_results.csv
    fold_dirs = sorted(config.output_dir.glob("fold_*"))
    basin_to_fold: Dict[int, int] = {}
    for fold_idx, fold_dir in enumerate(fold_dirs):
        br = fold_dir / "basin_results.csv"
        if br.exists():
            df = pd.read_csv(br)
            for bid in df["basin_id"].values:
                basin_to_fold[int(bid)] = fold_idx

    # Group basins by fold so we load each checkpoint once
    fold_basins: Dict[int, List[int]] = {}
    unassigned: List[int] = []
    for bid in basin_ids:
        fidx = basin_to_fold.get(bid)
        if fidx is not None:
            fold_basins.setdefault(fidx, []).append(bid)
        else:
            unassigned.append(bid)

    # Assign unassigned basins to fold 0 (or first available)
    if unassigned and fold_dirs:
        fallback = 0
        fold_basins.setdefault(fallback, []).extend(unassigned)
        print(f"  {len(unassigned)} basins not in any fold — using fold {fallback}")

    total_basins = sum(len(v) for v in fold_basins.values())
    print(f"\nSimulating {total_basins} basins across {len(fold_basins)} folds → {out_dir}")

    # Simulate fold by fold
    for fold_idx in sorted(fold_basins):
        bids = fold_basins[fold_idx]
        ckpt_path = fold_dirs[fold_idx] / "best_model.pt"
        if not ckpt_path.exists():
            print(f"  WARNING: {ckpt_path} not found — skipping {len(bids)} basins")
            continue

        model = build_model(config).to(device)
        norm_stats = load_checkpoint(ckpt_path, model, device)
        if norm_stats is None:
            print(f"  WARNING: no norm_stats in {ckpt_path} — skipping")
            continue

        print(f"  Fold {fold_idx}: {len(bids)} basins")
        for bid in tqdm(bids, desc=f"    fold {fold_idx}", leave=False):
            dynamic, static, dates, precip_mean = _normalise_basin(
                climate_data[bid], static_df, bid, config, norm_stats,
            )
            area = float(raw_area_km2.get(bid, 1.0))
            result = simulate_basin(
                model, dynamic, static, dates, precip_mean, area,
                seq_len=config.seq_len,
                batch_size=config.batch_size,
                device=device,
            )
            result.to_csv(out_dir / f"{bid}.csv", index=False)

    print(f"  Done — {total_basins} basin CSVs written to {out_dir}")
    return out_dir


# ---------------------------------------------------------------------------
# Ensemble simulation (all folds × all basins)
# ---------------------------------------------------------------------------

def _discover_basins_from_climate(climate_dir: Path) -> List[int]:
    """Discover basin IDs from a climate directory (no tier structure needed)."""
    ids = []
    for f in sorted(climate_dir.glob("climate_*.csv")):
        bid = int(f.stem.replace("climate_", ""))
        ids.append(bid)
    return ids


def _load_climate_from_dir(
    climate_dir: Path,
    basin_ids: List[int],
    dynamic_features: List[str],
) -> Dict[int, pd.DataFrame]:
    """Load raw daily climate from an arbitrary directory."""
    climate: Dict[int, pd.DataFrame] = {}
    for bid in basin_ids:
        cpath = climate_dir / f"climate_{bid}.csv"
        if cpath.exists():
            cdf = pd.read_csv(cpath, parse_dates=["date"], index_col="date")
            climate[bid] = cdf[dynamic_features]
    return climate


def _load_static_from_paths(
    basin_atlas_path: Path,
    climate_stats_path: Path,
    log_transform: List[str],
) -> tuple[pd.DataFrame, pd.Series]:
    """Load and prepare static attributes from explicit CSV paths."""
    return load_static_attributes(basin_atlas_path, climate_stats_path, log_transform)


def simulate_ensemble(
    config_path: Path,
    output_base: Path,
    target: str = "watersheds",
    device: torch.device | None = None,
) -> Path:
    """Run all k-fold models on every basin and write ensemble mean/min/max.

    Parameters
    ----------
    config_path : Path to the experiment TOML
    output_base : Root output directory (e.g. data/eval/sim/)
    target      : "watersheds" or "huc8" — determines input data sources
    device      : Torch device; auto-detected if None

    Returns
    -------
    Path to the output directory containing per-basin ensemble CSVs.
    """
    from src.paths import get_target_paths

    config = load_config(config_path)
    run_name = config.output_dir.name

    if device is None:
        device = pick_device()

    # Resolve input paths based on target
    target_paths = get_target_paths(target)
    climate_dir = target_paths["climate_dir"]
    basin_atlas_path = target_paths["basin_atlas_output"]
    climate_stats_path = target_paths["climate_stats_output"]

    out_dir = output_base / run_name / target / "historical"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Discover basins and load shared data
    basin_ids = _discover_basins_from_climate(climate_dir)

    # For training_watersheds, restrict to the basins that actually have flow
    # data in the tiered flow directories (avoids processing extra basins that
    # have climate files but were not used for training).
    if target == "training_watersheds":
        flow_basin_ids: set[int] = set()
        for tier in (1, 2, 3):
            tier_dir = config.flow_dir / f"tier_{tier}"
            if tier_dir.exists():
                for f in tier_dir.glob("*_cleaned.csv"):
                    flow_basin_ids.add(int(f.stem.replace("_cleaned", "")))
        basin_ids = [b for b in basin_ids if b in flow_basin_ids]

    climate_data = _load_climate_from_dir(
        climate_dir, basin_ids, config.dynamic_features,
    )
    static_df, raw_area_km2 = _load_static_from_paths(
        basin_atlas_path, climate_stats_path, config.log_transform_static,
    )

    # Keep only basins with both climate and static data
    basin_ids = [b for b in basin_ids if b in climate_data and b in static_df.index]
    print(f"\n{len(basin_ids)} basins with climate + static data in target={target}")

    # Load all fold checkpoints
    fold_dirs = sorted(config.output_dir.glob("fold_*"))
    models: list[tuple[torch.nn.Module, dict]] = []
    for fold_dir in fold_dirs:
        ckpt_path = fold_dir / "best_model.pt"
        if not ckpt_path.exists():
            print(f"  WARNING: {ckpt_path} not found — skipping fold")
            continue
        model = build_model(config).to(device)
        norm_stats = load_checkpoint(ckpt_path, model, device)
        if norm_stats is None:
            print(f"  WARNING: no norm_stats in {ckpt_path} — skipping")
            continue
        models.append((model, norm_stats))

    n_folds = len(models)
    if n_folds == 0:
        print("ERROR: No valid checkpoints found.")
        return out_dir
    print(f"Loaded {n_folds} fold checkpoints")

    print(f"Simulating {len(basin_ids)} basins × {n_folds} folds → {out_dir}\n")

    for bid in tqdm(basin_ids, desc="  ensemble"):
        tot_arrays: list[np.ndarray] = []
        fast_arrays: list[np.ndarray] = []
        slow_arrays: list[np.ndarray] = []
        dates = None

        for model, norm_stats in models:
            dynamic, static, d, precip_mean = _normalise_basin(
                climate_data[bid], static_df, bid, config, norm_stats,
            )
            if dates is None:
                dates = d
            q_tot, q_f, q_sl = _simulate_basin_raw(
                model, dynamic, static, precip_mean,
                seq_len=config.seq_len,
                batch_size=config.batch_size,
                device=device,
            )
            tot_arrays.append(q_tot)
            fast_arrays.append(q_f)
            slow_arrays.append(q_sl)

        if dates is None or len(tot_arrays) == 0:
            continue

        # Stack and compute ensemble statistics (mm/day)
        stacked_tot = np.stack(tot_arrays, axis=0)   # (n_folds, T_valid)
        stacked_fast = np.stack(fast_arrays, axis=0)
        stacked_slow = np.stack(slow_arrays, axis=0)

        # Convert mm/day → CFS
        area = float(raw_area_km2.get(bid, 1.0))
        cfs_scale = area * _MM_DAY_TO_CFS_FACTOR
        valid_idx = np.arange(config.seq_len - 1, len(dates))

        pd.DataFrame({
            "date": dates[valid_idx].strftime("%Y-%m-%d"),
            "q_mean": np.round(stacked_tot.mean(axis=0) * cfs_scale, 1),
            "q_min": np.round(stacked_tot.min(axis=0) * cfs_scale, 1),
            "q_max": np.round(stacked_tot.max(axis=0) * cfs_scale, 1),
            "q_fast_mean": np.round(stacked_fast.mean(axis=0) * cfs_scale, 1),
            "q_slow_mean": np.round(stacked_slow.mean(axis=0) * cfs_scale, 1),
        }).to_csv(out_dir / f"{bid}.csv", index=False)

    print(f"  Done — {len(basin_ids)} basin CSVs written to {out_dir}")
    return out_dir
