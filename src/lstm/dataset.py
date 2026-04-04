"""Data loading, fold creation, and PyTorch Dataset."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from .config import Config

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_all_data(config: Config):
    """Load climate, streamflow, and static-attribute tables.

    Returns
    -------
    basin_ids : list[int]
    climate_data : dict[int, pd.DataFrame]   – indexed by date
    flow_data : dict[int, pd.DataFrame]       – indexed by date, column 'flow'
    static_df : pd.DataFrame                  – indexed by PourPtID
    tier_map : dict[int, int]                 – basin_id → tier (1/2/3)
    """
    # 1. Discover basins & tiers from the tiered flow directory
    tier_map: Dict[int, int] = {}
    flow_data: Dict[int, pd.DataFrame] = {}

    for tier in (1, 2, 3):
        tier_dir = config.flow_dir / f"tier_{tier}"
        if not tier_dir.exists():
            continue
        for f in sorted(tier_dir.glob("*_cleaned.csv")):
            basin_id = int(f.stem.replace("_cleaned", ""))
            tier_map[basin_id] = tier
            df = pd.read_csv(f, parse_dates=["date"], index_col="date")
            flow_data[basin_id] = df[["flow"]]  # keep only flow column

    basin_ids = sorted(tier_map.keys())

    # 2. Load area-weighted daily climate for every basin that has flow
    climate_data: Dict[int, pd.DataFrame] = {}
    missing_climate: list = []
    for bid in basin_ids:
        cpath = config.climate_dir / f"climate_{bid}.csv"
        if cpath.exists():
            cdf = pd.read_csv(cpath, parse_dates=["date"], index_col="date")
            climate_data[bid] = cdf[config.dynamic_features]
        else:
            missing_climate.append(bid)

    if missing_climate:
        print(f"Warning: no climate file for {len(missing_climate)} basins – skipping them")
    basin_ids = [b for b in basin_ids if b in climate_data]

    # 3. Static attributes (merge two tables on PourPtID)
    basin_atlas = pd.read_csv(config.static_basin_atlas, index_col="PourPtID")
    climate_stats = pd.read_csv(config.static_climate, index_col="PourPtID")
    static_df = basin_atlas.join(climate_stats, how="inner")

    # Save raw area before log-transform (needed for cfs → mm/day conversion)
    raw_area_km2: pd.Series | None = None
    if "total_Shape_Area_km2" in static_df.columns:
        raw_area_km2 = static_df["total_Shape_Area_km2"].copy()

    # Log-transform heavily skewed features
    for feat in config.log_transform_static:
        if feat in static_df.columns:
            static_df[feat] = np.log10(static_df[feat].clip(lower=1e-6))

    # Fill any remaining NaN with column median
    static_df = static_df.fillna(static_df.median(numeric_only=True))

    # Keep only basins present in all three sources
    basin_ids = [b for b in basin_ids if b in static_df.index]
    tier_map = {b: tier_map[b] for b in basin_ids}
    flow_data = {b: flow_data[b] for b in basin_ids}
    climate_data = {b: climate_data[b] for b in basin_ids}

    # Convert flow: cfs → mm/day
    # q_mm_day = q_cfs * 0.0283168 m³/cfs * 86400 s/day * 1000 mm/m
    #            / (area_km2 * 1e6 m²/km²)
    #          = q_cfs * 2.44577 / area_km2
    CFS_TO_MM_DAY = 0.0283168 * 86400 * 1000 / 1e6   # ≈ 2.44577
    if raw_area_km2 is not None:
        converted = 0
        for bid in basin_ids:
            if bid in raw_area_km2.index:
                area = float(raw_area_km2[bid])
                if area > 0:
                    fd = flow_data[bid].copy()
                    fd["flow"] = fd["flow"] * CFS_TO_MM_DAY / area
                    flow_data[bid] = fd
                    converted += 1
        print(f"  Converted {converted}/{len(basin_ids)} basins: cfs → mm/day")

    return basin_ids, climate_data, flow_data, static_df, tier_map


# ---------------------------------------------------------------------------
# Fold creation  –  stratified 80/20 repeated random splits
# ---------------------------------------------------------------------------


def create_folds(
    basin_ids: List[int],
    tier_map: Dict[int, int],
    flow_data: Dict[int, pd.DataFrame],
    n_folds: int = 5,
    holdout_fraction: float = 0.2,  # kept for signature compat; ignored
    seed: int = 42,
) -> List[Tuple[List[int], List[int]]]:
    """Return *n_folds* non-overlapping (train_ids, val_ids) pairs.

    Basins are partitioned into *n_folds* groups via stratified random
    assignment (stratified by tier) so that every basin appears in
    exactly one validation fold.  Within each tier the basins are
    shuffled randomly, then dealt round-robin into fold buckets.

    Prints per-fold basin counts and flow-day splits so the actual
    data-volume balance is visible.
    """
    rng = np.random.RandomState(seed)

    # Pre-compute valid flow-days per basin
    flow_days: Dict[int, int] = {}
    for b in basin_ids:
        flow_days[b] = int(flow_data[b]["flow"].dropna().shape[0])

    total_flow_days = sum(flow_days.values())

    # Stratified partition: shuffle each tier, deal round-robin into folds
    fold_val_sets: List[List[int]] = [[] for _ in range(n_folds)]
    for t in (1, 2, 3):
        tier_basins = [b for b in basin_ids if tier_map[b] == t]
        rng.shuffle(tier_basins)
        for i, b in enumerate(tier_basins):
            fold_val_sets[i % n_folds].append(b)

    # Build (train, val) pairs and report splits
    folds: List[Tuple[List[int], List[int]]] = []
    print(f"\n  Fold partition  ({n_folds} folds, {len(basin_ids)} basins, "
          f"{total_flow_days:,} total flow-days)")
    for k in range(n_folds):
        val_ids = fold_val_sets[k]
        val_set = set(val_ids)
        train_ids = [b for b in basin_ids if b not in val_set]

        val_days = sum(flow_days[b] for b in val_ids)
        train_days = sum(flow_days[b] for b in train_ids)
        tier_counts = {t: sum(1 for b in val_ids if tier_map[b] == t)
                       for t in (1, 2, 3)}
        print(f"    Fold {k + 1}: val {len(val_ids):>3d} basins "
              f"(T1={tier_counts[1]}, T2={tier_counts[2]}, T3={tier_counts[3]})  "
              f"| val {val_days:>8,} days ({val_days / total_flow_days:.1%})  "
              f"| train {train_days:>8,} days ({train_days / total_flow_days:.1%})")

        folds.append((train_ids, val_ids))

    return folds


# ---------------------------------------------------------------------------
# Normalisation statistics  (computed from training basins only)
# ---------------------------------------------------------------------------


def compute_norm_stats(
    climate_data: Dict[int, pd.DataFrame],
    flow_data: Dict[int, pd.DataFrame],
    static_df: pd.DataFrame,
    train_ids: List[int],
    all_ids: List[int],
    config: Config,
) -> dict:
    """Compute normalisation constants.

    * Climate : global (mean, std) across all training timesteps.
    * Static  : global (mean, std) across training basins.
    * Flow    : per-basin std (training basins from their data;
                held-out basins from their own data – standard PUB
                convention, no temporal information leak).
    """
    # --- climate ---
    chunks = [climate_data[b].values for b in train_ids]
    all_clim = np.concatenate(chunks, axis=0)
    clim_mean = all_clim.mean(axis=0).astype(np.float32)
    clim_std = all_clim.std(axis=0).astype(np.float32)

    # --- static ---
    svs = static_df.loc[train_ids, config.static_features].values.astype(np.float32)
    stat_mean = svs.mean(axis=0)
    stat_std = svs.std(axis=0)

    # --- per-basin flow std ---
    flow_std_map: Dict[int, np.float32] = {}
    for bid in all_ids:
        if bid in flow_data:
            vals = flow_data[bid]["flow"].dropna().values
            std = vals.std() if len(vals) > 1 else 1.0
            flow_std_map[bid] = np.float32(max(std, 0.01))
        else:
            flow_std_map[bid] = np.float32(1.0)

    return {
        "climate": (clim_mean, clim_std),
        "static": (stat_mean, stat_std),
        "flow": flow_std_map,
    }


# ---------------------------------------------------------------------------
# Baseflow separation (Lyne–Hollick recursive digital filter)
# ---------------------------------------------------------------------------


def _lyne_hollick_baseflow(flow: np.ndarray, alpha: float = 0.925,
                           n_passes: int = 3) -> np.ndarray:
    """Separate baseflow using the Lyne–Hollick recursive digital filter.

    Uses *n_passes* (forward–backward–forward) following Nathan & McMahon
    (1990) to eliminate the phase shift from single-pass filtering and
    reduce initialisation sensitivity.

    Processes contiguous non-NaN segments independently so that data
    gaps don't introduce filter artefacts.  Returns an array the same
    shape as *flow* with NaN preserved where the input is NaN.
    """
    n = len(flow)
    baseflow = np.full(n, np.nan, dtype=np.float32)
    valid = ~np.isnan(flow)
    if not valid.any():
        return baseflow

    # Find start/end of each contiguous non-NaN segment
    edges = np.diff(np.concatenate([[0], valid.astype(np.int8), [0]]))
    starts = np.where(edges == 1)[0]
    ends = np.where(edges == -1)[0]

    c = (1.0 + alpha) / 2.0

    for s, e in zip(starts, ends):
        seg = flow[s:e].astype(np.float64)
        bf = seg.copy()

        for p in range(n_passes):
            # Alternate direction: forward on even passes, backward on odd
            series = bf if p % 2 == 0 else bf[::-1]
            qf = np.zeros(len(series), dtype=np.float64)
            for t in range(1, len(series)):
                qf[t] = alpha * qf[t - 1] + c * (series[t] - series[t - 1])
                if qf[t] < 0.0:
                    qf[t] = 0.0
            result = series - qf
            np.clip(result, 0.0, series, out=result)
            if p % 2 == 0:
                bf = result
            else:
                bf = result[::-1]

        baseflow[s:e] = bf.astype(np.float32)

    return baseflow


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------


class HydroDataset(Dataset):
    """Serves (x_dynamic, x_static, y_norm, y_components, basin_id, flow_std) tuples.

    x_dynamic      : (seq_len, n_dynamic) – normalised climate
    x_static       : (n_static,)          – normalised static attributes
    y_norm         : scalar                – flow / basin_std
    y_components   : (2,)                  – [quickflow, baseflow] normalised
    basin_id       : int
    flow_std       : scalar                – for denormalisation at eval time
    """

    def __init__(
        self,
        basin_ids: List[int],
        climate_data: Dict[int, pd.DataFrame],
        flow_data: Dict[int, pd.DataFrame],
        static_df: pd.DataFrame,
        config: Config,
        norm_stats: dict,
    ):
        super().__init__()
        self.seq_len = config.seq_len

        clim_mean, clim_std = norm_stats["climate"]
        stat_mean, stat_std = norm_stats["static"]

        self.samples: List[Tuple[int, int]] = []   # (basin_id, time_index)
        self.basin_data: Dict[int, dict] = {}

        for bid in tqdm(basin_ids, desc="  preparing basins", leave=False):
            cdf = climate_data[bid]
            dates = cdf.index
            n_days = len(dates)

            # normalised climate
            clim = ((cdf.values.astype(np.float32) - clim_mean) /
                    (clim_std + 1e-8))

            dynamic = clim

            # flow aligned to the climate date axis
            flow_arr = np.full(n_days, np.nan, dtype=np.float32)
            if bid in flow_data:
                aligned = flow_data[bid]["flow"].reindex(dates)
                flow_arr = aligned.values.astype(np.float32)

            # per-basin flow std
            fstd = norm_stats["flow"].get(bid, np.float32(1.0))

            # normalised static vector
            sv = static_df.loc[bid, config.static_features].values.astype(np.float32)
            sv_norm = (sv - stat_mean) / (stat_std + 1e-8)

            # Pre-convert to tensors to avoid per-sample conversion overhead
            self.basin_data[bid] = {
                "dynamic": torch.from_numpy(dynamic),
                "flow": flow_arr,                          # keep numpy for NaN checks
                "flow_std": fstd,
                "static": torch.from_numpy(sv_norm),
                "fstd_tensor": torch.tensor(fstd, dtype=torch.float32),
            }

            # valid sample indices: observed flow & enough lookback
            valid = np.where(~np.isnan(flow_arr))[0]
            valid = valid[valid >= self.seq_len - 1]
            self.samples.extend((bid, int(i)) for i in valid)

        # Pre-compute normalised flow targets as a tensor per basin
        for bid, bd in self.basin_data.items():
            flow_np = bd["flow"]
            fstd = bd["flow_std"]
            bd["flow_norm"] = torch.from_numpy(
                np.where(np.isnan(flow_np), 0.0, flow_np / (fstd + 1e-8)).astype(np.float32)
            )

            if config.aux_loss_weight > 0:
                # Lyne-Hollick separation: flow = quickflow + baseflow
                baseflow = _lyne_hollick_baseflow(flow_np, alpha=config.baseflow_alpha)
                quickflow = flow_np - baseflow

                scale = fstd + 1e-8
                bd["components_norm"] = torch.from_numpy(np.stack([
                    np.where(np.isnan(quickflow), 0.0, quickflow / scale),
                    np.where(np.isnan(baseflow), 0.0, baseflow / scale),
                ], axis=1).astype(np.float32))  # (T, 2)
            else:
                bd["components_norm"] = torch.zeros(len(flow_np), 2)

    # ---- Dataset interface ----

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        bid, tidx = self.samples[idx]
        bd = self.basin_data[bid]

        x_d = bd["dynamic"][tidx - self.seq_len + 1 : tidx + 1]   # (seq_len, n_dynamic)
        x_s = bd["static"]                                         # (n_static,)
        y = bd["flow_norm"][tidx]                                  # scalar tensor
        y_comp = bd["components_norm"][tidx]                       # (2,) — [fast, slow]

        return (x_d, x_s, y, y_comp, bid, bd["fstd_tensor"])
