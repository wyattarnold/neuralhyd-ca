"""Data loading, fold creation, normalisation, and PyTorch Dataset.

Typical call order::

    basin_data  = load_all_data(config)
    folds       = create_folds(basin_data, config)
    norm_stats  = compute_norm_stats(train_basins, basin_data, config)
    dataset     = HydroDataset(basin_data, basin_ids, norm_stats, config)

Key exports
-----------
load_all_data(config)
    Read climate CSVs, flow CSVs, and static attribute tables; return a
    dict mapping basin_id → combined DataFrame.
create_folds(basin_data, config)
    Produce ``config.n_folds`` stratified folds, keeping each tier
    (T1/T2/T3) proportionally represented in every held-out set.
compute_norm_stats(train_ids, basin_data, config)
    Compute global z-score statistics from training basins only
    (climate + static) and per-basin flow std for denormalisation.
HydroDataset
    ``torch.utils.data.Dataset`` returning 8-tuples:
    ``(x_dynamic, x_static, y_norm, y_components, basin_id, precip_mean,
    loss_weight, extreme_qs)``.  ``precip_mean`` is the per-basin
    denormalisation scale (mean daily precipitation, mm/day).
    ``loss_weight`` is the per-basin gradient-balancing weight
    (1 / var of flow/precip_mean).  ``extreme_qs`` is a (2,) tensor
    ``[y_q_start, y_q_top]`` giving per-basin start/end thresholds
    (normalised-flow units) for the extreme-flow aux-loss ramp.
"""

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


def load_static_attributes(
    basin_atlas_path: Path,
    climate_stats_path: Path,
    log_transform: List[str],
) -> Tuple[pd.DataFrame, pd.Series | None]:
    """Load and prepare static attributes from the two CSV sources.

    Joins BasinATLAS + Climate Statistics on ``PourPtID``, log-transforms
    the requested features (clipped at 1e-6), and fills remaining NaNs
    with the column median.  Returns ``(static_df, raw_area_km2)`` where
    ``raw_area_km2`` is the untransformed ``total_Shape_Area_km2`` series
    (needed for cfs ↔ mm/day conversion) or ``None`` if the column is
    absent.
    """
    basin_atlas = pd.read_csv(basin_atlas_path, index_col="PourPtID")
    climate_stats = pd.read_csv(climate_stats_path, index_col="PourPtID")
    static_df = basin_atlas.join(climate_stats, how="inner")

    raw_area_km2: pd.Series | None = None
    if "total_Shape_Area_km2" in static_df.columns:
        raw_area_km2 = static_df["total_Shape_Area_km2"].copy()

    for feat in log_transform:
        if feat in static_df.columns:
            static_df[feat] = np.log10(static_df[feat].clip(lower=1e-6))

    static_df = static_df.fillna(static_df.median(numeric_only=True))
    return static_df, raw_area_km2


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
    # 1. Load streamflow + tier from the zarr cube
    from src.data.io import load_flow_dataframes, load_climate_dataframes

    flow_data, tier_map = load_flow_dataframes(config.flow_zarr)
    basin_ids = sorted(tier_map.keys())

    # 2. Load daily climate from the zarr cube for every basin that has flow
    climate_dfs = load_climate_dataframes(
        config.climate_zarr, basin_ids=basin_ids, variables=config.dynamic_features
    )
    climate_data: Dict[int, pd.DataFrame] = {
        int(bid): df for bid, df in climate_dfs.items()
    }
    missing_climate = [b for b in basin_ids if b not in climate_data]
    if missing_climate:
        print(f"Warning: no climate data for {len(missing_climate)} basins – skipping them")
    basin_ids = [b for b in basin_ids if b in climate_data]

    # 3. Static attributes (merge tables on PourPtID)
    static_df, raw_area_km2 = load_static_attributes(
        config.static_basin_atlas, config.static_climate,
        config.log_transform_static,
    )

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
    * Scale   : per-basin mean daily precipitation (mm/day), computed from
                the basin's own climate record.  The flow target is
                normalised as ``y_norm = flow / precip_mean`` so the model
                learns a dimensionless runoff ratio.  This makes the
                denormalisation scale universally computable at inference
                time for ANY basin with a climate record — no reliance on
                observed flow data for unseen basins.
    """
    # --- climate ---
    chunks = [climate_data[b].values for b in train_ids]
    all_clim = np.concatenate(chunks, axis=0)
    clim_mean = all_clim.mean(axis=0).astype(np.float32)
    clim_std = all_clim.std(axis=0).astype(np.float32)

    # --- static ---
    eff_feats = config.effective_static_features
    # Features sourced from static_df vs computed from the climate window
    window_derived = set()
    if config.use_window_snow_fraction and "snow_fraction" in eff_feats:
        window_derived.add("snow_fraction")
    df_feats = [f for f in eff_feats if f not in window_derived]

    # Stats for CSV-based static features
    if df_feats:
        svs = static_df.loc[train_ids, df_feats].values.astype(np.float32)
        df_mean = svs.mean(axis=0)
        df_std = svs.std(axis=0)
    else:
        df_mean = np.array([], dtype=np.float32)
        df_std = np.array([], dtype=np.float32)

    # Stats for window-derived features (in order they appear in eff_feats)
    wd_means: List[np.float32] = []
    wd_stds: List[np.float32] = []
    if "snow_fraction" in window_derived:
        # Approximate training-set snow_fraction stats from full climate records
        basin_sf = []
        for bid in train_ids:
            cdf = climate_data[bid]
            tmean = (cdf["tmax_c"].values + cdf["tmin_c"].values) / 2.0
            precip = cdf["precip_mm"].values
            total_p = precip.sum()
            if total_p > 0:
                basin_sf.append(float(precip[tmean < 0.0].sum() / total_p))
            else:
                basin_sf.append(0.0)
        wd_means.append(np.float32(np.mean(basin_sf)))
        wd_stds.append(np.float32(np.std(basin_sf)))

    # Concatenate in the order of eff_feats
    stat_mean_parts: list = []
    stat_std_parts: list = []
    df_idx = 0
    wd_idx = 0
    for feat in eff_feats:
        if feat in window_derived:
            stat_mean_parts.append(wd_means[wd_idx])
            stat_std_parts.append(wd_stds[wd_idx])
            wd_idx += 1
        else:
            stat_mean_parts.append(df_mean[df_idx])
            stat_std_parts.append(df_std[df_idx])
            df_idx += 1
    stat_mean = np.array(stat_mean_parts, dtype=np.float32)
    stat_std = np.array(stat_std_parts, dtype=np.float32)

    # --- per-basin scale ---
    # The scale is the mean daily precipitation (mm/day), computed from each
    # basin's own climate record.  This is the static denorm factor; the
    # model's learned ScaleHead multiplies model outputs by an additional
    # per-basin factor on top, absorbing per-basin amplitude end-to-end.
    scale_map: Dict[int, np.float32] = {}
    precip_idx = config.dynamic_features.index("precip_mm")
    for bid in all_ids:
        if bid in climate_data:
            precip_vals = climate_data[bid].values[:, precip_idx]
            scale_map[bid] = np.float32(max(float(np.mean(precip_vals)), 0.01))
        else:
            scale_map[bid] = np.float32(1.0)

    # --- per-basin loss weight ---
    # Weight = 1 / max(var_b, min_var) ** exponent, where var_b is the
    # variance of the normalised target (flow / precip_mean) for basin b.
    # Balances per-basin gradient contributions; see Config fields
    # `basin_loss_weight_exponent` and `basin_loss_min_var` for rationale.
    loss_var_map: Dict[int, np.float32] = {}
    min_var = 0.1
    for bid in all_ids:
        if bid in flow_data and bid in climate_data:
            flow_vals = flow_data[bid]["flow"].dropna().values
            pmean = float(scale_map[bid])
            if len(flow_vals) > 1 and pmean > 0:
                var = float(np.var(flow_vals / pmean))
                loss_var_map[bid] = np.float32(max(var, min_var))
            else:
                loss_var_map[bid] = np.float32(1.0)
        else:
            loss_var_map[bid] = np.float32(1.0)

    # --- per-basin extreme-flow quantile thresholds ---
    # Computed on the *normalised* target (flow / scale_b) using only the
    # basin's observed days.  These are passed per-sample to the loss so
    # that "extreme" means the same thing at every basin (1-in-N-day event).
    # For basins without enough observed days, fall back to a conservative
    # default (start=3, top=8) on the normalised scale.
    q_start = float(config.extreme_start_quantile)
    q_top = float(config.extreme_top_quantile)
    y_q_start_map: Dict[int, np.float32] = {}
    y_q_top_map: Dict[int, np.float32] = {}
    for bid in all_ids:
        if bid in flow_data:
            flow_vals = flow_data[bid]["flow"].dropna().values
            s = float(scale_map[bid])
            if len(flow_vals) >= 100 and s > 0:
                y_norm = flow_vals / s
                y_q_start_map[bid] = np.float32(np.quantile(y_norm, q_start))
                y_q_top_map[bid] = np.float32(np.quantile(y_norm, q_top))
                continue
        y_q_start_map[bid] = np.float32(3.0)
        y_q_top_map[bid] = np.float32(8.0)

    return {
        "climate": (clim_mean, clim_std),
        "static": (stat_mean, stat_std),
        "scale": scale_map,
        "loss_var": loss_var_map,
        "y_q_start": y_q_start_map,
        "y_q_top": y_q_top_map,
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
# Lookback batching helper (shared by evaluate + simulate)
# ---------------------------------------------------------------------------


def make_lookback_batch(
    dynamic: torch.Tensor, indices: np.ndarray, seq_len: int,
) -> torch.Tensor:
    """Stack lookback windows ending at each index into a single tensor.

    Returns a (len(indices), seq_len, n_features) tensor.
    """
    return torch.stack(
        [dynamic[i - seq_len + 1 : i + 1] for i in indices]
    )


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------


class HydroDataset(Dataset):
    """Serves 8-tuples ``(x_dynamic, x_static, y_norm, y_components, basin_id, precip_mean, loss_weight, extreme_qs)``.

    x_dynamic     : (seq_len, n_dynamic) — normalised climate
    x_static      : (n_static,)          — normalised static attributes
    y_norm        : scalar               — flow / precip_mean (dimensionless)
    y_components  : (2,)                 — [quickflow, baseflow] / precip_mean
    basin_id      : int
    precip_mean   : scalar               — per-basin denormalisation scale (mm/day)
    loss_weight   : scalar               — per-basin gradient-balancing weight
    extreme_qs    : (2,)                 — [y_q_start, y_q_top] per-basin
                                           quantile thresholds (normalised flow
                                           units) for the aux-loss extreme ramp
    """

    def __init__(
        self,
        basin_ids: List[int],
        climate_data: Dict[int, pd.DataFrame],
        flow_data: Dict[int, pd.DataFrame],
        static_df: pd.DataFrame,
        config: Config,
        norm_stats: dict,
        device: torch.device | str | None = None,
    ):
        super().__init__()
        self.seq_len = config.seq_len
        self.use_window_snow_fraction = config.use_window_snow_fraction
        # Target device for the per-basin tensors that __getitem__ slices.
        # When set, the full dataset lives on-device and no host→device
        # transfers are needed per batch.  Must be None (or CPU) if the
        # DataLoader uses num_workers > 0 (workers can't share CUDA/MPS
        # tensors).  ``pmean_f`` / ``lw_f`` / ``extreme_qs_np`` remain on
        # CPU so legacy consumers of basin_data (e.g. evaluate_basin)
        # can still treat them as plain scalars / numpy.
        self.device = torch.device(device) if device is not None else None

        clim_mean, clim_std = norm_stats["climate"]
        stat_mean, stat_std = norm_stats["static"]

        # Determine which effective features come from static_df vs window
        eff_feats = config.effective_static_features
        window_derived = set()
        if self.use_window_snow_fraction and "snow_fraction" in eff_feats:
            window_derived.add("snow_fraction")
        df_feats = [f for f in eff_feats if f not in window_derived]

        # Build index mapping: for each effective feature, its position in
        # stat_mean / stat_std (which are ordered by eff_feats).
        self._eff_feats = eff_feats
        self._window_derived = window_derived
        # Positions of window-derived features in the final vector
        self._wd_positions: Dict[str, int] = {}
        for i, f in enumerate(eff_feats):
            if f in window_derived:
                self._wd_positions[f] = i
        # Norm stats for window-derived features (indexed by position)
        self._wd_mean = {f: stat_mean[i] for f, i in self._wd_positions.items()}
        self._wd_std = {f: stat_std[i] for f, i in self._wd_positions.items()}

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

            # per-basin denormalisation scale: mean daily precipitation (mm/day)
            pmean = norm_stats["scale"].get(bid, np.float32(1.0))
            # per-basin loss weight: 1 / max(var, min_var) ** exponent (see Config)
            lvar = float(norm_stats.get("loss_var", {}).get(bid, np.float32(1.0)))
            p = float(config.basin_loss_weight_exponent)
            min_var = float(config.basin_loss_min_var)
            lw = np.float32(1.0 / max(lvar, min_var) ** p) if p > 0 else np.float32(1.0)

            # normalised static vector (only CSV-sourced features)
            if df_feats:
                sv = static_df.loc[bid, df_feats].values.astype(np.float32)
            else:
                sv = np.array([], dtype=np.float32)

            # Build the full static vector with placeholders for window-derived
            sv_full = np.zeros(len(eff_feats), dtype=np.float32)
            df_idx = 0
            for i, f in enumerate(eff_feats):
                if f not in window_derived:
                    sv_full[i] = sv[df_idx]
                    df_idx += 1
                # window-derived slots stay 0 — filled per sample
            sv_norm = (sv_full - stat_mean) / (stat_std + 1e-8)

            bd_dict: dict = {
                "dynamic": torch.from_numpy(dynamic),
                "flow": flow_arr,                          # keep numpy for NaN checks
                "precip_mean": pmean,                      # denorm scale (mm/day)
                "static": torch.from_numpy(sv_norm),
                "precip_mean_tensor": torch.tensor(pmean, dtype=torch.float32),
                "loss_w_tensor": torch.tensor(lw, dtype=torch.float32),
                "extreme_qs_tensor": torch.tensor([
                    float(norm_stats.get("y_q_start", {}).get(bid, np.float32(3.0))),
                    float(norm_stats.get("y_q_top", {}).get(bid, np.float32(8.0))),
                ], dtype=torch.float32),
                "dates": dates,                            # climate DatetimeIndex
            }

            # Store raw precip & tmean for window snow_fraction computation
            if self.use_window_snow_fraction:
                bd_dict["precip_raw"] = cdf["precip_mm"].values.astype(np.float32)
                bd_dict["tmean_raw"] = (
                    (cdf["tmax_c"].values + cdf["tmin_c"].values) / 2.0
                ).astype(np.float32)

            self.basin_data[bid] = bd_dict

            # valid sample indices: observed flow & enough lookback
            valid = np.where(~np.isnan(flow_arr))[0]
            valid = valid[valid >= self.seq_len - 1]
            self.samples.extend((bid, int(i)) for i in valid)

        # Pre-compute normalised flow targets as a tensor per basin
        for bid, bd in self.basin_data.items():
            flow_np = bd["flow"]
            pmean = bd["precip_mean"]
            bd["flow_norm"] = torch.from_numpy(
                np.where(np.isnan(flow_np), 0.0, flow_np / (pmean + 1e-8)).astype(np.float32)
            )

            if config.aux_loss_weight > 0:
                # Lyne-Hollick separation: flow = quickflow + baseflow
                baseflow = _lyne_hollick_baseflow(flow_np, alpha=config.baseflow_alpha)
                quickflow = flow_np - baseflow

                scale = pmean + 1e-8
                bd["components_norm"] = torch.from_numpy(np.stack([
                    np.where(np.isnan(quickflow), 0.0, quickflow / scale),
                    np.where(np.isnan(baseflow), 0.0, baseflow / scale),
                ], axis=1).astype(np.float32))  # (T, 2)
            else:
                bd["components_norm"] = torch.zeros(len(flow_np), 2)

        # Optionally pre-move hot tensors to the training device so
        # __getitem__ returns on-device slices with zero H2D overhead.
        if self.device is not None and self.device.type != "cpu":
            hot_keys = (
                "dynamic", "static", "flow_norm", "components_norm",
                "precip_mean_tensor", "loss_w_tensor", "extreme_qs_tensor",
            )
            for bd in self.basin_data.values():
                for k in hot_keys:
                    if k in bd and isinstance(bd[k], torch.Tensor):
                        bd[k] = bd[k].to(self.device, non_blocking=True)

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

        # Compute window-derived static features on the fly
        if self.use_window_snow_fraction and "snow_fraction" in self._wd_positions:
            x_s = x_s.clone()
            start = tidx - self.seq_len + 1
            precip_win = bd["precip_raw"][start : tidx + 1]
            tmean_win = bd["tmean_raw"][start : tidx + 1]
            total_p = precip_win.sum()
            if total_p > 0:
                sf = float(precip_win[tmean_win < 0.0].sum() / total_p)
            else:
                sf = 0.0
            pos = self._wd_positions["snow_fraction"]
            std_val = float(self._wd_std["snow_fraction"])
            mean_val = float(self._wd_mean["snow_fraction"])
            x_s[pos] = (sf - mean_val) / (std_val + 1e-8)

        return (x_d, x_s, y, y_comp, bid,
                bd["precip_mean_tensor"], bd["loss_w_tensor"],
                bd["extreme_qs_tensor"])
