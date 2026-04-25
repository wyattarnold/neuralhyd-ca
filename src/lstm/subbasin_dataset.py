"""Subbasin-mode data loading and PyTorch Dataset.

In subbasin mode the training unit is still ``(gauge, date)`` but each
sample carries a **padded stack of sub-basin inputs** (HUC10 or HUC12)
together with the aggregation weights needed to combine subbasin
predictions into the gauge prediction.

Aggregation (applied in the training loop, not here):

    Q_gauge_mmday = sum_i  w_ig * y_hat_i_mmday

where ``w_ig = A_ig / A_gauge`` (subbasin i's overlap area inside gauge
g, divided by gauge area) and ``y_hat_i_mmday`` is the model's raw,
area-normalised prediction for subbasin i in mm/day.  No per-subbasin
de-normalisation is needed because the model is trained directly against
the observed gauge flow in mm/day.  Loss is MSE in mm/day space.

Key exports
-----------
load_subbasin_data(config)
    Read climate + static for every in-scope subbasin, load gauge flows,
    and parse the gauge×subbasin intersect table.  Returns a dict of
    arrays used by :class:`SubbasinHydroDataset` and
    :func:`compute_subbasin_norm_stats`.
compute_subbasin_norm_stats(train_gauges, bundle, config)
    Per-subbasin climate z-score + static z-score from training-gauge-
    reachable subbasins; per-gauge loss weight and extreme-quantile
    thresholds computed in mm/day space on observed gauge flow.
SubbasinHydroDataset
    Yields 9-tuples (gauge × day):
    ``(x_dynamic_stack, x_static_stack, mask, weights,
       y_mmday, y_components_mmday, gauge_id, loss_weight, extreme_qs)``.
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
from .dataset import (
    _lyne_hollick_baseflow,
    load_static_attributes,
)

# cfs → mm/day for a 1 km² basin (same constant as gauge-mode dataset)
_CFS_TO_MM_DAY = 0.0283168 * 86400 * 1000 / 1e6   # ≈ 2.44577


# ---------------------------------------------------------------------------
# Bundle loading
# ---------------------------------------------------------------------------


def _load_subbasin_intersect(csv_path: Path, id_col: str) -> pd.DataFrame:
    """Load the gauge × subbasin overlap table with normalised column types.

    ``id_col`` is e.g. ``"huc10"`` or ``"huc12"`` and must match the column
    produced by :mod:`src.data.subbasin_gauge_intersect`.
    """
    df = pd.read_csv(csv_path, dtype={id_col: str})
    area_col = f"{id_col}_area_km2"
    req = {"PourPtID", id_col, "gauge_area_km2", area_col, "overlap_area_km2"}
    missing = req - set(df.columns)
    if missing:
        raise RuntimeError(f"{csv_path} is missing columns: {missing}")
    df["PourPtID"] = df["PourPtID"].astype(int)
    df[id_col] = df[id_col].astype(str)
    return df


def _load_subbasin_statics(
    physical_csv: Path,
    climate_csv: Path,
    log_transform: List[str],
) -> pd.DataFrame:
    """Join subbasin static attributes with subbasin climate statistics.

    Mirrors :func:`load_static_attributes` but keeps subbasin IDs (strings)
    as the index.  Both input files use ``PourPtID`` as the ID column (the
    target-aware pipeline reuses the same column name for any polygon
    target).
    """
    phys = pd.read_csv(physical_csv, dtype={"PourPtID": str})
    phys = phys.set_index("PourPtID")
    clim = pd.read_csv(climate_csv, dtype={"PourPtID": str})
    clim = clim.set_index("PourPtID")
    df = phys.join(clim, how="inner")

    for feat in log_transform:
        if feat in df.columns:
            df[feat] = np.log10(df[feat].clip(lower=1e-6))

    df = df.fillna(df.median(numeric_only=True))
    return df


def load_subbasin_data(config: Config) -> dict:
    """Read the full subbasin-mode data bundle.

    Returns a dict with:
        gauge_ids        : list[int]
        gauge_tier       : dict[int, int]
        gauge_flow       : dict[int, pd.DataFrame] (column 'flow' in mm/day)
        gauge_area       : dict[int, float]    (km²)
        gauge_subbasins  : dict[int, list[str]]   ordered largest-overlap first
        gauge_weights    : dict[int, np.ndarray]  w_ig = A_ig / A_gauge
        subbasin_climate : dict[str, pd.DataFrame]  per-subbasin daily climate
        subbasin_static  : pd.DataFrame             (index = subbasin id str)
        n_max_subbasins  : int  (largest stack size; used for padding)
    """
    if (
        config.subbasin_intersect_csv is None
        or config.subbasin_climate_zarr is None
        or config.subbasin_static_attrs is None
        or config.subbasin_climate_stats is None
    ):
        raise ValueError("subbasin config paths must be set; see Config docstring")
    id_col = config.subbasin_level  # "huc10" or "huc12"
    area_col = f"{id_col}_area_km2"

    # ----- Gauge tier + flow (cfs) -----
    from src.data.io import load_flow_dataframes, load_climate_dataframes

    flow_cfs, tier_map = load_flow_dataframes(config.flow_zarr)

    # ----- Gauge static (for gauge area in km²) -----
    _, raw_area_km2 = load_static_attributes(
        config.static_basin_atlas, config.static_climate,
        config.log_transform_static,
    )
    if raw_area_km2 is None:
        raise RuntimeError(
            "Gauge static attributes must include 'total_Shape_Area_km2'"
        )

    # ----- Gauge × subbasin intersect -----
    isect = _load_subbasin_intersect(config.subbasin_intersect_csv, id_col)
    kept_gauges = set(isect["PourPtID"].unique())

    # Restrict gauges to those kept by the filter AND present in flow + area
    gauge_ids = sorted(
        b for b in kept_gauges
        if b in tier_map and b in flow_cfs and b in raw_area_km2.index
    )
    dropped = sorted(kept_gauges - set(gauge_ids))
    if dropped:
        print(f"  subbasin: {len(dropped)} gauges in intersect table "
              f"lacked flow/area; dropped: {dropped[:8]}{'...' if len(dropped) > 8 else ''}")

    # Convert gauge flow cfs → mm/day using the gauge area
    gauge_flow: Dict[int, pd.DataFrame] = {}
    gauge_area: Dict[int, float] = {}
    for bid in gauge_ids:
        area_km2 = float(raw_area_km2[bid])
        gauge_area[bid] = area_km2
        fd = flow_cfs[bid].copy()
        fd["flow"] = fd["flow"] * _CFS_TO_MM_DAY / area_km2
        gauge_flow[bid] = fd
    print(f"  subbasin: {len(gauge_ids)} gauges with valid flow+area "
          f"(T1={sum(1 for b in gauge_ids if tier_map[b]==1)}, "
          f"T2={sum(1 for b in gauge_ids if tier_map[b]==2)}, "
          f"T3={sum(1 for b in gauge_ids if tier_map[b]==3)})")

    # Build gauge → ordered subbasin list + weights (A_ig / A_gauge)
    gauge_subbasins: Dict[int, List[str]] = {}
    gauge_weights: Dict[int, np.ndarray] = {}
    needed_subbasins: set = set()
    n_max_subbasins = 0
    for bid, grp in isect.groupby("PourPtID"):
        if bid not in gauge_flow:
            continue
        grp = grp.sort_values("overlap_area_km2", ascending=False)
        hucs = grp[id_col].tolist()
        w = (grp["overlap_area_km2"].values / gauge_area[bid]).astype(np.float32)
        gauge_subbasins[bid] = hucs
        gauge_weights[bid] = w
        needed_subbasins.update(hucs)
        n_max_subbasins = max(n_max_subbasins, len(hucs))

    # ----- subbasin static attributes -----
    subbasin_static = _load_subbasin_statics(
        config.subbasin_static_attrs, config.subbasin_climate_stats,
        config.log_transform_static,
    )
    missing_static = [h for h in needed_subbasins if h not in subbasin_static.index]
    if missing_static:
        raise RuntimeError(
            f"{len(missing_static)} {id_col} values required by gauges are missing from "
            f"{config.subbasin_static_attrs.name}. First few: {missing_static[:5]}"
        )

    # ----- subbasin climate -----
    needed_list = sorted(needed_subbasins)
    print(f"  loading {id_col} climate from zarr: {len(needed_list)} subbasins")
    subbasin_climate_raw = load_climate_dataframes(
        config.subbasin_climate_zarr,
        basin_ids=needed_list,
        variables=config.dynamic_features,
    )
    subbasin_climate: Dict[str, pd.DataFrame] = {
        str(k): v for k, v in subbasin_climate_raw.items()
    }
    missing_clim = [h for h in needed_subbasins if h not in subbasin_climate]
    if missing_clim:
        raise RuntimeError(
            f"{len(missing_clim)} {id_col} climate records missing in "
            f"{config.subbasin_climate_zarr}. First few: {missing_clim[:5]}. "
            f"Run data-prep step for subbasin mode first."
        )

    # Keep only gauges whose subbasins are all present in climate (should be all
    # after the check above, but be defensive).
    ok_gauges = [
        b for b in gauge_ids
        if all(h in subbasin_climate for h in gauge_subbasins[b])
    ]
    if len(ok_gauges) < len(gauge_ids):
        print(f"  subbasin: dropped {len(gauge_ids) - len(ok_gauges)} gauges "
              f"missing at least one {id_col} climate file")
    gauge_ids = ok_gauges

    return {
        "gauge_ids": gauge_ids,
        "gauge_tier": {b: tier_map[b] for b in gauge_ids},
        "gauge_flow": {b: gauge_flow[b] for b in gauge_ids},
        "gauge_area": {b: gauge_area[b] for b in gauge_ids},
        "gauge_subbasins": {b: gauge_subbasins[b] for b in gauge_ids},
        "gauge_weights":   {b: gauge_weights[b] for b in gauge_ids},
        "subbasin_climate": subbasin_climate,
        "subbasin_static":  subbasin_static,
        "n_max_subbasins":  n_max_subbasins,
    }


# ---------------------------------------------------------------------------
# Normalisation stats (subbasin variant)
# ---------------------------------------------------------------------------


def compute_subbasin_norm_stats(
    train_gauges: List[int],
    bundle: dict,
    config: Config,
) -> dict:
    """Compute normalisation constants for subbasin mode.

    * Climate : global (mean, std) across all training-gauge-reachable subbasins
                (every subbasin that contributes to a training gauge).
    * Static  : global (mean, std) across those subbasins.
    * Gauge   : per-gauge observed-flow variance (mm/day) → loss weight;
                per-gauge quantile thresholds (mm/day) → extreme-aux ramp.

    The model is trained directly on mm/day targets, so no per-subbasin
    scale factor is produced here.  The jointly-learned ``ScaleHead`` on
    the static embedding absorbs per-subbasin amplitude end-to-end.
    """
    gauge_subbasins = bundle["gauge_subbasins"]
    subbasin_climate = bundle["subbasin_climate"]
    subbasin_static = bundle["subbasin_static"]
    gauge_flow = bundle["gauge_flow"]

    # Subbasins reachable via training gauges
    train_subbasins = sorted({
        h for b in train_gauges for h in gauge_subbasins[b]
    })

    # ----- climate z-score -----
    chunks = [subbasin_climate[h].values for h in train_subbasins]
    all_clim = np.concatenate(chunks, axis=0)
    clim_mean = all_clim.mean(axis=0).astype(np.float32)
    clim_std = all_clim.std(axis=0).astype(np.float32)

    # ----- static z-score -----
    eff_feats = config.effective_static_features
    # Window-derived features (snow_fraction) are not supported yet in subbasin
    # mode; enforce via config if/when they are.
    if config.use_window_snow_fraction:
        raise NotImplementedError(
            "use_window_snow_fraction is not yet supported in subbasin mode"
        )
    svs = subbasin_static.loc[train_subbasins, eff_feats].values.astype(np.float32)
    stat_mean = svs.mean(axis=0)
    stat_std = svs.std(axis=0)

    # ----- per-gauge loss weight (mm/day space) -----
    loss_var_map: Dict[int, np.float32] = {}
    for bid, fdf in gauge_flow.items():
        vals = fdf["flow"].dropna().values
        if len(vals) > 1:
            loss_var_map[bid] = np.float32(
                max(float(np.var(vals)), config.basin_loss_min_var)
            )
        else:
            loss_var_map[bid] = np.float32(1.0)

    # ----- per-gauge extreme quantile thresholds (mm/day) -----
    q_start = float(config.extreme_start_quantile)
    q_top = float(config.extreme_top_quantile)
    y_q_start_map: Dict[int, np.float32] = {}
    y_q_top_map: Dict[int, np.float32] = {}
    for bid, fdf in gauge_flow.items():
        vals = fdf["flow"].dropna().values
        if len(vals) >= 100:
            y_q_start_map[bid] = np.float32(np.quantile(vals, q_start))
            y_q_top_map[bid] = np.float32(np.quantile(vals, q_top))
        else:
            # Conservative defaults — scale by per-gauge mean flow so the
            # ramp still exists for data-poor basins.
            mean_flow = float(np.nanmean(vals)) if len(vals) else 1.0
            y_q_start_map[bid] = np.float32(max(3.0 * mean_flow, 0.1))
            y_q_top_map[bid] = np.float32(max(8.0 * mean_flow, 0.3))

    return {
        "climate": (clim_mean, clim_std),
        "static": (stat_mean, stat_std),
        "loss_var_gauge": loss_var_map,    # int → float
        "y_q_start_gauge": y_q_start_map,  # int → float (mm/day)
        "y_q_top_gauge":   y_q_top_map,    # int → float (mm/day)
    }


# ---------------------------------------------------------------------------
# Fold creation — nesting-aware stratified split
# ---------------------------------------------------------------------------


def _nesting_groups(gauge_subbasins: Dict[int, List[str]]) -> List[List[int]]:
    """Group gauges that share any subbasin via union-find.

    Two gauges are placed in the same group whenever their subbasin sets
    intersect — whether by strict nesting (one catchment fully contains
    the other) or partial overlap (sharing one or more subbasins without
    containment).  Either case constitutes input leakage: a HUC12 forcing
    tensor that drives one gauge in training would also drive the other
    in validation.  Transitivity is handled by union-find.
    """
    gauges = list(gauge_subbasins.keys())
    sub_sets = {g: set(hs) for g, hs in gauge_subbasins.items()}

    parent = list(range(len(gauges)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        parent[find(x)] = find(y)

    for i, a in enumerate(gauges):
        for j, b in enumerate(gauges[i + 1:], start=i + 1):
            if sub_sets[a] & sub_sets[b]:   # any shared subbasin = leakage
                union(i, j)

    from collections import defaultdict
    grp: Dict[int, List[int]] = defaultdict(list)
    for idx, g in enumerate(gauges):
        grp[find(idx)].append(g)
    return list(grp.values())


def create_subbasin_folds(
    gauge_ids: List[int],
    tier_map: Dict[int, int],
    gauge_flow: Dict[int, pd.DataFrame],
    gauge_subbasins: Dict[int, List[str]],
    n_folds: int = 5,
    seed: int = 42,
) -> List[Tuple[List[int], List[int]]]:
    """Stratified k-fold on gauge groups, respecting shared catchments.

    Gauges whose subbasin sets share **any** subbasin \u2014 whether by strict
    containment (nesting) or partial overlap \u2014 are grouped together and
    always assigned to the same fold.  This prevents the model from seeing
    a subbasin's forcing inputs during training while those same forcings
    contribute to a different gauge in the validation set.

    Within each tier the groups are stratified using LPT (longest-
    processing-time) deal: groups are sorted by total observed flow-days
    descending, then each is greedily assigned to the fold with the fewest
    accumulated days (ties broken by gauge count, then random).  The tier
    representative of a multi-gauge group is the tier of the member with
    the most observed flow-days.
    """
    rng = np.random.RandomState(seed)

    flow_days: Dict[int, int] = {
        b: int(gauge_flow[b]["flow"].dropna().shape[0]) for b in gauge_ids
    }
    total_flow_days = sum(flow_days.values())

    # Build nesting groups from the subbasin map
    subs_kept = {g: gauge_subbasins[g] for g in gauge_ids if g in gauge_subbasins}
    groups = _nesting_groups(subs_kept)
    # Add singletons for any gauge absent from the subbasin map (defensive)
    grouped_ids = {g for grp in groups for g in grp}
    for g in gauge_ids:
        if g not in grouped_ids:
            groups.append([g])

    nested = [grp for grp in groups if len(grp) > 1]
    if nested:
        print(f"\n  Nested gauge groups: {len(nested)} groups, "
              f"{sum(len(g) for g in nested)} gauges kept together")
        for grp in sorted(nested, key=len, reverse=True):
            print(f"    {sorted(grp)}")
    else:
        print("\n  No nested gauge groups detected — all gauges are independent.")

    def group_tier(grp: List[int]) -> int:
        """Tier of the member with most observed flow-days."""
        return tier_map[max(grp, key=lambda g: flow_days.get(g, 0))]

    # Stratified LPT (longest-processing-time) deal: within each tier,
    # sort groups by flow-days descending and greedily assign each group
    # to the fold with the fewest flow-days accumulated so far (ties
    # broken by fewer gauges, then random).  This balances both gauge
    # count and total flow-days far better than round-robin, while still
    # respecting tier stratification and never splitting a nested group.
    fold_val_groups: List[List[List[int]]] = [[] for _ in range(n_folds)]
    fold_days = [0] * n_folds
    fold_gauges = [0] * n_folds

    def group_days(grp: List[int]) -> int:
        return sum(flow_days.get(g, 0) for g in grp)

    for t in (1, 2, 3):
        tier_groups = [grp for grp in groups if group_tier(grp) == t]
        # Randomise first (tie-break jitter), then stable-sort desc on days
        rng.shuffle(tier_groups)
        tier_groups.sort(key=group_days, reverse=True)
        for grp in tier_groups:
            # Pick fold minimising (days, gauges, random jitter)
            jitter = rng.rand(n_folds)
            target = min(
                range(n_folds),
                key=lambda k: (fold_days[k], fold_gauges[k], jitter[k]),
            )
            fold_val_groups[target].append(grp)
            fold_days[target] += group_days(grp)
            fold_gauges[target] += len(grp)

    folds: List[Tuple[List[int], List[int]]] = []
    print(f"\n  Fold partition  ({n_folds} folds, {len(gauge_ids)} gauges in "
          f"{len(groups)} groups, {total_flow_days:,} total flow-days)")
    for k in range(n_folds):
        val_ids = [g for grp in fold_val_groups[k] for g in grp]
        val_set = set(val_ids)
        train_ids = [b for b in gauge_ids if b not in val_set]
        val_days = sum(flow_days[b] for b in val_ids)
        train_days = sum(flow_days[b] for b in train_ids)
        tier_counts = {t: sum(1 for b in val_ids if tier_map[b] == t)
                       for t in (1, 2, 3)}
        print(f"    Fold {k + 1}: val {len(val_ids):>3d} gauges "
              f"(T1={tier_counts[1]}, T2={tier_counts[2]}, T3={tier_counts[3]})  "
              f"| val {val_days:>8,} days ({val_days / total_flow_days:.1%})  "
              f"| train {train_days:>8,} days ({train_days / total_flow_days:.1%})")
        folds.append((train_ids, val_ids))

    return folds


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class SubbasinHydroDataset(Dataset):
    """Per-gauge daily samples with padded subbasin input stacks.

    ``__getitem__`` returns a 9-tuple

        ( x_dynamic_stack   (N_max, seq_len, n_dynamic),
          x_static_stack    (N_max, n_static),
          mask              (N_max,)    1.0 for real subbasin, 0 for padding,
          weights           (N_max,)    A_ig / A_gauge (0 for padding),
          y_mmday           scalar      observed gauge flow (mm/day),
          y_comp_mmday      (2,)        Lyne–Hollick [fast, slow] mm/day,
          gauge_id          int,
          loss_weight       scalar,
          extreme_qs        (2,) mm/day [q_start, q_top] ).

    Padding rows are filled with zeros (mask = 0, weight = 0) so the
    aggregation sum is unaffected.  The model is applied to the entire
    flattened stack; padded rows still run through the LSTM (unavoidable
    without dynamic batching) but contribute nothing to the loss.
    """

    def __init__(
        self,
        gauge_ids: List[int],
        bundle: dict,
        norm_stats: dict,
        config: Config,
        device: torch.device | str | None = None,
    ):
        super().__init__()
        self.seq_len = config.seq_len
        self.n_max_subbasins = bundle["n_max_subbasins"]
        self.device = torch.device(device) if device is not None else None

        clim_mean, clim_std = norm_stats["climate"]
        stat_mean, stat_std = norm_stats["static"]
        eff_feats = config.effective_static_features

        subbasin_climate = bundle["subbasin_climate"]
        subbasin_static = bundle["subbasin_static"]
        gauge_flow = bundle["gauge_flow"]
        gauge_subbasins = bundle["gauge_subbasins"]
        gauge_weights = bundle["gauge_weights"]
        loss_var_gauge = norm_stats["loss_var_gauge"]
        y_q_start_gauge = norm_stats["y_q_start_gauge"]
        y_q_top_gauge = norm_stats["y_q_top_gauge"]

        # ----- Cache normalised subbasin dynamics & static once -----
        self._sub_dyn: Dict[str, torch.Tensor] = {}
        self._sub_static: Dict[str, torch.Tensor] = {}
        self._sub_dates: Dict[str, pd.DatetimeIndex] = {}
        for h, cdf in subbasin_climate.items():
            dyn = ((cdf.values.astype(np.float32) - clim_mean) /
                   (clim_std + 1e-8))
            self._sub_dyn[h] = torch.from_numpy(dyn)
            sv = subbasin_static.loc[h, eff_feats].values.astype(np.float32)
            sv = (sv - stat_mean) / (stat_std + 1e-8)
            self._sub_static[h] = torch.from_numpy(sv)
            self._sub_dates[h] = cdf.index

        # ----- Build per-gauge bundle (flow + aligned subbasin indexing) -----
        self.samples: List[Tuple[int, int]] = []
        self.gauge_data: Dict[int, dict] = {}

        # For each gauge, choose the subbasin climate date axis as the common
        # time index.  All subbasins contributing to one gauge come from the
        # same climate source (VIC), so their date ranges are identical.
        # We use the first subbasin's dates as the reference.
        for bid in tqdm(gauge_ids, desc="  preparing gauges", leave=False):
            hucs = gauge_subbasins[bid]
            weights = gauge_weights[bid]
            ref_dates = self._sub_dates[hucs[0]]
            n_days = len(ref_dates)

            # Align flow to ref_dates
            fdf = gauge_flow[bid]["flow"].reindex(ref_dates)
            flow_arr = fdf.values.astype(np.float32)

            # Aux loss components (mm/day)
            if config.aux_loss_weight > 0:
                baseflow = _lyne_hollick_baseflow(flow_arr, alpha=config.baseflow_alpha)
                quickflow = flow_arr - baseflow
                y_comp = np.stack([
                    np.where(np.isnan(quickflow), 0.0, quickflow),
                    np.where(np.isnan(baseflow),  0.0, baseflow),
                ], axis=1).astype(np.float32)
            else:
                y_comp = np.zeros((n_days, 2), dtype=np.float32)

            # Per-gauge subbasin metadata (unpadded; per-batch padding
            # is handled by ``subbasin_collate``).
            k = len(hucs)
            n_static = len(eff_feats)
            static_stack = torch.zeros((k, n_static), dtype=torch.float32)
            for i, h in enumerate(hucs):
                static_stack[i] = self._sub_static[h]
            w_real = torch.from_numpy(weights.astype(np.float32))

            lvar = float(loss_var_gauge.get(bid, np.float32(1.0)))
            p_exp = float(config.basin_loss_weight_exponent)
            min_var = float(config.basin_loss_min_var)
            lw = np.float32(1.0 / max(lvar, min_var) ** p_exp) if p_exp > 0 else np.float32(1.0)

            bd = {
                "flow": flow_arr,
                "flow_tensor": torch.from_numpy(
                    np.where(np.isnan(flow_arr), 0.0, flow_arr).astype(np.float32)
                ),
                "y_comp_tensor": torch.from_numpy(y_comp),
                "weights":  w_real,           # (k,) unpadded
                "static_stack": static_stack, # (k, n_static) unpadded
                "k": k,                       # number of real subbasins
                "subbasin_ids": hucs,       # for dynamic slicing in __getitem__
                "loss_w_tensor": torch.tensor(lw, dtype=torch.float32),
                "extreme_qs_tensor": torch.tensor([
                    float(y_q_start_gauge.get(bid, 0.1)),
                    float(y_q_top_gauge.get(bid, 0.3)),
                ], dtype=torch.float32),
                "dates": ref_dates,
            }
            self.gauge_data[bid] = bd

            # Valid sample indices: observed flow & enough lookback
            valid = np.where(~np.isnan(flow_arr))[0]
            valid = valid[valid >= self.seq_len - 1]
            self.samples.extend((bid, int(i)) for i in valid)

        # ----- Optionally pre-move hot tensors to device -----
        if self.device is not None and self.device.type != "cpu":
            hot = ("flow_tensor", "y_comp_tensor",
                   "weights", "static_stack",
                   "loss_w_tensor", "extreme_qs_tensor")
            for bd in self.gauge_data.values():
                for hk in hot:
                    bd[hk] = bd[hk].to(self.device, non_blocking=True)
            for h in list(self._sub_dyn):
                self._sub_dyn[h] = self._sub_dyn[h].to(
                    self.device, non_blocking=True
                )

    # -- Dataset protocol --

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        bid, tidx = self.samples[idx]
        bd = self.gauge_data[bid]
        hucs = bd["subbasin_ids"]

        # Dynamic stack: (k, seq_len, n_dynamic) — unpadded.
        # Per-batch padding is handled by ``subbasin_collate``.
        stack = torch.stack(
            [self._sub_dyn[h][tidx - self.seq_len + 1 : tidx + 1] for h in hucs],
            dim=0,
        )

        return (
            stack,                         # x_dyn  (k, seq_len, n_dyn)
            bd["static_stack"],            # x_stat (k, n_static)
            bd["weights"],                 # w      (k,)
            bd["flow_tensor"][tidx],       # y      scalar (mm/day)
            bd["y_comp_tensor"][tidx],     # y_comp (2,)  mm/day
            bid,                           # gauge id
            bd["loss_w_tensor"],           # scalar
            bd["extreme_qs_tensor"],       # (2,)  mm/day
        )


# ---------------------------------------------------------------------------
# Per-batch padding collate
# ---------------------------------------------------------------------------


def subbasin_collate(batch):
    """Pad each batch to the max real subbasin count *within that batch*.

    The dataset returns variable-length per-gauge stacks ``(k, ...)``;
    this collate function stacks them into ``(B, N_batch, ...)`` tensors
    where ``N_batch = max(k_i)`` over the batch.  Padded rows have
    ``mask = 0`` and ``weights = 0`` so they contribute nothing to the
    aggregation or loss.  This avoids the large global padding to
    ``n_max_subbasins`` (which can be 100+ while the median gauge has
    only ~5–15 subbasins).
    """
    B = len(batch)
    ks = [item[0].shape[0] for item in batch]
    N = max(ks)
    seq_len, n_dyn = batch[0][0].shape[1], batch[0][0].shape[2]
    n_static = batch[0][1].shape[1]

    ref_dyn = batch[0][0]
    ref_static = batch[0][1]

    x_dyn = ref_dyn.new_zeros((B, N, seq_len, n_dyn))
    x_stat = ref_static.new_zeros((B, N, n_static))
    mask = ref_static.new_zeros((B, N))
    weights = ref_static.new_zeros((B, N))
    y = ref_static.new_zeros((B,))
    y_comp = ref_static.new_zeros((B, batch[0][4].shape[0]))
    bids = torch.empty((B,), dtype=torch.long)
    lw = ref_static.new_zeros((B,))
    qs = ref_static.new_zeros((B, batch[0][7].shape[0]))

    for i, (xd, xs, w, yv, yc, bid, lwv, qsv) in enumerate(batch):
        k = ks[i]
        x_dyn[i, :k] = xd
        x_stat[i, :k] = xs
        mask[i, :k] = 1.0
        weights[i, :k] = w
        y[i] = yv
        y_comp[i] = yc
        bids[i] = int(bid)
        lw[i] = lwv
        qs[i] = qsv

    return x_dyn, x_stat, mask, weights, y, y_comp, bids, lw, qs
