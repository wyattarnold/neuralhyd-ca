"""Data loading for the dPL+SAC-SMA model.

The model operates on **HUC12 lumped units** that aggregate to USGS gauges.
For each training basin we need:

* a per-HUC12 forcing CSV (precip_mm, tmax_c, tmin_c) at
  ``huc12_climate_dir/climate_<huc12>.csv``
* per-HUC12 static attributes from the BasinATLAS lev12 × WBDHU12 intersect
  (``huc12_static_csv``) — joined on the 12-digit HUC12 code
* per-HUC12 metadata (area, area-weight relative to gauge, elevation, lat)
* per-basin observed daily flow at the gauge (target)

Every HUC12 referenced by the manifest MUST have a forcing CSV and a
static row; missing inputs raise ``FileNotFoundError`` / ``KeyError``.

Public API
----------
:func:`load_dpl_data`  → dict of basin_id → :class:`BasinBundle`
:func:`create_folds`   → stratified spatial folds (basin level)
:func:`compute_norm_stats` → train-set z-score statistics
:class:`DplDataset`    → ``torch.utils.data.Dataset`` yielding
                         ``(unit_inputs, basin_targets, aux)`` tuples ready
                         for the model.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .config import DplConfig
from src.lstm.dataset import _lyne_hollick_baseflow

# cfs → mm/day per km²: 0.0283168 m³/cfs * 86400 s/day * 1000 mm/m / 1e6 m²/km²
_CFS_TO_MM_PER_KM2 = 0.0283168 * 86400 * 1000 / 1e6  # ≈ 2.44577


# ---------------------------------------------------------------------------
# Manifest & static attributes
# ---------------------------------------------------------------------------


def load_manifest(manifest_csv: Path) -> pd.DataFrame:
    """Load the basin↔HUC12 manifest.

    Expected columns (others ignored):
        ``PourPtID, huc12, gauge_area_km2, huc12_area_km2, overlap_area_km2,
        frac_of_gauge, frac_of_huc12``

    ``frac_of_gauge`` is the area weight used for aggregation.
    """
    df = pd.read_csv(manifest_csv)
    df["huc12"] = df["huc12"].astype(str).str.zfill(12)
    df["PourPtID"] = df["PourPtID"].astype(int)
    return df


def load_huc12_statics(
    physical_csv: Path,
    climate_stats_csv: Path,
    features: list[str],
) -> pd.DataFrame:
    """Load per-HUC12 static attributes from the two HUC12 static files.

    Mirrors :func:`src.lstm.subbasin_dataset._load_subbasin_statics`:
    Physical_Attributes_HUC12.csv (BasinATLAS-derived) and
    Climate_Statistics_HUC12.csv (derived climate statistics) are joined
    inner on ``PourPtID`` (the column holding the 12-digit HUC12 ID — its
    name is a legacy from the target-aware preparation pipeline; here it
    is the HUC12 code, not a gauge id).

    The ``ele_mt_uav`` (mean elevation, m) column is always retained so
    downstream code can use it for Snow-17.

    Returns
    -------
    DataFrame indexed by ``huc12`` (12-digit zero-padded string) containing
    the requested ``features`` plus ``ele_mt_uav``.
    """
    phys = pd.read_csv(physical_csv, dtype={"PourPtID": str}).set_index("PourPtID")
    clim = pd.read_csv(climate_stats_csv, dtype={"PourPtID": str}).set_index("PourPtID")
    df = phys.join(clim, how="inner")

    # Always keep elevation so the physics core has per-HUC12 elev.
    need = set(features) | {"ele_mt_uav"}
    missing = [c for c in features if c not in df.columns]
    if missing:
        raise KeyError(
            f"static files {physical_csv.name}+{climate_stats_csv.name} "
            f"are missing requested features: {missing}"
        )
    if "ele_mt_uav" not in df.columns:
        raise KeyError(
            f"{physical_csv.name} must contain 'ele_mt_uav' (per-HUC12 mean elevation)"
        )

    keep = [c for c in need if c in df.columns]
    df = df[keep].astype(float)
    df.index = df.index.astype(str).str.zfill(12)
    df.index.name = "huc12"
    return df


# ---------------------------------------------------------------------------
# Forcing & flow loading
# ---------------------------------------------------------------------------


# Note: flow + climate are loaded from zarr cubes via
# :func:`src.data.io.load_flow_dataframes` / :func:`load_climate_dataframes`
# in :func:`load_dpl_data`.


# ---------------------------------------------------------------------------
# BasinBundle: per-basin tensors for the forward pass
# ---------------------------------------------------------------------------


@dataclass
class BasinBundle:
    basin_id: int
    tier: int
    huc12s: list[str]
    dates: pd.DatetimeIndex                 # length T
    flow_mm: np.ndarray                      # (T,) observed flow as mm/day
    baseflow_mm: np.ndarray                  # (T,) Lyne–Hollick baseflow (mm/day)
    forcing: dict[str, np.ndarray]           # each (n_units, T)
    static: np.ndarray                       # (n_units, n_static_features)
    elev_m: np.ndarray                       # (n_units,)
    lat_rad: np.ndarray                      # (n_units,)
    area_weight: np.ndarray                  # (n_units,) sum ≈ 1
    basin_area_km2: float
    precip_mean_mm: float                    # for target normalisation
    q_extreme_start: float = 0.0             # ramp start (mm/day, target units)
    q_extreme_ramp: float = 1.0              # q_top - q_start (>=1e-4)

    @property
    def n_units(self) -> int:
        return len(self.huc12s)

    @property
    def n_days(self) -> int:
        return len(self.dates)


# ---------------------------------------------------------------------------
# Top-level loader
# ---------------------------------------------------------------------------


def load_dpl_data(config: DplConfig) -> Dict[int, BasinBundle]:
    """Discover all basins and return a dict ``basin_id -> BasinBundle``.

    All HUC12s referenced by the manifest are assumed to have a per-unit
    climate CSV at ``config.huc12_climate_dir/<huc12>.csv`` and a row in
    the per-HUC12 static attribute table.  Missing inputs raise a clear
    error instead of silently falling back to parent-basin values.

    The parent USGS-basin climate file is still read to compute the
    basin-level ``precip_mean_mm`` used as the target-normalisation
    scale (flow / basin_precip_mean → dimensionless runoff ratio).
    This is the one basin-level statistic that must remain area-
    representative; using a HUC12's own precip would couple the
    per-unit scaling to gA's predictions.
    """
    manifest = load_manifest(config.huc12_manifest_csv)
    statics = load_huc12_statics(
        config.huc12_physical_csv,
        config.huc12_climate_stats_csv,
        config.static_features,
    )

    # Log-transform configured columns on the HUC12 static table
    for feat in config.log_transform_static:
        if feat in statics.columns:
            statics[feat] = np.log10(statics[feat].clip(lower=1e-6))
    statics = statics.fillna(statics.median(numeric_only=True))

    # Parent basin latitude — used for Hamon daylength per HUC12 (broadcast).
    # Daylength varies <1 min across a 100 km-wide basin, so basin-centroid
    # latitude is accurate enough for daily PET.  If per-HUC12 centroid lat
    # becomes available, wire it in here.
    # Basin-centroid latitude for Hamon daylength — read from the USGS
    # site table (STATION_NO = basin_id, column LATITUDE in decimal degrees).
    usgs_table_path = config.data_dir / "raw" / "USGS_Table_1.csv"
    if not usgs_table_path.exists():
        raise FileNotFoundError(f"USGS site table missing: {usgs_table_path}")
    usgs_meta = pd.read_csv(usgs_table_path, index_col="STATION_NO")
    lat_col = "LATITUDE" if "LATITUDE" in usgs_meta.columns else "latitude"
    if lat_col not in usgs_meta.columns:
        raise KeyError(f"no latitude column in {usgs_table_path}")

    parent_climate_zarr = config.data_dir / "training" / "climate" / "watersheds.zarr"

    # Pre-load all flow + climate from zarr (cheap: a few hundred MB at most)
    from src.data.io import load_flow_dataframes, load_climate_dataframes

    flow_all, tier_all = load_flow_dataframes(config.flow_zarr)
    pour_ids = manifest["PourPtID"].unique()
    pour_ids_int = [int(b) for b in pour_ids]

    parent_climate_all = load_climate_dataframes(
        parent_climate_zarr, basin_ids=pour_ids_int
    )

    huc_needed = sorted({str(h).zfill(12) for h in manifest["huc12"].astype(str)})
    huc_climate_all = load_climate_dataframes(
        config.huc12_climate_zarr, basin_ids=huc_needed
    )

    bundles: Dict[int, BasinBundle] = {}

    for basin_id in pour_ids:
        basin_id_int = int(basin_id)
        if basin_id_int not in flow_all:
            continue
        flow_df = flow_all[basin_id_int]
        tier = tier_all[basin_id_int]

        rows = manifest[manifest["PourPtID"] == basin_id_int]
        if rows.empty:
            continue

        huc12_ids = rows["huc12"].astype(str).tolist()
        # Area weights — `frac_of_gauge` gives HUC12 contribution to the gauge.
        weights = rows["frac_of_gauge"].astype(float).values
        weights = weights / max(weights.sum(), 1e-9)
        basin_area = float(rows["gauge_area_km2"].iloc[0])

        # Parent climate — used only to compute basin-level precip_mean scale.
        if basin_id_int not in parent_climate_all:
            continue
        parent_clim = parent_climate_all[basin_id_int]

        # Per-HUC12 climate — required (no fallback).
        per_unit_clim: list[pd.DataFrame] = []
        for huc in huc12_ids:
            if huc not in huc_climate_all:
                raise FileNotFoundError(
                    f"HUC12 climate missing for {huc} in {config.huc12_climate_zarr} "
                    f"(basin {basin_id_int}).  Every HUC12 in the manifest must "
                    f"have a forcing entry."
                )
            per_unit_clim.append(huc_climate_all[huc])

        # Common date index = intersection of flow + every unit forcing
        common = per_unit_clim[0].index
        for u in per_unit_clim[1:]:
            common = common.intersection(u.index)
        common = common.intersection(flow_df.index)
        common = common.intersection(parent_clim.index)
        if len(common) < config.warmup_steps + 30:
            continue

        # Stack forcings: (n_units, T) per dynamic feature
        forcing_dict: dict[str, np.ndarray] = {}
        for feat in config.dynamic_features + ["tmax_c", "tmin_c", "precip_mm"]:
            if feat in forcing_dict:
                continue
            arr = np.stack([
                u.reindex(common)[feat].fillna(0.0).to_numpy(dtype=np.float32)
                for u in per_unit_clim
            ], axis=0)
            forcing_dict[feat] = arr
        # Always provide tavg, even if not in dynamic_features
        forcing_dict["tavg_c"] = 0.5 * (forcing_dict["tmax_c"] + forcing_dict["tmin_c"])

        # Per-HUC12 statics: strict lookup, no broadcast.
        missing = [h for h in huc12_ids if h not in statics.index]
        if missing:
            raise KeyError(
                f"HUC12 static rows missing for basin {basin_id_int}: "
                f"{missing[:5]}{'...' if len(missing) > 5 else ''}"
            )
        keep_feats = list(config.static_features)
        static_rows = statics.loc[huc12_ids, keep_feats]
        statics_arr = static_rows.astype(float).fillna(0.0).to_numpy(dtype=np.float32)

        # Per-HUC12 elevation from BasinATLAS
        elev_arr = statics.loc[huc12_ids, "ele_mt_uav"].astype(float).to_numpy(dtype=np.float32)

        # Latitude — basin-centroid broadcast to all HUC12s (daylength varies
        # negligibly at sub-basin scale).
        if basin_id_int not in usgs_meta.index:
            continue
        lat_deg = float(usgs_meta.loc[basin_id_int, lat_col])
        lat_arr = np.full(len(huc12_ids), np.deg2rad(lat_deg), dtype=np.float32)

        # Flow target — convert cfs → mm/day at gauge, then divide by precip_mean for stable scale
        flow_cfs = flow_df.reindex(common)["flow"].to_numpy(dtype=np.float32)
        flow_mm = flow_cfs * (_CFS_TO_MM_PER_KM2 / max(basin_area, 1e-6))
        precip_mean = float(parent_clim.reindex(common)["precip_mm"].mean())
        precip_mean = max(precip_mean, 1e-3)

        # Lyne–Hollick baseflow target (NaNs in flow propagate to NaN here;
        # downstream loss masks them out).
        baseflow_mm = _lyne_hollick_baseflow(
            flow_mm.astype(np.float64), alpha=config.baseflow_alpha
        ).astype(np.float32)

        # Per-basin quantiles for the peak-flow ramp on the primary loss.
        # Computed once on observed mm/day (ignoring NaNs); identical units
        # as the model's ``q_gauge`` output and the loss target.
        flow_finite = flow_mm[np.isfinite(flow_mm)]
        if flow_finite.size > 0:
            q_start = float(np.quantile(flow_finite, config.extreme_start_quantile))
            q_top = float(np.quantile(flow_finite, config.extreme_top_quantile))
        else:
            q_start, q_top = 0.0, 1.0
        q_ramp = max(q_top - q_start, 1e-4)

        bundles[basin_id_int] = BasinBundle(
            basin_id=basin_id_int,
            tier=tier,
            huc12s=huc12_ids,
            dates=common,
            flow_mm=flow_mm,
            baseflow_mm=baseflow_mm,
            forcing=forcing_dict,
            static=statics_arr,
            elev_m=elev_arr,
            lat_rad=lat_arr,
            area_weight=weights.astype(np.float32),
            basin_area_km2=basin_area,
            precip_mean_mm=precip_mean,
            q_extreme_start=q_start,
            q_extreme_ramp=q_ramp,
        )

    return bundles


# ---------------------------------------------------------------------------
# Folds, normalisation, dataset
# ---------------------------------------------------------------------------


def _nesting_groups(gauge_huc12s: dict[int, list[str]]) -> list[list[int]]:
    """Group gauges that share any HUC12 via union-find.

    Two gauges are placed in the same group whenever their HUC12 sets
    intersect — whether by strict nesting (one catchment fully contains
    the other) or partial overlap (sharing one or more HUC12s without
    containment).  Either case constitutes input leakage: a HUC12 forcing
    tensor that drives one gauge in training would also drive the other
    in validation.  Transitivity is handled by union-find.

    Mirrors :func:`src.lstm.subbasin_dataset._nesting_groups` so the dPL
    and subbasin-LSTM pipelines partition the universe of gauges
    consistently.
    """
    gauges = list(gauge_huc12s.keys())
    sub_sets = {g: set(hs) for g, hs in gauge_huc12s.items()}

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
            if sub_sets[a] & sub_sets[b]:   # any shared HUC12 = leakage
                union(i, j)

    from collections import defaultdict
    grp: dict[int, list[int]] = defaultdict(list)
    for idx, g in enumerate(gauges):
        grp[find(idx)].append(g)
    return list(grp.values())


def create_folds(bundles: dict[int, BasinBundle], config: DplConfig) -> list[tuple[list[int], list[int]]]:
    """Stratified k-fold over gauge groups, respecting shared catchments.

    Gauges whose HUC12 sets share **any** HUC12 \u2014 whether by strict
    containment (nesting) or partial overlap \u2014 are grouped together and
    always assigned to the same fold.  Without this, a HUC12's forcings
    would appear in training while the same forcings drive a different
    gauge in the val set \u2014 direct input leakage.

    Within each tier, groups are stratified using LPT (longest-processing-
    time) deal: groups are sorted by total observed flow-days descending,
    then each is greedily assigned to the fold with the fewest accumulated
    days (ties broken by gauge count, then random).  This balances both
    gauge count and total flow-days across folds while respecting tier
    stratification and the no-nested-split constraint.

    Mirrors :func:`src.lstm.subbasin_dataset.create_subbasin_folds` so dPL
    and subbasin-LSTM produce the same fold partition for a given seed.
    """
    rng = np.random.RandomState(config.seed)
    n_folds = config.n_folds

    gauge_ids = list(bundles.keys())
    tier_map = {bid: b.tier for bid, b in bundles.items()}
    flow_days: dict[int, int] = {
        bid: int(np.isfinite(b.flow_mm).sum()) for bid, b in bundles.items()
    }
    total_flow_days = sum(flow_days.values())

    # Build nesting groups from each gauge's HUC12 list
    gauge_huc12s = {bid: list(b.huc12s) for bid, b in bundles.items()}
    groups = _nesting_groups(gauge_huc12s)

    nested = [grp for grp in groups if len(grp) > 1]
    if nested:
        print(f"\n  Nested gauge groups: {len(nested)} groups, "
              f"{sum(len(g) for g in nested)} gauges kept together")
        for grp in sorted(nested, key=len, reverse=True):
            print(f"    {sorted(grp)}")
    else:
        print("\n  No nested gauge groups detected — all gauges are independent.")

    def group_tier(grp: list[int]) -> int:
        """Tier of the member with most observed flow-days."""
        return tier_map[max(grp, key=lambda g: flow_days.get(g, 0))]

    def group_days(grp: list[int]) -> int:
        return sum(flow_days.get(g, 0) for g in grp)

    # Stratified LPT deal
    fold_val_groups: list[list[list[int]]] = [[] for _ in range(n_folds)]
    fold_days = [0] * n_folds
    fold_gauges = [0] * n_folds

    for t in (1, 2, 3):
        tier_groups = [grp for grp in groups if group_tier(grp) == t]
        rng.shuffle(tier_groups)                            # tie-break jitter
        tier_groups.sort(key=group_days, reverse=True)      # stable LPT order
        for grp in tier_groups:
            jitter = rng.rand(n_folds)
            target = min(
                range(n_folds),
                key=lambda k: (fold_days[k], fold_gauges[k], jitter[k]),
            )
            fold_val_groups[target].append(grp)
            fold_days[target] += group_days(grp)
            fold_gauges[target] += len(grp)

    folds: list[tuple[list[int], list[int]]] = []
    print(f"\n  Fold partition  ({n_folds} folds, {len(gauge_ids)} gauges in "
          f"{len(groups)} groups, {total_flow_days:,} total flow-days)")
    for k in range(n_folds):
        val_ids = [g for grp in fold_val_groups[k] for g in grp]
        val_set = set(val_ids)
        train_ids = [g for g in gauge_ids if g not in val_set]
        val_days = sum(flow_days[g] for g in val_ids)
        train_days = sum(flow_days[g] for g in train_ids)
        tier_counts = {t: sum(1 for g in val_ids if tier_map[g] == t)
                       for t in (1, 2, 3)}
        print(f"    Fold {k + 1}: val {len(val_ids):>3d} gauges "
              f"(T1={tier_counts[1]}, T2={tier_counts[2]}, T3={tier_counts[3]})  "
              f"| val {val_days:>8,} days ({val_days / total_flow_days:.1%})  "
              f"| train {train_days:>8,} days ({train_days / total_flow_days:.1%})")
        folds.append((train_ids, val_ids))

    return folds


def compute_norm_stats(
    train_ids: list[int], bundles: dict[int, BasinBundle], config: DplConfig
) -> dict:
    """Compute z-score stats from training basins only.

    Climate stats are pooled across **HUC12 units** in the training set,
    not basins, since per-HUC12 inputs feed gA.  Static stats are computed
    in the same way.
    """
    # Climate
    clim_lists: dict[str, list[np.ndarray]] = {f: [] for f in config.dynamic_features}
    for bid in train_ids:
        b = bundles[bid]
        for f in config.dynamic_features:
            clim_lists[f].append(b.forcing[f].reshape(-1))
    clim_stats = {}
    for f, parts in clim_lists.items():
        arr = np.concatenate(parts)
        clim_stats[f] = {"mean": float(arr.mean()), "std": float(max(arr.std(), 1e-3))}

    # Statics
    static_stack = np.concatenate([bundles[b].static for b in train_ids], axis=0)
    static_mean = static_stack.mean(axis=0)
    static_std = static_stack.std(axis=0).clip(min=1e-3)

    # Flow scale
    precip_means = np.array([bundles[b].precip_mean_mm for b in train_ids])

    return {
        "climate": clim_stats,
        "static_mean": static_mean.astype(np.float32),
        "static_std": static_std.astype(np.float32),
        "precip_mean_mean": float(precip_means.mean()),
    }


# ---------------------------------------------------------------------------
# Dataset (basin-level samples)
# ---------------------------------------------------------------------------


class DplDataset(Dataset):
    """Yields fixed-length training windows aligned across all HUC12s of a basin.

    Each item is a dict ready for the model forward pass:
        ``x_dyn``      (n_units, T_full, n_dyn_norm)
        ``x_static``   (n_units, n_static_norm)
        ``forcing``    raw per-unit forcing dict (mm/day, °C, doy)
        ``aux``        elev / lat / area_weight (per unit) and basin-level fields
        ``y``          (T_loss,) target flow at gauge (raw mm/day; same units as model output)
        ``y_raw``      (T_loss,) raw flow mm/day (kept for backward-compatible eval code paths)

    ``T_full`` = ``warmup_steps + seq_len``; the loss is computed over the
    last ``seq_len`` days while the warmup buffer spins up SAC-SMA / Snow-17
    state.
    """

    def __init__(
        self,
        bundles: dict[int, BasinBundle],
        basin_ids: list[int],
        norm_stats: dict,
        config: DplConfig,
        is_train: bool = True,
    ):
        self.bundles = bundles
        self.basin_ids = list(basin_ids)
        self.config = config
        self.is_train = is_train
        self.window = config.warmup_steps + config.seq_len

        cs = norm_stats["climate"]
        self._dyn_means = np.array([cs[f]["mean"] for f in config.dynamic_features], dtype=np.float32)
        self._dyn_stds = np.array([cs[f]["std"] for f in config.dynamic_features], dtype=np.float32)
        self._static_mean = np.asarray(norm_stats["static_mean"], dtype=np.float32)
        self._static_std = np.asarray(norm_stats["static_std"], dtype=np.float32)

        # ------------------------------------------------------------------
        # Precompute per-bundle tensors once.  __getitem__ then becomes
        # pure slicing — no numpy allocation, no normalisation, no python
        # doy lookup per sample — which saves noticeable overhead when the
        # training loop pulls thousands of windows per epoch.
        # ``torch.from_numpy`` is zero-copy for contiguous float32 arrays.
        # ------------------------------------------------------------------
        dyn_means_t = torch.from_numpy(self._dyn_means)
        dyn_stds_t = torch.from_numpy(self._dyn_stds)
        static_mean_t = torch.from_numpy(self._static_mean)
        static_std_t = torch.from_numpy(self._static_std)
        feats = list(config.dynamic_features)

        self._cache: dict[int, dict] = {}
        for bid in self.basin_ids:
            b = bundles[bid]
            # Stack dynamic features and z-score once per bundle.
            dyn_np = np.stack([b.forcing[f] for f in feats], axis=-1).astype(np.float32, copy=False)
            dyn_t = (torch.from_numpy(dyn_np) - dyn_means_t) / dyn_stds_t     # (n, T, F)
            static_np = b.static.astype(np.float32, copy=False)
            static_t = (torch.from_numpy(static_np) - static_mean_t) / static_std_t
            doy_np = np.fromiter(
                (d.timetuple().tm_yday for d in b.dates),
                dtype=np.float32, count=len(b.dates),
            )
            self._cache[bid] = {
                "x_dyn": dyn_t,
                "x_static": static_t,
                "prcp": torch.from_numpy(b.forcing["precip_mm"]),          # (n, T)
                "tavg": torch.from_numpy(b.forcing["tavg_c"]),
                "doy_1d": torch.from_numpy(doy_np),                         # (T,)
                "flow": torch.from_numpy(b.flow_mm),                        # (T,)
                "baseflow": torch.from_numpy(b.baseflow_mm),                # (T,)
                "elev_m": torch.from_numpy(b.elev_m),
                "lat_rad": torch.from_numpy(b.lat_rad),
                "area_weight": torch.from_numpy(b.area_weight),
            }

    def __len__(self) -> int:
        return len(self.basin_ids)

    def __getitem__(self, idx: int) -> dict:
        bid = self.basin_ids[idx]
        b = self.bundles[bid]
        c = self._cache[bid]
        T = c["flow"].shape[0]
        if self.is_train:
            t0 = int(np.random.randint(0, max(T - self.window, 1)))
            t1 = t0 + self.window
        else:
            t0, t1 = 0, T

        prcp = c["prcp"][:, t0:t1]                        # (n, T_full)
        tavg = c["tavg"][:, t0:t1]
        # Broadcast doy (T,) → (n, T) without copying — use expand().
        doy = c["doy_1d"][t0:t1].unsqueeze(0).expand(prcp.shape[0], -1)
        x_dyn = c["x_dyn"][:, t0:t1, :]
        flow = c["flow"][t0:t1]
        baseflow = c["baseflow"][t0:t1]
        # Targets stay in raw mm/day — model `q_gauge` is the area-weighted
        # SAC-SMA gauge runoff in mm/day, so the loss is unit-consistent.
        # No precip_mean normalisation applied.
        y_target = flow
        y_base_target = baseflow

        return {
            "basin_id": b.basin_id,
            "tier": b.tier,
            "x_dyn": x_dyn,                                                 # (n, T_full, F)
            "x_static": c["x_static"],                                      # (n, S)
            "prcp": prcp,                                                   # (n, T_full)
            "tavg": tavg,
            "doy": doy,                                                     # (n, T_full)
            "elev_m": c["elev_m"],                                          # (n,)
            "lat_rad": c["lat_rad"],
            "area_weight": c["area_weight"],
            "y": y_target,                                                  # (T_full,) mm/day
            "y_base": y_base_target,                                        # (T_full,) mm/day
            "y_raw": flow,
            "precip_mean": torch.tensor(b.precip_mean_mm, dtype=torch.float32),
            "basin_area_km2": torch.tensor(b.basin_area_km2, dtype=torch.float32),
            "extreme_qs": torch.tensor([b.q_extreme_start, b.q_extreme_ramp],
                                       dtype=torch.float32),  # (2,) [q_start, ramp]
        }


# ---------------------------------------------------------------------------
# Collate: pack a list of basin-samples into model-ready tensors
# ---------------------------------------------------------------------------


def collate_basins(samples: list[dict]) -> dict:
    """Concatenate variable-sized per-basin samples along the units dim.

    Time-window length must match across the batch — guaranteed in
    training mode by sampling identical ``seq_len + warmup_steps``.  In
    eval mode each basin is processed individually so this collate is not
    used.
    """
    n_basins = len(samples)
    basin_index = []
    for b_idx, s in enumerate(samples):
        basin_index.extend([b_idx] * s["x_dyn"].shape[0])

    x_dyn = torch.cat([s["x_dyn"] for s in samples], dim=0)
    x_static = torch.cat([s["x_static"] for s in samples], dim=0)
    prcp = torch.cat([s["prcp"] for s in samples], dim=0)
    tavg = torch.cat([s["tavg"] for s in samples], dim=0)
    doy = torch.cat([s["doy"] for s in samples], dim=0)
    elev = torch.cat([s["elev_m"] for s in samples], dim=0)
    lat = torch.cat([s["lat_rad"] for s in samples], dim=0)
    aw = torch.cat([s["area_weight"] for s in samples], dim=0)

    y = torch.stack([s["y"] for s in samples], dim=0)
    y_base = torch.stack([s["y_base"] for s in samples], dim=0)
    y_raw = torch.stack([s["y_raw"] for s in samples], dim=0)
    precip_mean = torch.stack([s["precip_mean"] for s in samples], dim=0)
    extreme_qs = torch.stack([s["extreme_qs"] for s in samples], dim=0)  # (B, 2)
    tiers = torch.tensor([s["tier"] for s in samples], dtype=torch.long)
    basin_ids = torch.tensor([s["basin_id"] for s in samples], dtype=torch.long)

    return {
        "x_dyn": x_dyn,
        "x_static": x_static,
        "forcing": {"prcp": prcp, "tavg": tavg, "doy": doy},
        "aux": {
            "elev_m": elev,
            "lat_rad": lat,
            "area_weight": aw,
            "basin_index": torch.tensor(basin_index, dtype=torch.long),
            "n_basins": n_basins,
        },
        "y": y,
        "y_base": y_base,
        "y_raw": y_raw,
        "precip_mean": precip_mean,
        "extreme_qs": extreme_qs,
        "tiers": tiers,
        "basin_ids": basin_ids,
    }
