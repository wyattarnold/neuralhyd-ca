"""Create area-weighted average climate time series for each PourPtID.

Reads VICGrids intersect table and gridded meteo files to produce
area-weighted daily precipitation, tmax, and tmin for each watershed.
Output goes to data/training/climate/<target>/ (default ``target="watersheds"``)
or data/eval/climate/<target>/ when ``scope="eval"``.

The meteo directory must be supplied as an argument (it is a large external
dataset not stored in the repo).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.data.io import (
    CLIMATE_VARS,
    read_climate_zarr,
    write_climate_zarr,
)
from src.paths import (
    get_eval_target_paths,
    get_target_paths,
)


def read_meteo_file(lat: float, lon: float, meteo_dir: str) -> pd.DataFrame | None:
    """Read a gridded meteo file for a given lat/lon cell."""
    filepath = os.path.join(meteo_dir, f"data_{lat}_{lon}")

    if not os.path.exists(filepath):
        print(f"Warning: File not found: {filepath}")
        return None

    try:
        df = pd.read_csv(
            filepath,
            sep=r'\s+',
            header=None,
            names=['year', 'month', 'day', 'precip_mm', 'tmax_c', 'tmin_c'],
        )
        df['date'] = pd.to_datetime(df[['year', 'month', 'day']])
        return df
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None


def calculate_area_weighted_average(
    pourptid: str | int,
    grid_data: pd.DataFrame,
    meteo_dir: str,
) -> pd.DataFrame | None:
    """Calculate area-weighted average climate for a PourPtID."""
    all_data = []
    weights = []

    print(f"\nProcessing PourPtID {pourptid} with {len(grid_data)} grid cells...")

    for _, row in grid_data.iterrows():
        lat = row['Field1']
        lon = row['Field2']
        shape_area = row['Shape_Area']

        meteo_df = read_meteo_file(lat, lon, meteo_dir)
        if meteo_df is not None:
            all_data.append(meteo_df)
            weights.append(shape_area)
        else:
            print(f"  Skipping grid cell ({lat}, {lon}) - data not found")

    if not all_data:
        print(f"  No data found for PourPtID {pourptid}")
        return None

    min_date = max(df['date'].min() for df in all_data)
    max_date = min(df['date'].max() for df in all_data)
    print(f"  Date range: {min_date.date()} to {max_date.date()}")

    filtered_data = []
    for df in all_data:
        df_filtered = df[(df['date'] >= min_date) & (df['date'] <= max_date)].copy()
        df_filtered = df_filtered.sort_values('date').reset_index(drop=True)
        filtered_data.append(df_filtered)

    weights_arr = np.array(weights)
    normalized_weights = weights_arr / weights_arr.sum()

    result_df = filtered_data[0][['date', 'year', 'month', 'day']].copy()

    precip_weighted = np.zeros(len(result_df))
    tmax_weighted = np.zeros(len(result_df))
    tmin_weighted = np.zeros(len(result_df))

    for df, weight in zip(filtered_data, normalized_weights):
        precip_weighted += df['precip_mm'].values * weight
        tmax_weighted += df['tmax_c'].values * weight
        tmin_weighted += df['tmin_c'].values * weight

    result_df['precip_mm'] = precip_weighted
    result_df['tmax_c'] = tmax_weighted
    result_df['tmin_c'] = tmin_weighted

    print(f"  Successfully processed {len(filtered_data)} grid cells")
    return result_df


def main(
    meteo_dir: str | None = None,
    target: str = "watersheds",
    scope_ids: list | set | None = None,
    force: bool = False,
    scope: str = "training",
) -> pd.DataFrame | None:
    """Run the climate development pipeline.

    Parameters
    ----------
    meteo_dir : str or None
        Path to the directory containing gridded meteo files.
        Required — pass as CLI argument or directly.
    target : str
        Polygon target (``"watersheds"``, ``"huc8"``, ``"huc10"``, ``"huc12"``).
    scope_ids : list or set, optional
        When provided, restrict processing to this subset of PourPtIDs.
        Used by subbasin-mode data prep to limit climate generation to
        only the sub-basins touched by kept gauges (see
        ``<LEVEL>_In_Scope.csv`` written by ``subbasin_gauge_intersect``).
    force : bool
        If False (default), skip any PourPtID already present in the
        existing climate zarr cube.  Set True to regenerate from scratch.
    scope : {"training", "eval"}
        Selects the output root.  ``"training"`` (default) writes to
        ``data/training/climate/<target>.zarr``; ``"eval"`` writes the
        full-domain outputs to ``data/eval/climate/<target>.zarr``.  For
        ``target="watersheds"`` only ``"training"`` is valid.
    """
    if meteo_dir is None:
        if len(sys.argv) < 2:
            print("Usage: python -m src.data.develop_climate <METEO_DIR>")
            print("  METEO_DIR: path to gridded meteo files (data_{lat}_{lon})")
            sys.exit(1)
        meteo_dir = sys.argv[1]

    if scope == "eval":
        tp = get_eval_target_paths(target)
    else:
        tp = get_target_paths(target)
    vicgrids_file = tp["vicgrids_file"]
    climate_zarr = tp["climate_zarr"]
    # HUC ids are strings, gauge basin ids are integers.
    basin_id_dtype = "int64" if target in ("watersheds", "training_watersheds") else "str"

    print("=" * 80)
    print(f"Area-Weighted Climate Data Generator  [target={target}]")
    print("=" * 80)

    print(f"\nReading VICGrids file: {vicgrids_file}")
    vicgrids_df = pd.read_csv(vicgrids_file)
    print(f"Loaded {len(vicgrids_df)} records")
    print(f"Unique PourPtIDs: {vicgrids_df['PourPtID'].nunique()}")

    if scope_ids is not None:
        scope_set = set(str(s) for s in scope_ids)
        # Compare as string to be robust to int vs string HUC10 ids
        vicgrids_df = vicgrids_df[
            vicgrids_df['PourPtID'].astype(str).isin(scope_set)
        ].copy()
        print(f"Scope filter applied — {vicgrids_df['PourPtID'].nunique()} of "
              f"{len(scope_set)} requested IDs matched the VICGrids table")

    pourptids = list(vicgrids_df['PourPtID'].unique())

    # ``basin_data[bid] = DataFrame[date → climate vars]`` accumulates results
    # in memory; we write the full zarr cube once at the end.
    basin_data: dict[object, pd.DataFrame] = {}

    # Resume support: re-load any basins already in the zarr (unless --force).
    if not force and climate_zarr.exists():
        existing_basins, dates, arrays, _ = read_climate_zarr(climate_zarr)
        idx = pd.DatetimeIndex(dates)
        for i, bid in enumerate(existing_basins):
            df = pd.DataFrame({v: arrays[v][i] for v in CLIMATE_VARS}, index=idx)
            df.index.name = "date"
            basin_data[str(bid)] = df.dropna(how="all")
        existing_str = set(basin_data.keys())
        pourptids_todo = [p for p in pourptids if str(p) not in existing_str]
        n_skip = len(pourptids) - len(pourptids_todo)
        if n_skip:
            print(f"Resuming — {n_skip} PourPtIDs already in {climate_zarr.name} "
                  f"(pass --force to regenerate)")
        pourptids = pourptids_todo

    print(f"\nProcessing {len(pourptids)} PourPtIDs...")

    summary = []
    for pourptid in tqdm(pourptids, desc="PourPtIDs"):
        grid_data = vicgrids_df[vicgrids_df['PourPtID'] == pourptid]
        result_df = calculate_area_weighted_average(pourptid, grid_data, meteo_dir)

        if result_df is None:
            continue

        df_indexed = (
            result_df.set_index("date")[list(CLIMATE_VARS)].astype("float32")
        )
        basin_data[str(pourptid)] = df_indexed

        summary.append({
            'PourPtID': pourptid,
            'n_grids': len(grid_data),
            'total_area': grid_data['Shape_Area'].sum(),
            'start_date': df_indexed.index.min(),
            'end_date': df_indexed.index.max(),
            'n_records': len(df_indexed),
        })

    if not basin_data:
        print("\nNo basins to write — nothing to do.")
        return pd.DataFrame(summary)

    print(f"\nWriting zarr cube: {climate_zarr}  ({len(basin_data)} basins)")
    write_climate_zarr(
        climate_zarr,
        basin_data,
        basin_id_dtype=basin_id_dtype,
        overwrite=True,
    )
    summary_df = pd.DataFrame(summary)

    print("\n" + "=" * 80)
    print("Processing Complete!")
    print("=" * 80)
    print(f"\nSuccessfully processed {len(summary_df)} new PourPtIDs")
    print(f"Total basins in store: {len(basin_data)}")
    print(f"Zarr store: {climate_zarr}")

    if not summary_df.empty:
        print(f"\nSummary Statistics:")
        print(f"  Average grid cells per PourPtID: {summary_df['n_grids'].mean():.1f}")
        print(f"  Average records per PourPtID: {summary_df['n_records'].mean():.0f}")
        print(f"  Date range: {summary_df['start_date'].min()} to {summary_df['end_date'].max()}")

    return summary_df


if __name__ == "__main__":
    main()
