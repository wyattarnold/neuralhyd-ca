"""Create area-weighted average climate time series for each PourPtID.

Reads VICGrids intersect table and gridded meteo files to produce
area-weighted daily precipitation, tmax, and tmin for each watershed.
Output goes to data/training/climate/.

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

from src.paths import VICGRIDS_FILE, CLIMATE_DIR, get_target_paths


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


def main(meteo_dir: str | None = None, target: str = "watersheds") -> pd.DataFrame | None:
    """Run the climate development pipeline.

    Parameters
    ----------
    meteo_dir : str or None
        Path to the directory containing gridded meteo files.
        Required — pass as CLI argument or directly.
    target : str
        Polygon target (``"watersheds"``, ``"huc8"``, ``"huc10"``).
    """
    if meteo_dir is None:
        if len(sys.argv) < 2:
            print("Usage: python -m src.data.develop_climate <METEO_DIR>")
            print("  METEO_DIR: path to gridded meteo files (data_{lat}_{lon})")
            sys.exit(1)
        meteo_dir = sys.argv[1]

    tp = get_target_paths(target)
    vicgrids_file = tp["vicgrids_file"]
    climate_dir = tp["climate_dir"]

    print("=" * 80)
    print(f"Area-Weighted Climate Data Generator  [target={target}]")
    print("=" * 80)

    print(f"\nReading VICGrids file: {vicgrids_file}")
    vicgrids_df = pd.read_csv(vicgrids_file)
    print(f"Loaded {len(vicgrids_df)} records")
    print(f"Unique PourPtIDs: {vicgrids_df['PourPtID'].nunique()}")

    pourptids = vicgrids_df['PourPtID'].unique()
    print(f"\nProcessing {len(pourptids)} PourPtIDs...")

    climate_dir.mkdir(parents=True, exist_ok=True)

    summary = []
    for pourptid in tqdm(pourptids, desc="PourPtIDs"):
        grid_data = vicgrids_df[vicgrids_df['PourPtID'] == pourptid]
        result_df = calculate_area_weighted_average(pourptid, grid_data, meteo_dir)

        if result_df is not None:
            output_file = climate_dir / f"climate_{pourptid}.csv"
            result_df.to_csv(output_file, index=False)

            summary.append({
                'PourPtID': pourptid,
                'n_grids': len(grid_data),
                'total_area': grid_data['Shape_Area'].sum(),
                'start_date': result_df['date'].min(),
                'end_date': result_df['date'].max(),
                'n_records': len(result_df),
                'output_file': str(output_file),
            })

    summary_df = pd.DataFrame(summary)
    summary_file = climate_dir / "processing_summary.csv"
    summary_df.to_csv(summary_file, index=False)

    print("\n" + "=" * 80)
    print("Processing Complete!")
    print("=" * 80)
    print(f"\nSuccessfully processed {len(summary_df)} PourPtIDs")
    print(f"Output directory: {climate_dir}")
    print(f"Summary file: {summary_file}")

    if not summary_df.empty:
        print(f"\nSummary Statistics:")
        print(f"  Average grid cells per PourPtID: {summary_df['n_grids'].mean():.1f}")
        print(f"  Average records per PourPtID: {summary_df['n_records'].mean():.0f}")
        print(f"  Date range: {summary_df['start_date'].min()} to {summary_df['end_date'].max()}")

    return summary_df


if __name__ == "__main__":
    main()
