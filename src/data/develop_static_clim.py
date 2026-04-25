"""Compute climate statistics for each PourPtID from area-weighted climate files.

Calculates PET (Hargreaves), aridity index, snow fraction, and precipitation
event statistics.  Output goes to data/training/static/<target>/Climate_Statistics_<TARGET>.csv
(or data/eval/static/<target>/... when scope="eval").
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

from src.data.io import load_climate_dataframes
from src.paths import (
    get_eval_target_paths,
    get_target_paths,
)


def calculate_pet_hargreaves(
    tmax: pd.Series | np.ndarray,
    tmin: pd.Series | np.ndarray,
    date: pd.Series,
    latitude: float = 40.0,
) -> np.ndarray:
    """Calculate PET using Hargreaves equation (mm/day)."""
    tmean = (tmax + tmin) / 2

    if isinstance(date, pd.Series):
        doy = date.dt.dayofyear
    else:
        doy = date.dayofyear

    lat_rad = np.radians(latitude)
    solar_dec = 0.409 * np.sin(2 * np.pi * doy / 365 - 1.39)
    sunset_angle = np.arccos(-np.tan(lat_rad) * np.tan(solar_dec))
    dr = 1 + 0.033 * np.cos(2 * np.pi * doy / 365)
    Ra = (24 * 60 / np.pi) * 0.082 * dr * (
        sunset_angle * np.sin(lat_rad) * np.sin(solar_dec)
        + np.cos(lat_rad) * np.cos(solar_dec) * np.sin(sunset_angle)
    )

    pet = 0.0023 * Ra * (tmean + 17.8) * np.sqrt(np.maximum(tmax - tmin, 0))
    return pet


def calculate_event_statistics(
    data: pd.Series, threshold: float, above: bool = True
) -> tuple[float, float]:
    """Calculate frequency and average duration of events above/below threshold."""
    events = data >= threshold if above else data < threshold
    frequency = events.sum() / len(events)

    event_changes = np.diff(events.astype(int))
    event_starts = np.where(event_changes == 1)[0] + 1
    event_ends = np.where(event_changes == -1)[0] + 1

    if events.iloc[0]:
        event_starts = np.concatenate([[0], event_starts])
    if events.iloc[-1]:
        event_ends = np.concatenate([event_ends, [len(events)]])

    if len(event_starts) > 0:
        durations = event_ends - event_starts
        average_duration = durations.mean()
    else:
        average_duration = 0.0

    return frequency, average_duration


def process_climate_dataframe(df: pd.DataFrame, pourpoint_id: str) -> dict:
    """Compute climate statistics from a daily climate DataFrame.

    ``df`` is indexed by date with columns ``precip_mm``, ``tmax_c``, ``tmin_c``.
    """
    df = df.copy().dropna(how="all")
    df = df.reset_index().rename(columns={"index": "date"})
    if "date" not in df.columns:
        df.rename(columns={df.columns[0]: "date"}, inplace=True)
    df["date"] = pd.to_datetime(df["date"])

    df['pet_mm'] = calculate_pet_hargreaves(df['tmax_c'], df['tmin_c'], df['date'])
    df['tmean_c'] = (df['tmax_c'] + df['tmin_c']) / 2

    results: dict = {'PourPtID': pourpoint_id}

    results['precip_mean'] = df['precip_mm'].mean()
    results['pet_mean'] = df['pet_mm'].mean()

    if results['precip_mean'] > 0:
        results['aridity_index'] = results['pet_mean'] / results['precip_mean']
    else:
        results['aridity_index'] = np.nan

    snow_days = df['tmean_c'] < 0
    total_precip = df['precip_mm'].sum()
    if total_precip > 0:
        results['snow_fraction'] = df.loc[snow_days, 'precip_mm'].sum() / total_precip
    else:
        results['snow_fraction'] = 0.0

    high_precip_threshold = 5 * results['precip_mean']
    high_freq, high_dur = calculate_event_statistics(df['precip_mm'], high_precip_threshold, above=True)
    results['high_precip_freq'] = high_freq
    results['high_precip_dur'] = high_dur

    low_freq, low_dur = calculate_event_statistics(df['precip_mm'], 1.0, above=False)
    results['low_precip_freq'] = low_freq
    results['low_precip_dur'] = low_dur

    return results


def main(target: str = "watersheds", scope: str = "training") -> None:
    tp = get_eval_target_paths(target) if scope == "eval" else get_target_paths(target)
    climate_zarr = tp["climate_zarr"]
    climate_stats_output = tp["climate_stats_output"]

    print("=" * 70)
    print(f"Climate Statistics Calculator  [target={target}]")
    print("=" * 70)

    if not climate_zarr.exists():
        print(f"ERROR: Climate zarr not found: {climate_zarr}")
        return

    print(f"\nReading climate cube: {climate_zarr}")
    climate_dfs = load_climate_dataframes(climate_zarr)
    print(f"Found {len(climate_dfs)} basins to process")

    if len(climate_dfs) == 0:
        print("ERROR: No basins in climate cube")
        return

    results = []
    print("\nProcessing basins...")

    items = list(climate_dfs.items())
    for i, (pourpoint_id, df) in enumerate(items, 1):
        pid = str(pourpoint_id)
        if i % 50 == 0 or i == len(items):
            print(f"  Processing {i}/{len(items)}: {pid}")

        try:
            result = process_climate_dataframe(df, pid)
            results.append(result)
        except Exception as e:
            print(f"  ERROR processing {pid}: {e}")
            continue

    results_df = pd.DataFrame(results)

    for col in results_df.columns:
        if col != 'PourPtID' and pd.api.types.is_numeric_dtype(results_df[col]):
            results_df[col] = results_df[col].round(3)

    climate_stats_output.parent.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving results to: {climate_stats_output.name}")
    results_df.to_csv(climate_stats_output, index=False)

    print(f"\nResults summary:")
    print(f"  - Successfully processed: {len(results_df)} pourpoints")
    print(f"  - Output columns: {len(results_df.columns)}")

    print(f"\nFirst few rows of results:")
    print(results_df.head())
    print(f"\nClimate statistics summary:")
    print(results_df.describe())

    print("\n" + "=" * 70)
    print("Processing complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
