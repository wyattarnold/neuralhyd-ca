"""Verify area-weighted climate data by calculating monthly averages.

Reads climate files from data/training/climate/ and produces monthly average
tables plus summary statistics.  Output goes to data/prepare/climate_verification/.
"""
from __future__ import annotations

import os

import pandas as pd
from tqdm import tqdm

from src.paths import CLIMATE_DIR, CLIMATE_VERIFICATION_DIR


def calculate_monthly_averages(climate_file: str, pourptid: str | int) -> pd.DataFrame:
    """Calculate monthly averages from daily climate data."""
    df = pd.read_csv(climate_file)
    df['date'] = pd.to_datetime(df['date'])

    monthly_avg = df.groupby('month').agg({
        'precip_mm': 'mean',
        'tmax_c': 'mean',
        'tmin_c': 'mean',
    }).reset_index()

    monthly_avg.columns = ['month', 'precip_mm_avg', 'tmax_c_avg', 'tmin_c_avg']

    month_names = {
        1: 'January', 2: 'February', 3: 'March', 4: 'April',
        5: 'May', 6: 'June', 7: 'July', 8: 'August',
        9: 'September', 10: 'October', 11: 'November', 12: 'December',
    }
    monthly_avg['month_name'] = monthly_avg['month'].map(month_names)
    monthly_avg = monthly_avg[['month', 'month_name', 'precip_mm_avg', 'tmax_c_avg', 'tmin_c_avg']]

    monthly_avg['precip_mm_avg'] = monthly_avg['precip_mm_avg'].round(2)
    monthly_avg['tmax_c_avg'] = monthly_avg['tmax_c_avg'].round(2)
    monthly_avg['tmin_c_avg'] = monthly_avg['tmin_c_avg'].round(2)

    return monthly_avg


def create_summary_statistics(all_monthly_data: dict) -> pd.DataFrame:
    """Create summary statistics across all PourPtIDs."""
    combined = []
    for pourptid, monthly_df in all_monthly_data.items():
        temp_df = monthly_df.copy()
        temp_df['PourPtID'] = pourptid
        combined.append(temp_df)

    combined_df = pd.concat(combined, ignore_index=True)

    summary = combined_df.groupby(['month', 'month_name']).agg({
        'precip_mm_avg': ['mean', 'std', 'min', 'max'],
        'tmax_c_avg': ['mean', 'std', 'min', 'max'],
        'tmin_c_avg': ['mean', 'std', 'min', 'max'],
    }).reset_index()

    summary.columns = [
        'month', 'month_name',
        'precip_mean', 'precip_std', 'precip_min', 'precip_max',
        'tmax_mean', 'tmax_std', 'tmax_min', 'tmax_max',
        'tmin_mean', 'tmin_std', 'tmin_min', 'tmin_max',
    ]

    for col in summary.columns:
        if col not in ['month', 'month_name']:
            summary[col] = summary[col].round(2)

    return summary


def main() -> None:
    print("=" * 80)
    print("Climate Data Verification - Monthly Averages")
    print("=" * 80)

    # Read processing summary to get list of PourPtIDs
    summary_file = CLIMATE_DIR / "processing_summary.csv"
    print(f"\nReading processing summary: {summary_file}")
    summary_df = pd.read_csv(summary_file)
    print(f"Found {len(summary_df)} PourPtIDs to process")

    CLIMATE_VERIFICATION_DIR.mkdir(parents=True, exist_ok=True)

    all_monthly_data = {}
    failed_pourptids = []

    for _, row in tqdm(summary_df.iterrows(), total=len(summary_df), desc="Processing PourPtIDs"):
        pourptid = row['PourPtID']

        # Use the repo path rather than whatever was saved in the summary
        climate_file = CLIMATE_DIR / f"climate_{pourptid}.csv"
        if not climate_file.exists():
            print(f"\nWarning: File not found for PourPtID {pourptid}: {climate_file}")
            failed_pourptids.append(pourptid)
            continue

        try:
            monthly_avg = calculate_monthly_averages(str(climate_file), pourptid)
            all_monthly_data[pourptid] = monthly_avg

            output_file = CLIMATE_VERIFICATION_DIR / f"monthly_avg_{pourptid}.csv"
            monthly_avg.to_csv(output_file, index=False)
        except Exception as e:
            print(f"\nError processing PourPtID {pourptid}: {e}")
            failed_pourptids.append(pourptid)

    print("\nCreating summary statistics...")
    summary_stats = create_summary_statistics(all_monthly_data)
    stats_file = CLIMATE_VERIFICATION_DIR / "monthly_averages_summary.csv"
    summary_stats.to_csv(stats_file, index=False)

    print("Creating combined monthly averages file...")
    combined = []
    for pourptid, monthly_df in all_monthly_data.items():
        temp_df = monthly_df.copy()
        temp_df['PourPtID'] = pourptid
        combined.append(temp_df)

    combined_df = pd.concat(combined, ignore_index=True)
    combined_file = CLIMATE_VERIFICATION_DIR / "monthly_averages_all_pourptids.csv"
    combined_df.to_csv(combined_file, index=False)

    print("\n" + "=" * 80)
    print("Verification Complete!")
    print("=" * 80)
    print(f"\nSuccessfully processed {len(all_monthly_data)} PourPtIDs")
    print(f"Failed PourPtIDs: {len(failed_pourptids)}")
    if failed_pourptids:
        print(f"  Failed IDs: {failed_pourptids}")

    print(f"\nOutput directory: {CLIMATE_VERIFICATION_DIR}")
    print(f"Summary statistics file: {stats_file}")
    print(f"Combined file: {combined_file}")


if __name__ == "__main__":
    main()
