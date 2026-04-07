"""Retrieve daily USGS flow and water-temperature data for all stations in USGS_Table_1.csv.

Downloads daily values for discharge (00060) and water temperature (00010)
and saves to data/raw/usgs/{STATION_NO}.csv.
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path
from typing import Optional

import pandas as pd
import dataretrieval.nwis as nwis

from src.paths import STATION_TABLE, RAW_USGS_DIR


def fetch_and_save_station(station_no: str, start_year: int, end_year: int) -> Optional[Path]:
    """Fetch daily values for a station and save to CSV.

    Requests discharge (00060) and water temperature (00010).
    Returns the path to the written file, or None if no data.
    """
    start_date = f"{int(start_year)}-01-01"
    end_date = f"{int(end_year)}-12-31"

    df = nwis.get_record(
        sites=str(station_no),
        service="dv",
        start=start_date,
        end=end_date,
        parameterCd="00060,00010",
    )

    if df is None or len(df) == 0:
        print(f"[WARN] No data returned for station {station_no} in {start_year}-{end_year}")
        return None

    RAW_USGS_DIR.mkdir(parents=True, exist_ok=True)

    out_path = RAW_USGS_DIR / f"{station_no}.csv"
    df_reset = df.reset_index()
    df_reset.to_csv(out_path, index=False)
    return out_path


def main() -> None:
    if not STATION_TABLE.exists():
        print(f"[ERROR] Could not find station list at {STATION_TABLE}")
        sys.exit(1)

    try:
        stations = pd.read_csv(STATION_TABLE)
    except Exception as e:
        print(f"[ERROR] Failed to read {STATION_TABLE}: {e}")
        sys.exit(1)

    required_cols = {"STATION_NO", "REF_BEGIN_YEAR", "REF_END_YEAR"}
    missing = required_cols - set(stations.columns)
    if missing:
        print(f"[ERROR] Missing required columns in USGS_Table_1.csv: {sorted(missing)}")
        sys.exit(1)

    total = len(stations)
    print(f"Found {total} stations. Starting downloads to {RAW_USGS_DIR} ...")

    success = 0
    failures = 0
    has_temp = 0
    for idx, row in stations.iterrows():
        station_no = str(row["STATION_NO"]).strip()
        start_year = int(row["REF_BEGIN_YEAR"]) if not pd.isna(row["REF_BEGIN_YEAR"]) else None
        end_year = int(row["REF_END_YEAR"]) if not pd.isna(row["REF_END_YEAR"]) else None

        if not station_no or start_year is None or end_year is None:
            print(f"[{idx+1}/{total}] [SKIP] Missing station or year range: {station_no} {start_year}-{end_year}")
            failures += 1
            continue

        print(f"[{idx+1}/{total}] Fetching {station_no} for {start_year}-{end_year} ...", end=" ")
        try:
            out = fetch_and_save_station(station_no, start_year, end_year)
            if out is None:
                print("no data.")
                failures += 1
            else:
                saved = pd.read_csv(out, nrows=0)
                temp_cols = [c for c in saved.columns if "00010" in c]
                temp_flag = f" (+temp: {', '.join(temp_cols)})" if temp_cols else ""
                if temp_cols:
                    has_temp += 1
                print(f"saved -> {out.name}{temp_flag}")
                success += 1
        except Exception:
            print("failed.")
            traceback.print_exc()
            failures += 1

    print(f"\nDone. Success: {success}, Failures: {failures}, Total: {total}")
    print(f"Stations with water temperature data: {has_temp}/{success}")


if __name__ == "__main__":
    main()
