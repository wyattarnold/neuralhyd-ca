"""Watershed QA/QC: Flow (mm/day) vs Precipitation (mm/day).

Converts daily CFS to mm/day using watershed area, compares against
area-weighted precipitation to flag suspect watersheds, and sorts
cleaned flow files into tier subdirectories under data/training/flow/.
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.io import load_climate_dataframes, write_flow_zarr, water_year
from src.paths import (
    BASIN_ATLAS_OUTPUT,
    CLIMATE_WATERSHEDS_ZARR,
    FLOW_CLEANED_STRICT_DIR,
    FLOW_ZARR,
    STEP_8_OUTPUT_DIR,
    WATERSHED_GEOMETRY,
)

# Conversion factor: CFS -> mm/day over km2
CFS_TO_MM_FACTOR = 0.028316847 * 86400.0 / 1000.0  # = 2.44657...

# Thresholds
LT_RATIO_HIGH = 0.8
LT_RATIO_LOW  = 0.1
MAX_FLOW_MM   = 500.0


def load_areas(path: str | Path) -> dict[str, float]:
    """Load PourPtID -> area_km2 from BasinATLAS CSV."""
    areas = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = row["PourPtID"].strip()
            area = float(row["total_Shape_Area_km2"])
            areas[pid] = area
    return areas


def load_flow(path: str | Path) -> list[dict]:
    """Load flow file -> list of dicts with date, flow (cfs)."""
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                flow_val = float(row["flow"])
            except (ValueError, KeyError):
                continue
            rows.append({"date": row["date"], "flow_cfs": flow_val})
    return rows


def load_climate_from_zarr(zarr_path: Path) -> dict[str, dict[str, dict]]:
    """Load all climate from zarr cube into ``{pid: {date: {precip_mm, tmean_c}}}``."""
    dfs = load_climate_dataframes(zarr_path)
    out: dict[str, dict[str, dict]] = {}
    for bid, df in dfs.items():
        # df index is daily DatetimeIndex; convert once.
        precip = df["precip_mm"].to_numpy()
        tmean = ((df["tmax_c"].to_numpy() + df["tmin_c"].to_numpy()) / 2.0)
        date_strs = df.index.strftime("%Y-%m-%d").to_numpy()
        out[str(bid)] = {
            d: {"precip_mm": float(p), "tmean_c": float(t)}
            for d, p, t in zip(date_strs, precip, tmean)
            if not (np.isnan(p) or np.isnan(t))
        }
    return out


def monthly_regression_metrics(merged_data: list[dict]) -> dict:
    """Monthly OLS: flow_mm ~ precip_mm + tmean_c + prev_12mo_precip."""
    monthly = defaultdict(lambda: {"flow_mm": 0.0, "precip_mm": 0.0, "tmean_sum": 0.0, "n": 0})
    for r in merged_data:
        ym = r["date"][:7]
        monthly[ym]["flow_mm"] += r["flow_mm"]
        monthly[ym]["precip_mm"] += r["precip_mm"]
        monthly[ym]["tmean_sum"] += r["tmean_c"]
        monthly[ym]["n"] += 1

    sorted_months = sorted(monthly.keys())
    precip_by_month = [monthly[ym]["precip_mm"] for ym in sorted_months]

    rows = []
    for i, ym in enumerate(sorted_months):
        v = monthly[ym]
        if v["n"] == 0 or i < 12:
            continue
        prev_12mo_precip = sum(precip_by_month[i - 12:i])
        rows.append((v["flow_mm"], v["precip_mm"], v["tmean_sum"] / v["n"], prev_12mo_precip))

    nan_result = {"r2": float("nan"), "rsr": float("nan"), "pbias": float("nan"),
                  "pearson_r": float("nan"), "mkge": float("nan")}

    if len(rows) < 5:
        return nan_result

    X = np.array([[1.0, r[1], r[2], r[3]] for r in rows])
    y = np.array([r[0] for r in rows])
    try:
        beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        y_sim = X @ beta
        ss_res = float(np.sum((y - y_sim) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        if ss_tot == 0:
            return nan_result
        r2 = 1.0 - ss_res / ss_tot
        rsr = float(np.sqrt(ss_res / ss_tot))
        pbias = float(100.0 * np.sum(y_sim - y) / np.sum(y)) if np.sum(y) != 0 else float("nan")

        corr_mat = np.corrcoef(y, y_sim)
        pearson_r = float(corr_mat[0, 1])

        mean_obs = float(y.mean())
        mean_sim = float(y_sim.mean())
        std_obs = float(y.std(ddof=1)) if len(y) > 1 else float("nan")
        std_sim = float(y_sim.std(ddof=1)) if len(y_sim) > 1 else float("nan")
        if mean_obs == 0 or mean_sim == 0 or std_obs == 0:
            mkge = float("nan")
        else:
            beta_b = mean_sim / mean_obs
            gamma_v = (std_sim / mean_sim) / (std_obs / mean_obs)
            mkge = float(1.0 - np.sqrt((pearson_r - 1) ** 2 + (beta_b - 1) ** 2 + (gamma_v - 1) ** 2))

        return {"r2": r2, "rsr": rsr, "pbias": pbias, "pearson_r": pearson_r, "mkge": mkge}
    except Exception:
        return nan_result


def main(include_cdec: bool = False) -> None:
    STEP_8_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    areas = load_areas(BASIN_ATLAS_OUTPUT)
    if include_cdec and WATERSHED_GEOMETRY.exists():
        ws = pd.read_csv(WATERSHED_GEOMETRY, dtype={"Pour Point ID": str})
        for _, row in ws.iterrows():
            pid = str(row["Pour Point ID"]).strip()
            if pid not in areas and pid.startswith("990000"):
                areas[pid] = float(row["Area Square Kilometers"])
    print(f"Loaded {len(areas)} watershed areas.")

    flow_files = sorted(FLOW_CLEANED_STRICT_DIR.glob("*_cleaned.csv"))
    print(f"Found {len(flow_files)} flow files.\n")

    print(f"Loading climate from {CLIMATE_WATERSHEDS_ZARR.name} ...")
    climate_by_pid = load_climate_from_zarr(CLIMATE_WATERSHEDS_ZARR)
    print(f"  loaded {len(climate_by_pid)} basins from zarr")

    summary_rows = []
    flag_rows = []
    annual_detail_rows = []
    tier_rows = []

    for fpath in flow_files:
        fname = fpath.name
        pid = fname.replace("_cleaned.csv", "")

        if pid not in areas:
            flag_rows.append({"PourPtID": pid, "Flag": "NO_AREA", "Detail": "Missing from BasinATLAS"})
            continue

        area_km2 = areas[pid]
        if area_km2 <= 0:
            flag_rows.append({"PourPtID": pid, "Flag": "ZERO_AREA", "Detail": f"area={area_km2}"})
            continue

        flow_data = load_flow(fpath)
        if not flow_data:
            flag_rows.append({"PourPtID": pid, "Flag": "EMPTY_FLOW", "Detail": "No valid flow records"})
            continue

        clim_data_full = climate_by_pid.get(pid)
        if clim_data_full is None:
            flag_rows.append({"PourPtID": pid, "Flag": "NO_CLIMATE", "Detail": "Missing climate in zarr"})
            continue
        flow_dates = set(r["date"] for r in flow_data)
        clim_data = {d: v for d, v in clim_data_full.items() if d in flow_dates}

        merged = []
        for r in flow_data:
            if r["date"] in clim_data:
                c = clim_data[r["date"]]
                merged.append({
                    "date": r["date"],
                    "flow_cfs": r["flow_cfs"],
                    "flow_mm": r["flow_cfs"] * CFS_TO_MM_FACTOR / area_km2,
                    "precip_mm": c["precip_mm"],
                    "tmean_c": c["tmean_c"],
                })
        if not merged:
            flag_rows.append({"PourPtID": pid, "Flag": "NO_OVERLAP", "Detail": "No date overlap between flow and climate"})
            continue
        flow_data = merged

        # --- Runoff ratios by water year ---
        wy_flow = defaultdict(float)
        wy_precip = defaultdict(float)
        wy_days = defaultdict(int)
        total_flow_mm = 0.0
        total_precip_mm = 0.0

        for r in flow_data:
            wy = water_year(r["date"])
            wy_flow[wy] += r["flow_mm"]
            wy_precip[wy] += r["precip_mm"]
            wy_days[wy] += 1
            total_flow_mm += r["flow_mm"]
            total_precip_mm += r["precip_mm"]

        annual_ratios = {}
        for wy in sorted(wy_flow.keys()):
            if wy_days[wy] >= 300 and wy_precip[wy] > 0:
                annual_ratios[wy] = wy_flow[wy] / wy_precip[wy]
            if wy_days[wy] >= 300:
                ratio = wy_flow[wy] / wy_precip[wy] if wy_precip[wy] > 0 else float("nan")
                annual_detail_rows.append({
                    "PourPtID": pid,
                    "water_year": wy,
                    "n_days": wy_days[wy],
                    "total_flow_mm": round(wy_flow[wy], 2),
                    "total_precip_mm": round(wy_precip[wy], 2),
                    "runoff_ratio": round(ratio, 4) if not np.isnan(ratio) else np.nan,
                })

        lt_ratio = total_flow_mm / total_precip_mm if total_precip_mm > 0 else float("nan")

        ratio_values = list(annual_ratios.values())
        min_ratio = min(ratio_values) if ratio_values else float("nan")
        max_ratio = max(ratio_values) if ratio_values else float("nan")
        mean_ratio = sum(ratio_values) / len(ratio_values) if ratio_values else float("nan")

        # --- Daily flow_mm extremes ---
        flow_mm_vals = [r["flow_mm"] for r in flow_data]
        max_flow_mm = max(flow_mm_vals)
        mean_flow_mm = sum(flow_mm_vals) / len(flow_mm_vals)
        mean_flow_cfs = sum(r["flow_cfs"] for r in flow_data) / len(flow_data)

        precip_vals = [r["precip_mm"] for r in flow_data]
        mean_precip_mm = sum(precip_vals) / len(precip_vals)
        max_precip_mm = max(precip_vals) if precip_vals else 0

        n_zero = sum(1 for v in flow_mm_vals if v == 0)
        zero_frac = n_zero / len(flow_mm_vals) if flow_mm_vals else 0

        # --- Regression metrics ---
        metrics = monthly_regression_metrics(flow_data)
        r2, rsr, pbias = metrics["r2"], metrics["rsr"], metrics["pbias"]
        pearson_r, mkge = metrics["pearson_r"], metrics["mkge"]
        tier_rows.append((str(fpath), r2))

        # --- Compile flags ---
        flags = []

        if np.isnan(lt_ratio):
            flags.append("LT_RATIO_NAN")
        elif lt_ratio > LT_RATIO_HIGH:
            flags.append(f"LT_RATIO_HIGH({lt_ratio:.3f})")
        elif lt_ratio < LT_RATIO_LOW:
            flags.append(f"LT_RATIO_LOW({lt_ratio:.4f})")

        if max_flow_mm > MAX_FLOW_MM:
            flags.append(f"MAX_FLOW_MM({max_flow_mm:.1f})")

        if max_precip_mm > 0 and (max_flow_mm / max_precip_mm) > 5:
            flags.append(f"MAX_FLOW_PRECIP_RATIO({max_flow_mm / max_precip_mm:.1f})")

        if area_km2 < 1.0:
            flags.append(f"TINY_AREA({area_km2:.4f}km2)")

        flag_str = "; ".join(flags) if flags else "PASS"

        summary_rows.append({
            "PourPtID": pid,
            "area_km2": round(area_km2, 4),
            "n_days": len(flow_data),
            "n_water_years": len(annual_ratios),
            "mean_flow_cfs": round(mean_flow_cfs, 3),
            "mean_flow_mm": round(mean_flow_mm, 4),
            "mean_precip_mm": round(mean_precip_mm, 4),
            "lt_runoff_ratio": round(lt_ratio, 4) if not np.isnan(lt_ratio) else np.nan,
            "min_annual_ratio": round(min_ratio, 4) if not np.isnan(min_ratio) else np.nan,
            "mean_annual_ratio": round(mean_ratio, 4) if not np.isnan(mean_ratio) else np.nan,
            "max_annual_ratio": round(max_ratio, 4) if not np.isnan(max_ratio) else np.nan,
            "max_flow_mm": round(max_flow_mm, 2),
            "max_precip_mm": round(max_precip_mm, 2),
            "zero_flow_frac": round(zero_frac, 4),
            "monthly_regression_r2": round(r2, 4) if not np.isnan(r2) else np.nan,
            "monthly_regression_rsr": round(rsr, 4) if not np.isnan(rsr) else np.nan,
            "monthly_regression_pbias": round(pbias, 2) if not np.isnan(pbias) else np.nan,
            "monthly_regression_pearson_r": round(pearson_r, 4) if not np.isnan(pearson_r) else np.nan,
            "monthly_regression_mkge": round(mkge, 4) if not np.isnan(mkge) else np.nan,
            "flags": flag_str,
        })

        if flags:
            for fl in flags:
                flag_rows.append({"PourPtID": pid, "Flag": fl.split("(")[0], "Detail": fl})

        r2_str = f"{r2:.3f}" if not np.isnan(r2) else "NaN"
        rsr_str = f"{rsr:.3f}" if not np.isnan(rsr) else "NaN"
        pbias_str = f"{pbias:.1f}" if not np.isnan(pbias) else "NaN"
        mkge_str = f"{mkge:.3f}" if not np.isnan(mkge) else "NaN"
        print(f"  {pid}: lt_ratio={lt_ratio:.4f}, r2={r2_str}, rsr={rsr_str}, pbias={pbias_str}%, mkge={mkge_str}, flags={flag_str}")

    # --- Write outputs ---
    if summary_rows:
        sum_path = STEP_8_OUTPUT_DIR / "qaqc_flow_vs_precip_summary.csv"
        pd.DataFrame(summary_rows).to_csv(sum_path, index=False)
        print(f"\nSummary: {sum_path} ({len(summary_rows)} watersheds)")

    flag_path = STEP_8_OUTPUT_DIR / "qaqc_flow_vs_precip_flags.csv"
    pd.DataFrame(flag_rows, columns=["PourPtID", "Flag", "Detail"]).to_csv(flag_path, index=False)
    print(f"Flags:   {flag_path} ({len(flag_rows)} flags)")

    if annual_detail_rows:
        ann_path = STEP_8_OUTPUT_DIR / "qaqc_annual_runoff_ratios.csv"
        pd.DataFrame(annual_detail_rows).to_csv(ann_path, index=False)
        print(f"Annual:  {ann_path} ({len(annual_detail_rows)} water-year rows)")

    # --- Build flow.zarr from tier classification ---
    tier_map: dict[int, int] = {}
    flow_data_map: dict[int, pd.DataFrame] = {}
    n_tiers = {1: 0, 2: 0, 3: 0, "unclassified": 0}
    for fpath_str, r2 in tier_rows:
        fpath = Path(fpath_str)
        pid = fpath.stem.replace("_cleaned", "")
        try:
            bid = int(pid)
        except ValueError:
            continue
        if np.isnan(r2):  # NaN
            n_tiers["unclassified"] += 1
            continue
        if r2 > 0.6:
            tier = 1
        elif r2 >= 0.2:
            tier = 2
        else:
            tier = 3
        df = pd.read_csv(
            fpath, usecols=["date", "flow"], parse_dates=["date"]
        ).set_index("date")
        df["flow"] = df["flow"].astype("float32")
        flow_data_map[bid] = df
        tier_map[bid] = tier
        n_tiers[tier] += 1

    print(f"\nTier classification:")
    print(f"  tier_1 (R2 > 0.6):         {n_tiers[1]} basins")
    print(f"  tier_2 (0.2 <= R2 <= 0.6): {n_tiers[2]} basins")
    print(f"  tier_3 (R2 < 0.2):         {n_tiers[3]} basins")
    if n_tiers["unclassified"]:
        print(f"  unclassified (NaN R2):     {n_tiers['unclassified']} basins")

    if flow_data_map:
        print(f"\nWriting {FLOW_ZARR}  ({len(flow_data_map)} basins)")
        write_flow_zarr(FLOW_ZARR, flow_data_map, tier_map=tier_map, overwrite=True)

    # --- Print overall summary ---
    n_pass = sum(1 for r in summary_rows if r["flags"] == "PASS")
    n_flag = sum(1 for r in summary_rows if r["flags"] != "PASS")
    print(f"\n{'='*60}")
    print(f"RESULTS: {n_pass} PASS, {n_flag} FLAGGED out of {len(summary_rows)} watersheds")
    print(f"{'='*60}")

    if n_flag > 0:
        print(f"\nFlagged watersheds:")
        print(f"{'PourPtID':<15} {'LT Ratio':<12} {'Max Flow mm':<14} {'Flags'}")
        print("-" * 90)
        for r in summary_rows:
            if r["flags"] != "PASS":
                print(f"{r['PourPtID']:<15} {r['lt_runoff_ratio']:<12} {r['max_flow_mm']:<14} {r['flags']}")


if __name__ == "__main__":
    main()
