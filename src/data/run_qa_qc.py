"""Comprehensive QA/QC for climate and cleaned flow data.

Sections and output files (written to data/prepare/):
  1  Climate QA        -> sec1_climate_qa.csv
  2  Flow QA           -> sec2_flow_qa.csv
  3  Cross-dataset     -> sec3_cross_dataset.csv
  4  Raw vs Cleaned    -> sec4_raw_vs_cleaned.csv
  5  Zero Water Years  -> sec5_zero_wy_audit.csv
  6  Zero-WY Drops     -> sec6_null_flow_drops.csv
  7  Cleaned Strict    -> sec7_cleaned_strict.csv
  8  Excluded Basins   -> sec8_excluded_basins.csv + figures/excluded_basins/
  Report               -> qa_qc_report.txt
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

from src.paths import (
    CLIMATE_DIR,
    FLOW_CLEANED_DIR,
    FLOW_DROPPED_DIR,
    FLOW_CLEANED_STRICT_DIR,
    STEP_7_OUTPUT_DIR,
    RAW_USGS_DIR,
)

# ---------------------------------------------------------------------------
# Value-range thresholds
# ---------------------------------------------------------------------------
PRECIP_MAX_MM = 600.0
TMAX_MAX_C    =  45.0
TMAX_MIN_C    = -20.0
TMIN_MAX_C    =  40.0
TMIN_MIN_C    = -25.0

def _tval(v): return f"neg{abs(int(v))}" if v < 0 else str(int(v))
COL_PRECIP_HI = f"precip_over_{int(PRECIP_MAX_MM)}mm"
COL_TMAX_HI   = f"tmax_above_{_tval(TMAX_MAX_C)}c"
COL_TMAX_LO   = f"tmax_below_{_tval(TMAX_MIN_C)}c"
COL_TMIN_HI   = f"tmin_above_{_tval(TMIN_MAX_C)}c"
COL_TMIN_LO   = f"tmin_below_{_tval(TMIN_MIN_C)}c"

INTENTIONAL_EXCLUSIONS = {
    "11299000": "wrong gage identifier (reservoir storage, not flow)",
    "11446220": "water temperature gage, not flow",
    "11406999": "suspect observations",
    "11355500": "suspect baseflow",
    "11195500": "suspect observations",
    "11274790": "suspect observations",
    "11376500": "suspect baseflow",
    "11400000": "suspect baseflow",
    "10308783": "tiny, sparse observations",
}

FLOW_COL_ORDER = ["00060_Mean", "00060_2_Mean", "00054_Observation at 24:00"]

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def _sid(p: Path, prefix: str = "", suffix: str = "") -> str:
    n = p.stem
    if prefix: n = n.removeprefix(prefix)
    if suffix: n = n.removesuffix(suffix)
    return n

def _gap_stats(dates: pd.Series) -> dict:
    dates = pd.to_datetime(dates).sort_values().reset_index(drop=True)
    if dates.empty:
        return {"gap_count": 0, "max_gap_days": 0, "total_missing_days": 0}
    expected = pd.date_range(dates.iloc[0], dates.iloc[-1], freq="D")
    missing  = len(expected.difference(dates))
    diffs    = dates.diff().dt.days.dropna()
    gaps     = diffs[diffs > 1]
    return {
        "gap_count":          int(len(gaps)),
        "max_gap_days":       int(gaps.max()) if len(gaps) else 0,
        "total_missing_days": missing,
    }

def _pct(n, total): return round(100.0 * n / total, 4) if total else np.nan

def _wy(dt) -> int:
    dt = pd.Timestamp(dt)
    return dt.year + 1 if dt.month >= 10 else dt.year

def _detect_flow_col(cols):
    for c in FLOW_COL_ORDER:
        if c in cols: return c
    candidates = [c for c in cols if "60" in c and c.endswith("_Mean")]
    return candidates[0] if candidates else None

def _parse_raw_datetimes(series: pd.Series) -> pd.Series:
    if series.dt.tz is not None:
        series = series.dt.tz_convert(None)
    return series.dt.normalize()

def _s(s): return int(s.sum()) if s.notna().any() else 0
def _c(cond): return int(cond.sum())


# ============================================================================
# Section runners
# ============================================================================

def _run_climate_qa() -> pd.DataFrame:
    print("=== Section 1: Climate QA ===")
    rows = []
    for fp in sorted(CLIMATE_DIR.glob("climate_*.csv")):
        sid = _sid(fp, prefix="climate_")
        try:
            df = pd.read_csv(fp, parse_dates=["date"])
        except Exception as e:
            rows.append({"station_id": sid, "error": str(e)}); continue

        n   = len(df)
        g   = _gap_stats(df["date"])
        np_ = int(df["precip_mm"].isna().sum())
        nt  = int(df["tmax_c"].isna().sum())
        ntn = int(df["tmin_c"].isna().sum())
        neg = int((df["precip_mm"] < 0).sum())
        hp  = int((df["precip_mm"] > PRECIP_MAX_MM).sum())
        txh = int((df["tmax_c"] > TMAX_MAX_C).sum())
        txl = int((df["tmax_c"] < TMAX_MIN_C).sum())
        tnh = int((df["tmin_c"] > TMIN_MAX_C).sum())
        tnl = int((df["tmin_c"] < TMIN_MIN_C).sum())
        inv = int((df["tmax_c"] < df["tmin_c"]).sum())

        rows.append({
            "station_id":         sid,
            "n_rows":             n,
            "date_start":         df["date"].min().date(),
            "date_end":           df["date"].max().date(),
            "gap_count":          g["gap_count"],
            "max_gap_days":       g["max_gap_days"],
            "total_missing_days": g["total_missing_days"],
            "duplicate_dates":    int(df["date"].duplicated().sum()),
            "null_precip_mm":     np_,
            "null_tmax_c":        nt,
            "null_tmin_c":        ntn,
            "neg_precip_count":   neg,
            COL_PRECIP_HI:        hp,
            COL_TMAX_HI:          txh,
            COL_TMAX_LO:          txl,
            COL_TMIN_HI:          tnh,
            COL_TMIN_LO:          tnl,
            "tmax_lt_tmin":       inv,
            "precip_max_mm":      round(df["precip_mm"].max(), 4),
            "precip_p99_mm":      round(df["precip_mm"].quantile(0.99), 4),
            "tmax_max_c":         round(df["tmax_c"].max(), 4),
            "tmax_min_c":         round(df["tmax_c"].min(), 4),
            "tmin_max_c":         round(df["tmin_c"].max(), 4),
            "tmin_min_c":         round(df["tmin_c"].min(), 4),
            "any_flag":           (np_ + nt + ntn + neg + hp + inv) > 0,
        })

    clim_df = pd.DataFrame(rows)
    clim_df.to_csv(STEP_7_OUTPUT_DIR / "sec1_climate_qa.csv", index=False)
    print(f"  {len(clim_df)} stations -> sec1_climate_qa.csv")
    return clim_df


def _run_flow_qa() -> pd.DataFrame:
    print("=== Section 2: Flow QA ===")
    rows = []
    for fp in sorted(FLOW_CLEANED_DIR.glob("*_cleaned.csv")):
        sid = _sid(fp, suffix="_cleaned")
        try:
            df = pd.read_csv(fp, parse_dates=["date"])
        except Exception as e:
            rows.append({"station_id": sid, "error": str(e)}); continue

        n   = len(df)
        g   = _gap_stats(df["date"])
        nf  = int(df["flow"].isna().sum())
        neg = int((df["flow"] < 0).sum())
        z   = int((df["flow"] == 0).sum())

        flow_cd_counts = {}
        if "flow_cd" in df.columns:
            for code, cnt in df["flow_cd"].value_counts(dropna=False).items():
                flow_cd_counts[f"flow_cd_{code}"] = int(cnt)

        fit_false = np.nan
        if "flow_7day_fit" in df.columns:
            fit_false = int((df["flow_7day_fit"] == False).sum())  # noqa: E712

        tlt = np.nan
        if "tmax_c" in df.columns and "tmin_c" in df.columns:
            tlt = int((df["tmax_c"] < df["tmin_c"]).sum())

        rec = {
            "station_id":         sid,
            "n_rows":             n,
            "date_start":         df["date"].min().date(),
            "date_end":           df["date"].max().date(),
            "gap_count":          g["gap_count"],
            "max_gap_days":       g["max_gap_days"],
            "total_missing_days": g["total_missing_days"],
            "duplicate_dates":    int(df["date"].duplicated().sum()),
            "null_flow":          nf,
            "null_precip_mm":     int(df["precip_mm"].isna().sum()) if "precip_mm" in df else np.nan,
            "null_tmax_c":        int(df["tmax_c"].isna().sum())    if "tmax_c"    in df else np.nan,
            "null_tmin_c":        int(df["tmin_c"].isna().sum())    if "tmin_c"    in df else np.nan,
            "neg_flow_count":     neg,
            "zero_flow_count":    z,
            "flow_7day_fit_false": fit_false,
            "tmax_lt_tmin":       tlt,
            "flow_mean":          round(df["flow"].mean(), 4),
            "flow_median":        round(df["flow"].median(), 4),
            "flow_min":           round(df["flow"].min(), 4),
            "flow_max":           round(df["flow"].max(), 4),
            "flow_p999":          round(df["flow"].quantile(0.999), 4),
            "any_flag":           (nf + neg) > 0,
        }
        rec.update(flow_cd_counts)
        rows.append(rec)

    flow_df = pd.DataFrame(rows)
    flow_df.to_csv(STEP_7_OUTPUT_DIR / "sec2_flow_qa.csv", index=False)
    print(f"  {len(flow_df)} stations -> sec2_flow_qa.csv")
    return flow_df


def _run_cross_dataset(clim_df: pd.DataFrame, flow_df: pd.DataFrame) -> pd.DataFrame:
    print("=== Section 3: Cross-dataset alignment ===")

    clim_sids = set(clim_df["station_id"].astype(str))
    flow_sids = set(flow_df["station_id"].astype(str))
    in_both   = clim_sids & flow_sids
    clim_only = clim_sids - flow_sids
    flow_only = flow_sids - clim_sids

    clim_idx = clim_df.set_index("station_id")[["date_start", "date_end"]].to_dict("index")
    flow_idx = flow_df.set_index("station_id")[["date_start", "date_end"]].to_dict("index")

    rows = []
    for sid in sorted(in_both):
        c = clim_idx[sid]; f = flow_idx[sid]
        cs = pd.Timestamp(c["date_start"]); ce = pd.Timestamp(c["date_end"])
        fs = pd.Timestamp(f["date_start"]); fe = pd.Timestamp(f["date_end"])
        ov_s = max(cs, fs); ov_e = min(ce, fe)
        rows.append({
            "station_id":               sid,
            "climate_start":            c["date_start"],
            "climate_end":              c["date_end"],
            "flow_start":               f["date_start"],
            "flow_end":                 f["date_end"],
            "overlap_days":             int((ov_e - ov_s).days) + 1 if ov_e >= ov_s else 0,
            "flow_days_before_climate": max(0, int((cs - fs).days)),
            "flow_days_after_climate":  max(0, int((fe - ce).days)),
            "has_full_coverage":        cs <= fs and ce >= fe,
        })

    for sid in sorted(clim_only):
        c = clim_idx[sid]
        rows.append({
            "station_id":    sid,
            "climate_start": c["date_start"],
            "climate_end":   c["date_end"],
            "note":          INTENTIONAL_EXCLUSIONS.get(sid, "climate only - no flow file"),
        })

    for sid in sorted(flow_only):
        f = flow_idx[sid]
        rows.append({
            "station_id": sid,
            "flow_start": f["date_start"],
            "flow_end":   f["date_end"],
            "note":       "flow only - no climate file",
        })

    cross_df = pd.DataFrame(rows).sort_values("station_id")
    cross_df.to_csv(STEP_7_OUTPUT_DIR / "sec3_cross_dataset.csv", index=False)
    print(f"  {len(cross_df)} stations -> sec3_cross_dataset.csv")
    return cross_df


def _run_raw_vs_cleaned() -> pd.DataFrame:
    print("=== Section 4: Raw vs Cleaned reconciliation ===")
    rows = []
    for fp_clean in sorted(FLOW_CLEANED_DIR.glob("*_cleaned.csv")):
        sid        = _sid(fp_clean, suffix="_cleaned")
        fp_raw     = RAW_USGS_DIR     / f"{sid}.csv"
        fp_dropped = FLOW_DROPPED_DIR / f"{sid}_dropped.csv"

        try:
            cl = pd.read_csv(fp_clean, parse_dates=["date"], usecols=["date"])
            cleaned_dates = set(cl["date"].dt.date)
        except Exception as e:
            rows.append({"station_id": sid, "error": f"cleaned: {e}"}); continue

        dropped_dates: set = set()
        if fp_dropped.exists():
            try:
                dr = pd.read_csv(fp_dropped, parse_dates=["date"], usecols=["date"])
                dropped_dates = set(dr["date"].dt.date)
            except Exception:
                pass
        dropped_count = len(dropped_dates)

        if not fp_raw.exists():
            rows.append({
                "station_id": sid, "raw_file_exists": False,
                "cleaned_days": len(cleaned_dates), "dropped_days": dropped_count,
            }); continue

        try:
            rw       = pd.read_csv(fp_raw, parse_dates=["datetime"])
            rw["date"] = _parse_raw_datetimes(rw["datetime"])
            flow_col   = _detect_flow_col(rw.columns)

            if flow_col is None:
                rows.append({
                    "station_id": sid, "raw_file_exists": True,
                    "raw_no_discharge_col": True,
                    "raw_total_rows": len(rw),
                    "cleaned_days": len(cleaned_dates), "dropped_days": dropped_count,
                    "note": INTENTIONAL_EXCLUSIONS.get(sid, "no discharge col"),
                }); continue

            null_mask       = rw[flow_col].isna()
            raw_null_dates  = set(rw.loc[ null_mask, "date"].dt.date)
            raw_valid_dates = set(rw.loc[~null_mask, "date"].dt.date)
            raw_null        = len(raw_null_dates)
            raw_valid       = len(raw_valid_dates)

        except Exception as e:
            rows.append({"station_id": sid, "error": f"raw: {e}"}); continue

        if cleaned_dates:
            c_min        = min(cleaned_dates); c_max = max(cleaned_dates)
            window_dates = set(pd.date_range(c_min, c_max, freq="D").date)
            gap_dates    = window_dates - cleaned_dates
        else:
            gap_dates = set()

        rows.append({
            "station_id":                 sid,
            "raw_file_exists":            True,
            "raw_total_rows":             len(rw),
            "raw_null_flow_days":         raw_null,
            "raw_valid_flow_days":        raw_valid,
            "cleaned_days":               len(cleaned_dates),
            "dropped_days":               dropped_count,
            "cleaned_plus_dropped":       len(cleaned_dates) + dropped_count,
            "pct_valid_retained":         _pct(len(cleaned_dates), raw_valid),
            "pct_valid_dropped":          _pct(dropped_count, raw_valid),
            "total_gap_days":             len(gap_dates),
            "gaps_from_raw_null":         len(gap_dates & raw_null_dates),
            "gaps_from_dropped":          len(gap_dates & dropped_dates),
            "gaps_raw_valid_unaccounted": len((gap_dates & raw_valid_dates) - dropped_dates),
            "gaps_unexplained":           len(gap_dates - raw_null_dates - dropped_dates - raw_valid_dates),
        })

    audit_df = pd.DataFrame(rows)
    audit_df.to_csv(STEP_7_OUTPUT_DIR / "sec4_raw_vs_cleaned.csv", index=False)
    print(f"  {len(audit_df)} stations -> sec4_raw_vs_cleaned.csv")
    return audit_df


def _run_zero_wy_detection() -> pd.DataFrame:
    print("=== Section 5: Zero water year detection ===")
    zero_wy_rows = []
    for fp_raw in sorted(RAW_USGS_DIR.glob("*.csv")):
        sid = fp_raw.stem
        try:
            rw = pd.read_csv(fp_raw, parse_dates=["datetime"])
            rw["date"]   = _parse_raw_datetimes(rw["datetime"])
            flow_col = _detect_flow_col(rw.columns)
            if flow_col is None:
                continue
            rw["_wy"] = rw["date"].apply(_wy)
            for wy, grp in rw.groupby("_wy"):
                valid  = grp[flow_col].dropna()
                if len(valid) == 0:
                    continue
                n_zero = int((valid == 0).sum())
                if n_zero == len(valid):
                    zero_wy_rows.append({
                        "station_id":   sid,
                        "water_year":   int(wy),
                        "n_raw_days":   len(grp),
                        "n_valid_days": len(valid),
                        "n_zero_days":  n_zero,
                    })
        except Exception:
            continue

    zero_wy_df = (
        pd.DataFrame(zero_wy_rows).sort_values(["station_id", "water_year"])
        if zero_wy_rows
        else pd.DataFrame(columns=["station_id", "water_year",
                                    "n_raw_days", "n_valid_days", "n_zero_days"])
    )
    zero_wy_df.to_csv(STEP_7_OUTPUT_DIR / "sec5_zero_wy_audit.csv", index=False)
    n_wy_stations = zero_wy_df["station_id"].nunique() if not zero_wy_df.empty else 0
    print(f"  {n_wy_stations} stations, {len(zero_wy_df)} flagged WYs -> sec5_zero_wy_audit.csv")
    return zero_wy_df


def _run_null_flow_drops() -> tuple[int, int, pd.DataFrame]:
    print("=== Section 6: Null flow drops ===")
    null_drop_rows = []
    n_null_changed = 0
    total_null_moved = 0

    FLOW_DROPPED_DIR.mkdir(parents=True, exist_ok=True)

    for fp_clean in sorted(FLOW_CLEANED_DIR.glob("*_cleaned.csv")):
        sid        = _sid(fp_clean, suffix="_cleaned")
        fp_dropped = FLOW_DROPPED_DIR / f"{sid}_dropped.csv"

        try:
            cf = pd.read_csv(fp_clean, parse_dates=["date"])
        except Exception as e:
            print(f"  WARNING: {sid} read error - {e}"); continue

        to_move = cf[cf["flow"].isna()]
        to_keep = cf[cf["flow"].notna()]

        if len(to_move) == 0:
            continue

        if fp_dropped.exists():
            existing = pd.read_csv(fp_dropped, parse_dates=["date"])
            new_drop = (pd.concat([existing, to_move], ignore_index=True)
                        .sort_values("date").reset_index(drop=True))
        else:
            new_drop = to_move.sort_values("date").reset_index(drop=True)

        to_keep.to_csv(fp_clean,   index=False)
        new_drop.to_csv(fp_dropped, index=False)

        n_null_changed   += 1
        total_null_moved += len(to_move)
        null_drop_rows.append({"station_id": sid, "null_days_moved": len(to_move)})
        print(f"  {sid}: moved {len(to_move)} null-flow rows")

    if n_null_changed == 0:
        print("  No null flow rows found in cleaned files.")

    null_drop_df = (
        pd.DataFrame(null_drop_rows) if null_drop_rows
        else pd.DataFrame(columns=["station_id", "null_days_moved"])
    )
    null_drop_df.to_csv(STEP_7_OUTPUT_DIR / "sec6_null_flow_drops.csv", index=False)
    print(f"  {n_null_changed} stations modified, {total_null_moved} days moved -> sec6_null_flow_drops.csv")
    return n_null_changed, total_null_moved, null_drop_df


def _run_cleaned_strict() -> tuple[pd.DataFrame, int, int]:
    print("=== Section 7: cleaned_strict ===")
    STRICT_EXCLUDE = {"A, e", "A, R"}
    # Remove stale files (e.g. previously-excluded basins) before rewriting
    if FLOW_CLEANED_STRICT_DIR.exists():
        for old in FLOW_CLEANED_STRICT_DIR.glob("*_cleaned.csv"):
            old.unlink()
    FLOW_CLEANED_STRICT_DIR.mkdir(parents=True, exist_ok=True)

    strict_rows = []
    for fp_clean in sorted(FLOW_CLEANED_DIR.glob("*_cleaned.csv")):
        sid = _sid(fp_clean, suffix="_cleaned")

        if sid in INTENTIONAL_EXCLUSIONS:
            print(f"  {sid}: skipped (excluded — {INTENTIONAL_EXCLUSIONS[sid]})")
            continue

        try:
            cf = pd.read_csv(fp_clean, parse_dates=["date"])
        except Exception as e:
            print(f"  WARNING: {sid} read error - {e}"); continue

        if "flow_cd" in cf.columns:
            exclude_mask = cf["flow_cd"].isin(STRICT_EXCLUDE)
        else:
            exclude_mask = pd.Series(False, index=cf.index)

        strict = cf[~exclude_mask]
        strict.to_csv(FLOW_CLEANED_STRICT_DIR / f"{sid}_cleaned.csv", index=False)

        strict_rows.append({
            "station_id":       sid,
            "cleaned_days":     len(cf),
            "strict_days":      len(strict),
            "excluded_Ae":      int(cf[cf["flow_cd"] == "A, e"].shape[0]) if "flow_cd" in cf.columns else 0,
            "excluded_AR":      int(cf[cf["flow_cd"] == "A, R"].shape[0]) if "flow_cd" in cf.columns else 0,
        })

    strict_df = pd.DataFrame(strict_rows)
    strict_df.to_csv(STEP_7_OUTPUT_DIR / "sec7_cleaned_strict.csv", index=False)
    n_strict_affected = _c(strict_df["excluded_Ae"] + strict_df["excluded_AR"] > 0)
    total_strict_excluded = _s(strict_df["excluded_Ae"]) + _s(strict_df["excluded_AR"])
    print(f"  {len(strict_df)} stations written to cleaned_strict/")
    print(f"  {n_strict_affected} stations had rows excluded, {total_strict_excluded} days total -> sec7_cleaned_strict.csv")
    return strict_df, n_strict_affected, total_strict_excluded


def _run_excluded_basins() -> pd.DataFrame:
    """Section 8: plot observations for intentionally excluded basins."""
    print("=== Section 8: Excluded basin observation plots ===")
    fig_dir = STEP_7_OUTPUT_DIR / "figures" / "excluded_basins"
    fig_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for sid, reason in sorted(INTENTIONAL_EXCLUSIONS.items()):
        fp_clean = FLOW_CLEANED_DIR / f"{sid}_cleaned.csv"
        fp_raw   = RAW_USGS_DIR / f"{sid}.csv"

        # Try cleaned first, fall back to raw
        df = None
        source = None
        if fp_clean.exists():
            try:
                df = pd.read_csv(fp_clean, parse_dates=["date"])
                source = "cleaned"
            except Exception:
                pass

        if df is None and fp_raw.exists():
            try:
                rw = pd.read_csv(fp_raw, parse_dates=["datetime"])
                rw["date"] = _parse_raw_datetimes(rw["datetime"])
                flow_col = _detect_flow_col(rw.columns)
                if flow_col is not None:
                    rw = rw.rename(columns={flow_col: "flow"})
                    df = rw[["date", "flow"]].dropna(subset=["flow"])
                    source = "raw"
            except Exception:
                pass

        n_obs = len(df) if df is not None else 0
        rows.append({
            "station_id": sid,
            "reason":     reason,
            "source":     source or "none",
            "n_obs":      n_obs,
            "date_start": df["date"].min().date() if df is not None and n_obs else None,
            "date_end":   df["date"].max().date() if df is not None and n_obs else None,
        })

        if df is None or n_obs == 0:
            print(f"  {sid}: no data found — skipping plot")
            continue

        # --- Plot ---
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(df["date"], df["flow"], linewidth=0.5, color="steelblue", alpha=0.8)
        ax.set_title(f"{sid} — {reason} (n={n_obs:,}, source={source})", fontsize=11)
        ax.set_xlabel("Date")
        ax.set_ylabel("Flow (cfs)")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(fig_dir / f"{sid}_observations.png", dpi=150)
        plt.close(fig)
        print(f"  {sid}: plotted {n_obs:,} obs -> {sid}_observations.png")

    excl_df = pd.DataFrame(rows)
    excl_df.to_csv(STEP_7_OUTPUT_DIR / "sec8_excluded_basins.csv", index=False)
    print(f"  {len(excl_df)} excluded basins -> sec8_excluded_basins.csv")
    return excl_df


# ============================================================================
# Report generation
# ============================================================================

def _write_report(
    clim_df, flow_df, cross_df, audit_df, zero_wy_df,
    n_null_changed, total_null_moved, null_drop_df,
    strict_df, n_strict_affected, total_strict_excluded,
    excl_df,
) -> None:
    print("=== Writing qa_qc_report.txt ===")

    # Derived sets for the report
    clim_sids = set(clim_df["station_id"].astype(str))
    flow_sids = set(flow_df["station_id"].astype(str))
    in_both   = clim_sids & flow_sids
    clim_only = clim_sids - flow_sids
    flow_only = flow_sids - clim_sids
    n_wy_stations = zero_wy_df["station_id"].nunique() if not zero_wy_df.empty else 0

    lines = []
    L = lines.append

    L("=" * 70)
    L("QA/QC REPORT -- climate & cleaned flow data")
    L(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")
    L("=" * 70)
    L("")

    # -- Section 1: Climate
    L("\u2501" * 70)
    L("SECTION 1: CLIMATE")
    L("\u2501" * 70)
    L(f"  Station files         : {len(clim_df)}")
    vc = clim_df[clim_df["n_rows"] > 0] if "n_rows" in clim_df.columns else clim_df
    if not vc.empty and "date_start" in vc.columns:
        span = (pd.to_datetime(vc["date_end"]) - pd.to_datetime(vc["date_start"])).dt.days
        L(f"  Date range            : {vc['date_start'].min()} -- {vc['date_end'].max()}")
        L(f"  Median record length  : {span.median():.0f} days")
    L("")
    L("  GAPS / NULLS")
    for col, label in [
        ("gap_count", "Stations with >=1 gap"),
        ("null_precip_mm", "Stations with null precip_mm"),
        ("null_tmax_c", "Stations with null tmax_c"),
        ("null_tmin_c", "Stations with null tmin_c"),
    ]:
        if col in vc.columns:
            L(f"  {label:40s}: {_c(vc[col] > 0)}")
    if "total_missing_days" in vc.columns:
        L(f"  {'Total missing days':40s}: {_s(vc['total_missing_days']):,}")
    L("")

    # -- Section 2: Flow
    L("\u2501" * 70)
    L("SECTION 2: STREAMFLOW")
    L("\u2501" * 70)
    L(f"  Station files         : {len(flow_df)}")
    vf = flow_df[flow_df["n_rows"] > 0] if "n_rows" in flow_df.columns else flow_df
    if not vf.empty and "date_start" in vf.columns:
        span_f = (pd.to_datetime(vf["date_end"]) - pd.to_datetime(vf["date_start"])).dt.days
        L(f"  Date range            : {vf['date_start'].min()} -- {vf['date_end'].max()}")
        L(f"  Median record length  : {span_f.median():.0f} days")
    L("")
    for col, label in [
        ("null_flow", "Stations with null flow"),
        ("neg_flow_count", "Stations with negative flow"),
        ("zero_flow_count", "Stations with any zero flow"),
    ]:
        if col in vf.columns:
            L(f"  {label:40s}: {_c(vf[col] > 0)}")
    L("")

    # -- Section 3: Cross-dataset
    L("\u2501" * 70)
    L("SECTION 3: CROSS-DATASET ALIGNMENT")
    L("\u2501" * 70)
    L(f"  Stations in both datasets : {len(in_both)}")
    L(f"  Climate only (no flow)    : {len(clim_only)}")
    for sid in sorted(clim_only):
        L(f"    {sid}  -> {INTENTIONAL_EXCLUSIONS.get(str(sid), 'no flow file')}")
    L(f"  Flow only (no climate)    : {len(flow_only)}")
    for sid in sorted(flow_only):
        L(f"    {sid}")
    L("")

    # -- Section 4: Raw vs Cleaned
    L("\u2501" * 70)
    L("SECTION 4: RAW vs CLEANED RECONCILIATION")
    L("\u2501" * 70)
    has_err      = "error" in audit_df.columns
    va           = audit_df[audit_df["error"].isna()] if has_err else audit_df
    reconcilable = va
    if "raw_no_discharge_col" in va.columns:
        no_disc = va[va.get("raw_no_discharge_col", pd.Series(False, index=va.index)) == True]  # noqa: E712
        reconcilable = va[~va.index.isin(no_disc.index)]
    L(f"  Stations audited                 : {len(va)}")
    if "raw_file_exists" in va.columns:
        L(f"  Stations missing raw file        : {_c(va['raw_file_exists'] == False)}")  # noqa: E712
    if "cleaned_days" in reconcilable.columns:
        L(f"  Days retained (cleaned)          : {_s(reconcilable['cleaned_days']):>12,}")
    if "dropped_days" in reconcilable.columns:
        L(f"  Days dropped (outlier removal)   : {_s(reconcilable['dropped_days']):>12,}")
    L("")

    # -- Section 5: Zero WY
    L("\u2501" * 70)
    L("SECTION 5: ZERO WATER YEAR DETECTION")
    L("\u2501" * 70)
    L(f"  Stations with >=1 all-zero WY : {n_wy_stations}")
    L(f"  Total all-zero WYs flagged    : {len(zero_wy_df)}")
    L("")

    # -- Section 6: Null flow drops
    L("\u2501" * 70)
    L("SECTION 6: NULL FLOW DROPS APPLIED")
    L("\u2501" * 70)
    L(f"  Stations modified  : {n_null_changed}")
    L(f"  Total days moved   : {total_null_moved}")
    L("")

    # -- Section 7: Cleaned strict
    L("\u2501" * 70)
    L("SECTION 7: CLEANED_STRICT")
    L("\u2501" * 70)
    L(f"  Stations written                 : {len(strict_df)}")
    L(f"  Stations with excluded rows      : {n_strict_affected}")
    L(f"  Total excluded (A,e + A,R) days  : {total_strict_excluded:,}")
    L("")

    # -- Section 8: Excluded basins
    L("\u2501" * 70)
    L("SECTION 8: EXCLUDED BASINS")
    L("\u2501" * 70)
    L(f"  Basins excluded       : {len(excl_df)}")
    for _, row in excl_df.iterrows():
        obs_info = f"n={row['n_obs']:,}" if row["n_obs"] else "no data"
        L(f"    {row['station_id']:12s}  {row['reason']:40s}  ({obs_info})")
    L(f"  Observation plots     : {STEP_7_OUTPUT_DIR / 'figures' / 'excluded_basins'}/")
    L("")

    # -- Output file index
    L("-" * 70)
    L("OUTPUT FILES")
    L("  sec1_climate_qa.csv      - per-station climate quality metrics")
    L("  sec2_flow_qa.csv         - per-station flow quality metrics")
    L("  sec3_cross_dataset.csv   - overlap / alignment between datasets")
    L("  sec4_raw_vs_cleaned.csv  - raw vs cleaned reconciliation & gap classification")
    L("  sec5_zero_wy_audit.csv   - all-zero water years detected in raw data")
    L("  sec6_null_flow_drops.csv - stations where null-flow rows were moved to dropped")
    L("  sec7_cleaned_strict.csv  - per-station summary of A,e / A,R exclusions")
    L("  sec8_excluded_basins.csv - excluded basins with reasons and obs counts")
    L("  figures/excluded_basins/ - observation time-series plots for excluded basins")
    L("  qa_qc_report.txt         - this report")
    L("-" * 70)

    report_text = "\n".join(lines)
    with open(STEP_7_OUTPUT_DIR / "qa_qc_report.txt", "w", encoding="utf-8") as f:
        f.write(report_text)

    print(report_text)
    print()
    print("All outputs written to:", STEP_7_OUTPUT_DIR)


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    STEP_7_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    clim_df  = _run_climate_qa()
    flow_df  = _run_flow_qa()
    cross_df = _run_cross_dataset(clim_df, flow_df)
    audit_df = _run_raw_vs_cleaned()
    zero_wy_df = _run_zero_wy_detection()
    n_null_changed, total_null_moved, null_drop_df = _run_null_flow_drops()
    strict_df, n_strict_affected, total_strict_excluded = _run_cleaned_strict()
    excl_df = _run_excluded_basins()

    _write_report(
        clim_df, flow_df, cross_df, audit_df, zero_wy_df,
        n_null_changed, total_null_moved, null_drop_df,
        strict_df, n_strict_affected, total_strict_excluded,
        excl_df,
    )


if __name__ == "__main__":
    main()
