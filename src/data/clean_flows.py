"""Clean raw USGS flow files using exceedance-based precip filtering.

For each site:
  1. Merge raw USGS flow with area-weighted climate
  2. Iteratively drop extreme-flow days where 3-day precip is implausibly low
     (criteria from original USGS_qaqc.ipynb)
  3. Save cleaned CSV to data/intermediate/flow_cleaned/<site>_cleaned.csv
  4. Save dropped rows to data/intermediate/flow_dropped/<site>_dropped.csv
  5. Save one QA figure per site to data/prepare/figures/<site>_exceedance.png

The figure shows the top-15 exceedance points with:
  - Green markers = retained
  - Red markers   = filtered out
  - Stacked bars  = same-day precip (orange) and 3-day total (blue)
  - Date labels   = secondary x-axis
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data.io import load_climate_dataframes
from src.paths import (
    CLIMATE_STATS_OUTPUT,
    CLIMATE_WATERSHEDS_ZARR,
    FLOW_CLEANED_DIR,
    FLOW_DROPPED_DIR,
    STEP_6_OUTPUT_DIR,
    RAW_USGS_DIR,
)

FIGURES_DIR = STEP_6_OUTPUT_DIR / "figures"

# Discharge column fallback order (mirrors notebook)
FLOW_COL_ORDER = ["00060_Mean", "00060_2_Mean", "00054_Observation at 24:00"]
TOP_N = 15  # number of extreme events to inspect

# Snowmelt exemption — avoid filtering legitimate melt-driven peaks
SNOW_FRAC_THRESH = 0.1       # basin snow_fraction above which exemption applies
MELT_MONTHS = {3, 4, 5, 6, 7}  # Mar–Jul
WINTER_PRECIP_WINDOW = 280    # rolling lookback (days) for winter accumulation
WINTER_PRECIP_MIN = 100       # mm cumulative to trigger exemption


def _detect_flow_col(cols) -> str | None:
    for c in FLOW_COL_ORDER:
        if c in cols:
            return c
    candidates = [c for c in cols if "60" in c and c.endswith("_Mean")]
    return candidates[0] if candidates else None


def _build_exc_df(df: pd.DataFrame) -> pd.DataFrame:
    """Add exceedance rank column, sorted ascending by flow."""
    out = df.copy().sort_values("flow", ascending=True)
    T = len(out)
    out["x_exc"] = np.arange(1, T + 1) * 100.0 / (T + 1)
    return out


def _filter_site(
    df: pd.DataFrame,
    exempt: pd.Index | None = None,
) -> tuple[pd.DataFrame, list]:
    """Iteratively drop extreme events with implausibly low 3-day precip.

    Dates in *exempt* (snowmelt-season dates with sufficient winter precip)
    are never dropped, even if they trip the precipitation criteria.

    Returns (filtered_df, dropped_dates).
    """
    exc_mast = _build_exc_df(df)
    drop_dates: list = []
    _exempt = exempt if exempt is not None else pd.Index([])

    t = True
    while t:
        top = exc_mast.iloc[-TOP_N:]

        # Criteria 1: 3-day precip < 1 mm
        bad1 = top[top["pr_3day"] < 1].index.tolist()
        bad1 = [d for d in bad1 if d not in _exempt]
        if bad1:
            exc_mast = exc_mast.drop(bad1)
            drop_dates.extend(bad1)
            continue

        # Criteria 2: 3-day precip < 10% of top-N mean AND outside 7-day flow window
        top = exc_mast.iloc[-TOP_N:]
        thresh = np.nanmean(top["pr_3day"]) * 0.1
        bad2_cond = (top["pr_3day"] < thresh) & (top["flow_7day_fit"] == False)  # noqa: E712
        bad2 = top[bad2_cond].index.tolist()
        bad2 = [d for d in bad2 if d not in _exempt]
        if bad2:
            exc_mast = exc_mast.drop(bad2)
            drop_dates.extend(bad2)
            # Re-check criteria 1 after criteria 2 drops
            continue

        t = False

    filtered_df = df.drop(index=drop_dates, errors="ignore")
    return filtered_df, drop_dates


def _plot_site(
    df_orig: pd.DataFrame,
    exc_full: pd.DataFrame,
    drop_dates: list,
    site: str,
    out_path: Path,
) -> None:
    """Two-panel figure: flow time series / combined exceedance+precip bars."""
    top = exc_full.iloc[-TOP_N:].copy()
    kept    = top[~top.index.isin(drop_dates)]
    dropped = top[top.index.isin(drop_dates)]

    tick_locs = top["x_exc"].values
    width = np.min(np.diff(tick_locs)) * 0.8 if len(tick_locs) > 1 else 0.05

    fig, axes = plt.subplots(2, 1, figsize=(9, 8))

    # ── Panel 1: Flow time series + precipitation ──────────────────────────
    ax0 = axes[0]
    ax0.plot(df_orig.index, df_orig["flow"], label="Flow", color="blue", alpha=0.7)
    ax0.set_ylabel("Flow (cfs)")
    ax0.set_title(f"Site {site} — Flow Time Series")

    ax0r = ax0.twinx()
    ax0r.plot(df_orig.index, df_orig["precip_mm"], label="Precipitation",
              color="r", alpha=0.3)
    ax0r.set_ylabel("Precipitation (mm)")

    h0, l0 = ax0.get_legend_handles_labels()
    h0r, l0r = ax0r.get_legend_handles_labels()
    ax0.legend(h0 + h0r, l0 + l0r, loc="upper left", fontsize=8)

    # ── Panel 2: Precip bars (background) + exceedance scatter (foreground) ─
    ax1 = axes[1]
    ax1.bar(top["x_exc"], top["pr_3day"],   width=width, label="pr_3day",
            color="#5B9BD5", alpha=1.0, zorder=2)
    ax1.bar(top["x_exc"], top["precip_mm"], width=width, label="precip_mm",
            color="#FF8C00", alpha=1.0, zorder=3)
    ax1.set_ylabel("Precipitation (mm)")
    ax1.grid(True, linestyle="--", alpha=0.5)

    ax1r = ax1.twinx()
    ax1r.plot(top["x_exc"], top["flow"], color="green", linewidth=0.8,
              alpha=0.4, zorder=4)
    if len(kept):
        ax1r.scatter(kept["x_exc"], kept["flow"], color="green", zorder=5,
                     s=60, label="kept", marker="o")
    if len(dropped):
        ax1r.scatter(dropped["x_exc"], dropped["flow"], color="red", zorder=6,
                     s=80, label="filtered out", marker="X")
    ax1r.set_ylabel("Flow (cfs)")

    # Combined legend
    h1, l1 = ax1.get_legend_handles_labels()
    h1r, l1r = ax1r.get_legend_handles_labels()
    ax1.legend(h1 + h1r, l1 + l1r, loc="upper left", fontsize=8)

    # Date labels on secondary x-axis below panel 2
    ax1b = ax1.twiny()
    ax1b.plot(top["x_exc"], top["pr_3day"], color="none")
    ax1b.set_xticks(top["x_exc"])
    ax1b.xaxis.set_ticks_position("bottom")
    ax1b.xaxis.set_label_position("bottom")
    ax1b.spines["bottom"].set_position(("axes", -0.38))
    ax1b.spines["bottom"].set_visible(True)
    date_labels = top.index.strftime("%Y-%m-%d").tolist()
    ax1b.set_xticklabels(date_labels, rotation=90, ha="right", fontsize=7)
    ax1b.set_xlabel("Date")
    ax1b.set_xlim(ax1.get_xlim())

    plt.subplots_adjust(hspace=0.35)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close("all")


def process_site(
    site: str,
    snow_fraction: float = 0.0,
    climate_df: pd.DataFrame | None = None,
) -> dict:
    """Process a single site. Returns dict with status, kept, dropped.

    ``climate_df`` is the per-basin climate DataFrame indexed by date with
    columns ``precip_mm``, ``tmax_c``, ``tmin_c`` (loaded once from zarr by
    the caller).
    """
    flow_path    = RAW_USGS_DIR / f"{site}.csv"
    out_clean    = FLOW_CLEANED_DIR / f"{site}_cleaned.csv"
    out_dropped  = FLOW_DROPPED_DIR / f"{site}_dropped.csv"
    out_fig      = FIGURES_DIR / f"{site}_filter.png"

    if not flow_path.exists():
        return {"status": f"[SKIP] no raw flow: {flow_path.name}"}
    if climate_df is None or climate_df.empty:
        return {"status": f"[SKIP] no climate for {site}"}

    # ── Load & merge ───────────────────────────────────────────────────
    flow_df = pd.read_csv(flow_path)
    flow_df["date"] = pd.to_datetime(flow_df["datetime"].str.split(" ", expand=True)[0])

    climate_df = (
        climate_df[["precip_mm", "tmax_c", "tmin_c"]].dropna(how="all").reset_index()
    )

    merged = (pd.merge(flow_df, climate_df, on="date", how="inner")
                .sort_values("date").reset_index(drop=True)
                .set_index("date", drop=False))

    flow_col = _detect_flow_col(merged.columns)
    if flow_col is None:
        return {"status": "[SKIP] no discharge column"}

    cd_col = f"{flow_col}_cd" if f"{flow_col}_cd" in merged.columns else None
    cols = [flow_col] + ([cd_col] if cd_col else []) + ["precip_mm", "tmax_c", "tmin_c"]
    merged = merged[cols]
    new_cols = ["flow"] + (["flow_cd"] if cd_col else []) + ["precip_mm", "tmax_c", "tmin_c"]
    merged.columns = new_cols
    if "flow_cd" not in merged.columns:
        merged["flow_cd"] = pd.NA

    # Remove negative flows
    merged.loc[merged["flow"] < 0, "flow"] = np.nan

    # 3-day backward precip sum
    merged["pr_3day"] = merged["precip_mm"].rolling(window=3, min_periods=1).sum()

    # 7-day backward flow average and fit flag
    merged["flow_7day"] = merged["flow"].rolling(window=7, min_periods=1).mean()
    merged["flow_7day_fit"] = (
        (merged["flow"] >= merged["flow_7day"] * 0.25) &
        (merged["flow"] <= merged["flow_7day"] * 1.75)
    )

    orig = merged.copy()

    # ── Snowmelt exemption ─────────────────────────────────────────────────
    exempt: pd.Index | None = None
    if snow_fraction > SNOW_FRAC_THRESH:
        winter_accum = merged["precip_mm"].rolling(
            window=WINTER_PRECIP_WINDOW, min_periods=1,
        ).sum()
        is_melt = merged.index.month.isin(MELT_MONTHS)
        exempt = merged.index[is_melt & (winter_accum >= WINTER_PRECIP_MIN)]

    # ── Filter ─────────────────────────────────────────────────────────────
    exc_full = _build_exc_df(merged)
    _, drop_dates = _filter_site(merged, exempt=exempt)

    # ── Plot ───────────────────────────────────────────────────────────────
    _plot_site(orig, exc_full, drop_dates, site, out_fig)

    cleaned = orig.drop(index=drop_dates, errors="ignore")

    # ── Save outputs ───────────────────────────────────────────────────────
    dropped = orig.loc[orig.index.isin(drop_dates)]

    cleaned.to_csv(out_clean, index=True)
    dropped.to_csv(out_dropped, index=True)

    return {
        "status": f"[OK] kept={len(cleaned)}, dropped={len(dropped)}",
        "kept": len(cleaned),
        "dropped": len(dropped),
    }


def main() -> None:
    FLOW_CLEANED_DIR.mkdir(parents=True, exist_ok=True)
    FLOW_DROPPED_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Load snow_fraction per basin for snowmelt exemption
    snow_map: dict[str, float] = {}
    if CLIMATE_STATS_OUTPUT.exists():
        clim_df = pd.read_csv(CLIMATE_STATS_OUTPUT)
        snow_map = dict(
            zip(clim_df["PourPtID"].astype(str), clim_df["snow_fraction"])
        )

    sites = sorted(p.stem for p in RAW_USGS_DIR.glob("*.csv"))
    print(f"Found {len(sites)} sites.")

    print(f"Loading climate from {CLIMATE_WATERSHEDS_ZARR.name} ...")
    climate_dfs_raw = load_climate_dataframes(CLIMATE_WATERSHEDS_ZARR)
    climate_dfs = {str(b): df for b, df in climate_dfs_raw.items()}
    print(f"  loaded {len(climate_dfs)} basins")

    metrics = []
    for i, site in enumerate(sites, 1):
        sf = snow_map.get(site, 0.0)
        print(f"[{i}/{len(sites)}] {site} (snow_frac={sf:.2f}) ... ", end="", flush=True)
        result = process_site(site, snow_fraction=sf, climate_df=climate_dfs.get(site))
        # detect which flow column was used for site_metrics.csv
        flow_path = RAW_USGS_DIR / f"{site}.csv"
        flow_col = None
        if flow_path.exists():
            try:
                cols = pd.read_csv(flow_path, nrows=0).columns.tolist()
                flow_col = _detect_flow_col(cols) or "unknown"
            except Exception:
                pass
        metrics.append({
            "site": site,
            "metric": flow_col or "unknown",
            "kept": result.get("kept", ""),
            "dropped": result.get("dropped", ""),
        })
        print(result["status"])

    metrics_df = pd.DataFrame(metrics)
    metrics_path = STEP_6_OUTPUT_DIR / "site_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"\nDone. site_metrics -> {metrics_path}")


if __name__ == "__main__":
    main()
