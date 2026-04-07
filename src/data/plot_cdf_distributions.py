"""
CDF distribution plots for key watershed variables, grouped by R² tier.

Generates figures:
  1) Summary 2×3: average CDF per tier
  2) Per-tier 2×2 (×3): individual watershed CDFs
  3) Annual Q/P ratio 1×3
  4) Static attributes CDFs
  5) Monthly averages 2×2
  6) Observed streamflow days CDF
  7) Per-tier monthly averages (×3)

Outputs go to data/prepare/tier_characteristics/.
"""
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

from src.paths import (
    CLIMATE_DIR,
    FLOW_DIR,
    BASIN_ATLAS_OUTPUT,
    CLIMATE_STATS_OUTPUT,
    QAQC_FLOW_PRECIP_CSV,
    TIER_CHARACTERISTICS_DIR,
)

# CFS → mm/day:  cfs * 2.4466 / area_km2
CFS_TO_MM_FACTOR = 0.0283168 * 86400 / 1e6 * 1000  # ≈ 2.4466

PRECIP_THRESH = 0.1

TIER_COLORS = {
    "tier_1": "#196cbe",
    "tier_2": "#4bdd63",
    "tier_3": "#d1503c",
}
TIER_LABELS = {
    "tier_1": "Tier 1 — R² > 0.60",
    "tier_2": "Tier 2 — 0.20 ≤ R² ≤ 0.60",
    "tier_3": "Tier 3 — R² < 0.20",
}
TIER_ORDER = ["tier_1", "tier_2", "tier_3"]

STATIC_ATTRS = [
    ("total_Shape_Area_km2", "Watershed Area (km²)",     True),
    ("ele_mt_uav",           "Mean Elevation (m)",        False),
    ("slp_dg_uav",           "Mean Slope (°×10)",         False),
    ("for_pc_use",           "Forest Cover (%)",          False),
    ("cly_pc_uav",           "Clay Fraction (%)",         False),
    ("slt_pc_uav",           "Silt Fraction (%)",         False),
    ("snd_pc_uav",           "Sand Fraction (%)",         False),
    ("precip_mean",          "Mean Daily Precip (mm)",    False),
    ("pet_mean",             "Mean Daily PET (mm)",       False),
    ("aridity_index",        "Aridity Index (PET/P)",     False),
    ("snow_fraction",        "Snow Fraction",             False),
    ("high_precip_freq",     "High Precip Frequency",     False),
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def average_cdf(sorted_arrays, n_points=500):
    if len(sorted_arrays) == 0:
        return np.array([]), np.array([])
    all_vals = np.concatenate(sorted_arrays)
    x_grid = np.linspace(np.nanmin(all_vals), np.nanmax(all_vals), n_points)
    cdf_stack = np.zeros((len(sorted_arrays), n_points))
    for i, vals in enumerate(sorted_arrays):
        cdf_y = np.arange(1, len(vals) + 1) / len(vals)
        cdf_stack[i] = np.interp(x_grid, vals, cdf_y, left=0.0, right=1.0)
    return x_grid, cdf_stack.mean(axis=0)


def average_cdf_log(sorted_arrays, n_points=500):
    if len(sorted_arrays) == 0:
        return np.array([]), np.array([])
    all_vals = np.concatenate(sorted_arrays)
    lo = np.nanmin(all_vals[all_vals > 0]) if np.any(all_vals > 0) else 1e-6
    hi = np.nanmax(all_vals)
    if lo >= hi:
        lo, hi = hi * 0.1, hi * 1.1
    x_grid = np.geomspace(lo, hi, n_points)
    cdf_stack = np.zeros((len(sorted_arrays), n_points))
    for i, vals in enumerate(sorted_arrays):
        cdf_y = np.arange(1, len(vals) + 1) / len(vals)
        cdf_stack[i] = np.interp(x_grid, vals, cdf_y, left=0.0, right=1.0)
    return x_grid, cdf_stack.mean(axis=0)


def add_tier_legend(fig):
    handles = [
        mlines.Line2D([], [], color=TIER_COLORS[t], linewidth=2, label=TIER_LABELS[t])
        for t in TIER_ORDER
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=10,
               frameon=True, fancybox=True, bbox_to_anchor=(0.5, 0.01))


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    TIER_CHARACTERISTICS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load QAQC for area lookup ─────────────────────────────────────────────
    print("Loading QAQC data (for area lookup) …")
    area_by_pid: dict[str, float] = {}
    with open(QAQC_FLOW_PRECIP_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            area_by_pid[row["PourPtID"].strip()] = float(row["area_km2"])
    print(f"  {len(area_by_pid)} watersheds with area data")

    # ── Per-watershed CDF data ────────────────────────────────────────────────
    KEYS = ["flow_mm", "tmax", "tmin", "precip_nz", "annual_qp", "nonzero_precip_frac"]
    cdf_by_tier = {t: {k: [] for k in KEYS} for t in TIER_ORDER}
    n_flow_days_by_tier = {t: [] for t in TIER_ORDER}

    MONTHLY_VARS = ["flow_mm", "tmax", "tmin", "precip"]
    monthly_by_tier = {t: {v: [] for v in MONTHLY_VARS} for t in TIER_ORDER}

    print("Processing watersheds …")
    n_processed = 0
    for tier in TIER_ORDER:
        tier_flow_dir = FLOW_DIR / tier
        if not tier_flow_dir.exists():
            print(f"  WARNING: {tier_flow_dir} not found, skipping")
            continue

        for flow_file in sorted(tier_flow_dir.glob("*_cleaned.csv")):
            pid = flow_file.stem.replace("_cleaned", "")
            area = area_by_pid.get(pid)
            if area is None:
                continue

            clim_file = CLIMATE_DIR / f"climate_{pid}.csv"
            if not clim_file.exists():
                continue

            clim    = pd.read_csv(clim_file, parse_dates=["date"])
            flow_df = pd.read_csv(flow_file, parse_dates=["date"])
            n_processed += 1

            flow_mm = flow_df["flow"].values * CFS_TO_MM_FACTOR / area
            cdf_by_tier[tier]["flow_mm"].append(np.sort(flow_mm))
            n_flow_days_by_tier[tier].append(len(flow_df))

            cdf_by_tier[tier]["tmax"].append(np.sort(clim["tmax_c"].dropna().values))
            cdf_by_tier[tier]["tmin"].append(np.sort(clim["tmin_c"].dropna().values))

            nz = clim.loc[clim["precip_mm"] > PRECIP_THRESH, "precip_mm"].values
            if len(nz) > 0:
                cdf_by_tier[tier]["precip_nz"].append(np.sort(nz))
            cdf_by_tier[tier]["nonzero_precip_frac"].append(
                (clim["precip_mm"] > PRECIP_THRESH).mean())

            clim["month"] = clim["date"].dt.month
            monthly_by_tier[tier]["tmax"].append(
                clim.groupby("month")["tmax_c"].mean().reindex(range(1, 13)).values)
            monthly_by_tier[tier]["tmin"].append(
                clim.groupby("month")["tmin_c"].mean().reindex(range(1, 13)).values)
            monthly_by_tier[tier]["precip"].append(
                clim.groupby("month")["precip_mm"].mean().reindex(range(1, 13)).values)
            flow_df["month"]         = flow_df["date"].dt.month
            flow_df["flow_mm_daily"] = flow_df["flow"] * CFS_TO_MM_FACTOR / area
            monthly_by_tier[tier]["flow_mm"].append(
                flow_df.groupby("month")["flow_mm_daily"].mean().reindex(range(1, 13)).values)

            flow_df["flow_mm"] = flow_mm
            merged = flow_df[["date", "flow_mm"]].merge(
                clim[["date", "precip_mm"]], on="date", how="inner")
            merged["water_year"] = merged["date"].dt.year
            merged.loc[merged["date"].dt.month >= 10, "water_year"] += 1
            wy = merged.groupby("water_year").agg(
                total_q=("flow_mm", "sum"), total_p=("precip_mm", "sum"))
            wy = wy[wy["total_p"] > 10]
            if len(wy) > 0:
                cdf_by_tier[tier]["annual_qp"].append(
                    np.sort((wy["total_q"] / wy["total_p"]).values))

            if n_processed % 50 == 0:
                print(f"  {n_processed} watersheds processed …")

    print(f"  Done — {n_processed} total watersheds")

    # ── Static attributes ─────────────────────────────────────────────────────
    print("Loading static attributes …")
    pid_to_tier: dict[str, str] = {}
    for tier in TIER_ORDER:
        tier_dir = FLOW_DIR / tier
        if tier_dir.exists():
            for f in tier_dir.glob("*_cleaned.csv"):
                pid_to_tier[f.stem.replace("_cleaned", "")] = tier

    basin_df      = pd.read_csv(BASIN_ATLAS_OUTPUT)
    basin_df["PourPtID"] = basin_df["PourPtID"].astype(str).str.strip()
    clim_stats_df = pd.read_csv(CLIMATE_STATS_OUTPUT)
    clim_stats_df["PourPtID"] = clim_stats_df["PourPtID"].astype(str).str.strip()
    static_df = basin_df.merge(clim_stats_df, on="PourPtID", how="outer")
    static_df["tier"] = static_df["PourPtID"].map(pid_to_tier)
    static_df = static_df.dropna(subset=["tier"])

    # ── Common plot config ────────────────────────────────────────────────────
    month_labels    = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                       "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    months          = np.arange(1, 13)
    monthly_panels  = [
        ("flow_mm", "Streamflow (mm/day)"),
        ("tmax",    "Tmax (°C)"),
        ("tmin",    "Tmin (°C)"),
        ("precip",  "Precipitation (mm/day)"),
    ]

    # ── FIGURE 1 — Summary average CDFs ──────────────────────────────────────
    print("Plotting Figure 1 — summary averages …")
    fig1, axes1 = plt.subplots(2, 3, figsize=(16, 9))
    fig1.suptitle("Average CDF by Tier — Summary", fontsize=14, fontweight="bold", y=0.97)
    summary_panels = [
        ("flow_mm",             "Daily Streamflow (mm)",                          True,  "array"),
        ("tmax",                "Daily Tmax (°C)",                                 False, "array"),
        ("tmin",                "Daily Tmin (°C)",                                 False, "array"),
        ("precip_nz",           f"Non-Zero Precip (>{PRECIP_THRESH} mm)",          True,  "array"),
        ("nonzero_precip_frac", f"Non-Zero Precip Day Fraction (>{PRECIP_THRESH} mm)", False, "scalar"),
        ("annual_qp",           "Annual Q/P Ratio",                                True,  "array"),
    ]
    for ax, (key, title, log_x, mode) in zip(axes1.flat, summary_panels):
        for tier in TIER_ORDER:
            arrays = cdf_by_tier[tier][key]
            if not arrays:
                continue
            if mode == "scalar":
                vals  = np.sort(arrays)
                cdf_y = np.arange(1, len(vals) + 1) / len(vals)
                ax.step(vals, cdf_y, color=TIER_COLORS[tier], linewidth=2, alpha=0.9)
            else:
                xg, yg = (average_cdf_log if log_x else average_cdf)(arrays)
                ax.plot(xg, yg, color=TIER_COLORS[tier], linewidth=2, alpha=0.9)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_ylabel("CDF")
        ax.set_ylim(0, 1.02)
        ax.grid(True, alpha=0.3)
        if log_x:
            ax.set_xscale("log")
    add_tier_legend(fig1)
    fig1.tight_layout(rect=[0, 0.05, 1, 0.95])
    fig1.savefig(TIER_CHARACTERISTICS_DIR / "cdf_summary.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig1)
    print("  Saved → cdf_summary.pdf")

    # ── FIGURE 2 — Per-tier individual CDFs ──────────────────────────────────
    print("Plotting Figure 2 — per-tier individual CDFs …")
    detail_panels = [
        ("flow_mm",   "Daily Streamflow (mm)",                   True),
        ("tmax",      "Daily Tmax (°C)",                          False),
        ("tmin",      "Daily Tmin (°C)",                          False),
        ("precip_nz", f"Non-Zero Precip (>{PRECIP_THRESH} mm)",  True),
    ]
    for tier in TIER_ORDER:
        fig, axes = plt.subplots(2, 2, figsize=(12, 9))
        n_ws = len(cdf_by_tier[tier]["flow_mm"])
        fig.suptitle(f"{TIER_LABELS[tier]}  ({n_ws} watersheds)",
                     fontsize=14, fontweight="bold", y=0.97)
        for ax, (key, title, log_x) in zip(axes.flat, detail_panels):
            for vals in cdf_by_tier[tier][key]:
                if len(vals) == 0:
                    continue
                cdf_y = np.arange(1, len(vals) + 1) / len(vals)
                ax.plot(vals, cdf_y, color=TIER_COLORS[tier], alpha=0.4, linewidth=0.7)
            ax.set_title(title, fontsize=11, fontweight="bold")
            ax.set_ylabel("CDF")
            ax.set_ylim(0, 1.02)
            ax.grid(True, alpha=0.3)
            if log_x:
                ax.set_xscale("log")
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        fname = f"cdf_detail_{tier}.pdf"
        fig.savefig(TIER_CHARACTERISTICS_DIR / fname, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved → {fname}")

    # ── FIGURE 3 — Annual Q/P ratio 1×3 ──────────────────────────────────────
    print("Plotting Figure 3 — annual Q/P ratio …")
    fig3, axes3 = plt.subplots(1, 3, figsize=(16, 5))
    fig3.suptitle("Annual Q/P Ratio CDFs by Tier", fontsize=14, fontweight="bold", y=1.0)
    for ax, tier in zip(axes3, TIER_ORDER):
        for vals in cdf_by_tier[tier]["annual_qp"]:
            if len(vals) == 0:
                continue
            cdf_y = np.arange(1, len(vals) + 1) / len(vals)
            ax.plot(vals, cdf_y, color=TIER_COLORS[tier], alpha=0.4, linewidth=0.7)
        n_ws = len(cdf_by_tier[tier]["annual_qp"])
        ax.set_title(f"{TIER_LABELS[tier]}  ({n_ws} ws)", fontsize=10, fontweight="bold")
        ax.set_ylabel("CDF")
        ax.set_ylim(0, 1.02)
        ax.set_xscale("log")
        ax.set_xlabel("Annual Q/P Ratio")
        ax.grid(True, alpha=0.3)
    fig3.tight_layout()
    fig3.savefig(TIER_CHARACTERISTICS_DIR / "cdf_annual_qp_ratio.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig3)
    print("  Saved → cdf_annual_qp_ratio.pdf")

    # ── FIGURE 4 — Static attributes by tier ─────────────────────────────────
    print("Plotting Figure 4 — static attributes …")
    n_attrs = len(STATIC_ATTRS)
    ncols   = 3
    nrows   = int(np.ceil(n_attrs / ncols))
    fig4, axes4 = plt.subplots(nrows, ncols, figsize=(16, 3.5 * nrows))
    fig4.suptitle("Static Attribute CDFs by Tier", fontsize=14, fontweight="bold", y=0.98)
    for ax, (col, title, log_x) in zip(axes4.flat, STATIC_ATTRS):
        for tier in TIER_ORDER:
            subset = static_df.loc[static_df["tier"] == tier, col].dropna().values
            if len(subset) == 0:
                continue
            vals  = np.sort(subset)
            cdf_y = np.arange(1, len(vals) + 1) / len(vals)
            ax.step(vals, cdf_y, color=TIER_COLORS[tier], linewidth=1.5, alpha=0.9,
                    label=TIER_LABELS[tier])
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_ylabel("CDF")
        ax.set_ylim(0, 1.02)
        ax.grid(True, alpha=0.3)
        if log_x:
            ax.set_xscale("log")
    for ax in axes4.flat[n_attrs:]:
        ax.set_visible(False)
    add_tier_legend(fig4)
    fig4.tight_layout(rect=[0, 0.04, 1, 0.96])
    fig4.savefig(TIER_CHARACTERISTICS_DIR / "cdf_static_attributes.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig4)
    print("  Saved → cdf_static_attributes.pdf")

    # ── FIGURE 5 — Monthly averages 2×2 ──────────────────────────────────────
    print("Plotting Figure 5 — monthly averages …")
    fig5, axes5 = plt.subplots(2, 2, figsize=(12, 9))
    fig5.suptitle("Monthly Average by Tier", fontsize=14, fontweight="bold", y=0.97)
    for ax, (var, title) in zip(axes5.flat, monthly_panels):
        for tier in TIER_ORDER:
            arr_list = monthly_by_tier[tier][var]
            if not arr_list:
                continue
            stack     = np.array(arr_list)
            mean_vals = np.nanmean(stack, axis=0)
            ax.plot(months, mean_vals, color=TIER_COLORS[tier], linewidth=2,
                    alpha=0.9, marker="o", markersize=4, label=TIER_LABELS[tier])
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xticks(months)
        ax.set_xticklabels(month_labels, fontsize=8)
        ax.set_ylabel(title.split("(")[-1].rstrip(")") if "(" in title else title)
        ax.grid(True, alpha=0.3)
    add_tier_legend(fig5)
    fig5.tight_layout(rect=[0, 0.05, 1, 0.95])
    fig5.savefig(TIER_CHARACTERISTICS_DIR / "monthly_averages.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig5)
    print("  Saved → monthly_averages.pdf")

    # ── FIGURE 6 — Observed streamflow days ──────────────────────────────────
    print("Plotting Figure 6 — observed streamflow days …")
    fig6, ax6 = plt.subplots(figsize=(8, 5))
    ax6.set_title("Number of Observed Streamflow Days", fontsize=13, fontweight="bold")
    for tier in TIER_ORDER:
        vals = np.sort(n_flow_days_by_tier[tier])
        if len(vals) == 0:
            continue
        cdf_y = np.arange(1, len(vals) + 1) / len(vals)
        ax6.step(vals, cdf_y, color=TIER_COLORS[tier], linewidth=2, alpha=0.9,
                 label=TIER_LABELS[tier])
    ax6.set_xlabel("Number of Days")
    ax6.set_ylabel("CDF")
    ax6.set_ylim(0, 1.02)
    ax6.grid(True, alpha=0.3)
    ax6.legend(fontsize=10)
    fig6.tight_layout()
    fig6.savefig(TIER_CHARACTERISTICS_DIR / "cdf_observed_flow_days.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig6)
    print("  Saved → cdf_observed_flow_days.pdf")

    # ── FIGURE 7 — Per-tier monthly averages (individual watersheds) ──────────
    print("Plotting Figure 7 — per-tier monthly averages …")
    for tier in TIER_ORDER:
        fig, axes = plt.subplots(2, 2, figsize=(12, 9))
        n_ws = len(monthly_by_tier[tier]["flow_mm"])
        fig.suptitle(f"{TIER_LABELS[tier]}  ({n_ws} watersheds) — Monthly Averages",
                     fontsize=14, fontweight="bold", y=0.97)
        for ax, (var, title) in zip(axes.flat, monthly_panels):
            arr_list = monthly_by_tier[tier][var]
            for ws_vals in arr_list:
                ax.plot(months, ws_vals, color=TIER_COLORS[tier], alpha=0.3, linewidth=0.7)
            if arr_list:
                stack     = np.array(arr_list)
                mean_vals = np.nanmean(stack, axis=0)
                ax.plot(months, mean_vals, color="black", linewidth=2.5,
                        alpha=0.9, marker="o", markersize=4, label="Tier Mean")
            ax.set_title(title, fontsize=11, fontweight="bold")
            ax.set_xticks(months)
            ax.set_xticklabels(month_labels, fontsize=8)
            ax.set_ylabel(title.split("(")[-1].rstrip(")") if "(" in title else title)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8, loc="upper right")
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        fname = f"monthly_detail_{tier}.pdf"
        fig.savefig(TIER_CHARACTERISTICS_DIR / fname, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved → {fname}")

    print("Done.")


if __name__ == "__main__":
    main()
