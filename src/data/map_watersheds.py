"""
Professional academic-style map of California USGS watersheds,
colored by regression R² tier from the QAQC analysis.

Outputs go to data/prepare/map_watersheds/.
"""
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import csv
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from pathlib import Path

from src.data.paths import (
    WATERSHED_GEOJSON,
    QAQC_FLOW_PRECIP_CSV,
    MAP_WATERSHEDS_DIR,
)

# ── Tier colors & labels ──────────────────────────────────────────────────────
TIER_COLORS = {
    "tier_1":        "#196cbe",
    "tier_2":        "#4bdd63",
    "tier_3":        "#d1503c",
    "unclassified":  "#aaaaaa",
}
TIER_LABELS = {
    "tier_1":        "Tier 1 — R² > 0.60",
    "tier_2":        "Tier 2 — 0.20 ≤ R² ≤ 0.60",
    "tier_3":        "Tier 3 — R² < 0.20",
    "unclassified":  "Unclassified (insufficient data)",
}


def main() -> None:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

    CA_ALBERS = ccrs.AlbersEqualArea(
        central_longitude=-120.0,
        central_latitude=37.5,
        standard_parallels=(34.0, 40.5),
    )
    GEODETIC = ccrs.Geodetic()

    MAP_WATERSHEDS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load QAQC R² results ──────────────────────────────────────────────────
    print("Loading QAQC tier data …")
    r2_by_id: dict[str, dict] = {}
    with open(QAQC_FLOW_PRECIP_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            pid = row["PourPtID"].strip()
            r2_str = row.get("monthly_regression_r2", "NaN").strip()
            try:
                r2 = float(r2_str)
            except ValueError:
                r2 = float("nan")
            if r2 != r2:  # NaN
                tier = "unclassified"
            elif r2 > 0.6:
                tier = "tier_1"
            elif r2 >= 0.2:
                tier = "tier_2"
            else:
                tier = "tier_3"
            r2_by_id[pid] = {"r2": r2, "tier": tier}

    # ── Load watershed GeoJSON ────────────────────────────────────────────────
    print("Loading watershed data …")
    gdf = gpd.read_file(WATERSHED_GEOJSON)
    gdf = gdf.to_crs(epsg=4326)
    gdf["geometry"] = gdf["geometry"].simplify(tolerance=0.01, preserve_topology=True)

    gdf["pid"]  = gdf["Pour Point ID"].astype(str).str.strip()
    gdf["tier"] = gdf["pid"].map(lambda x: r2_by_id.get(x, {}).get("tier", "unclassified"))
    gdf["r2"]   = gdf["pid"].map(lambda x: r2_by_id.get(x, {}).get("r2", float("nan")))

    # ── Figure / axes ─────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(10, 13), dpi=150)
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0.08, 0.08, 0.82, 0.82], projection=CA_ALBERS)
    ax.set_extent([-124.8, -113.5, 32.2, 42.2], crs=GEODETIC)

    # ── Background features ───────────────────────────────────────────────────
    ax.add_feature(
        cfeature.NaturalEarthFeature("physical", "ocean", "10m",
                                     edgecolor="none", facecolor="#c8dff0"), zorder=0)
    ax.add_feature(
        cfeature.NaturalEarthFeature("physical", "land", "10m",
                                     edgecolor="none", facecolor="#f0ece4"), zorder=1)
    ax.add_feature(
        cfeature.NaturalEarthFeature("physical", "lakes", "10m",
                                     edgecolor="#7aaacf", facecolor="#c8dff0", linewidth=0.4), zorder=2)
    ax.add_feature(
        cfeature.NaturalEarthFeature("physical", "rivers_lake_centerlines", "10m",
                                     edgecolor="#7aaacf", facecolor="none", linewidth=0.35), zorder=3)
    ax.add_feature(
        cfeature.NaturalEarthFeature("cultural", "admin_1_states_provinces", "10m",
                                     edgecolor="#888888", facecolor="none", linewidth=0.6), zorder=4)
    ax.add_feature(
        cfeature.NaturalEarthFeature("cultural", "admin_0_countries", "10m",
                                     edgecolor="#555555", facecolor="none", linewidth=0.9), zorder=4)

    # ── Watersheds colored by tier ────────────────────────────────────────────
    print("Plotting watersheds …")
    for tier in ["unclassified", "tier_1", "tier_2", "tier_3"]:
        subset = gdf[gdf["tier"] == tier]
        for _, row in subset.iterrows():
            ax.add_geometries(
                [row.geometry], crs=GEODETIC,
                facecolor=TIER_COLORS[tier], edgecolor="#1a3a5c",
                linewidth=0.35, alpha=0.82, zorder=5,
            )

    # ── Gridlines ─────────────────────────────────────────────────────────────
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      x_inline=False, y_inline=False,
                      linewidth=0.5, color="grey", alpha=0.5, linestyle="--")
    gl.top_labels    = False
    gl.right_labels  = False
    gl.xlocator      = mticker.FixedLocator([-124, -122, -120, -118, -116, -114])
    gl.ylocator      = mticker.FixedLocator([33, 35, 37, 39, 41])
    gl.xformatter    = LONGITUDE_FORMATTER
    gl.yformatter    = LATITUDE_FORMATTER
    gl.xlabel_style  = {"size": 7, "color": "#333333"}
    gl.ylabel_style  = {"size": 7, "color": "#333333"}

    # ── Legend ────────────────────────────────────────────────────────────────
    legend_elements = []
    for t in ["tier_1", "tier_2", "tier_3", "unclassified"]:
        n = (gdf["tier"] == t).sum()
        if n == 0:
            continue
        legend_elements.append(mpatches.Patch(
            facecolor=TIER_COLORS[t], edgecolor="#1a3a5c",
            linewidth=0.6, alpha=0.82,
            label=f"{TIER_LABELS[t]}  (n={n})",
        ))
    ax.legend(handles=legend_elements, loc="upper right", fontsize=7.5,
              title="Monthly Regression R²\n(flow ~ precip + temp + lag-12mo precip)",
              title_fontsize=7, frameon=True, framealpha=0.93,
              edgecolor="#aaaaaa", handlelength=1.4, handletextpad=0.6, borderpad=0.8)

    # ── Scale bar ─────────────────────────────────────────────────────────────
    def _draw_scalebar(ax, lon0, lat0, length_km=200, color="#222222"):
        deg_per_km = 1 / (111.32 * np.cos(np.radians(lat0)))
        lon1 = lon0 + length_km * deg_per_km
        ax.plot([lon0, lon1], [lat0, lat0], transform=GEODETIC,
                color=color, linewidth=2.5, solid_capstyle="butt", zorder=10)
        for lx in [lon0, lon1]:
            ax.plot([lx, lx], [lat0 - 0.04, lat0 + 0.04], transform=GEODETIC,
                    color=color, linewidth=2.5, zorder=10)
        mid = (lon0 + lon1) / 2
        ax.text(mid, lat0 - 0.18, f"{length_km} km", transform=GEODETIC,
                ha="center", va="top", fontsize=7.5, color=color, fontweight="bold", zorder=10)
        ax.text(lon0, lat0 + 0.08, "0", transform=GEODETIC,
                ha="center", va="bottom", fontsize=6.5, color=color, zorder=10)

    _draw_scalebar(ax, lon0=-123.5, lat0=32.7, length_km=200)

    # ── North arrow ───────────────────────────────────────────────────────────
    ax.annotate("", xy=(0.94, 0.16), xytext=(0.94, 0.10), xycoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", color="#222222", lw=1.5, mutation_scale=14),
                zorder=11)
    ax.text(0.94, 0.17, "N", transform=ax.transAxes,
            ha="center", va="bottom", fontsize=9, fontweight="bold", color="#222222", zorder=11)

    # ── Titles ────────────────────────────────────────────────────────────────
    fig.text(0.5, 0.935,
             "USGS Streamflow Gauging Station Watersheds — Regression Quality Tiers",
             ha="center", va="bottom", fontsize=13, fontweight="bold", color="#111111")
    fig.text(0.5, 0.918,
             "California — Training Dataset  ·  HydroSHEDS Delineation  ·  "
             "Tiers based on monthly OLS R² (flow ~ precip + temp + 12-month lag precip)",
             ha="center", va="bottom", fontsize=8, color="#444444", style="italic")
    fig.text(0.08, 0.055,
             "Sources: USGS National Water Information System; HydroSHEDS (WWF); "
             "Natural Earth. Projection: Albers Equal Area (NAD 83).",
             ha="left", va="top", fontsize=6.2, color="#666666")

    for spine in ax.spines.values():
        spine.set_edgecolor("#555555")
        spine.set_linewidth(0.8)

    # ── Save ──────────────────────────────────────────────────────────────────
    out_png = MAP_WATERSHEDS_DIR / "map_watersheds.png"
    out_pdf = MAP_WATERSHEDS_DIR / "map_watersheds.pdf"
    print(f"Saving → {out_png}")
    fig.savefig(out_png, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Saving → {out_pdf}")
    fig.savefig(out_pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("Done.")


if __name__ == "__main__":
    main()
