"""Replicate the ArcGIS Intersect table between a geo/attribute layer and a
target polygon layer, writing the result to data/prepare/geo_ops/.

Usage (standalone):
    python -m src.data.geo_intersect --geo static --target watersheds
    python -m src.data.geo_intersect --geo static --target huc8
    python -m src.data.geo_intersect --geo static --target huc10

Via the pipeline:
    python scripts/prepare_data.py --geo-intersect --geo static --target watersheds
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd

from src.paths import (
    GEO_OPS_DIR, BASIN_ATLAS_CLIPPED, VIC_GRIDS_GPKG,
    WATERSHEDS_GPKG, WBDHU8_GPKG, WBDHU10_GPKG,
)

# ── Layer registry ─────────────────────────────────────────────────────────────

GEO_LAYERS: dict[str, str] = {
    "static": "BasinATLAS_v10_lev12",
    "meteo":  "VICGrids_CAORNV_LatLong",
}

TARGET_LAYERS: dict[str, str] = {
    "watersheds": "USGS_Training_Watersheds",
    "huc8":       "WBDHU8",
    "huc10":      "WBDHU10",
}

# GeoPackage source for each geo layer (from data/raw/gis/)
_GEO_GPKG: dict[str, Path] = {
    "static": BASIN_ATLAS_CLIPPED,
    "meteo":  VIC_GRIDS_GPKG,
}

# GeoPackage source for each target (from data/raw/gis/)
_TARGET_GPKG: dict[str, Path] = {
    "watersheds": WATERSHEDS_GPKG,
    "huc8":       WBDHU8_GPKG,
    "huc10":      WBDHU10_GPKG,
}

# Output filename for each (geo, target) combination
_OUTPUT_NAMES: dict[tuple[str, str], str] = {
    ("static", "watersheds"): "BasinATLAS_v10_lev12_Intersect_Watersheds.csv",
    ("static", "huc8"):       "BasinATLAS_v10_lev12_Intersect_HUC8.csv",
    ("static", "huc10"):      "BasinATLAS_v10_lev12_Intersect_HUC10.csv",
    ("meteo",  "watersheds"): "VICGrids_Intersect_Watersheds.csv",
    ("meteo",  "huc8"):       "VICGrids_Intersect_HUC8.csv",
    ("meteo",  "huc10"):      "VICGrids_Intersect_HUC10.csv",
}

# Equal-area CRS for accurate area/perimeter computation.
# EPSG:6414 = NAD83(2011) / California Albers (metres).
# NOTE: EPSG:3310 (NAD83 / CA Albers) requires PROJ grid-shift files that may
#       not be installed; EPSG:6414 is functionally equivalent without the
#       datum-shift dependency.
_PROJECTED_CRS = "EPSG:6414"

# Native ID column for each target layer — renamed to PourPtID in the output
# so all downstream pipeline steps can group by a single column name.
_TARGET_ID_COL: dict[str, str] = {
    "watersheds": "PourPtID",
    "huc8":       "huc8",
    "huc10":      "huc10",
}


# ── Core function ──────────────────────────────────────────────────────────────

def run_intersect(geo: str, target: str) -> Path:
    """Intersect *geo* layer with *target* layer and write CSV to GEO_OPS_DIR.

    Parameters
    ----------
    geo:
        Key into GEO_LAYERS (e.g. ``"static"``).
    target:
        Key into TARGET_LAYERS (e.g. ``"watersheds"``, ``"huc8"``, ``"huc10"``).

    Returns
    -------
    Path
        Path to the written CSV.
    """
    if geo not in GEO_LAYERS:
        raise ValueError(f"Unknown --geo '{geo}'. Choices: {list(GEO_LAYERS)}")
    if target not in TARGET_LAYERS:
        raise ValueError(f"Unknown --target '{target}'. Choices: {list(TARGET_LAYERS)}")

    geo_layer    = GEO_LAYERS[geo]
    target_layer = TARGET_LAYERS[target]
    out_name     = _OUTPUT_NAMES[(geo, target)]
    out_path     = GEO_OPS_DIR / out_name

    GEO_OPS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load layers ────────────────────────────────────────────────────────────
    tgt_gpkg = _TARGET_GPKG[target]
    if not tgt_gpkg.exists():
        raise FileNotFoundError(
            f"GeoPackage not found: {tgt_gpkg}\n"
            f"Run data/prepare/geo_ops/export_gdb_layers.py first."
        )
    print(f"Loading target layer : {tgt_gpkg.name}")
    tgt: gpd.GeoDataFrame = gpd.read_file(str(tgt_gpkg))
    print(f"  {len(tgt)} features, CRS: {tgt.crs}")

    geo_gpkg = _GEO_GPKG[geo]
    if not geo_gpkg.exists():
        raise FileNotFoundError(
            f"GeoPackage not found: {geo_gpkg}\n"
            f"Expected at: {geo_gpkg}"
        )
    print(f"Loading geo layer    : {geo_gpkg.name}")
    geo_gdf: gpd.GeoDataFrame = gpd.read_file(str(geo_gpkg))
    print(f"  {len(geo_gdf)} features, CRS: {geo_gdf.crs}")

    # ── Stash original FIDs before overlay renames indices ────────────────────
    tgt_fid_col = f"FID_{target_layer}"
    geo_fid_col = f"FID_{geo_layer}"
    tgt["_tgt_fid"] = range(len(tgt))
    geo_gdf["_geo_fid"] = range(len(geo_gdf))

    # Drop ArcGIS-managed shape columns — we recompute after intersection
    for df in (tgt, geo_gdf):
        for col in ("Shape_Length", "Shape_Area"):
            if col in df.columns:
                df.drop(columns=col, inplace=True)

    # ── Fix invalid geometries before reprojection ─────────────────────────
    import shapely
    for label, df in (("target", tgt), ("geo", geo_gdf)):
        invalid = ~df.geometry.is_valid
        if invalid.any():
            print(f"  Fixing {invalid.sum()} invalid geometries in {label} layer …")
            df.geometry = shapely.make_valid(df.geometry.values)

    # ── Reproject to equal-area CRS for accurate geometry metrics ─────────────
    print(f"Reprojecting to {_PROJECTED_CRS} …")
    tgt_proj = tgt.to_crs(_PROJECTED_CRS)
    geo_proj = geo_gdf.to_crs(_PROJECTED_CRS)

    # ── Spatial intersection ──────────────────────────────────────────────────
    print("Computing intersection (this may take a few minutes for large layers) …")
    intersect: gpd.GeoDataFrame = gpd.overlay(
        tgt_proj, geo_proj, how="intersection", keep_geom_type=False
    )
    print(f"  Intersection produced {len(intersect)} rows")

    # ── Shape metrics from intersection geometry ───────────────────────────────
    intersect["Shape_Area"]   = intersect.geometry.area        # m²
    intersect["Shape_Length"] = intersect.geometry.length      # m

    # ── Rename FID helper columns ─────────────────────────────────────────────
    intersect.rename(columns={"_tgt_fid": tgt_fid_col, "_geo_fid": geo_fid_col}, inplace=True)

    # ── Standardise target ID to PourPtID ─────────────────────────────────────
    native_id = _TARGET_ID_COL.get(target, "PourPtID")
    if native_id != "PourPtID" and native_id in intersect.columns:
        intersect.rename(columns={native_id: "PourPtID"}, inplace=True)

    # ── Column ordering ────────────────────────────────────────────────────────
    # BasinATLAS (static): target FID first, then target attrs, then geo.
    # Meteo grids: geo FID first (matches original ArcGIS output convention where
    #              VICGrids was the Input Features layer).
    tgt_attr_cols = [c for c in tgt.columns    if c not in ("geometry", "_tgt_fid")]
    if native_id != "PourPtID":
        tgt_attr_cols = ["PourPtID" if c == native_id else c for c in tgt_attr_cols]
    geo_attr_cols = [c for c in geo_gdf.columns if c not in ("geometry", "_geo_fid")]

    if geo == "meteo":
        ordered = (
            [geo_fid_col]
            + geo_attr_cols
            + [tgt_fid_col]
            + tgt_attr_cols
            + ["Shape_Length", "Shape_Area"]
        )
    else:
        ordered = (
            [tgt_fid_col]
            + tgt_attr_cols
            + [geo_fid_col]
            + geo_attr_cols
            + ["Shape_Length", "Shape_Area"]
        )
    ordered = [c for c in ordered if c in intersect.columns]

    out_df = intersect[ordered].copy()
    first_sort = geo_fid_col if geo == "meteo" else tgt_fid_col
    second_sort = tgt_fid_col if geo == "meteo" else geo_fid_col
    out_df = out_df.sort_values([first_sort, second_sort]).reset_index(drop=True)

    print(f"Writing {len(out_df)} rows → {out_path}")
    out_df.to_csv(out_path, index=False)
    return out_path


# ── Pipeline entry point ───────────────────────────────────────────────────────

def main(geo: str = "static", target: str = "watersheds") -> None:
    out = run_intersect(geo=geo, target=target)
    print(f"\nDone → {out}")


# ── Standalone CLI ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a GIS intersect table from the project GDB.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--geo",
        choices=list(GEO_LAYERS),
        default="static",
        help="Attribute layer to intersect (static = BasinATLAS_v10_lev12, meteo = VICGrids_CAORNV_LatLong).",
    )
    parser.add_argument(
        "--target",
        choices=list(TARGET_LAYERS),
        default="watersheds",
        help="Target polygon layer.",
    )
    args = parser.parse_args()
    main(geo=args.geo, target=args.target)
