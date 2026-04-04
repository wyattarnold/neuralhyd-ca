"""
Aggregate VIC-Sim daily total runoff (RUNOFF + BASEFLOW) to watershed polygon
boundaries.

Processes all annual NC files (1951–2020) and writes one wide-format CSV per
boundary layer:

  aggregated/huc6_runoff.csv
  aggregated/huc8_runoff.csv
  aggregated/huc10_runoff.csv
  aggregated/huc12_runoff.csv
  aggregated/training_watersheds_runoff.csv

Each CSV has a 'date' column plus one column per polygon (named by the layer's
ID field).  Values are volumetric runoff in CFS, computed as the cos(lat)-weighted
mean total runoff depth (mm/day) × polygon area (m², from EPSG:5070 Conus Albers
projection) × unit conversion, rounded to the nearest integer.  Polygons with
no grid cells inside are silently dropped.

VIC partitions total runoff into two components:
  - RUNOFF:   surface (fast) runoff
  - BASEFLOW: subsurface (slow) drainage
Both must be summed to get total streamflow.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
import xarray as xr
from pyproj import Transformer
from shapely.strtree import STRtree
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

HERE = Path(__file__).parent
RUNOFF_DIR = HERE / "runoff"
BASEFLOW_DIR = HERE / "baseflow"
OUT_DIR = HERE / "aggregated"
GDB = Path(__file__).parents[4] / "data" / "raw" / "neuralhyd-ca.gdb"
WS_PATH = Path(__file__).parents[4] / "data" / "raw" / "watershed_geometry" / "output_watershed.geojson"

OUT_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Layer definitions  (path, id_column, output_stem)
# ---------------------------------------------------------------------------

LAYERS = [
    (GDB, "WBDHU6",  "huc6",          "huc6_runoff"),
    (GDB, "WBDHU8",  "huc8",          "huc8_runoff"),
    (GDB, "WBDHU10", "huc10",         "huc10_runoff"),
    (GDB, "WBDHU12", "huc12",         "huc12_runoff"),
    (WS_PATH, None,  "Pour Point ID", "training_watersheds_runoff"),
]

# mm/day × m² → CFS:  depth_m/day × area_m² = vol_m³/day;  ÷86400 → m³/s;  ×35.3147 → CFS
_MM_DAY_M2_TO_CFS = 35.3147 / (1_000.0 * 86_400.0)

# Equal-area CRS for accurate polygon area computation
_EQUAL_AREA_CRS = "EPSG:5070"  # NAD83 / Conus Albers


# ---------------------------------------------------------------------------
# Compute polygon areas in m²
# ---------------------------------------------------------------------------

def compute_areas_m2(gdf: gpd.GeoDataFrame, id_col: str) -> dict[str, float]:
    """Return {polygon_id: area_m²} using an equal-area projection."""
    gdf_ea = gdf[[id_col, "geometry"]].to_crs(_EQUAL_AREA_CRS)
    return {str(pid): float(area)
            for pid, area in zip(gdf_ea[id_col], gdf_ea.geometry.area)}


# ---------------------------------------------------------------------------
# Build pixel masks for one boundary layer
# ---------------------------------------------------------------------------

def build_masks(
    gdf: gpd.GeoDataFrame,
    id_col: str,
    lat2d: np.ndarray,
    lon2d: np.ndarray,
    src_crs: str = "EPSG:4326",
) -> dict[str, np.ndarray]:
    """Return {polygon_id: flat_pixel_indices} for all polygons with ≥1 pixel."""

    nlat, nlon = lat2d.shape
    n_pixels = nlat * nlon

    # Reproject pixel centres to the layer's CRS if needed
    layer_crs = gdf.crs.to_epsg()
    if layer_crs is None or str(layer_crs) == "4326" or str(layer_crs) == "4269":
        # NAD83 / WGS84 — treat as geographic, use lon/lat directly
        px = lon2d.ravel()
        py = lat2d.ravel()
    else:
        # Projected — transform lon/lat → layer CRS
        transformer = Transformer.from_crs(src_crs, gdf.crs, always_xy=True)
        px, py = transformer.transform(lon2d.ravel(), lat2d.ravel())

    # Build shapely Points array (shapely 2.x vectorised)
    points = shapely.points(px, py)

    # STRtree over polygon geometries, then bulk within-query
    tree = STRtree(gdf.geometry.values)
    # Returns shape (2, K): row0 = input (point) index, row1 = tree (polygon) index
    pt_idx, poly_idx = tree.query(points, predicate="within")

    masks: dict[str, np.ndarray] = {}
    for p_i in np.unique(poly_idx):
        pid = gdf.iloc[p_i][id_col]
        masks[str(pid)] = pt_idx[poly_idx == p_i]

    return masks


# ---------------------------------------------------------------------------
# Area weights — cos(lat) for each pixel (flat index)
# ---------------------------------------------------------------------------

def make_cos_weights(lat2d: np.ndarray) -> np.ndarray:
    """Flat cos(lat) weights matching the ravelled lat2d order."""
    return np.cos(np.deg2rad(lat2d.ravel()))


# ---------------------------------------------------------------------------
# Aggregate one year
# ---------------------------------------------------------------------------

def aggregate_year(
    runoff_path: Path,
    baseflow_path: Path,
    masks_list: list[dict[str, np.ndarray]],
    cos_w: np.ndarray,
) -> list[pd.DataFrame]:
    """
    Load one year's RUNOFF + BASEFLOW NCs, sum them, apply each layer's masks,
    return list of DataFrames (rows=days, cols=polygon_ids).
    """
    ds_r = xr.open_dataset(runoff_path)
    ds_b = xr.open_dataset(baseflow_path)
    # Both have shape (lat, lon, time) in mm/day
    runoff_arr = ds_r["RUNOFF"].values    # (lat, lon, time)
    baseflow_arr = ds_b["BASEFLOW"].values  # (lat, lon, time)
    total = runoff_arr + baseflow_arr
    # Reshape to (time, lat*lon)
    total = total.reshape(-1, total.shape[2]).T
    dates = pd.to_datetime(ds_r["time"].values)
    ds_r.close()
    ds_b.close()

    results = []
    for masks in masks_list:
        rows: dict[str, np.ndarray] = {}
        for pid, idx in masks.items():
            w = cos_w[idx]                           # (npix,)
            data = total[:, idx]                    # (ntime, npix)
            valid = ~np.isnan(data)                  # (ntime, npix)
            # weighted sum over valid pixels; NaN where no valid pixels exist
            w_broadcast = w[np.newaxis, :]           # (1, npix)
            wsum = np.where(valid, w_broadcast, 0.0).sum(axis=1)   # (ntime,)
            wdata = np.where(valid, data * w_broadcast, 0.0).sum(axis=1)
            with np.errstate(invalid="ignore", divide="ignore"):
                rows[pid] = np.where(wsum > 0, wdata / wsum, np.nan)
        results.append(pd.DataFrame(rows, index=dates))

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    nc_files = sorted(RUNOFF_DIR.glob("*.nc"))
    if not nc_files:
        raise FileNotFoundError(f"No .nc files found in {RUNOFF_DIR}")

    # Verify matching BASEFLOW files exist
    bf_files = sorted(BASEFLOW_DIR.glob("*.nc"))
    bf_stems = {f.stem for f in bf_files}
    missing = [f.stem for f in nc_files if f.stem not in bf_stems]
    if missing:
        raise FileNotFoundError(
            f"BASEFLOW files missing for years: {missing}. "
            f"Run download.py to fetch them."
        )

    print(f"Found {len(nc_files)} annual NC files ({nc_files[0].stem}–{nc_files[-1].stem})")

    # Build lat/lon grid from the first file
    ds0 = xr.open_dataset(nc_files[0])
    lats = ds0["lat"].values   # (450,)
    lons = ds0["lon"].values   # (426,)
    ds0.close()

    lat2d, lon2d = np.meshgrid(lats, lons, indexing="ij")   # (450, 426)
    cos_w = make_cos_weights(lat2d)

    # Build masks and areas once per layer
    print("Building pixel masks …")
    layers_meta = []
    masks_list = []
    areas_list = []  # list of {polygon_id: area_m²}
    for src_path, gdb_layer, id_col, out_stem in LAYERS:
        print(f"  {out_stem} …", end=" ", flush=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            kwargs = {"layer": gdb_layer} if gdb_layer is not None else {}
            gdf = gpd.read_file(src_path, **kwargs)
        masks = build_masks(gdf, id_col, lat2d, lon2d)
        areas = compute_areas_m2(gdf, id_col)
        print(f"{len(masks)}/{len(gdf)} polygons have ≥1 pixel")
        layers_meta.append((out_stem, id_col))
        masks_list.append(masks)
        areas_list.append(areas)

    # Accumulate DataFrames across years
    accumulated: list[list[pd.DataFrame]] = [[] for _ in LAYERS]

    for nc_path in tqdm(nc_files, desc="Processing years"):
        bf_path = BASEFLOW_DIR / nc_path.name
        year_dfs = aggregate_year(nc_path, bf_path, masks_list, cos_w)
        for i, df in enumerate(year_dfs):
            accumulated[i].append(df)

    # Concatenate, convert mm/day → CFS, and write
    print("Writing output CSVs …")
    for i, (out_stem, _) in enumerate(layers_meta):
        out_path = OUT_DIR / f"{out_stem}.csv"
        df = pd.concat(accumulated[i]).sort_index()
        df.index.name = "date"
        areas = areas_list[i]
        # multiply each column by its polygon area × unit conversion, then round
        for col in df.columns:
            if col in areas:
                df[col] = (df[col] * areas[col] * _MM_DAY_M2_TO_CFS).round(0)
        df.to_csv(out_path, float_format="%.0f")
        print(f"  {out_path.relative_to(HERE.parent.parent.parent.parent)}  "
              f"({len(df)} days × {df.shape[1]} polygons)")

    print("Done.")


if __name__ == "__main__":
    main()
