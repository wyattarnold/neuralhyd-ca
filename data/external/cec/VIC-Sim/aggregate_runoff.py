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
ID field).  Values are volumetric runoff in CFS, computed as the
overlap-fraction × cos(lat)-weighted mean total runoff depth (mm/day) × polygon
area (m², from EPSG:5070 Conus Albers projection) × unit conversion, rounded to
the nearest integer.  Grid cells partially overlapping a polygon are weighted by
their fractional overlap area, so even sub-grid-cell watersheds receive valid
estimates.  Polygons with no intersecting grid cells are silently dropped.

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
WS_PATH = Path(__file__).parents[4] / "data" / "training" / "watersheds" / "watersheds.geojson"

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
# Mask type: polygon_id → (flat_pixel_indices, overlap_fraction_weights)
# ---------------------------------------------------------------------------

Mask = dict[str, tuple[np.ndarray, np.ndarray]]


# ---------------------------------------------------------------------------
# Build pixel masks for one boundary layer
# ---------------------------------------------------------------------------

def build_masks(
    gdf: gpd.GeoDataFrame,
    id_col: str,
    lat2d: np.ndarray,
    lon2d: np.ndarray,
    src_crs: str = "EPSG:4326",
) -> Mask:
    """Return {polygon_id: (flat_pixel_indices, overlap_fraction_weights)}.

    Grid cells are constructed as boxes around each pixel centre.  Each
    polygon is intersected with overlapping grid cells, and the weight for
    each cell is the fraction of the cell area covered by the polygon.
    Watersheds smaller than a single cell still receive valid estimates
    (the overlapping cell gets a fractional weight).
    """
    dlat = abs(float(lat2d[1, 0] - lat2d[0, 0]))
    dlon = abs(float(lon2d[0, 1] - lon2d[0, 0]))

    # Build grid cell boxes in geographic coords
    lats_flat = lat2d.ravel()
    lons_flat = lon2d.ravel()
    cell_boxes = shapely.box(
        lons_flat - dlon / 2, lats_flat - dlat / 2,
        lons_flat + dlon / 2, lats_flat + dlat / 2,
    )
    cell_areas = shapely.area(cell_boxes)

    # Bring polygons into geographic coords to match the grid
    polys = gdf[[id_col, "geometry"]].to_crs(src_crs)

    # Bulk spatial query: which cells intersect which polygons
    tree = STRtree(cell_boxes)
    poly_idx, cell_idx = tree.query(polys.geometry.values, predicate="intersects")

    if len(poly_idx) == 0:
        return {}

    # Vectorised intersection → area fractions
    int_areas = shapely.area(
        shapely.intersection(polys.geometry.values[poly_idx], cell_boxes[cell_idx])
    )
    fractions = int_areas / cell_areas[cell_idx]

    # Group by polygon
    masks: Mask = {}
    for p_i in np.unique(poly_idx):
        sel = poly_idx == p_i
        pid = str(polys.iloc[p_i][id_col])
        c_idx = cell_idx[sel]
        frac = fractions[sel]
        valid = frac > 1e-9  # filter out edge touches
        if valid.any():
            masks[pid] = (c_idx[valid], frac[valid])

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

def _weighted_mean(data_2d: np.ndarray, cos_w: np.ndarray, masks: Mask, dates) -> pd.DataFrame:
    """Compute overlap-fraction × cos-weighted spatial mean. data_2d shape = (ntime, npix)."""
    rows: dict[str, np.ndarray] = {}
    for pid, (idx, frac) in masks.items():
        w = cos_w[idx] * frac
        data = data_2d[:, idx]
        valid = ~np.isnan(data)
        w_broadcast = w[np.newaxis, :]
        wsum = np.where(valid, w_broadcast, 0.0).sum(axis=1)
        wdata = np.where(valid, data * w_broadcast, 0.0).sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            rows[pid] = np.where(wsum > 0, wdata / wsum, np.nan)
    return pd.DataFrame(rows, index=dates)


def aggregate_year(
    runoff_path: Path,
    baseflow_path: Path,
    masks_list: list[Mask],
    cos_w: np.ndarray,
    components: bool = False,
) -> tuple[list[pd.DataFrame], list[pd.DataFrame] | None, list[pd.DataFrame] | None]:
    """
    Load one year's RUNOFF + BASEFLOW NCs, sum them, apply each layer's masks.

    Returns (total_dfs, baseflow_dfs, surface_dfs).  When *components* is False
    the second and third elements are None (backward-compatible).
    """
    ds_r = xr.open_dataset(runoff_path)
    ds_b = xr.open_dataset(baseflow_path)
    runoff_arr = ds_r["RUNOFF"].values      # (lat, lon, time) mm/day
    baseflow_arr = ds_b["BASEFLOW"].values  # (lat, lon, time) mm/day
    total_arr = runoff_arr + baseflow_arr
    # Reshape to (time, npix)
    total_flat = total_arr.reshape(-1, total_arr.shape[2]).T
    dates = pd.to_datetime(ds_r["time"].values)
    ds_r.close()
    ds_b.close()

    total_results = [_weighted_mean(total_flat, cos_w, m, dates) for m in masks_list]

    if not components:
        return total_results, None, None

    bf_flat = baseflow_arr.reshape(-1, baseflow_arr.shape[2]).T
    sf_flat = runoff_arr.reshape(-1, runoff_arr.shape[2]).T
    # Only compute components for the last layer (training_watersheds)
    bf_results = [None] * len(masks_list)
    sf_results = [None] * len(masks_list)
    last = len(masks_list) - 1
    bf_results[last] = _weighted_mean(bf_flat, cos_w, masks_list[last], dates)
    sf_results[last] = _weighted_mean(sf_flat, cos_w, masks_list[last], dates)
    return total_results, bf_results, sf_results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--components", action="store_true",
                        help="Also output baseflow and surface runoff CSVs")
    args = parser.parse_args()

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
        print(f"{len(masks)}/{len(gdf)} polygons have ≥1 overlapping cell")
        layers_meta.append((out_stem, id_col))
        masks_list.append(masks)
        areas_list.append(areas)

    # Accumulate DataFrames across years
    n_layers = len(LAYERS)
    accumulated_total: list[list[pd.DataFrame]] = [[] for _ in range(n_layers)]
    accumulated_bf: list[list[pd.DataFrame]] = [[] for _ in range(n_layers)]
    accumulated_sf: list[list[pd.DataFrame]] = [[] for _ in range(n_layers)]

    for nc_path in tqdm(nc_files, desc="Processing years"):
        bf_path = BASEFLOW_DIR / nc_path.name
        total_dfs, bf_dfs, sf_dfs = aggregate_year(
            nc_path, bf_path, masks_list, cos_w, components=args.components,
        )
        for i, df in enumerate(total_dfs):
            accumulated_total[i].append(df)
        if bf_dfs is not None:
            for i, df in enumerate(bf_dfs):
                if df is not None:
                    accumulated_bf[i].append(df)
            for i, df in enumerate(sf_dfs):
                if df is not None:
                    accumulated_sf[i].append(df)

    # Helper to concat, convert mm/day → CFS, and write
    def _write(accumulated, areas, stem_suffix=""):
        for i, (out_stem, _) in enumerate(layers_meta):
            out_path = OUT_DIR / f"{out_stem}{stem_suffix}.csv"
            df = pd.concat(accumulated[i]).sort_index()
            df.index.name = "date"
            a = areas_list[i]
            for col in df.columns:
                if col in a:
                    df[col] = (df[col] * a[col] * _MM_DAY_M2_TO_CFS).round(0)
            df.to_csv(out_path, float_format="%.0f")
            print(f"  {out_path.relative_to(HERE.parent.parent.parent.parent)}  "
                  f"({len(df)} days × {df.shape[1]} polygons)")

    print("Writing output CSVs …")
    _write(accumulated_total, areas_list)
    if args.components:
        # Only write component CSVs for training_watersheds (last layer)
        def _write_last(accumulated, areas, stem_suffix):
            i = len(layers_meta) - 1
            out_stem, _ = layers_meta[i]
            out_path = OUT_DIR / f"{out_stem}{stem_suffix}.csv"
            df = pd.concat(accumulated[i]).sort_index()
            df.index.name = "date"
            a = areas_list[i]
            for col in df.columns:
                if col in a:
                    df[col] = (df[col] * a[col] * _MM_DAY_M2_TO_CFS).round(0)
            df.to_csv(out_path, float_format="%.0f")
            print(f"  {out_path.relative_to(HERE.parent.parent.parent.parent)}  "
                  f"({len(df)} days × {df.shape[1]} polygons)")
        print("Writing baseflow CSV (training_watersheds) …")
        _write_last(accumulated_bf, areas_list, "_baseflow")
        print("Writing surface runoff CSV (training_watersheds) …")
        _write_last(accumulated_sf, areas_list, "_surface")

    print("Done.")


if __name__ == "__main__":
    main()
