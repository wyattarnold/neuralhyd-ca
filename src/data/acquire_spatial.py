"""Acquire DEM and stream-network spatial features for CA watersheds.

Downloads 30 m elevation data from USGS 3DEP and NHDPlus flowlines via
the HyRiver suite, computes per-basin hypsometric, terrain, and
network-topology features.

Requires additional packages (not in base environment)::

    pip install py3dep pynhd rasterio rioxarray pyflwdir

Outputs:
    data/training/static/DEM_Attributes_Watersheds.csv
        ele_p05 .. ele_p95, ele_min, ele_max, ele_range, ele_std,
        hyp_integral, twi_p05 .. twi_p95, twi_mean, twi_std

    data/training/static/Network_Attributes_Watersheds.csv
        max_strahler, drain_density_km, n_reaches, bifurc_ratio,
        main_chan_length_km, main_chan_sinuosity

    data/training/static/Width_Functions_Watersheds.csv
        PourPtID + 32 normalised distance-to-outlet histogram bins
        (wf_00 .. wf_31).  Each row sums to 1.

Called from ``prepare_data.py --analysis acquire_spatial`` or directly::

    python -m src.data.acquire_spatial                  # all basins, fresh
    python -m src.data.acquire_spatial --resume         # skip cached basins
    python -m src.data.acquire_spatial --basin 11522300 # one basin (test)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.paths import STATIC_DIR, WATERSHED_GEOJSON, QA_DIR

# ---- Outputs ---------------------------------------------------------------
DEM_OUTPUT = STATIC_DIR / "DEM_Attributes_Watersheds.csv"
NET_OUTPUT = STATIC_DIR / "Network_Attributes_Watersheds.csv"
WF_OUTPUT  = STATIC_DIR / "Width_Functions_Watersheds.csv"
CACHE_DIR  = QA_DIR / "spatial_cache"

# Redirect HyRiver (async-retriever) HTTP cache into the same folder.
# Must be set before any py3dep/pynhd imports trigger async_retriever.
os.environ.setdefault(
    "HYRIVER_CACHE_NAME",
    str(CACHE_DIR / "aiohttp_cache.sqlite"),
)

N_WF_BINS = 32  # width-function histogram resolution

# Seconds to sleep between API calls (be kind to USGS servers)
REQUEST_DELAY = 1.5

# CRS for length/area calculations (same as source geojson)
PROJ_CRS = "EPSG:6414"  # NAD83 California Albers (metres)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _nan_to_none(obj: object) -> object:
    """Recursively convert NaN → None for JSON serialisation."""
    if isinstance(obj, dict):
        return {k: _nan_to_none(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_nan_to_none(v) for v in obj]
    if isinstance(obj, float) and np.isnan(obj):
        return None
    return obj


def _none_to_nan(obj: object) -> object:
    """Recursively convert None → NaN when reading cache."""
    if isinstance(obj, dict):
        return {k: _none_to_nan(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_none_to_nan(v) for v in obj]
    if obj is None:
        return np.nan
    return obj


def _find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Return the first matching column name, case-insensitive."""
    lower_map = {c.lower(): c for c in df.columns}
    for name in candidates:
        if name.lower() in lower_map:
            return lower_map[name.lower()]
    return None


# ---------------------------------------------------------------------------
# Load watersheds
# ---------------------------------------------------------------------------
def load_watersheds() -> gpd.GeoDataFrame:
    """Load watershed polygons and reproject to EPSG:4326 for API queries."""
    gdf = gpd.read_file(WATERSHED_GEOJSON)
    gdf = gdf.rename(columns={"Pour Point ID": "PourPtID"})
    gdf["PourPtID"] = gdf["PourPtID"].astype(int)
    gdf["area_km2"] = gdf["Area Square Kilometers"].astype(float)
    return gdf.to_crs(epsg=4326)


# ---------------------------------------------------------------------------
# DEM download helper
# ---------------------------------------------------------------------------
def _download_dem(geom):
    """Download and clip 30 m DEM from 3DEP.  Returns xarray DataArray (EPSG:4326)."""
    import py3dep

    dem = py3dep.get_dem(geom, resolution=30, crs=4326)
    try:
        import rioxarray  # noqa: F401  -- registers .rio accessor
        dem = dem.rio.clip([geom], crs=4326)
    except Exception:
        pass  # fall back to bounding-box DEM; NaN masking handles it
    return dem


# ---------------------------------------------------------------------------
# DEM features
# ---------------------------------------------------------------------------
def compute_dem_features(dem, basin_id: int) -> dict:
    """Hypsometric features from a clipped DEM xarray.

    Parameters
    ----------
    dem : xarray.DataArray  (EPSG:4326 elevation in metres)
    basin_id : USGS pour-point ID

    Returns
    -------
    dict with hypsometric statistics
    """
    elev = dem.values.ravel()
    elev = elev[~np.isnan(elev)]

    if len(elev) < 10:
        raise ValueError(f"Basin {basin_id}: only {len(elev)} valid DEM pixels")

    pcts = np.percentile(elev, [5, 10, 25, 50, 75, 90, 95])
    ele_min = float(np.min(elev))
    ele_max = float(np.max(elev))
    ele_range = ele_max - ele_min

    return {
        "PourPtID":     basin_id,
        "ele_p05":      round(float(pcts[0]), 1),
        "ele_p10":      round(float(pcts[1]), 1),
        "ele_p25":      round(float(pcts[2]), 1),
        "ele_p50":      round(float(pcts[3]), 1),
        "ele_p75":      round(float(pcts[4]), 1),
        "ele_p90":      round(float(pcts[5]), 1),
        "ele_p95":      round(float(pcts[6]), 1),
        "ele_min":      round(ele_min, 1),
        "ele_max":      round(ele_max, 1),
        "ele_range":    round(ele_range, 1),
        "ele_std":      round(float(np.std(elev)), 2),
        "hyp_integral": round(float((np.mean(elev) - ele_min) / ele_range), 4)
                        if ele_range > 0 else 0.5,
    }


# ---------------------------------------------------------------------------
# Terrain features (TWI + width function) — requires pyflwdir
# ---------------------------------------------------------------------------
def compute_terrain_features(
    dem, basin_id: int, n_bins: int = N_WF_BINS,
) -> tuple[dict, list[float]]:
    """Compute TWI quantiles and geomorphic width function from DEM.

    The **width function** W(d) is the normalised histogram of flow-distance
    from every DEM pixel to the basin outlet.  It encodes how contributing
    area is distributed along the travel-time axis — a compact 1-D signature
    of basin shape and drainage topology.

    **TWI** = ln(a / tan β) characterises saturation-excess runoff potential.

    Parameters
    ----------
    dem : xarray.DataArray  (EPSG:4326 elevation)
    basin_id : USGS pour-point ID
    n_bins : number of width-function histogram bins  (default 32)

    Returns
    -------
    twi_dict : dict of TWI statistics  (twi_p05 .. twi_p95, twi_mean, twi_std)
    width_func : list[float] of *n_bins* normalised histogram bin values
    """
    import pyflwdir
    import rioxarray  # noqa: F401

    # ---- Reproject to metric CRS for distance / area calculations ----------
    dem_proj = dem.rio.reproject(PROJ_CRS)
    elev = dem_proj.values.squeeze().astype(np.float64)
    if elev.ndim != 2:
        raise ValueError(f"Basin {basin_id}: expected 2-D DEM, got {elev.shape}")

    nodata = dem_proj.rio.nodata
    nodata_val = float(nodata) if nodata is not None else -9999.0

    transform = dem_proj.rio.transform()
    dx = abs(transform.a)   # cell width  (metres)
    dy = abs(transform.e)   # cell height (metres)

    valid = np.isfinite(elev) & (elev != nodata_val)
    elev_work = elev.copy()
    elev_work[~valid] = nodata_val

    # ---- D8 flow direction via pyflwdir ------------------------------------
    flw = pyflwdir.from_dem(
        data=elev_work,
        nodata=nodata_val,
        transform=transform,
        latlon=False,
    )

    # ---- TWI ---------------------------------------------------------------
    acc = flw.upstream_area(unit="cell")           # upstream cell count
    specific_area = acc * dy                        # m² per unit contour length

    grad_y, grad_x = np.gradient(elev_work, dy, dx)
    slope = np.sqrt(grad_x ** 2 + grad_y ** 2)
    slope = np.clip(slope, 1e-3, None)              # avoid log singularity

    with np.errstate(invalid="ignore", divide="ignore"):
        twi = np.log(specific_area / slope)
    twi_valid = twi[valid & (acc > 0)]
    twi_valid = twi_valid[np.isfinite(twi_valid)]

    if len(twi_valid) < 10:
        twi_dict = {f"twi_p{p:02d}": np.nan for p in (5, 10, 25, 50, 75, 90, 95)}
        twi_dict.update(twi_mean=np.nan, twi_std=np.nan)
    else:
        q = np.percentile(twi_valid, [5, 10, 25, 50, 75, 90, 95])
        twi_dict = {
            "twi_p05": round(float(q[0]), 3),
            "twi_p10": round(float(q[1]), 3),
            "twi_p25": round(float(q[2]), 3),
            "twi_p50": round(float(q[3]), 3),
            "twi_p75": round(float(q[4]), 3),
            "twi_p90": round(float(q[5]), 3),
            "twi_p95": round(float(q[6]), 3),
            "twi_mean": round(float(np.mean(twi_valid)), 3),
            "twi_std":  round(float(np.std(twi_valid)), 3),
        }

    # ---- Width function (flow-distance to outlet) --------------------------
    n_cells = elev.size
    ncols = elev.shape[1]
    idxs_ds = flw.idxs_ds

    # Outlet = pixel with maximum flow accumulation
    outlet_flat = int(acc.ravel().argmax())

    # Cell-to-downstream-neighbour step distances (vectorised)
    idx_all = np.arange(n_cells)
    row_all, col_all = np.divmod(idx_all, ncols)
    row_ds, col_ds   = np.divmod(idxs_ds, ncols)
    step_dist = np.sqrt(((row_all - row_ds).astype(np.float64) * dy) ** 2 +
                        ((col_all - col_ds).astype(np.float64) * dx) ** 2)

    # Propagate outlet → headwaters in topological order (rank-by-rank)
    dist = np.full(n_cells, np.nan)
    dist[outlet_flat] = 0.0
    rank_flat = flw.rank.ravel()
    max_rank = int(rank_flat.max())

    for r in range(max_rank + 1):
        cells = np.where(rank_flat == r)[0]
        if len(cells) == 0:
            continue
        ds = idxs_ds[cells]
        known = np.isfinite(dist[ds])
        upd = cells[known]
        dist[upd] = dist[idxs_ds[upd]] + step_dist[upd]

    # Mark remaining pits with zero distance
    pits = (idxs_ds == idx_all) & np.isnan(dist)
    dist[pits] = 0.0

    dist_valid = dist[valid.ravel()]
    dist_valid = dist_valid[np.isfinite(dist_valid)]

    if len(dist_valid) < 10 or dist_valid.max() <= 0:
        width_func = [0.0] * n_bins
    else:
        counts, _ = np.histogram(dist_valid, bins=n_bins,
                                 range=(0, dist_valid.max()))
        total = counts.sum()
        width_func = [round(float(c / total), 6) for c in counts]

    return twi_dict, width_func


# ---------------------------------------------------------------------------
# Stream network features
# ---------------------------------------------------------------------------
def compute_network_features(geom, basin_id: int, area_km2: float) -> dict:
    """Download NHDPlus flowlines and compute topology descriptors.

    Parameters
    ----------
    geom : shapely Polygon (EPSG:4326)
    basin_id : USGS pour-point ID
    area_km2 : basin area in km²

    Returns
    -------
    dict with network topology metrics
    """
    from pynhd import WaterData

    wd = WaterData("nhdflowline_network")
    flowlines = wd.bygeom(geom, geo_crs=4326)

    if flowlines is None or len(flowlines) == 0:
        raise ValueError(f"Basin {basin_id}: no NHDPlus flowlines returned")

    # ---- Identify columns (names vary by service version) ----
    order_col  = _find_col(flowlines, ["streamorde", "stream_order"])
    length_col = _find_col(flowlines, ["lengthkm", "length_km"])

    # Compute length from geometry if column missing
    if length_col is None:
        fl_proj = flowlines.to_crs(PROJ_CRS)
        flowlines["_len_km"] = fl_proj.geometry.length / 1000.0
        length_col = "_len_km"

    lengths = flowlines[length_col].astype(float)
    total_len_km = float(lengths.sum())

    result: dict = {
        "PourPtID":            basin_id,
        "n_reaches":           len(flowlines),
        "total_stream_len_km": round(total_len_km, 3),
        "drain_density_km":    round(total_len_km / area_km2, 4) if area_km2 > 0 else np.nan,
    }

    # ---- Strahler order & bifurcation ratio ----
    if order_col is not None:
        orders = flowlines[order_col].dropna().astype(int)
        orders = orders[orders > 0]
        if len(orders) > 0:
            result["max_strahler"] = int(orders.max())
            counts = orders.value_counts().sort_index()
            if len(counts) > 1:
                ratios = []
                for i in range(len(counts) - 1):
                    n_low = counts.iloc[i]
                    n_high = counts.iloc[i + 1]
                    if n_high > 0:
                        ratios.append(n_low / n_high)
                result["bifurc_ratio"] = round(float(np.mean(ratios)), 3) if ratios else np.nan
            else:
                result["bifurc_ratio"] = np.nan
        else:
            result["max_strahler"] = np.nan
            result["bifurc_ratio"] = np.nan
    else:
        result["max_strahler"] = np.nan
        result["bifurc_ratio"] = np.nan

    # ---- Main channel length & sinuosity ----
    max_ord = result.get("max_strahler", np.nan)
    if order_col is not None and not (isinstance(max_ord, float) and np.isnan(max_ord)):
        from shapely.ops import linemerge, unary_union

        main_mask = flowlines[order_col].astype(float) >= max_ord
        main_fl = flowlines.loc[main_mask]
        main_len_km = float(main_fl[length_col].astype(float).sum())
        result["main_chan_length_km"] = round(main_len_km, 3)

        # Sinuosity = path length / straight-line distance (in projected CRS)
        if len(main_fl) > 0:
            main_proj = main_fl.to_crs(PROJ_CRS)
            combined = unary_union(main_proj.geometry)

            # linemerge needs a collection; skip it for a single LineString
            if combined.geom_type == "MultiLineString":
                merged = linemerge(combined)
                # If still multi after merge, take the longest piece
                if merged.geom_type == "MultiLineString":
                    merged = max(merged.geoms, key=lambda g: g.length)
            elif combined.geom_type == "LineString":
                merged = combined
            else:
                merged = None

            if merged is not None and merged.geom_type == "LineString" and len(merged.coords) >= 2:
                from shapely.geometry import Point
                p1 = Point(merged.coords[0])
                p2 = Point(merged.coords[-1])
                straight_m = p1.distance(p2)
                path_m = merged.length
                result["main_chan_sinuosity"] = (
                    round(path_m / straight_m, 3) if straight_m > 0 else np.nan
                )
            else:
                result["main_chan_sinuosity"] = np.nan
        else:
            result["main_chan_sinuosity"] = np.nan
    else:
        result["main_chan_length_km"] = np.nan
        result["main_chan_sinuosity"] = np.nan

    return result


# ---------------------------------------------------------------------------
# Per-basin orchestrator
# ---------------------------------------------------------------------------
def process_basin(
    row: pd.Series,
    cache_dir: Path,
    resume: bool,
) -> tuple[dict | None, dict | None, list[float] | None]:
    """Process one basin (DEM + terrain + network) with JSON caching."""
    basin_id = int(row["PourPtID"])
    area_km2 = float(row["area_km2"])
    geom = row.geometry

    cache_file = cache_dir / f"{basin_id}.json"
    dem_feat: dict | None = None
    net_feat: dict | None = None
    terrain: dict | None = None  # {"twi": {...}, "width_function": [...]}

    # ---- Resume from cache ----
    if resume and cache_file.exists():
        with open(cache_file) as f:
            cached = json.load(f)
        dem = cached.get("dem")
        net = cached.get("network")
        ter = cached.get("terrain")
        # Only skip if all three succeeded
        if dem is not None and net is not None and ter is not None:
            dem_merged = {**_none_to_nan(dem), **_none_to_nan(ter.get("twi", {}))}
            return dem_merged, _none_to_nan(net), ter.get("width_function")
        # Preserve previously-successful results; only re-fetch failures
        if dem is not None:
            dem_feat = _none_to_nan(dem)
        if net is not None:
            net_feat = _none_to_nan(net)
        if ter is not None:
            terrain = ter

    # ---- Download DEM once (shared by hypsometric + terrain features) ----
    dem_xr = None
    if dem_feat is None or terrain is None:
        try:
            dem_xr = _download_dem(geom)
        except Exception as e:
            print(f"  [DEM DL FAIL] {basin_id}: {e}")
        time.sleep(REQUEST_DELAY)

    # ---- Hypsometric features ----
    if dem_feat is None and dem_xr is not None:
        try:
            dem_feat = compute_dem_features(dem_xr, basin_id)
        except Exception as e:
            print(f"  [DEM FAIL] {basin_id}: {e}")

    # ---- Terrain features (TWI + width function) ----
    if terrain is None and dem_xr is not None:
        try:
            twi_dict, wf = compute_terrain_features(dem_xr, basin_id)
            terrain = {"twi": twi_dict, "width_function": wf}
        except Exception as e:
            print(f"  [TERRAIN FAIL] {basin_id}: {e}")

    # ---- Network ----
    if net_feat is None:
        try:
            net_feat = compute_network_features(geom, basin_id, area_km2)
        except Exception as e:
            print(f"  [NET FAIL] {basin_id}: {e}")
        time.sleep(REQUEST_DELAY)

    # ---- Cache ----
    cache_dir.mkdir(parents=True, exist_ok=True)
    payload = _nan_to_none({"dem": dem_feat, "network": net_feat, "terrain": terrain})
    with open(cache_file, "w") as f:
        json.dump(payload, f, indent=2)

    # Merge TWI into DEM features for the output CSV
    merged_dem = None
    if dem_feat is not None:
        merged_dem = dict(dem_feat)
        if terrain is not None and "twi" in terrain:
            twi = terrain["twi"]
            merged_dem.update(_none_to_nan(twi) if isinstance(twi, dict) else {})

    wf_out = terrain.get("width_function") if terrain is not None else None
    return merged_dem, net_feat, wf_out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def _check_deps() -> None:
    """Verify optional spatial dependencies are installed."""
    missing = []
    for pkg in ("py3dep", "pynhd", "rasterio", "rioxarray"):
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"Missing packages: {', '.join(missing)}")
        print("Install with:\n  pip install py3dep pynhd rasterio rioxarray pyflwdir")
        sys.exit(1)
    try:
        __import__("pyflwdir")
    except ImportError:
        print("WARNING: pyflwdir not installed — TWI and width-function "
              "features will be skipped.\n  pip install pyflwdir")


def main(*, resume: bool = False, basin: int | None = None) -> None:
    """Acquire spatial features for all (or one) watershed.

    Parameters
    ----------
    resume : bool
        If True, skip basins that already have complete cache files.
    basin : int | None
        If set, process only this basin ID (for testing).
    """
    _check_deps()

    # ---- Load basins ----
    gdf = load_watersheds()
    print(f"Loaded {len(gdf)} watersheds  (reprojected to EPSG:4326)")

    if basin is not None:
        gdf = gdf[gdf["PourPtID"] == basin]
        if len(gdf) == 0:
            print(f"Basin {basin} not found in {WATERSHED_GEOJSON}")
            sys.exit(1)

    # ---- Process ----
    dem_rows: list[dict] = []
    net_rows: list[dict] = []
    wf_rows: list[dict] = []

    for _, row in tqdm(gdf.iterrows(), total=len(gdf), desc="Basins"):
        dem_feat, net_feat, wf = process_basin(row, CACHE_DIR, resume)
        if dem_feat is not None:
            dem_rows.append(dem_feat)
        if net_feat is not None:
            net_rows.append(net_feat)
        if wf is not None:
            wf_row = {"PourPtID": int(row["PourPtID"])}
            wf_row.update({f"wf_{i:02d}": v for i, v in enumerate(wf)})
            wf_rows.append(wf_row)

    # ---- Save DEM + TWI features ----
    if dem_rows:
        dem_df = pd.DataFrame(dem_rows).sort_values("PourPtID").reset_index(drop=True)
        DEM_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        dem_df.to_csv(DEM_OUTPUT, index=False)
        print(f"\nDEM + TWI features → {DEM_OUTPUT}  ({len(dem_df)} basins)")
        print(dem_df.drop(columns="PourPtID").describe().round(2).to_string())

    # ---- Save network features ----
    if net_rows:
        net_df = pd.DataFrame(net_rows).sort_values("PourPtID").reset_index(drop=True)
        NET_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        net_df.to_csv(NET_OUTPUT, index=False)
        print(f"\nNetwork features → {NET_OUTPUT}  ({len(net_df)} basins)")
        print(net_df.drop(columns="PourPtID").describe().round(2).to_string())

    # ---- Save width-function features ----
    if wf_rows:
        wf_df = pd.DataFrame(wf_rows).sort_values("PourPtID").reset_index(drop=True)
        WF_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        wf_df.to_csv(WF_OUTPUT, index=False)
        print(f"\nWidth functions → {WF_OUTPUT}  ({len(wf_df)} basins)")

    # ---- Summary ----
    n = len(gdf)
    n_dem_fail = n - len(dem_rows)
    n_net_fail = n - len(net_rows)
    n_wf_fail  = n - len(wf_rows)
    if n_dem_fail or n_net_fail or n_wf_fail:
        print(f"\nFailures: {n_dem_fail} DEM, {n_net_fail} network, {n_wf_fail} terrain")
        print("Re-run with --resume to retry only failed basins")
    else:
        print(f"\nAll {n} basins processed successfully")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download DEM + NHDPlus features for CA watersheds",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip basins that already have complete cache files",
    )
    parser.add_argument(
        "--basin", type=int, default=None,
        help="Process a single basin ID (for testing)",
    )
    args = parser.parse_args()
    main(resume=args.resume, basin=args.basin)
