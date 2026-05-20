"""Build the subbasin ↔ gauge-basin intersection table for HUC-based workflows.

Supports any Watershed Boundary Dataset level (default ``huc12``).  For
every USGS training watershed, compute the set of subbasin polygons that
overlap it and the area of each overlap (km², in an equal-area CRS).
Apply the gauge-filter rule ("gauge too small to be a subbasin") and
write level-tagged CSVs to ``data/prepare/geo_ops/``:

    {LEVEL}_Intersect_Watersheds.csv
        columns: PourPtID, <level>, gauge_area_km2, <level>_area_km2,
                 overlap_area_km2, frac_of_gauge, frac_of_<level>, and
                 optional BasinATLAS routing proxies including
                 flow_length_to_gauge_km and gauge_flow_length_km

    {LEVEL}_Kept_Gauges.csv
        columns: PourPtID, n_<level>, total_overlap_km2, gauge_area_km2,
                 max_frac_of_gauge, reason_dropped (empty if kept), kept

    {LEVEL}_In_Scope.csv
        columns: <level>, <level>_area_km2   (unique subbasins used)

where ``{LEVEL}`` is e.g. ``HUC12`` and ``<level>`` is the lower-case id
column (``huc10`` or ``huc12``) in the source WBD geopackage.

Filter rule (``gauge_min_fraction_of_subbasin``, default 0.70):
    A gauge is dropped only when it sits inside a **single** subbasin
    (i.e. its entire area overlaps exactly one subbasin after sliver
    removal) AND the gauge is less than 70 % of that subbasin's area.
    Gauges that span two or more subbasins are always kept because they
    have no "single overlying subbasin" to compare against.

Standalone:
    python -m src.data.subbasin_gauge_intersect --level huc12
Via pipeline:
    python scripts/prepare_data.py --step 9
"""
from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import pandas as pd
import shapely

from src.data.io import load_flow_dataframes
from src.paths import (
    BASIN_ATLAS_INPUT,
    FLOW_ZARR,
    GEO_OPS_DIR,
    WATERSHEDS_GPKG,
    WBDHU10_GPKG,
    WBDHU12_GPKG,
)

# Equal-area CRS (metres) for accurate area/overlap math.
_PROJECTED_CRS = "EPSG:6414"

# NLDI / NHDPlus WaterData settings
_NLDI_CACHE_DIR = GEO_OPS_DIR / "nldi_cache"
_NLDI_CHUNK_SIZE = 300     # max COMIDs per WaterData request
_NLDI_UPSTREAM_KM = 1000   # upstream navigation distance (km); covers any CA watershed

_LEVEL_GPKG = {
    "huc10": WBDHU10_GPKG,
    "huc12": WBDHU12_GPKG,
}


def _id_width(level: str) -> int:
    return 12 if level.lower() == "huc12" else 10


def _load_basinatlas_routing_distances(level: str, id_col: str) -> pd.DataFrame:
    """Return area-weighted BasinATLAS distance attributes by WBD unit (global fallback).

    This is the legacy approach: for each WBD unit it averages DIST_SINK over
    ALL BasinATLAS polygons that overlap it anywhere in the global intersect.
    Because large upstream BasinATLAS sub-basins can span a HUC12 and dominate
    the area-weighted average, this tends to overestimate dist_sink_km for
    outlet HUC12s.  Use :func:`_load_routing_distances_watershed_filtered`
    instead when the BA-Watersheds intersect file is available.
    """
    level_upper = level.upper()
    source = GEO_OPS_DIR / f"BasinATLAS_v10_lev12_Intersect_{level_upper}.csv"
    if not source.exists():
        print(
            f"  WARNING: {source.name} not found; routing distance columns will be omitted. "
            f"Run --geo-intersect --geo static --target {level} first."
        )
        return pd.DataFrame()

    header = pd.read_csv(source, nrows=0).columns
    source_id_col = "PourPtID" if "PourPtID" in header else id_col
    needed = [source_id_col, "Shape_Area", "DIST_SINK", "DIST_MAIN"]
    missing = [col for col in needed if col not in header]
    if missing:
        print(
            f"  WARNING: {source.name} missing {missing}; routing distance columns will be omitted."
        )
        return pd.DataFrame()

    df = pd.read_csv(source, usecols=needed, dtype={source_id_col: str})
    df[source_id_col] = df[source_id_col].astype(str).str.zfill(_id_width(level))
    df["Shape_Area"] = pd.to_numeric(df["Shape_Area"], errors="coerce").clip(lower=0)
    for col in ("DIST_SINK", "DIST_MAIN"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df.loc[df[col] == -9999, col] = pd.NA

    rows: list[dict[str, float | str]] = []
    for unit_id, group in df.groupby(source_id_col):
        weights = group["Shape_Area"].fillna(0.0).to_numpy(dtype=float)
        out: dict[str, float | str] = {id_col: str(unit_id).zfill(_id_width(level))}
        for source_col, out_col in (("DIST_SINK", "dist_sink_km"), ("DIST_MAIN", "dist_main_km")):
            values = group[source_col].to_numpy(dtype=float)
            valid = pd.notna(values) & (weights > 0)
            if valid.any():
                out[out_col] = float((values[valid] * weights[valid]).sum() / weights[valid].sum())
            else:
                out[out_col] = float("nan")
        rows.append(out)
    return pd.DataFrame(rows)


def _load_routing_distances_watershed_filtered(level: str, id_col: str) -> pd.DataFrame:
    """Per-(gauge, HUC) DIST_SINK via HYBAS_ID join between BA-Watersheds and BA-HUC.

    For each training watershed, this function restricts the BasinATLAS polygons
    used for DIST_SINK computation to those that fall *within the watershed boundary*
    (from ``BasinATLAS_v10_lev12_Intersect_Watersheds.csv``).  It then maps each
    BA polygon to its WBD unit via the HYBAS_ID in
    ``BasinATLAS_v10_lev12_Intersect_{LEVEL}.csv``.

    Advantages over the global per-HUC fallback
    --------------------------------------------
    * Eliminates contamination from large BA sub-basins belonging to a different
      river network that happen to overlap a WBD HUC12 polygon globally.
    * Weights by the watershed-clipped Shape_Area, so cross-boundary BA fragments
      that barely clip the training watershed boundary are naturally down-weighted.
    * The gauge outlet identification (``min dist_sink_km`` within the watershed
      group) correctly finds the downstream-most HUC12 because it uses only BA
      polygons from within the watershed.

    Returns a DataFrame with columns ``[PourPtID, <id_col>, dist_sink_km,
    dist_main_km]`` — one row per (gauge, WBD unit) pair that has BA polygon
    coverage inside the watershed.  Returns an empty DataFrame if either source
    file is missing.
    """
    level_upper = level.upper()
    ba_ws_path = BASIN_ATLAS_INPUT          # BasinATLAS_v10_lev12_Intersect_Watersheds.csv
    ba_huc_path = GEO_OPS_DIR / f"BasinATLAS_v10_lev12_Intersect_{level_upper}.csv"

    if not ba_ws_path.exists():
        print(
            f"  WARNING: {ba_ws_path.name} not found; "
            "falling back to global HUC-level routing distances."
        )
        return pd.DataFrame()

    if not ba_huc_path.exists():
        print(
            f"  WARNING: {ba_huc_path.name} not found; routing distance columns will be omitted. "
            f"Run --geo-intersect --geo static --target {level} first."
        )
        return pd.DataFrame()

    # --- Load BA-Watersheds: one row per (gauge, BasinATLAS polygon) ----------
    ws_header = pd.read_csv(ba_ws_path, nrows=0).columns
    ws_needed = [c for c in ["PourPtID", "HYBAS_ID", "DIST_SINK", "DIST_MAIN", "Shape_Area"]
                 if c in ws_header]
    if not {"PourPtID", "HYBAS_ID", "DIST_SINK", "Shape_Area"}.issubset(ws_needed):
        print(
            f"  WARNING: {ba_ws_path.name} is missing required columns "
            "(need PourPtID, HYBAS_ID, DIST_SINK, Shape_Area); skipping watershed filter."
        )
        return pd.DataFrame()

    ba_ws = pd.read_csv(ba_ws_path, usecols=ws_needed,
                        dtype={"PourPtID": str, "HYBAS_ID": str})
    ba_ws["Shape_Area"] = pd.to_numeric(ba_ws["Shape_Area"], errors="coerce").clip(lower=0)
    for col in ("DIST_SINK", "DIST_MAIN"):
        if col in ba_ws.columns:
            ba_ws[col] = pd.to_numeric(ba_ws[col], errors="coerce")
            ba_ws.loc[ba_ws[col] == -9999, col] = pd.NA

    # --- Load BA-HUC: HYBAS_ID → WBD unit id ---------------------------------
    huc_header = pd.read_csv(ba_huc_path, nrows=0).columns
    huc_id_src = "PourPtID" if "PourPtID" in huc_header else id_col
    if "HYBAS_ID" not in huc_header or huc_id_src not in huc_header:
        print(
            f"  WARNING: {ba_huc_path.name} missing HYBAS_ID or id column; "
            "routing filter skipped."
        )
        return pd.DataFrame()

    ba_huc = pd.read_csv(ba_huc_path, usecols=[huc_id_src, "HYBAS_ID", "Shape_Area"],
                         dtype={huc_id_src: str, "HYBAS_ID": str})
    ba_huc[huc_id_src] = ba_huc[huc_id_src].astype(str).str.zfill(_id_width(level))
    ba_huc["Shape_Area"] = pd.to_numeric(ba_huc["Shape_Area"], errors="coerce").clip(lower=0)
    ba_huc = ba_huc.rename(columns={huc_id_src: id_col, "Shape_Area": "Shape_Area_huc"})
    ba_huc = ba_huc.drop_duplicates(["HYBAS_ID", id_col])

    # --- Join: restrict to BA polygons that are both in the watershed and a HUC -
    merged = ba_ws.merge(ba_huc[["HYBAS_ID", id_col, "Shape_Area_huc"]], on="HYBAS_ID", how="inner")
    if merged.empty:
        print(
            f"  WARNING: No HYBAS_ID overlap between {ba_ws_path.name} and "
            f"{ba_huc_path.name}; skipping watershed filter."
        )
        return pd.DataFrame()

    # Weight = min(Shape_Area_ws, Shape_Area_huc) — approximates the BA polygon
    # area that falls in BOTH the watershed and the HUC.  This correctly
    # down-weights polygons that are large in the watershed but barely clip the
    # HUC (or vice-versa), avoiding inflated influence from cross-boundary
    # polygons.
    merged["weight"] = merged[["Shape_Area", "Shape_Area_huc"]].min(axis=1)

    # --- Per-(gauge, HUC): area-weighted avg DIST_SINK using intersection-area weight ---
    rows: list[dict[str, float | str]] = []
    for (gauge_id, huc_id), grp in merged.groupby(["PourPtID", id_col]):
        w = grp["weight"].fillna(0.0).to_numpy(dtype=float)
        row: dict[str, float | str] = {"PourPtID": str(gauge_id), id_col: str(huc_id)}
        for col, out_col in (("DIST_SINK", "dist_sink_km"), ("DIST_MAIN", "dist_main_km")):
            if col not in grp.columns:
                continue
            v = grp[col].to_numpy(dtype=float)
            valid = pd.notna(v) & (w > 0)
            row[out_col] = (
                float((v[valid] * w[valid]).sum() / w[valid].sum())
                if valid.any()
                else float("nan")
            )
        rows.append(row)

    result = pd.DataFrame(rows)
    n_gauges = result["PourPtID"].nunique() if not result.empty else 0
    n_units = result[id_col].nunique() if not result.empty else 0
    print(
        f"  Watershed-filtered routing distances: "
        f"{n_gauges} gauges × {n_units} unique {level_upper}s"
    )
    return result


def _load_routing_distances_nldi(
    level: str,
    id_col: str,
    overlap_df: pd.DataFrame,
) -> pd.DataFrame:
    """Fetch true along-network routing distances via NLDI + NHDPlus WaterData.

    For each USGS gauge in *overlap_df*, navigates upstream tributaries via
    the NLDI API, then fetches full flowline attributes (``pathlength``,
    geometry) from the NHDPlus WaterData service.  Each flowline is spatially
    joined to the WBD polygons that intersect that gauge, and the outlet reach
    per WBD unit is taken as the reach with the minimum ``pathlength`` (i.e.
    closest to the network mouth).

    NHDPlus ``pathlength`` is the cumulative distance from the network mouth to
    each reach's outlet — the same concept as BasinATLAS ``DIST_SINK`` — so the
    returned ``dist_sink_km`` column slots directly into the existing downstream
    logic that computes ``flow_length_to_gauge_km`` as a difference.

    Advantages over the BasinATLAS proxy
    -------------------------------------
    * Uses the authoritative US stream-network topology (NHDPlus V2); no
      HydroSHEDS basin-boundary misalignment for coastal CA gauges.
    * Distances follow the actual stream network, not polygon-area-weighted
      averages of coarse HydroSHEDS sub-basins.
    * HUC12 assignment via spatial join to the WBD geopackage — no dependency
      on BasinATLAS intersection files.

    Caching
    -------
    Upstream flowlines for each gauge are saved as GeoParquet files under
    ``data/prepare/geo_ops/nldi_cache/flowlines_{site_id}.parquet`` so the
    expensive API calls are made only once.  Delete a cache file to force
    a re-fetch for a specific gauge.

    Parameters
    ----------
    level : str
        WBD level (``"huc12"`` or ``"huc10"``).
    id_col : str
        Column name for the WBD unit id (``"huc12"`` or ``"huc10"``).
    overlap_df : pd.DataFrame
        Gauge × WBD overlap table (must contain ``PourPtID`` and *id_col*
        columns) — used to determine which gauges and WBD units to query.

    Returns
    -------
    pd.DataFrame
        Columns ``[PourPtID, id_col, dist_sink_km, dist_main_km]``, one row
        per (gauge, WBD unit) pair with NHD coverage.  Returns an empty
        DataFrame if ``pynhd`` is not installed or all gauge queries fail.
    """
    try:
        from pynhd import NLDI, WaterData  # type: ignore[import]
    except ImportError:
        print(
            "  pynhd not installed; NLDI routing distances unavailable. "
            "Install with: pip install pynhd>=0.16"
        )
        return pd.DataFrame()

    _NLDI_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    id_width = _id_width(level)
    level_upper = level.upper()

    # Load WBD polygons restricted to the HUC units that appear in overlap_df.
    scope_hucs = set(overlap_df[id_col].astype(str).str.zfill(id_width))
    print(f"  Loading {level_upper} boundaries for {len(scope_hucs)} in-scope units …")
    huc_gdf: gpd.GeoDataFrame = gpd.read_file(str(_LEVEL_GPKG[level]))
    huc_gdf[id_col] = huc_gdf[id_col].astype(str).str.zfill(id_width)
    huc_gdf = huc_gdf[huc_gdf[id_col].isin(scope_hucs)][[id_col, "geometry"]].copy()
    huc_gdf = huc_gdf.to_crs(_PROJECTED_CRS)

    nldi = NLDI()
    wd = WaterData("nhdflowline_network")

    gauge_ids = overlap_df["PourPtID"].astype(str).unique().tolist()
    n_total = len(gauge_ids)
    rows: list[dict[str, float | str]] = []
    n_ok = 0

    for i, site_id in enumerate(gauge_ids, 1):
        cache_path = _NLDI_CACHE_DIR / f"flowlines_{site_id}.parquet"
        print(f"  [{i:>3}/{n_total}] USGS-{site_id}", end=" … ", flush=True)

        # ── Load cached or fetch from NLDI + WaterData ───────────────────────
        if cache_path.exists():
            fl = gpd.read_parquet(str(cache_path))
            print(f"(cached, {len(fl)} reaches)", flush=True)
        else:
            try:
                upstream = nldi.navigate_byid(
                    fsource="nwissite",
                    fid=f"USGS-{site_id}",
                    navigation="upstreamTributaries",
                    source="flowlines",
                    distance=_NLDI_UPSTREAM_KM,
                )
            except Exception as exc:
                print(f"NLDI navigate failed: {exc}")
                continue

            if upstream is None or (hasattr(upstream, "empty") and upstream.empty):
                print("no upstream flowlines returned")
                continue

            comid_col = next(
                (c for c in upstream.columns if "comid" in c.lower()), None
            )
            if comid_col is None:
                print(f"no COMID column in NLDI response: {list(upstream.columns)}")
                continue

            comids = upstream[comid_col].astype(str).tolist()

            # Fetch full attributes (pathlength, geometry) in chunks.
            fl_parts: list[gpd.GeoDataFrame] = []
            for j in range(0, len(comids), _NLDI_CHUNK_SIZE):
                chunk = comids[j : j + _NLDI_CHUNK_SIZE]
                try:
                    part = wd.byid("comid", chunk)
                    fl_parts.append(part)
                except Exception as exc_chunk:
                    print(
                        f"\n    WaterData chunk {j // _NLDI_CHUNK_SIZE + 1} failed: {exc_chunk}",
                        end="",
                    )

            if not fl_parts:
                print(" — WaterData returned no data")
                continue

            fl = gpd.GeoDataFrame(pd.concat(fl_parts, ignore_index=True))
            fl.to_parquet(str(cache_path))
            print(f"(fetched {len(fl)} reaches, cached)", flush=True)

        if fl.empty:
            continue

        # ── Normalise column names (pynhd may vary case) ─────────────────────
        fl = fl.rename(columns={c: c.lower() for c in fl.columns})

        if "pathlength" not in fl.columns:
            print(
                f"  WARNING: 'pathlength' missing for {site_id}; "
                f"available columns: {list(fl.columns)}"
            )
            continue

        fl["pathlength"] = pd.to_numeric(fl["pathlength"], errors="coerce")
        fl = fl[fl["pathlength"].notna()].copy()
        if fl.empty:
            continue

        # ── Spatial join flowlines → WBD units for this gauge ────────────────
        fl_proj = fl[["pathlength", "geometry"]].to_crs(_PROJECTED_CRS)

        gauge_huc_ids = set(
            overlap_df.loc[
                overlap_df["PourPtID"].astype(str) == site_id, id_col
            ].astype(str).str.zfill(id_width)
        )
        gauge_hucs = huc_gdf[huc_gdf[id_col].isin(gauge_huc_ids)].copy()
        if gauge_hucs.empty:
            continue

        joined = gpd.sjoin(
            fl_proj,
            gauge_hucs[[id_col, "geometry"]],
            how="inner",
            predicate="intersects",
        )
        if joined.empty:
            continue

        # Per WBD unit: outlet reach = minimum pathlength (closest to sink).
        for huc_id, grp in joined.groupby(id_col):
            rows.append({
                "PourPtID": site_id,
                id_col: str(huc_id).zfill(id_width),
                "dist_sink_km": float(grp["pathlength"].min()),
                "dist_main_km": float("nan"),  # no direct NHDPlus equivalent
            })
        n_ok += 1

    result = pd.DataFrame(rows)
    n_g = result["PourPtID"].nunique() if not result.empty else 0
    n_u = result[id_col].nunique() if not result.empty else 0
    print(
        f"  NLDI routing distances complete: {n_g}/{n_total} gauges successful, "
        f"{n_u} unique {level_upper}s covered"
    )
    return result


def _add_routing_distance_columns(
    overlap_df: pd.DataFrame,
    level: str,
    id_col: str,
    use_nldi: bool = False,
) -> pd.DataFrame:
    """Add per-row HUC distance and relative flow-length proxy columns.

    Source priority (highest to lowest):

    1. **NLDI / NHDPlus WaterData** (when *use_nldi* is ``True``) — true
       along-network distances using ``pathlength`` from NHDPlus V2.  Any
       (gauge, WBD-unit) pairs not covered by NLDI are filled from source 2.
    2. **Watershed-filtered BasinATLAS** — per-(gauge, HUC) DIST_SINK using
       only BA polygons within the training watershed boundary.
    3. **Global per-HUC BasinATLAS** — legacy fallback when source 2 files
       are absent.
    """
    id_width = _id_width(level)
    routing = pd.DataFrame()
    use_per_gauge = False

    # ── Source 1: NLDI ───────────────────────────────────────────────────────
    if use_nldi:
        print(f"  Fetching NLDI routing distances for {level.upper()} …")
        routing = _load_routing_distances_nldi(level, id_col, overlap_df)
        if not routing.empty:
            use_per_gauge = True
            # Gap-fill: pairs not covered by NLDI → try BasinATLAS
            covered = set(
                zip(
                    routing["PourPtID"].astype(str),
                    routing[id_col].astype(str).str.zfill(id_width),
                )
            )
            all_pairs = set(
                zip(
                    overlap_df["PourPtID"].astype(str),
                    overlap_df[id_col].astype(str).str.zfill(id_width),
                )
            )
            missing = all_pairs - covered
            if missing:
                print(
                    f"  NLDI: {len(missing)} (gauge, {level.upper()}) pairs without NHD "
                    "coverage; supplementing from BasinATLAS …"
                )
                ba = _load_routing_distances_watershed_filtered(level, id_col)
                ba_per_gauge = not ba.empty
                if ba.empty:
                    ba = _load_basinatlas_routing_distances(level, id_col)
                    ba_per_gauge = False
                if not ba.empty:
                    if ba_per_gauge:
                        ba["_key"] = (
                            ba["PourPtID"].astype(str)
                            + ":"
                            + ba[id_col].astype(str).str.zfill(id_width)
                        )
                        ba = ba[
                            ba["_key"].isin({f"{g}:{h}" for g, h in missing})
                        ].drop(columns=["_key"])
                    else:
                        missing_huc_ids = {h for _, h in missing}
                        ba = ba[
                            ba[id_col].astype(str).str.zfill(id_width).isin(missing_huc_ids)
                        ]
                    if not ba.empty:
                        routing = pd.concat([routing, ba], ignore_index=True)

    # ── Sources 2 & 3: BasinATLAS fallback ───────────────────────────────────
    if routing.empty:
        routing = _load_routing_distances_watershed_filtered(level, id_col)
        use_per_gauge = not routing.empty
        if not use_per_gauge:
            routing = _load_basinatlas_routing_distances(level, id_col)

    if routing.empty:
        return overlap_df

    out = overlap_df.copy()
    out[id_col] = out[id_col].astype(str).str.zfill(_id_width(level))

    if use_per_gauge:
        # Per-(gauge, HUC) merge — two-key join.
        # Align PourPtID to str so the merge works regardless of whether it
        # came in as int64 (shapefile) or str (CSV load).
        out["PourPtID"] = out["PourPtID"].astype(str)
        out = out.merge(routing, on=["PourPtID", id_col], how="left")
    else:
        # Legacy per-HUC merge — single key
        out = out.merge(routing, on=id_col, how="left")

    if "dist_sink_km" not in out.columns or out["dist_sink_km"].isna().all():
        return out

    outlet = out.groupby("PourPtID")["dist_sink_km"].transform("min")
    headwater = out.groupby("PourPtID")["dist_sink_km"].transform("max")
    out["outlet_dist_sink_km"] = outlet
    out["gauge_flow_length_km"] = (headwater - outlet).clip(lower=0.0).fillna(0.0)
    out["flow_length_to_gauge_km"] = (out["dist_sink_km"] - outlet).clip(lower=0.0).fillna(0.0)
    denom = out["gauge_flow_length_km"].where(out["gauge_flow_length_km"] > 0, pd.NA)
    out["relative_flow_length_to_gauge"] = (out["flow_length_to_gauge_km"] / denom).fillna(0.0)
    print(
        "  Added routing distance proxies: dist_sink_km, dist_main_km, "
        "flow_length_to_gauge_km, gauge_flow_length_km"
    )
    return out


def _add_inferred_topology_columns(overlap_df: pd.DataFrame, target_col: str, id_col: str) -> pd.DataFrame:
    """Infer basin-local downstream links from target-relative flow length.

    This creates a lightweight routing topology for differentiable network
    routing.  Each unit links to the nearest intersecting unit with a smaller
    ``flow_length_to_gauge_km``; units with no downstream candidate are gauge
    outlets.  It is intentionally conservative and can be replaced by an
    explicit hydrography graph later without changing the downstream schema.
    """
    out = overlap_df.copy()
    if "flow_length_to_gauge_km" not in out.columns:
        out["routing_downstream_huc12"] = ""
        out["routing_reach_length_km"] = 0.0
        out["routing_order"] = 0
        out["routing_is_outlet"] = 1
        return out

    out["routing_downstream_huc12"] = ""
    out["routing_reach_length_km"] = 0.0
    out["routing_order"] = 0
    out["routing_is_outlet"] = 1
    for _target, group in out.groupby(target_col, sort=False):
        lengths = pd.to_numeric(group["flow_length_to_gauge_km"], errors="coerce").fillna(0.0).clip(lower=0.0)
        unit_ids = group[id_col].astype(str).str.zfill(_id_width(id_col)).to_dict()
        ordered = sorted(group.index.tolist(), key=lambda idx: (-float(lengths.loc[idx]), unit_ids[idx]))
        for rank, idx in enumerate(ordered):
            out.at[idx, "routing_order"] = rank
        for idx in group.index:
            current = float(lengths.loc[idx])
            candidates = [j for j in group.index if float(lengths.loc[j]) < current - 1e-6]
            if not candidates:
                continue
            best_len = max(float(lengths.loc[j]) for j in candidates)
            tied = sorted(
                [j for j in candidates if abs(float(lengths.loc[j]) - best_len) <= 1e-6],
                key=lambda j: unit_ids[j],
            )
            downstream_idx = tied[0]
            out.at[idx, "routing_downstream_huc12"] = unit_ids[downstream_idx]
            out.at[idx, "routing_reach_length_km"] = max(current - float(lengths.loc[downstream_idx]), 0.0)
            out.at[idx, "routing_is_outlet"] = 0
    return out


def run_intersect(
    level: str = "huc12",
    gauge_min_fraction_of_subbasin: float = 0.70,
    use_nldi: bool = False,
) -> tuple[Path, Path, Path]:
    """Compute gauge × subbasin overlap areas and apply the filter rule.

    Parameters
    ----------
    level :
        WBD level to use: ``"huc10"`` or ``"huc12"``.
    gauge_min_fraction_of_subbasin :
        Minimum ratio ``A_gauge / A_subbasin`` for a gauge whose entire
        footprint sits inside a single subbasin.  Below this, the gauge
        is dropped.  Gauges that span two or more subbasins are kept
        unconditionally.

    Returns
    -------
    (overlap_csv, kept_gauges_csv, in_scope_csv)
    """
    level = level.lower()
    if level not in _LEVEL_GPKG:
        raise ValueError(f"level must be one of {sorted(_LEVEL_GPKG)}; got {level!r}")
    gpkg = _LEVEL_GPKG[level]
    level_upper = level.upper()
    id_col = level  # "huc10" or "huc12" — lower case in WBD layers
    area_col = f"{level}_area_km2"

    GEO_OPS_DIR.mkdir(parents=True, exist_ok=True)
    out_overlap  = GEO_OPS_DIR / f"{level_upper}_Intersect_Watersheds.csv"
    out_gauges   = GEO_OPS_DIR / f"{level_upper}_Kept_Gauges.csv"
    out_in_scope = GEO_OPS_DIR / f"{level_upper}_In_Scope.csv"

    print(f"Loading gauges     : {WATERSHEDS_GPKG.name}")
    gauges: gpd.GeoDataFrame = gpd.read_file(str(WATERSHEDS_GPKG))
    print(f"  {len(gauges)} gauges, CRS: {gauges.crs}")

    # Restrict to gauges that have a QA/QC-cleaned flow record in flow.zarr.
    # The watersheds gpkg contains the full combined pre-QA pool, so
    # without this filter the intersect would carry gauges that the LSTM
    # dataset later drops with a noisy warning, and `In_Scope.csv` would
    # include subbasins touched only by QA-failed gauges.
    qaqc_ids: set[int] = set()
    if FLOW_ZARR.exists():
        flow_dfs, _ = load_flow_dataframes(FLOW_ZARR)
        for bid in flow_dfs.keys():
            try:
                qaqc_ids.add(int(bid))
            except (TypeError, ValueError):
                continue
    if not qaqc_ids:
        raise RuntimeError(
            f"No QA/QC-cleaned flow records found in {FLOW_ZARR}; "
            f"run prepare_data.py steps 1\u20138 first."
        )
    before = len(gauges)
    gauges = gauges[gauges["PourPtID"].astype(int).isin(qaqc_ids)].copy()
    n_dropped_qa = before - len(gauges)
    if n_dropped_qa:
        print(f"  filtered to QA/QC pool: {len(gauges)} of {before} "
              f"({n_dropped_qa} pre-QA gauges dropped)")

    print(f"Loading {level_upper:<10}: {gpkg.name}")
    subs: gpd.GeoDataFrame = gpd.read_file(str(gpkg))
    print(f"  {len(subs)} {level_upper} polygons, CRS: {subs.crs}")

    gauge_id_col = "PourPtID"
    if gauge_id_col not in gauges.columns:
        raise RuntimeError(
            f"Expected '{gauge_id_col}' in watersheds gpkg; found {list(gauges.columns)}"
        )
    if id_col not in subs.columns:
        if level_upper in subs.columns:
            subs = subs.rename(columns={level_upper: id_col})
        else:
            raise RuntimeError(
                f"Expected '{id_col}' in {gpkg.name}; found {list(subs.columns)}"
            )

    for label, df in (("gauges", gauges), (level_upper, subs)):
        invalid = ~df.geometry.is_valid
        if invalid.any():
            print(f"  Fixing {invalid.sum()} invalid geometries in {label} …")
            df.geometry = shapely.make_valid(df.geometry.values)

    print(f"Reprojecting to {_PROJECTED_CRS} for equal-area overlap …")
    gauges_proj = gauges[[gauge_id_col, "geometry"]].to_crs(_PROJECTED_CRS)
    subs_proj = subs[[id_col, "geometry"]].to_crs(_PROJECTED_CRS)

    gauges_proj["gauge_area_km2"] = gauges_proj.geometry.area / 1e6
    subs_proj[area_col] = subs_proj.geometry.area / 1e6

    gauge_union_bbox = gauges_proj.total_bounds
    touch = subs_proj.cx[
        gauge_union_bbox[0]:gauge_union_bbox[2],
        gauge_union_bbox[1]:gauge_union_bbox[3],
    ]
    print(f"  Gauge bbox touches {len(touch)} {level_upper} polygons")

    print(f"Computing overlay (gauge ∩ {level}) …")
    overlay = gpd.overlay(
        gauges_proj, touch, how="intersection", keep_geom_type=False,
    )
    overlay["overlap_area_km2"] = overlay.geometry.area / 1e6
    print(f"  overlay produced {len(overlay)} pair rows")

    before = len(overlay)
    overlay = overlay[overlay["overlap_area_km2"] >= 1.0].copy()
    print(f"  dropped {before - len(overlay)} sliver rows (< 1 km²)")

    overlap_df = overlay[
        [gauge_id_col, id_col, "gauge_area_km2", area_col, "overlap_area_km2"]
    ].copy()
    overlap_df["frac_of_gauge"] = overlap_df["overlap_area_km2"] / overlap_df["gauge_area_km2"]
    overlap_df[f"frac_of_{level}"] = overlap_df["overlap_area_km2"] / overlap_df[area_col]
    overlap_df = _add_routing_distance_columns(overlap_df, level, id_col, use_nldi=use_nldi)

    # Apply filter rule per gauge:
    # drop only if the gauge overlaps exactly ONE subbasin AND the gauge
    # is smaller than threshold * subbasin_area.
    drop_reason: dict = {}
    kept_rows: list = []
    for pid, grp in overlap_df.groupby(gauge_id_col):
        grp_sorted = grp.sort_values("overlap_area_km2", ascending=False)
        n_overlaps = len(grp_sorted)
        max_frac_gauge = float(grp_sorted["frac_of_gauge"].iloc[0])
        gauge_area = float(grp_sorted["gauge_area_km2"].iloc[0])
        top_sub_area = float(grp_sorted[area_col].iloc[0])

        if (n_overlaps == 1
                and gauge_area < gauge_min_fraction_of_subbasin * top_sub_area):
            drop_reason[pid] = (
                f"single overlying {level_upper}; gauge_area={gauge_area:.1f} km² < "
                f"{gauge_min_fraction_of_subbasin:.0%} × {level_upper}_area={top_sub_area:.1f} km²"
            )
            continue
        kept_rows.append({
            "PourPtID": pid,
            f"n_{level}": n_overlaps,
            "total_overlap_km2": float(grp_sorted["overlap_area_km2"].sum()),
            "gauge_area_km2": gauge_area,
            "max_frac_of_gauge": max_frac_gauge,
            "gauge_flow_length_km": float(grp_sorted.get("gauge_flow_length_km", pd.Series([0.0])).iloc[0]),
        })

    kept_df = pd.DataFrame(kept_rows)
    kept_ids = set(kept_df["PourPtID"]) if len(kept_df) else set()

    all_summary_rows: list = []
    for pid, grp in overlap_df.groupby(gauge_id_col):
        grp_sorted = grp.sort_values("overlap_area_km2", ascending=False)
        all_summary_rows.append({
            "PourPtID": pid,
            f"n_{level}": len(grp_sorted),
            "total_overlap_km2": float(grp_sorted["overlap_area_km2"].sum()),
            "gauge_area_km2": float(grp_sorted["gauge_area_km2"].iloc[0]),
            "max_frac_of_gauge": float(grp_sorted["frac_of_gauge"].iloc[0]),
            "gauge_flow_length_km": float(grp_sorted.get("gauge_flow_length_km", pd.Series([0.0])).iloc[0]),
            "reason_dropped": drop_reason.get(pid, ""),
            "kept": pid in kept_ids,
        })
    summary_df = pd.DataFrame(all_summary_rows).sort_values("PourPtID")

    overlap_kept = overlap_df[overlap_df[gauge_id_col].isin(kept_ids)].copy()
    overlap_kept = overlap_kept.sort_values(
        [gauge_id_col, "overlap_area_km2"], ascending=[True, False],
    )
    overlap_kept = _add_inferred_topology_columns(overlap_kept, gauge_id_col, id_col)

    in_scope = (
        overlap_kept[[id_col, area_col]]
        .drop_duplicates(subset=[id_col])
        .sort_values(id_col)
        .reset_index(drop=True)
    )

    print(f"\nFilter summary:")
    print(f"  level               : {level_upper}")
    print(f"  input gauges        : {len(overlap_df[gauge_id_col].unique())}")
    print(f"  dropped (filter)    : {len(drop_reason)}")
    print(f"  kept gauges         : {len(kept_ids)}")
    print(f"  in-scope {level_upper:<6}     : {len(in_scope)}")
    if kept_df.empty:
        print("  WARNING: no gauges kept. Check filter parameters.")
    else:
        col = f"n_{level}"
        print(
            f"  {level_upper}s per gauge    : min={kept_df[col].min()}, "
            f"median={int(kept_df[col].median())}, "
            f"max={kept_df[col].max()}, "
            f"mean={kept_df[col].mean():.1f}"
        )

    print(f"\nWriting {out_overlap.name} ({len(overlap_kept)} rows)")
    overlap_kept.to_csv(out_overlap, index=False)
    print(f"Writing {out_gauges.name} ({len(summary_df)} rows)")
    summary_df.to_csv(out_gauges, index=False)
    print(f"Writing {out_in_scope.name} ({len(in_scope)} rows)")
    in_scope.to_csv(out_in_scope, index=False)

    return out_overlap, out_gauges, out_in_scope


def main(
    level: str = "huc12",
    gauge_min_fraction_of_subbasin: float = 0.70,
    use_nldi: bool = False,
) -> None:
    run_intersect(
        level=level,
        gauge_min_fraction_of_subbasin=gauge_min_fraction_of_subbasin,
        use_nldi=use_nldi,
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Build subbasin × gauge-basin intersection for HUC-based workflows."
    )
    ap.add_argument("--level", choices=sorted(_LEVEL_GPKG), default="huc12",
                    help="WBD level to intersect against (default: huc12).")
    ap.add_argument("--gauge-min-fraction-of-subbasin", type=float, default=0.70,
                    help="Drop a gauge that sits inside exactly one subbasin when "
                         "its area is less than this fraction of the subbasin area "
                         "(default: 0.70). Gauges spanning multiple subbasins are "
                         "always kept.")
    ap.add_argument("--use-nldi", action="store_true", default=False,
                    help="Use NLDI + NHDPlus WaterData for true along-network routing "
                         "distances (requires pynhd; falls back to BasinATLAS for any "
                         "gauge/unit pairs not covered). Results cached per gauge under "
                         "data/prepare/geo_ops/nldi_cache/.")
    args = ap.parse_args()
    main(
        level=args.level,
        gauge_min_fraction_of_subbasin=args.gauge_min_fraction_of_subbasin,
        use_nldi=args.use_nldi,
    )
