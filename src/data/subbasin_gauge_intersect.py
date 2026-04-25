"""Build the subbasin ↔ gauge-basin intersection table for subbasin-mode training.

Supports any Watershed Boundary Dataset level (default ``huc12``).  For
every USGS training watershed, compute the set of subbasin polygons that
overlap it and the area of each overlap (km², in an equal-area CRS).
Apply the gauge-filter rule ("gauge too small to be a subbasin") and
write level-tagged CSVs to ``data/prepare/geo_ops/``:

    {LEVEL}_Intersect_Watersheds.csv
        columns: PourPtID, <level>, gauge_area_km2, <level>_area_km2,
                 overlap_area_km2, frac_of_gauge, frac_of_<level>

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
    FLOW_ZARR,
    GEO_OPS_DIR,
    WATERSHEDS_GPKG,
    WBDHU10_GPKG,
    WBDHU12_GPKG,
)

# Equal-area CRS (metres) for accurate area/overlap math.
_PROJECTED_CRS = "EPSG:6414"

_LEVEL_GPKG = {
    "huc10": WBDHU10_GPKG,
    "huc12": WBDHU12_GPKG,
}


def run_intersect(
    level: str = "huc12",
    gauge_min_fraction_of_subbasin: float = 0.70,
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

    # Restrict to gauges that have a QA/QC-cleaned flow file in any tier.
    # The watersheds gpkg contains the full pre-QA pool (~219 polygons), so
    # without this filter the intersect would carry gauges that the LSTM
    # dataset later drops with a noisy warning, and `In_Scope.csv` would
    # include subbasins touched only by QA-failed gauges.
    # Restrict to gauges that have a QA/QC-cleaned flow record in flow.zarr.
    # The watersheds gpkg contains the full pre-QA pool (~219 polygons), so
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
            "reason_dropped": drop_reason.get(pid, ""),
            "kept": pid in kept_ids,
        })
    summary_df = pd.DataFrame(all_summary_rows).sort_values("PourPtID")

    overlap_kept = overlap_df[overlap_df[gauge_id_col].isin(kept_ids)].copy()
    overlap_kept = overlap_kept.sort_values(
        [gauge_id_col, "overlap_area_km2"], ascending=[True, False],
    )

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
) -> None:
    run_intersect(
        level=level,
        gauge_min_fraction_of_subbasin=gauge_min_fraction_of_subbasin,
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Build subbasin × gauge-basin intersection for subbasin-mode training."
    )
    ap.add_argument("--level", choices=sorted(_LEVEL_GPKG), default="huc12",
                    help="WBD level to intersect against (default: huc12).")
    ap.add_argument("--gauge-min-fraction-of-subbasin", type=float, default=0.70,
                    help="Drop a gauge that sits inside exactly one subbasin when "
                         "its area is less than this fraction of the subbasin area "
                         "(default: 0.70). Gauges spanning multiple subbasins are "
                         "always kept.")
    args = ap.parse_args()
    main(
        level=args.level,
        gauge_min_fraction_of_subbasin=args.gauge_min_fraction_of_subbasin,
    )
