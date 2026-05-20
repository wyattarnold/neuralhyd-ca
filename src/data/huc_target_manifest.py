"""Build HUC12-to-HUC10/HUC8 target manifests for dPL simulation.

The dPL model learns local HUC12 runoff generation and then aggregates/routs
HUC12 outputs to a requested outlet.  Gauge training uses
``HUC12_Intersect_Watersheds.csv``; this module builds the analogous full-domain
manifests for WBD HUC10 and HUC8 target outlets.

Outputs land in ``data/prepare/geo_ops/``:

``HUC12_Intersect_HUC10.csv`` / ``HUC12_Intersect_HUC8.csv``
    One row per target-HUC12 overlap, with area weights and target-relative
    flow-distance proxies.

``HUC10_Targets.csv`` / ``HUC8_Targets.csv``
    One row per target with coverage and flow-length summary fields.

``HUC12_In_Scope_HUC10.csv`` / ``HUC12_In_Scope_HUC8.csv``
    Unique HUC12 IDs needed for each target level.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import pandas as pd

from src.paths import GEO_OPS_DIR, WBDHU8_GPKG, WBDHU10_GPKG, WBDHU12_GPKG

_PROJECTED_CRS = "EPSG:6414"

_TARGET_GPKG: dict[str, Path] = {
    "huc8": WBDHU8_GPKG,
    "huc10": WBDHU10_GPKG,
}

_ID_WIDTH: dict[str, int] = {
    "huc8": 8,
    "huc10": 10,
    "huc12": 12,
}


def _find_id_col(gdf: gpd.GeoDataFrame, level: str) -> str:
    match = next((col for col in gdf.columns if col.lower() == level), None)
    if match is None:
        raise KeyError(f"Expected {level!r} id column in {list(gdf.columns)[:20]}")
    return match


def _standardize_id(series: pd.Series, level: str) -> pd.Series:
    return series.astype(str).str.strip().str.zfill(_ID_WIDTH[level])


def _load_huc12_routing_distances() -> pd.DataFrame:
    """Return area-weighted BasinATLAS distance attributes by HUC12."""
    source = GEO_OPS_DIR / "BasinATLAS_v10_lev12_Intersect_HUC12.csv"
    if not source.exists():
        raise FileNotFoundError(
            f"HUC12 BasinATLAS intersect not found: {source}. Run "
            "scripts/prepare_data.py --geo-intersect --geo static --target huc12 first."
        )

    header = pd.read_csv(source, nrows=0).columns
    source_id_col = "PourPtID" if "PourPtID" in header else "huc12"
    needed = [source_id_col, "Shape_Area", "DIST_SINK", "DIST_MAIN"]
    missing = [col for col in needed if col not in header]
    if missing:
        raise KeyError(f"{source.name} is missing required columns: {missing}")

    df = pd.read_csv(source, usecols=needed, dtype={source_id_col: str})
    df[source_id_col] = _standardize_id(df[source_id_col], "huc12")
    df["Shape_Area"] = pd.to_numeric(df["Shape_Area"], errors="coerce").clip(lower=0)
    for col in ("DIST_SINK", "DIST_MAIN"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df.loc[df[col] == -9999, col] = pd.NA

    rows: list[dict[str, float | str]] = []
    for huc12, group in df.groupby(source_id_col):
        weights = group["Shape_Area"].fillna(0.0).to_numpy(dtype=float)
        out: dict[str, float | str] = {"huc12": str(huc12).zfill(_ID_WIDTH["huc12"])}
        for source_col, out_col in (("DIST_SINK", "dist_sink_km"), ("DIST_MAIN", "dist_main_km")):
            values = group[source_col].to_numpy(dtype=float)
            valid = pd.notna(values) & (weights > 0)
            out[out_col] = (
                float((values[valid] * weights[valid]).sum() / weights[valid].sum())
                if valid.any()
                else float("nan")
            )
        rows.append(out)
    return pd.DataFrame(rows)


def _add_inferred_topology_columns(manifest: pd.DataFrame, target_level: str) -> pd.DataFrame:
    """Infer target-local downstream HUC12 links from flow length to target."""
    out = manifest.copy()
    out["routing_downstream_huc12"] = ""
    out["routing_reach_length_km"] = 0.0
    out["routing_order"] = 0
    out["routing_is_outlet"] = 1
    if "flow_length_to_target_km" not in out.columns:
        return out

    for _target, group in out.groupby(target_level, sort=False):
        lengths = pd.to_numeric(group["flow_length_to_target_km"], errors="coerce").fillna(0.0).clip(lower=0.0)
        huc_ids = group["huc12"].astype(str).str.zfill(_ID_WIDTH["huc12"]).to_dict()
        ordered = sorted(group.index.tolist(), key=lambda idx: (-float(lengths.loc[idx]), huc_ids[idx]))
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
                key=lambda j: huc_ids[j],
            )
            downstream_idx = tied[0]
            out.at[idx, "routing_downstream_huc12"] = huc_ids[downstream_idx]
            out.at[idx, "routing_reach_length_km"] = max(current - float(lengths.loc[downstream_idx]), 0.0)
            out.at[idx, "routing_is_outlet"] = 0
    return out


def build_manifest(target_level: str, min_overlap_km2: float = 0.1) -> tuple[Path, Path, Path]:
    """Build one HUC12-to-target manifest.

    Parameters
    ----------
    target_level:
        ``"huc10"`` or ``"huc8"``.
    min_overlap_km2:
        Small overlay fragments below this area are dropped.

    Returns
    -------
    ``(manifest_csv, summary_csv, in_scope_csv)``.
    """
    target_level = target_level.lower()
    if target_level not in _TARGET_GPKG:
        raise ValueError(f"target_level must be one of {sorted(_TARGET_GPKG)}; got {target_level!r}")

    target_upper = target_level.upper()
    GEO_OPS_DIR.mkdir(parents=True, exist_ok=True)
    out_manifest = GEO_OPS_DIR / f"HUC12_Intersect_{target_upper}.csv"
    out_summary = GEO_OPS_DIR / f"{target_upper}_Targets.csv"
    out_in_scope = GEO_OPS_DIR / f"HUC12_In_Scope_{target_upper}.csv"

    print(f"Loading target {target_upper:<5}: {_TARGET_GPKG[target_level].name}")
    targets = gpd.read_file(_TARGET_GPKG[target_level])
    target_id_col = _find_id_col(targets, target_level)
    print(f"  {len(targets)} {target_upper} polygons, CRS: {targets.crs}")

    print(f"Loading source HUC12: {WBDHU12_GPKG.name}")
    huc12 = gpd.read_file(WBDHU12_GPKG)
    huc12_id_col = _find_id_col(huc12, "huc12")
    print(f"  {len(huc12)} HUC12 polygons, CRS: {huc12.crs}")

    targets = targets[[target_id_col, "geometry"]].rename(columns={target_id_col: target_level})
    huc12 = huc12[[huc12_id_col, "geometry"]].rename(columns={huc12_id_col: "huc12"})
    targets[target_level] = _standardize_id(targets[target_level], target_level)
    huc12["huc12"] = _standardize_id(huc12["huc12"], "huc12")

    print(f"Reprojecting to {_PROJECTED_CRS} for equal-area areas ...")
    targets_proj = targets.to_crs(_PROJECTED_CRS).copy()
    huc12_proj = huc12.to_crs(_PROJECTED_CRS).copy()
    target_area_col = f"{target_level}_area_km2"
    targets_proj[target_area_col] = targets_proj.geometry.area / 1e6
    huc12_proj["huc12_area_km2"] = huc12_proj.geometry.area / 1e6

    # WBD IDs are hierarchical: a HUC12's first 10 digits are its HUC10, and
    # first 8 digits are its HUC8.  Using the prefix avoids a large polygon
    # overlay and is the intended relationship for nested WBD levels.
    huc12_proj[target_level] = huc12_proj["huc12"].str[:_ID_WIDTH[target_level]]
    target_areas = targets_proj[[target_level, target_area_col]].drop_duplicates(target_level)
    huc12_areas = huc12_proj[[target_level, "huc12", "huc12_area_km2"]].copy()
    manifest = huc12_areas.merge(target_areas, on=target_level, how="inner")
    manifest["overlap_area_km2"] = manifest["huc12_area_km2"]
    before = len(manifest)
    manifest = manifest[manifest["overlap_area_km2"] >= min_overlap_km2].copy()
    print(
        f"  prefix membership produced {before} target-HUC12 rows; "
        f"dropped {before - len(manifest)} tiny HUC12 rows (< {min_overlap_km2:g} km2)"
    )
    manifest["frac_of_target"] = manifest["overlap_area_km2"] / manifest[target_area_col]
    manifest["frac_of_huc12"] = manifest["overlap_area_km2"] / manifest["huc12_area_km2"]
    manifest["area_weight"] = manifest["frac_of_target"]

    routing = _load_huc12_routing_distances()
    manifest = manifest.merge(routing, on="huc12", how="left")
    if manifest["dist_sink_km"].isna().all():
        raise RuntimeError("All HUC12 dist_sink_km values are missing after routing-distance merge")

    outlet = manifest.groupby(target_level)["dist_sink_km"].transform("min")
    headwater = manifest.groupby(target_level)["dist_sink_km"].transform("max")
    manifest["target_outlet_dist_sink_km"] = outlet
    manifest["target_flow_length_km"] = (headwater - outlet).clip(lower=0.0).fillna(0.0)
    manifest["flow_length_to_target_km"] = (manifest["dist_sink_km"] - outlet).clip(lower=0.0).fillna(0.0)
    denom = manifest["target_flow_length_km"].where(manifest["target_flow_length_km"] > 0, pd.NA)
    manifest["relative_flow_length_to_target"] = (manifest["flow_length_to_target_km"] / denom).fillna(0.0)

    # Compatibility aliases for DplDataset-style loaders that currently expect
    # gauge-oriented names. These are target-relative values in this manifest.
    manifest["PourPtID"] = manifest[target_level]
    manifest["gauge_area_km2"] = manifest[target_area_col]
    manifest["frac_of_gauge"] = manifest["frac_of_target"]
    manifest["gauge_flow_length_km"] = manifest["target_flow_length_km"]
    manifest["outlet_dist_sink_km"] = manifest["target_outlet_dist_sink_km"]
    manifest["flow_length_to_gauge_km"] = manifest["flow_length_to_target_km"]
    manifest["relative_flow_length_to_gauge"] = manifest["relative_flow_length_to_target"]
    manifest = _add_inferred_topology_columns(manifest, target_level)

    ordered = [
        target_level,
        "PourPtID",
        "huc12",
        target_area_col,
        "gauge_area_km2",
        "huc12_area_km2",
        "overlap_area_km2",
        "area_weight",
        "frac_of_target",
        "frac_of_gauge",
        "frac_of_huc12",
        "dist_sink_km",
        "dist_main_km",
        "target_outlet_dist_sink_km",
        "outlet_dist_sink_km",
        "target_flow_length_km",
        "gauge_flow_length_km",
        "flow_length_to_target_km",
        "flow_length_to_gauge_km",
        "relative_flow_length_to_target",
        "relative_flow_length_to_gauge",
        "routing_downstream_huc12",
        "routing_reach_length_km",
        "routing_order",
        "routing_is_outlet",
    ]
    manifest = manifest[ordered].sort_values([target_level, "huc12"]).reset_index(drop=True)

    summary = (
        manifest.groupby(target_level)
        .agg(
            n_huc12=("huc12", "count"),
            total_overlap_km2=("overlap_area_km2", "sum"),
            target_area_km2=(target_area_col, "first"),
            target_flow_length_km=("target_flow_length_km", "first"),
            outlet_dist_sink_km=("target_outlet_dist_sink_km", "first"),
            max_flow_length_to_target_km=("flow_length_to_target_km", "max"),
        )
        .reset_index()
    )
    summary["coverage_ratio"] = summary["total_overlap_km2"] / summary["target_area_km2"]
    summary = summary.sort_values(target_level).reset_index(drop=True)

    in_scope = (
        manifest[["huc12", "huc12_area_km2"]]
        .drop_duplicates(subset=["huc12"])
        .sort_values("huc12")
        .reset_index(drop=True)
    )

    print(f"Writing {out_manifest.name} ({len(manifest)} rows)")
    manifest.to_csv(out_manifest, index=False)
    print(f"Writing {out_summary.name} ({len(summary)} rows)")
    summary.to_csv(out_summary, index=False)
    print(f"Writing {out_in_scope.name} ({len(in_scope)} rows)")
    in_scope.to_csv(out_in_scope, index=False)
    print(
        f"  {target_upper}: targets={len(summary)} HUC12 rows={len(manifest)} "
        f"coverage median={summary['coverage_ratio'].median():.3f} "
        f"min={summary['coverage_ratio'].min():.3f} max={summary['coverage_ratio'].max():.3f}"
    )
    return out_manifest, out_summary, out_in_scope


def main(target_level: str = "all", min_overlap_km2: float = 0.1) -> None:
    target_level = target_level.lower()
    levels = ["huc10", "huc8"] if target_level == "all" else [target_level]
    for level in levels:
        build_manifest(level, min_overlap_km2=min_overlap_km2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build HUC12-to-HUC10/HUC8 manifests")
    parser.add_argument("--target-level", default="all", choices=["all", "huc10", "huc8"])
    parser.add_argument("--min-overlap-km2", type=float, default=0.1)
    args = parser.parse_args()
    main(target_level=args.target_level, min_overlap_km2=args.min_overlap_km2)