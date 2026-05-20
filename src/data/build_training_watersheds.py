"""Build the combined USGS + CDEC training watershed catalog.

This is the earliest preparation step for gauge-mode training data.  It
materialises a single ``watersheds`` target layer from the original USGS
training watershed GeoPackage plus the CDEC full-natural-flow watershed
GeoPackage, then canonicalises CDEC daily FNF files into the raw-flow CSV
shape consumed by the existing cleaning/QA pipeline.
"""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd

from src.paths import (
    CDEC_DAILY_FNF_DIR,
    CDEC_FNF_GPKG,
    CDEC_FNF_KEY,
    CDEC_FNF_SHAPEFILE_DIR,
    RAW_USGS_DIR,
    USGS_WATERSHEDS_GPKG,
    WATERSHED_GEOJSON,
    WATERSHED_GEOMETRY,
    WATERSHEDS_DIR,
    WATERSHEDS_GPKG,
)

PROJECTED_CRS = "EPSG:6414"
WATERSHEDS_LAYER = "Training_Watersheds"

# Fallback crosswalk for legacy FNF_key.csv files without a GEO_ID column.
CDEC_FNF_TO_DESCRIPTION: dict[str, str] = {
    "FOL": "FOL_I",
    "MIL": "MILLE",
    "MKM": "PRD_C",
    "MRC": "LK_MC",
    "NHG": "N_HOG",
    "NML": "N_MEL",
    "ORO": "OROVI",
    "PNF": "KINGS",
    "ISB": "ISB",
    "SCC": "SCC",
    "SHA": "SHAST",
    "TLG": "DPR_I",
    "TRM": "TRM",
    "YRS": "SMART",
}

CDEC_FLOW_CATALOG = CDEC_DAILY_FNF_DIR / "cdec_daily_fnf_catalog.csv"
CDEC_LEGACY_STAGING_REPORT = CDEC_DAILY_FNF_DIR / "staged_to_raw_flow.csv"
CDEC_MAPPING_AUDIT = CDEC_DAILY_FNF_DIR / "fnf_mapping_audit.csv"


def _standardise_raw_columns(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Return raw-target columns expected by the GIS intersect step."""
    out = gdf.copy()
    rename = {
        "Pour Point ID": "PourPtID",
        "Data Resolution": "DataResolution",
        "Area Square Kilometers": "AreaSqKm",
    }
    out.rename(columns={k: v for k, v in rename.items() if k in out.columns}, inplace=True)
    keep = [
        "PourPtID",
        "Description",
        "DataResolution",
        "AreaSqKm",
        "Shape_Length",
        "Shape_Area",
        "geometry",
    ]
    missing = [col for col in keep if col not in out.columns]
    if missing:
        raise KeyError(f"Watershed layer is missing required columns: {missing}")
    out = out[keep].copy()
    out["PourPtID"] = out["PourPtID"].astype("int64")
    return gpd.GeoDataFrame(out, geometry="geometry", crs=gdf.crs)


def _load_cdec_key() -> pd.DataFrame:
    if not CDEC_FNF_KEY.exists():
        return pd.DataFrame(
            columns=[
                "FNF",
                "key_geo_id",
                "key_cdec_label",
                "key_id",
                "key_pourptid",
                "key_area_mi2",
                "key_area_km2",
            ]
        )

    key_df = pd.read_csv(
        CDEC_FNF_KEY,
        dtype={"FNF": str, "GEO_ID": str, "CDEC_LABEL": str, "PourPtID": str},
    )
    key_df["FNF"] = key_df["FNF"].astype(str).str.strip().str.upper()
    key_df["key_geo_id"] = key_df["GEO_ID"].fillna("").astype(str).str.strip() if "GEO_ID" in key_df else ""
    key_df["key_cdec_label"] = key_df["CDEC_LABEL"].fillna("") if "CDEC_LABEL" in key_df else ""
    key_id = key_df["ID"] if "ID" in key_df else pd.Series(pd.NA, index=key_df.index)
    key_area = key_df["AREA_MI2"] if "AREA_MI2" in key_df else pd.Series(pd.NA, index=key_df.index)
    key_df["key_id"] = pd.to_numeric(key_id, errors="coerce").astype("Int64")
    expected_pourptid = _expected_pourptid(key_df["key_id"])
    if "PourPtID" in key_df:
        key_pourptid = pd.to_numeric(key_df["PourPtID"], errors="coerce").astype("Int64")
        key_df["key_pourptid"] = key_pourptid.fillna(expected_pourptid)
    else:
        key_df["key_pourptid"] = expected_pourptid
    key_df["key_area_mi2"] = pd.to_numeric(key_area, errors="coerce")
    key_df["key_area_km2"] = key_df["key_area_mi2"] * 2.58999
    return key_df[
        [
            "FNF",
            "key_geo_id",
            "key_cdec_label",
            "key_id",
            "key_pourptid",
            "key_area_mi2",
            "key_area_km2",
        ]
    ].copy()


def _expected_pourptid(ids: pd.Series) -> pd.Series:
    key_id = pd.to_numeric(ids, errors="coerce").astype("Int64")
    return key_id.apply(
        lambda value: pd.NA if pd.isna(value) else 990000000 + int(value)
    ).astype("Int64")


def ensure_cdec_key_pourptid() -> pd.DataFrame:
    """Add/validate the synthetic PourPtID column in FNF_key.csv."""
    if not CDEC_FNF_KEY.exists():
        raise FileNotFoundError(f"CDEC FNF key not found: {CDEC_FNF_KEY}")

    raw = pd.read_csv(CDEC_FNF_KEY, dtype=str)
    if "ID" not in raw.columns:
        raise ValueError("FNF_key.csv is missing required ID column")
    expected = _expected_pourptid(raw["ID"])
    if expected.isna().any():
        missing_codes = raw.loc[expected.isna(), "FNF"].tolist() if "FNF" in raw else []
        raise ValueError(f"FNF_key.csv has invalid ID values for: {missing_codes}")

    changed = False
    if "PourPtID" in raw.columns:
        existing = pd.to_numeric(raw["PourPtID"], errors="coerce").astype("Int64")
        mismatch = existing.notna() & (existing != expected)
        if mismatch.any():
            bad = raw.loc[mismatch, [col for col in ("FNF", "ID", "PourPtID") if col in raw.columns]]
            raise ValueError(f"FNF_key.csv has mismatched PourPtID values:\n{bad}")
        filled = existing.fillna(expected).astype("Int64")
        changed = not existing.equals(filled)
        raw["PourPtID"] = filled.astype(str)
    else:
        insert_at = raw.columns.get_loc("ID") + 1
        raw.insert(insert_at, "PourPtID", expected.astype(str))
        changed = True

    if changed:
        raw.to_csv(CDEC_FNF_KEY, index=False)
    return _load_cdec_key()


def build_cdec_fnf_geopackage(shapefile_dir: Path = CDEC_FNF_SHAPEFILE_DIR) -> gpd.GeoDataFrame:
    """Build data/raw/gis/cdec_fnf.gpkg from FNF_key.csv and watershed shapefiles."""
    key_df = ensure_cdec_key_pourptid()
    if key_df.empty:
        raise FileNotFoundError(f"CDEC FNF key not found or empty: {CDEC_FNF_KEY}")
    if key_df["key_geo_id"].eq("").any():
        missing_codes = key_df.loc[key_df["key_geo_id"].eq(""), "FNF"].tolist()
        raise ValueError(f"FNF_key.csv is missing GEO_ID values for: {missing_codes}")
    if key_df["key_id"].isna().any():
        missing_codes = key_df.loc[key_df["key_id"].isna(), "FNF"].tolist()
        raise ValueError(f"FNF_key.csv is missing numeric ID values for: {missing_codes}")
    if key_df["key_pourptid"].isna().any():
        missing_codes = key_df.loc[key_df["key_pourptid"].isna(), "FNF"].tolist()
        raise ValueError(f"FNF_key.csv is missing PourPtID values for: {missing_codes}")

    missing_shapes = [
        shapefile_dir / f"{row.key_geo_id}.shp"
        for row in key_df.itertuples(index=False)
        if not (shapefile_dir / f"{row.key_geo_id}.shp").exists()
    ]
    if missing_shapes:
        raise FileNotFoundError(
            "Missing CDEC watershed shapefiles:\n" + "\n".join(str(path) for path in missing_shapes)
        )

    records = []
    for row in key_df.itertuples(index=False):
        path = shapefile_dir / f"{row.key_geo_id}.shp"
        shp = gpd.read_file(path)
        if shp.empty:
            raise ValueError(f"CDEC watershed shapefile has no features: {path}")
        if shp.crs is None:
            shp = shp.set_crs("EPSG:4326")
        shp = shp.to_crs("EPSG:4326")
        geometry = shp.geometry.unary_union
        records.append({
            "PourPtID": int(row.key_pourptid),
            "Description": str(row.key_geo_id),
            "DataResolution": "",
            "AreaSqKm": float(row.key_area_km2),
            "Shape_Length": geometry.length,
            "Shape_Area": geometry.area,
            "FNF": row.FNF,
            "CDEC_LABEL": row.key_cdec_label,
            "CDEC_ID": int(row.key_id),
            "AREA_MI2": float(row.key_area_mi2),
            "geometry": geometry,
        })

    cdec = gpd.GeoDataFrame(records, geometry="geometry", crs="EPSG:4326")
    if cdec["PourPtID"].duplicated().any():
        dupes = cdec.loc[cdec["PourPtID"].duplicated(), "PourPtID"].tolist()
        raise ValueError(f"Duplicate CDEC synthetic PourPtID values: {dupes}")

    CDEC_FNF_GPKG.parent.mkdir(parents=True, exist_ok=True)
    if CDEC_FNF_GPKG.exists():
        CDEC_FNF_GPKG.unlink()
    cdec.to_file(CDEC_FNF_GPKG, layer="cdec_fnf", driver="GPKG")
    return cdec


def _export_training_watershed_files(combined_wgs84: gpd.GeoDataFrame) -> None:
    """Write combined GeoPackage, CSV, and GeoJSON watershed products."""
    WATERSHEDS_GPKG.parent.mkdir(parents=True, exist_ok=True)
    WATERSHEDS_DIR.mkdir(parents=True, exist_ok=True)

    raw_cols = [
        "PourPtID",
        "Description",
        "DataResolution",
        "AreaSqKm",
        "Shape_Length",
        "Shape_Area",
        "geometry",
    ]
    if WATERSHEDS_GPKG.exists():
        WATERSHEDS_GPKG.unlink()
    combined_wgs84[raw_cols].to_file(
        WATERSHEDS_GPKG,
        layer=WATERSHEDS_LAYER,
        driver="GPKG",
    )

    projected = combined_wgs84.to_crs(PROJECTED_CRS).copy()
    projected["OBJECTID"] = range(1, len(projected) + 1)
    projected["Pour Point ID"] = projected["PourPtID"].astype("int64")
    projected["Data Resolution"] = projected["DataResolution"]
    projected["Area Square Kilometers"] = projected["AreaSqKm"].astype(float)
    projected["Shape_Length"] = projected.geometry.length
    projected["Shape_Area"] = projected.geometry.area

    export_cols = [
        "OBJECTID",
        "Pour Point ID",
        "Description",
        "Data Resolution",
        "Area Square Kilometers",
        "Shape_Length",
        "Shape_Area",
    ]
    projected[export_cols].to_csv(WATERSHED_GEOMETRY, index=False)
    projected[export_cols + ["geometry"]].to_file(WATERSHED_GEOJSON, driver="GeoJSON")


def build_training_watershed_catalog(include_cdec: bool = True) -> gpd.GeoDataFrame:
    """Build and export the combined USGS + CDEC watershed target layer."""
    if not USGS_WATERSHEDS_GPKG.exists():
        raise FileNotFoundError(f"USGS watershed GeoPackage not found: {USGS_WATERSHEDS_GPKG}")

    usgs = _standardise_raw_columns(gpd.read_file(USGS_WATERSHEDS_GPKG))
    if not include_cdec:
        combined = usgs.reset_index(drop=True)
        _export_training_watershed_files(combined)
        return combined

    build_cdec_fnf_geopackage()

    cdec = _standardise_raw_columns(gpd.read_file(CDEC_FNF_GPKG, layer="cdec_fnf"))
    cdec["Description"] = "CDEC FNF " + cdec["Description"].astype(str)

    common_crs = usgs.crs or "EPSG:4326"
    usgs = usgs.to_crs(common_crs)
    cdec = cdec.to_crs(common_crs)
    combined = gpd.GeoDataFrame(
        pd.concat([usgs, cdec], ignore_index=True),
        geometry="geometry",
        crs=common_crs,
    )
    if combined["PourPtID"].duplicated().any():
        dupes = combined.loc[combined["PourPtID"].duplicated(), "PourPtID"].tolist()
        raise ValueError(f"Duplicate PourPtID values in combined watersheds: {dupes}")

    combined = combined.reset_index(drop=True)
    _export_training_watershed_files(combined)
    return combined


def _load_cdec_flow_mapping(cdec: gpd.GeoDataFrame) -> tuple[dict[str, int], pd.DataFrame]:
    """Return FNF code -> PourPtID and a reporting DataFrame."""
    key_df = ensure_cdec_key_pourptid()
    cdec_by_desc = cdec.set_index(cdec["Description"].astype(str), drop=False)
    mapping: dict[str, int] = {}
    rows: list[dict[str, object]] = []

    key_by_code = key_df.set_index("FNF", drop=False).to_dict("index") if not key_df.empty else {}
    codes = sorted(set(CDEC_FNF_TO_DESCRIPTION) | set(key_by_code))

    for code in codes:
        key_row = key_by_code.get(code, {})
        report_row: dict[str, object] = {
            "FNF": code,
            "key_geo_id": key_row.get("key_geo_id", ""),
            "key_cdec_label": key_row.get("key_cdec_label", ""),
            "key_id": key_row.get("key_id", pd.NA),
            "key_pourptid": key_row.get("key_pourptid", pd.NA),
            "key_area_mi2": key_row.get("key_area_mi2", pd.NA),
            "key_area_km2": key_row.get("key_area_km2", pd.NA),
            "PourPtID": "",
            "gpkg_description": "",
            "gpkg_area_km2": pd.NA,
            "area_diff_pct": pd.NA,
        }

        description = key_row.get("key_geo_id")
        mapping_source = "mapped_from_key_geo_id"
        if not description:
            description = CDEC_FNF_TO_DESCRIPTION.get(code)
            mapping_source = "mapped_from_fallback_crosswalk"
        if description is None:
            report_row["mapping_status"] = "no_gpkg_crosswalk"
        elif description in cdec_by_desc.index:
            cdec_row = cdec_by_desc.loc[description]
            pid = int(cdec_row["PourPtID"])
            mapping[code] = pid
            gpkg_area = float(cdec_row["AreaSqKm"])
            key_area = report_row["key_area_km2"]
            if pd.notna(key_area) and float(key_area) > 0:
                report_row["area_diff_pct"] = abs(gpkg_area - float(key_area)) / float(key_area) * 100.0
            report_row.update({
                "PourPtID": pid,
                "gpkg_description": description,
                "gpkg_area_km2": gpkg_area,
                "mapping_status": mapping_source,
            })
        else:
            report_row.update({
                "gpkg_description": description,
                "mapping_status": "crosswalk_description_not_in_gpkg",
            })
        rows.append(report_row)
    return mapping, pd.DataFrame(rows)


def remove_staged_cdec_fnf_flows(remove_reports: bool = True) -> list[Path]:
    """Remove legacy generated synthetic CDEC flow CSVs from RAW_USGS_DIR."""
    removed: list[Path] = []
    for path in sorted(RAW_USGS_DIR.glob("990000*.csv")):
        path.unlink()
        removed.append(path)
    if remove_reports:
        for path in (CDEC_FLOW_CATALOG, CDEC_LEGACY_STAGING_REPORT, CDEC_MAPPING_AUDIT):
            if path.exists():
                path.unlink()
    return removed


def _iter_cdec_fnf_text_files() -> list[Path]:
    paths = list(CDEC_DAILY_FNF_DIR.glob("FNF_*_cfs.txt"))
    paths.extend(CDEC_DAILY_FNF_DIR.glob("FNF_*_cfs.text"))
    return sorted(set(paths))


def _read_cdec_fnf_file(path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=["year", "month", "day", "flow", "date"],
        dtype={"year": "Int64", "month": "Int64", "day": "Int64", "date": str},
    )
    dates = pd.to_datetime(df["date"], errors="coerce")
    out = pd.DataFrame({
        "datetime": dates.dt.strftime("%Y-%m-%d 00:00:00"),
        "00060_Mean": pd.to_numeric(df["flow"], errors="coerce"),
        "00060_Mean_cd": pd.NA,
    })
    return out.dropna(subset=["datetime"])


def stage_cdec_fnf_flows() -> pd.DataFrame:
    """Convert CDEC daily FNF text files into canonical raw CDEC CSVs."""
    if not CDEC_DAILY_FNF_DIR.exists():
        raise FileNotFoundError(f"CDEC daily FNF directory not found: {CDEC_DAILY_FNF_DIR}")
    if not CDEC_FNF_GPKG.exists():
        raise FileNotFoundError(f"CDEC FNF watershed GeoPackage not found: {CDEC_FNF_GPKG}")

    cdec = _standardise_raw_columns(gpd.read_file(CDEC_FNF_GPKG, layer="cdec_fnf"))
    flow_mapping, mapping_report = _load_cdec_flow_mapping(cdec)
    CDEC_DAILY_FNF_DIR.mkdir(parents=True, exist_ok=True)
    mapping_report.to_csv(CDEC_MAPPING_AUDIT, index=False)

    text_by_code = {
        path.stem.removeprefix("FNF_").removesuffix("_cfs").upper(): path
        for path in _iter_cdec_fnf_text_files()
    }
    report_rows = []
    for code, pid in sorted(flow_mapping.items()):
        text_path = text_by_code.get(code)
        out_path = CDEC_DAILY_FNF_DIR / f"{pid}.csv"
        legacy_path = RAW_USGS_DIR / f"{pid}.csv"
        status = "missing_source"

        if text_path is not None:
            out_df = _read_cdec_fnf_file(text_path)
            out_df.to_csv(out_path, index=False)
            text_path.unlink()
            status = "converted_from_legacy_text"
        elif out_path.exists():
            out_df = pd.read_csv(out_path)
            status = "exists"
        elif legacy_path.exists():
            out_df = pd.read_csv(legacy_path)
            out_df.to_csv(out_path, index=False)
            status = "migrated_from_raw_usgs"
        else:
            out_df = pd.DataFrame()

        report_rows.append({
            "FNF": code,
            "PourPtID": pid,
            "source_file": text_path.name if text_path is not None else (legacy_path.name if status == "migrated_from_raw_usgs" else ""),
            "output_file": str(out_path) if out_path.exists() or status != "missing_source" else "",
            "n_rows": len(out_df),
            "n_missing_flow": int(out_df["00060_Mean"].isna().sum()) if "00060_Mean" in out_df else "",
            "status": status,
        })

    for code, path in sorted(text_by_code.items()):
        if code in flow_mapping:
            continue
        pid = flow_mapping.get(code)
        if pid is None:
            report_rows.append({
                "FNF": code,
                "PourPtID": "",
                "source_file": path.name,
                "output_file": "",
                "n_rows": 0,
                "n_missing_flow": "",
                "status": "no_geometry_mapping",
            })

    remove_staged_cdec_fnf_flows(remove_reports=False)

    staged_report = pd.DataFrame(report_rows).sort_values("FNF")
    if mapping_report.empty:
        merged_report = staged_report
    else:
        mapping_cols = [col for col in mapping_report.columns if col != "PourPtID"]
        merged_report = staged_report.merge(mapping_report[mapping_cols], on="FNF", how="left")
    merged_report.to_csv(CDEC_FLOW_CATALOG, index=False)
    if CDEC_LEGACY_STAGING_REPORT.exists():
        CDEC_LEGACY_STAGING_REPORT.unlink()
    return merged_report


def main(stage_flows: bool = True, include_cdec: bool = True) -> None:
    combined = build_training_watershed_catalog(include_cdec=include_cdec)
    n_cdec = int(combined["PourPtID"].astype(str).str.startswith("990").sum())
    print(f"Combined training watersheds: {len(combined)} total ({n_cdec} CDEC)")
    print(f"  GeoPackage: {WATERSHEDS_GPKG}")
    print(f"  CSV:        {WATERSHED_GEOMETRY}")
    print(f"  GeoJSON:    {WATERSHED_GEOJSON}")

    if not include_cdec:
        removed = remove_staged_cdec_fnf_flows()
        print(f"CDEC excluded; removed {len(removed)} staged raw-flow files")
    elif stage_flows:
        report = stage_cdec_fnf_flows()
        ready_statuses = {"converted_from_legacy_text", "exists", "migrated_from_raw_usgs"}
        staged = int(report["status"].isin(ready_statuses).sum()) if "status" in report else 0
        unmapped = report.loc[~report["status"].isin(ready_statuses), "FNF"].tolist() if "status" in report else []
        print(f"CDEC FNF daily CSVs ready: {staged}")
        if unmapped:
            print(f"  Unmapped FNF files (no staged flow): {unmapped}")
        print(f"  Catalog: {CDEC_FLOW_CATALOG}")
        print(f"  Mapping audit: {CDEC_MAPPING_AUDIT}")


if __name__ == "__main__":
    main()