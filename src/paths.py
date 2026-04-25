"""Centralized path configuration for the neuralhyd-ca project.

All paths are relative to PROJECT_ROOT (the repo root containing scripts/, src/, data/).
Used by src.data, src.lstm, src.eval, and scripts/.
"""
from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Project root  (contains scripts/, src/, data/)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Top-level directories
# ---------------------------------------------------------------------------
DATA_DIR     = PROJECT_ROOT / "data"
RAW_DIR      = DATA_DIR / "raw"
TRAINING_DIR = DATA_DIR / "training"
QA_DIR       = DATA_DIR / "prepare"
SCRIPTS_DIR  = PROJECT_ROOT / "scripts"
EVAL_DIR     = DATA_DIR / "eval"

# ---------------------------------------------------------------------------
# Raw / source data
# ---------------------------------------------------------------------------
STATION_TABLE        = RAW_DIR / "USGS_Table_1.csv"
RAW_USGS_DIR         = RAW_DIR / "usgs"

# Individual GeoPackage exports from the GDB (run data/prepare/geo_ops/export_gdb_layers.py)
GIS_DIR              = RAW_DIR / "gis"
WATERSHEDS_GPKG      = GIS_DIR / "USGS_Training_Watersheds.gpkg"
POUR_POINTS_GPKG     = GIS_DIR / "USGS_Training_Watersheds_Pour_Points.gpkg"
VIC_GRIDS_GPKG       = GIS_DIR / "VICGrids_CAORNV_LatLong.gpkg"
WBDHU4_GPKG          = GIS_DIR / "WBDHU4.gpkg"
WBDHU6_GPKG          = GIS_DIR / "WBDHU6.gpkg"
WBDHU8_GPKG          = GIS_DIR / "WBDHU8.gpkg"
WBDHU10_GPKG         = GIS_DIR / "WBDHU10.gpkg"
WBDHU12_GPKG         = GIS_DIR / "WBDHU12.gpkg"

# Watershed boundaries (CSV + GeoJSON)
WATERSHEDS_DIR       = TRAINING_DIR / "watersheds"
WATERSHED_GEOMETRY   = WATERSHEDS_DIR / "watersheds.csv"
WATERSHED_GEOJSON    = WATERSHEDS_DIR / "watersheds.geojson"

# ---------------------------------------------------------------------------
# Training data (final outputs used by the model)
#
# Layout: ``data/training/{climate,static}/<polygon-target>/`` for static
# attributes; daily climate + flow time series are consolidated into compact
# zarr cubes (see below).  ``<polygon-target>`` is one of ``watersheds``
# (USGS gauge basins), ``huc8``, ``huc10``, ``huc12``.
# ---------------------------------------------------------------------------
TRAINING_CLIMATE_ROOT = TRAINING_DIR / "climate"
TRAINING_STATIC_ROOT  = TRAINING_DIR / "static"
STATIC_DIR           = TRAINING_STATIC_ROOT / "watersheds"
TRAINING_OUTPUT_DIR  = TRAINING_DIR / "output"

# ---------------------------------------------------------------------------
# Compact zarr cube stores (single source of truth for daily time series).
# ---------------------------------------------------------------------------
CLIMATE_WATERSHEDS_ZARR = TRAINING_CLIMATE_ROOT / "watersheds.zarr"
CLIMATE_HUC12_ZARR      = TRAINING_CLIMATE_ROOT / "huc12.zarr"
FLOW_ZARR               = TRAINING_DIR / "flow.zarr"

# Static attribute inputs (GIS intersect tables — in prepare/geo_ops/)
GEO_OPS_DIR          = QA_DIR / "geo_ops"
VICGRIDS_FILE        = GEO_OPS_DIR / "VICGrids_Intersect_Watersheds.csv"
BASIN_ATLAS_INPUT    = GEO_OPS_DIR / "BasinATLAS_v10_lev12_Intersect_Watersheds.csv"
BASIN_ATLAS_CLIPPED  = GIS_DIR / "BasinATLAS_v10_lev12_clipped.gpkg"

# Static attribute outputs (weighted averages)
BASIN_ATLAS_OUTPUT   = STATIC_DIR / "Physical_Attributes_Watersheds.csv"
CLIMATE_STATS_OUTPUT = STATIC_DIR / "Climate_Statistics_Watersheds.csv"

# ---------------------------------------------------------------------------
# Target-aware path resolution (watersheds / huc8 / huc10)
# ---------------------------------------------------------------------------

_TARGET_SUFFIX: dict[str, str] = {
    "watersheds":          "Watersheds",
    "training_watersheds": "Watersheds",
    "huc8":                "HUC8",
    "huc10":               "HUC10",
    "huc12":               "HUC12",
}


def get_target_paths(target: str = "watersheds") -> dict[str, Path]:
    """Return input/output paths for a given polygon target.

    All targets now live under ``data/training/{climate,static}/<target>/``.
    For HUC targets the *full-domain* climate/static outputs additionally land
    in ``data/eval/{climate,static}/<target>/`` (see ``get_eval_target_paths``);
    the training subset is copied into the training tree by step 9 so trainable
    configs only need a single root.
    """
    suffix = _TARGET_SUFFIX[target]
    key = "watersheds" if target in ("watersheds", "training_watersheds") else target
    static_dir  = TRAINING_STATIC_ROOT / key
    return {
        "climate_zarr":         TRAINING_CLIMATE_ROOT / f"{key}.zarr",
        "static_dir":           static_dir,
        "vicgrids_file":        GEO_OPS_DIR / f"VICGrids_Intersect_{suffix}.csv",
        "basin_atlas_input":    GEO_OPS_DIR / f"BasinATLAS_v10_lev12_Intersect_{suffix}.csv",
        "basin_atlas_output":   static_dir / f"Physical_Attributes_{suffix}.csv",
        "climate_stats_output": static_dir / f"Climate_Statistics_{suffix}.csv",
    }


def get_eval_target_paths(target: str) -> dict[str, Path]:
    """Return paths under ``data/eval/{climate,static}/<target>/``.

    Used by step 9 to write *full-domain* HUC8/10/12 climate + static outputs.
    Only the manifest subset is later copied into the training tree.
    """
    suffix = _TARGET_SUFFIX[target]
    static_dir  = EVAL_DIR / "static" / target
    return {
        "climate_zarr":         EVAL_DIR / "climate" / f"{target}.zarr",
        "static_dir":           static_dir,
        "vicgrids_file":        GEO_OPS_DIR / f"VICGrids_Intersect_{suffix}.csv",
        "basin_atlas_input":    GEO_OPS_DIR / f"BasinATLAS_v10_lev12_Intersect_{suffix}.csv",
        "basin_atlas_output":   static_dir / f"Physical_Attributes_{suffix}.csv",
        "climate_stats_output": static_dir / f"Climate_Statistics_{suffix}.csv",
    }


# ---------------------------------------------------------------------------
# Evaluation & external comparison data
# ---------------------------------------------------------------------------
SIM_DIR              = EVAL_DIR / "sim"
VIC_RUNOFF_CSV       = DATA_DIR / "external" / "cec" / "VIC-Sim" / "aggregated" / "training_watersheds_runoff.csv"
VIC_CAL_DIR          = DATA_DIR / "external" / "cec" / "VIC-Calibration"

# ---------------------------------------------------------------------------
# Intermediate data (cleaned flows before tier-sorting)
# ---------------------------------------------------------------------------
STEP_3_OUTPUT_DIR        = QA_DIR / "verify_climate_data"
STEP_6_OUTPUT_DIR        = QA_DIR / "flow_precip_exceedance_filter"
STEP_7_OUTPUT_DIR        = QA_DIR / "qa_qc_report"
STEP_8_OUTPUT_DIR        = QA_DIR / "qa_qc_tier_sort"

FLOW_CLEANED_DIR         = STEP_6_OUTPUT_DIR / "flow_cleaned"
FLOW_DROPPED_DIR         = STEP_6_OUTPUT_DIR / "flow_dropped"
FLOW_CLEANED_STRICT_DIR  = STEP_7_OUTPUT_DIR / "flow_cleaned_strict"

# ---------------------------------------------------------------------------
# QA outputs
# ---------------------------------------------------------------------------
QA_OUTPUT_DIR            = QA_DIR
CLIMATE_VERIFICATION_DIR = STEP_3_OUTPUT_DIR
QAQC_FLOW_PRECIP_CSV    = STEP_8_OUTPUT_DIR / "qaqc_flow_vs_precip_summary.csv"

# ---------------------------------------------------------------------------
# Analysis outputs
# ---------------------------------------------------------------------------
MAP_WATERSHEDS_DIR       = QA_DIR / "map_watersheds"
TIER_CHARACTERISTICS_DIR = QA_DIR / "tier_characteristics"
SPATIAL_ANALYSIS_DIR     = QA_DIR / "spatial_analysis"

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DEFAULT_CONFIG           = SCRIPTS_DIR / "config.toml"
