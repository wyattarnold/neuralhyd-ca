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

# Watershed boundaries (CSV + GeoJSON)
WATERSHEDS_DIR       = TRAINING_DIR / "watersheds"
WATERSHED_GEOMETRY   = WATERSHEDS_DIR / "watersheds.csv"
WATERSHED_GEOJSON    = WATERSHEDS_DIR / "watersheds.geojson"

# ---------------------------------------------------------------------------
# Training data (final outputs used by the model)
# ---------------------------------------------------------------------------
CLIMATE_DIR          = TRAINING_DIR / "climate"
STATIC_DIR           = TRAINING_DIR / "static"
FLOW_DIR             = TRAINING_DIR / "flow"        # tier_1/, tier_2/, tier_3/
TRAINING_OUTPUT_DIR  = TRAINING_DIR / "output"

# Static attribute inputs (GIS intersect tables — in prepare/geo_ops/)
GEO_OPS_DIR          = QA_DIR / "geo_ops"
VICGRIDS_FILE        = GEO_OPS_DIR / "VICGrids_Intersect_Watersheds.csv"
BASIN_ATLAS_INPUT    = GEO_OPS_DIR / "BasinATLAS_v10_lev12_Intersect_Watersheds.csv"

# Static attribute outputs (weighted averages)
BASIN_ATLAS_OUTPUT   = STATIC_DIR / "Physical_Attributes_Watersheds.csv"
CLIMATE_STATS_OUTPUT = STATIC_DIR / "Climate_Statistics_Watersheds.csv"

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

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DEFAULT_CONFIG           = SCRIPTS_DIR / "config.toml"
