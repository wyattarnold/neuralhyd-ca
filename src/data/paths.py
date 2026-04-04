"""Centralized path configuration for the neuralhyd-ca data pipeline.

All paths are relative to PROJECT_ROOT (the repo root containing main.py).
"""
from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Project root  (contains scripts/, src/, data/)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# ---------------------------------------------------------------------------
# Top-level data directories
# ---------------------------------------------------------------------------
DATA_DIR     = PROJECT_ROOT / "data"
RAW_DIR      = DATA_DIR / "raw"
TRAINING_DIR = DATA_DIR / "training"
QA_DIR       = DATA_DIR / "prepare"

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

# Static attribute inputs (GIS intersect tables — in prepare/geo_ops/)
GEO_OPS_DIR          = QA_DIR / "geo_ops"
VICGRIDS_FILE        = GEO_OPS_DIR / "VICGrids_Intersect_Watersheds.csv"
BASIN_ATLAS_INPUT    = GEO_OPS_DIR / "BasinATLAS_v10_lev12_Intersect_Watersheds.csv"

# Static attribute outputs (weighted averages)
BASIN_ATLAS_OUTPUT   = STATIC_DIR / "Physical_Attributes_Watersheds.csv"
CLIMATE_STATS_OUTPUT = STATIC_DIR / "Climate_Statistics_Watersheds.csv"

# ---------------------------------------------------------------------------
# Intermediate data (cleaned flows before tier-sorting)
# Step 6 outputs: flow_cleaned/, flow_dropped/, flow_precip_filter_exceedance/, site_metrics.csv
# Step 7 outputs: sec*.csv, qa_qc_report.txt, flow_cleaned_strict/
# Step 8 outputs: qaqc_*.csv
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
QAQC_FLOW_PRECIP_CSV     = STEP_8_OUTPUT_DIR / "qaqc_flow_vs_precip_summary.csv"

# ---------------------------------------------------------------------------
# Analysis outputs
# ---------------------------------------------------------------------------
MAP_WATERSHEDS_DIR       = QA_DIR / "map_watersheds"
TIER_CHARACTERISTICS_DIR = QA_DIR / "tier_characteristics"
