# Data Directory

## Model Inputs

`training/` — All final training/evaluation data, read directly by the model code (paths set in `config.toml`).

| Directory | Contents |
|---|---|
| `training/climate/` | Daily climate CSVs per basin (`climate_<basin_id>.csv`): precip_mm, tmax_c, tmin_c (1915–2018) |
| `training/flow/` | Quality-filtered daily streamflow + climate, split by tier (`tier_{1,2,3}/<basin_id>_cleaned.csv`) |
| `training/static/` | Basin-level attributes: `Physical_Attributes_Watersheds.csv` (physical/land-cover) and `Climate_Statistics_Watersheds.csv` (long-term climate normals) |
| `training/watersheds/` | Watershed boundaries (`watersheds.geojson`) and pour-point table (`watersheds.csv`) |
| `training/output/` | Model outputs created at runtime — per-fold checkpoints (`best_model.pt`), basin results, and predicted timeseries |

## Raw Source Data

`raw/` — Immutable downloads, never modified by the pipeline.

- `usgs/` — Original USGS daily streamflow downloads (216 basins)
- `watershed_geometry/` — Watershed boundaries and pour point shapefiles

## Processing Pipeline

`prepare/` — Intermediate outputs produced by the data preparation pipeline (`prepare_data.py` at the repo root).

| Directory | Contents |
|---|---|
| `prepare/geo_ops/` | GIS intersection tables from ArcGIS (`BasinATLAS_v10_lev12_Intersect_Watersheds.csv`, `VICGrids_Intersect_Watersheds.csv`) — inputs to steps 4 and 5 |
| `prepare/verify_climate_data/` | Step 3 outputs — monthly average verification CSVs per basin |
| `prepare/flow_precip_exceedance_filter/` | Step 6 outputs — cleaned/dropped flow CSVs, per-site figures, site metrics |
| `prepare/qa_qc_report/` | Step 7 outputs — QA/QC section CSVs (`sec1_` through `sec7_`), full report, strictly-cleaned flow files |
| `prepare/qa_qc_tier_sort/` | Step 8 outputs — flow/precip QA CSVs; also writes final tier-sorted files to `training/flow/` |
| `prepare/map_watersheds/` | Analysis — watershed map (PNG + PDF) |
| `prepare/tier_characteristics/` | Analysis — tier characterisation CDF figures (PDF) |

## External Comparison Data

`external/` — Third-party model results used for benchmarking, not produced by this project.

- `cec/` — CEC process-based model outputs (NOAH-MP and VIC, calibrated + regionalized KGE values)
