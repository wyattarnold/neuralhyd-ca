"""
Pre-build optimised data files for the Streamflow Explorer app.

Outputs go to app/data/:
  geojson/{layer_key}.geojson           — simplified polygons (EPSG:4326)
  timeseries/vic_{layer_key}.parquet    — VIC-Sim runoff (CFS, Int32)
  timeseries/obs.parquet                — observed streamflow (CFS, Int32)
  timeseries/lstm_pred.parquet          — LSTM dual predicted total Q (CFS, Int32)
  timeseries/lstm_fast.parquet          — LSTM dual fast pathway (CFS, Int32)
  timeseries/lstm_slow.parquet          — LSTM dual slow pathway (CFS, Int32)

Run once (or whenever source data changes):
    python -m app.build_data
"""

from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO = Path(__file__).resolve().parents[1]
GDB = REPO / "data" / "raw" / "neuralhyd-ca.gdb"
WS_PATH = REPO / "data" / "training" / "watersheds" / "watersheds.geojson"
AGG = REPO / "data" / "external" / "cec" / "VIC-Sim" / "aggregated"

FLOW_DIR = REPO / "data" / "training" / "flow"
STATIC_CSV = REPO / "data" / "training" / "static" / "Physical_Attributes_Watersheds.csv"
CLIMATE_STATIC_CSV = REPO / "data" / "training" / "static" / "Climate_Statistics_Watersheds.csv"
VIC_KGE_CSV = REPO / "data" / "external" / "cec" / "model_kge_comparison.csv"
LSTM_DIR = REPO / "data" / "training" / "output" / "dual_lstm_kfold"

OUT = Path(__file__).parent / "data"
GEO_DIR = OUT / "geojson"
TS_DIR = OUT / "timeseries"

GEO_DIR.mkdir(parents=True, exist_ok=True)
TS_DIR.mkdir(parents=True, exist_ok=True)

# mm/day → CFS: CFS = mm/day × area_km² / 2.44577
_MM_DAY_TO_CFS_FACTOR = 1.0 / 2.44577

# ---------------------------------------------------------------------------
# Layer definitions
# ---------------------------------------------------------------------------

LAYERS: list[tuple[str, str, Path, float, Path | None, str | None]] = [
    # (key,  id_col,  src_path,  simplify_tol, vic_csv,  gdb_layer)
    ("huc8",  "huc8",  GDB, 0.0005, AGG / "huc8_runoff.csv",  "WBDHU8"),
    ("huc10", "huc10", GDB, 0.0005, AGG / "huc10_runoff.csv", "WBDHU10"),
    ("training_watersheds", "Pour Point ID", WS_PATH, 0.0001, AGG / "training_watersheds_runoff.csv", None),
]

# ---------------------------------------------------------------------------
# GeoJSON builder
# ---------------------------------------------------------------------------

def build_geojson(key: str, id_col: str, src_path: Path, tol: float, gdb_layer: str | None = None) -> None:
    out_path = GEO_DIR / f"{key}.geojson"
    print(f"  GeoJSON {key} … ", end="", flush=True)

    kwargs = {"layer": gdb_layer} if gdb_layer is not None else {}
    gdf = gpd.read_file(src_path, **kwargs).to_crs("EPSG:4326")
    gdf["geometry"] = gdf.geometry.simplify(tol, preserve_topology=True)

    # Keep only id + name (if present) + geometry
    keep = [id_col, "geometry"]
    if "name" in gdf.columns:
        keep.insert(1, "name")
    gdf = gdf[[c for c in keep if c in gdf.columns]]
    gdf[id_col] = gdf[id_col].astype(str)

    out_path.write_text(gdf.to_json())
    size_mb = out_path.stat().st_size / 1e6
    print(f"{len(gdf)} polygons → {size_mb:.2f} MB")


def build_training_watersheds_geojson(tol: float) -> None:
    """Build training_watersheds GeoJSON with tier, obs dates, KGE and NSE properties."""
    out_path = GEO_DIR / "training_watersheds.geojson"
    print("  GeoJSON training_watersheds … ", end="", flush=True)

    # --- Basin tier + LSTM KGE/NSE from 5-fold basin_results ---
    tier_map: dict[str, int] = {}
    lstm_kge_map: dict[str, float] = {}
    lstm_nse_map: dict[str, float] = {}
    for br in sorted(LSTM_DIR.glob("fold_*/basin_results.csv")):
        df = pd.read_csv(br, usecols=["basin_id", "tier", "kge", "nse"])
        for _, row in df.iterrows():
            bid = str(int(row["basin_id"]))
            tier_map[bid] = int(row["tier"])
            lstm_kge_map[bid] = float(row["kge"])
            lstm_nse_map[bid] = float(row["nse"])

    # --- VIC regionalized KGE ---
    vic_kge_map: dict[str, float] = {}
    if VIC_KGE_CSV.exists():
        vic_df = pd.read_csv(VIC_KGE_CSV, usecols=["GageID", "VIC_KGE_Regionalized"])
        for _, row in vic_df.iterrows():
            if pd.notna(row["VIC_KGE_Regionalized"]):
                vic_kge_map[str(row["GageID"])] = float(row["VIC_KGE_Regionalized"])

    # --- VIC NSE (computed from obs vs VIC timeseries) ---
    vic_nse_map: dict[str, float] = {}
    vic_ts_path = AGG / "training_watersheds_runoff.csv"
    if vic_ts_path.exists():
        vic_wide = pd.read_csv(vic_ts_path, index_col="date", parse_dates=True)
        for tier in [1, 2, 3]:
            for csv in sorted((FLOW_DIR / f"tier_{tier}").glob("*_cleaned.csv")):
                bid = csv.stem.replace("_cleaned", "")
                if bid not in vic_wide.columns:
                    continue
                obs_df = pd.read_csv(csv, usecols=["date", "flow"], index_col="date", parse_dates=True)
                merged = obs_df.join(vic_wide[[bid]].rename(columns={bid: "vic"}), how="inner").dropna()
                if len(merged) < 365:
                    continue
                obs_arr = merged["flow"].values
                sim_arr = merged["vic"].values
                mean_obs = obs_arr.mean()
                ss_res = np.sum((obs_arr - sim_arr) ** 2)
                ss_tot = np.sum((obs_arr - mean_obs) ** 2)
                if ss_tot > 0:
                    vic_nse_map[bid] = float(1.0 - ss_res / ss_tot)

    # --- Observed record date ranges ---
    obs_range: dict[str, tuple[str, str]] = {}
    for tier in [1, 2, 3]:
        for csv in sorted((FLOW_DIR / f"tier_{tier}").glob("*_cleaned.csv")):
            bid = csv.stem.replace("_cleaned", "")
            dates = pd.read_csv(csv, usecols=["date", "flow"])
            valid = dates.dropna(subset=["flow"])
            if len(valid):
                obs_range[bid] = (str(valid["date"].iloc[0]), str(valid["date"].iloc[-1]))

    # --- Load and simplify geometry ---
    gdf = gpd.read_file(WS_PATH).to_crs("EPSG:4326")
    gdf["geometry"] = gdf.geometry.simplify(tol, preserve_topology=True)
    gdf["Pour Point ID"] = gdf["Pour Point ID"].astype(str)
    keep = ["Pour Point ID", "geometry"]
    gdf = gdf[[c for c in keep if c in gdf.columns]].copy()

    # Attach metadata
    gdf["tier"] = gdf["Pour Point ID"].map(tier_map)
    gdf["obs_start"] = gdf["Pour Point ID"].map(lambda b: obs_range.get(b, (None, None))[0])
    gdf["obs_end"]   = gdf["Pour Point ID"].map(lambda b: obs_range.get(b, (None, None))[1])
    gdf["lstm_kge"]  = gdf["Pour Point ID"].map(lstm_kge_map)
    gdf["vic_kge"]   = gdf["Pour Point ID"].map(vic_kge_map)
    gdf["lstm_nse"]  = gdf["Pour Point ID"].map(lstm_nse_map)
    gdf["vic_nse"]   = gdf["Pour Point ID"].map(vic_nse_map)

    out_path.write_text(gdf.to_json())
    size_mb = out_path.stat().st_size / 1e6
    print(f"{len(gdf)} polygons → {size_mb:.2f} MB")


# ---------------------------------------------------------------------------
# VIC Parquet builder (wide-format CSV → Parquet)
# ---------------------------------------------------------------------------

def build_vic_parquet(key: str, vic_csv: Path) -> None:
    out_path = TS_DIR / f"vic_{key}.parquet"
    print(f"  vic_{key} … ", end="", flush=True)

    df = pd.read_csv(vic_csv, index_col="date", parse_dates=True)
    df = df.round(1)

    df.to_parquet(out_path, engine="pyarrow", compression="zstd")
    size_mb = out_path.stat().st_size / 1e6
    print(f"{df.shape[0]} days × {df.shape[1]} cols → {size_mb:.2f} MB")


# ---------------------------------------------------------------------------
# Observed streamflow builder (per-basin CSVs → single wide Parquet)
# ---------------------------------------------------------------------------

def build_obs_parquet() -> None:
    """Read cleaned flow CSVs across all tiers, convert CFS → wide Parquet."""
    out_path = TS_DIR / "obs.parquet"
    print("  obs … ", end="", flush=True)

    series: dict[str, pd.Series] = {}
    for tier in [1, 2, 3]:
        tier_dir = FLOW_DIR / f"tier_{tier}"
        for csv in sorted(tier_dir.glob("*_cleaned.csv")):
            bid = csv.stem.replace("_cleaned", "")
            df = pd.read_csv(csv, usecols=["date", "flow"], index_col="date", parse_dates=True)
            series[bid] = df["flow"]

    wide = pd.DataFrame(series)
    wide = wide.round(1)
    wide.index.name = "date"
    wide.to_parquet(out_path, engine="pyarrow", compression="zstd")
    size_mb = out_path.stat().st_size / 1e6
    print(f"{wide.shape[0]} days × {wide.shape[1]} basins → {size_mb:.2f} MB")


# ---------------------------------------------------------------------------
# LSTM dual results builder (per-fold/per-basin CSVs → wide Parquets)
# ---------------------------------------------------------------------------

def build_lstm_parquets() -> None:
    """Combine LSTM dual results across all folds into wide Parquets (CFS).

    Each basin appears in exactly one fold (spatial cross-validation holdout).
    Predictions are in mm/day — convert to CFS using basin area.
    Dates come from the corresponding observed flow file (same length, aligned).
    """
    # Load basin areas for mm/day → CFS conversion
    static = pd.read_csv(STATIC_CSV, usecols=["PourPtID", "total_Shape_Area_km2"])
    area_map = dict(zip(static["PourPtID"].astype(str), static["total_Shape_Area_km2"]))

    pred_series: dict[str, pd.Series] = {}
    fast_series: dict[str, pd.Series] = {}
    slow_series: dict[str, pd.Series] = {}

    for fold_dir in sorted(LSTM_DIR.glob("fold_*")):
        ts_dir = fold_dir / "timeseries"
        if not ts_dir.is_dir():
            continue
        for csv in sorted(ts_dir.glob("*.csv")):
            bid = csv.stem
            area = area_map.get(bid)
            if area is None:
                continue
            cfs_factor = area * _MM_DAY_TO_CFS_FACTOR

            # Read LSTM results (no date column — rows align with obs file)
            lstm = pd.read_csv(csv)

            # Find dates from the observed flow file
            obs_file = None
            for tier in [1, 2, 3]:
                candidate = FLOW_DIR / f"tier_{tier}" / f"{bid}_cleaned.csv"
                if candidate.exists():
                    obs_file = candidate
                    break
            if obs_file is None:
                continue

            obs_dates = pd.read_csv(obs_file, usecols=["date"], parse_dates=True)["date"]
            if len(obs_dates) != len(lstm):
                # Align by taking the trailing min(len) rows from each
                n = min(len(obs_dates), len(lstm))
                obs_dates = obs_dates.iloc[-n:].reset_index(drop=True)
                lstm = lstm.iloc[-n:].reset_index(drop=True)

            idx = pd.DatetimeIndex(obs_dates, name="date")

            pred_series[bid] = pd.Series((lstm["pred"].values * cfs_factor).round(1), index=idx)
            fast_series[bid] = pd.Series((lstm["q_fast"].values * cfs_factor).round(1), index=idx)
            slow_series[bid] = pd.Series((lstm["q_slow"].values * cfs_factor).round(1), index=idx)

    for name, data in [("lstm_pred", pred_series), ("lstm_fast", fast_series), ("lstm_slow", slow_series)]:
        out_path = TS_DIR / f"{name}.parquet"
        print(f"  {name} … ", end="", flush=True)
        wide = pd.DataFrame(data).round(1)
        wide.index.name = "date"
        wide.to_parquet(out_path, engine="pyarrow", compression="zstd")
        size_mb = out_path.stat().st_size / 1e6
        print(f"{wide.shape[0]} days × {wide.shape[1]} basins → {size_mb:.2f} MB")


# ---------------------------------------------------------------------------
# California outline builder (dissolve HUC-8 polygons)
# ---------------------------------------------------------------------------

def build_ca_outline() -> None:
    """Build simplified CA boundary by dissolving all HUC-8 polygons."""
    out_path = GEO_DIR / "ca_outline.geojson"
    print("  CA outline … ", end="", flush=True)

    gdf = gpd.read_file(GDB, layer="WBDHU8").to_crs("EPSG:4326")
    dissolved = gdf.dissolve()
    dissolved["geometry"] = dissolved.geometry.simplify(0.005, preserve_topology=True)
    dissolved = dissolved[["geometry"]]
    out_path.write_text(dissolved.to_json())
    size_mb = out_path.stat().st_size / 1e6
    print(f"→ {size_mb:.2f} MB")


# ---------------------------------------------------------------------------
# Static attributes JSON builder (per-basin lookup)
# ---------------------------------------------------------------------------

_DISPLAY_ATTRS = {
    "total_Shape_Area_km2": ("Area", "km²", 1),
    "ele_mt_uav": ("Elevation", "m", 0),
    "slp_dg_uav": ("Slope", "°", 1),
    "for_pc_use": ("Forest Cover", "%", 1),
    "cly_pc_uav": ("Clay", "%", 1),
    "slt_pc_uav": ("Silt", "%", 1),
    "snd_pc_uav": ("Sand", "%", 1),
    "precip_mean": ("Mean Precip", "mm/day", 2),
    "pet_mean": ("Mean PET", "mm/day", 2),
    "aridity_index": ("Aridity Index", "", 2),
    "snow_fraction": ("Snow Fraction", "", 2),
}


def build_static_attrs() -> None:
    """Build JSON lookup of display-friendly static attributes per basin."""
    out_path = OUT / "static_attrs.json"
    print("  static_attrs … ", end="", flush=True)

    atlas = pd.read_csv(STATIC_CSV)
    atlas["PourPtID"] = atlas["PourPtID"].astype(str)
    atlas = atlas.set_index("PourPtID")

    climate = pd.read_csv(CLIMATE_STATIC_CSV)
    climate["PourPtID"] = climate["PourPtID"].astype(str)
    climate = climate.set_index("PourPtID")

    combined = atlas.join(climate, rsuffix="_clim")

    result = {}
    for bid in combined.index:
        row = combined.loc[bid]
        attrs = {}
        for col, (label, unit, decimals) in _DISPLAY_ATTRS.items():
            val = row.get(col)
            if pd.notna(val):
                attrs[col] = {
                    "label": label,
                    "value": round(float(val), decimals),
                    "unit": unit,
                }
        result[bid] = attrs

    out_path.write_text(json.dumps(result))
    size_mb = out_path.stat().st_size / 1e6
    print(f"{len(result)} basins → {size_mb:.2f} MB")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("Building GeoJSON files …")
    for key, id_col, shp_path, tol, vic_csv, gdb_layer in LAYERS:
        if key == "training_watersheds":
            build_training_watersheds_geojson(tol)
        else:
            build_geojson(key, id_col, shp_path, tol, gdb_layer)

    print("\nBuilding CA outline …")
    build_ca_outline()

    print("\nBuilding static attributes …")
    build_static_attrs()

    print("\nBuilding VIC-Sim Parquet files …")
    for key, id_col, shp_path, tol, vic_csv, gdb_layer in LAYERS:
        if vic_csv is not None and vic_csv.exists():
            build_vic_parquet(key, vic_csv)
        else:
            print(f"  vic_{key} … skipped (no CSV)")

    print("\nBuilding observed streamflow Parquet …")
    build_obs_parquet()

    print("\nBuilding LSTM dual result Parquets …")
    build_lstm_parquets()

    print("\nDone. Output:")
    for f in sorted((OUT).rglob("*")):
        if f.is_file():
            print(f"  {f.relative_to(REPO)}  ({f.stat().st_size/1e6:.2f} MB)")


if __name__ == "__main__":
    main()
