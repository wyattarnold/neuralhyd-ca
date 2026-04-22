"""
Pre-build optimised data files for the Streamflow Explorer app.

Outputs go to app/data/:
  geojson/{layer_key}.geojson           — simplified polygons (EPSG:4326)
  timeseries/vic_{layer_key}.parquet    — VIC-Sim total runoff (CFS, Int32)
  timeseries/vic_baseflow_{lk}.parquet  — VIC-Sim baseflow component (CFS)
  timeseries/vic_surface_{lk}.parquet   — VIC-Sim surface runoff component (CFS)
  timeseries/obs.parquet                — observed streamflow (CFS, Int32)
  timeseries/obs_baseflow.parquet       — Lyne–Hollick baseflow (CFS)
  timeseries/lstm_pred_{layer}.parquet       — LSTM dual ensemble mean total Q (CFS)
  timeseries/lstm_pred_min_{layer}.parquet   — LSTM dual ensemble min total Q (CFS)
  timeseries/lstm_pred_max_{layer}.parquet   — LSTM dual ensemble max total Q (CFS)
  timeseries/lstm_fast_{layer}.parquet       — LSTM dual fast pathway mean (CFS)
  timeseries/lstm_slow_{layer}.parquet       — LSTM dual slow pathway mean (CFS)
  timeseries/lstm_single_pred_{layer}.parquet     — LSTM single ensemble mean total Q (CFS)
  timeseries/lstm_single_pred_min_{layer}.parquet — LSTM single ensemble min total Q (CFS)
  timeseries/lstm_single_pred_max_{layer}.parquet — LSTM single ensemble max total Q (CFS)

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
GIS_DIR = REPO / "data" / "raw" / "gis"
HUC8_GPKG = GIS_DIR / "WBDHU8.gpkg"
HUC10_GPKG = GIS_DIR / "WBDHU10.gpkg"
WATERSHEDS_GPKG = GIS_DIR / "USGS_Training_Watersheds.gpkg"
AGG = REPO / "data" / "external" / "cec" / "VIC-Sim" / "aggregated"

FLOW_DIR = REPO / "data" / "training" / "flow"
STATIC_CSV = REPO / "data" / "training" / "static" / "Physical_Attributes_Watersheds.csv"
CLIMATE_STATIC_CSV = REPO / "data" / "training" / "static" / "Climate_Statistics_Watersheds.csv"
VIC_KGE_CSV = REPO / "data" / "external" / "cec" / "model_kge_comparison.csv"
LSTM_DIR = REPO / "data" / "training" / "output" / "dual_lstm_kfold"
LSTM_SINGLE_DIR = REPO / "data" / "training" / "output" / "single_lstm_kfold"
SIM_DIR = REPO / "data" / "eval" / "sim"

OUT = Path(__file__).parent / "data"
GEO_DIR = OUT / "geojson"
TS_DIR = OUT / "timeseries"

GEO_DIR.mkdir(parents=True, exist_ok=True)
TS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Layer definitions
# ---------------------------------------------------------------------------

LAYERS: list[tuple[str, str, Path, float, Path | None]] = [
    # (key,  id_col,  src_path,  simplify_tol, vic_csv)
    ("huc8",  "huc8",  HUC8_GPKG, 0.0005, AGG / "huc8_runoff.csv"),
    ("training_watersheds", "PourPtID", WATERSHEDS_GPKG, 0.0001, AGG / "training_watersheds_runoff.csv"),
]

# ---------------------------------------------------------------------------
# GeoJSON builder
# ---------------------------------------------------------------------------

def build_geojson(key: str, id_col: str, src_path: Path, tol: float) -> None:
    out_path = GEO_DIR / f"{key}.geojson"
    print(f"  GeoJSON {key} … ", end="", flush=True)

    gdf = gpd.read_file(src_path).to_crs("EPSG:4326")
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

    # --- Basin tier + LSTM Dual KGE/NSE from 5-fold basin_results ---
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

    # --- LSTM Single KGE/NSE from 5-fold basin_results ---
    lstm_single_nse_map: dict[str, float] = {}
    if LSTM_SINGLE_DIR.exists():
        for br in sorted(LSTM_SINGLE_DIR.glob("fold_*/basin_results.csv")):
            df = pd.read_csv(br, usecols=["basin_id", "nse"])
            for _, row in df.iterrows():
                bid = str(int(row["basin_id"]))
                lstm_single_nse_map[bid] = float(row["nse"])

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
    gdf = gpd.read_file(WATERSHEDS_GPKG).to_crs("EPSG:4326")
    gdf["geometry"] = gdf.geometry.simplify(tol, preserve_topology=True)
    gdf = gdf.rename(columns={"PourPtID": "Pour Point ID"})
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
    gdf["lstm_single_nse"] = gdf["Pour Point ID"].map(lstm_single_nse_map)
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
# LSTM sim results builder (per-basin sim CSVs → wide Parquets)
# ---------------------------------------------------------------------------

def _build_sim_parquets(
    sim_dir: Path,
    layer_key: str,
    model_label: str,
    columns: dict[str, str],
    fallback_columns: dict[str, str] | None = None,
) -> None:
    """Read per-basin sim CSVs and write wide Parquets.

    Parameters
    ----------
    sim_dir   : directory containing <basin_id>.csv files (already in CFS)
    layer_key : e.g. "training_watersheds", "huc8"
    model_label : display name for print messages
    columns   : mapping from sim CSV column → output parquet name prefix,
                e.g. {"q_mean": "lstm_pred", "q_fast_mean": "lstm_fast", ...}
    fallback_columns : optional per-output fallback source column used when
                ``src_col`` is missing from a basin CSV.  Lets training_watersheds
                held-out sims (``q_total/q_fast/q_slow``) fill both the mean
                and min/max outputs with the same single-fold prediction.
    """
    if not sim_dir.is_dir():
        for prefix in columns.values():
            print(f"  {prefix}_{layer_key} … skipped (no sim directory)")
        return

    csvs = sorted(sim_dir.glob("*.csv"))
    if not csvs:
        for prefix in columns.values():
            print(f"  {prefix}_{layer_key} … skipped (no CSVs)")
        return

    # Read all basin CSVs once
    basin_data: dict[str, pd.DataFrame] = {}
    for csv in csvs:
        bid = csv.stem
        df = pd.read_csv(csv, parse_dates=["date"], index_col="date")
        if not df.empty:
            basin_data[bid] = df

    # Build one wide Parquet per output column
    for src_col, prefix in columns.items():
        out_path = TS_DIR / f"{prefix}_{layer_key}.parquet"
        print(f"  {prefix}_{layer_key} … ", end="", flush=True)

        fb_col = fallback_columns.get(prefix) if fallback_columns else None

        def _pick(df: pd.DataFrame) -> pd.Series | None:
            if src_col in df.columns:
                return df[src_col]
            if fb_col is not None and fb_col in df.columns:
                return df[fb_col]
            return None

        series = {bid: s for bid, df in basin_data.items()
                  if (s := _pick(df)) is not None}
        if not series:
            print("skipped (no data)")
            continue

        wide = pd.DataFrame(series)
        wide.index.name = "date"
        wide.to_parquet(out_path, engine="pyarrow", compression="zstd")
        size_mb = out_path.stat().st_size / 1e6
        print(f"{wide.shape[0]} days × {wide.shape[1]} basins → {size_mb:.2f} MB")


def build_lstm_parquets() -> None:
    """Build LSTM dual sim Parquets for all available layer types."""
    sim_base = SIM_DIR / "dual_lstm_kfold"
    # Ensemble columns are preferred; fall back to held-out single-fold
    # columns (training_watersheds uses q_total/q_fast/q_slow for the
    # unbiased out-of-fold prediction — min/max then collapse to the mean).
    fallback = {
        "lstm_pred":     "q_total",
        "lstm_fast":     "q_fast",
        "lstm_slow":     "q_slow",
        "lstm_pred_min": "q_total",
        "lstm_pred_max": "q_total",
    }
    for layer_key in ("training_watersheds", "huc8"):
        sim_dir = sim_base / layer_key / "historical"
        _build_sim_parquets(sim_dir, layer_key, "LSTM Dual", {
            "q_mean":      "lstm_pred",
            "q_fast_mean": "lstm_fast",
            "q_slow_mean": "lstm_slow",
            "q_min":       "lstm_pred_min",
            "q_max":       "lstm_pred_max",
        }, fallback_columns=fallback)


# ---------------------------------------------------------------------------
# LSTM single results builder
# ---------------------------------------------------------------------------

def build_lstm_single_parquets() -> None:
    """Build LSTM single sim Parquets for all available layer types."""
    sim_base = SIM_DIR / "single_lstm_kfold"
    fallback = {
        "lstm_single_pred":     "q_total",
        "lstm_single_pred_min": "q_total",
        "lstm_single_pred_max": "q_total",
    }
    for layer_key in ("training_watersheds", "huc8"):
        sim_dir = sim_base / layer_key / "historical"
        _build_sim_parquets(sim_dir, layer_key, "LSTM Single", {
            "q_mean": "lstm_single_pred",
            "q_min":  "lstm_single_pred_min",
            "q_max":  "lstm_single_pred_max",
        }, fallback_columns=fallback)


# ---------------------------------------------------------------------------
# Observed baseflow (Lyne–Hollick digital filter)
# ---------------------------------------------------------------------------

def _lyne_hollick(flow: np.ndarray, alpha: float = 0.925, passes: int = 3) -> np.ndarray:
    """Return baseflow via Lyne–Hollick recursive digital filter.

    Forward–backward filtering with ``passes`` total passes for zero phase lag.
    """
    q = flow.copy().astype(float)
    n = len(q)
    for p in range(passes):
        qf = np.zeros(n)
        direction = 1 if p % 2 == 0 else -1
        rng = range(n) if direction == 1 else range(n - 1, -1, -1)
        prev_i = None
        for i in rng:
            if prev_i is None:
                qf[i] = 0.0
            else:
                qf[i] = alpha * qf[prev_i] + (1 + alpha) / 2 * (q[i] - q[prev_i])
            qf[i] = max(0.0, min(qf[i], q[i]))
            prev_i = i
        q = q - qf  # q becomes baseflow after subtracting quickflow
    return np.maximum(q, 0.0)


def build_obs_baseflow_parquet() -> None:
    """Apply Lyne–Hollick filter to observed flow and save baseflow Parquet."""
    out_path = TS_DIR / "obs_baseflow.parquet"
    print("  obs_baseflow … ", end="", flush=True)

    series: dict[str, pd.Series] = {}
    for tier in [1, 2, 3]:
        tier_dir = FLOW_DIR / f"tier_{tier}"
        for csv in sorted(tier_dir.glob("*_cleaned.csv")):
            bid = csv.stem.replace("_cleaned", "")
            df = pd.read_csv(csv, usecols=["date", "flow"], index_col="date", parse_dates=True)
            flow = df["flow"].values
            # Fill NaN gaps with 0 for filter stability, restore NaN after
            mask = np.isnan(flow)
            flow_filled = np.where(mask, 0.0, flow)
            bf = _lyne_hollick(flow_filled)
            bf[mask] = np.nan
            series[bid] = pd.Series(bf.round(1), index=df.index)

    wide = pd.DataFrame(series)
    wide.index.name = "date"
    wide.to_parquet(out_path, engine="pyarrow", compression="zstd")
    size_mb = out_path.stat().st_size / 1e6
    print(f"{wide.shape[0]} days × {wide.shape[1]} basins → {size_mb:.2f} MB")


# ---------------------------------------------------------------------------
# VIC component Parquet builders (baseflow + surface runoff)
# ---------------------------------------------------------------------------

def build_vic_component_parquets() -> None:
    """Build VIC baseflow and surface runoff Parquets for training_watersheds."""
    for component in ("baseflow", "surface"):
        csv_path = AGG / f"training_watersheds_runoff_{component}.csv"
        out_path = TS_DIR / f"vic_{component}_training_watersheds.parquet"
        print(f"  vic_{component}_training_watersheds … ", end="", flush=True)
        if not csv_path.exists():
            print("skipped (no CSV — run aggregate_runoff.py --components)")
            continue
        df = pd.read_csv(csv_path, index_col="date", parse_dates=True)
        df = df.round(1)
        df.to_parquet(out_path, engine="pyarrow", compression="zstd")
        size_mb = out_path.stat().st_size / 1e6
        print(f"{df.shape[0]} days × {df.shape[1]} cols → {size_mb:.2f} MB")


# ---------------------------------------------------------------------------
# California outline builder (dissolve HUC-8 polygons)
# ---------------------------------------------------------------------------

def build_ca_outline() -> None:
    """Build simplified CA boundary by dissolving all HUC-8 polygons."""
    out_path = GEO_DIR / "ca_outline.geojson"
    print("  CA outline … ", end="", flush=True)

    gdf = gpd.read_file(HUC8_GPKG).to_crs("EPSG:4326")
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
    for key, id_col, shp_path, tol, vic_csv in LAYERS:
        if key == "training_watersheds":
            build_training_watersheds_geojson(tol)
        else:
            build_geojson(key, id_col, shp_path, tol)

    print("\nBuilding CA outline …")
    build_ca_outline()

    print("\nBuilding static attributes …")
    build_static_attrs()

    print("\nBuilding VIC-Sim Parquet files …")
    for key, id_col, shp_path, tol, vic_csv in LAYERS:
        if vic_csv is not None and vic_csv.exists():
            build_vic_parquet(key, vic_csv)
        else:
            print(f"  vic_{key} … skipped (no CSV)")

    print("\nBuilding observed streamflow Parquet …")
    build_obs_parquet()

    print("\nBuilding observed baseflow (Lyne–Hollick) Parquet …")
    build_obs_baseflow_parquet()

    print("\nBuilding LSTM dual result Parquets …")
    build_lstm_parquets()

    print("\nBuilding LSTM single result Parquets …")
    build_lstm_single_parquets()

    print("\nBuilding VIC component Parquets …")
    build_vic_component_parquets()

    print("\nDone. Output:")
    for f in sorted((OUT).rglob("*")):
        if f.is_file():
            print(f"  {f.relative_to(REPO)}  ({f.stat().st_size/1e6:.2f} MB)")


if __name__ == "__main__":
    main()
