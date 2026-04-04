"""In-memory app state — geometry + lazy timeseries loading.

Design for memory efficiency:
- GeoJSON is pre-built (by app/build_data.py) and read from disk as-is.
  No geopandas at runtime.
- Timeseries data is stored as Parquet (columnar, zstd-compressed).
  ``LazyParquet.read_column()`` reads only the single column requested,
  so resident memory scales with one polygon at a time, not the full table.

Data sources per layer:
  HUC layers       -> VIC-Sim only
  training_watersheds -> VIC-Sim + observed (CFS) + LSTM dual (pred, fast, slow)

Run ``python -m app.build_data`` after source data changes to refresh.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Lazy Parquet column reader
# ---------------------------------------------------------------------------

class LazyParquet:
    """Read single columns from a Parquet file without loading the whole table."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._columns: list[str] | None = None

    @property
    def columns(self) -> list[str]:
        if self._columns is None:
            import pyarrow.parquet as pq
            # Exclude the pandas index column ("date") — it's metadata, not a polygon ID
            all_names = pq.read_schema(self.path).names
            self._columns = [c for c in all_names if c != "date"]
        return self._columns

    def read_column(self, col: str) -> pd.Series:
        """Read a single polygon's timeseries (DatetimeIndex, Int32 CFS)."""
        df = pd.read_parquet(self.path, columns=[col], engine="pyarrow")
        return df[col]


# ---------------------------------------------------------------------------
# Layer descriptor
# ---------------------------------------------------------------------------

_APP_DATA = Path(__file__).parent / "data"


def _lazy(filename: str) -> LazyParquet | None:
    path = _APP_DATA / "timeseries" / filename
    return LazyParquet(path) if path.exists() else None


class Layer:
    """One watershed boundary layer (e.g. HUC8 or training_watersheds)."""

    def __init__(self, key: str, name: str, id_col: str) -> None:
        self.key = key
        self.name = name
        self.id_col = id_col

        geo_path = _APP_DATA / "geojson" / f"{key}.geojson"
        self._geojson: dict | None = None
        self._geo_path = geo_path if geo_path.exists() else None

        # Timeseries sources (all LazyParquet, None if file missing)
        self.vic = _lazy(f"vic_{key}.parquet")
        self.obs = _lazy("obs.parquet") if key == "training_watersheds" else None
        self.lstm_pred = _lazy("lstm_pred.parquet") if key == "training_watersheds" else None
        self.lstm_fast = _lazy("lstm_fast.parquet") if key == "training_watersheds" else None
        self.lstm_slow = _lazy("lstm_slow.parquet") if key == "training_watersheds" else None

    @property
    def geojson(self) -> dict:
        if self._geojson is None:
            if self._geo_path is None:
                raise FileNotFoundError(
                    f"Pre-built GeoJSON not found for layer '{self.key}'. "
                    "Run: python -m app.build_data"
                )
            self._geojson = json.loads(self._geo_path.read_text())
        return self._geojson

    @property
    def polygon_count(self) -> int:
        return len(self.geojson["features"])

    @property
    def available_series(self) -> list[str]:
        out = []
        for attr in ("vic", "obs", "lstm_pred", "lstm_fast", "lstm_slow"):
            if getattr(self, attr) is not None:
                out.append(attr)
        return out


# ---------------------------------------------------------------------------
# AppState singleton
# ---------------------------------------------------------------------------

_LAYER_DEFS = [
    ("huc8",             "HUC-8",             "huc8"),
    ("huc10",            "HUC-10",            "huc10"),
    ("training_watersheds", "Training Watersheds", "Pour Point ID"),
]


class AppState:
    layers: dict[str, Layer]
    static_attrs: dict  # basin_id → { col: {label, value, unit} }
    ca_outline: dict    # GeoJSON FeatureCollection

    def __init__(self) -> None:
        self.layers = {}
        self.static_attrs = {}
        self.ca_outline = {}

    def load(self) -> None:
        for key, name, id_col in _LAYER_DEFS:
            self.layers[key] = Layer(key, name, id_col)
        # Static attributes
        sa_path = _APP_DATA / "static_attrs.json"
        if sa_path.exists():
            self.static_attrs = json.loads(sa_path.read_text())
        # CA outline
        ca_path = _APP_DATA / "geojson" / "ca_outline.geojson"
        if ca_path.exists():
            self.ca_outline = json.loads(ca_path.read_text())


state = AppState()
