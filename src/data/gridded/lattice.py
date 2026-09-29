"""The statewide 1/16° Livneh lattice shared by every gridded product.

Cell centres sit at ``0.03125 + k/16`` degrees — odd multiples of 1/32, so
every centre and every edge is an exact binary fraction and survives any
float round-trip.  The product grid is the bounding box of the 13,786 cells
of the WGEN NonDetrend-Unsplit statewide store:

    lat   32.59375 … 43.34375    173 rows, ascending (south → north)
    lon -124.34375 … -113.90625  168 columns, ascending (west → east)

Cells outside the store are NaN in every NetCDF and 0 in ``mask``.  The
lattice is defined arithmetically here, never from the VICGrids gpkg (its
polygon vertices are off k/16 by up to 1.3e-5°).

Cell keys are ``f"{lat:.5f}_{lon:.5f}"`` — identical to the store's file
names (``data_<key>``).  Values are always placed by key → ``(ilat, ilon)``,
never by row order.

UTM zones (AlphaEarth tiles are per zone): 10N west of -120°, 11N between
-120° and -114°, 12N east of -114°.  Both zone edges are lattice cell edges
(k/16), so no cell straddles a zone.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Grid definition
# ---------------------------------------------------------------------------
CELL_DEG  = 0.0625           # 1/16°
HALF_CELL = 0.03125          # 1/32°; centres at HALF_CELL + k * CELL_DEG
LAT0      = 32.59375         # southernmost row centre
LON0      = -124.34375       # westernmost column centre
NLAT      = 173
NLON      = 168
N_CELLS   = 13_786           # cells in the WGEN NonDetrend-Unsplit statewide store

# Daily record of the store (all 26 Feb-29s included; 37,986 = 26 × 1461).
START_DATE = "1915-01-01"
END_DATE   = "2018-12-31"
N_DAYS     = 37_986
TIME_UNITS = f"days since {START_DATE}"
CALENDAR   = "standard"

#: UTM zone edges (degrees east) and the zone number on each side.
ZONE_EDGES = (-120.0, -114.0)
ZONE_NUMBERS = (10, 11, 12)

_STORE_PREFIX = "data_"


# ---------------------------------------------------------------------------
# Axes, keys and indices
# ---------------------------------------------------------------------------
def lat_axis() -> np.ndarray:
    """Row centres, ascending, float64 (exact)."""
    return LAT0 + CELL_DEG * np.arange(NLAT, dtype=np.float64)


def lon_axis() -> np.ndarray:
    """Column centres, ascending, float64 (exact)."""
    return LON0 + CELL_DEG * np.arange(NLON, dtype=np.float64)


def axis_bounds(centres: np.ndarray) -> np.ndarray:
    """``(n, 2)`` cell edges from centres (± 1/32°, exact)."""
    c = np.asarray(centres, dtype=np.float64)
    return np.stack([c - HALF_CELL, c + HALF_CELL], axis=1)


def time_axis() -> pd.DatetimeIndex:
    """The daily 1915-01-01 → 2018-12-31 record."""
    t = pd.date_range(START_DATE, END_DATE, freq="D")
    assert len(t) == N_DAYS
    return t


def cell_key(lat: float, lon: float) -> str:
    return f"{lat:.5f}_{lon:.5f}"


def on_lattice(v: np.ndarray | float) -> np.ndarray:
    """True where ``v`` is a cell centre (``0.03125 + k/16``)."""
    k = (np.asarray(v, dtype=np.float64) - HALF_CELL) / CELL_DEG
    return np.abs(k - np.rint(k)) < 1e-9


def parse_key(key: str) -> tuple[float, float]:
    """``"<lat>_<lon>"`` → (lat, lon); raises if off the lattice."""
    lat_s, lon_s = str(key).split("_")
    lat, lon = float(lat_s), float(lon_s)
    if not (on_lattice(lat) and on_lattice(lon)):
        raise ValueError(f"cell key {key!r} is not on the 1/16° lattice")
    return lat, lon


def grid_index(lat: np.ndarray | float, lon: np.ndarray | float) -> tuple[np.ndarray, np.ndarray]:
    """Dense ``(ilat, ilon)`` of cell centres; raises outside the grid."""
    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)
    if not (on_lattice(lat).all() and on_lattice(lon).all()):
        raise ValueError("coordinates off the 1/16° lattice")
    ilat = np.rint((lat - LAT0) / CELL_DEG).astype(np.int64)
    ilon = np.rint((lon - LON0) / CELL_DEG).astype(np.int64)
    if ilat.min() < 0 or ilat.max() >= NLAT or ilon.min() < 0 or ilon.max() >= NLON:
        raise ValueError("coordinates outside the statewide grid")
    return ilat, ilon


def utm_zone(lon: np.ndarray | float) -> np.ndarray:
    """UTM zone number (10/11/12) of each cell centre."""
    lon = np.asarray(lon, dtype=np.float64)
    return np.select([lon < ZONE_EDGES[0], lon < ZONE_EDGES[1]],
                     [ZONE_NUMBERS[0], ZONE_NUMBERS[1]], ZONE_NUMBERS[2]).astype(np.int8)


# ---------------------------------------------------------------------------
# Cell list
# ---------------------------------------------------------------------------
def store_filename(key: str) -> str:
    """Forcing-store file name of a cell (``data_<lat>_<lon>``)."""
    return f"{_STORE_PREFIX}{key}"


def _cells_frame(lat: np.ndarray, lon: np.ndarray) -> pd.DataFrame:
    ilat, ilon = grid_index(lat, lon)
    df = pd.DataFrame({
        "key": [cell_key(a, b) for a, b in zip(lat, lon)],
        "lat": lat, "lon": lon, "ilat": ilat, "ilon": ilon,
        "utm_zone": utm_zone(lon),
    })
    df = df.sort_values(["lat", "lon"], kind="mergesort").reset_index(drop=True)
    if df["key"].duplicated().any():
        raise ValueError("duplicate cell keys")
    # Zone edges must be cell edges: no cell may straddle one.
    for edge in ZONE_EDGES:
        straddle = (df["lon"] - HALF_CELL < edge) & (df["lon"] + HALF_CELL > edge)
        if straddle.any():
            raise ValueError(f"{int(straddle.sum())} cells straddle the UTM edge {edge}°")
    return df


def scan_meteo_dir(meteo_dir: str | os.PathLike) -> pd.DataFrame:
    """The cell list of the forcing store: one row per ``data_<key>`` file.

    Columns ``key, lat, lon, ilat, ilon, utm_zone``, sorted by (lat, lon).
    Non-``data_`` entries (e.g. an editor's ``.vscode``) are ignored.  Every
    file name must round-trip through :func:`cell_key`.
    """
    names = sorted(n for n in os.listdir(meteo_dir) if n.startswith(_STORE_PREFIX))
    if not names:
        raise FileNotFoundError(f"no {_STORE_PREFIX}* files in {meteo_dir}")
    keys = [n[len(_STORE_PREFIX):] for n in names]
    ll = np.array([parse_key(k) for k in keys], dtype=np.float64)
    df = _cells_frame(ll[:, 0], ll[:, 1])
    drift = set(df["key"]) ^ set(keys)
    if drift:
        raise ValueError(f"{len(drift)} store file names do not round-trip through "
                         f"cell_key, e.g. {sorted(drift)[:3]}")
    return df


def write_grid_csv(cells: pd.DataFrame, path: str | os.PathLike) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cells.to_csv(path, index=False, lineterminator="\n")


def read_grid_csv(path: str | os.PathLike) -> pd.DataFrame:
    """Read a cell list written by :func:`write_grid_csv`; re-validates it."""
    raw = pd.read_csv(path, dtype={"key": str})
    df = _cells_frame(raw["lat"].to_numpy(np.float64), raw["lon"].to_numpy(np.float64))
    if not df["key"].equals(raw["key"].reset_index(drop=True)):
        raise ValueError(f"{path}: keys disagree with lat/lon or are not (lat, lon)-sorted")
    return df


def mask_from_cells(cells: pd.DataFrame) -> np.ndarray:
    """``(NLAT, NLON)`` uint8: 1 on store cells, 0 elsewhere."""
    m = np.zeros((NLAT, NLON), dtype=np.uint8)
    m[cells["ilat"].to_numpy(), cells["ilon"].to_numpy()] = 1
    return m
