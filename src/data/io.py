"""Compact storage for per-basin climate and streamflow timeseries.

The training and evaluation pipelines used to keep one CSV per basin per
variable group (~5500 small files, ~13.5 GB on disk).  This module
replaces that layout with a small set of zarr v3 cubes:

    data/training/climate/watersheds.zarr    # 210 USGS gauges
    data/training/climate/huc12.zarr         # in-scope HUC12 manifest
    data/training/flow.zarr                  # 210 USGS gauges + tier coord
    data/eval/climate/{huc8,huc10,huc12}.zarr

Each store is a zarr group containing:

    basin   : (N,)        int64 USGS site numbers, or  variable-length str
                          for HUC scopes
    time    : (T,)        int32 days since 1970-01-01 (np.datetime64[D] view)
    <var>   : (N, T)      float32, NaN where missing
              (climate: precip_mm, tmax_c, tmin_c
               flow:    flow [mm/day], plus tier coord)

Chunks are ``(64, 4096)`` along ``(basin, time)`` with zstd compression,
which gives ~10× compression on this data and cheap per-basin slicing.
The file ``year/month/day`` columns from the legacy CSVs are dropped;
they are derivable from ``time``.

This module deliberately does **not** depend on xarray — the project's
neuralhyd env has a broken xarray import path.  Pure ``zarr`` + numpy is
sufficient.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

import zarr


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CLIMATE_VARS: Tuple[str, ...] = ("precip_mm", "tmax_c", "tmin_c")
FLOW_VAR: str = "flow"

# Chunking for (basin, time).  64 × 4096 ≈ 1 MB at float32 → fast random
# basin reads while staying friendly to bulk scans.
_CHUNKS = (64, 4096)
_CODECS = ("zstd", 3)


def _date_index(time_days: np.ndarray) -> pd.DatetimeIndex:
    """Convert ``int32`` days-since-epoch to a pandas DatetimeIndex."""
    return pd.DatetimeIndex(time_days.astype("datetime64[D]"))


def _days_since_epoch(dates: pd.DatetimeIndex) -> np.ndarray:
    return dates.values.astype("datetime64[D]").astype("int32")


# ---------------------------------------------------------------------------
# Climate cube
# ---------------------------------------------------------------------------


def write_climate_zarr(
    path: Path,
    basin_data: Mapping[object, pd.DataFrame],
    *,
    variables: Sequence[str] = CLIMATE_VARS,
    basin_id_dtype: str = "int64",
    overwrite: bool = True,
) -> None:
    """Write a climate cube from a ``{basin_id: DataFrame}`` mapping.

    Each DataFrame must be indexed by ``date`` (or have a ``date`` column)
    and contain at least the columns listed in ``variables``.  The store is
    written with a unified time axis spanning the union of all basin
    DataFrames; missing days are filled with NaN.

    Parameters
    ----------
    path : output store path (will be created/replaced)
    basin_data : mapping basin_id → DataFrame
    variables : column names to persist (default: precip_mm, tmax_c, tmin_c)
    basin_id_dtype : ``"int64"`` (USGS gauges) or ``"str"`` (HUC scopes)
    overwrite : whether to delete an existing store first
    """
    path = Path(path)
    if overwrite and path.exists():
        import shutil
        shutil.rmtree(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if not basin_data:
        raise ValueError("basin_data is empty")

    # Sort basins for deterministic ordering
    if basin_id_dtype == "int64":
        basins_sorted = sorted(basin_data.keys(), key=int)
        basin_arr = np.asarray([int(b) for b in basins_sorted], dtype=np.int64)
    else:
        basins_sorted = sorted(basin_data.keys(), key=str)
        basin_arr = np.asarray([str(b) for b in basins_sorted], dtype=object)

    # Build unified time axis.  All training-watershed climate spans
    # 1915-01-01 → 2018-12-31; HUC scopes use the same window.  We compute
    # the union to be robust.
    starts: list[pd.Timestamp] = []
    ends: list[pd.Timestamp] = []
    for b in basins_sorted:
        df = basin_data[b]
        idx = df.index if isinstance(df.index, pd.DatetimeIndex) else pd.to_datetime(df["date"])
        starts.append(idx.min())
        ends.append(idx.max())
    start = min(starts)
    end = max(ends)
    time_index = pd.date_range(start, end, freq="D")
    n_basin = len(basins_sorted)
    n_time = len(time_index)

    g = zarr.open_group(str(path), mode="w")
    g.attrs["start_date"] = start.strftime("%Y-%m-%d")
    g.attrs["end_date"] = end.strftime("%Y-%m-%d")
    g.attrs["variables"] = list(variables)
    g.attrs["basin_id_dtype"] = basin_id_dtype

    # basin coord
    if basin_id_dtype == "int64":
        g.create_array("basin", shape=(n_basin,), dtype="int64",
                       chunks=(n_basin,))[:] = basin_arr
    else:
        g.create_array("basin", shape=(n_basin,), dtype=str,
                       chunks=(n_basin,))[:] = basin_arr

    # time coord (int32 days since epoch — easy to interpret, small)
    g.create_array("time", shape=(n_time,), dtype="int32",
                   chunks=(min(n_time, 16384),))[:] = _days_since_epoch(time_index)

    # Pre-build buffers per variable in (n_basin, n_time) layout
    chunk_t = min(n_time, _CHUNKS[1])
    chunk_b = min(n_basin, _CHUNKS[0])

    for var in variables:
        arr = np.full((n_basin, n_time), np.nan, dtype=np.float32)
        for i, b in enumerate(basins_sorted):
            df = basin_data[b]
            if isinstance(df.index, pd.DatetimeIndex):
                idx = df.index
            else:
                idx = pd.to_datetime(df["date"])
            # locate offsets
            offs = (idx - start).days.values.astype(np.int64)
            mask = (offs >= 0) & (offs < n_time)
            arr[i, offs[mask]] = df[var].values[mask].astype(np.float32, copy=False)
        za = g.create_array(
            var, shape=(n_basin, n_time), dtype="float32",
            chunks=(chunk_b, chunk_t),
            compressors=zarr.codecs.BloscCodec(cname="zstd", clevel=3),
        )
        za[:] = arr


def read_climate_zarr(
    path: Path,
    *,
    variables: Sequence[str] | None = None,
) -> tuple[np.ndarray, pd.DatetimeIndex, Dict[str, np.ndarray], str]:
    """Return ``(basins, dates, {var: (N, T) ndarray}, basin_id_dtype)``."""
    path = Path(path)
    g = zarr.open_group(str(path), mode="r")
    basins = g["basin"][:]
    dates = _date_index(g["time"][:])
    if variables is None:
        variables = list(g.attrs.get("variables", CLIMATE_VARS))
    data = {v: g[v][:] for v in variables}
    basin_id_dtype = str(g.attrs.get("basin_id_dtype", "int64"))
    return basins, dates, data, basin_id_dtype


def load_climate_dataframes(
    path: Path,
    basin_ids: Iterable | None = None,
    variables: Sequence[str] | None = None,
) -> Dict[object, pd.DataFrame]:
    """Return ``{basin_id: DataFrame}`` indexed by date.

    Drops trailing all-NaN rows so the returned DataFrame matches the
    shape of the original per-basin CSV for that basin.  Trims at the
    union of variable validity.
    """
    basins, dates, data, basin_id_dtype = read_climate_zarr(path, variables=variables)
    variables = list(data.keys())
    out: Dict[object, pd.DataFrame] = {}

    if basin_ids is not None:
        wanted = set(basin_ids)
    else:
        wanted = None

    is_str = basin_id_dtype != "int64"
    for i, b in enumerate(basins):
        key = str(b) if is_str else int(b)
        if wanted is not None and key not in wanted:
            continue
        rows = {v: data[v][i] for v in variables}
        df = pd.DataFrame(rows, index=dates)
        df.index.name = "date"
        # Drop fully-NaN rows at edges (HUC files may have shorter records)
        valid = df.notna().any(axis=1).values
        if valid.any():
            first = int(np.argmax(valid))
            last = len(valid) - int(np.argmax(valid[::-1]))
            df = df.iloc[first:last]
        out[key] = df
    return out


# ---------------------------------------------------------------------------
# Flow cube
# ---------------------------------------------------------------------------


def write_flow_zarr(
    path: Path,
    flow_data: Mapping[int, pd.DataFrame],
    tier_map: Mapping[int, int] | None = None,
    *,
    overwrite: bool = True,
) -> None:
    """Write a flow cube from ``{basin_id: DataFrame[flow]}``.

    Each DataFrame must be indexed by ``date`` (or have a ``date`` column)
    and contain a ``flow`` column.  ``tier_map`` is optional; if given it
    is stored as a ``tier`` int8 coordinate aligned with ``basin``.
    """
    path = Path(path)
    if overwrite and path.exists():
        import shutil
        shutil.rmtree(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    basins_sorted = sorted(int(b) for b in flow_data.keys())
    starts, ends = [], []
    for b in basins_sorted:
        df = flow_data[b]
        idx = df.index if isinstance(df.index, pd.DatetimeIndex) else pd.to_datetime(df["date"])
        starts.append(idx.min()); ends.append(idx.max())
    start = min(starts)
    end = max(ends)
    time_index = pd.date_range(start, end, freq="D")
    n_basin = len(basins_sorted)
    n_time = len(time_index)

    g = zarr.open_group(str(path), mode="w")
    g.attrs["start_date"] = start.strftime("%Y-%m-%d")
    g.attrs["end_date"] = end.strftime("%Y-%m-%d")
    g.attrs["variables"] = [FLOW_VAR]

    g.create_array("basin", shape=(n_basin,), dtype="int64",
                   chunks=(n_basin,))[:] = np.asarray(basins_sorted, dtype=np.int64)
    g.create_array("time", shape=(n_time,), dtype="int32",
                   chunks=(min(n_time, 16384),))[:] = _days_since_epoch(time_index)

    if tier_map is not None:
        tier_arr = np.asarray([int(tier_map.get(b, 0)) for b in basins_sorted], dtype=np.int8)
        g.create_array("tier", shape=(n_basin,), dtype="int8",
                       chunks=(n_basin,))[:] = tier_arr

    chunk_t = min(n_time, _CHUNKS[1])
    chunk_b = min(n_basin, _CHUNKS[0])
    arr = np.full((n_basin, n_time), np.nan, dtype=np.float32)
    for i, b in enumerate(basins_sorted):
        df = flow_data[b]
        if isinstance(df.index, pd.DatetimeIndex):
            idx = df.index
        else:
            idx = pd.to_datetime(df["date"])
        offs = (idx - start).days.values.astype(np.int64)
        mask = (offs >= 0) & (offs < n_time)
        arr[i, offs[mask]] = df[FLOW_VAR].values[mask].astype(np.float32, copy=False)
    za = g.create_array(
        FLOW_VAR, shape=(n_basin, n_time), dtype="float32",
        chunks=(chunk_b, chunk_t),
        compressors=zarr.codecs.BloscCodec(cname="zstd", clevel=3),
    )
    za[:] = arr


def read_flow_zarr(path: Path) -> tuple[np.ndarray, pd.DatetimeIndex, np.ndarray, np.ndarray | None]:
    """Return ``(basins, dates, flow_arr (N, T), tier_arr | None)``."""
    path = Path(path)
    g = zarr.open_group(str(path), mode="r")
    basins = g["basin"][:]
    dates = _date_index(g["time"][:])
    flow = g[FLOW_VAR][:]
    tier = g["tier"][:] if "tier" in g else None
    return basins, dates, flow, tier


def load_flow_dataframes(
    path: Path,
) -> tuple[Dict[int, pd.DataFrame], Dict[int, int]]:
    """Return ``({basin_id: DataFrame[flow]}, {basin_id: tier})``.

    Trims trailing/leading all-NaN rows per basin to match the shape of
    the legacy per-basin CSV.  Internal NaN gaps are preserved.
    """
    basins, dates, flow, tier = read_flow_zarr(path)
    out_flow: Dict[int, pd.DataFrame] = {}
    out_tier: Dict[int, int] = {}
    for i, b in enumerate(basins):
        bi = int(b)
        series = flow[i]
        valid = ~np.isnan(series)
        if not valid.any():
            continue
        first = int(np.argmax(valid))
        last = len(valid) - int(np.argmax(valid[::-1]))
        df = pd.DataFrame({"flow": series[first:last]}, index=dates[first:last])
        df.index.name = "date"
        out_flow[bi] = df
        if tier is not None:
            out_tier[bi] = int(tier[i])
    return out_flow, out_tier
