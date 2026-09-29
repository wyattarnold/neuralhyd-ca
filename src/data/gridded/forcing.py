"""Statewide daily forcing: WGEN NonDetrend-Unsplit store → one NetCDF per variable.

Input
-----
The DWR WGEN NonDetrend-Unsplit statewide store: 13,786 TAB-delimited CRLF
ASCII files ``data_<lat>_<lon>`` with columns ``year month day precip_mm
tmax_c tmin_c`` and 37,986 daily rows, 1915-01-01 … 2018-12-31 including
every Feb 29.  Precip carries 3 decimals and temperatures carry 2.  The store is
bit-identical to the one the current neuralhyd-ca training climate was built
from.  It is the ``Historical_Unsplit`` baseline of DWR's "Gridded Weather
Generator Perturbations …" release, meaning the historical record *without*
temperature detrending.  Precipitation is the Livneh-lineage unsplit product
(Pierce et al. 2021).  Temperature is Livneh et al. (2013) for 1915-2015 and a
PRISM-based extension for 2016-2018.

Corrections
-----------
The code changes only these two things.  Every other value is the ASCII
value parsed exactly and cast to float32.

1. **x10 misplaced-decimal precip spikes.**  This is DWR's upstream rule
   applied statewide: a day in June, July or August (``X10_MONTHS``) with raw
   precip >= 150.0 mm (``X10_THRESHOLD_MM``) is divided by 10.  The divide is
   done in float64 on the parsed value and the result is then cast to float32.
   Statewide it hits 620 cell-days on 445 cells over 44 dates.  The largest
   JJA value left alone is 149.904 mm and the smallest corrected one is
   150.041 mm.  :func:`check_x10_product_a` gates the rule statewide against
   DWR WGEN Product A, which applies the same correction upstream and stores
   ``round(raw/10, 2)``: the full run matched the 620 rule days to Product
   A's 620 corrections exactly, and found one Product A edit that is not the
   rule (``PA_KNOWN_NON_RULE``, kept raw here).  The rule is a correction
   *convention*, not a measurement.  It cannot tell a real storm from an
   artefact: 1977-08-16/17 is divided anyway, although it plausibly overlaps a
   real tropical-storm remnant.  It also leaves 100-150 mm values on the
   artefact dates alone.  Non-summer extremes are never touched, for example
   814.6 mm at 37.59375_-118.21875 on 1967-01-25.  That is why the data
   variables carry no ``valid_range``.
2. **Inverted temperatures.**  About 181,660 cell-days (0.035 %) have
   tmin > tmax, and they are worst on the coast: 38.15625_-122.96875 has
   6.1 % of its days inverted.  These days are swapped so that tmin = min and
   tmax = max, which leaves the daily mean unchanged.  Per-cell counts are in
   ``swap_count``, which is identical in both temperature files.

Temperature source seam
-----------------------
From 1915 to 2015 the temperature is Livneh.  From 2016 to 2018 it is a
PRISM-based extension; the WGEN README says "Livneh temperature (1915-2015)
corrected to PRISM observations (2016-2018)".  The inversions drop to about
zero from 2016-01-01.  The counts before and after the seam are recorded in
the file attributes and in the provenance.

Output (``out_dir``, default ``data/gridded``)
----------------------------------------------
::

    livneh_precip_mm_daily_1915-2018.nc  precip_mm(time, lat, lon), x10 pair
                                         table (dim x10_pair), x10_count(lat, lon)
    livneh_tmax_c_daily_1915-2018.nc     tmax_c(time, lat, lon), swap_count(lat, lon)
    livneh_tmin_c_daily_1915-2018.nc     tmin_c(time, lat, lon), swap_count(lat, lon)
    precip_x10_corrections.csv           key,lat,lon,date,raw_mm,corrected_mm
    grid_cells.csv                       the store's cell list (lattice.scan_meteo_dir)
    SHA256SUMS, provenance.toml          table ["forcing"]

There is one file per variable so that each stays under GitHub LFS's 2 GB
cap; they come to roughly 0.47, 1.25 and 1.3 GB.  Values are float32 with
zlib(4) + shuffle and NaN outside the store cells (``mask == 0``).  Chunks are
(time=1461, lat=16, lon=16).  1461 days is four years and divides the record
exactly (26 x 1461 = 37,986).  A 16 x 16 cell chunk balances a per-cell series
read (~0.15-0.2 s) against a statewide per-day map (~0.5 s).

Build
-----
The grid is processed in 16-row bands aligned to the chunk grid (0-16, 16-32,
…, 160-173).  For each band, a thread pool reads, hashes, parses and corrects
the band's files.  pandas' C parser releases the GIL, so this runs at about
4 ms per file on 8 threads.  The results are scattered into one float32
``(time, rows, lon)`` slab per variable, and each slab is written with a
single ``var[:, r0:r1, :] = slab``.  While one band is compressed and written,
the next band parses and a separate thread hashes the finished slabs.  Peak
memory is about 2.5 GB, two bands of 1.2 GB.

Parsing uses pandas' C engine at its default "high" precision.  For these
tokens that is exact: every value is an integer of at most 7 digits scaled by
10^-k with k <= 3, which the parser computes as an exact integer divided by
an exact power of ten.  That division is correctly rounded, so the result is
bitwise equal to ``float_precision="round_trip"`` (checked on sampled files).

Hashes
------
content_sha256 (per variable, independent of the library)
    sha256 over the float32 little-endian values in (lat, lon, time) C order
    across the whole 173 x 168 grid.  That is each cell's full daily series in
    turn, with cells in ascending (lat, lon) order.  NaN cells are included
    as the canonical quiet NaN 0x7fc00000.  The file's own sha256 changes on
    every rebuild (``history``, ``date_created``, the HDF5/netCDF-C build);
    this hash does not.
input_tree_sha256
    sha256 over ``"".join(f"{filename} {sha256_hex}\\n")`` of the store files
    read, sorted by file name.

Usage (via ``scripts/prepare_gridded.py``)
------------------------------------------
::

    python scripts/prepare_gridded.py forcing --meteo-dir <store>
    python scripts/prepare_gridded.py forcing --meteo-dir <store> --rows 96:112 --out-dir <scratch>
    python scripts/prepare_gridded.py check-x10 --meteo-dir <store> --product-a-dir <PA dir> --sample 40
    python scripts/prepare_gridded.py verify --meteo-dir <store> [--full]
"""
from __future__ import annotations

import hashlib
import io
import os
import time
from collections import defaultdict
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

import netCDF4
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.data.gridded import lattice as L
from src.data.gridded import ncio
from src.paths import (
    GRIDDED_DIR,
    GRIDDED_FORCING_NC,
    GRIDDED_GRID_CSV,
    GRIDDED_PROVENANCE,
    GRIDDED_SHA256SUMS,
    GRIDDED_X10_CSV,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
VARS: tuple[str, ...] = ("precip_mm", "tmax_c", "tmin_c")
X10_MONTHS = (6, 7, 8)
X10_THRESHOLD_MM = 150.0

X10_RULE = (f"precip_mm divided by 10 where month in {X10_MONTHS} and raw precip >= "
            f"{X10_THRESHOLD_MM} mm (DWR upstream misplaced-decimal rule; float64 "
            f"divide of the parsed ASCII value, then float32 cast)")

CHUNKS = (1461, 16, 16)                  # (time, lat, lon); 26 x 1461 = N_DAYS
BAND_ROWS = CHUNKS[1]                    # build/verify bands follow the chunk rows
SEAM_DATE = "2016-01-01"                 # Livneh → PRISM-extension temperature seam

# Store columns: year month day precip_mm tmax_c tmin_c
_C_MONTH, _C_PRECIP, _C_TMAX, _C_TMIN = 1, 3, 4, 5
_N_COLS = 6

_VAR_ATTRS: dict[str, dict[str, str]] = {
    "precip_mm": {"long_name": "daily precipitation",
                  "standard_name": "lwe_thickness_of_precipitation_amount",
                  "units": "mm", "cell_methods": "time: sum area: mean"},
    "tmax_c": {"long_name": "daily maximum air temperature",
               "standard_name": "air_temperature",
               "units": "degC", "cell_methods": "time: maximum area: mean"},
    "tmin_c": {"long_name": "daily minimum air temperature",
               "standard_name": "air_temperature",
               "units": "degC", "cell_methods": "time: minimum area: mean"},
}

REFERENCES = (
    "Livneh, B., et al. (2013), A long-term hydrologically based dataset of land surface "
    "fluxes and states for the conterminous United States: Update and extensions, "
    "J. Climate 26, 9384-9392; "
    "Pierce, D. W., et al. (2021), An extreme-preserving long-term gridded daily "
    "precipitation dataset for the conterminous United States, J. Hydrometeorology "
    "22(7), 1883-1895; "
    "PRISM Climate Group, Oregon State University; "
    "California Department of Water Resources, \"Gridded Weather Generator Perturbations "
    "of Historical Detrended and Stochastically Generated Temperature and Precipitation "
    "for the State of CA and HUC8s\", https://data.ca.gov/dataset/gridded-weather-generator-"
    "perturbations-of-historical-detrended-and-stochastically-generated-te "
    "(Historical_Unsplit = non-temperature-detrended historical baseline)"
)

CONTENT_HASH_DEFINITION = (
    "sha256 over the float32 little-endian values of the variable in (lat, lon, time) "
    "C order over the whole 173 x 168 grid including NaN cells (canonical quiet NaN "
    "0x7fc00000), i.e. each cell's full daily series in turn, cells in ascending "
    "(lat, lon) order; independent of the HDF5/netCDF-C build"
)
INPUT_TREE_HASH_DEFINITION = (
    'sha256 over "".join(f"{filename} {sha256_hex}\\n") of the store files read, '
    "sorted by file name"
)

# Product A check (DWR WGEN Product A stores round(raw/10, 2) on x10 days and
# round(raw, 2) elsewhere; its temperatures are detrended and ignored).
_PA_PREFIX = "meteo_"
_PA_TOL = 0.0051        # |PA - expected| allowed (0.01 mm rounding + float slack)
_PA_CORR_TOL = 0.006    # |PA - raw| above this counts as a PA correction
_CHECK_WORKERS = 8

_NAN_BLOCK = np.full(1 << 20, np.nan, dtype=np.float32).tobytes()


# ---------------------------------------------------------------------------
# One store file
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _ymd() -> np.ndarray:
    """``(N_DAYS, 3)`` float64 year/month/day of the lattice time axis."""
    t = L.time_axis()
    return np.stack([t.year, t.month, t.day], axis=1).astype(np.float64)


@lru_cache(maxsize=1)
def _seam_index() -> int:
    return int((pd.Timestamp(SEAM_DATE) - pd.Timestamp(L.START_DATE)).days)


def read_store_file(path) -> tuple[bytes, np.ndarray]:
    """Read one store file → (raw bytes, float64 array (N_DAYS, 6)). Validates: exactly N_DAYS rows x 6 cols,
    year/month/day exactly equal lattice.time_axis(), all finite, precip >= 0. Raises ValueError naming the file."""
    path = Path(path)
    raw = path.read_bytes()
    try:
        df = pd.read_csv(io.BytesIO(raw), sep="\t", header=None, dtype=np.float64,
                         engine="c")
    except Exception as exc:                                  # noqa: BLE001
        raise ValueError(f"{path}: unparseable ({type(exc).__name__}: {exc})") from exc
    values = df.to_numpy()
    if values.shape != (L.N_DAYS, _N_COLS):
        raise ValueError(f"{path}: shape {values.shape}, expected ({L.N_DAYS}, {_N_COLS})")
    if not np.isfinite(values).all():
        bad = np.argwhere(~np.isfinite(values))[0]
        raise ValueError(f"{path}: non-finite value at row {bad[0]} col {bad[1]}")
    ymd = _ymd()
    if not np.array_equal(values[:, :3], ymd):
        row = int(np.flatnonzero((values[:, :3] != ymd).any(axis=1))[0])
        raise ValueError(f"{path}: date {values[row, :3].astype(int).tolist()} at row {row}, "
                         f"expected {ymd[row].astype(int).tolist()}")
    if (values[:, _C_PRECIP] < 0).any():
        row = int(np.flatnonzero(values[:, _C_PRECIP] < 0)[0])
        raise ValueError(f"{path}: negative precip {values[row, _C_PRECIP]} at row {row}")
    return raw, values


def _x10_mask(values: np.ndarray) -> np.ndarray:
    return np.isin(values[:, _C_MONTH], X10_MONTHS) & (values[:, _C_PRECIP] >= X10_THRESHOLD_MM)


def correct_cell(values: np.ndarray) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """float64 (N_DAYS,6) → ({var: float32 (N_DAYS,)}, x10_day_index int array, swapped bool (N_DAYS,)).
    Applies the x10 rule (float64 divide, then float32 cast) and the tmin/tmax swap."""
    if values.shape != (L.N_DAYS, _N_COLS) or values.dtype != np.float64:
        raise ValueError(f"expected float64 ({L.N_DAYS}, {_N_COLS}), got "
                         f"{values.dtype} {values.shape}")
    rule = _x10_mask(values)
    precip = values[:, _C_PRECIP].copy()
    precip[rule] = precip[rule] / 10.0
    tmax, tmin = values[:, _C_TMAX], values[:, _C_TMIN]
    swapped = tmin > tmax
    out = {
        "precip_mm": precip.astype(np.float32),
        "tmax_c": np.maximum(tmax, tmin).astype(np.float32),
        "tmin_c": np.minimum(tmax, tmin).astype(np.float32),
    }
    return out, np.flatnonzero(rule), swapped


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def _bands() -> list[tuple[int, int]]:
    return [(r0, min(r0 + BAND_ROWS, L.NLAT)) for r0 in range(0, L.NLAT, BAND_ROWS)]


def _parse_rows(rows: tuple[int, int] | str | None) -> tuple[int, int] | None:
    if rows is None:
        return None
    if isinstance(rows, str):
        a, b = rows.split(":")
        rows = (int(a), int(b))
    a, b = int(rows[0]), int(rows[1])
    if not 0 <= a < b <= L.NLAT:
        raise ValueError(f"rows {a}:{b} outside 0:{L.NLAT} or empty")
    return a, b


def _out_paths(out_dir: Path) -> dict[str, Path]:
    return {var: out_dir / GRIDDED_FORCING_NC[var].name for var in VARS}


def _fmt(v: float | np.floating) -> str:
    """Shortest decimal that round-trips the value at its own precision."""
    return np.format_float_positional(v, unique=True, trim="-")


def _feed_nan(h: "hashlib._Hash", n_values: int) -> None:
    """Feed ``n_values`` canonical float32 NaNs into ``h`` (an all-NaN band)."""
    per = len(_NAN_BLOCK) // 4
    full, rest = divmod(n_values, per)
    for _ in range(full):
        h.update(_NAN_BLOCK)
    if rest:
        h.update(_NAN_BLOCK[: rest * 4])


def _hash_slab(h: "hashlib._Hash", slab: np.ndarray) -> None:
    """Feed a ``(time, rows, lon)`` slab into ``h`` in (lat, lon, time) order.

    Row by row it produces the same bytes as ``slab.transpose(1, 2, 0)`` but
    needs only about 25 MB of temporaries per row instead of two slab copies.
    """
    for r in range(slab.shape[1]):
        ncio.sha256_array(slab[:, r, :].T, h)


def _utc_date() -> str:
    return datetime.now(timezone.utc).date().isoformat()


def _date_str(day_index: int) -> str:
    return (pd.Timestamp(L.START_DATE) + pd.Timedelta(days=int(day_index))).date().isoformat()


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
@dataclass
class _CellResult:
    key: str
    ilat: int
    ilon: int
    sha256: str
    x10_idx: np.ndarray        # day indices (int64)
    x10_raw: np.ndarray        # float64 raw precip on those days
    x10_corr: np.ndarray       # float32 stored precip on those days
    n_swap: int
    n_swap_post: int           # swaps on/after SEAM_DATE
    n_equal: int               # days with tmin == tmax


def _process_cell(meteo_dir: Path, key: str, ilat: int, ilon: int, r0: int,
                  slabs: dict[str, np.ndarray]) -> _CellResult:
    """Worker: read → sha256 → parse → correct → scatter into the band slabs."""
    raw, values = read_store_file(meteo_dir / L.store_filename(key))
    digest = hashlib.sha256(raw).hexdigest()
    del raw
    out, x10, swapped = correct_cell(values)
    for var in VARS:
        slabs[var][:, ilat - r0, ilon] = out[var]
    return _CellResult(
        key=key, ilat=ilat, ilon=ilon, sha256=digest,
        x10_idx=x10, x10_raw=values[x10, _C_PRECIP].copy(),
        x10_corr=out["precip_mm"][x10].copy(),
        n_swap=int(swapped.sum()), n_swap_post=int(swapped[_seam_index():].sum()),
        n_equal=int((values[:, _C_TMAX] == values[:, _C_TMIN]).sum()),
    )


def _submit_band(ex: ThreadPoolExecutor, meteo_dir: Path, band_cells: pd.DataFrame,
                 r0: int, r1: int) -> tuple[dict[str, np.ndarray], list[Future]]:
    slabs = {var: np.full((L.N_DAYS, r1 - r0, L.NLON), np.nan, dtype=np.float32)
             for var in VARS}
    futs = [ex.submit(_process_cell, meteo_dir, key, int(i), int(j), r0, slabs)
            for key, i, j in zip(band_cells["key"], band_cells["ilat"], band_cells["ilon"])]
    return slabs, futs


def _global_attrs(var: str, *, meteo_dir: Path, date_created: str, git: dict[str, str],
                  command: str, subset: tuple[int, int] | None, n_cells: int) -> dict[str, Any]:
    long = _VAR_ATTRS[var]["long_name"]
    dirty = ", dirty tree" if git.get("git_dirty") == "yes" else ""
    attrs: dict[str, Any] = {
        "title": f"Livneh 1/16-degree {long} for California, 1915-2018 "
                 f"(WGEN NonDetrend-Unsplit historical baseline)",
        "summary": (
            f"{long.capitalize()} ({var}) on the dense 173 x 168 1/16-degree lat/lon "
            f"lattice covering {n_cells} of the {L.N_CELLS} cells of the DWR WGEN "
            f"NonDetrend-Unsplit statewide store, {L.START_DATE} to {L.END_DATE} "
            f"({L.N_DAYS} days incl. Feb 29). "
            f"Values are the store's ASCII values cast to float32 after two corrections "
            f"(x10 misplaced-decimal summer precip spikes divided by 10; tmin > tmax days "
            f"swapped). NaN outside the store cells (mask == 0). One file per variable: "
            f"precip_mm, tmax_c, tmin_c."),
        "source": (
            f"DWR WGEN NonDetrend-Unsplit statewide store ({meteo_dir.name}): "
            f"{L.N_CELLS} TAB-delimited ASCII files data_<lat>_<lon> with columns year "
            f"month day precip_mm tmax_c tmin_c. Precipitation: Livneh-lineage unsplit "
            f"daily precipitation (Pierce et al. 2021). Temperature: Livneh et al. (2013) "
            f"1915-2015, PRISM-based extension 2016-2018."),
        "references": REFERENCES,
        "license": ("The terms of the source datasets apply (see references); this "
                    "derived file asserts no terms of its own."),
        "history": (f"{date_created}: {command} (neuralhyd-ca git "
                    f"{git.get('git_commit') or 'unknown'}{dirty})"),
        "date_created": date_created,
        "time_coverage_start": L.START_DATE,
        "time_coverage_end": L.END_DATE,
        "time_coverage_resolution": "P1D",
        "comment": (f"GDAL-based readers (QGIS, gdal_translate, rasterio, "
                    f"rioxarray.open_rasterio, R terra) turn each of the {L.N_DAYS} days "
                    f"into a band and by default stop at 32768 bands (2004-09-17) with "
                    f"only a log warning: set GDAL_MAX_BAND_COUNT=65536.  xarray and "
                    f"netCDF4 are unaffected."),
    }
    if subset is not None:
        attrs["subset_rows"] = f"{subset[0]}:{subset[1]}"
    return attrs


def _create_var(ds: netCDF4.Dataset, var: str) -> netCDF4.Variable:
    v = ds.createVariable(var, "f4", ("time", "lat", "lon"), chunksizes=CHUNKS,
                          fill_value=np.float32(np.nan), **ncio.ZLIB)
    ncio.set_attrs(v, {**_VAR_ATTRS[var], "grid_mapping": "crs"})
    v.set_auto_maskandscale(False)     # write the NaN slabs as-is (no masked_invalid pass)
    return v


def _existing_build_problem(out_dir: Path, finals: dict[str, Path],
                            subset: tuple[int, int] | None) -> str:
    """``""`` when the existing products are one complete build that
    SHA256SUMS and provenance [forcing] both describe (for the requested
    rows); otherwise what is wrong.  Cheap checks first, then the sha256 of
    every file (a few seconds for the full ~3 GB)."""
    sums = ncio.read_sha256sums(out_dir / GRIDDED_SHA256SUMS.name)
    prov = ncio.read_provenance(out_dir / GRIDDED_PROVENANCE.name).get("forcing", {})
    if not prov:
        return f"{GRIDDED_PROVENANCE.name} has no [forcing] table"
    want_rows = f"{subset[0]}:{subset[1]}" if subset else None
    if prov.get("subset_rows") != want_rows:
        return (f"they were built for rows {prov.get('subset_rows') or 'all'}, "
                f"requested {want_rows or 'all'}")
    pfiles = {f.get("name"): f for f in prov.get("files", {}).values() if isinstance(f, dict)}
    paths = [*finals.values(), out_dir / GRIDDED_X10_CSV.name]
    for p in paths:
        f = pfiles.get(p.name)
        if not p.exists():
            return f"{p.name} is missing"
        if f is None or p.name not in sums:
            return f"{p.name} is not listed in {GRIDDED_SHA256SUMS.name} and {GRIDDED_PROVENANCE.name}"
        if f.get("sha256") != sums[p.name]:
            return f"{GRIDDED_SHA256SUMS.name} and {GRIDDED_PROVENANCE.name} disagree on {p.name}"
        if f.get("bytes") != p.stat().st_size:
            return f"{p.name} has {p.stat().st_size} bytes, provenance says {f.get('bytes')}"
    print(f"Checking the existing forcing files against {GRIDDED_SHA256SUMS.name} ...")
    for p in paths:
        if ncio.sha256_file(p) != sums[p.name]:
            return f"{p.name} does not match {GRIDDED_SHA256SUMS.name}"
    return ""


def build_forcing(meteo_dir, *, out_dir=GRIDDED_DIR, rows: tuple[int, int] | None = None, workers: int = 8,
                  force: bool = False) -> dict:
    """Build the three forcing NetCDFs (+ x10 CSV, grid_cells.csv, sums, provenance).

    ``rows=(a, b)`` is a smoke-test mode.  Only cells with ``a <= ilat < b``
    are written, the grid stays the full 173 x 168 with NaN elsewhere,
    ``mask`` marks the written cells only, and every file carries
    ``subset_rows = "a:b"``.  It refuses to write into ``data/gridded``.

    Without ``force`` an existing set is kept (summary ``skipped=True``) only
    if SHA256SUMS and provenance [forcing] describe exactly these files and
    rows; an inconsistent set (an interrupted or mixed build) is an error
    that asks for ``--force``.  Git LFS pointer stubs count as missing.  The
    outputs are written and hashed as ``.part`` files and then moved into
    place together (:func:`ncio.finalize_all`).
    """
    meteo_dir = Path(meteo_dir)
    out_dir = Path(out_dir)
    subset = _parse_rows(rows)
    if subset is not None and ncio.is_repo_gridded_dir(out_dir):
        raise SystemExit(f"--rows is a smoke-test mode; refusing to write into {GRIDDED_DIR} "
                         f"(pass --out-dir)")

    finals = _out_paths(out_dir)
    stubs = [p.name for p in finals.values() if ncio.is_lfs_pointer(p)]
    if stubs:
        print(f"{', '.join(stubs)} in {out_dir} are Git LFS pointer stubs, not products; "
              f"rebuilding over them.  (To fetch the committed files instead: "
              f"{ncio.LFS_PULL_HINT})")
    elif not force and all(p.exists() for p in finals.values()):
        problem = _existing_build_problem(out_dir, finals, subset)
        if problem:
            raise SystemExit(f"Forcing NetCDFs exist in {out_dir} but {problem}: an "
                             f"interrupted or mixed build.  Pass --force to rebuild "
                             f"(`prepare_gridded.py verify` shows what is wrong).")
        print(f"Forcing NetCDFs already exist in {out_dir} — pass --force to regenerate")
        return {"skipped": True, "out_dir": str(out_dir),
                "files": {var: str(p) for var, p in finals.items()}}

    t_start = time.perf_counter()
    print(f"Scanning {meteo_dir} ...")
    all_cells = L.scan_meteo_dir(meteo_dir)
    if subset is None and len(all_cells) != L.N_CELLS:
        raise SystemExit(f"{meteo_dir}: {len(all_cells)} store files, expected {L.N_CELLS} "
                         f"(full build needs the whole store; --rows for a smoke test)")
    out_dir.mkdir(parents=True, exist_ok=True)

    # grid_cells.csv: the full store scan (also in subset mode, for verify).
    grid_csv = out_dir / GRIDDED_GRID_CSV.name
    if grid_csv.exists():
        existing = L.read_grid_csv(grid_csv)
        if not existing["key"].equals(all_cells["key"]):
            raise SystemExit(f"{grid_csv} differs from the scan of {meteo_dir} "
                             f"({len(existing)} vs {len(all_cells)} cells); regenerate it "
                             f"with `prepare_gridded.py grid` first")
    else:
        L.write_grid_csv(all_cells, grid_csv)
        print(f"Wrote {grid_csv} ({len(all_cells)} cells)")

    if subset is None:
        cells = all_cells
    else:
        a, b = subset
        cells = all_cells[(all_cells["ilat"] >= a) & (all_cells["ilat"] < b)].reset_index(drop=True)
        if cells.empty:
            raise SystemExit(f"no store cells in rows {a}:{b}")
    print(f"{len(cells)} cells" + (f" (subset rows {subset[0]}:{subset[1]})" if subset else "")
          + f", {L.N_DAYS} days, {workers} workers -> {out_dir}")

    date_created = _utc_date()
    git = ncio.git_state()
    command = ncio.command_line()
    datasets: dict[str, netCDF4.Dataset] = {}
    ncvars: dict[str, netCDF4.Variable] = {}
    hashes = {var: hashlib.sha256() for var in VARS}
    results: list[_CellResult] = []
    band_log: list[dict[str, Any]] = []

    ex = ThreadPoolExecutor(max_workers=workers)
    hasher = ThreadPoolExecutor(max_workers=1)      # FIFO → per-variable band order
    ok = False
    try:
        for var in VARS:
            ds = ncio.create_dataset(
                ncio.part_path(finals[var]), cells, n_time=L.N_DAYS,
                global_attrs=_global_attrs(var, meteo_dir=meteo_dir, date_created=date_created,
                                           git=git, command=command, subset=subset,
                                           n_cells=len(cells)))
            datasets[var] = ds
            ncvars[var] = _create_var(ds, var)

        bands = _bands()
        band_cells = {rb: cells[(cells["ilat"] >= rb[0]) & (cells["ilat"] < rb[1])]
                      for rb in bands}
        todo = [rb for rb in bands if len(band_cells[rb])]
        pending = {todo[0]: _submit_band(ex, meteo_dir, band_cells[todo[0]], *todo[0])}
        hash_futs: list[Future] = []
        for r0, r1 in bands:
            n_band = len(band_cells[(r0, r1)])
            if n_band == 0:                       # subset mode: NaN-only band
                for var in VARS:
                    hash_futs.append(hasher.submit(_feed_nan, hashes[var],
                                                   L.N_DAYS * (r1 - r0) * L.NLON))
                continue
            t0 = time.perf_counter()
            slabs, futs = pending.pop((r0, r1))
            band_results = [f.result() for f in futs]
            t1 = time.perf_counter()
            for f in hash_futs:                   # bound memory: previous band hashed
                f.result()
            hash_futs = []
            t2 = time.perf_counter()
            k = todo.index((r0, r1))
            if k + 1 < len(todo):                 # parse the next band while writing
                nxt = todo[k + 1]
                pending[nxt] = _submit_band(ex, meteo_dir, band_cells[nxt], *nxt)
            for var in VARS:
                ncvars[var][:, r0:r1, :] = slabs[var]
                hash_futs.append(hasher.submit(_hash_slab, hashes[var], slabs[var]))
            del slabs
            t3 = time.perf_counter()
            results.extend(band_results)
            band_log.append({"rows": f"{r0}:{r1}", "cells": n_band,
                             "parse_wait_s": round(t1 - t0, 2), "hash_wait_s": round(t2 - t1, 2),
                             "write_s": round(t3 - t2, 2)})
            print(f"  band {r0:3d}-{r1:3d}  {n_band:5d} cells  parse-wait {t1 - t0:6.1f}s  "
                  f"hash-wait {t2 - t1:5.1f}s  write {t3 - t2:6.1f}s  "
                  f"({t3 - t_start:7.1f}s elapsed)")
        t_h = time.perf_counter()
        for f in hash_futs:
            f.result()
        t_hash_tail = time.perf_counter() - t_h

        # --- per-cell tables -------------------------------------------------
        results.sort(key=lambda r: (r.ilat, r.ilon))
        x10_count = np.zeros((L.NLAT, L.NLON), dtype=np.int16)
        swap_count = np.zeros((L.NLAT, L.NLON), dtype=np.int32)
        pairs = []
        for r in results:
            x10_count[r.ilat, r.ilon] = len(r.x10_idx)
            swap_count[r.ilat, r.ilon] = r.n_swap
            for t, raw_v, corr_v in zip(r.x10_idx, r.x10_raw, r.x10_corr):
                pairs.append((int(t), r.ilat, r.ilon, r.key, float(raw_v), corr_v))
        pairs.sort(key=lambda p: (p[0], p[1], p[2]))
        n_pairs = len(pairs)
        x10_cells = len({p[3] for p in pairs})
        x10_dates = len({p[0] for p in pairs})
        n_swap = int(swap_count.sum())
        n_swap_post = sum(r.n_swap_post for r in results)
        swap_cells = int((swap_count > 0).sum())
        n_equal = sum(r.n_equal for r in results)
        tree = hashlib.sha256("".join(
            f"{L.store_filename(r.key)} {r.sha256}\n"
            for r in sorted(results, key=lambda r: L.store_filename(r.key))).encode("utf-8"))
        input_tree_sha256 = tree.hexdigest()

        corrections = (
            f"x10: {n_pairs} cell-days on {x10_cells} cells ({x10_dates} dates) where month in "
            f"{X10_MONTHS} and raw precip >= {X10_THRESHOLD_MM} mm were divided by 10 (DWR "
            f"upstream misplaced-decimal rule; float64 divide, then float32 cast); see the "
            f"x10_* variables of the precip file and precip_x10_corrections.csv. "
            f"Temperature swap: tmin > tmax on {n_swap} cell-days ({swap_cells} cells) swapped "
            f"so tmin = min, tmax = max (daily mean unchanged); see swap_count. "
            f"No other value is altered.")
        seam_note = (
            f"Temperature is Livneh et al. (2013) for 1915-2015; 2016-2018 is a PRISM-based "
            f"extension (WGEN README: \"Livneh temperature (1915-2015) corrected to PRISM "
            f"observations (2016-2018)\"). Expect a seam at {SEAM_DATE}: tmin > tmax "
            f"inversions number {n_swap - n_swap_post} cell-days before it and "
            f"{n_swap_post} from it on.")
        common = {"input_tree_sha256": input_tree_sha256, "corrections": corrections,
                  "temperature_source_note": seam_note}

        # --- precip extras ---------------------------------------------------
        ds = datasets["precip_mm"]
        ncio.set_attrs(ncvars["precip_mm"], {
            "x10_rule": X10_RULE,
            "x10_months": np.array(X10_MONTHS, dtype=np.int32),
            "x10_threshold_mm": float(X10_THRESHOLD_MM),
        })
        ds.createDimension("x10_pair", n_pairs)
        lat_ax, lon_ax = L.lat_axis(), L.lon_axis()
        tab = {
            "x10_lat": ("f8", np.array([lat_ax[p[1]] for p in pairs], dtype=np.float64),
                        {"long_name": "latitude of the x10-corrected cell",
                         "units": "degrees_north"}),
            "x10_lon": ("f8", np.array([lon_ax[p[2]] for p in pairs], dtype=np.float64),
                        {"long_name": "longitude of the x10-corrected cell",
                         "units": "degrees_east"}),
            "x10_time": ("i4", np.array([p[0] for p in pairs], dtype=np.int32),
                         {"long_name": "day of the x10-corrected value",
                          "units": L.TIME_UNITS, "calendar": L.CALENDAR}),
            "x10_raw_mm": ("f4", np.array([p[4] for p in pairs], dtype=np.float32),
                           {"long_name": "raw store precipitation before the x10 correction",
                            "units": "mm"}),
            "x10_corrected_mm": ("f4", np.array([p[5] for p in pairs], dtype=np.float32),
                                 {"long_name": "stored precipitation after the x10 correction "
                                               "(equals precip_mm at the pair)",
                                  "units": "mm"}),
        }
        for name, (dtype, values, attrs) in tab.items():
            v = ds.createVariable(name, dtype, ("x10_pair",))
            ncio.set_attrs(v, {**attrs, "comment": "pairs sorted by (time, lat, lon)"})
            if n_pairs:
                v[:] = values
        v = ds.createVariable("x10_count", "i2", ("lat", "lon"), **ncio.ZLIB)
        ncio.set_attrs(v, {"long_name": "number of x10-corrected days in the cell",
                           "units": "1", "grid_mapping": "crs"})
        v[:] = x10_count
        # --- temperature extras ----------------------------------------------
        for var in ("tmax_c", "tmin_c"):
            ncio.set_attrs(ncvars[var], {"comment": "tmin > tmax days of the store swapped "
                                                    "(tmin = min, tmax = max); see swap_count"})
            v = datasets[var].createVariable("swap_count", "i4", ("lat", "lon"), **ncio.ZLIB)
            ncio.set_attrs(v, {"long_name": "days with tmin > tmax in the store (swapped)",
                               "units": "1", "grid_mapping": "crs"})
            v[:] = swap_count
        for var in VARS:
            ncio.set_attrs(datasets[var], common)
        ok = True
    finally:
        ex.shutdown(wait=True, cancel_futures=True)
        hasher.shutdown(wait=True, cancel_futures=True)
        for ds in datasets.values():
            try:
                ds.close()
            except Exception:                                  # noqa: BLE001
                pass
        if not ok:
            for p in finals.values():
                ncio.part_path(p).unlink(missing_ok=True)

    # --- x10 CSV + output hashes, all on the .part files ---------------------
    # Nothing is renamed until every output is written and hashed, and then
    # the whole set moves in one step (ncio.finalize_all), so an interrupt or
    # a file held open elsewhere never leaves a mix of old and new products.
    x10_csv = out_dir / GRIDDED_X10_CSV.name
    outputs = [*finals.values(), x10_csv]
    try:
        pd.DataFrame({
            "key": [p[3] for p in pairs],
            "lat": [f"{lat_ax[p[1]]:.5f}" for p in pairs],
            "lon": [f"{lon_ax[p[2]]:.5f}" for p in pairs],
            "date": [_date_str(p[0]) for p in pairs],
            "raw_mm": [_fmt(np.float64(p[4])) for p in pairs],
            "corrected_mm": [_fmt(np.float32(p[5])) for p in pairs],
        }).to_csv(ncio.part_path(x10_csv), index=False, lineterminator="\n")
        print("  hashing the outputs ...")
        file_sha = {p.name: ncio.sha256_file(ncio.part_path(p)) for p in outputs}
        file_sha[grid_csv.name] = ncio.sha256_file(grid_csv)
        sizes = {p.name: ncio.part_path(p).stat().st_size for p in outputs}
        sizes[grid_csv.name] = grid_csv.stat().st_size
    except BaseException:
        for p in outputs:
            ncio.part_path(p).unlink(missing_ok=True)
        raise
    ncio.finalize_all(outputs)

    # --- sums + provenance -----------------------------------------------------
    content = {var: hashes[var].hexdigest() for var in VARS}
    ncio.update_sha256sums(file_sha, out_dir / GRIDDED_SHA256SUMS.name)
    files: dict[str, dict[str, Any]] = {}
    for var in VARS:
        p = finals[var]
        files[var] = {"name": p.name, "sha256": file_sha[p.name],
                      "bytes": sizes[p.name], "content_sha256": content[var]}
    for label, p in (("grid_cells", grid_csv), ("precip_x10_corrections", x10_csv)):
        files[label] = {"name": p.name, "sha256": file_sha[p.name], "bytes": sizes[p.name]}
    table: dict[str, Any] = {
        "date_created": date_created,
        "command": command,
        "input_dir": meteo_dir.name,
        "input_tree_sha256": input_tree_sha256,
        "input_tree_hash_definition": INPUT_TREE_HASH_DEFINITION,
        "n_input_files": len(results),
        "n_cells": len(results),
        "time_start": L.START_DATE,
        "time_end": L.END_DATE,
        "n_days": L.N_DAYS,
        "chunks": list(CHUNKS),
        "content_hash_definition": CONTENT_HASH_DEFINITION,
    }
    if subset is not None:
        table["subset_rows"] = f"{subset[0]}:{subset[1]}"
    table.update({
        "files": files,
        "x10": {"rule": X10_RULE, "months": list(X10_MONTHS),
                "threshold_mm": float(X10_THRESHOLD_MM), "n_pairs": n_pairs,
                "n_cells": x10_cells, "n_dates": x10_dates},
        "temp_swap": {"n_cell_days": n_swap, "n_cells": swap_cells, "n_equal_days": n_equal,
                      "n_cell_days_before_seam": n_swap - n_swap_post,
                      "n_cell_days_from_seam": n_swap_post, "seam_date": SEAM_DATE},
        "encoding": {"dtype": "float32", "fill_value": "NaN", "zlib": True,
                     "complevel": int(ncio.ZLIB["complevel"]), "shuffle": True},
        "library_versions": ncio.library_versions(),
        "git": git,
    })
    ncio.update_provenance("forcing", table, out_dir / GRIDDED_PROVENANCE.name)

    elapsed = time.perf_counter() - t_start
    print(f"\n  x10 pairs     {n_pairs} on {x10_cells} cells, {x10_dates} dates")
    print(f"  tmin>tmax     {n_swap} cell-days swapped on {swap_cells} cells "
          f"({n_swap_post} from {SEAM_DATE}); tmin==tmax {n_equal}")
    for var in VARS:
        p = finals[var]
        print(f"  {p.name:40s} {p.stat().st_size / 1e9:6.3f} GB  content {content[var][:16]}...")
    print(f"  hash tail {t_hash_tail:.1f}s;  total {elapsed:.1f}s")
    return {
        "skipped": False, "out_dir": str(out_dir), "subset_rows": table.get("subset_rows"),
        "n_cells": len(results), "input_tree_sha256": input_tree_sha256,
        "files": files, "x10": table["x10"], "temp_swap": table["temp_swap"],
        "bands": band_log, "elapsed_s": round(elapsed, 1),
    }


# ---------------------------------------------------------------------------
# Product A gate
# ---------------------------------------------------------------------------
#: Product A differences that are NOT the x10 rule and that this product
#: deliberately does not reproduce (the raw store value is kept).  Found by
#: the first statewide check (2026-09-28): one isolated upstream edit on the
#: 1974-07-08 artifact storm, ratio 1.18, all neighbouring cells untouched.
#: (key, date) -> (raw_mm, product_a_mm).  Reported, not failed.
PA_KNOWN_NON_RULE: dict[tuple[str, str], tuple[float, float]] = {
    ("37.40625_-122.34375", "1974-07-08"): (29.037, 24.58),
}


def _read_product_a(path: Path) -> np.ndarray:
    """Product A ``meteo_<key>`` → float64 (N_DAYS, ≥4); validates rows + dates."""
    try:
        values = pd.read_csv(path, sep=r"\s+", header=None, dtype=np.float64,
                             engine="c").to_numpy()
    except Exception as exc:                                  # noqa: BLE001
        raise ValueError(f"{path}: unparseable ({type(exc).__name__}: {exc})") from exc
    if values.shape[0] != L.N_DAYS or values.shape[1] < 4:
        raise ValueError(f"{path}: shape {values.shape}, expected ({L.N_DAYS}, >=4)")
    if not np.array_equal(values[:, :3], _ymd()):
        raise ValueError(f"{path}: dates differ from {L.START_DATE}..{L.END_DATE}")
    if not np.isfinite(values[:, 3]).all():
        row = int(np.flatnonzero(~np.isfinite(values[:, 3]))[0])
        raise ValueError(f"{path}: non-finite precip {values[row, 3]} at row {row} "
                         f"({_date_str(row)})")
    return values


def _is_known_non_rule(key: str, date: str, raw: float, pa: float) -> bool:
    """True if (key, date) is listed in PA_KNOWN_NON_RULE with these values."""
    known = PA_KNOWN_NON_RULE.get((key, date))
    return known is not None and abs(raw - known[0]) <= 5e-4 and abs(pa - known[1]) <= 5e-3


def _cell_has_rule(meteo_dir: Path, key: str) -> tuple[str, int]:
    _, values = read_store_file(meteo_dir / L.store_filename(key))
    return key, int(_x10_mask(values).sum())


def _compare_product_a(meteo_dir: Path, pa_dir: Path, key: str) -> dict[str, Any]:
    """One cell: expected = where(rule, raw/10, raw) vs Product A precip."""
    out: dict[str, Any] = {"key": key, "n_rule": 0, "n_pa_corr": 0, "max_diff": 0.0,
                           "rows": [], "error": None}
    try:
        _, values = read_store_file(meteo_dir / L.store_filename(key))
        pa = _read_product_a(pa_dir / f"{_PA_PREFIX}{key}")[:, 3]
    except (ValueError, OSError) as exc:
        out["error"] = str(exc)
        return out
    raw = values[:, _C_PRECIP]
    rule = _x10_mask(values)
    expected = np.where(rule, raw / 10.0, raw)
    diff = np.abs(pa - expected)
    pa_corr = (np.abs(pa - raw) > _PA_CORR_TOL) & (np.abs(pa - raw / 10.0) <= _PA_TOL)
    out["n_rule"] = int(rule.sum())
    out["n_pa_corr"] = int(pa_corr.sum())
    known = np.zeros(diff.shape, dtype=bool)
    for t in np.flatnonzero(~(diff <= _PA_TOL)):          # NaN counts as a mismatch
        if rule[t] and abs(pa[t] - raw[t]) <= _PA_TOL:
            kind = "rule_day_not_corrected_by_pa"
        elif not rule[t] and pa_corr[t]:
            kind = "pa_correction_missed_by_rule"
        elif _is_known_non_rule(key, _date_str(t), raw[t], pa[t]):
            kind = "known_non_rule"
            known[t] = True
        else:
            kind = "other"
        out["rows"].append({"key": key, "date": _date_str(t), "kind": kind,
                            "month": int(values[t, _C_MONTH]), "raw_mm": _fmt(raw[t]),
                            "expected_mm": _fmt(expected[t]), "product_a_mm": _fmt(pa[t]),
                            "abs_diff": _fmt(np.float64(round(float(diff[t]), 6)))})
    kept = diff[~known]                                   # listed non-rule edits reported apart
    out["max_diff"] = float(np.nanmax(kept)) if np.isfinite(kept).any() else float("inf")
    return out


def check_x10_product_a(meteo_dir, product_a_dir, *, sample: int | None = None, seed: int = 0,
                        out_csv=None) -> bool:
    """Gate the x10 rule against DWR WGEN Product A (same correction, applied upstream).

    For each selected cell the expected precip is ``where(rule, raw/10, raw)``,
    and every day must satisfy ``|PA - expected| <= 0.0051`` (Product A rounds
    to 0.01 mm).  All cells are selected by default.  With ``sample=N`` the
    selection is every cell that has a rule pair plus N random others; finding
    the rule cells means parsing the whole local store first (about 1.5 min
    on 8 threads).  Mismatches are classified as (i) a rule day Product A did not
    correct, (ii) a Product A correction the rule missed, or (iii) anything
    else.  They are written to ``out_csv`` if given.  Product A's temperatures
    are detrended and are ignored.  Returns True iff clean.
    """
    t_start = time.perf_counter()
    meteo_dir, pa_dir = Path(meteo_dir), Path(product_a_dir)
    cells = L.scan_meteo_dir(meteo_dir)
    pa_keys = {n[len(_PA_PREFIX):] for n in os.listdir(pa_dir) if n.startswith(_PA_PREFIX)}
    store_keys = set(cells["key"])
    ok = True
    missing = sorted(store_keys - pa_keys)
    extra = sorted(pa_keys - store_keys)
    if missing:
        ok = False
        print(f"  FAIL  {len(missing)} store cells have no Product A file, e.g. {missing[:3]}")
    if extra:
        if len(cells) == L.N_CELLS:
            ok = False
            print(f"  FAIL  {len(extra)} Product A files have no store cell, e.g. {extra[:3]}")
        else:
            print(f"  note  store is a subset ({len(cells)} cells); {len(extra)} Product A "
                  f"files outside it are not checked")
    keys = [k for k in cells["key"] if k in pa_keys]

    if sample is None:
        selected = keys
        n_rule_cells = None
        print(f"Checking all {len(selected)} cells against {pa_dir} ...")
    else:
        print(f"Scanning {len(keys)} store files for rule days ...")
        rule_keys: list[str] = []
        with ThreadPoolExecutor(max_workers=_CHECK_WORKERS) as ex:
            futs = [ex.submit(_cell_has_rule, meteo_dir, k) for k in keys]
            for f in tqdm(as_completed(futs), total=len(futs), desc="store scan"):
                k, n = f.result()
                if n:
                    rule_keys.append(k)
        rule_set = set(rule_keys)
        others = [k for k in keys if k not in rule_set]
        rng = np.random.default_rng(seed)
        n_rand = min(int(sample), len(others))
        rand = [others[i] for i in sorted(rng.choice(len(others), n_rand, replace=False))]
        order = {k: n for n, k in enumerate(keys)}
        selected = sorted(rule_set | set(rand), key=order.__getitem__)
        n_rule_cells = len(rule_set)
        print(f"Checking {len(rule_set)} rule cells + {n_rand} random others "
              f"(seed {seed}) against {pa_dir} ...")
    t_scan = time.perf_counter() - t_start

    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    n_rule = n_pa_corr = 0
    max_diff = 0.0
    with ThreadPoolExecutor(max_workers=_CHECK_WORKERS) as ex:
        futs = [ex.submit(_compare_product_a, meteo_dir, pa_dir, k) for k in selected]
        for f in tqdm(as_completed(futs), total=len(futs), desc="Product A"):
            r = f.result()
            if r["error"]:
                errors.append(r["error"])
                continue
            n_rule += r["n_rule"]
            n_pa_corr += r["n_pa_corr"]
            max_diff = max(max_diff, r["max_diff"])
            rows.extend(r["rows"])
    rows.sort(key=lambda r: (r["date"], r["key"]))
    kinds = pd.Series([r["kind"] for r in rows], dtype=object).value_counts()
    n_i = int(kinds.get("rule_day_not_corrected_by_pa", 0))
    n_ii = int(kinds.get("pa_correction_missed_by_rule", 0))
    n_iii = int(kinds.get("other", 0))
    known = [r for r in rows if r["kind"] == "known_non_rule"]
    failing = [r for r in rows if r["kind"] != "known_non_rule"]

    if out_csv is not None:
        out_csv = Path(out_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows, columns=["key", "date", "kind", "month", "raw_mm", "expected_mm",
                                    "product_a_mm", "abs_diff"]
                     ).to_csv(out_csv, index=False, lineterminator="\n")
        print(f"  mismatches -> {out_csv}")

    elapsed = time.perf_counter() - t_start
    print(f"\n  cells checked                 {len(selected) - len(errors)}"
          + (f"  ({n_rule_cells} with rule pairs)" if n_rule_cells is not None else ""))
    print(f"  rule pairs (x10 days)         {n_rule}")
    print(f"  Product A x10 corrections     {n_pa_corr}")
    print(f"  max |PA - expected|           {max_diff:.4f} mm  (tolerance {_PA_TOL}; "
          f"known non-rule edits excluded)")
    print(f"  (i)   rule days PA left raw   {n_i}")
    print(f"  (ii)  PA corrections missed   {n_ii}")
    print(f"  (iii) other mismatches        {n_iii}")
    print(f"  known non-rule PA edits       {len(known)} of {len(PA_KNOWN_NON_RULE)} listed "
          f"(raw kept; PA_KNOWN_NON_RULE)")
    for r in known:
        print(f"        {r['key']} {r['date']}: raw {r['raw_mm']} mm, Product A "
              f"{r['product_a_mm']} mm")
    print(f"  unreadable cells              {len(errors)}")
    for e in errors[:5]:
        print(f"        {e}")
    print(f"  time: scan {t_scan:.1f}s, total {elapsed:.1f}s")
    ok = ok and not failing and not errors and n_rule == n_pa_corr
    print(f"  {'PASS' if ok else 'FAIL'}  x10 rule vs Product A")
    return ok


# ---------------------------------------------------------------------------
# Verify
# ---------------------------------------------------------------------------
class _Report:
    def __init__(self) -> None:
        self.ok = True

    def __call__(self, passed: bool, label: str, detail: str = "") -> bool:
        self.ok &= bool(passed)
        print(f"  {'PASS' if passed else 'FAIL'}  {label}" + (f": {detail}" if detail else ""))
        return bool(passed)


def _bits_equal(a: np.ndarray, b: np.ndarray) -> bool:
    a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
    return a.shape == b.shape and np.array_equal(a.view(np.uint32), b.view(np.uint32))


def _read_cell_series(v: netCDF4.Variable,
                      ij: list[tuple[int, int]]) -> dict[tuple[int, int], np.ndarray]:
    """Full daily series of cells, one read per 16 x 16 chunk column."""
    groups: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
    for i, j in ij:
        groups[(i // CHUNKS[1], j // CHUNKS[2])].append((i, j))
    out = {}
    for (bi, bj), members in groups.items():
        r0, c0 = bi * CHUNKS[1], bj * CHUNKS[2]
        block = v[:, r0:min(r0 + CHUNKS[1], L.NLAT), c0:min(c0 + CHUNKS[2], L.NLON)]
        for i, j in members:
            out[(i, j)] = np.array(block[:, i - r0, j - c0])
    return out


def verify_forcing(out_dir=GRIDDED_DIR, *, meteo_dir=None, sample: int = 64, seed: int = 0, full: bool = False) -> bool:
    """Check the forcing products in ``out_dir``; print PASS/FAIL lines; return overall bool.

    The checks are: the files are products, not Git LFS pointer stubs; exact
    lattice coordinates and time axis; the encoding; masks identical across
    files and equal to ``grid_cells.csv`` (only the subset rows when
    ``subset_rows`` is set; a full product must cover all 13,786 cells, and
    a ``--rows`` smoke build in ``data/gridded`` fails); values finite
    exactly where ``mask == 1`` on each 16-row band's first, last and one
    random day, with precip >= 0, no June-August precip >= 150 mm (the x10
    postcondition) and tmin <= tmax on those days; the x10 table (sorted,
    in-mask, JJA, >= threshold, equal to precip at the pairs, matching
    ``x10_count`` and the CSV); ``swap_count`` identical in both temperature
    files; and SHA256SUMS plus the provenance file hashes.  ``full`` checks
    every day the same way and recomputes each content hash against the
    provenance.  If
    ``meteo_dir`` is given, ``sample`` random cells plus the x10-table cells
    (at most 64) are re-read through :func:`read_store_file` and
    :func:`correct_cell` and must equal the files bitwise.
    """
    t_start = time.perf_counter()
    out_dir = Path(out_dir)
    rep = _Report()
    rng = np.random.default_rng(seed)
    finals = _out_paths(out_dir)
    missing = [p.name for p in finals.values() if not p.exists()]
    if not rep(not missing, "files present", ", ".join(missing)):
        return False
    stubs = [p.name for p in finals.values() if ncio.is_lfs_pointer(p)]
    if not rep(not stubs, "files are products, not Git LFS pointer stubs",
               f"{', '.join(stubs)}; fetch them with {ncio.LFS_PULL_HINT}" if stubs else ""):
        return False
    prov = ncio.read_provenance(out_dir / GRIDDED_PROVENANCE.name).get("forcing", {})
    is_repo_dir = ncio.is_repo_gridded_dir(out_dir)

    ds = {var: netCDF4.Dataset(finals[var], "r") for var in VARS}
    try:
        for d in ds.values():
            d.set_auto_maskandscale(False)

        # --- coordinates / encoding --------------------------------------------
        bad = []
        for var, d in ds.items():
            t = d["time"]
            if not (np.array_equal(d["lat"][:], L.lat_axis())
                    and np.array_equal(d["lon"][:], L.lon_axis())
                    and np.array_equal(d["lat_bnds"][:], L.axis_bounds(L.lat_axis()))
                    and np.array_equal(d["lon_bnds"][:], L.axis_bounds(L.lon_axis()))
                    and np.array_equal(t[:], np.arange(L.N_DAYS))
                    and t.units == L.TIME_UNITS and t.calendar == L.CALENDAR):
                bad.append(var)
        rep(not bad, "coords + time axis exactly the lattice", ", ".join(bad))
        bad = []
        for var, d in ds.items():
            v = d[var]
            f = v.filters()
            fill = np.float32(v.getncattr("_FillValue"))
            if not (v.dtype == np.float32 and v.dimensions == ("time", "lat", "lon")
                    and list(v.chunking()) == list(CHUNKS) and f.get("zlib")
                    and f.get("complevel") == ncio.ZLIB["complevel"] and f.get("shuffle")
                    and np.isnan(fill)):
                bad.append(f"{var} dtype={v.dtype} chunks={v.chunking()} filters={f}")
        rep(not bad, f"encoding float32 {CHUNKS} zlib4+shuffle NaN fill", "; ".join(bad))

        # --- mask ---------------------------------------------------------------
        subsets = {var: d.__dict__.get("subset_rows") for var, d in ds.items()}
        subset = _parse_rows(subsets["precip_mm"]) if subsets["precip_mm"] else None
        rep(len(set(subsets.values())) == 1, "subset_rows identical across files",
            str(subsets) if len(set(subsets.values())) > 1 else
            (f"subset {subsets['precip_mm']}" if subset else "full grid"))
        if is_repo_dir:
            rep(not any(subsets.values()), f"no --rows smoke-test product in {GRIDDED_DIR.name}",
                f"subset_rows {sorted({s for s in subsets.values() if s})}"
                if any(subsets.values()) else "")
        grid_csv = out_dir / GRIDDED_GRID_CSV.name
        if grid_csv.exists():
            cells = L.read_grid_csv(grid_csv)
        elif meteo_dir is not None:
            cells = L.scan_meteo_dir(meteo_dir)
            print(f"  note  {grid_csv.name} missing; using the scan of {meteo_dir}")
        else:
            rep(False, f"{grid_csv.name} present (or pass --meteo-dir)")
            return False
        if subset is not None:
            in_rows = (cells["ilat"] >= subset[0]) & (cells["ilat"] < subset[1])
            cells = cells[in_rows].reset_index(drop=True)
        mask_expected = L.mask_from_cells(cells)
        masks = {var: d["mask"][:] for var, d in ds.items()}
        same = all(np.array_equal(m, masks["precip_mm"]) for m in masks.values())
        rep(same and np.array_equal(masks["precip_mm"], mask_expected),
            "mask identical across files and equal to grid cells",
            f"{int(mask_expected.sum())} cells")
        if subset is None:          # build_forcing writes a full product only from the full store
            n_mask = int(masks["precip_mm"].astype(bool).sum())
            rep(n_mask == L.N_CELLS, f"full product covers all {L.N_CELLS} cells",
                f"{n_mask} cells" if n_mask != L.N_CELLS else "")
        mask = mask_expected.astype(bool)
        jja = np.isin(L.time_axis().month, X10_MONTHS)

        # --- finite exactly where mask -------------------------------------------
        bad = []
        n_days_checked = 0
        for r0, r1 in _bands():
            days = sorted({0, L.N_DAYS - 1, int(rng.integers(L.N_DAYS))})
            m = mask[r0:r1]
            for day in days:
                vals = {var: ds[var][var][day, r0:r1, :] for var in VARS}
                n_days_checked += 1
                for var, a in vals.items():
                    if not np.array_equal(np.isfinite(a), m):
                        bad.append(f"{var} day {day} rows {r0}:{r1}")
                if (vals["precip_mm"][m] < 0).any():
                    bad.append(f"precip < 0 day {day} rows {r0}:{r1}")
                if jja[day] and (vals["precip_mm"][m] >= X10_THRESHOLD_MM).any():
                    bad.append(f"June-August precip >= {X10_THRESHOLD_MM} (x10 rule not "
                               f"applied) day {day} rows {r0}:{r1}")
                if (vals["tmin_c"][m] > vals["tmax_c"][m]).any():
                    bad.append(f"tmin > tmax day {day} rows {r0}:{r1}")
        rep(not bad, f"finite exactly on mask, precip >= 0, no June-August precip >= "
                     f"{X10_THRESHOLD_MM:g}, tmin <= tmax ({n_days_checked} band-days per "
                     f"variable)", "; ".join(bad[:5]))

        if full:
            content = {var: hashlib.sha256() for var in VARS}
            bad = []
            for r0, r1 in _bands():
                m = mask[r0:r1][None]
                band = {}
                for var in VARS:
                    a = ds[var][var][:, r0:r1, :]
                    if not np.array_equal(np.isfinite(a), np.broadcast_to(m, a.shape)):
                        bad.append(f"{var} rows {r0}:{r1} finite pattern")
                    ncio.sha256_array(a.transpose(1, 2, 0), content[var])
                    band[var] = a
                mm = np.broadcast_to(m, band["precip_mm"].shape)
                if (band["precip_mm"][mm] < 0).any():
                    bad.append(f"precip < 0 rows {r0}:{r1}")
                # x10 postcondition: raw/10 < 150 for every raw < 1500, so no
                # June-August value >= 150 mm may survive the rule
                spikes = band["precip_mm"][jja][:, m[0]] >= X10_THRESHOLD_MM
                if spikes.any():
                    t, c = np.argwhere(spikes)[0]
                    i, j = np.argwhere(m[0])[c]
                    bad.append(f"{int(spikes.sum())} June-August precip >= "
                               f"{X10_THRESHOLD_MM:g} mm rows {r0}:{r1} (x10 rule not "
                               f"applied), e.g. {L.cell_key(L.lat_axis()[r0 + i], L.lon_axis()[j])} "
                               f"{_date_str(np.flatnonzero(jja)[t])}")
                if (band["tmin_c"][mm] > band["tmax_c"][mm]).any():
                    bad.append(f"tmin > tmax rows {r0}:{r1}")
                del band
            rep(not bad, "full scan: every day finite exactly on mask, precip >= 0, "
                         f"no June-August precip >= {X10_THRESHOLD_MM:g}, tmin <= tmax",
                "; ".join(bad[:5]))
            pfiles = prov.get("files", {})
            bad = [var for var in VARS
                   if pfiles.get(var, {}).get("content_sha256") != content[var].hexdigest()]
            rep(not bad, "content_sha256 recomputed == provenance",
                ", ".join(f"{var} {content[var].hexdigest()[:16]}..." for var in bad))

        # --- x10 table ------------------------------------------------------------
        dp = ds["precip_mm"]
        tt = dp["x10_time"][:].astype(np.int64)
        tlat = dp["x10_lat"][:]
        tlon = dp["x10_lon"][:]
        traw = dp["x10_raw_mm"][:]
        tcorr = dp["x10_corrected_mm"][:]
        ti, tj = (L.grid_index(tlat, tlon) if len(tt) else
                  (np.zeros(0, np.int64), np.zeros(0, np.int64)))
        bad = []
        order = np.lexsort((tj, ti, tt))
        if not np.array_equal(order, np.arange(len(tt))):
            bad.append("not sorted by (time, lat, lon)")
        if len(tt) and not mask[ti, tj].all():
            bad.append("pair outside mask")
        months = L.time_axis()[tt].month if len(tt) else np.zeros(0)
        if not np.isin(months, X10_MONTHS).all():
            bad.append("pair outside months " + str(X10_MONTHS))
        if (traw < X10_THRESHOLD_MM).any():
            bad.append("raw below threshold")
        if not np.allclose(tcorr, traw.astype(np.float64) / 10.0, rtol=1e-6, atol=0):
            bad.append("corrected != raw/10")
        cnt = np.zeros((L.NLAT, L.NLON), dtype=np.int64)
        np.add.at(cnt, (ti, tj), 1)
        if not np.array_equal(cnt, dp["x10_count"][:].astype(np.int64)):
            bad.append("x10_count != pairs per cell")
        # precip at the pairs, one read per chunk
        got = np.empty(len(tt), dtype=np.float32)
        groups: dict[tuple[int, int, int], list[int]] = defaultdict(list)
        for n, (t, i, j) in enumerate(zip(tt, ti, tj)):
            groups[(t // CHUNKS[0], i // CHUNKS[1], j // CHUNKS[2])].append(n)
        for (bt, bi, bj), members in groups.items():
            t0, r0, c0 = bt * CHUNKS[0], bi * CHUNKS[1], bj * CHUNKS[2]
            block = dp["precip_mm"][t0:t0 + CHUNKS[0], r0:r0 + CHUNKS[1], c0:c0 + CHUNKS[2]]
            for n in members:
                got[n] = block[tt[n] - t0, ti[n] - r0, tj[n] - c0]
        if not _bits_equal(got, tcorr):
            bad.append(f"precip_mm at pairs != x10_corrected_mm "
                       f"({int((got != tcorr).sum())} pairs)")
        x10_csv = out_dir / GRIDDED_X10_CSV.name
        if x10_csv.exists():
            csv = pd.read_csv(x10_csv, dtype={"key": str, "date": str, "raw_mm": str,
                                              "corrected_mm": str})
            keys = [L.cell_key(a, b) for a, b in zip(tlat, tlon)]
            dates = [_date_str(t) for t in tt]
            if not (len(csv) == len(tt) and csv["key"].tolist() == keys
                    and csv["date"].tolist() == dates
                    and _bits_equal(np.array(csv["raw_mm"].tolist(), dtype=np.float64)
                                    .astype(np.float32), traw)
                    and _bits_equal(np.array(csv["corrected_mm"].tolist(), dtype=np.float32),
                                    tcorr)):
                bad.append(f"{x10_csv.name} disagrees with the NetCDF table")
        else:
            bad.append(f"{x10_csv.name} missing")
        rep(not bad, f"x10 table ({len(tt)} pairs, {len(set(zip(ti, tj)))} cells, "
                     f"{len(set(tt.tolist()))} dates)", "; ".join(bad))

        # --- swap_count -----------------------------------------------------------
        sx, sn = ds["tmax_c"]["swap_count"][:], ds["tmin_c"]["swap_count"][:]
        rep(np.array_equal(sx, sn) and (sx >= 0).all() and not sx[~mask].any(),
            f"swap_count identical in both temperature files "
            f"({int(sx.sum())} cell-days, {int((sx > 0).sum())} cells)")

        # --- store re-read ----------------------------------------------------------
        if meteo_dir is not None:
            meteo_dir = Path(meteo_dir)
            ci = cells["ilat"].to_numpy()
            cj = cells["ilon"].to_numpy()
            n_rand = min(int(sample), len(cells))
            pick = set(rng.choice(len(cells), n_rand, replace=False).tolist())
            x10_cells = sorted(set(zip(ti.tolist(), tj.tolist())))
            if len(x10_cells) > 64:
                keep = sorted(rng.choice(len(x10_cells), 64, replace=False))
                x10_cells = [x10_cells[k] for k in keep]
            pos = {(int(i), int(j)): n for n, (i, j) in enumerate(zip(ci, cj))}
            pick |= {pos[c] for c in x10_cells}
            pick = sorted(pick)
            ij = [(int(ci[n]), int(cj[n])) for n in pick]
            with ThreadPoolExecutor(max_workers=_CHECK_WORKERS) as ex:
                redo = list(ex.map(lambda n: correct_cell(read_store_file(
                    meteo_dir / L.store_filename(cells["key"].iat[n]))[1]), pick))
            series = {var: _read_cell_series(ds[var][var], ij) for var in VARS}
            xc = dp["x10_count"][:]
            bad = []
            for (i, j), (out, x10, swapped) in zip(ij, redo):
                key = L.cell_key(L.lat_axis()[i], L.lon_axis()[j])
                for var in VARS:
                    if not _bits_equal(series[var][(i, j)], out[var]):
                        bad.append(f"{key} {var}")
                sel = (ti == i) & (tj == j)
                if not np.array_equal(np.sort(tt[sel]), x10) or xc[i, j] != len(x10):
                    bad.append(f"{key} x10 table")
                if sx[i, j] != int(swapped.sum()):
                    bad.append(f"{key} swap_count")
            rep(not bad, f"store re-read bitwise equal ({len(ij)} cells: {n_rand} random + "
                         f"{len(x10_cells)} x10 cells)", "; ".join(bad[:5]))
    finally:
        for d in ds.values():
            d.close()

    # --- SHA256SUMS / provenance ------------------------------------------------------
    sums = ncio.read_sha256sums(out_dir / GRIDDED_SHA256SUMS.name)
    names = [p.name for p in finals.values()] + [GRIDDED_GRID_CSV.name, GRIDDED_X10_CSV.name]
    actual = {n: ncio.sha256_file(out_dir / n) for n in names if (out_dir / n).exists()}
    bad = [n for n in names if sums.get(n) is None or sums.get(n) != actual.get(n)]
    rep(not bad, f"SHA256SUMS match ({len(names)} files)", ", ".join(bad))
    pfiles = prov.get("files", {})
    prov_sha = {f.get("name"): f.get("sha256") for f in pfiles.values() if isinstance(f, dict)}
    bad = [n for n in names if prov_sha.get(n) != actual.get(n)]
    rep(not bad, "provenance [forcing] file hashes match", ", ".join(bad))

    elapsed = time.perf_counter() - t_start
    print(f"  {'PASS' if rep.ok else 'FAIL'}  forcing verify ({elapsed:.1f}s)")
    return rep.ok
