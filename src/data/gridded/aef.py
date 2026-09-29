"""Statewide AlphaEarth satellite embeddings (Google Earth Engine) → per 1/16° cell: 2017, and the 2017-2025 mean.

Source: ``GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL`` (``CATALOG_URL``) — AlphaEarth
Foundations (Brown et al. 2025, arXiv:2507.22291), CC-BY 4.0: "The AlphaEarth Foundations
Satellite Embedding dataset is produced by Google and Google DeepMind."  One
image per UTM tile (163.84 km, 10 m) per CALENDAR year from 2017; bands
A00..A63 are a unit-length 64-d embedding per pixel (stored int8, served
dequantised as sign(q)*(q/127.5)**2).  DATASET_VERSION is per year (1.1 for
2017 as of 2026-09) — recorded from the images, not the catalog prose.

Products, both keyed to the 13,786 store cells (``grid_cells.csv``):

- ``data/gridded/alphaearth_2017.nc`` — for each cell the mean of the
  calendar-2017 pixel vectors over the cell rectangle.  ONE year, no temporal
  averaging: 2017 is the collection's earliest year (the burn refuses to
  start if Earth Engine lists any image before 2017-01-01) and the one
  closest to the 1915-2018 forcing record.
- ``data/gridded/alphaearth_2017-2025_mean.nc`` (:func:`assemble_mean`) —
  the same per-year cell mean for every annual layer in ``MEAN_YEARS``
  (``embedding_year``), then the EQUAL-WEIGHT mean over the years in which
  the cell has valid pixels (``embedding``), not renormalised.  Its 2017
  layer is bit for bit the 2017 product.

Every cell is reduced from Earth Engine for every year.
:func:`compare_reference` optionally compares the bank and the mean offline
with an external reference bank in the same format (read only).

**Method** (the same for every year):

- **No mosaic().**  The UTM zone edges -120° (10N | 11N) and -114°
  (11N | 12N) are lattice cell edges, so every cell lies in one zone and is
  reduced on its own zone's tiles only, in the tile's native UTM grid; the
  per-tile weighted sums are combined.  Tiles of one zone (163.84 km grid)
  never overlap and cover every cell exactly: sum(w) / expected = 1 on tile
  corners and the -120 cells, and assembly refuses valid_frac > 1 + 1e-6,
  which double counting would cause.  At 10 m this equals the zone mosaic's
  mean to 1e-12, at ~1/3 fewer EECU.  (A zone's tiles carry ~84 m of valid,
  slightly different pixels past its edge; they are never used.)  Zone 12N
  holds only 3 cells (east of -114°), too few to turn up in a random sample,
  so :func:`check` always samples all of them and the 11N cells on the -114°
  edge, and the assemblers warn about any zone-edge or 12N cell whose
  coverage is not complete.
- **scale 15 m, native UTM** (``scale`` accepts only 10 or 15).  EE's
  pyramid levels >= 20 m are L2-RENORMALIZED block means (despite
  pyramidingPolicy MEAN in the asset metadata): asking for 19-20 m reads
  them and inflates |mean| by ~4e-3.  At 15 m EE reads the full-resolution
  level on a nearest-neighbour lattice that takes 4/9 of the 10 m pixels:
  <= 2e-4 per band (a tiny same-sign bias), cos >= 0.9999998, |mean|
  unchanged (<= 6e-5), vs the full 10 m mean, at 12.8 vs 23.3 EECU-s per
  cell-year — ~49 EECU-h per statewide year (~89 at 10 m): the 2017 burn
  ~49 EECU-h, 2018-2025 another ~392 (the noncommercial Community tier is
  150 EECU-h/month).  16 m aliases (1e-3), so never "any value below 19".
  :func:`check` re-verifies one year against an independent zone-mosaic
  reduction at 10 m.
- mean = sum(v*w) / sum(w) with EE's fractional boundary weights (``w`` is
  a constant band carrying A00's exact mask); ``valid_frac`` = sum(w) / the
  same sum of an unmasked constant on the same grid (``expected_15m.npz``,
  year-independent).
- **Masked pixels.**  Coastal cells may be partly or wholly masked (e.g.
  open ocean): a cell that at least one tile feature reached (``nt`` > 0)
  but with W == 0 is banked and assembled as NaN for that year (2017
  product: ``aef_flag`` no_valid_pixels; mean: left out of that cell's mean,
  ``n_years`` < 9).  A cell NO tile feature reached (``nt`` == 0, outside
  every tile footprint of the year) is a hard error — its unit is not
  banked — as is a null band sum (a schema change).

**Resume / safety.**  Each (year, unit) — a unit is one zone's 1-degree
block, split to <= ``chunk`` cells, the same plan every year — is banked
atomically as ``aef_parts/<year>/<unit>.npz`` with the cell KEYS it holds
(plus ``S``, ``W``, ``nt``, the tile ids and the scale); a re-run skips
banked units and assembling places values by key.  An empty or null result
is never banked.  ``run.json`` pins chunk, scale and the grid (sha1 of its
keys) for the whole bank and lists the years runs were started for
(``years``); a bank begun by the 2017-only pipeline carries ``"year": 2017``
instead, which is accepted as is and kept when ``years`` is added.
``run.lock`` stops two runs from spending quota on the same units; one pool
serves every (year, unit) job of a run, year by year.  Ctrl+C cancels the
queued units (in-flight requests finish and are banked).  Before the first
unit the burn writes ``inventory_<year>.json`` per requested year (the tiles
over the state and their versions; 2017's also holds the earliest-year
check, the other years a has-images check) and ``expected_<scale>m.npz``.

Needs an authenticated earthengine-api (``earthengine authenticate``) and
an EE-registered cloud project (``--project``, no default).  ``ee`` is
imported only by the Earth Engine actions (:func:`run`, :func:`check`), so
dry-run / status / assemble / compare / verify work without it.  Behind the
DWR TLS proxy the Windows trust store has to be injected before
``import ee`` — done in ``_ee`` when pip's vendored truststore is
importable.

Usage (via ``scripts/prepare_gridded.py aef``)
-----
    python scripts/prepare_gridded.py aef --dry-run                           # what would run, no EE
    python scripts/prepare_gridded.py aef --run --project P --max-units 2     # smoke test
    python scripts/prepare_gridded.py aef --run --project P                   # the 2017 burn, resumable
    python scripts/prepare_gridded.py aef --status
    python scripts/prepare_gridded.py aef --assemble
    python scripts/prepare_gridded.py aef --check 40 --project P              # ~23.3 EECU-s per cell
    python scripts/prepare_gridded.py aef --dry-run --years all               # the 2017-2025 mean
    python scripts/prepare_gridded.py aef --run --project P --years all       # ~392 EECU-h on top of 2017
    python scripts/prepare_gridded.py aef --assemble-mean
    python scripts/prepare_gridded.py aef --check 30 --year 2021 --project P
    python scripts/prepare_gridded.py aef --compare-parts <ref_dir>           # offline, partials only
    python scripts/prepare_gridded.py aef --compare-parts <ref_dir> --compare-mean <ref_mean.npz>
    python scripts/prepare_gridded.py verify --skip-forcing
"""
from __future__ import annotations

import concurrent.futures as cf
import datetime as dt
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import netCDF4
import numpy as np
import pandas as pd

from src.data.gridded import lattice as L
from src.data.gridded import ncio
from src.paths import (
    GRIDDED_AEF_MEAN_NC,
    GRIDDED_AEF_NC,
    GRIDDED_AEF_PARTS_DIR,
    GRIDDED_DIR,
    GRIDDED_GRID_CSV,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
COLL = "GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL"
BANDS = [f"A{i:02d}" for i in range(64)]
YEAR = 2017                                # the collection's earliest year: the single-year
                                           # product, and the default burn / check year
MEAN_YEARS = tuple(range(2017, 2026))      # every annual layer -> the multi-year mean
ZONES = {"10N": "EPSG:32610", "11N": "EPSG:32611", "12N": "EPSG:32612"}
ZONE_SPAN = (-126.0, -108.0)               # west edge of 10N, east edge of 12N
CELL_DEG = L.CELL_DEG
SCALES = (10.0, 15.0)                      # the verified reduction scales (see docstring)
DEFAULT_SCALE = 15.0
EECU_S_PER_CELL = {15.0: 12.8, 10.0: 23.3}   # measured EECU-s per cell-year
ATTRIBUTION = ("The AlphaEarth Foundations Satellite Embedding dataset is produced "
               "by Google and Google DeepMind.")
REFERENCE = ('Brown, C. F., et al. 2025, "AlphaEarth Foundations: An embedding field '
             'model for accurate and efficient global mapping from sparse label data", '
             'arXiv:2507.22291')
LICENSE = "CC-BY-4.0 (https://creativecommons.org/licenses/by/4.0/)"
#: URI of the licensed material (CC-BY 4.0 section 3(a)(1)(A)(v)).
CATALOG_URL = ("https://developers.google.com/earth-engine/datasets/catalog/"
               "GOOGLE_SATELLITE_EMBEDDING_V1_ANNUAL")

#: Gates shared by assemble / verify / check.
FULL_TOL = 1e-6                            # valid_frac >= 1 - FULL_TOL counts as full
LOW_FRAC = 0.5                             # valid_frac below this sets low_coverage
CHECK_MAX_D = 2e-4                         # per band, vs the 10 m zone-mosaic mean
CHECK_MAX_DNORM = 1e-3                     # | |m| - |m10| |
REF_TOL = 1e-9                             # compare_reference: same reduction, same
                                           # pixels -> equal to rounding (~1e-13)
#: compare_reference, the mean: both files hold float32 casts of the same
#: float64 formula, so identical partials give identical values, and partials
#: that agree to ~1e-13 can differ by one float32 ulp (< 6e-8 below 1).
REF_MEAN_TOL = 1e-7
MEAN_TOL = 1e-6                            # verify: embedding vs the mean of embedding_year
COS_TOL = 1e-5                             # verify: year_cos_min recomputed from float32

#: EE error text -> what to do.  SPLIT: the request is too big, halve it (no
#: retry).  FATAL: deterministic, abort the burn.  Anything else (429 / 5xx /
#: "too many concurrent aggregations" / dropped connections, and the reduced
#: concurrency of an over-quota "restricted" project) is retried with backoff.
SPLIT = ("timed out", "deadline", "memory limit", "too large")
FATAL = ("permission", "not found", "not authorized", "does not have", "did not match",
         "not registered", "invalid argument", "unknown band")

_FLAG_PARTIAL, _FLAG_NO_VALID, _FLAG_LOW, _FLAG_FEW_YEARS = 1, 2, 4, 8
_CONTENT_HASH_DEFINITION = (
    "sha256 over the float32 little-endian values of `embedding` in stored (band, lat, lon) "
    f"C order over the whole 64 x {L.NLAT} x {L.NLON} grid, NaN (0x7FC00000) outside the "
    "cells with valid pixels")
_MEAN_CONTENT_HASH_DEFINITION = (
    "content_sha256: sha256 over the float32 little-endian values of `embedding` in stored "
    f"(band, lat, lon) C order over the whole 64 x {L.NLAT} x {L.NLON} grid, NaN (0x7FC00000) "
    "outside the cells with valid pixels in at least one year; content_sha256_embedding_year: "
    "the same over `embedding_year` in stored (year, band, lat, lon) C order "
    f"({len(MEAN_YEARS)} x 64 x {L.NLAT} x {L.NLON}), NaN outside the cell-years with valid "
    "pixels")


class Fatal(RuntimeError):
    """An EE error that retrying cannot fix."""


# ---------------------------------------------------------------------------
# Small utils
# ---------------------------------------------------------------------------
def _year_dir(parts_dir: str | os.PathLike, year: int = YEAR) -> Path:
    return Path(parts_dir) / str(year)


def _check_years(years) -> tuple[int, ...]:
    """Sorted, de-duplicated years, each one of ``MEAN_YEARS`` (the annual layers)."""
    ys = tuple(sorted({int(y) for y in years}))
    bad = [y for y in ys if y not in MEAN_YEARS]
    if not ys or bad:
        sys.exit(f"years {bad or '(none)'}: the annual layers are "
                 f"{MEAN_YEARS[0]}-{MEAN_YEARS[-1]}")
    return ys


def _years_label(years) -> str:
    """``2017``, ``2017-2025`` (a contiguous run) or ``2017, 2019``."""
    ys = [int(y) for y in years]
    if len(ys) > 1 and ys == list(range(ys[0], ys[-1] + 1)):
        return f"{ys[0]}-{ys[-1]}"
    return ", ".join(map(str, ys))


def _atomic_npz(path: Path, **arrays) -> None:
    """Write ``arrays`` so a kill mid-write can never leave a half-file behind."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    with open(tmp, "wb") as fh:
        np.savez_compressed(fh, **arrays)
    tmp.replace(path)


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    tmp.write_text(text, encoding="utf-8", newline="\n")
    tmp.replace(path)


def _load(path: Path) -> dict:
    try:
        with np.load(path, allow_pickle=False) as z:
            return {k: z[k] for k in z.files}
    except Exception as e:                                   # noqa: BLE001
        sys.exit(f"{path}: unreadable ({e}) -- delete it and re-run")


def _check_scale(scale: float) -> float:
    scale = float(scale)
    if scale not in SCALES:
        sys.exit(f"scale {scale:g} m is not one of the verified scales {SCALES} "
                 "(16 m aliases, >= 19 m reads renormalised pyramid levels)")
    return scale


def _safe_stdout() -> None:
    """EE errors can carry non-ASCII; never let a print kill the burn."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="backslashreplace")  # type: ignore[union-attr]


def _ee(project: str | None):
    """Initialise earthengine-api (imported here only, so this module loads
    without it)."""
    if not project:
        sys.exit("--project is required for Earth Engine actions "
                 "(an EE-registered cloud project id)")
    try:
        import pip._vendor.truststore as truststore         # DWR TLS proxy
        truststore.inject_into_ssl()
    except Exception:                                        # noqa: BLE001
        pass
    try:
        import ee
        ee.Initialize(project=project)
        # EE's interactive limit is 5 min; without a client deadline a dropped
        # connection can hang a request forever.
        ee.data.setDeadline(360_000)
    except Exception as e:                                   # noqa: BLE001
        sys.exit(f"earthengine-api not ready ({e}); run `earthengine authenticate` "
                 "and pass --project <an-EE-registered-cloud-project>")
    return ee


def _retry(fn, what: str, tries: int = 4):
    for k in range(tries):
        try:
            return fn()
        except Exception as e:                               # noqa: BLE001
            msg = str(e).lower()
            if any(s in msg for s in SPLIT):
                raise
            if any(s in msg for s in FATAL):
                raise Fatal(f"{what}: {e}") from e
            if k == tries - 1:
                raise
            wait = min(300, 30 * 2 ** k)
            print(f"    retry {what} in {wait}s ({str(e)[:120]})", flush=True)
            time.sleep(wait)
    raise RuntimeError("unreachable")


# ---------------------------------------------------------------------------
# Grid and work units
# ---------------------------------------------------------------------------
def _grid(grid_csv: str | os.PathLike) -> pd.DataFrame:
    """The cell list (``lattice.read_grid_csv``) plus the zone label."""
    grid_csv = Path(grid_csv)
    if not grid_csv.exists():
        sys.exit(f"{grid_csv} missing -- run `prepare_gridded.py grid` first")
    g = L.read_grid_csv(grid_csv)
    g["zone"] = [f"{int(z)}N" for z in g["utm_zone"]]
    h = CELL_DEG / 2.0
    if g["lon"].min() - h < ZONE_SPAN[0] or g["lon"].max() + h > ZONE_SPAN[1]:
        sys.exit(f"{grid_csv}: cells outside UTM zones {'/'.join(ZONES)}")
    return g


def _grid_sha1(g: pd.DataFrame) -> str:
    """Identity of the cell set (independent of the CSV's bytes)."""
    return hashlib.sha1("\n".join(g["key"]).encode("utf-8")).hexdigest()


def _units(g: pd.DataFrame, chunk: int) -> list[tuple[str, str, np.ndarray]]:
    """Work units: (zone, name, row indices) -- one 1-degree block of one zone,
    split to <= ``chunk`` cells, so a request touches few tiles.  The plan
    does not depend on the year."""
    out = []
    for zone in ZONES:
        z = g[g["zone"] == zone]
        if z.empty:
            continue
        blk = (np.floor(z["lat"]).astype(int).astype(str) + "_"
               + (-np.floor(z["lon"])).astype(int).astype(str))
        for b, rows in z.groupby(blk, sort=True):
            idx = rows.sort_values(["lat", "lon"]).index.to_numpy()
            for k, lo in enumerate(range(0, len(idx), chunk)):
                out.append((zone, f"{zone}_{b}_{k}", idx[lo:lo + chunk]))
    return out


def _todo(g: pd.DataFrame, parts_dir: Path, chunk: int, max_units: int | None,
          year: int = YEAR):
    """(units of ``year`` not banked yet, every unit of the plan)."""
    units = _units(g, chunk)
    if max_units is not None:
        units = units[:max_units]
    ydir = _year_dir(parts_dir, year)
    return [u for u in units if not (ydir / f"{u[1]}.npz").exists()], units


# ---------------------------------------------------------------------------
# Burn bookkeeping: lock + manifest
# ---------------------------------------------------------------------------
def _lock(parts_dir: Path) -> Path:
    """One burn at a time.  A lock left by a killed run is taken over when its
    PID is gone."""
    lock = parts_dir / "run.lock"
    parts_dir.mkdir(parents=True, exist_ok=True)
    for _ in range(2):
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(os.getpid()).encode())
            os.close(fd)
            return lock
        except FileExistsError:
            pid = lock.read_text().strip()
            if os.name == "nt":
                alive = pid in subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"],
                                              capture_output=True, text=True).stdout
            else:
                alive = Path(f"/proc/{pid}").exists()
            if alive:
                sys.exit(f"another run (PID {pid}) holds {lock}")
            print(f"  taking over the stale lock of PID {pid}", flush=True)
            lock.unlink(missing_ok=True)
    sys.exit(f"could not take {lock}")


def _manifest_want(g: pd.DataFrame, chunk: int, scale: float) -> dict:
    """What ``run.json`` pins: the unit plan (chunk + grid) and the scale,
    shared by every year of the bank."""
    return {"chunk": int(chunk), "scale": float(scale), "grid_sha1": _grid_sha1(g)}


def _manifest_diff(have: dict, want: dict, skip: tuple[str, ...] = ()) -> dict:
    """``{key: (banked, now)}`` for every pinned key that differs."""
    return {k: (have.get(k), v) for k, v in want.items() if k not in skip and have.get(k) != v}


def _manifest_years(have: dict) -> list[int]:
    """Years runs were started for: ``years``, plus the single ``year`` of a
    bank begun by the 2017-only pipeline (legacy run.json, still accepted)."""
    ys = {int(y) for y in have.get("years", [])}
    if have.get("year") is not None:
        ys.add(int(have["year"]))
    return sorted(ys)


def _read_manifest(parts_dir: Path) -> dict | None:
    path = parts_dir / "run.json"
    if not path.exists():
        return None
    have = json.loads(path.read_text())
    bad = [y for y in _manifest_years(have) if y not in MEAN_YEARS]
    if bad:
        sys.exit(f"{path}: years {bad} are not annual layers of this pipeline "
                 f"({MEAN_YEARS[0]}-{MEAN_YEARS[-1]})")
    return have


def _manifest(parts_dir: Path, g: pd.DataFrame, chunk: int, scale: float,
              years: tuple[int, ...] = (YEAR,)) -> None:
    """Pin chunk + scale + the grid for the whole bank (unit names and cell
    sets must not change between runs or years) and record ``years``.

    A legacy 2017-only ``run.json`` (``{"year": 2017, chunk, scale,
    grid_sha1}``) is accepted as is: a 2017 run leaves it byte for byte
    untouched, and a run that adds years rewrites it atomically with every
    key kept plus ``years`` (so the 2017-only pipeline still accepts it)."""
    want = _manifest_want(g, chunk, scale)
    have = _read_manifest(parts_dir)
    path = parts_dir / "run.json"
    if have is None:
        _atomic_text(path, json.dumps({**want, "years": list(years)}, indent=1))
        return
    diff = _manifest_diff(have, want)
    if diff:
        sys.exit(f"{path} was banked with a different setup {diff} "
                 f"(banked, now): finish with the banked settings, or move {parts_dir} aside")
    ys = sorted(set(_manifest_years(have)) | set(years))
    if ys != _manifest_years(have):
        _atomic_text(path, json.dumps({**have, "years": ys}, indent=1))


# ---------------------------------------------------------------------------
# The reduction
# ---------------------------------------------------------------------------
def _collection(ee, year: int = YEAR):
    return ee.ImageCollection(COLL).filterDate(f"{year}-01-01", f"{year + 1}-01-01")


def _fc(ee, g: pd.DataFrame, idx: np.ndarray):
    """One planar EPSG:4326 cell rectangle per row of ``idx``, tagged with its
    grid row ``i``."""
    h = CELL_DEG / 2.0
    lat, lon = g["lat"].to_numpy(), g["lon"].to_numpy()
    return ee.FeatureCollection([
        ee.Feature(ee.Geometry.Rectangle([lon[i] - h, lat[i] - h, lon[i] + h, lat[i] + h],
                                         proj="EPSG:4326", geodesic=False), {"i": int(i)})
        for i in idx])


def _bbox(ee, g: pd.DataFrame, idx: np.ndarray | None = None):
    h = CELL_DEG / 2.0
    s = g if idx is None else g.loc[idx]
    return ee.Geometry.Rectangle([s.lon.min() - h, s.lat.min() - h, s.lon.max() + h, s.lat.max() + h],
                                 proj="EPSG:4326", geodesic=False)


def _sum_features(features: list[dict], idx: np.ndarray, what: str):
    """Per-tile reduceRegions features -> (S [n,64], W [n], nt [n], image ids).

    ``nt`` counts the tile features returned per cell.  Reducer.sum gives 0,
    never null, on no data: a None or a missing key is a schema change."""
    pos = {int(i): k for k, i in enumerate(idx)}
    S = np.zeros((len(idx), 64))
    W = np.zeros(len(idx))
    nt = np.zeros(len(idx), dtype=np.int32)
    imgs: set[str] = set()
    for f in features:
        p = f["properties"]
        k = pos[int(p["i"])]
        v = [p.get(b) for b in BANDS]
        if p.get("w") is None or any(x is None for x in v):
            raise RuntimeError(f"{what}: null band sum (cell row {p['i']}, tile "
                               f"{p.get('img')}) -- schema change?")
        S[k] += v
        W[k] += p["w"]
        nt[k] += 1
        imgs.add(str(p["img"]))
    return S, W, nt, sorted(imgs)


def _reduce(ee, g: pd.DataFrame, zone: str, idx: np.ndarray, scale: float,
            year: int = YEAR):
    """Per-tile weighted sums of the ``year`` embedding over cells ``idx`` (one
    zone) -> (S [n,64], W [n], nt [n], image ids).  Halves the request when EE
    says it is too big."""
    fc = _fc(ee, g, idx)
    coll = (_collection(ee, year).filter(ee.Filter.eq("UTM_ZONE", zone))
            .filterBounds(_bbox(ee, g, idx)))

    def per_tile(img):
        w = img.select(0).multiply(0).add(1).rename("w")       # A00's exact mask
        return (img.addBands(w)
                .reduceRegions(collection=fc.filterBounds(img.geometry()),
                               reducer=ee.Reducer.sum(), crs=img.projection().crs(),
                               scale=scale)
                .map(lambda f: f.setGeometry(None).set("img", img.get("system:index"))))

    try:
        res = _retry(lambda: coll.map(per_tile).flatten().getInfo(),
                     f"{year} {zone} n={len(idx)}")
    except Fatal:
        raise
    except Exception as e:                                   # noqa: BLE001
        if not any(s in str(e).lower() for s in SPLIT) or len(idx) < 2:
            raise
        h = len(idx) // 2
        print(f"    {year} {zone}: too big at n={len(idx)} -> split", flush=True)
        a = _reduce(ee, g, zone, idx[:h], scale, year)
        b = _reduce(ee, g, zone, idx[h:], scale, year)
        return (np.vstack([a[0], b[0]]), np.concatenate([a[1], b[1]]),
                np.concatenate([a[2], b[2]]), sorted(set(a[3]) | set(b[3])))
    return _sum_features(res["features"], idx, f"{year} {zone}")


def _check_unit(S: np.ndarray, W: np.ndarray, nt: np.ndarray, imgs: list[str],
                year: int = YEAR) -> int:
    """Raise unless the unit may be banked; returns its number of fully masked
    (nt > 0, W == 0) cells."""
    if (nt == 0).any() or not imgs:
        raise RuntimeError(f"{int((nt == 0).sum())}/{len(nt)} cells without any tile feature "
                           f"(tiles {imgs}) -- not banked; a cell outside every {year} tile "
                           "footprint needs a look before re-running")
    if not (np.isfinite(S).all() and np.isfinite(W).all()) or (W < 0).any():
        raise RuntimeError("non-finite or negative sums -- not banked")
    return int((W == 0).sum())


# ---------------------------------------------------------------------------
# Inventories (one per year)
# ---------------------------------------------------------------------------
def _check_key(year: int) -> str:
    """The inventory's image-count check: 2017's is the earliest-year guard
    (no image before 2017-01-01), every other year's only requires images."""
    return "earliest_year_check" if year == YEAR else "year_check"


def _check_label(year: int) -> str:
    return "earliest-year check" if year == YEAR else f"{year} year check"


def _check_failed(year: int, path: Path | None, chk: dict) -> str:
    """The refusal when a year's image-count check fails (2017: the wording of
    the 2017-only pipeline)."""
    why = (f"{YEAR} is no longer the collection's earliest year (or has no images)"
           if year == YEAR else f"{COLL} lists no {year} image")
    return (f"{str(path) + ': ' if path is not None else ''}{_check_label(year)} failed {chk} "
            f"-- {why}; stop and decide")


def _read_inventory(parts_dir: Path, year: int = YEAR) -> dict | None:
    path = parts_dir / f"inventory_{year}.json"
    if not path.exists():
        return None
    inv = json.loads(path.read_text())
    key = _check_key(year)
    if not isinstance(inv, dict) or "tiles" not in inv or key not in inv:
        sys.exit(f"{path}: not an inventory of this pipeline (no tiles / {key})")
    if int(inv.get("year", year)) != year:
        sys.exit(f"{path}: holds year {inv.get('year')}, not {year}")
    return inv


def _validate_inventory(inv: dict, g: pd.DataFrame, path: Path, year: int = YEAR) -> None:
    """Images in the year (2017: and none before it), >= 1 tile in every zone
    holding cells, one DATASET_VERSION over those zones' tiles."""
    chk = inv[_check_key(year)]
    if not chk.get("passed") or int(chk.get("images_in_year", 0)) <= 0:
        sys.exit(_check_failed(year, path, chk))
    zones = set(g["zone"])
    have = {t.get("UTM_ZONE") for t in inv["tiles"]}
    missing = sorted(zones - have)
    if missing:
        sys.exit(f"{path}: no {year} tile in zone(s) {missing} that hold grid cells")
    dvs = sorted({str(t.get("DATASET_VERSION")) for t in inv["tiles"] if t.get("UTM_ZONE") in zones})
    if len(dvs) != 1:
        sys.exit(f"{path}: mixed DATASET_VERSION {dvs} over the {year} tiles (the collection is "
                 "being republished?) -- stop and decide; delete the inventory to re-query")


def _inventory(ee, g: pd.DataFrame, parts_dir: Path, year: int = YEAR) -> dict:
    """The tile images of ``year`` over the grid and their version properties,
    plus the image-count check (2017: the earliest-year guard) ->
    ``inventory_<year>.json`` (queried once per year)."""
    path = parts_dir / f"inventory_{year}.json"
    inv = _read_inventory(parts_dir, year)
    key = _check_key(year)
    if inv is None:
        sizes = {"year": _collection(ee, year).size()}
        if year == YEAR:
            sizes["before"] = ee.ImageCollection(COLL).filterDate("1900-01-01", f"{YEAR}-01-01").size()
        n = _retry(lambda: ee.Dictionary(sizes).getInfo(), _check_label(year))
        c = (_collection(ee, year).filterBounds(_bbox(ee, g))
             .filter(ee.Filter.inList("UTM_ZONE", list(ZONES))))
        props = ["UTM_ZONE", "DATASET_VERSION", "MODEL_VERSION", "PROCESSING_SOFTWARE_VERSION"]
        rows = _retry(lambda: c.map(lambda im: ee.Feature(None, im.toDictionary(props).set(
            "id", im.get("system:index")))).getInfo(), f"inventory {year}")
        tiles = sorted((f["properties"] for f in rows["features"]),
                       key=lambda r: (str(r.get("UTM_ZONE")), str(r.get("id"))))
        h = CELL_DEG / 2.0
        today = dt.date.today().isoformat()
        if year == YEAR:
            chk = {"images_before_year": int(n["before"]), "images_in_year": int(n["year"]),
                   "passed": int(n["before"]) == 0 and int(n["year"]) > 0, "checked": today}
        else:
            chk = {"images_in_year": int(n["year"]), "passed": int(n["year"]) > 0,
                   "checked": today}
        inv = {
            "collection": COLL, "year": year, "zones": list(ZONES),
            "bbox_wsen": [float(g.lon.min() - h), float(g.lat.min() - h),
                          float(g.lon.max() + h), float(g.lat.max() + h)],
            key: chk,
            "earthengine_api": getattr(ee, "__version__", ""),
            "tiles": tiles,
        }
        if not chk["passed"]:                             # not written: re-queried next run
            sys.exit(_check_failed(year, None, chk))
        _atomic_text(path, json.dumps(inv, indent=1))
    _validate_inventory(inv, g, path, year)
    vers = sorted({(r.get("DATASET_VERSION"), r.get("MODEL_VERSION"),
                    r.get("PROCESSING_SOFTWARE_VERSION")) for r in inv["tiles"]})
    per_zone = pd.Series([t.get("UTM_ZONE") for t in inv["tiles"]]).value_counts().sort_index()
    count = (f"images before {YEAR}: {inv[key]['images_before_year']}" if year == YEAR
             else f"images in {year}: {inv[key]['images_in_year']}")
    print(f"  inventory {year}: {len(inv['tiles'])} tiles {per_zone.to_dict()}, versions {vers}; "
          f"{count}", flush=True)
    return inv


def _expected(ee, g: pd.DataFrame, parts_dir: Path, scale: float) -> None:
    """Fractional pixel count of each FULL cell on the same grid (unmasked
    constant) -> the denominator of valid_frac, for every year.  Costs
    ~nothing to compute."""
    path = parts_dir / f"expected_{scale:g}m.npz"
    if path.exists():
        return
    n = np.full(len(g), np.nan)
    for zone, crs in ZONES.items():
        idx = g.index[g["zone"] == zone].to_numpy()
        for lo in range(0, len(idx), 1000):
            sub = idx[lo:lo + 1000]
            r = _retry(lambda: ee.Image.constant(1).rename("n").reduceRegions(
                collection=_fc(ee, g, sub), reducer=ee.Reducer.sum(), crs=crs, scale=scale)
                .map(lambda f: f.setGeometry(None)).getInfo(), f"expected {zone}")
            for f in r["features"]:   # one band, one reducer -> property "sum"
                n[int(f["properties"]["i"])] = f["properties"]["sum"]
    if np.isnan(n).any():
        sys.exit(f"expected counts missing for {int(np.isnan(n).sum())} cells")
    _atomic_npz(path, n=n, keys=g["key"].to_numpy().astype("U"))
    print(f"  expected pixel counts @ {scale:g} m: median {np.nanmedian(n):.0f}", flush=True)


# ---------------------------------------------------------------------------
# Reading the bank
# ---------------------------------------------------------------------------
def _banked(g: pd.DataFrame, parts_dir: Path, scale: float | None, year: int = YEAR) -> dict:
    """Sum every banked partial of ``year`` into grid order, placed by cell KEY.

    ``seen`` counts how often each cell was banked (must end up exactly 1)."""
    gi = pd.Index(g["key"])
    n = len(g)
    S = np.zeros((n, 64))
    W = np.zeros(n)
    nt = np.zeros(n, dtype=np.int64)
    seen = np.zeros(n, dtype=np.int64)
    imgs: set[str] = set()
    scales: set[float] = set()
    parts = sorted(_year_dir(parts_dir, year).glob("*.npz"))
    for p in parts:
        z = _load(p)
        if "nt" not in z or "keys" not in z:
            sys.exit(f"{p}: no `keys`/`nt` -- not a partial of this pipeline")
        s = float(z["scale"])
        scales.add(s)
        if scale is not None and s != scale:
            sys.exit(f"{p}: banked at {s:g} m, reading {scale:g} m")
        pos = gi.get_indexer(z["keys"].astype(str))
        if (pos < 0).any():
            sys.exit(f"{p}: {int((pos < 0).sum())} banked keys not in grid_cells.csv "
                     "(grid changed since banking?)")
        np.add.at(S, pos, z["S"])
        np.add.at(W, pos, z["W"])
        np.add.at(nt, pos, z["nt"])
        np.add.at(seen, pos, 1)
        imgs |= set(z["imgs"].astype(str).tolist())
    return {"S": S, "W": W, "nt": nt, "seen": seen, "imgs": imgs, "n_parts": len(parts),
            "scales": sorted(scales)}


def _expected_counts(g: pd.DataFrame, parts_dir: Path, scale: float) -> np.ndarray | None:
    ep = parts_dir / f"expected_{scale:g}m.npz"
    if not ep.exists():
        return None
    e = _load(ep)
    return pd.Series(e["n"], index=e["keys"].astype(str)).reindex(g["key"]).to_numpy(np.float64)


def _edge_cells(g: pd.DataFrame) -> np.ndarray:
    """Cells adjacent to a UTM zone edge (lon = edge +- 1/32) or in 12N."""
    lons = [e + s * L.HALF_CELL for e in L.ZONE_EDGES for s in (-1.0, 1.0)]
    return np.isin(g["lon"].to_numpy(), lons) | (g["zone"].to_numpy() == "12N")


def _setup_for_assembly(g: pd.DataFrame, parts_dir: Path, scale: float) -> tuple[dict, np.ndarray]:
    """run.json (scale + grid must match; chunk is only recorded) and the
    expected counts (positive everywhere) shared by both assemblers."""
    man = _read_manifest(parts_dir)
    if man is None:
        sys.exit(f"{parts_dir / 'run.json'} missing -- nothing banked")
    diff = _manifest_diff(man, _manifest_want(g, man.get("chunk", 0), scale), skip=("chunk",))
    if diff:
        sys.exit(f"the partials were banked with {diff} (run.json, now)")
    exp = _expected_counts(g, parts_dir, scale)
    if exp is None:
        sys.exit(f"no expected counts at {scale:g} m in {parts_dir} -- nothing banked at this scale")
    if not (np.isfinite(exp) & (exp > 0)).all():
        sys.exit(f"expected_{scale:g}m.npz: {int((~(np.isfinite(exp) & (exp > 0))).sum())} "
                 "grid cells without a positive expected count")
    return man, exp


# ---------------------------------------------------------------------------
# Public API: plan, burn, status
# ---------------------------------------------------------------------------
def dry_run(*, grid_csv=GRIDDED_GRID_CSV, parts_dir=GRIDDED_AEF_PARTS_DIR, chunk: int = 150,
            scale: float = DEFAULT_SCALE, max_units: int | None = None,
            years=(YEAR,)) -> None:
    """What ``run`` would reduce for ``years``, and its EECU cost.  No Earth Engine."""
    scale = _check_scale(scale)
    years = _check_years(years)
    parts_dir = Path(parts_dir)
    g = _grid(grid_csv)
    plan = {y: _todo(g, parts_dir, chunk, max_units, y)[0] for y in years}
    units = _todo(g, parts_dir, chunk, max_units, years[0])[1]
    print(f"  grid: {len(g)} cells ({grid_csv}); parts: {parts_dir}")
    for zone in ZONES:
        zu = [u for u in units if u[0] == zone]
        zt = sum(1 for y in years for u in plan[y] if u[0] == zone)
        print(f"  {zone}: {len(zu):4d} units, {sum(len(u[2]) for u in zu):5d} cells, "
              f"{zt:4d} units to do" + (f" over {len(years)} years" if len(years) > 1 else ""))
    if len(years) > 1:
        for y in years:
            print(f"    {y}: {len(plan[y]):4d}/{len(units)} units to do "
                  f"({sum(len(u[2]) for u in plan[y]):5d} cells)")
    n_todo = sum(len(plan[y]) for y in years)
    cells = sum(len(u[2]) for y in years for u in plan[y])
    eecu = cells * EECU_S_PER_CELL[scale]
    print(f"{n_todo} of {len(units) * len(years)} units to reduce ({cells} cells, <= {chunk} each) "
          f"@ {scale:g} m for {_years_label(years)}")
    if max_units is not None:
        print(f"  (--max-units {max_units}: only the first {max_units} units of the plan"
              + (" per year" if len(years) > 1 else "")
              + "; already-banked units among them are skipped)")
    print(f"  EECU estimate: ~{eecu:,.0f} EECU-s = {eecu / 3600:.1f} EECU-h "
          f"({EECU_S_PER_CELL[scale]} EECU-s per cell at {scale:g} m; inventory + expected "
          "counts ~0); Community tier = 150 EECU-h/month")
    have = _read_manifest(parts_dir)
    if have is not None:
        want = _manifest_want(g, chunk, scale)
        diff = _manifest_diff(have, want)
        if diff:
            print(f"  NOTE: run.json pins a different setup {diff} (banked, now) -- "
                  "--run would refuse these settings")
        else:
            print(f"  run.json: matches (chunk {chunk}, scale {scale:g} m, grid sha1 "
                  f"{want['grid_sha1'][:12]}); years started {_manifest_years(have)}"
                  + ("; legacy 2017-only form, kept" if "year" in have else ""))
    inv_have = [y for y in years if (parts_dir / f"inventory_{y}.json").exists()]
    inv_new = [y for y in years if y not in inv_have]
    print(f"  inventories: {'present for ' + _years_label(inv_have) if inv_have else 'none yet'}"
          + (f"; --run queries {_years_label(inv_new)} first" if inv_new else ""))


def run(project: str, *, grid_csv=GRIDDED_GRID_CSV, parts_dir=GRIDDED_AEF_PARTS_DIR,
        chunk: int = 150, scale: float = DEFAULT_SCALE, workers: int = 8,
        max_units: int | None = None, years=(YEAR,)) -> None:
    """The burn: an inventory per year (2017: + the earliest-year guard),
    expected counts, then every unbanked (year, unit), ``workers`` requests at
    a time from one pool, year by year.  Resumable.  The default (2017 only)
    continues a bank of the 2017-only pipeline unchanged."""
    scale = _check_scale(scale)
    years = _check_years(years)
    parts_dir = Path(parts_dir)
    g = _grid(grid_csv)
    _safe_stdout()
    ee = _ee(project)
    lock = _lock(parts_dir)
    failed = 0
    try:
        _manifest(parts_dir, g, chunk, scale, years)
        for y in years:
            _inventory(ee, g, parts_dir, y)
        _expected(ee, g, parts_dir, scale)
        jobs = []
        for y in years:
            todo, units = _todo(g, parts_dir, chunk, max_units, y)
            if len(todo) < len(units):
                print(f"Resuming {y} — {len(units) - len(todo)} of {len(units)} units already "
                      f"banked in {_year_dir(parts_dir, y)}", flush=True)
            jobs += [(y, zone, name, idx) for zone, name, idx in todo]
        cells = sum(len(j[3]) for j in jobs)
        print(f"{len(jobs)} units to reduce ({cells} cells, <= {chunk} each) for "
              f"{_years_label(years)} @ {scale:g} m, {workers} workers, "
              f"~{cells * EECU_S_PER_CELL[scale] / 3600:.1f} EECU-h", flush=True)
        keys = g["key"].to_numpy().astype("U")

        def one(job):
            year, zone, name, idx = job
            t0 = time.time()
            S, W, nt, imgs = _reduce(ee, g, zone, idx, scale, year)
            n_zero = _check_unit(S, W, nt, imgs, year)
            _atomic_npz(_year_dir(parts_dir, year) / f"{name}.npz", keys=keys[idx], S=S, W=W,
                        nt=nt, imgs=np.array(imgs, dtype="U"), scale=np.array(scale))
            return len(idx), n_zero, time.time() - t0

        done = streak = 0
        t0 = time.time()
        ex = cf.ThreadPoolExecutor(workers)
        futs = {ex.submit(one, j): j for j in jobs}
        try:
            for fut in cf.as_completed(futs):
                year, _, name, _ = futs[fut]
                try:
                    n, n_zero, dts = fut.result()
                except Fatal as e:
                    print(f"  FATAL {year} {name}: {e}", flush=True)
                    raise
                except Exception as e:                       # noqa: BLE001
                    failed += 1
                    streak += 1
                    print(f"  FAIL {year} {name}: {str(e)[:300]}", flush=True)
                    if streak >= 2 * workers:
                        raise RuntimeError(f"{streak} consecutive failures -- stopping")
                    continue
                done += 1
                streak = 0
                note = f"  ({n_zero} fully masked, banked as NaN)" if n_zero else ""
                print(f"  {done}/{len(jobs)} {year} {name}: {n} cells {dts:5.1f}s  "
                      f"[{(time.time() - t0) / 60:.1f} min]{note}", flush=True)
        except BaseException:
            print("stopping: cancelling queued units (in-flight ones finish and are banked)",
                  flush=True)
            ex.shutdown(wait=True, cancel_futures=True)
            raise
        ex.shutdown(wait=True)
    finally:
        lock.unlink(missing_ok=True)
    status(grid_csv=grid_csv, parts_dir=parts_dir, years=years)
    if failed:
        sys.exit(f"{failed} units failed (not banked) -- re-run to retry them")


def status(*, grid_csv=GRIDDED_GRID_CSV, parts_dir=GRIDDED_AEF_PARTS_DIR, years=None) -> None:
    """What is banked so far, per year (default: every year of ``MEAN_YEARS``).
    No Earth Engine."""
    parts_dir = Path(parts_dir)
    g = _grid(grid_csv)
    years = MEAN_YEARS if years is None else _check_years(years)
    have = _read_manifest(parts_dir)
    print(f"  parts: {parts_dir}")
    print(f"  setup: {json.dumps(have) if have else '(no run yet)'}")
    if have is not None and have.get("grid_sha1") != _grid_sha1(g):
        print(f"  WARNING run.json grid_sha1 {have.get('grid_sha1')} != this grid {_grid_sha1(g)}")
    exps = sorted(p.name for p in parts_dir.glob("expected_*m.npz"))
    print(f"  expected counts: {', '.join(exps) if exps else '(not yet)'}")
    zone = g["zone"].to_numpy()
    n_units = len(_units(g, int(have["chunk"]))) if have is not None else 0
    complete = []
    for y in years:
        inv = _read_inventory(parts_dir, y)
        b = _banked(g, parts_dir, None, y)
        if inv is None and b["n_parts"] == 0:
            print(f"  {y}: nothing banked, no inventory")
            continue
        if inv is None:
            print(f"  inventory {y}: (not yet)")
        else:
            zc = pd.Series([t.get("UTM_ZONE") for t in inv["tiles"]]).value_counts().sort_index()
            dv = sorted({str(t.get("DATASET_VERSION")) for t in inv["tiles"]})
            print(f"  inventory {y}: {len(inv['tiles'])} tiles {zc.to_dict()}, DATASET_VERSION "
                  f"{dv}, {'earliest-year check' if y == YEAR else 'year check'} "
                  f"{inv[_check_key(y)]}")
        seen, W, nt = b["seen"], b["W"], b["nt"]
        dup = int((seen > 1).sum())
        zero = int(((seen > 0) & (W == 0)).sum())
        print(f"  {y}: {b['n_parts']:4d} partials, {int((seen > 0).sum()):5d}/{len(g)} cells"
              + (f"  ({dup} banked twice!)" if dup else "")
              + (f"  ({zero} fully masked)" if zero else "")
              + (f"  ({int(((seen > 0) & (nt == 0)).sum())} with nt == 0!)"
                 if ((seen > 0) & (nt == 0)).any() else "")
              + (f"  scales {b['scales']}" if len(b["scales"]) > 1 else ""))
        if have is not None:
            n_todo = len(_todo(g, parts_dir, int(have["chunk"]), None, y)[0])
            units_text = f"; units {n_units - n_todo}/{n_units} banked (chunk {have['chunk']})"
        else:
            units_text = ""
        print("    " + ", ".join(f"{z} {int((seen[zone == z] > 0).sum())}/{int((zone == z).sum())}"
                                 for z in ZONES if (zone == z).any()) + units_text)
        if (seen == 1).all() and inv is not None:
            complete.append(y)
    if tuple(years) == MEAN_YEARS:
        print(f"  {MEAN_YEARS[0]}-{MEAN_YEARS[-1]} mean: {len(complete)}/{len(MEAN_YEARS)} years "
              f"complete{' -- ready for --assemble-mean' if len(complete) == len(MEAN_YEARS) else ''}"
              + (f" (complete: {_years_label(complete)})" if 0 < len(complete) < len(MEAN_YEARS)
                 else ""))


# ---------------------------------------------------------------------------
# Assembling
# ---------------------------------------------------------------------------
def assemble(*, grid_csv=GRIDDED_GRID_CSV, parts_dir=GRIDDED_AEF_PARTS_DIR,
             out_dir=GRIDDED_DIR, scale: float = DEFAULT_SCALE) -> dict:
    """Banked 2017 partials -> ``alphaearth_2017.nc`` (+ SHA256SUMS, provenance)."""
    scale = _check_scale(scale)
    parts_dir, out_dir = Path(parts_dir), Path(out_dir)
    g = _grid(grid_csv)
    n = len(g)
    man, exp = _setup_for_assembly(g, parts_dir, scale)

    b = _banked(g, parts_dir, scale, YEAR)
    S, W, nt, seen, imgs = b["S"], b["W"], b["nt"], b["seen"], b["imgs"]
    if (seen != 1).any():
        sys.exit(f"{int((seen == 0).sum())} of {n} cells not banked, {int((seen > 1).sum())} "
                 f"banked more than once ({b['n_parts']} partials) -- finish the run "
                 "(--status) or remove the duplicate partials")
    if (nt == 0).any():
        sys.exit(f"{int((nt == 0).sum())} banked cells with nt == 0 -- corrupt partials")
    vf = W / exp
    if vf.max() > 1 + FULL_TOL:  # tiles of one zone overlapping would double-count
        c = int(np.argmax(vf))
        sys.exit(f"valid_frac {vf[c]:.6f} > 1 at {g['key'][c]}: a pixel was counted twice")

    inv_path = parts_dir / f"inventory_{YEAR}.json"
    inv = _read_inventory(parts_dir, YEAR)
    if inv is None:
        sys.exit(f"{inv_path} missing -- re-run to rebuild the inventory")
    _validate_inventory(inv, g, inv_path, YEAR)
    tiles = {t["id"]: t for t in inv["tiles"]}
    unknown = sorted(imgs - set(tiles))
    dvs = sorted({str(tiles[i].get("DATASET_VERSION")) for i in imgs if i in tiles})
    if unknown:                    # the collection was republished mid-burn
        sys.exit(f"{len(unknown)} tiles used by the partials are not in {inv_path.name} "
                 f"(e.g. {unknown[:3]}) -- delete {_year_dir(parts_dir)} + its inventory and re-run")
    if len(dvs) != 1:
        sys.exit(f"mixed DATASET_VERSION {dvs} over the tiles used -- delete "
                 f"{_year_dir(parts_dir)} + its inventory and re-run")
    used = sorted(imgs)
    mvs = sorted({str(tiles[i].get("MODEL_VERSION")) for i in used})
    psvs = sorted({str(tiles[i].get("PROCESSING_SOFTWARE_VERSION")) for i in used})
    eyc = inv["earliest_year_check"]

    zero = W == 0
    valid = ~zero
    if not valid.any():
        sys.exit("no cell has valid pixels")
    emb = np.full((n, 64), np.nan)
    emb[valid] = S[valid] / W[valid, None]
    emb32 = emb.astype(np.float32)
    norm = np.full(n, np.nan)
    norm[valid] = np.linalg.norm(emb32[valid].astype(np.float64), axis=1)
    partial = vf < 1 - FULL_TOL
    low = vf < LOW_FRAC
    flag = (partial * _FLAG_PARTIAL | zero * _FLAG_NO_VALID | low * _FLAG_LOW).astype(np.uint8)
    susp = partial & _edge_cells(g)
    if zero.any():
        print(f"  NOTE {int(zero.sum())} cells without valid pixels (NaN, aef_flag no_valid_pixels): "
              f"{g['key'][zero].tolist()[:10]}")
    if susp.any():
        print(f"  WARNING {int(susp.sum())} zone-edge-adjacent / 12N cells with valid_frac < 1 "
              "(a zone's tiles should cover its edge cells fully): "
              + ", ".join(f"{k} {v:.4f}" for k, v in zip(g["key"][susp][:10], vf[susp][:10])))
    n_big = int((norm[valid] > 1 + FULL_TOL).sum())
    if n_big:
        print(f"  WARNING {n_big} cells with |embedding| > 1 (a mean of unit vectors) -- "
              "verify will fail; inspect before publishing")

    # -- dense grids ---------------------------------------------------------
    ilat, ilon = g["ilat"].to_numpy(), g["ilon"].to_numpy()
    grid_emb = np.full((64, L.NLAT, L.NLON), np.nan, dtype=np.float32)
    grid_emb[:, ilat[valid], ilon[valid]] = emb32[valid].T
    grid_vf = np.full((L.NLAT, L.NLON), np.nan, dtype=np.float32)
    grid_vf[ilat, ilon] = vf.astype(np.float32)
    grid_norm = np.full((L.NLAT, L.NLON), np.nan, dtype=np.float32)
    grid_norm[ilat[valid], ilon[valid]] = norm[valid].astype(np.float32)
    grid_nt = np.zeros((L.NLAT, L.NLON), dtype=np.int8)
    grid_nt[ilat, ilon] = np.clip(nt, 0, 127).astype(np.int8)
    grid_zone = np.zeros((L.NLAT, L.NLON), dtype=np.int8)
    grid_zone[ilat, ilon] = g["utm_zone"].to_numpy(np.int8)
    grid_flag = np.zeros((L.NLAT, L.NLON), dtype=np.uint8)
    grid_flag[ilat, ilon] = flag

    # -- attributes ----------------------------------------------------------
    today = dt.date.today().isoformat()
    git = ncio.git_state()
    cmd = ncio.command_line()
    how = _how(scale)
    eyc_text = _eyc_text(eyc)
    attrs = {
        "title": f"AlphaEarth Foundations satellite embedding, calendar year {YEAR}, "
                 "1/16-degree cell means for California",
        "summary": (f"Per cell of the statewide 1/16-degree Livneh lattice, the mean of the "
                    f"{YEAR} AlphaEarth Foundations 64-band pixel embeddings (A00..A63) over the "
                    "cell rectangle; one static vector per cell, no temporal averaging.  NaN "
                    "outside the domain (mask == 0) and on cells without valid pixels "
                    "(aef_flag no_valid_pixels)."),
        "source": (f"Google Earth Engine {COLL}, calendar year {YEAR} ({len(used)} UTM tile "
                   f"images); DATASET_VERSION {dvs[0]}, MODEL_VERSION {'/'.join(mvs)}, "
                   f"PROCESSING_SOFTWARE_VERSION {'/'.join(psvs)}; {CATALOG_URL}"),
        "attribution": ATTRIBUTION,
        "license": LICENSE,
        "references": REFERENCE,
        "modifications": (
            "Modified from the original (CC-BY-4.0 section 3(a)(1)(B)): the 10 m pixel "
            "embeddings were aggregated to 1/16-degree cell means over each cell rectangle; "
            f"each cell was reduced on its own UTM zone's tiles in native UTM on {how}; the "
            "cell means are not unit length."),
        "method": _method_text(scale),
        "year": np.int32(YEAR),
        "scale_m": float(scale),
        "images_used": ", ".join(used),
        "dataset_version": dvs[0],
        "earliest_year_check": eyc_text,
        "history": f"{today}: {cmd} (neuralhyd-ca git {git['git_commit'][:12] or 'unknown'}"
                   + (", dirty tree" if git["git_dirty"] == "yes" else "") + ")",
        "date_created": today,
    }

    # -- write ---------------------------------------------------------------
    final = out_dir / GRIDDED_AEF_NC.name
    out_dir.mkdir(parents=True, exist_ok=True)
    ds = ncio.create_dataset(ncio.part_path(final), g, global_attrs=attrs)
    try:
        _write_bands(ds)
        v = ds.createVariable("embedding", "f4", ("band", "lat", "lon"),
                              fill_value=np.float32(np.nan), chunksizes=(64, L.NLAT, L.NLON),
                              **ncio.ZLIB)
        v.setncatts({
            "long_name": f"AlphaEarth Foundations satellite embedding, {YEAR} cell mean",
            "units": "1", "cell_methods": "area: mean", "grid_mapping": "crs",
            "coordinates": "band_name",
            "comment": ("Mean of unit-length 64-d pixel embeddings over the cell: NOT unit "
                        "length.  norm < 1 measures within-cell heterogeneity; do not "
                        "renormalise.  Values are already dequantised by Earth Engine."),
        })
        v[:] = grid_emb
        for name, dtype, data, fill, a in (
            ("valid_frac", "f4", grid_vf, np.float32(np.nan), {
                "long_name": "fraction of the cell rectangle covered by valid (unmasked) "
                             f"{YEAR} AlphaEarth pixels", "units": "1"}),
            ("norm", "f4", grid_norm, np.float32(np.nan), {
                "long_name": "Euclidean norm of the cell-mean embedding", "units": "1"}),
            ("n_tiles", "i1", grid_nt, None, {
                "long_name": "number of AlphaEarth UTM tiles that reached the cell "
                             "(0 outside the domain)", "units": "1"}),
            ("utm_zone", "i1", grid_zone, None, _UTM_ZONE_ATTRS),
            ("aef_flag", "u1", grid_flag, None, {
                "long_name": "AlphaEarth coverage flags",
                "flag_masks": np.array([1, 2, 4], dtype=np.uint8),
                "flag_meanings": "partial_coverage no_valid_pixels low_coverage",
                "comment": (f"partial_coverage: valid_frac < 1 - {FULL_TOL:g}; "
                            "no_valid_pixels: sum(w) == 0 (embedding NaN); low_coverage: "
                            f"valid_frac < {LOW_FRAC:g}")}),
        ):
            kw = {"fill_value": fill} if fill is not None else {}
            var = ds.createVariable(name, dtype, ("lat", "lon"), **kw, **ncio.ZLIB)
            ncio.set_attrs(var, {**a, "grid_mapping": "crs"})
            var[:] = data
    finally:
        ds.close()
    ncio.finalize_all([final])

    sha = ncio.sha256_file(final)
    csha = ncio.sha256_array(grid_emb).hexdigest()
    size = final.stat().st_size
    nv = norm[valid]
    counts = {"cells": n, "valid": int(valid.sum()), "zero_w": int(zero.sum()),
              "partial": int(partial.sum()), "low": int(low.sum()),
              "edge_or_12n_partial": int(susp.sum()), "norm_gt_1": n_big}
    ncio.update_sha256sums({final.name: sha}, out_dir / "SHA256SUMS")
    ncio.update_provenance("alphaearth", {
        "file": final.name, "sha256": sha, "bytes": size, "content_sha256": csha,
        "content_hash_definition": _CONTENT_HASH_DEFINITION,
        "collection": COLL, "source_url": CATALOG_URL, "year": YEAR, "bands": "A00..A63",
        "dataset_version": dvs[0], "model_version": "/".join(mvs),
        "processing_software_version": "/".join(psvs),
        "images_used": used, "n_images_used": len(used), "n_inventory_tiles": len(tiles),
        "earliest_year_check": {k: eyc[k] for k in ("images_before_year", "images_in_year",
                                                    "passed", "checked")},
        "scale_m": scale, "chunk": int(man["chunk"]), "grid_sha1": _grid_sha1(g),
        "n_partials": b["n_parts"],
        "valid_frac_min": float(vf[valid].min()),
        "norm_min": float(nv.min()), "norm_max": float(nv.max()),
        "norm_median": float(np.median(nv)),
        "counts": counts,
        "attribution": ATTRIBUTION, "license": LICENSE, "reference": REFERENCE,
        "earthengine_api": str(inv.get("earthengine_api", "")),
        "library_versions": ncio.library_versions(),
        "git": git, "command": cmd, "date_created": today,
    }, out_dir / "provenance.toml")

    print(f"wrote {final} ({size / 1e6:.2f} MB): {n} cells x 64, {int(valid.sum())} valid"
          + (f", {int(zero.sum())} without valid pixels" if zero.any() else "")
          + f"; norm {nv.min():.3f}-{nv.max():.3f} (median {np.median(nv):.3f}), valid_frac "
          f"min {vf[valid].min():.4f} ({int(partial.sum())} partial, {int(low.sum())} low); "
          f"{len(used)} tiles, DATASET_VERSION {dvs[0]}")
    return {"path": str(final), "sha256": sha, "bytes": size, "content_sha256": csha,
            **counts, "valid_frac_min": float(vf[valid].min()),
            "norm_min": float(nv.min()), "norm_max": float(nv.max()),
            "dataset_version": dvs[0], "images_used": len(used)}


_UTM_ZONE_ATTRS = {
    "long_name": "UTM zone (north) whose tiles the cell was reduced on (0 outside the domain)",
    "flag_values": np.array([0, 10, 11, 12], dtype=np.int8),
    "flag_meanings": "outside_domain utm_10n utm_11n utm_12n"}


def _how(scale: float) -> str:
    return ("the full 10 m pixel grid" if scale == 10 else
            "a 15 m nearest-neighbour lattice of the full-resolution level (4/9 of the 10 m "
            "pixels; <= 2e-4 per band, cos >= 0.9999998 vs the full 10 m mean)")


def _method_text(scale: float) -> str:
    return (
        f"mean = sum(v*w) / sum(w) per cell, w = a constant band carrying A00's mask with "
        "Earth Engine's fractional boundary weights; per-tile ee.Reducer.sum() via "
        f"reduceRegions in the tile's native UTM projection at scale {scale:g} m, sums "
        "combined over the tiles of the cell's own zone (10N west of -120 deg, 11N to "
        "-114 deg, 12N east; both edges are cell edges; no mosaic).  valid_frac = sum(w) / "
        "the same sum of an unmasked constant on the same grid.  Values are Earth "
        "Engine's dequantised floats (sign(q)*(q/127.5)**2), averaged in float64 and "
        "stored as float32.")


def _eyc_text(eyc: dict) -> str:
    return (f"{eyc['images_before_year']} images before {YEAR}-01-01 and "
            f"{eyc['images_in_year']} images in {YEAR} in {COLL} (checked {eyc['checked']}): "
            f"{YEAR} is the collection's earliest year")


def _write_bands(ds: netCDF4.Dataset) -> None:
    """The ``band`` dimension + its index and name coordinates."""
    ds.createDimension("band", 64)
    v = ds.createVariable("band", "i4", ("band",))
    v.setncatts({"long_name": "AlphaEarth embedding band index (0 = A00 ... 63 = A63)"})
    v[:] = np.arange(64, dtype=np.int32)
    v = ds.createVariable("band_name", str, ("band",))
    v.setncatts({"long_name": "AlphaEarth embedding band name"})
    v[:] = np.array(BANDS, dtype=object)


def assemble_mean(*, grid_csv=GRIDDED_GRID_CSV, parts_dir=GRIDDED_AEF_PARTS_DIR,
                  out_dir=GRIDDED_DIR, scale: float = DEFAULT_SCALE) -> dict:
    """Banked partials of every year in ``MEAN_YEARS`` ->
    ``alphaearth_2017-2025_mean.nc`` (+ SHA256SUMS, provenance ``[alphaearth_mean]``).

    Per cell and year the cell mean exactly as :func:`assemble` computes it
    (``embedding_year``; its 2017 layer is bit for bit ``alphaearth_2017.nc``),
    NaN where the year has no valid pixels.  ``embedding`` = the EQUAL-WEIGHT
    mean over the years with valid pixels (W > 0), NOT renormalised: the
    float64 sum of the per-year means over those years divided by their
    number, cast to float32 once (so identical partials give identical
    float32 values); NaN where no year has valid pixels.  Refuses -- listing
    every missing year and cell -- unless every year is complete: each cell
    banked exactly once, no nt == 0, valid_frac <= 1 + 1e-6, a valid
    inventory holding every tile used, and one DATASET_VERSION per year (the
    years may differ)."""
    scale = _check_scale(scale)
    parts_dir, out_dir = Path(parts_dir), Path(out_dir)
    g = _grid(grid_csv)
    n, n_all = len(g), len(MEAN_YEARS)
    label = _years_label(MEAN_YEARS)
    keys = g["key"].to_numpy()
    man, exp = _setup_for_assembly(g, parts_dir, scale)

    # -- every year complete?  (collect every problem before refusing) --------
    banks: dict[int, dict] = {}
    missing: list[int] = []
    problems: list[str] = []
    for y in MEAN_YEARS:
        b = _banked(g, parts_dir, scale, y)
        if b["n_parts"] == 0:
            missing.append(y)
            continue
        seen, nt = b["seen"], b["nt"]
        bad = []
        if (seen == 0).any():
            bad.append(f"{int((seen == 0).sum())} of {n} cells not banked "
                       f"(e.g. {keys[seen == 0][:3].tolist()})")
        if (seen > 1).any():
            bad.append(f"{int((seen > 1).sum())} cells banked more than once "
                       f"(e.g. {keys[seen > 1][:3].tolist()})")
        if ((seen > 0) & (nt == 0)).any():
            bad.append(f"{int(((seen > 0) & (nt == 0)).sum())} cells with nt == 0 (corrupt partials)")
        if not (parts_dir / f"inventory_{y}.json").exists():
            bad.append(f"no inventory_{y}.json")
        if bad:
            problems.append(f"{y} ({b['n_parts']} partials): " + "; ".join(bad))
        banks[y] = b
    if missing or problems:
        lines = ([f"years with nothing banked: {missing}"] if missing else []) + problems
        sys.exit(f"refusing to assemble the {label} mean -- every year must be complete:\n    "
                 + "\n    ".join(lines)
                 + f"\n  finish the burn (aef --run --years all; aef --status) and re-run")

    # -- per year: cell means, coverage, versions -----------------------------
    means, fracs = [], []
    per_year: dict[str, dict] = {}
    eyc = None
    for y in MEAN_YEARS:
        b = banks[y]
        S, W, imgs = b["S"], b["W"], b["imgs"]
        vf = W / exp
        if vf.max() > 1 + FULL_TOL:  # tiles of one zone overlapping would double-count
            c = int(np.argmax(vf))
            sys.exit(f"valid_frac {vf[c]:.6f} > 1 at {keys[c]} in {y}: a pixel was counted twice")
        inv_path = parts_dir / f"inventory_{y}.json"
        inv = _read_inventory(parts_dir, y)
        _validate_inventory(inv, g, inv_path, y)
        tiles = {t["id"]: t for t in inv["tiles"]}
        unknown = sorted(imgs - set(tiles))
        if unknown:                # the collection was republished mid-burn
            sys.exit(f"{y}: {len(unknown)} tiles used by the partials are not in {inv_path.name} "
                     f"(e.g. {unknown[:3]}) -- delete {_year_dir(parts_dir, y)} + its inventory "
                     "and re-run that year")
        used = sorted(imgs)
        dvs = sorted({str(tiles[i].get("DATASET_VERSION")) for i in used})
        if len(dvs) != 1:
            sys.exit(f"{y}: mixed DATASET_VERSION {dvs} over the tiles used -- delete "
                     f"{_year_dir(parts_dir, y)} + its inventory and re-run that year")
        if y == YEAR:
            eyc = inv["earliest_year_check"]
        with np.errstate(invalid="ignore", divide="ignore"):
            means.append(S / W[:, None])           # NaN where W == 0; = assemble's 2017 means
        fracs.append(vf)
        zero = W == 0
        per_year[str(y)] = {
            "dataset_version": dvs[0],
            "model_version": "/".join(sorted({str(tiles[i].get("MODEL_VERSION")) for i in used})),
            "processing_software_version": "/".join(
                sorted({str(tiles[i].get("PROCESSING_SOFTWARE_VERSION")) for i in used})),
            "n_images_used": len(used), "images_used": used, "n_inventory_tiles": len(tiles),
            "images_in_year": int(inv[_check_key(y)]["images_in_year"]),
            "earthengine_api": str(inv.get("earthengine_api", "")),
            "n_partials": b["n_parts"], "zero_w": int(zero.sum()),
            "partial": int((vf < 1 - FULL_TOL).sum()), "low": int((vf < LOW_FRAC).sum()),
            "valid_frac_min": float(vf[~zero].min()) if (~zero).any() else float("nan"),
        }

    E = np.stack(means, axis=1)                    # (cells, years, 64), float64
    F = np.stack(fracs, axis=1)                    # (cells, years)
    ok = F > 0
    n_years = ok.sum(axis=1)
    valid = n_years > 0
    if not valid.any():
        sys.exit("no cell has valid pixels in any year")
    with np.errstate(invalid="ignore", divide="ignore"):
        # equal-weight mean over the valid years, then each year's cosine to it
        emb = np.where(ok[..., None], E, 0.0).sum(axis=1) / n_years[:, None]
        cos = (np.einsum("cyk,ck->cy", E, emb)
               / (np.linalg.norm(E, axis=2) * np.linalg.norm(emb, axis=1)[:, None]))
    emb32 = emb.astype(np.float32)
    norm = np.full(n, np.nan)
    norm[valid] = np.linalg.norm(emb32[valid].astype(np.float64), axis=1)
    cos_min = np.where(ok, cos, np.inf).min(axis=1)
    cos_min[~valid] = np.nan
    partial = (F < 1 - FULL_TOL).any(axis=1)
    low = (F < LOW_FRAC).any(axis=1)
    few = n_years < n_all
    flag = (partial * _FLAG_PARTIAL | ~valid * _FLAG_NO_VALID | low * _FLAG_LOW
            | few * _FLAG_FEW_YEARS).astype(np.uint8)
    susp = partial & _edge_cells(g)
    dvs_year = [per_year[str(y)]["dataset_version"] for y in MEAN_YEARS]
    nimg_year = [per_year[str(y)]["n_images_used"] for y in MEAN_YEARS]
    if few.any():
        print(f"  NOTE {int(few.sum())} cells without valid pixels in some year (left out of their "
              f"mean; n_years < {n_all}, aef_flag missing_years), {int((~valid).sum())} in every "
              f"year (NaN): {keys[few][:10].tolist()}")
    if susp.any():
        print(f"  WARNING {int(susp.sum())} zone-edge-adjacent / 12N cells with valid_frac < 1 in "
              f"some year (a zone's tiles should cover its edge cells fully): "
              + ", ".join(f"{k} {v:.4f}" for k, v in zip(keys[susp][:10], F[susp].min(axis=1)[:10])))
    n_big = int((norm[valid] > 1 + FULL_TOL).sum())
    if n_big:
        print(f"  WARNING {n_big} cells with |embedding| > 1 (a mean of unit vectors) -- "
              "verify will fail; inspect before publishing")

    # -- dense grids ---------------------------------------------------------
    ilat, ilon = g["ilat"].to_numpy(), g["ilon"].to_numpy()
    grid_emb = np.full((64, L.NLAT, L.NLON), np.nan, dtype=np.float32)
    grid_emb[:, ilat[valid], ilon[valid]] = emb32[valid].T
    grid_ey = np.full((n_all, 64, L.NLAT, L.NLON), np.nan, dtype=np.float32)
    for k in range(n_all):
        v = ok[:, k]
        grid_ey[k][:, ilat[v], ilon[v]] = E[v, k].astype(np.float32).T
    grid_vf = np.full((n_all, L.NLAT, L.NLON), np.nan, dtype=np.float32)
    grid_vf[:, ilat, ilon] = F.T.astype(np.float32)
    grid_ny = np.zeros((L.NLAT, L.NLON), dtype=np.int8)
    grid_ny[ilat, ilon] = n_years.astype(np.int8)
    grid_norm = np.full((L.NLAT, L.NLON), np.nan, dtype=np.float32)
    grid_norm[ilat[valid], ilon[valid]] = norm[valid].astype(np.float32)
    grid_cos = np.full((L.NLAT, L.NLON), np.nan, dtype=np.float32)
    grid_cos[ilat[valid], ilon[valid]] = cos_min[valid].astype(np.float32)
    grid_zone = np.zeros((L.NLAT, L.NLON), dtype=np.int8)
    grid_zone[ilat, ilon] = g["utm_zone"].to_numpy(np.int8)
    grid_flag = np.zeros((L.NLAT, L.NLON), dtype=np.uint8)
    grid_flag[ilat, ilon] = flag

    # -- attributes ----------------------------------------------------------
    today = dt.date.today().isoformat()
    git = ncio.git_state()
    cmd = ncio.command_line()

    def per_year_text(vals) -> str:
        return ", ".join(f"{y}: {v}" for y, v in zip(MEAN_YEARS, vals))

    attrs = {
        "title": f"AlphaEarth Foundations satellite embedding, {label} mean of the annual "
                 "1/16-degree cell means for California",
        "summary": (f"Per cell of the statewide 1/16-degree Livneh lattice and per calendar year "
                    f"{label}, the mean of the AlphaEarth Foundations 64-band pixel embeddings "
                    "(A00..A63) over the cell rectangle (embedding_year), and their equal-weight "
                    "mean over the years with valid pixels (embedding): one static vector per "
                    "cell.  NaN outside the domain (mask == 0), in embedding_year on cell-years "
                    "without valid pixels, and in embedding where no year has any."),
        "source": (f"Google Earth Engine {COLL}, calendar years {label} ({sum(nimg_year)} UTM "
                   f"tile images); DATASET_VERSION per year {per_year_text(dvs_year)}; "
                   f"{CATALOG_URL}"),
        "source_url": CATALOG_URL,
        "attribution": ATTRIBUTION,
        "license": LICENSE,
        "references": REFERENCE,
        "modifications": (
            "Modified from the original (CC-BY-4.0 section 3(a)(1)(B)): for each calendar "
            f"year {label} the 10 m pixel embeddings were aggregated to 1/16-degree cell "
            "means over each cell rectangle; each cell was reduced on its own UTM zone's "
            f"tiles in native UTM on {_how(scale)}; the per-year cell means were then "
            "averaged with equal weight over the years in which the cell has valid pixels; "
            "neither the per-year nor the multi-year means are unit length."),
        "method": (_method_text(scale) + "  embedding = the equal-weight mean of "
                   "embedding_year over the years with valid pixels (sum(w) > 0), computed "
                   "in float64 from the per-year means; not renormalised."),
        "years": np.array(MEAN_YEARS, dtype=np.int32),
        "scale_m": float(scale),
        "dataset_version": per_year_text(dvs_year),
        "images_per_year": per_year_text(nimg_year),
        "earliest_year_check": _eyc_text(eyc),
        "history": f"{today}: {cmd} (neuralhyd-ca git {git['git_commit'][:12] or 'unknown'}"
                   + (", dirty tree" if git["git_dirty"] == "yes" else "") + ")",
        "date_created": today,
    }

    # -- write ---------------------------------------------------------------
    final = out_dir / GRIDDED_AEF_MEAN_NC.name
    out_dir.mkdir(parents=True, exist_ok=True)
    ds = ncio.create_dataset(ncio.part_path(final), g, global_attrs=attrs)
    try:
        _write_bands(ds)
        ds.createDimension("year", n_all)
        v = ds.createVariable("year", "i4", ("year",))
        v.setncatts({"long_name": "calendar year of the AlphaEarth annual layer", "units": "1"})
        v[:] = np.array(MEAN_YEARS, dtype=np.int32)
        v = ds.createVariable("dataset_version", str, ("year",))
        v.setncatts({"long_name": "AlphaEarth DATASET_VERSION of the year's tiles "
                                  "(one per year, from the Earth Engine image properties)"})
        v[:] = np.array(dvs_year, dtype=object)
        v = ds.createVariable("n_images_used", "i4", ("year",))
        v.setncatts({"long_name": "AlphaEarth UTM tile images used in the year", "units": "1"})
        v[:] = np.array(nimg_year, dtype=np.int32)
        v = ds.createVariable("embedding", "f4", ("band", "lat", "lon"),
                              fill_value=np.float32(np.nan), chunksizes=(64, L.NLAT, L.NLON),
                              **ncio.ZLIB)
        v.setncatts({
            "long_name": f"AlphaEarth Foundations satellite embedding, {label} equal-weight "
                         "mean of the annual cell means",
            "units": "1", "cell_methods": "area: mean time: mean", "grid_mapping": "crs",
            "coordinates": "band_name",
            "comment": ("Equal-weight mean of embedding_year over the years with valid pixels "
                        "(n_years): NOT unit length.  norm < 1 measures within-cell AND "
                        "between-year heterogeneity; do not renormalise.  Values are already "
                        "dequantised by Earth Engine."),
        })
        v[:] = grid_emb
        v = ds.createVariable("embedding_year", "f4", ("year", "band", "lat", "lon"),
                              fill_value=np.float32(np.nan),
                              chunksizes=(1, 64, L.NLAT, L.NLON), **ncio.ZLIB)
        v.setncatts({
            "long_name": "AlphaEarth Foundations satellite embedding, cell mean of each "
                         "calendar year",
            "units": "1", "cell_methods": "area: mean", "grid_mapping": "crs",
            "coordinates": "band_name",
            "comment": (f"Computed exactly as alphaearth_{YEAR}.nc embedding (its {YEAR} layer is "
                        "bit for bit that file's embedding); NaN on cell-years without valid "
                        "pixels.  Not unit length."),
        })
        v[:] = grid_ey
        v = ds.createVariable("valid_frac", "f4", ("year", "lat", "lon"),
                              fill_value=np.float32(np.nan), chunksizes=(1, L.NLAT, L.NLON),
                              **ncio.ZLIB)
        ncio.set_attrs(v, {"long_name": "fraction of the cell rectangle covered by valid "
                                        "(unmasked) AlphaEarth pixels of the year",
                           "units": "1", "grid_mapping": "crs"})
        v[:] = grid_vf
        for name, dtype, data, fill, a in (
            ("n_years", "i1", grid_ny, None, {
                "long_name": f"number of years {label} with valid pixels in the cell, i.e. "
                             "averaged into embedding (0 outside the domain)", "units": "1"}),
            ("norm", "f4", grid_norm, np.float32(np.nan), {
                "long_name": "Euclidean norm of the multi-year mean embedding", "units": "1",
                "comment": "< 1 measures within-cell and between-year heterogeneity"}),
            ("year_cos_min", "f4", grid_cos, np.float32(np.nan), {
                "long_name": "minimum over the valid years of the cosine between the year's "
                             "cell mean and the multi-year mean", "units": "1"}),
            ("utm_zone", "i1", grid_zone, None, _UTM_ZONE_ATTRS),
            ("aef_flag", "u1", grid_flag, None, {
                "long_name": "AlphaEarth coverage flags over the years",
                "flag_masks": np.array([1, 2, 4, 8], dtype=np.uint8),
                "flag_meanings": "partial_coverage no_valid_pixels low_coverage missing_years",
                "comment": (f"partial_coverage: valid_frac < 1 - {FULL_TOL:g} in any year; "
                            "no_valid_pixels: sum(w) == 0 in every year (embedding NaN); "
                            f"low_coverage: valid_frac < {LOW_FRAC:g} in any year; "
                            f"missing_years: n_years < {n_all}")}),
        ):
            kw = {"fill_value": fill} if fill is not None else {}
            var = ds.createVariable(name, dtype, ("lat", "lon"), **kw, **ncio.ZLIB)
            ncio.set_attrs(var, {**a, "grid_mapping": "crs"})
            var[:] = data
    finally:
        ds.close()
    ncio.finalize_all([final])

    sha = ncio.sha256_file(final)
    csha = ncio.sha256_array(grid_emb).hexdigest()
    csha_y = ncio.sha256_array(grid_ey).hexdigest()
    size = final.stat().st_size
    nv, cv = norm[valid], cos_min[valid]
    fv = F[ok]
    counts = {"cells": n, "valid": int(valid.sum()), "all_years": int((n_years == n_all).sum()),
              "some_years": int((valid & few).sum()), "no_valid": int((~valid).sum()),
              "partial_any": int(partial.sum()), "low_any": int(low.sum()),
              "edge_or_12n_partial": int(susp.sum()), "norm_gt_1": n_big}
    ncio.update_sha256sums({final.name: sha}, out_dir / "SHA256SUMS")
    ncio.update_provenance("alphaearth_mean", {
        "file": final.name, "sha256": sha, "bytes": size, "content_sha256": csha,
        "content_sha256_embedding_year": csha_y,
        "content_hash_definition": _MEAN_CONTENT_HASH_DEFINITION,
        "collection": COLL, "source_url": CATALOG_URL, "years": list(MEAN_YEARS),
        "bands": "A00..A63",
        "mean": ("equal weight over the years with valid pixels (sum(w) > 0), not "
                 "renormalised; float64 sum of the per-year means / n_years"),
        "dataset_version": dvs_year, "n_images_used": nimg_year,
        "earliest_year_check": {k: eyc[k] for k in ("images_before_year", "images_in_year",
                                                    "passed", "checked")},
        "scale_m": scale, "chunk": int(man["chunk"]), "grid_sha1": _grid_sha1(g),
        "n_partials": sum(banks[y]["n_parts"] for y in MEAN_YEARS),
        "valid_frac_min": float(fv.min()),
        "norm_min": float(nv.min()), "norm_max": float(nv.max()),
        "norm_median": float(np.median(nv)),
        "year_cos_min_min": float(cv.min()), "year_cos_min_median": float(np.median(cv)),
        "counts": counts,
        "attribution": ATTRIBUTION, "license": LICENSE, "reference": REFERENCE,
        "library_versions": ncio.library_versions(),
        "git": git, "command": cmd, "date_created": today,
        "per_year": per_year,
    }, out_dir / "provenance.toml")

    print(f"wrote {final} ({size / 1e6:.2f} MB): {n} cells x 64 x {n_all} years, "
          f"{int((n_years == n_all).sum())} with every year"
          + (f", {int((valid & few).sum())} with fewer" if (valid & few).any() else "")
          + (f", {int((~valid).sum())} with none (NaN)" if (~valid).any() else "")
          + f"; norm {nv.min():.3f}-{nv.max():.3f} (median {np.median(nv):.3f}), year-vs-mean "
          f"cos min {cv.min():.3f} (median {np.median(cv):.3f}), valid_frac min {fv.min():.4f}; "
          f"tiles/yr {nimg_year}; versions {dvs_year}")
    return {"path": str(final), "sha256": sha, "bytes": size, "content_sha256": csha,
            "content_sha256_embedding_year": csha_y, **counts,
            "norm_min": float(nv.min()), "norm_max": float(nv.max()),
            "year_cos_min_min": float(cv.min()), "dataset_version": dvs_year}


# ---------------------------------------------------------------------------
# Verification against Earth Engine
# ---------------------------------------------------------------------------
def _check_sample(g: pd.DataFrame, pool: np.ndarray, vf: np.ndarray, n: int,
                  seed: int) -> tuple[np.ndarray, dict[str, int], list[str]]:
    """Stratified check sample from ``pool`` (banked cells with valid pixels).

    Always all 12N cells and the 11N cells on the -114 edge; the rest of ``n``
    split between both sides of -120, partially covered cells and random.
    Returns the cell rows, the count per stratum and each cell's stratum."""
    rng = np.random.default_rng(seed)
    lon, zone = g["lon"].to_numpy(), g["zone"].to_numpy()
    e0, e1 = L.ZONE_EDGES
    h = L.HALF_CELL
    chosen: list[int] = []
    labels: list[str] = []
    counts: dict[str, int] = {}
    for name, m in (("12N", zone == "12N"), (f"11N @ {e1 - h}", lon == e1 - h)):
        idx = [int(i) for i in np.flatnonzero(pool & m) if i not in chosen]
        chosen += idx
        labels += [name] * len(idx)
        counts[name] = len(idx)
    per = max(n - len(chosen), 0) // 4
    for name, m in ((f"10N @ {e0 - h}", lon == e0 - h), (f"11N @ {e0 + h}", lon == e0 + h),
                    ("partial", vf < 1 - FULL_TOL)):
        cand = np.setdiff1d(np.flatnonzero(pool & m), chosen)
        take = rng.choice(cand, size=min(per, cand.size), replace=False) if per else []
        chosen += [int(i) for i in take]
        labels += [name] * len(take)
        counts[name] = len(take)
    cand = np.setdiff1d(np.flatnonzero(pool), chosen)
    take = rng.choice(cand, size=min(max(n - len(chosen), 0), cand.size), replace=False)
    chosen += [int(i) for i in take]
    labels += ["random"] * len(take)
    counts["random"] = len(take)
    return np.array(chosen, dtype=np.int64), counts, labels


def _check_tolerance(vf: np.ndarray) -> np.ndarray:
    """Per-cell multiplier of the check gates.

    1 on fully covered cells (and where valid_frac is unknown): the 2e-4 /
    1e-3 gates are set for full cells.  The 15 m lattice keeps 4/9 of the
    10 m pixels, so its error against the 10 m mean grows roughly as
    1/sqrt(number of valid pixels): partial cells with valid_frac >= LOW_FRAC
    get 1/sqrt(valid_frac) (<= 1.41x).  Low-coverage cells (valid_frac <
    LOW_FRAC, flagged in the product) are reported but not gated (NaN)."""
    vf = np.asarray(vf, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(~(vf < 1 - FULL_TOL), 1.0,
                        np.where(vf >= LOW_FRAC, 1.0 / np.sqrt(vf), np.nan))


def _gate(got: np.ndarray, ref: np.ndarray, vf: np.ndarray,
          labels: list[str]) -> tuple[bool, list[str]]:
    """Gate every sampled cell at its own tolerance (:func:`_check_tolerance`);
    returns the verdict and one summary line per stratum."""
    mult = _check_tolerance(vf)
    gated = np.isfinite(mult)
    missing = np.isnan(ref).any(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        d = np.abs(got - ref).max(axis=1)
        ng, nr = np.linalg.norm(got, axis=1), np.linalg.norm(ref, axis=1)
        dn = np.abs(ng - nr)
        cos = np.einsum("ck,ck->c", got, ref) / (ng * nr)
        within = (d < CHECK_MAX_D * mult) & (dn < CHECK_MAX_DNORM * mult)
    over = gated & ~missing & ~within
    lab = np.array(labels, dtype=object)
    lines = []
    for name in dict.fromkeys(labels):
        s = (lab == name) & ~missing
        n_miss = int(((lab == name) & missing).sum())
        if not s.any():
            lines.append(f"{name:<18s} {n_miss} cells, no mosaic mean")
            continue
        sg, si = s & gated, s & ~gated
        n_part = int((sg & (mult > 1)).sum())
        stats = (f"max|d| {d[sg].max():.2e}  max d|m| {dn[sg].max():.1e}  "
                 f"min cos {cos[sg].min():.8f}" if sg.any() else "no gated cell")
        lines.append(
            f"{name:<18s} {int(s.sum()):3d} cells  {stats}"
            + (f"  ({n_part} partial, tolerance x1/sqrt(valid_frac))" if n_part else "")
            + (f"  ({int(si.sum())} low-coverage, reported only: max|d| {d[si].max():.2e})"
               if si.any() else "")
            + (f"  ({n_miss} without mosaic mean)" if n_miss else "")
            + (f"  {int(over[s].sum())} OVER TOLERANCE" if over[s].any() else ""))
    return bool(not missing.any() and not over.any()), lines


def _product_cells(path: Path, g: pd.DataFrame, idx: np.ndarray, year: int | None,
                   hint: str) -> np.ndarray | None:
    """A product's vectors at the sampled cells (cells x 64, float64), read
    before any Earth Engine call: ``embedding`` (``year`` None) or the
    ``year`` layer of ``embedding_year``.  None, with a note, when the file is
    absent or a Git LFS pointer stub; an unreadable file stops the check."""
    if not path.exists():
        print(f"  note  {path} not assembled yet: not gated ({hint})")
        return None
    if ncio.is_lfs_pointer(path):
        print(f"  note  {path.name} is a Git LFS pointer stub: not gated "
              f"({ncio.LFS_PULL_HINT} fetches it)")
        return None
    try:
        with netCDF4.Dataset(path) as ds:
            ds.set_auto_mask(False)
            if year is None:
                e = ds["embedding"][:]
            else:
                ys = [int(v) for v in ds["year"][:]]
                if year not in ys:
                    sys.exit(f"{path}: no {year} layer (years {ys})")
                e = ds["embedding_year"][ys.index(year)]
    except (OSError, KeyError, IndexError) as exc:
        sys.exit(f"{path}: unreadable ({type(exc).__name__}: {exc}) -- re-assemble or "
                 "move it aside before spending Earth Engine quota")
    return e[:, g["ilat"].to_numpy()[idx], g["ilon"].to_numpy()[idx]].T.astype(np.float64)


def check(project: str, n: int, *, year: int = YEAR, grid_csv=GRIDDED_GRID_CSV,
          parts_dir=GRIDDED_AEF_PARTS_DIR, out_dir=GRIDDED_DIR, scale: float = DEFAULT_SCALE,
          seed: int = 0) -> bool:
    """Banked ``year`` partials (and the assembled products, when present) vs
    an INDEPENDENT reduction of a stratified sample: Reducer.mean on each
    zone's own ``year`` tile mosaic at the full 10 m in the zone's UTM CRS.

    Gate per cell: max|d| < 2e-4 per band and | |m| - |m10| | < 1e-3
    (reading a renormalised pyramid level shows up as |m| inflated by
    ~4e-3) on fully covered cells.  Partial cells with valid_frac >= 0.5 get
    both tolerances x 1/sqrt(valid_frac); cells below 0.5 (low_coverage) are
    reported but not gated (:func:`_check_tolerance`).  One line per stratum
    shows its max|d|.  Cells without valid pixels are excluded (the mosaic
    mean there is null).  The products gated too: ``alphaearth_2017.nc``
    for 2017, and the ``year`` layer of ``embedding_year`` in the 2017-2025
    mean.  Run it after assembling so one check gates the partials and the
    NetCDFs; the products are read before any Earth Engine call, and a Git
    LFS pointer stub is skipped with a note."""
    scale = _check_scale(scale)
    year = _check_years([year])[0]
    parts_dir, out_dir = Path(parts_dir), Path(out_dir)
    g = _grid(grid_csv)
    b = _banked(g, parts_dir, scale, year)
    exp = _expected_counts(g, parts_dir, scale)
    with np.errstate(invalid="ignore", divide="ignore"):
        vf = b["W"] / exp if exp is not None else np.full(len(g), np.nan)
    pool = (b["seen"] == 1) & (b["W"] > 0)
    if not pool.any():
        print(f"FAIL  check: nothing banked for {year} in {parts_dir} yet")
        return False
    idx, counts, labels = _check_sample(g, pool, vf, n, seed)

    # the products, read before spending quota
    prods: list[tuple[str, np.ndarray]] = []
    if year == YEAR:
        prod = out_dir / GRIDDED_AEF_NC.name
        got_p = _product_cells(prod, g, idx, None,
                               "only the banked partials are gated; run --assemble first to "
                               "gate the NetCDF too")
        if got_p is not None:
            prods.append((prod.name, got_p))
    mean = out_dir / GRIDDED_AEF_MEAN_NC.name
    got_m = _product_cells(mean, g, idx, year, f"run --assemble-mean to gate its {year} layer too")
    if got_m is not None:
        prods.append((f"{mean.name} embedding_year[{year}]", got_m))

    eecu = idx.size * EECU_S_PER_CELL[10.0]
    print(f"check {year}: {idx.size} cells {counts} vs the 10 m zone-mosaic mean -- "
          f"~{eecu:,.0f} EECU-s ({eecu / 3600:.2f} EECU-h, ~{EECU_S_PER_CELL[10.0]} per cell)",
          flush=True)
    print(f"  gate per cell: max|d| < {CHECK_MAX_D:g} per band and ||m|-|m10|| < "
          f"{CHECK_MAX_DNORM:g} on full cells, x 1/sqrt(valid_frac) on partial cells with "
          f"valid_frac >= {LOW_FRAC:g}; valid_frac < {LOW_FRAC:g} reported only", flush=True)
    _safe_stdout()
    ee = _ee(project)
    ref = np.full((idx.size, 64), np.nan)
    pos = {int(i): k for k, i in enumerate(idx)}
    zone = g["zone"].to_numpy()
    for z, crs in ZONES.items():
        mine = idx[zone[idx] == z]
        if mine.size == 0:
            continue
        mos = _collection(ee, year).filter(ee.Filter.eq("UTM_ZONE", z)).mosaic()
        for lo in range(0, mine.size, 50):
            sub = mine[lo:lo + 50]
            r = _retry(lambda: mos.reduceRegions(collection=_fc(ee, g, sub),
                                                 reducer=ee.Reducer.mean(), crs=crs, scale=10)
                       .map(lambda f: f.setGeometry(None)).getInfo(), f"check {year} {z}")
            for f in r["features"]:
                p = f["properties"]
                if p.get("A00") is not None:
                    ref[pos[int(p["i"])]] = [p[bn] for bn in BANDS]
    if np.isnan(ref).any():
        print(f"  FAIL  {int(np.isnan(ref).any(axis=1).sum())} sampled cells got no mosaic mean")
    got = b["S"][idx] / b["W"][idx, None]
    ok, lines = _gate(got, ref, vf[idx], labels)
    print(f"  {'PASS' if ok else 'FAIL'}  banked {year} partials vs 10 m zone mosaic")
    for line in lines:
        print(f"          {line}")
    for name, got_p in prods:
        ok_p, lines_p = _gate(got_p, ref, vf[idx], labels)
        print(f"  {'PASS' if ok_p else 'FAIL'}  {name} vs 10 m zone mosaic")
        for line in lines_p:
            print(f"          {line}")
        ok = ok and ok_p
    print(f"{'PASS' if ok else 'FAIL'}  check {year} ({idx.size} cells)")
    return ok


# ---------------------------------------------------------------------------
# Offline comparison with an external reference bank
# ---------------------------------------------------------------------------
def _compare_year(rp: Path, g: pd.DataFrame, parts_dir: Path, scale: float,
                  year: int) -> bool | None:
    """One year of :func:`compare_reference`: the verdict, or None when no
    banked cell overlaps yet."""
    files = sorted((rp / str(year)).glob("*.npz"))
    tk, tS, tW, timgs = [], [], [], set()
    for p in files:
        with np.load(p, allow_pickle=False) as z:
            if float(z["scale"]) != scale:
                print(f"  {year}: FAIL  {p} banked at {float(z['scale']):g} m, ours {scale:g} m")
                return False
            tk += z["keys"].astype(str).tolist()
            tS.append(z["S"])
            tW.append(z["W"])
            timgs |= set(z["imgs"].astype(str).tolist())
    tS, tW = np.vstack(tS), np.concatenate(tW)
    if len(set(tk)) != len(tk):
        print(f"  {year}: FAIL  the reference banked {len(tk) - len(set(tk))} keys twice")
        return False
    b = _banked(g, parts_dir, scale, year)
    pos = pd.Index(g["key"]).get_indexer(tk)
    outside = pos < 0
    ov = ~outside
    ov[ov] = b["seen"][pos[ov]] == 1
    n_ov = int(ov.sum())
    print(f"  {year}: reference {len(tk)} cells in {len(files)} partials ({int(outside.sum())} "
          f"outside the statewide grid); ours {int((b['seen'] > 0).sum())}/{len(g)} banked; "
          f"overlap {n_ov} (reference cells not banked by us yet: {int((~outside).sum()) - n_ov})")
    if n_ov == 0:
        print(f"  {year}: no overlapping banked cells yet -- not compared")
        return None
    p = pos[ov]
    wt, wo = tW[ov], b["W"][p]
    rel_w = np.abs(wo - wt) / wt
    with np.errstate(invalid="ignore", divide="ignore"):
        d = np.abs(tS[ov] / wt[:, None] - b["S"][p] / wo[:, None]).max(axis=1)
    d = np.where(wo > 0, d, np.inf)             # we found no valid pixels where the reference did
    k = int(np.argmax(d))
    ok = bool(d.max() <= REF_TOL and rel_w.max() <= REF_TOL)
    common = timgs & b["imgs"]
    print(f"  {year}: max |S/W - ours| {d.max():.2e} (at {np.asarray(tk)[ov][k]}), max |dW|/W "
          f"{rel_w.max():.2e}, cells over {REF_TOL:g}: "
          f"{int(((d > REF_TOL) | (rel_w > REF_TOL)).sum())}; (info) tile ids reference "
          f"{len(timgs)}, ours {len(b['imgs'])}, common {len(common)}  "
          f"-> {'PASS' if ok else 'FAIL'}")
    return ok


def _compare_mean(ours: Path, theirs: Path, g: pd.DataFrame) -> bool:
    """Our ``alphaearth_2017-2025_mean.nc`` vs a reference mean npz on the
    overlapping cells (read without pickles, so cells are matched by its
    float64 ``lat``/``lon``, never by an object-array ``keys``)."""
    need = ("lat", "lon", "emb", "n_years", "years", "dataset_version", "valid_frac",
            "year_cos_min", "norm")
    try:
        with np.load(theirs, allow_pickle=False) as z:
            miss = [k for k in need if k not in z.files]
            if miss:
                print(f"  mean: FAIL  {theirs} lacks {miss} -- not a reference mean")
                return False
            t = {k: z[k] for k in need}
    except Exception as e:                                   # noqa: BLE001
        print(f"  mean: FAIL  {theirs}: unreadable ({e})")
        return False
    if ncio.is_lfs_pointer(ours):
        print(f"  mean: FAIL  {ours.name} is a Git LFS pointer stub ({ncio.LFS_PULL_HINT})")
        return False
    with netCDF4.Dataset(ours) as ds:
        ds.set_auto_mask(False)
        o_years = [int(v) for v in ds["year"][:]]
        o_dv = [str(v) for v in ds["dataset_version"][:]]
        o_emb, o_ny = ds["embedding"][:], ds["n_years"][:]
        o_vf, o_cm, o_norm = ds["valid_frac"][:], ds["year_cos_min"][:], ds["norm"][:]
    t_years = [int(y) for y in t["years"]]
    t_dv = [str(v) for v in t["dataset_version"]]
    if t_years != o_years:
        print(f"  mean: FAIL  reference years {t_years} != ours {o_years}")
        return False
    tkeys = [L.cell_key(a, b) for a, b in zip(t["lat"], t["lon"])]
    pos = pd.Index(g["key"]).get_indexer(tkeys)
    inside = pos >= 0
    ilat, ilon = g["ilat"].to_numpy()[pos[inside]], g["ilon"].to_numpy()[pos[inside]]
    ours_e = o_emb[:, ilat, ilon].T.astype(np.float64)
    theirs_e = t["emb"][inside].astype(np.float64)
    nan_o, nan_t = np.isnan(ours_e).any(axis=1), np.isnan(theirs_e).any(axis=1)
    both = ~nan_o & ~nan_t
    d = np.abs(ours_e[both] - theirs_e[both]).max(axis=1) if both.any() else np.zeros(0)
    exact = int((o_emb[:, ilat, ilon].T[both].view(np.uint32)
                 == t["emb"][inside][both].view(np.uint32)).all(axis=1).sum())
    ny_eq = o_ny[ilat, ilon] == t["n_years"][inside]
    dv_eq = t_dv == o_dv
    with np.errstate(invalid="ignore"):
        d_vf = np.nanmax(np.abs(o_vf[:, ilat, ilon].T - t["valid_frac"][inside]))
        d_cm = np.nanmax(np.abs(o_cm[ilat, ilon] - t["year_cos_min"][inside]))
        d_nm = np.nanmax(np.abs(o_norm[ilat, ilon] - t["norm"][inside]))
    n_in = int(inside.sum())
    print(f"  mean: reference {theirs}: {len(tkeys)} cells ({len(tkeys) - n_in} outside the "
          f"statewide grid), years {_years_label(t_years)}, dataset_version {t_dv}")
    print(f"  mean: max |emb - ours| {d.max() if d.size else float('nan'):.2e} over {int(both.sum())} "
          f"cells ({exact} bit-identical), NaN mismatches {int((nan_o != nan_t).sum())}, n_years "
          f"equal on {int(ny_eq.sum())}/{n_in}, dataset_version per year equal: {dv_eq}"
          + ("" if dv_eq else f" (ours {o_dv})"))
    print(f"  mean: (info) max |d| valid_frac {d_vf:.1e}, year_cos_min {d_cm:.1e}, norm {d_nm:.1e}")
    ok = bool(n_in > 0 and both.sum() == n_in and d.max() <= REF_MEAN_TOL
              and ny_eq.all() and dv_eq)
    print(f"  mean: {'PASS' if ok else 'FAIL'}  {n_in} overlapping cells within "
          f"{REF_MEAN_TOL:g}")
    return ok


def compare_reference(ref_parts_dir, *, grid_csv=GRIDDED_GRID_CSV,
                      parts_dir=GRIDDED_AEF_PARTS_DIR, out_dir=GRIDDED_DIR,
                      scale: float = DEFAULT_SCALE, ref_mean=None) -> bool:
    """Offline comparison with an external reference bank in the same format
    (read only): per-year partials ``<ref_parts_dir>/<year>/*.npz`` holding
    ``keys``, ``S``, ``W``, ``imgs`` and ``scale`` (an optional ``run.json``
    must pin the same scale; an optional ``expected_<scale>m.npz`` is
    compared for information).

    Every year both banks hold: same collection, year, method and scale ->
    the cell means and weights of the overlapping keys must agree to
    rounding (gate REF_TOL = 1e-9).  With ``ref_mean`` (a mean npz holding
    ``lat``, ``lon``, ``emb``, ``n_years``, ``years``, ``dataset_version``,
    ``valid_frac``, ``year_cos_min`` and ``norm``), the ``embedding`` of
    ``alphaearth_2017-2025_mean.nc`` in ``out_dir`` must also equal its
    ``emb`` on the overlapping cells: identical partials give identical
    float32 values, partials that agree to ~1e-13 at most one float32 ulp
    apart (gate REF_MEAN_TOL = 1e-7), with equal ``n_years`` and the same
    DATASET_VERSION per year; a mean missing on either side is then a
    failure.  Without ``ref_mean`` only the partials are compared.  PASS
    needs at least one comparison made and every one to pass."""
    scale = _check_scale(scale)
    rp, parts_dir, out_dir = Path(ref_parts_dir), Path(parts_dir), Path(out_dir)
    man = rp / "run.json"
    if man.exists() and float(json.loads(man.read_text()).get("scale", scale)) != scale:
        print(f"FAIL  compare_reference: {man} scale {json.loads(man.read_text())['scale']} "
              f"!= {scale:g}")
        return False
    rp_years = [y for y in MEAN_YEARS if any((rp / str(y)).glob("*.npz"))]
    if not rp_years:
        print(f"FAIL  compare_reference: no partials of {_years_label(MEAN_YEARS)} in {rp}")
        return False
    g = _grid(grid_csv)
    print(f"  reference partials: {rp} (years {_years_label(rp_years)}); ours: {parts_dir}")
    verdicts: dict[int, bool] = {}
    for y in rp_years:
        if not any(_year_dir(parts_dir, y).glob("*.npz")):
            print(f"  {y}: nothing banked by us yet -- not compared")
            continue
        ok_y = _compare_year(rp, g, parts_dir, scale, y)
        if ok_y is not None:
            verdicts[y] = ok_y
    ep_t, ep_o = rp / f"expected_{scale:g}m.npz", parts_dir / f"expected_{scale:g}m.npz"
    if ep_t.exists() and ep_o.exists():                      # information only
        with np.load(ep_t, allow_pickle=False) as z:
            et = pd.Series(z["n"], index=z["keys"].astype(str))
        eo = _expected_counts(g, parts_dir, scale)
        e_ov = et.reindex(g["key"]).to_numpy()
        m = np.isfinite(e_ov) & np.isfinite(eo)
        if m.any():
            print(f"  (info) expected counts: {int(m.sum())} common cells, max rel diff "
                  f"{np.max(np.abs(eo[m] - e_ov[m]) / e_ov[m]):.2e}")

    ok_mean = None
    ours = out_dir / GRIDDED_AEF_MEAN_NC.name
    if ref_mean is None:
        print("  mean: no reference mean given (--compare-mean) -- only the partials compared")
    elif not ours.exists():
        print(f"  mean: {ours} not assembled -- not compared (aef --assemble-mean)")
        print("FAIL  compare_reference: --compare-mean given but there is no mean of ours")
        return False
    elif not Path(ref_mean).exists():
        print(f"  mean: reference mean {ref_mean} not found -- not compared (FAIL)")
        ok_mean = False
    else:
        ok_mean = _compare_mean(ours, Path(ref_mean), g)

    if not verdicts and ok_mean is None:
        print("FAIL  compare_reference: nothing overlapping to compare")
        return False
    ok = all(verdicts.values()) and ok_mean is not False
    what = [f"{len(verdicts)} year(s) of partials ({_years_label(sorted(verdicts))})"
            if verdicts else "no partials"]
    if ok_mean is not None:
        what.append("the mean")
    failed = [str(y) for y, v in verdicts.items() if not v] + (["mean"] if ok_mean is False else [])
    print(f"{'PASS' if ok else 'FAIL'}  compare_reference: {' + '.join(what)}"
          + ("" if ok else f"; failed: {', '.join(failed)}"))
    return ok


# ---------------------------------------------------------------------------
# Offline verification of the products
# ---------------------------------------------------------------------------
def verify_aef(out_dir=GRIDDED_DIR, *, grid_csv=None) -> bool:
    """Offline consistency checks of ``alphaearth_2017.nc`` and, when present,
    ``alphaearth_2017-2025_mean.nc`` (the optional mean; its absence is
    reported, not failed)."""
    out_dir = Path(out_dir)
    grid_csv = Path(grid_csv) if grid_csv is not None else out_dir / GRIDDED_GRID_CSV.name
    ok = _verify_2017(out_dir, grid_csv)
    mean = out_dir / GRIDDED_AEF_MEAN_NC.name
    if mean.exists():
        ok = _verify_mean(out_dir, grid_csv) and ok
    else:
        print(f"  note  {mean.name} not present (the optional {_years_label(MEAN_YEARS)} mean, "
              "aef --assemble-mean) -- not verified")
    return ok


def _lattice_ok(lat, lon, latb, lonb) -> bool:
    return bool(np.array_equal(lat, L.lat_axis()) and np.array_equal(lon, L.lon_axis())
                and np.array_equal(latb, L.axis_bounds(L.lat_axis()))
                and np.array_equal(lonb, L.axis_bounds(L.lon_axis())))


def _verify_2017(out_dir: Path, grid_csv: Path) -> bool:
    """``alphaearth_2017.nc``."""
    path = out_dir / GRIDDED_AEF_NC.name
    ok_all = True

    def rep(label: str, ok: bool, detail: str = "") -> None:
        nonlocal ok_all
        ok_all &= bool(ok)
        print(f"  {'PASS' if ok else 'FAIL'}  {label}" + (f" ({detail})" if detail else ""))

    print(f"verify {path}")
    if not path.exists() or not grid_csv.exists():
        rep("inputs present", False, f"{path.name}: {path.exists()}, {grid_csv}: {grid_csv.exists()}")
        return False
    if ncio.is_lfs_pointer(path):
        rep(f"{path.name} is the product, not a Git LFS pointer stub", False,
            f"fetch it with {ncio.LFS_PULL_HINT}")
        return False
    g = L.read_grid_csv(grid_csv)
    try:
        with netCDF4.Dataset(path) as ds:
            ds.set_auto_mask(False)
            lat, lon = ds["lat"][:], ds["lon"][:]
            latb, lonb = ds["lat_bnds"][:], ds["lon_bnds"][:]
            mask = ds["mask"][:]
            band, names = ds["band"][:], list(ds["band_name"][:])
            ev = ds["embedding"]
            emb = ev[:]
            edims, edtype = ev.dimensions, ev.dtype
            vf, norm = ds["valid_frac"][:], ds["norm"][:]
            nt, zone, flag = ds["n_tiles"][:], ds["utm_zone"][:], ds["aef_flag"][:]
            gattrs = {k: ds.getncattr(k) for k in ds.ncattrs()}
    except (OSError, KeyError, IndexError) as e:
        rep("file readable with every variable", False, f"{type(e).__name__}: {e}")
        return False

    rep("lat/lon are the lattice", _lattice_ok(lat, lon, latb, lonb))
    rep(f"mask equals {grid_csv.name}", np.array_equal(mask, L.mask_from_cells(g)),
        f"{int(mask.sum())} vs {len(g)} cells")
    rep("64 bands A00..A63", np.array_equal(band, np.arange(64)) and names == BANDS
        and edims == ("band", "lat", "lon") and emb.shape == (64, L.NLAT, L.NLON)
        and edtype == np.float32)

    inm = mask == 1
    no_valid = inm & ((flag & _FLAG_NO_VALID) != 0)
    valid = inm & ~no_valid
    fin = np.isfinite(emb)
    rep("embedding finite exactly on valid cells, NaN elsewhere",
        bool(fin[:, valid].all() and np.isnan(emb[:, ~valid]).all()),
        f"{int(valid.sum())} valid, {int(no_valid.sum())} no_valid_pixels")
    vfv = vf[valid]
    rep("valid_frac in (0, 1+1e-6] on valid cells, 0 on no-valid cells, NaN outside",
        bool((vfv > 0).all() and (vfv <= 1 + FULL_TOL).all() and (vf[no_valid] == 0).all()
             and np.isnan(vf[~inm]).all()),
        f"min {vfv.min():.4f}" if vfv.size else "")
    want = np.zeros_like(flag)
    with np.errstate(invalid="ignore"):
        want[inm] = ((vf[inm] < 1 - FULL_TOL) * _FLAG_PARTIAL | (vf[inm] == 0) * _FLAG_NO_VALID
                     | (vf[inm] < LOW_FRAC) * _FLAG_LOW).astype(np.uint8)
    rep("aef_flag consistent with valid_frac", np.array_equal(flag, want),
        f"partial {int(((flag & _FLAG_PARTIAL) != 0).sum())}, low "
        f"{int(((flag & _FLAG_LOW) != 0).sum())}")
    en = np.linalg.norm(emb[:, valid].astype(np.float64), axis=0)
    nv = norm[valid].astype(np.float64)
    rep("norm == |embedding| (1e-6) and <= 1+1e-6, NaN elsewhere",
        bool(np.allclose(nv, en, rtol=0, atol=1e-6) and (nv <= 1 + FULL_TOL).all()
             and np.isnan(norm[~valid]).all()),
        f"{nv.min():.4f}-{nv.max():.4f}" if nv.size else "")
    rep("n_tiles >= 1 in the domain, 0 outside", bool((nt[inm] >= 1).all() and (nt[~inm] == 0).all()))
    zz = np.broadcast_to(L.utm_zone(lon)[None, :], zone.shape)
    rep("utm_zone matches the cell lon, 0 outside",
        bool(np.array_equal(zone[inm], zz[inm]) and (zone[~inm] == 0).all()))
    rep("attribution / license / year attributes",
        gattrs.get("attribution") == ATTRIBUTION and gattrs.get("license") == LICENSE
        and int(gattrs.get("year", -1)) == YEAR)

    prov = ncio.read_provenance(out_dir / "provenance.toml").get("alphaearth", {})
    csha = ncio.sha256_array(emb).hexdigest()
    rep("content_sha256 matches provenance.toml", prov.get("content_sha256") == csha,
        "no [alphaearth] section" if not prov else "")
    sums = ncio.read_sha256sums(out_dir / "SHA256SUMS")
    fsha = ncio.sha256_file(path)
    rep("SHA256SUMS matches", sums.get(path.name) == fsha,
        "no entry" if path.name not in sums else "")
    print(f"{'PASS' if ok_all else 'FAIL'}  {path.name}")
    return ok_all


def _verify_mean(out_dir: Path, grid_csv: Path) -> bool:
    """``alphaearth_2017-2025_mean.nc``: layout, NaN pattern, the mean
    recomputed from ``embedding_year``, derived fields, flags, hashes, and
    its 2017 layer bit for bit against ``alphaearth_2017.nc`` (when that is
    present)."""
    path = out_dir / GRIDDED_AEF_MEAN_NC.name
    n_all = len(MEAN_YEARS)
    ok_all = True

    def rep(label: str, ok: bool, detail: str = "") -> None:
        nonlocal ok_all
        ok_all &= bool(ok)
        print(f"  {'PASS' if ok else 'FAIL'}  {label}" + (f" ({detail})" if detail else ""))

    print(f"verify {path}")
    if not grid_csv.exists():
        rep("inputs present", False, f"{grid_csv}: missing")
        return False
    if ncio.is_lfs_pointer(path):
        rep(f"{path.name} is the product, not a Git LFS pointer stub", False,
            f"fetch it with {ncio.LFS_PULL_HINT}")
        return False
    g = L.read_grid_csv(grid_csv)
    try:
        with netCDF4.Dataset(path) as ds:
            ds.set_auto_mask(False)
            lat, lon = ds["lat"][:], ds["lon"][:]
            latb, lonb = ds["lat_bnds"][:], ds["lon_bnds"][:]
            mask = ds["mask"][:]
            band, names = ds["band"][:], list(ds["band_name"][:])
            years = [int(v) for v in ds["year"][:]]
            dvs = [str(v) for v in ds["dataset_version"][:]]
            nimg = ds["n_images_used"][:]
            ev, eyv = ds["embedding"], ds["embedding_year"]
            emb, ey = ev[:], eyv[:]
            layout = (ev.dimensions, ev.dtype, eyv.dimensions, eyv.dtype)
            vf, ny = ds["valid_frac"][:], ds["n_years"][:]
            norm, cmin = ds["norm"][:], ds["year_cos_min"][:]
            zone, flag = ds["utm_zone"][:], ds["aef_flag"][:]
            gattrs = {k: ds.getncattr(k) for k in ds.ncattrs()}
    except (OSError, KeyError, IndexError) as e:
        rep("file readable with every variable", False, f"{type(e).__name__}: {e}")
        return False

    rep("lat/lon are the lattice", _lattice_ok(lat, lon, latb, lonb))
    rep(f"mask equals {grid_csv.name}", np.array_equal(mask, L.mask_from_cells(g)),
        f"{int(mask.sum())} vs {len(g)} cells")
    rep(f"64 bands A00..A63, years {_years_label(MEAN_YEARS)}, embedding (band, lat, lon) and "
        "embedding_year (year, band, lat, lon) float32",
        np.array_equal(band, np.arange(64)) and names == BANDS and years == list(MEAN_YEARS)
        and layout == (("band", "lat", "lon"), np.float32, ("year", "band", "lat", "lon"),
                       np.float32)
        and emb.shape == (64, L.NLAT, L.NLON) and ey.shape == (n_all, 64, L.NLAT, L.NLON),
        f"years {years}")
    rep("one dataset_version and >= 1 image per year",
        len(dvs) == n_all and all(dvs) and (np.asarray(nimg) >= 1).all(),
        ", ".join(f"{y}: {v}" for y, v in zip(years, dvs)))

    inm = mask == 1
    fin_y = np.isfinite(ey).all(axis=1)                    # (year, lat, lon)
    nan_y = np.isnan(ey).all(axis=1)
    with np.errstate(invalid="ignore"):
        vy = inm[None] & (vf > 0)                          # cell-years with valid pixels
        rep("valid_frac in [0, 1+1e-6] in the domain, NaN outside",
            bool((vf[:, inm] >= 0).all() and (vf[:, inm] <= 1 + FULL_TOL).all()
                 and np.isnan(vf[:, ~inm]).all()),
            f"min {np.nanmin(vf[:, inm]):.4f}, {int((vf[:, inm] == 0).sum())} cell-years at 0")
    rep("embedding_year finite exactly on cell-years with valid_frac > 0, NaN elsewhere",
        bool(np.array_equal(fin_y, vy) and np.array_equal(nan_y, ~vy)),
        f"{int(vy.sum())} cell-years")
    nfin = fin_y.sum(axis=0)
    rep("n_years == number of years with a finite embedding_year, 0 outside",
        bool(np.array_equal(ny[inm].astype(np.int64), nfin[inm]) and (ny[~inm] == 0).all()),
        f"{int((ny[inm] == n_all).sum())} cells with all {n_all}, "
        f"{int(((ny[inm] > 0) & (ny[inm] < n_all)).sum())} with fewer, {int((ny[inm] == 0).sum())} "
        "with none")
    valid = inm & (ny > 0)
    rep("embedding finite exactly where n_years > 0, NaN elsewhere",
        bool(np.isfinite(emb[:, valid]).all() and np.isnan(emb[:, ~valid]).all()),
        f"{int(valid.sum())} cells")
    with np.errstate(invalid="ignore", divide="ignore"):
        mean64 = (np.where(fin_y[:, None], ey, 0.0).astype(np.float64).sum(axis=0)
                  / nfin[None].astype(np.float64))
    dmax = float(np.abs(emb[:, valid].astype(np.float64) - mean64[:, valid]).max()) if valid.any() else 0.0
    rep(f"embedding == mean of embedding_year over the valid years ({MEAN_TOL:g})",
        dmax <= MEAN_TOL, f"max |d| {dmax:.1e}")
    e64 = emb[:, valid].astype(np.float64)
    en = np.linalg.norm(e64, axis=0)
    nv = norm[valid].astype(np.float64)
    rep("norm == |embedding| (1e-6) and <= 1+1e-6, NaN elsewhere",
        bool(np.allclose(nv, en, rtol=0, atol=1e-6) and (nv <= 1 + FULL_TOL).all()
             and np.isnan(norm[~valid]).all()),
        f"{nv.min():.4f}-{nv.max():.4f}" if nv.size else "")
    with np.errstate(invalid="ignore", divide="ignore"):
        eyv64 = ey[:, :, valid].astype(np.float64)         # (year, band, cells)
        cos = (np.einsum("ybc,bc->yc", eyv64, e64)
               / (np.linalg.norm(eyv64, axis=1) * en[None]))
    cm = np.where(fin_y[:, valid], cos, np.inf).min(axis=0)
    dcos = float(np.abs(cm - cmin[valid]).max()) if valid.any() else 0.0
    rep(f"year_cos_min == min cosine(year, mean) over the valid years ({COS_TOL:g}), NaN elsewhere",
        dcos <= COS_TOL and bool(np.isnan(cmin[~valid]).all()),
        f"max |d| {dcos:.1e}, min {np.nanmin(cmin):.3f}" if valid.any() else "")
    want = np.zeros_like(flag)
    with np.errstate(invalid="ignore"):
        vfi = vf[:, inm]
        want[inm] = (((vfi < 1 - FULL_TOL).any(axis=0)) * _FLAG_PARTIAL
                     | (ny[inm] == 0) * _FLAG_NO_VALID
                     | ((vfi < LOW_FRAC).any(axis=0)) * _FLAG_LOW
                     | (ny[inm] < n_all) * _FLAG_FEW_YEARS).astype(np.uint8)
    rep("aef_flag consistent with valid_frac and n_years", np.array_equal(flag, want),
        f"partial {int(((flag & _FLAG_PARTIAL) != 0).sum())}, low "
        f"{int(((flag & _FLAG_LOW) != 0).sum())}, missing_years "
        f"{int(((flag & _FLAG_FEW_YEARS) != 0).sum())}")
    zz = np.broadcast_to(L.utm_zone(lon)[None, :], zone.shape)
    rep("utm_zone matches the cell lon, 0 outside",
        bool(np.array_equal(zone[inm], zz[inm]) and (zone[~inm] == 0).all()))
    rep("attribution / license / source_url / years attributes",
        gattrs.get("attribution") == ATTRIBUTION and gattrs.get("license") == LICENSE
        and gattrs.get("source_url") == CATALOG_URL
        and [int(y) for y in np.atleast_1d(gattrs.get("years", []))] == list(MEAN_YEARS))

    single = out_dir / GRIDDED_AEF_NC.name
    if YEAR not in years:
        rep(f"embedding_year holds {YEAR}", False)
    elif not single.exists() or ncio.is_lfs_pointer(single):
        print(f"  note  {single.name} {'absent' if not single.exists() else 'is an LFS stub'}: "
              f"its bit-for-bit match with embedding_year[{YEAR}] not checked")
    else:
        try:
            with netCDF4.Dataset(single) as ds:
                ds.set_auto_mask(False)
                e17 = ds["embedding"][:]
            same = e17.shape == ey[years.index(YEAR)].shape and np.array_equal(
                e17.view(np.uint32), ey[years.index(YEAR)].view(np.uint32))
        except (OSError, KeyError, IndexError) as e:
            same = False
            print(f"  note  {single.name} unreadable: {type(e).__name__}: {e}")
        rep(f"embedding_year[{YEAR}] == {single.name} embedding bit for bit", same)

    prov = ncio.read_provenance(out_dir / "provenance.toml").get("alphaearth_mean", {})
    rep("content_sha256 (embedding) matches provenance.toml",
        prov.get("content_sha256") == ncio.sha256_array(emb).hexdigest(),
        "no [alphaearth_mean] section" if not prov else "")
    rep("content_sha256_embedding_year matches provenance.toml",
        prov.get("content_sha256_embedding_year") == ncio.sha256_array(ey).hexdigest(),
        "no [alphaearth_mean] section" if not prov else "")
    sums = ncio.read_sha256sums(out_dir / "SHA256SUMS")
    rep("SHA256SUMS matches", sums.get(path.name) == ncio.sha256_file(path),
        "no entry" if path.name not in sums else "")
    print(f"{'PASS' if ok_all else 'FAIL'}  {path.name}")
    return ok_all
