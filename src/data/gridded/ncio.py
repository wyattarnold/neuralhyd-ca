"""Shared NetCDF, checksum and provenance helpers for the gridded products.

Every product file is NETCDF4 (HDF5) with zlib + shuffle only — zstd/blosc
filters are not portable (the neuralhyd env's netCDF4 wheel cannot load its
own zstd plugin).  Each file carries the same CF skeleton:

    lat(lat), lon(lon)            float64 cell centres, ascending
    lat_bnds, lon_bnds (·, nv)    float64 cell edges (± 1/32°)
    crs                           latitude_longitude grid mapping
    mask(lat, lon)                uint8, 1 on the 13,786 store cells

Files are written to ``<name>.part`` and renamed on success, so a killed
build never leaves a truncated product behind.  :func:`finalize_all` moves a
set of products into place together (or not at all).

A file's sha256 changes on EVERY rebuild, even when every value is
identical: the files embed ``history`` and ``date_created``, and the bytes
also depend on the HDF5 / netCDF-C build.  The provenance therefore also
records a *content* hash of the decoded values, which is stable across
rebuilds and libraries, plus the library versions.  :func:`command_line`
drops the directory part of absolute paths, so committed files carry no
local user paths.

Default clones hold the NetCDFs as Git LFS pointer stubs (``.lfsconfig``
fetchexclude); :func:`is_lfs_pointer` lets the build and the verifiers tell
a stub from a product.

``provenance.toml`` and ``SHA256SUMS`` sit in plain git (``*.json`` would be
an LFS pointer, unreadable on a clone without LFS).
"""
from __future__ import annotations

import hashlib
import os
import platform
import re
import subprocess
import sys
import tomllib
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

import netCDF4
import numpy as np
import pandas as pd

from src.data.gridded import lattice as L
from src.paths import GRIDDED_PROVENANCE, GRIDDED_SHA256SUMS, PROJECT_ROOT

#: Encoding of every gridded data variable.
ZLIB = dict(zlib=True, complevel=4, shuffle=True)

_HASH_BUF = 16 * 1024 * 1024
_BARE_KEY = re.compile(r"[A-Za-z0-9_-]+")

#: OGC WKT (GDAL WKT1 flavour, with EPSG authority codes) of the lattice's
#: CRS.  GDAL, rioxarray and pyproj read it from ``crs_wkt`` / ``spatial_ref``;
#: without it they build a CRS with an undefined datum from the ellipsoid.
EPSG4326_WKT = (
    'GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,'
    'AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0,'
    'AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,'
    'AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4326"]]'
)

_LFS_POINTER_PREFIX = b"version https://git-lfs.github.com/spec/v1"
#: How to fetch the committed NetCDFs over their pointer stubs.
LFS_PULL_HINT = 'git lfs pull --include="data/gridded/*.nc" --exclude=""'


# ---------------------------------------------------------------------------
# NetCDF skeleton
# ---------------------------------------------------------------------------
def create_dataset(path: str | os.PathLike, cells: pd.DataFrame, *,
                   n_time: int | None = None,
                   global_attrs: Mapping[str, Any] | None = None) -> netCDF4.Dataset:
    """Open ``path`` for writing with the shared CF skeleton filled in.

    ``n_time`` adds an int32 ``time`` coordinate (days since 1915-01-01,
    standard calendar).  The caller adds its data variables and closes it.
    """
    ds = netCDF4.Dataset(path, "w", format="NETCDF4")
    if n_time is not None:
        ds.createDimension("time", n_time)
    ds.createDimension("lat", L.NLAT)
    ds.createDimension("lon", L.NLON)
    ds.createDimension("nv", 2)

    if n_time is not None:
        t = ds.createVariable("time", "i4", ("time",))
        t.setncatts({"standard_name": "time", "long_name": "time", "axis": "T",
                     "units": L.TIME_UNITS, "calendar": L.CALENDAR})
        t[:] = np.arange(n_time, dtype=np.int32)

    lat, lon = L.lat_axis(), L.lon_axis()
    for name, values, std, units, axis in (
        ("lat", lat, "latitude", "degrees_north", "Y"),
        ("lon", lon, "longitude", "degrees_east", "X"),
    ):
        v = ds.createVariable(name, "f8", (name,))
        v.setncatts({"standard_name": std, "long_name": f"{std} of cell centre",
                     "units": units, "axis": axis, "bounds": f"{name}_bnds"})
        v[:] = values
        b = ds.createVariable(f"{name}_bnds", "f8", (name, "nv"))
        b[:] = L.axis_bounds(values)

    crs = ds.createVariable("crs", "i4")
    crs.setncatts({
        "grid_mapping_name": "latitude_longitude",
        "longitude_of_prime_meridian": 0.0,
        "semi_major_axis": 6378137.0,
        "inverse_flattening": 298.257223563,
        "geographic_crs_name": "WGS 84",
        "horizontal_datum_name": "WGS_1984",
        "reference_ellipsoid_name": "WGS 84",
        "prime_meridian_name": "Greenwich",
        "crs_wkt": EPSG4326_WKT,
        "spatial_ref": EPSG4326_WKT,          # the attribute older GDAL reads
        "epsg_code": "EPSG:4326",
        "comment": "Geographic lat/lon lattice of the Livneh 1/16-degree product",
    })

    m = ds.createVariable("mask", "u1", ("lat", "lon"), **ZLIB)
    m.setncatts({"long_name": "cell of the WGEN NonDetrend-Unsplit statewide store",
                 "flag_values": np.array([0, 1], dtype=np.uint8),
                 "flag_meanings": "outside_domain in_domain",
                 "grid_mapping": "crs"})
    m[:] = L.mask_from_cells(cells)

    edges_lat = L.axis_bounds(lat)
    edges_lon = L.axis_bounds(lon)
    attrs = {
        "Conventions": "CF-1.8",
        "geospatial_lat_min": float(edges_lat[0, 0]),
        "geospatial_lat_max": float(edges_lat[-1, 1]),
        "geospatial_lon_min": float(edges_lon[0, 0]),
        "geospatial_lon_max": float(edges_lon[-1, 1]),
        "geospatial_lat_resolution": "0.0625 degree",
        "geospatial_lon_resolution": "0.0625 degree",
        "grid": (f"dense {L.NLAT} x {L.NLON} lat/lon lattice, cell centres at "
                 f"0.03125 + k/16 deg, lat ascending; {len(cells)} in-domain cells "
                 f"(mask == 1), NaN elsewhere"),
    }
    attrs.update(global_attrs or {})
    set_attrs(ds, attrs)
    return ds


def set_attrs(obj: netCDF4.Dataset | netCDF4.Variable, attrs: Mapping[str, Any]) -> None:
    """Set attributes, skipping ``None`` values."""
    obj.setncatts({k: v for k, v in attrs.items() if v is not None})


def part_path(final: str | os.PathLike) -> Path:
    """Temporary sibling a product is written to before :func:`finalize`."""
    final = Path(final)
    return final.with_name(final.name + ".part")


def finalize(final: str | os.PathLike) -> Path:
    """Atomically move ``<final>.part`` into place."""
    final = Path(final)
    os.replace(part_path(final), final)
    return final


def _backup_path(final: Path) -> Path:
    # ends in .part so the gitignore rule for data/gridded/*.part covers it
    return final.with_name(final.name + ".old.part")


def finalize_all(finals: list[str | os.PathLike]) -> list[Path]:
    """Move every ``<final>.part`` into place together, or none of them.

    The existing products are first moved aside to ``<final>.old.part``.
    On Windows that is the step that fails when another program (xarray,
    Panoply, Excel) holds a product open, and at that point nothing new is
    in place yet.  Then the new files are renamed in and the backups are
    deleted.  On any failure the old files are restored, the new ``.part``
    files are deleted and ``SystemExit`` names the file to close.
    """
    finals = [Path(p) for p in finals]
    moved: list[Path] = []                 # finals moved aside to their backup
    placed: list[Path] = []                # new files renamed into place
    try:
        for f in finals:
            if f.exists():
                os.replace(f, _backup_path(f))
                moved.append(f)
        for f in finals:
            os.replace(part_path(f), f)
            placed.append(f)
    except OSError as exc:
        culprit = Path(getattr(exc, "filename", "") or finals[0]).name
        for f in placed:
            f.unlink(missing_ok=True)
        for f in moved:
            try:
                os.replace(_backup_path(f), f)
            except OSError:
                print(f"  WARNING could not restore {f.name} from {_backup_path(f).name}")
        for f in finals:
            part_path(f).unlink(missing_ok=True)
        raise SystemExit(
            f"could not move the new products into place ({type(exc).__name__}: {exc}).  "
            f"Nothing was replaced.  Close any program that has {culprit} (or another "
            f"product in {finals[0].parent}) open and re-run.") from exc
    for f in finals:
        _backup_path(f).unlink(missing_ok=True)
    return finals


def is_lfs_pointer(path: str | os.PathLike) -> bool:
    """True if ``path`` is a Git LFS pointer stub rather than the real file."""
    path = Path(path)
    try:
        if path.stat().st_size >= 1024:
            return False
        with open(path, "rb") as fh:
            return fh.read(len(_LFS_POINTER_PREFIX)) == _LFS_POINTER_PREFIX
    except OSError:
        return False


# ---------------------------------------------------------------------------
# Hashes
# ---------------------------------------------------------------------------
def sha256_file(path: str | os.PathLike) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(_HASH_BUF):
            h.update(chunk)
    return h.hexdigest()


def sha256_array(a: np.ndarray, h: "hashlib._Hash | None" = None) -> "hashlib._Hash":
    """Feed ``a`` (C order, little-endian) into ``h`` (a new sha256 if None)."""
    h = h or hashlib.sha256()
    a = np.ascontiguousarray(a)
    if a.dtype.byteorder == ">":
        a = a.astype(a.dtype.newbyteorder("<"))
    h.update(a.tobytes())
    return h


# ---------------------------------------------------------------------------
# Environment / provenance
# ---------------------------------------------------------------------------
def library_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "netCDF4": netCDF4.__version__,
        "netcdf_c": netCDF4.__netcdf4libversion__,
        "hdf5": netCDF4.__hdf5libversion__,
    }


def git_state() -> dict[str, str]:
    """HEAD commit of this repo and whether the tree is dirty ('' if unknown)."""
    def _git(*args: str) -> str:
        try:
            return subprocess.run(["git", *args], cwd=PROJECT_ROOT, capture_output=True,
                                  text=True, timeout=30).stdout.strip()
        except Exception:                                    # noqa: BLE001
            return ""
    return {"git_commit": _git("rev-parse", "HEAD"),
            "git_dirty": "yes" if _git("status", "--porcelain", "--untracked-files=no") else "no"}


def _short_arg(arg: str) -> str:
    """An absolute path reduced to its last component (``--opt=PATH`` too)."""
    if arg.startswith("--") and "=" in arg:
        opt, value = arg.split("=", 1)
        return f"{opt}={_short_arg(value)}"
    if os.path.isabs(arg) or arg.startswith(("/", "\\")):
        name = PurePosixPath(arg.replace("\\", "/").rstrip("/")).name
        return name or arg
    return arg


def command_line() -> str:
    """The invoking command, script name only and absolute paths shortened to
    their last component (committed files must not carry local user paths)."""
    return " ".join([Path(sys.argv[0]).name, *(_short_arg(a) for a in sys.argv[1:])])


# ---------------------------------------------------------------------------
# provenance.toml  (read-modify-write, one table per product)
# ---------------------------------------------------------------------------
def _toml_value(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        return repr(float(v))
    if isinstance(v, (list, tuple, np.ndarray)):
        return "[" + ", ".join(_toml_value(x) for x in v) + "]"
    s = str(v).replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
    return f'"{s}"'


def _toml_key(k: Any) -> str:
    """A key bare when TOML allows it, else quoted (a file name's dot would
    otherwise open a nested table)."""
    k = str(k)
    return k if _BARE_KEY.fullmatch(k) else _toml_value(k)


def _toml_table(lines: list[str], path: tuple[str, ...], table: Mapping[str, Any]) -> None:
    scalars = {k: v for k, v in table.items() if not isinstance(v, Mapping)}
    subs = {k: v for k, v in table.items() if isinstance(v, Mapping)}
    lines.append("[" + ".".join(_toml_key(p) for p in path) + "]")
    lines.extend(f"{_toml_key(k)} = {_toml_value(v)}" for k, v in scalars.items())
    lines.append("")
    for k, v in subs.items():
        _toml_table(lines, (*path, str(k)), v)


def read_provenance(path: str | os.PathLike = GRIDDED_PROVENANCE) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        return {}
    with open(path, "rb") as fh:
        return tomllib.load(fh)


def update_provenance(section: str, table: Mapping[str, Any],
                      path: str | os.PathLike = GRIDDED_PROVENANCE) -> Path:
    """Replace one top-level table of ``provenance.toml``; keep the others."""
    path = Path(path)
    doc = read_provenance(path)
    doc[section] = dict(table)
    lines = ["# Provenance of the statewide 1/16-degree gridded products.",
             "# Written by scripts/prepare_gridded.py -- do not edit by hand.", ""]
    for name in sorted(doc):
        _toml_table(lines, (name,), doc[name])
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    tmp.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    os.replace(tmp, path)
    return path


# ---------------------------------------------------------------------------
# SHA256SUMS  (``sha256sum -c`` format, names relative to the gridded dir)
# ---------------------------------------------------------------------------
def read_sha256sums(path: str | os.PathLike = GRIDDED_SHA256SUMS) -> dict[str, str]:
    path = Path(path)
    if not path.exists():
        return {}
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            digest, name = line.split(maxsplit=1)
            out[name.lstrip("*")] = digest
    return out


def update_sha256sums(files: Mapping[str, str],
                      path: str | os.PathLike = GRIDDED_SHA256SUMS) -> Path:
    """Merge ``{file name: sha256}`` into ``SHA256SUMS`` (sorted by name)."""
    path = Path(path)
    sums = read_sha256sums(path)
    sums.update(files)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    tmp.write_text("".join(f"{d}  {n}\n" for n, d in sorted(sums.items())),
                   encoding="utf-8", newline="\n")
    os.replace(tmp, path)
    return path
