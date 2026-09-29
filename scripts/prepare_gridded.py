"""Statewide 1/16° gridded inputs — preparation and QA.

Builds the dense (lat, lon) products on the Livneh 1/16° lattice under
``data/gridded/`` (Git LFS objects, excluded from the default fetch by
``.lfsconfig``; dataset card ``data/gridded/README.md``):

  grid_cells.csv                  the 13,786 cells of the forcing store
                                  (key, lat, lon, ilat, ilon, utm_zone)
  livneh_{precip_mm,tmax_c,tmin_c}_daily_1915-2018.nc
                                  daily forcing, one file per variable
                                  (GitHub LFS caps a file at 2 GB)
  precip_x10_corrections.csv      every corrected JJA cell-day
  alphaearth_2017.nc              AlphaEarth 2017 64-band cell-mean embedding
  alphaearth_2017-2025_mean.nc    optional: the 2017-2025 per-year cell means
                                  and their equal-weight mean
  SHA256SUMS, provenance.toml     checksums + how each product was made

Subcommands:
  grid       Scan the WGEN NonDetrend-Unsplit store → grid_cells.csv, the
             cell list every product is keyed to.  Cheap.  Refuses to
             replace an existing grid_cells.csv that disagrees with the
             store, and refuses to write an incomplete cell list into
             data/gridded.
  forcing    Store → one daily NetCDF per variable.  Applies the DWR x10
             precip rule (month in JJA and raw >= 150 mm → divide by 10)
             and swaps tmin/tmax where tmin > tmax.  Skips when all three
             NetCDFs exist and match SHA256SUMS and provenance.toml (pass
             --force to regenerate; an inconsistent set is an error, and
             Git LFS pointer stubs count as missing).  --rows A:B builds
             only a band of lattice rows for smoke tests (scratch --out-dir
             only).
  check-x10  Statewide gate of the x10 rule against DWR WGEN Product A,
             which applies the same correction upstream (its temperatures
             are detrended and ignored).
  aef        AlphaEarth per-cell mean embeddings from Earth Engine: 2017,
             and optionally every annual layer 2017-2025 for the
             multi-year mean.  Exactly one action: --dry-run (no EE),
             --run (the burn; banked partials per year, resumable;
             --years picks the years, default 2017, 'all' = 2017-2025),
             --status, --assemble (2017 partials → alphaearth_2017.nc),
             --assemble-mean (every year's partials → the 2017-2025 mean),
             --check N (independent 10 m re-reduction of N sampled cells of
             one --year) or --compare-parts DIR (offline comparison with
             an external reference bank in the same format; with
             --compare-mean NPZ also the multi-year mean).  --run and
             --check spend EE quota and need --project.
  verify     Offline QA of grid_cells.csv, the forcing NetCDFs and the
             AlphaEarth NetCDFs (the mean when present) against SHA256SUMS
             / provenance.toml; with --meteo-dir it also re-reads a sample
             of store cells and requires bitwise equality.

Every subcommand that writes or reads products takes --out-dir (default
data/gridded).  Point it at a scratch directory for trial builds so
data/gridded is never touched; the aef partials and grid list then default
to <out-dir>/aef_parts and <out-dir>/grid_cells.csv.

The exit status is non-zero whenever a check fails (check-x10, aef --check,
aef --compare-parts, verify), so the commands can gate a pipeline.

Environment: the neuralhyd conda env (netCDF4) for everything; aef --run and
--check additionally need an authenticated earthengine-api.  Nothing here
imports torch, and each subcommand imports only its own module (aef never
loads the forcing code).  A full forcing build or AlphaEarth burn runs for a
long time — launch it from your own terminal.

Usage:
  python scripts/prepare_gridded.py grid --meteo-dir /path/to/WGEN_NonDetrend_Unsplit_Statewide
  python scripts/prepare_gridded.py forcing --meteo-dir /path/to/store --rows 80:96 --out-dir /path/to/scratch
  python scripts/prepare_gridded.py forcing --meteo-dir /path/to/store --workers 8
  python scripts/prepare_gridded.py check-x10 --meteo-dir /path/to/store --product-a-dir /path/to/Product_A/1 --out-csv x10_mismatches.csv
  python scripts/prepare_gridded.py aef --dry-run
  python scripts/prepare_gridded.py aef --run --project <ee-project> --max-units 2     # smoke test
  python scripts/prepare_gridded.py aef --run --project <ee-project>                   # the burn, resumable
  python scripts/prepare_gridded.py aef --status
  python scripts/prepare_gridded.py aef --assemble
  python scripts/prepare_gridded.py aef --check 30 --project <ee-project>
  python scripts/prepare_gridded.py aef --compare-parts /path/to/reference/aef_parts [--compare-mean /path/to/reference_mean.npz]
  python scripts/prepare_gridded.py aef --dry-run --years all
  python scripts/prepare_gridded.py aef --run --project <ee-project> --years all      # 2018-2025 too
  python scripts/prepare_gridded.py aef --assemble-mean
  python scripts/prepare_gridded.py aef --check 30 --year 2021 --project <ee-project>
  python scripts/prepare_gridded.py verify --meteo-dir /path/to/store
  python scripts/prepare_gridded.py verify --full --skip-aef

Typical order for a fresh build:
  grid → forcing → check-x10 → verify --skip-aef
       → aef --dry-run → aef --run [→ aef --status] → aef --assemble
       → aef --check N → verify
  2017-2025 mean (optional, ~392 EECU-h on top of 2017's ~49):
       aef --dry-run --years all → aef --run --years all → aef --assemble-mean
       → aef --check N --year Y → verify
"""
from __future__ import annotations

import argparse
import sys
import traceback
from collections.abc import Callable
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.paths import (  # noqa: E402  (path set up above)
    GRIDDED_AEF_PARTS_DIR,
    GRIDDED_DIR,
    GRIDDED_GRID_CSV,
    GRIDDED_SHA256SUMS,
    PROJECT_ROOT,
)

_SEP = "=" * 70

_EPILOG = """\
typical order for a fresh build:
  grid -> forcing -> check-x10 -> verify --skip-aef
       -> aef --dry-run -> aef --run [-> aef --status] -> aef --assemble
       -> aef --check N -> verify
  2017-2025 mean (optional):
       aef --dry-run --years all -> aef --run --years all -> aef --assemble-mean
       -> aef --check N --year Y -> verify

Products land in data/gridded/ unless --out-dir is given; use a scratch
--out-dir for trial builds.  Run 'prepare_gridded.py COMMAND --help' for
the options of each command.
"""


def _banner(label: str) -> None:
    print(f"\n{_SEP}")
    print(label)
    print(_SEP)


# ---------------------------------------------------------------------------
# Argument helpers
# ---------------------------------------------------------------------------
def _rel(path: Path) -> str:
    """``path`` relative to the repo root for help texts (else as is)."""
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)


def _existing_dir(text: str) -> Path:
    p = Path(text)
    if not p.is_dir():
        raise argparse.ArgumentTypeError(f"not a directory: {text}")
    return p


def _positive_int(text: str) -> int:
    try:
        v = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected an integer, got {text!r}") from None
    if v < 1:
        raise argparse.ArgumentTypeError(f"must be >= 1, got {v}")
    return v


def _nonneg_int(text: str) -> int:
    try:
        v = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected an integer, got {text!r}") from None
    if v < 0:
        raise argparse.ArgumentTypeError(f"must be >= 0, got {v}")
    return v


def _year_arg(text: str) -> int | str:
    """One ``--years`` item: a calendar year or ``all`` (checked against
    aef.MEAN_YEARS in _cmd_aef, so building the parser imports no aef code)."""
    if text.strip().lower() == "all":
        return "all"
    try:
        return int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"expected a year (2017-2025) or 'all', got {text!r}") from None


def _row_range(text: str) -> tuple[int, int]:
    """``"A:B"`` → ``(A, B)``: lattice rows ``A <= ilat < B``."""
    try:
        a, b = (int(x) for x in text.split(":"))
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"expected A:B (two integers, e.g. 80:96), got {text!r}") from None
    if not 0 <= a < b:
        raise argparse.ArgumentTypeError(f"need 0 <= A < B, got {text!r}")
    return a, b


def _guarded(label: str, fn: Callable[[], bool]) -> bool:
    """Run one verifier; an exception counts as a failure, not a crash."""
    try:
        return bool(fn())
    except Exception as exc:                                 # noqa: BLE001
        traceback.print_exc()
        print(f"  FAIL  {label}: {type(exc).__name__}: {exc}")
        return False


# ---------------------------------------------------------------------------
# grid
# ---------------------------------------------------------------------------
def _print_grid_summary(cells, meteo_dir: Path) -> None:
    from src.data.gridded import lattice as L

    n_dense = L.NLAT * L.NLON
    lat, lon = L.lat_axis(), L.lon_axis()
    ilat, ilon = cells["ilat"], cells["ilon"]
    per_row = cells.groupby("ilat").size()
    zones = cells["utm_zone"].value_counts().sort_index()
    print(f"  Store      : {meteo_dir}")
    print(f"  Cells      : {len(cells):,}  (expected {L.N_CELLS:,}; "
          f"{len(cells) / n_dense:.1%} of the {n_dense:,}-cell dense grid)")
    print(f"  Grid       : {L.NLAT} rows x {L.NLON} cols, 1/16 deg, "
          f"lat {lat[0]:.5f}..{lat[-1]:.5f}, lon {lon[0]:.5f}..{lon[-1]:.5f}")
    print(f"  Rows used  : {ilat.nunique()} of {L.NLAT} (ilat {ilat.min()}..{ilat.max()}), "
          f"{per_row.min()}..{per_row.max()} cells per row")
    print(f"  Cols used  : {ilon.nunique()} of {L.NLON} (ilon {ilon.min()}..{ilon.max()})")
    print("  UTM zones  : " + ", ".join(f"{z}N {c:,}" for z, c in zones.items()))
    if (ilat.min(), ilat.max(), ilon.min(), ilon.max()) != (0, L.NLAT - 1, 0, L.NLON - 1):
        print("  WARNING: the cells do not span the full product grid "
              "(the grid is defined as the bounding box of the full store)")


def _cmd_grid(args: argparse.Namespace) -> int:
    from src.data.gridded import lattice as L
    from src.data.gridded import ncio

    out_dir: Path = args.out_dir
    target = out_dir / GRIDDED_GRID_CSV.name
    _banner(f"GRID: forcing store -> {target}")

    cells = L.scan_meteo_dir(args.meteo_dir)
    _print_grid_summary(cells, args.meteo_dir)

    if len(cells) != L.N_CELLS:
        msg = f"the store has {len(cells):,} cells, expected {L.N_CELLS:,}"
        if ncio.is_repo_gridded_dir(out_dir):
            print(f"\n  ERROR: {msg}; refusing to write an incomplete cell list "
                  f"into {_rel(GRIDDED_DIR)} (use a scratch --out-dir)")
            return 1
        print(f"\n  WARNING: {msg} (scratch --out-dir, continuing)")

    if target.exists():
        try:
            old = L.read_grid_csv(target)
        except (ValueError, KeyError) as exc:
            print(f"\n  ERROR: existing {target} is not a valid cell list: {exc}")
            return 1
        if not old.equals(cells):
            print(f"\n  ERROR: existing {target} ({len(old):,} cells) disagrees with "
                  f"the store scan ({len(cells):,} cells).")
            print("  Every gridded product and AEF partial is keyed to this list.  "
                  "If the store really changed, delete the file by hand and "
                  "rebuild every product.")
            return 1
        print(f"\n  {target} exists and matches the store scan, kept as is")
    else:
        L.write_grid_csv(cells, target)
        print(f"\n  Wrote {target}")

    digest = ncio.sha256_file(target)
    sums = ncio.update_sha256sums({target.name: digest}, out_dir / GRIDDED_SHA256SUMS.name)
    print(f"  sha256 {digest}")
    print(f"  Updated {sums}")
    return 0


# ---------------------------------------------------------------------------
# forcing / check-x10
# ---------------------------------------------------------------------------
def _cmd_forcing(args: argparse.Namespace) -> int:
    from src.data.gridded import lattice as L
    from src.data.gridded import ncio

    if args.rows is not None:
        a, b = args.rows
        if b > L.NLAT:
            args.parser.error(f"--rows {a}:{b}: B must be <= {L.NLAT}")
        if ncio.is_repo_gridded_dir(args.out_dir):
            args.parser.error(f"--rows builds a smoke-test subset and never writes "
                              f"{_rel(GRIDDED_DIR)}; pass a scratch --out-dir")
        label = f"rows {a}:{b}"
    else:
        label = "all rows"
    _banner(f"FORCING: WGEN store -> daily NetCDFs ({label}) in {args.out_dir}")

    from src.data.gridded import forcing
    forcing.build_forcing(args.meteo_dir, out_dir=args.out_dir, rows=args.rows,
                          workers=args.workers, force=args.force)
    return 0


def _cmd_check_x10(args: argparse.Namespace) -> int:
    scope = "all cells" if args.sample is None else f"rule cells + {args.sample} random"
    _banner(f"CHECK-X10: x10 precip rule vs DWR WGEN Product A ({scope})")

    from src.data.gridded import forcing
    ok = forcing.check_x10_product_a(args.meteo_dir, args.product_a_dir,
                                     sample=args.sample, seed=args.seed,
                                     out_csv=args.out_csv)
    return 0 if ok else 1        # the verdict line is printed by check_x10_product_a


# ---------------------------------------------------------------------------
# aef
# ---------------------------------------------------------------------------
def _cmd_aef(args: argparse.Namespace) -> int:
    out_dir: Path = args.out_dir
    grid_csv: Path = args.grid_csv or out_dir / GRIDDED_GRID_CSV.name
    parts_dir: Path = args.parts_dir or out_dir / GRIDDED_AEF_PARTS_DIR.name
    if (args.run or args.check is not None) and not args.project:
        args.parser.error("--run and --check spend Earth Engine quota and need "
                          "--project (an EE-registered cloud project id)")
    if args.years is not None and not (args.run or args.dry_run):
        args.parser.error("--years applies to --run and --dry-run (--status reports every "
                          "year, --assemble-mean needs all of them, --check takes --year)")
    if args.year is not None and args.check is None:
        args.parser.error("--year applies to --check (the year it re-reduces)")
    if args.compare_mean is not None and args.compare_parts is None:
        args.parser.error("--compare-mean applies to --compare-parts")
    paths = dict(grid_csv=grid_csv, parts_dir=parts_dir)

    from src.data.gridded import aef

    span = f"{aef.MEAN_YEARS[0]}-{aef.MEAN_YEARS[-1]}"
    years: tuple[int, ...] = (aef.YEAR,)
    if args.years is not None:
        if "all" in args.years:
            if len(args.years) > 1:
                args.parser.error("--years all takes no other year")
            years = aef.MEAN_YEARS
        else:
            bad = sorted(set(args.years) - set(aef.MEAN_YEARS))
            if bad:
                args.parser.error(f"--years {bad}: the annual layers are {span}")
            years = tuple(sorted(set(args.years)))
    check_year = aef.YEAR if args.year is None else args.year
    if check_year not in aef.MEAN_YEARS:
        args.parser.error(f"--year {check_year}: the annual layers are {span}")
    label = aef._years_label(years)

    if args.dry_run:
        _banner(f"AEF: dry run for {label} (no Earth Engine calls)")
        aef.dry_run(**paths, chunk=args.chunk, scale=args.scale, max_units=args.max_units,
                    years=years)
        return 0
    if args.status:
        _banner(f"AEF: status of {parts_dir}")
        aef.status(**paths)
        return 0
    if args.run:
        _banner(f"AEF: Earth Engine reduction of {label} -> {parts_dir} (resumable)")
        aef.run(args.project, **paths, chunk=args.chunk, scale=args.scale,
                workers=args.workers, max_units=args.max_units, years=years)
        return 0
    if args.assemble:
        _banner(f"AEF: assemble the {aef.YEAR} partials -> {out_dir}")
        aef.assemble(**paths, out_dir=out_dir, scale=args.scale)
        return 0
    if args.assemble_mean:
        _banner(f"AEF: assemble the {span} mean -> {out_dir}")
        aef.assemble_mean(**paths, out_dir=out_dir, scale=args.scale)
        return 0
    if args.check is not None:
        _banner(f"AEF: independent 10 m check of {args.check} cells ({check_year})")
        ok = aef.check(args.project, args.check, year=check_year, **paths, out_dir=out_dir,
                       scale=args.scale, seed=args.seed)
        print(f"\n  {'PASS' if ok else 'FAIL'}  AEF check {check_year} vs 10 m zone mosaic")
        return 0 if ok else 1
    if args.compare_parts is not None:
        _banner(f"AEF: offline comparison with the reference bank in {args.compare_parts}")
        ok = aef.compare_reference(args.compare_parts, **paths, out_dir=out_dir,
                                   scale=args.scale, ref_mean=args.compare_mean)
        print(f"\n  {'PASS' if ok else 'FAIL'}  AEF vs reference")
        return 0 if ok else 1
    raise AssertionError("argparse enforces exactly one aef action")


# ---------------------------------------------------------------------------
# verify
# ---------------------------------------------------------------------------
def _verify_grid(out_dir: Path) -> bool | None:
    """grid_cells.csv valid, complete and matching SHA256SUMS (None if absent)."""
    from src.data.gridded import lattice as L
    from src.data.gridded import ncio

    path = out_dir / GRIDDED_GRID_CSV.name
    if not path.exists():
        print(f"  SKIP  grid: {path} not present")
        return None
    if ncio.is_lfs_pointer(path):
        print(f"  FAIL  grid: {path} is a Git LFS pointer stub (install Git LFS and run "
              f"`git lfs pull`)")
        return False
    try:
        cells = L.read_grid_csv(path)
    except (ValueError, KeyError) as exc:
        print(f"  FAIL  grid: {path} is not a valid cell list: {exc}")
        return False
    ok = True
    if len(cells) == L.N_CELLS:
        print(f"  PASS  grid: {len(cells):,} cells, keys/lat/lon/indices consistent")
    else:
        print(f"  FAIL  grid: {len(cells):,} cells, expected {L.N_CELLS:,}")
        ok = False
    listed = ncio.read_sha256sums(out_dir / GRIDDED_SHA256SUMS.name).get(path.name)
    actual = ncio.sha256_file(path)
    if listed is None:
        print(f"  FAIL  grid: {path.name} not listed in {GRIDDED_SHA256SUMS.name}")
        ok = False
    elif listed != actual:
        print(f"  FAIL  grid: sha256 {actual} != {GRIDDED_SHA256SUMS.name} {listed}")
        ok = False
    else:
        print(f"  PASS  grid: sha256 matches {GRIDDED_SHA256SUMS.name}")
    return ok


def _cmd_verify(args: argparse.Namespace) -> int:
    out_dir: Path = args.out_dir
    results: dict[str, bool] = {}

    _banner(f"VERIFY: cell list in {out_dir}")
    grid_ok = _verify_grid(out_dir)
    if grid_ok is not None:
        results["grid"] = grid_ok

    if not args.skip_forcing:
        _banner("VERIFY: forcing NetCDFs"
                + (" (full: every day + content hashes)" if args.full else ""))

        def _forcing() -> bool:
            from src.data.gridded import forcing
            return forcing.verify_forcing(out_dir, meteo_dir=args.meteo_dir,
                                          sample=args.sample, seed=args.seed,
                                          full=args.full)
        results["forcing"] = _guarded("forcing", _forcing)

    if not args.skip_aef:
        _banner("VERIFY: AlphaEarth NetCDFs (2017; the 2017-2025 mean when present)")

        def _aef() -> bool:
            from src.data.gridded import aef
            return aef.verify_aef(out_dir)
        results["alphaearth"] = _guarded("alphaearth", _aef)

    _banner("VERIFY: summary")
    if not results:
        print("  FAIL  nothing was verified")
        return 1
    for name, ok in results.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    return 0 if all(results.values()) else 1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _add_out_dir(sp: argparse.ArgumentParser, what: str) -> None:
    sp.add_argument(
        "--out-dir", type=Path, default=GRIDDED_DIR, metavar="DIR",
        help=f"Product directory {what} (default: {_rel(GRIDDED_DIR)}).  Use a "
             "scratch directory for trial builds.",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="prepare_gridded.py",
        description="Statewide 1/16-degree gridded inputs: Livneh daily forcing "
                    "(WGEN NonDetrend-Unsplit store) + AlphaEarth static embeddings "
                    "(2017, and the optional 2017-2025 mean).",
        epilog=_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    # grid ------------------------------------------------------------------
    sp = sub.add_parser(
        "grid", help="scan the forcing store -> grid_cells.csv",
        description="Scan the WGEN NonDetrend-Unsplit store and write "
                    "grid_cells.csv (key, lat, lon, ilat, ilon, utm_zone), the cell "
                    "list every gridded product is keyed to, then record its sha256 "
                    "in SHA256SUMS.  An existing grid_cells.csv is kept only if it "
                    "matches the scan; a mismatch is an error.",
    )
    sp.add_argument("--meteo-dir", type=_existing_dir, required=True, metavar="DIR",
                    help="WGEN NonDetrend-Unsplit statewide store (data_<lat>_<lon> files).")
    _add_out_dir(sp, "to write grid_cells.csv and SHA256SUMS into")
    sp.set_defaults(func=_cmd_grid, parser=sp)

    # forcing ---------------------------------------------------------------
    sp = sub.add_parser(
        "forcing", help="forcing store -> one daily NetCDF per variable",
        description="Build livneh_{precip_mm,tmax_c,tmin_c}_daily_1915-2018.nc from "
                    "the store: x10 precip rule (month in 6-8 and raw >= 150 mm -> "
                    "divide by 10), tmin/tmax swap where tmin > tmax, float32 + "
                    "zlib, NaN outside the store cells.  Also writes "
                    "precip_x10_corrections.csv, SHA256SUMS and the 'forcing' "
                    "section of provenance.toml.  Skips when all three NetCDFs "
                    "exist and match SHA256SUMS and provenance.toml, unless "
                    "--force; a mismatching set is an error.",
    )
    sp.add_argument("--meteo-dir", type=_existing_dir, required=True, metavar="DIR",
                    help="WGEN NonDetrend-Unsplit statewide store (data_<lat>_<lon> files).")
    _add_out_dir(sp, "to write the NetCDFs into")
    sp.add_argument("--rows", type=_row_range, default=None, metavar="A:B",
                    help="Smoke test: only lattice rows A <= ilat < B (the files keep "
                         "the full grid, NaN elsewhere).  Needs a scratch --out-dir.")
    sp.add_argument("--workers", type=_positive_int, default=8, metavar="N",
                    help="Threads parsing store files within a 16-row band (default: 8).")
    sp.add_argument("--force", action="store_true", default=False,
                    help="Regenerate even when all three NetCDFs already exist (also "
                         "the way out of an interrupted or mixed build).")
    sp.set_defaults(func=_cmd_forcing, parser=sp)

    # check-x10 -------------------------------------------------------------
    sp = sub.add_parser(
        "check-x10", help="gate the x10 precip rule against DWR WGEN Product A",
        description="Compare the x10-corrected store precipitation with DWR WGEN "
                    "Product A (which applies the same correction upstream; its "
                    "detrended temperatures are ignored).  Requires |PA - expected| "
                    "<= 0.0051 mm everywhere and reports rule days PA did not "
                    "correct, PA corrections the rule missed and any other mismatch.  "
                    "Exit status 1 on any mismatch.",
    )
    sp.add_argument("--meteo-dir", type=_existing_dir, required=True, metavar="DIR",
                    help="WGEN NonDetrend-Unsplit statewide store (data_<lat>_<lon> files).")
    sp.add_argument("--product-a-dir", type=_existing_dir, required=True, metavar="DIR",
                    help="DWR WGEN Product A realisation directory (meteo_<key> files).")
    sp.add_argument("--sample", type=_nonneg_int, default=None, metavar="N",
                    help="Check every cell with a rule pair plus N random others "
                         "(default: every cell).")
    sp.add_argument("--seed", type=int, default=0, metavar="S",
                    help="Random seed for --sample (default: 0).")
    sp.add_argument("--out-csv", type=Path, default=None, metavar="PATH",
                    help="Write the mismatching cell-days to this CSV.")
    sp.set_defaults(func=_cmd_check_x10, parser=sp)

    # aef -------------------------------------------------------------------
    sp = sub.add_parser(
        "aef", help="AlphaEarth cell-mean embeddings: 2017 and the 2017-2025 mean (Earth Engine)",
        description="AlphaEarth Foundations satellite embedding "
                    "(GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL, bands A00..A63): mean pixel "
                    "vector over each cell rectangle per calendar year, reduced per UTM "
                    "zone on native tiles.  alphaearth_2017.nc holds 2017 alone; the "
                    "optional alphaearth_2017-2025_mean.nc holds every year 2017-2025 and "
                    "their equal-weight mean.  Partials are banked per year and work "
                    "unit, so --run is resumable.  Exactly one action is required.  "
                    "--run and --check spend Earth Engine quota (~49 EECU-h per year "
                    "at 15 m).",
    )
    act = sp.add_argument_group("action (exactly one)")
    one = act.add_mutually_exclusive_group(required=True)
    one.add_argument("--dry-run", action="store_true",
                     help="Work units, cells, units left and EECU estimate for --years "
                          "(no EE calls).")
    one.add_argument("--status", action="store_true",
                     help="What is banked so far, per year (no EE calls).")
    one.add_argument("--run", action="store_true",
                     help="Reduce every unbanked unit of --years on Earth Engine (needs "
                          "--project).")
    one.add_argument("--assemble", action="store_true",
                     help="Banked 2017 partials -> alphaearth_2017.nc + SHA256SUMS + "
                          "provenance [alphaearth].")
    one.add_argument("--assemble-mean", action="store_true",
                     help="Banked partials of every year 2017-2025 -> "
                          "alphaearth_2017-2025_mean.nc + SHA256SUMS + provenance "
                          "[alphaearth_mean].  Refuses unless every year is complete.")
    one.add_argument("--check", type=_positive_int, default=None, metavar="N",
                     help="Independent 10 m zone-mosaic re-reduction of N stratified "
                          "sample cells of --year, ~23.3 EECU-s per cell (needs --project).  "
                          "Gate per cell: max |d| < 2e-4 per band and ||m|-|m10|| < 1e-3 on "
                          "full cells, x 1/sqrt(valid_frac) on partial cells with "
                          "valid_frac >= 0.5; lower coverage is reported only.  Run "
                          "after assembling to gate the NetCDFs too (2017: "
                          "alphaearth_2017.nc; any year: its layer of the mean).")
    one.add_argument("--compare-parts", type=_existing_dir, default=None, metavar="DIR",
                     help="Offline: compare the banked partials of every year with an "
                          "external reference bank in the same format (DIR/<year>/*.npz "
                          "with keys, S, W, imgs, scale; never modified).  S/W and W must "
                          "agree to 1e-9 on the overlapping cells.  Add --compare-mean to "
                          "compare the assembled mean too.")
    sp.add_argument("--years", type=_year_arg, nargs="+", default=None, metavar="Y",
                    help="Calendar years for --run / --dry-run: any of 2017..2025, or 'all' "
                         "(default: 2017, the single-year product).")
    sp.add_argument("--year", type=int, default=None, metavar="Y",
                    help="Year whose partials --check re-reduces (default: 2017).")
    sp.add_argument("--compare-mean", type=Path, default=None, metavar="NPZ",
                    help="With --compare-parts: a reference mean npz (lat, lon, emb, "
                         "n_years, years, dataset_version, valid_frac, year_cos_min, "
                         "norm) that alphaearth_2017-2025_mean.nc must match within 1e-7 "
                         "on the overlapping cells (default: the mean is not compared).")
    sp.add_argument("--project", default=None, metavar="P",
                    help="Earth Engine cloud project id (required for --run and "
                         "--check; no default).")
    sp.add_argument("--out-dir", type=Path, default=GRIDDED_DIR, metavar="DIR",
                    help=f"Product directory for --assemble/--assemble-mean/--check/"
                         f"--compare-parts (default: {_rel(GRIDDED_DIR)}).  Use a scratch "
                         "directory for trial builds.")
    sp.add_argument("--parts-dir", type=Path, default=None, metavar="DIR",
                    help=f"Banked partials (default: <out-dir>/{GRIDDED_AEF_PARTS_DIR.name}).")
    sp.add_argument("--grid-csv", type=Path, default=None, metavar="PATH",
                    help=f"Cell list (default: <out-dir>/{GRIDDED_GRID_CSV.name}).")
    sp.add_argument("--workers", type=_positive_int, default=8, metavar="N",
                    help="Concurrent Earth Engine requests for --run (default: 8).")
    sp.add_argument("--chunk", type=_positive_int, default=150, metavar="N",
                    help="Max cells per request; pinned in run.json (default: 150).")
    sp.add_argument("--scale", type=float, default=15.0, choices=(10.0, 15.0),
                    metavar="{10,15}",
                    help="Reduction scale in metres; pinned in run.json.  15 reads the "
                         "full-resolution level on a nearest-neighbour lattice (<= 2e-4 "
                         "per band vs 10 m) at ~55%% of the EECU (default: 15).")
    sp.add_argument("--max-units", type=_positive_int, default=None, metavar="N",
                    help="Only the first N work units of the plan, per year (smoke test); "
                         "already-banked units among them are skipped, so repeating "
                         "the same N does no new work.")
    sp.add_argument("--seed", type=int, default=0, metavar="S",
                    help="Sampling seed for --check (default: 0).")
    sp.set_defaults(func=_cmd_aef, parser=sp)

    # verify ----------------------------------------------------------------
    sp = sub.add_parser(
        "verify", help="offline QA of every product + SHA256SUMS",
        description="Offline QA: grid_cells.csv, the three forcing NetCDFs, "
                    "alphaearth_2017.nc and, when present, alphaearth_2017-2025_mean.nc "
                    "(lattice coordinates, masks, NaN pattern, x10 table, swap counts, "
                    "the mean recomputed from its years, SHA256SUMS).  With --meteo-dir, "
                    "re-reads a sample of store cells and requires bitwise "
                    "equality.  Exit status 1 on any failure.",
    )
    _add_out_dir(sp, "to verify")
    sp.add_argument("--meteo-dir", type=_existing_dir, default=None, metavar="DIR",
                    help="Also re-read store cells and compare bit for bit with the NetCDFs.")
    sp.add_argument("--sample", type=_nonneg_int, default=64, metavar="N",
                    help="Random store cells to re-read with --meteo-dir, on top of "
                         "the x10-table cells (default: 64).")
    sp.add_argument("--seed", type=int, default=0, metavar="S",
                    help="Random seed for the sampled days and cells (default: 0).")
    sp.add_argument("--full", action="store_true", default=False,
                    help="Check every day and recompute the content hashes (slow).")
    sp.add_argument("--skip-forcing", action="store_true", default=False,
                    help="Do not verify the forcing NetCDFs.")
    sp.add_argument("--skip-aef", action="store_true", default=False,
                    help="Do not verify the AlphaEarth NetCDFs.")
    sp.set_defaults(func=_cmd_verify, parser=sp)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
