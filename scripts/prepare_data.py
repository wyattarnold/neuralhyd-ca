"""Data pipeline — preparation and QA/QC.

Steps:
  0. Build training watershed catalog; optionally add CDEC FNF watersheds and canonical daily-flow CSVs
  1. Retrieve raw USGS flow data
  2. Develop area-weighted climate time series  (requires --meteo-dir)
  3. Verify climate data (monthly averages)
  4. Develop BasinATLAS static attributes (weighted averages)
  5. Develop climate statistics
  6. Clean raw USGS/CDEC flows (exceedance/precip filtering) → data/prepare/flow_precip_exceedance_filter/
  7. Comprehensive QA/QC report
  8. Flow vs precipitation QA + tier sorting → data/training/flow.zarr

Output routing:
  --target watersheds       → data/training/climate/watersheds.zarr + data/training/static/watersheds/
  --target huc8|huc10|huc12 → data/eval/climate/<level>.zarr + data/eval/static/<level>/  (full domain, inference only)

Geo intersect (prerequisite for steps 2 and 4 — one-time GIS operation; runs after step 0):
  --geo-intersect                              Run the GIS intersect
  --geo   static|meteo                         Attribute layer (BasinATLAS or VICGrids)
  --target watersheds|huc8|huc10|huc12         Target polygon layer (also used by steps 2, 4, 5)

Analysis (run after steps 0-8 are complete):
  --analysis map_watersheds       CA watershed map colored by regression tier
  --analysis tier_characteristics CDF/monthly-average figures by tier
  --analysis flow_extremes        Flow distribution analysis for extreme-loss calibration

Usage:
  python prepare_data.py --meteo-dir /path/to/meteo               # run steps 0-8
  python prepare_data.py --step 0 --include-cdec                 # combined watershed catalog + CDEC FNF canonical CSVs
  python prepare_data.py --step 0 --exclude-cdec                 # USGS-only watershed catalog
  python prepare_data.py --step 1                                # single step
  python prepare_data.py --step 2 --meteo-dir /path/to/meteo
  python prepare_data.py --diagnose-missing-grids --target huc12 --meteo-dir /path/to/meteo
  python prepare_data.py --geo-intersect --geo static --target watersheds
  python prepare_data.py --analysis map_watersheds
  python prepare_data.py --analysis tier_characteristics
  python prepare_data.py --analysis flow_extremes
  python prepare_data.py --analysis map_watersheds --analysis tier_characteristics

Typical order for a fresh run:
  0 → --geo-intersect → 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

_SEP = "=" * 70


def _banner(label: str) -> None:
    print(f"\n{_SEP}")
    print(label)
    print(_SEP)


def main() -> None:
    parser = argparse.ArgumentParser(description="Data pipeline")
    parser.add_argument(
        "--step", type=int, action="append", default=None, choices=range(9),
        help="Step(s) to run (0-8). Omit to run all of them.",
    )
    parser.add_argument(
        "--meteo-dir", type=str, default=None,
        help="Path to gridded meteo files (required for step 2).",
    )
    parser.add_argument(
        "--diagnose-missing-grids", action="store_true", default=False,
        help=(
            "Step 2 preflight: compare requested VIC grid cells against the "
            "meteo archive and report missing files without processing climate series."
        ),
    )
    parser.add_argument(
        "--analysis", type=str, action="append", default=None,
        choices=["map_watersheds", "tier_characteristics", "flow_extremes"],
        metavar="NAME",
        help=(
            "Analysis script(s) to run after the main pipeline. "
            "Choices: map_watersheds, tier_characteristics, flow_extremes. "
            "Can be specified multiple times."
        ),
    )
    parser.add_argument(
        "--geo-intersect", action="store_true", default=False,
        help="Run the GIS intersect table generator after step 0 (prerequisite for steps 2 and 4).",
    )
    parser.add_argument(
        "--geo", type=str, default="static",
        choices=["static", "meteo"],
        help="Attribute layer to intersect (static = BasinATLAS_v10_lev12, meteo = VICGrids_CAORNV_LatLong).",
    )
    parser.add_argument(
        "--target", type=str, default="watersheds",
        choices=["watersheds", "huc8", "huc10", "huc12"],
        help="Target polygon layer (used by steps 2/4/5 and --geo-intersect).",
    )
    parser.add_argument(
        "--force", action="store_true", default=False,
        help=(
            "Force regeneration of every basin in the climate zarr cube "
            "(step 2).  Default skips basins already in the cube."
        ),
    )
    cdec_group = parser.add_mutually_exclusive_group()
    cdec_group.add_argument(
        "--include-cdec", dest="include_cdec", action="store_true", default=True,
        help="Include CDEC FNF watersheds (step 0) and their staged flows (steps 6 and 8) (default).",
    )
    cdec_group.add_argument(
        "--exclude-cdec", dest="include_cdec", action="store_false",
        help="Build USGS-only watershed products (step 0) and skip CDEC flows in steps 6 and 8.",
    )
    args = parser.parse_args()

    # HUC targets always write to data/eval/ (full-domain, inference-only
    # outputs).  Only the `watersheds` target writes into data/training/.
    scope = "eval" if args.target in ("huc8", "huc10", "huc12") else "training"

    # Determine which steps to run.
    if args.step is not None:
        steps = args.step
    elif args.analysis or args.geo_intersect or args.diagnose_missing_grids:
        steps = []
    else:
        steps = list(range(9))  # 0–8

    # Validate up front so a default run does not download flows (step 1)
    # before failing on the missing meteo archive.
    if 2 in steps and args.meteo_dir is None:
        parser.error("--meteo-dir is required for step 2")

    if args.diagnose_missing_grids:
        _banner("DIAGNOSTIC: Missing meteo grids")
        if args.meteo_dir is None:
            parser.error("--meteo-dir is required for --diagnose-missing-grids")
        from src.data.develop_climate import diagnose_missing_grids as run_diagnose_missing_grids
        run_diagnose_missing_grids(
            meteo_dir=args.meteo_dir,
            target=args.target,
            scope=scope,
        )

    if 0 in steps:
        mode = "USGS + CDEC" if args.include_cdec else "USGS-only"
        _banner(f"STEP 0: Build {mode} training watersheds")
        from src.data.build_training_watersheds import main as run_build_training_watersheds
        run_build_training_watersheds(stage_flows=True, include_cdec=args.include_cdec)

    if args.geo_intersect:
        _banner(f"GEO INTERSECT: --geo {args.geo} --target {args.target}")
        from src.data.geo_intersect import main as run_geo_intersect
        run_geo_intersect(geo=args.geo, target=args.target, include_cdec=args.include_cdec)

    if 1 in steps:
        _banner("STEP 1: Retrieve raw USGS flow data")
        from src.data.retrieve_flows import main as run_retrieve
        run_retrieve()

    if 2 in steps:
        _banner("STEP 2: Develop area-weighted climate time series")
        from src.data.develop_climate import main as run_climate
        run_climate(meteo_dir=args.meteo_dir, target=args.target, force=args.force, scope=scope)

    if 3 in steps:
        _banner("STEP 3: Verify climate data (monthly averages)")
        from src.data.verify_climate import main as run_verify
        run_verify()

    if 4 in steps:
        _banner("STEP 4: Develop BasinATLAS static attributes")
        from src.data.develop_static_attr import main as run_static_attr
        run_static_attr(target=args.target, scope=scope)

    if 5 in steps:
        _banner("STEP 5: Develop climate statistics")
        from src.data.develop_static_clim import main as run_static_clim
        run_static_clim(target=args.target, scope=scope)

    if 6 in steps:
        _banner("STEP 6: Clean raw USGS flows (exceedance/precip filtering)")
        from src.data.clean_flows import main as run_clean
        run_clean(include_cdec=args.include_cdec)

    if 7 in steps:
        _banner("STEP 7: Comprehensive QA/QC")
        from src.data.run_qa_qc import main as run_qaqc
        run_qaqc()

    if 8 in steps:
        _banner("STEP 8: Flow vs precipitation QA + tier sorting")
        from src.data.flow_precip_qaqc import main as run_flow_precip
        run_flow_precip(include_cdec=args.include_cdec)

    if steps:
        _banner("Pipeline complete.")

    for name in args.analysis or []:
        if name == "map_watersheds":
            _banner("ANALYSIS: Map watersheds")
            from src.data.map_watersheds import main as run_map
            run_map()
        elif name == "tier_characteristics":
            _banner("ANALYSIS: Tier characteristics (CDF plots)")
            from src.data.plot_cdf_distributions import main as run_cdf
            run_cdf()
        elif name == "flow_extremes":
            _banner("ANALYSIS: Flow extremes (threshold calibration)")
            from src.data.analyse_flow_extremes import main as run_extremes
            run_extremes()


if __name__ == "__main__":
    main()
