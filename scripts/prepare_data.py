"""Data pipeline — preparation and QA/QC.

Steps:
  1. Retrieve raw USGS flow data
  2. Develop area-weighted climate time series  (requires --meteo-dir)
  3. Verify climate data (monthly averages)
  4. Develop BasinATLAS static attributes (weighted averages)
  5. Develop climate statistics
  6. Clean raw USGS flows (exceedance/precip filtering) → figures/flow_precip_filter_exceedance/
  7. Comprehensive QA/QC report
  8. Flow vs precipitation QA + tier sorting → data/training/flow/
  9. (subbasin mode) Cross-reference QA-passing gauges with already-built
     subbasin static/climate data and copy the in-scope subset into the
     training tree.  Assumes steps 2/4/5 have already been run with
     `--target huc12` (or huc10) so the full-domain outputs exist under
     data/eval/{climate,static}/<level>/.

Output routing:
  --target watersheds       → data/training/{climate,static}/watersheds/
  --target huc8|huc10|huc12 → data/eval/{climate,static}/<level>/    (full domain)
                              Step 9 then materialises the in-scope subset
                              under data/training/{climate,static}/<level>/.

Geo intersect (prerequisite for step 4 — one-time GIS operation):
  --geo-intersect                              Run the GIS intersect
  --geo   static|meteo                         Attribute layer (BasinATLAS or VICGrids)
  --target watersheds|huc8|huc10               Target polygon layer (also used by steps 2, 4, 5)

Analysis (run after steps 1-8 are complete):
  --analysis map_watersheds       CA watershed map colored by regression tier
  --analysis tier_characteristics CDF/monthly-average figures by tier
  --analysis flow_extremes        Flow distribution analysis for extreme-loss calibration

Usage:
  python prepare_data.py                                          # run all steps
  python prepare_data.py --step 1                                 # single step
  python prepare_data.py --step 2 --meteo-dir /path/to/meteo
  python prepare_data.py --geo-intersect --geo static --target watersheds
  python prepare_data.py --analysis map_watersheds
  python prepare_data.py --analysis tier_characteristics
  python prepare_data.py --analysis flow_extremes
  python prepare_data.py --analysis map_watersheds --analysis tier_characteristics

Typical order for a fresh run:
  --geo-intersect → 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> None:
    parser = argparse.ArgumentParser(description="Data pipeline")
    parser.add_argument(
        "--step", type=int, action="append", default=None,
        help="Step(s) to run (1-8 default; 9 builds subbasin-mode data). "
             "Omit to run 1-8 in order.",
    )
    parser.add_argument(
        "--meteo-dir", type=str, default=None,
        help="Path to gridded meteo files (required for step 2).",
    )
    parser.add_argument(
        "--analysis", type=str, action="append", default=None,
        choices=["map_watersheds", "tier_characteristics", "flow_extremes"],
        metavar="NAME",
        help=(
            "Analysis script(s) to run after the main pipeline. "
            "Choices: map_watersheds, tier_characteristics. "
            "Can be specified multiple times."
        ),
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="For acquire_spatial: skip basins with complete cache files.",
    )
    parser.add_argument(
        "--basin", type=int, default=None,
        help="For acquire_spatial: process only this basin ID.",
    )
    parser.add_argument(
        "--geo-intersect", action="store_true", default=False,
        help="Run the GIS intersect table generator (prerequisite for step 4).",
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
        "--subbasin-level", type=str, default="huc12",
        choices=["huc10", "huc12"],
        help="WBD level used by step 9 (default: huc12).",
    )
    parser.add_argument(
        "--gauge-min-fraction-of-subbasin", type=float, default=0.70,
        help=(
            "Step 9a filter (default: 0.70). A gauge that overlaps exactly "
            "one subbasin is dropped only when its area is less than this "
            "fraction of the subbasin's area.  Gauges that span two or more "
            "subbasins are always kept."
        ),
    )
    parser.add_argument(
        "--force", action="store_true", default=False,
        help=(
            "Force regeneration of climate CSVs even when the output file "
            "already exists (step 2).  Default skips existing files."
        ),
    )
    args = parser.parse_args()

    # HUC targets always write to data/eval/ (full-domain outputs).  Only the
    # `watersheds` target writes directly into data/training/.  The training
    # subset for HUC runs is materialised by step 9 (cross-reference + copy).
    scope = "eval" if args.target in ("huc8", "huc10", "huc12") else "training"

    steps = args.step or ([] if (args.analysis or args.geo_intersect) else [1, 2, 3, 4, 5, 6, 7, 8])

    if 1 in steps:
        print("\n" + "=" * 70)
        print("STEP 1: Retrieve raw USGS flow data")
        print("=" * 70)
        from src.data.retrieve_flows import main as run_retrieve
        run_retrieve()

    if 2 in steps:
        print("\n" + "=" * 70)
        print("STEP 2: Develop area-weighted climate time series")
        print("=" * 70)
        if args.meteo_dir is None:
            print("ERROR: --meteo-dir is required for step 2")
            sys.exit(1)
        from src.data.develop_climate import main as run_climate
        run_climate(meteo_dir=args.meteo_dir, target=args.target, force=args.force, scope=scope)

    if 3 in steps:
        print("\n" + "=" * 70)
        print("STEP 3: Verify climate data (monthly averages)")
        print("=" * 70)
        from src.data.verify_climate import main as run_verify
        run_verify()

    if 4 in steps:
        print("\n" + "=" * 70)
        print("STEP 4: Develop BasinATLAS static attributes")
        print("=" * 70)
        from src.data.develop_static_attr import main as run_static_attr
        run_static_attr(target=args.target, scope=scope)

    if 5 in steps:
        print("\n" + "=" * 70)
        print("STEP 5: Develop climate statistics")
        print("=" * 70)
        from src.data.develop_static_clim import main as run_static_clim
        run_static_clim(target=args.target, scope=scope)

    if 6 in steps:
        print("\n" + "=" * 70)
        print("STEP 6: Clean raw USGS flows (exceedance/precip filtering)")
        print("=" * 70)
        from src.data.clean_flows import main as run_clean
        run_clean()

    if 7 in steps:
        print("\n" + "=" * 70)
        print("STEP 7: Comprehensive QA/QC")
        print("=" * 70)
        from src.data.run_qa_qc import main as run_qaqc
        run_qaqc()

    if 8 in steps:
        print("\n" + "=" * 70)
        print("STEP 8: Flow vs precipitation QA + tier sorting")
        print("=" * 70)
        from src.data.flow_precip_qaqc import main as run_flow_precip
        run_flow_precip()

    if 9 in steps:
        # Subbasin-mode data build: gauge × subbasin intersect, then copy
        # the in-scope subset of the (already-built) full-domain static +
        # climate outputs into the training tree.  Default level is HUC12
        # (pass `--subbasin-level huc10` for the coarser level).
        #
        # Prerequisite: steps 2/4/5 must already have been run with
        # `--target <level>` so data/eval/{climate,static}/<level>/ is
        # populated.  Step 9 itself does NOT (re)build climate or static
        # attributes — that is intentional, to avoid conflicting with the
        # earlier steps.
        level = args.subbasin_level
        level_upper = level.upper()
        print("\n" + "=" * 70)
        print(f"STEP 9: {level_upper} subbasin gauge cross-reference")
        print("=" * 70)
        from src.data.subbasin_gauge_intersect import main as run_subbasin_intersect

        print(f"  9a. gauge × {level_upper} intersect + filter rule "
              f"(min frac of subbasin = {args.gauge_min_fraction_of_subbasin:.0%})")
        run_subbasin_intersect(
            level=level,
            gauge_min_fraction_of_subbasin=args.gauge_min_fraction_of_subbasin,
        )

        # Copy the manifest (in-scope) subset into the training tree so
        # trainable configs only need a single root.  Full-domain copies
        # stay in data/eval/ for ungauged-basin simulation.
        print(f"\n  9b. Copy manifest subset → data/training/{{climate,static}}/{level}/")
        from src.data.copy_subbasin_to_training import (
            main as run_copy_subbasin_to_training,
        )
        run_copy_subbasin_to_training(level=level)

    if steps:
        print("\n" + "=" * 70)
        print("Pipeline complete.")
        print("=" * 70)

    analysis = args.analysis or []

    if "map_watersheds" in analysis:
        print("\n" + "=" * 70)
        print("ANALYSIS: Map watersheds")
        print("=" * 70)
        from src.data.map_watersheds import main as run_map
        run_map()

    if "tier_characteristics" in analysis:
        print("\n" + "=" * 70)
        print("ANALYSIS: Tier characteristics (CDF plots)")
        print("=" * 70)
        from src.data.plot_cdf_distributions import main as run_cdf
        run_cdf()

    if "flow_extremes" in analysis:
        print("\n" + "=" * 70)
        print("ANALYSIS: Flow extremes (threshold calibration)")
        print("=" * 70)
        from src.data.analyse_flow_extremes import main as run_extremes
        run_extremes()

    if args.geo_intersect:
        print("\n" + "=" * 70)
        print(f"GEO INTERSECT: --geo {args.geo} --target {args.target}")
        print("=" * 70)
        from src.data.geo_intersect import main as run_geo_intersect
        run_geo_intersect(geo=args.geo, target=args.target)


if __name__ == "__main__":
    main()
