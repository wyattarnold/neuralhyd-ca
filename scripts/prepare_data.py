"""Data pipeline — preparation and QA/QC.

Steps:
  0. Build training watershed catalog; optionally add CDEC FNF watersheds and canonical daily-flow CSVs
  1. Retrieve raw USGS flow data
  2. Develop area-weighted climate time series  (requires --meteo-dir)
  3. Verify climate data (monthly averages)
  4. Develop BasinATLAS static attributes (weighted averages)
  5. Develop climate statistics
  6. Clean raw USGS flows (exceedance/precip filtering) → figures/flow_precip_filter_exceedance/
  7. Comprehensive QA/QC report
  8. Flow vs precipitation QA + tier sorting → data/training/flow/
  9. Cross-reference QA-passing gauges with already-built HUC subbasin
      static/climate data and copy the in-scope subset into the training
      tree.  Assumes steps 2/4/5 have already been run with
     `--target huc12` (or huc10) so the full-domain outputs exist under
     data/eval/{climate,static}/<level>/.
     Auto-included in the default run when --target is huc10 or huc12.
 10. Build HUC12 → HUC10/HUC8 simulation manifests with target-relative
     area weights and routing-distance features.

Output routing:
  --target watersheds       → data/training/{climate,static}/watersheds/
  --target huc8|huc10|huc12 → data/eval/{climate,static}/<level>/    (full domain)
                              Step 9 then materialises the in-scope subset
                              under data/training/{climate,static}/<level>/.

Geo intersect (prerequisite for step 4 — one-time GIS operation):
  --geo-intersect                              Run the GIS intersect
  --geo   static|meteo                         Attribute layer (BasinATLAS or VICGrids)
  --target watersheds|huc8|huc10               Target polygon layer (also used by steps 2, 4, 5)

Analysis (run after steps 0-8 are complete):
  --analysis map_watersheds       CA watershed map colored by regression tier
  --analysis tier_characteristics CDF/monthly-average figures by tier
  --analysis flow_extremes        Flow distribution analysis for extreme-loss calibration

Usage:
  python prepare_data.py                                          # run steps 0-8 (+ step 9 if --target huc10/huc12)
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
  0 → --geo-intersect → 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8 [→ 9 if --target huc10/huc12]
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
        "--step", type=int, action="append", default=None,
        help=(
            "Step(s) to run. Omit to run the default set (0-8; step 9 is also "
            "auto-included when --target is huc10 or huc12). Step 10 always "
            "requires explicit --step 10."
        ),
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
        "--simulation-target-level", type=str, default="all",
        choices=["all", "huc10", "huc8"],
        help="Target level used by step 10 HUC12-to-target manifests (default: all).",
    )
    parser.add_argument(
        "--simulation-min-overlap-km2", type=float, default=0.1,
        help="Step 10: drop target-HUC12 overlay fragments smaller than this area (default: 0.1).",
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
    parser.add_argument(
        "--use-nldi", action="store_true", default=False,
        help=(
            "Step 9: use NLDI + NHDPlus WaterData for true along-network routing "
            "distances (requires pynhd).  Per-gauge flowlines are cached under "
            "data/prepare/geo_ops/nldi_cache/.  BasinATLAS is used as a gap-fill "
            "for any gauge/unit pairs not covered by NHD."
        ),
    )
    cdec_group = parser.add_mutually_exclusive_group()
    cdec_group.add_argument(
        "--include-cdec", dest="include_cdec", action="store_true", default=True,
        help="Step 0: include CDEC FNF watersheds and staged raw-flow files (default).",
    )
    cdec_group.add_argument(
        "--exclude-cdec", dest="include_cdec", action="store_false",
        help="Step 0: build USGS-only watershed products and remove staged CDEC raw-flow files.",
    )
    args = parser.parse_args()

    # HUC targets always write to data/eval/ (full-domain outputs).  Only the
    # `watersheds` target writes directly into data/training/.  The training
    # subset for HUC runs is materialised by step 9 (cross-reference + copy).
    scope = "eval" if args.target in ("huc8", "huc10", "huc12") else "training"

    # Determine which steps to run.
    if args.step is not None:
        steps = args.step
    elif args.analysis or args.geo_intersect or args.diagnose_missing_grids:
        steps = []
    else:
        steps = list(range(9))  # 0–8
        if args.target in ("huc10", "huc12"):
            steps.append(9)

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

    if 1 in steps:
        _banner("STEP 1: Retrieve raw USGS flow data")
        from src.data.retrieve_flows import main as run_retrieve
        run_retrieve()

    if 2 in steps:
        _banner("STEP 2: Develop area-weighted climate time series")
        if args.meteo_dir is None:
            parser.error("--meteo-dir is required for step 2")
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
        run_clean()

    if 7 in steps:
        _banner("STEP 7: Comprehensive QA/QC")
        from src.data.run_qa_qc import main as run_qaqc
        run_qaqc()

    if 8 in steps:
        _banner("STEP 8: Flow vs precipitation QA + tier sorting")
        from src.data.flow_precip_qaqc import main as run_flow_precip
        run_flow_precip(include_cdec=args.include_cdec)

    if 9 in steps:
        level = args.subbasin_level
        _banner(f"STEP 9: {level.upper()} subbasin gauge cross-reference")
        from src.data.subbasin_gauge_intersect import main as run_subbasin_intersect

        print(f"  9a. gauge × {level.upper()} intersect + filter rule "
              f"(min frac of subbasin = {args.gauge_min_fraction_of_subbasin:.0%})")
        run_subbasin_intersect(
            level=level,
            gauge_min_fraction_of_subbasin=args.gauge_min_fraction_of_subbasin,
            use_nldi=args.use_nldi,
        )

        print(f"\n  9b. Copy manifest subset → data/training/{{climate,static}}/{level}/")
        from src.data.copy_subbasin_to_training import main as run_copy_subbasin_to_training
        run_copy_subbasin_to_training(level=level)

    if 10 in steps:
        _banner("STEP 10: HUC12 → HUC10/HUC8 simulation manifests")
        from src.data.huc_target_manifest import main as run_huc_target_manifest
        run_huc_target_manifest(
            target_level=args.simulation_target_level,
            min_overlap_km2=args.simulation_min_overlap_km2,
        )

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

    if args.geo_intersect:
        _banner(f"GEO INTERSECT: --geo {args.geo} --target {args.target}")
        from src.data.geo_intersect import main as run_geo_intersect
        run_geo_intersect(geo=args.geo, target=args.target, include_cdec=args.include_cdec)


if __name__ == "__main__":
    main()
