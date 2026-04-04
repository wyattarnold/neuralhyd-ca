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

Analysis (run after steps 1-8 are complete):
  --analysis map_watersheds       CA watershed map colored by regression tier
  --analysis tier_characteristics CDF/monthly-average figures by tier

Usage:
  python prepare_data.py                          # run all steps in order
  python prepare_data.py --step 1                 # single step
  python prepare_data.py --step 2 --meteo-dir /path/to/meteo
  python prepare_data.py --analysis map_watersheds
  python prepare_data.py --analysis tier_characteristics
  python prepare_data.py --analysis map_watersheds --analysis tier_characteristics

Typical order for a fresh run: 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8
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
        help="Step(s) to run (1-8). Omit to run all in order.",
    )
    parser.add_argument(
        "--meteo-dir", type=str, default=None,
        help="Path to gridded meteo files (required for step 2).",
    )
    parser.add_argument(
        "--analysis", type=str, action="append", default=None,
        choices=["map_watersheds", "tier_characteristics"],
        metavar="NAME",
        help=(
            "Analysis script(s) to run after the main pipeline. "
            "Choices: map_watersheds, tier_characteristics. "
            "Can be specified multiple times."
        ),
    )
    args = parser.parse_args()

    steps = args.step or ([] if args.analysis else [1, 2, 3, 4, 5, 6, 7, 8])

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
        run_climate(meteo_dir=args.meteo_dir)

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
        run_static_attr()

    if 5 in steps:
        print("\n" + "=" * 70)
        print("STEP 5: Develop climate statistics")
        print("=" * 70)
        from src.data.develop_static_clim import main as run_static_clim
        run_static_clim()

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


if __name__ == "__main__":
    main()
