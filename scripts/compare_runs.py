#!/usr/bin/env python3
"""Compare held-out basin metrics across k-fold training runs.

Two modes:

* **Pairwise** (default) -- given two or more run names, print a tier-median
  table (one column per run) plus paired deltas vs. the first run, computed on
  the basins shared by every run.  With exactly two runs it also prints the
  per-basin delta distribution and the biggest movers.

* **Multi-seed** (``--seeds``) -- given one or more *base* run names, expand
  each into its seed replicates (``<base>`` and ``<base>__seed*``) and report
  the per-tier median together with the across-seed spread (min / median / max),
  so an apparent "win" can be judged against initialisation noise.

Runs are resolved under ``data/training/output/<name>/`` and read from each
``fold_*/basin_results.csv``.

Usage::

    python scripts/compare_runs.py single_lstm dual_lstm
    python scripts/compare_runs.py single_lstm dual_lstm --metrics nse kge
    python scripts/compare_runs.py single_lstm --seeds        # across-seed spread
    python scripts/compare_runs.py single_lstm moe_lstm --seeds
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.paths import TRAINING_OUTPUT_DIR

ALL_METRICS = ["nse", "kge", "fhv", "fehv", "flv"]


def load_run(run_dir: Path) -> pd.DataFrame:
    """Concatenate every fold's basin_results.csv for a run."""
    frames = []
    for fold_dir in sorted(run_dir.glob("fold_*")):
        csv = fold_dir / "basin_results.csv"
        if not csv.exists():
            continue
        df = pd.read_csv(csv)
        df["fold"] = int(fold_dir.name.split("_")[1])
        df["run"] = run_dir.name
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No fold_*/basin_results.csv found under {run_dir}")
    return pd.concat(frames, ignore_index=True)


def discover_seed_runs(root: Path, base: str) -> list[Path]:
    """Return the base run dir (if present) plus all <base>__seed* siblings."""
    runs = []
    if (root / base).is_dir():
        runs.append(root / base)
    runs.extend(sorted(p for p in root.glob(f"{base}__seed*") if p.is_dir()))
    if not runs:
        raise FileNotFoundError(f"No runs found for base {base!r} under {root}")
    return runs


# ---------------------------------------------------------------------------
# Pairwise comparison
# ---------------------------------------------------------------------------
def compare_pairwise(run_dirs: list[Path], metrics: list[str]) -> None:
    tables = {d.name: load_run(d).set_index("basin_id") for d in run_dirs}
    names = list(tables)

    shared = set.intersection(*(set(t.index) for t in tables.values()))
    print(f"Runs: {', '.join(names)}")
    for name, t in tables.items():
        print(f"  {name}: {len(t)} basin rows across {t['fold'].nunique()} folds")
    print(f"Shared basins (in all runs): {len(shared)}\n")
    if not shared:
        print("No shared basins -- cannot compare.")
        return

    tiers = sorted(tables[names[0]].loc[list(shared), "tier"].unique())
    base = names[0]

    print("=" * 78)
    print(f"TIER-MEDIAN METRICS on {len(shared)} shared basins  (delta vs. {base})")
    print("=" * 78)
    header = f"{'tier':>5} {'n':>5}  {'metric':<6}" + "".join(f"{n:>12}" for n in names)
    print(header)
    for tier in list(tiers) + ["ALL"]:
        if tier == "ALL":
            ids = list(shared)
        else:
            ids = [b for b in shared if tables[base].loc[b, "tier"] == tier]
        n = len(ids)
        for m in metrics:
            cells = []
            base_med = float(np.median(tables[base].loc[ids, m]))
            for name in names:
                med = float(np.median(tables[name].loc[ids, m]))
                if name == base:
                    cells.append(f"{med:>12.3f}")
                else:
                    d = med - base_med
                    cells.append(f"{med:>8.3f}{('+' if d >= 0 else '')}{d:>.3f}")
            tlabel = tier if tier == "ALL" else int(tier)
            print(f"{str(tlabel):>5} {n:>5}  {m:<6}" + "".join(cells))
        print()

    # Richer per-basin view only makes sense for an exact pair.
    if len(names) == 2:
        _pairwise_distribution(tables[names[0]], tables[names[1]], list(shared), metrics)


def _pairwise_distribution(a: pd.DataFrame, b: pd.DataFrame,
                           shared: list, metrics: list[str]) -> None:
    paired = a.loc[shared, ["tier"] + metrics].join(
        b.loc[shared, metrics], lsuffix="_a", rsuffix="_b"
    )
    for m in metrics:
        paired[f"d_{m}"] = paired[f"{m}_b"] - paired[f"{m}_a"]

    print("=" * 78)
    print(f"PER-BASIN DELTA DISTRIBUTION  ({b['run'].iloc[0]} - {a['run'].iloc[0]})")
    print("=" * 78)
    print(f"{'metric':<6} {'mean':>10} {'median':>10} {'std':>10} {'min':>10} {'max':>10} {'>0':>8}")
    for m in metrics:
        d = paired[f"d_{m}"].dropna()
        n_pos = int((d > 0).sum())
        print(f"{m:<6} {d.mean():>10.3f} {d.median():>10.3f} {d.std():>10.3f} "
              f"{d.min():>10.3f} {d.max():>10.3f} {n_pos:>4}/{len(d):<3}")
    print()

    print("=" * 78)
    print("TOP 10 BASINS BY |delta NSE|")
    print("=" * 78)
    top = paired.assign(abs_d=paired["d_nse"].abs()).nlargest(10, "abs_d")[
        ["tier", "nse_a", "nse_b", "d_nse", "kge_a", "kge_b", "d_kge"]
    ]
    top.index.name = "basin_id"
    print(top.round(3).to_string())
    print()


# ---------------------------------------------------------------------------
# Multi-seed spread
# ---------------------------------------------------------------------------
def compare_seeds(root: Path, bases: list[str], metrics: list[str]) -> None:
    for base in bases:
        seed_dirs = discover_seed_runs(root, base)
        print("=" * 78)
        print(f"MULTI-SEED SPREAD: {base}  ({len(seed_dirs)} replicate(s))")
        print(f"  {', '.join(d.name for d in seed_dirs)}")
        print("=" * 78)

        # Per-replicate tier-median table -> {tier: {metric: [vals across seeds]}}
        reps = [load_run(d) for d in seed_dirs]
        tiers = sorted(set().union(*(set(r["tier"].unique()) for r in reps)))

        print(f"{'tier':>5} {'metric':<6} {'median':>10} {'min':>10} {'max':>10} {'spread':>10}")
        for tier in list(tiers) + ["ALL"]:
            for m in metrics:
                vals = []
                for r in reps:
                    sub = r if tier == "ALL" else r[r["tier"] == tier]
                    if len(sub):
                        vals.append(float(sub[m].median()))
                if not vals:
                    continue
                vmed = float(np.median(vals))
                vmin, vmax = min(vals), max(vals)
                tlabel = tier if tier == "ALL" else int(tier)
                print(f"{str(tlabel):>5} {m:<6} {vmed:>10.3f} {vmin:>10.3f} "
                      f"{vmax:>10.3f} {vmax - vmin:>10.3f}")
            print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", help="Run name(s) under the output root.")
    parser.add_argument("--metrics", nargs="+", default=ALL_METRICS,
                        choices=ALL_METRICS, help="Metrics to report.")
    parser.add_argument("--root", type=Path, default=TRAINING_OUTPUT_DIR,
                        help="Output root containing run dirs.")
    parser.add_argument("--seeds", action="store_true",
                        help="Treat each run as a base and report across-seed spread.")
    args = parser.parse_args()

    if args.seeds:
        compare_seeds(args.root, args.runs, args.metrics)
        return

    if len(args.runs) < 2:
        parser.error("pairwise comparison needs >= 2 runs (or use --seeds)")
    run_dirs = [args.root / r for r in args.runs]
    missing = [d for d in run_dirs if not d.is_dir()]
    if missing:
        parser.error(f"run dir(s) not found: {', '.join(str(d) for d in missing)}")
    compare_pairwise(run_dirs, args.metrics)


if __name__ == "__main__":
    main()
