"""Compare basin-level results between the original dual_lstm run and dual_lstm_v1.

Reads all fold basin_results.csv files from each run, computes paired
metric deltas on the intersection of basin_ids, and prints summary
statistics.
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path("data/training/output")
RUN_A = ROOT / "dual_lstm"          # original 2026-04-20 run
RUN_B = ROOT / "dual_lstm_v1"       # post-cleanup re-run


def load_all(run_dir: Path) -> pd.DataFrame:
    frames = []
    for fold_dir in sorted(run_dir.glob("fold_*")):
        df = pd.read_csv(fold_dir / "basin_results.csv")
        df["fold"] = int(fold_dir.name.split("_")[1])
        df["run"] = run_dir.name
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


a = load_all(RUN_A)
b = load_all(RUN_B)

print(f"Original ({RUN_A.name}): {len(a)} basin rows across {a['fold'].nunique()} folds")
print(f"v1       ({RUN_B.name}): {len(b)} basin rows across {b['fold'].nunique()} folds")
print()

# Universe and overlap
ids_a = set(a["basin_id"])
ids_b = set(b["basin_id"])
shared = ids_a & ids_b
only_a = ids_a - ids_b
only_b = ids_b - ids_a
print(f"Basin set sizes: original={len(ids_a)}, v1={len(ids_b)}")
print(f"  shared:   {len(shared)}")
print(f"  only in original (dropped from v1): {len(only_a)}")
print(f"  only in v1 (new CDEC FNF basins):   {len(only_b)}")
print()

# Build paired delta table on shared basins
metrics = ["nse", "kge", "fhv", "fehv", "flv"]
a_s = a[a["basin_id"].isin(shared)].set_index("basin_id")[["tier"] + metrics]
b_s = b[b["basin_id"].isin(shared)].set_index("basin_id")[metrics]
paired = a_s.join(b_s, lsuffix="_a", rsuffix="_b")

for m in metrics:
    paired[f"d_{m}"] = paired[f"{m}_b"] - paired[f"{m}_a"]


# ---- Tier-median table: original vs v1, computed on the SHARED basins
print("=" * 76)
print("TIER-MEDIAN METRICS  (computed only on shared basins, paired comparison)")
print("=" * 76)
print(f"{'tier':>6} {'n':>5}  {'metric':<6} {'original':>10} {'v1':>10} {'delta':>10}")
for tier in sorted(paired["tier"].unique()):
    sub = paired[paired["tier"] == tier]
    n = len(sub)
    for m in metrics:
        med_a = sub[f"{m}_a"].median()
        med_b = sub[f"{m}_b"].median()
        d = med_b - med_a
        sign = "+" if d >= 0 else ""
        print(f"{int(tier):>6} {n:>5}  {m:<6} {med_a:>10.3f} {med_b:>10.3f} {sign}{d:>9.3f}")
    print()

# Overall median across all shared basins
sub = paired
n = len(sub)
print(f"{'ALL':>6} {n:>5}")
for m in metrics:
    med_a = sub[f"{m}_a"].median()
    med_b = sub[f"{m}_b"].median()
    d = med_b - med_a
    sign = "+" if d >= 0 else ""
    print(f"{'':>6} {'':>5}  {m:<6} {med_a:>10.3f} {med_b:>10.3f} {sign}{d:>9.3f}")
print()


# ---- Per-basin delta distribution
print("=" * 76)
print("DELTA DISTRIBUTION  (per-basin v1 - original, on shared basins)")
print("=" * 76)
print(f"{'metric':<6} {'mean':>10} {'median':>10} {'std':>10} {'min':>10} {'max':>10} {'>0':>6}")
for m in metrics:
    d = paired[f"d_{m}"].dropna()
    n_pos = int((d > 0).sum())
    print(f"{m:<6} {d.mean():>10.3f} {d.median():>10.3f} {d.std():>10.3f} {d.min():>10.3f} {d.max():>10.3f} {n_pos:>4}/{len(d):<3}")
print()

# ---- Biggest individual movers (by |d_nse|)
print("=" * 76)
print("TOP 10 BASINS BY |delta NSE|")
print("=" * 76)
top = paired.assign(abs_d=paired["d_nse"].abs()).nlargest(10, "abs_d")[
    ["tier", "nse_a", "nse_b", "d_nse", "kge_a", "kge_b", "d_kge"]
]
top.index.name = "basin_id"
print(top.round(3).to_string())
print()

# ---- New basins in v1 (their performance — no comparison possible)
if only_b:
    print("=" * 76)
    print(f"NEW BASINS IN V1 (n={len(only_b)}) — performance summary, no baseline")
    print("=" * 76)
    new_b = b[b["basin_id"].isin(only_b)]
    for tier in sorted(new_b["tier"].unique()):
        sub = new_b[new_b["tier"] == tier]
        print(f"  tier {int(tier)}  n={len(sub):>3}  "
              f"NSE med {sub['nse'].median():>6.3f}  KGE med {sub['kge'].median():>6.3f}  "
              f"FHV med {sub['fhv'].median():>7.2f}  FLV med {sub['flv'].median():>7.2f}")
