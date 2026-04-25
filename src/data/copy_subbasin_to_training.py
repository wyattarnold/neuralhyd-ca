"""Copy the in-scope (manifest) subset of subbasin climate + static outputs
from ``data/eval/{climate,static}/<level>/`` into
``data/training/{climate,static}/<level>/``.

The eval tree holds full-domain outputs (used for ungauged-basin simulation);
the training tree holds the subset referenced by training gauges so trainable
configs only need a single root.

The set of in-scope subbasin IDs is read from
``data/prepare/geo_ops/<LEVEL>_In_Scope.csv``.
"""
from __future__ import annotations

import pandas as pd

from src.data.io import load_climate_dataframes, write_climate_zarr
from src.paths import (
    GEO_OPS_DIR,
    get_eval_target_paths,
    get_target_paths,
)


def _load_in_scope_ids(level: str) -> set[str]:
    in_scope_csv = GEO_OPS_DIR / f"{level.upper()}_In_Scope.csv"
    if not in_scope_csv.exists():
        raise FileNotFoundError(
            f"In-scope manifest not found: {in_scope_csv} "
            f"— run step 9a (subbasin_gauge_intersect) first."
        )
    df = pd.read_csv(in_scope_csv, dtype=str)
    return set(df[level].astype(str))


def _filter_climate_zarr(eval_zarr, train_zarr, ids: set[str]) -> int:
    """Read eval climate cube, keep in-scope ids, write training cube."""
    if not eval_zarr.exists():
        raise FileNotFoundError(f"Eval climate zarr not found: {eval_zarr}")
    train_zarr.parent.mkdir(parents=True, exist_ok=True)
    dfs = load_climate_dataframes(eval_zarr)
    kept = {bid: df for bid, df in dfs.items() if str(bid) in ids}
    write_climate_zarr(train_zarr, kept, basin_id_dtype="str", overwrite=True)
    return len(kept)


def _copy_static(eval_dir, train_dir, ids: set[str], level: str) -> None:
    """Copy filtered Physical_Attributes + Climate_Statistics CSVs."""
    train_dir.mkdir(parents=True, exist_ok=True)
    suffix = level.upper()
    for stem in (f"Physical_Attributes_{suffix}.csv",
                 f"Climate_Statistics_{suffix}.csv"):
        src = eval_dir / stem
        if not src.exists():
            print(f"  WARNING: {src} not found, skipping")
            continue
        df = pd.read_csv(src, dtype={"PourPtID": str})
        df = df[df["PourPtID"].astype(str).isin(ids)].copy()
        out = train_dir / stem
        df.to_csv(out, index=False)
        print(f"  wrote {out.name}: {len(df)} rows")


def main(level: str = "huc12") -> None:
    level = level.lower()
    ids = _load_in_scope_ids(level)
    print(f"  in-scope {level.upper()} ids: {len(ids)}")

    eval_paths  = get_eval_target_paths(level)
    train_paths = get_target_paths(level)

    n = _filter_climate_zarr(eval_paths["climate_zarr"], train_paths["climate_zarr"], ids)
    print(f"  filtered {n} basins → {train_paths['climate_zarr']}")

    _copy_static(eval_paths["static_dir"], train_paths["static_dir"], ids, level)


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else "huc12")
