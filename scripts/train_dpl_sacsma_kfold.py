#!/usr/bin/env python3
"""K-fold training entry point for the dPL+SAC-SMA model.

Mirrors ``scripts/train_kfold.py`` but uses the physics-based package
:mod:`src.dpl`.  Basins are the unit of splitting; HUC12s are the
hydrologic units on which SAC-SMA + Snow17 run.

Usage
-----
    python scripts/train_dpl_sacsma_kfold.py
    python scripts/train_dpl_sacsma_kfold.py scripts/config_dpl_sacsma_kfold.toml
"""

from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import pandas as pd
import torch

from src.dpl import (
    build_model,
    compute_norm_stats,
    create_folds,
    evaluate_fold,
    load_dpl_config,
    load_dpl_data,
    train_model,
)

_SCRIPTS_DIR = Path(__file__).resolve().parent
_DEFAULT_CONFIG = _SCRIPTS_DIR / "config_dpl_sacsma_kfold.toml"


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()


def _pick_device() -> torch.device:
    if torch.cuda.is_available():
        # cuDNN autotuner for conv1d routing / LSTM kernels.
        torch.backends.cudnn.benchmark = True
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    cfg_path = Path(sys.argv[1]) if len(sys.argv) > 1 else _DEFAULT_CONFIG
    config = load_dpl_config(cfg_path)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    log_file = config.output_dir / "train.log"
    log_handle = open(log_file, "a", encoding="utf-8")
    tee = _Tee(sys.stdout, log_handle)
    sys.stdout = tee
    sys.stderr = tee

    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] dPL+SAC-SMA k-fold run")
    print(f"  config:     {cfg_path}")
    print(f"  output_dir: {config.output_dir}")

    device = _pick_device()
    print(f"  device:     {device}")

    print("\nloading bundles ...")
    bundles = load_dpl_data(config)
    print(f"  basins loaded: {len(bundles)}")
    if not bundles:
        print("no basins found — aborting")
        return

    folds = create_folds(bundles, config)
    n_dynamic = len(config.dynamic_features)
    n_static = len(config.static_features)

    all_rows = []
    for k, (train_ids, val_ids) in enumerate(folds):
        print(f"\n=== fold {k} | train {len(train_ids)} | val {len(val_ids)} ===")
        fold_dir = config.output_dir / f"fold_{k}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        norm_stats = compute_norm_stats(train_ids, bundles, config)

        train_model(
            config,
            bundles,
            train_ids,
            val_ids,
            norm_stats,
            fold_dir,
            device,
            n_dynamic=n_dynamic,
            n_static=n_static,
        )

        # Reload best model for eval
        model = build_model(config, n_dynamic=n_dynamic, n_static=n_static).to(device)
        ckpt = torch.load(fold_dir / "best_model.pt", map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        results = evaluate_fold(
            model, bundles, val_ids, norm_stats, config, device,
            out_dir=fold_dir / "timeseries",
        )
        rows = [
            {"fold": k, "basin_id": r.basin_id, "tier": r.tier,
             "nse": r.nse, "kge": r.kge, "fhv": r.fhv, "fehv": r.fehv, "flv": r.flv}
            for r in results
        ]
        df = pd.DataFrame(rows)
        df.to_csv(fold_dir / "basin_results.csv", index=False)
        all_rows.extend(rows)

        print(f"  fold {k} tier medians:")
        for tier in sorted(df["tier"].unique()):
            sub = df[df["tier"] == tier]
            print(f"    T{tier}: n={len(sub):3d}  NSE={sub['nse'].median():+.3f}"
                  f"  KGE={sub['kge'].median():+.3f}  FHV={sub['fhv'].median():+.3f}")

    if all_rows:
        all_df = pd.DataFrame(all_rows)
        all_df.to_csv(config.output_dir / "all_fold_results.csv", index=False)
        print("\nAll folds complete.  Overall tier medians:")
        for tier in sorted(all_df["tier"].unique()):
            sub = all_df[all_df["tier"] == tier]
            print(f"  T{tier}: n={len(sub):3d}  NSE={sub['nse'].median():+.3f}"
                  f"  KGE={sub['kge'].median():+.3f}"
                  f"  FHV={sub['fhv'].median():+.3f}"
                  f"  FLV={sub['flv'].median():+.3f}")


if __name__ == "__main__":
    main()
