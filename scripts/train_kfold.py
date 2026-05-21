#!/usr/bin/env python3
"""Unified entry point for k-fold spatial cross-validation.

For each fold ~20 % of basins per tier (T1 rainfall / T2 transitional /
T3 snow) are held out as unseen test watersheds, exercising ungauged-basin
generalisation.  Basins — not timesteps — are the unit of splitting.

Usage
-----
Pass a model-family TOML file to run a named experiment; the output
directory is derived automatically from the filename unless the TOML sets
``output_dir`` explicitly::

    python scripts/train_kfold.py scripts/cfg_dual_lstm.toml
    python scripts/train_kfold.py scripts/cfg_single_lstm.toml

Outputs (written to ``config.output_dir``):
    all_fold_results.csv           Tier-median NSE/KGE/FHV/FLV per fold
    fold_<n>/best_model.pt         Checkpoint: model weights + norm_stats
    fold_<n>/basin_results.csv     Per-basin metrics for the held-out set
    fold_<n>/timeseries/           Observed vs predicted CSV per basin
"""

from __future__ import annotations

import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import pandas as pd
import torch

from src.lstm.config import load_config
from src.lstm.dataset import (
    HydroDataset,
    compute_norm_stats,
    create_folds as create_lstm_folds,
    load_all_data,
)
from src.lstm.evaluate import evaluate_fold as evaluate_lstm_fold
from src.lstm.model import build_model as build_lstm_model
from src.lstm.train import pick_device, seed_everything, train_model as train_lstm_model


class _Tee:
    """Write to both a file and the original stream."""

    def __init__(self, stream, path: Path):
        self._stream = stream
        self._fh = open(path, "w", encoding="utf-8")

    def write(self, data: str) -> int:
        self._stream.write(data)
        self._fh.write(data)
        return len(data)

    def flush(self) -> None:
        self._stream.flush()
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Train and evaluate a neuralhyd-ca model with k-fold CV.")
    parser.add_argument(
        "config",
        help="Path to a TOML config file, e.g. scripts/cfg_dual_lstm.toml",
    )
    args = parser.parse_args()
    config_path = Path(args.config).resolve()

    config = load_config(config_path)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    # Preserve the exact config used for this run alongside its outputs.
    shutil.copy2(config_path, config.output_dir / config_path.name)

    # ---- tee stdout to log.txt ----
    tee = _Tee(sys.stdout, config.output_dir / "log.txt")
    _original_stdout = sys.stdout
    sys.stdout = tee

    try:
        _main_lstm(config, config_path)
    finally:
        sys.stdout = _original_stdout
        tee.close()


def _main_lstm(config, config_path: Path, *, device=None) -> None:  # noqa: D401
    print(f"Run started: {datetime.now().isoformat(timespec='seconds')}")
    print(f"Config: {config_path}")
    print(f"Output: {config.output_dir}")
    print(f"Model type: {config.model_type}")
    print()

    seed_everything(config.seed)

    # ---- device ----
    device = pick_device()
    print(f"Device: {device}")

    # ---- CUDA performance knobs (no-ops on MPS / CPU) ----
    if device.type == "cuda":
        if config.cudnn_benchmark:
            torch.backends.cudnn.benchmark = True
        if config.tf32:
            torch.set_float32_matmul_precision("high")
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    # ---- load data ----
    print("Loading data ...")
    basin_ids, climate_data, flow_data, static_df, tier_map = load_all_data(config)
    n_per_tier = {t: sum(1 for v in tier_map.values() if v == t) for t in (1, 2, 3)}
    print(
        f"  {len(basin_ids)} basins  "
        f"(T1={n_per_tier[1]}, T2={n_per_tier[2]}, T3={n_per_tier[3]})"
    )

    # ---- folds ----
    folds = create_lstm_folds(
        basin_ids, tier_map, flow_data,
        config.n_folds, config.seed,
    )

    all_results: list[pd.DataFrame] = []

    for fold_idx, (train_ids, val_ids) in enumerate(folds):
        print(f"\n{'=' * 60}")
        print(f"FOLD {fold_idx + 1}/{config.n_folds}  "
              f"(train {len(train_ids)}  / val {len(val_ids)} basins)")
        print("=" * 60)

        # normalisation from training basins only
        norm = compute_norm_stats(
            climate_data, flow_data, static_df,
            train_ids, basin_ids, config,
        )

        # datasets
        print(f"  Building training dataset  ({len(train_ids)} basins) ...")
        train_ds = HydroDataset(
            train_ids, climate_data, flow_data, static_df, config, norm,
        )
        print(f"    {len(train_ds):,} training samples  "
              f"(batch size {config.batch_size} -> {len(train_ds) // config.batch_size:,} batches/epoch)")

        print(f"  Building validation dataset  ({len(val_ids)} held-out basins) ...")
        val_ds = HydroDataset(
            val_ids, climate_data, flow_data, static_df, config, norm,
        )
        print(f"    {len(val_ds):,} validation samples")

        # loaders
        train_workers = config.num_workers
        train_pin = device.type == "cuda"
        train_pw = train_workers > 0
        # Val loader: tensors on CPU -- use workers + pinning when on CUDA.
        val_workers = config.num_workers
        val_pin = device.type == "cuda"
        val_pw = val_workers > 0
        train_loader = torch.utils.data.DataLoader(
            train_ds, batch_size=config.batch_size, shuffle=True,
            num_workers=train_workers, pin_memory=train_pin,
            persistent_workers=train_pw,
        )
        val_loader = torch.utils.data.DataLoader(
            val_ds, batch_size=config.batch_size, shuffle=False,
            num_workers=val_workers, pin_memory=val_pin,
            persistent_workers=val_pw,
        )

        # model
        model = build_lstm_model(config).to(device)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Model: {config.model_type} -- {n_params:,} parameters  (device: {device})")

        # train
        print("  Training ...")
        model, history = train_lstm_model(
            model, train_loader, val_loader, config, fold_idx, device,
            norm_stats=norm,
        )

        # save loss curve for this fold
        fold_dir = config.output_dir / f"fold_{fold_idx}"
        history_df = pd.DataFrame(history)
        history_df.to_csv(fold_dir / "loss_curve.csv", index=False)

        # evaluate held-out basins
        df = evaluate_lstm_fold(model, val_ds, tier_map, config, fold_idx, device)
        all_results.append(df)

    # ---- aggregate ----
    combined = pd.concat(all_results, ignore_index=True)

    print(f"\n{'=' * 60}")
    print("AGGREGATE ACROSS ALL FOLDS")
    print("=" * 60)
    print(f"  {'Tier':<8}{'N':<6}{'NSE med':>10}{'KGE med':>10}{'FHV med':>10}{'FLV med':>10}")
    print(f"  {'-' * 54}")
    for tier in sorted(combined["tier"].unique()):
        sub = combined[combined["tier"] == tier]
        print(
            f"  {tier:<8}{len(sub):<6}"
            f"{sub['nse'].median():>10.3f}{sub['kge'].median():>10.3f}"
            f"{sub['fhv'].median():>10.1f}{sub['flv'].median():>10.1f}"
        )
    print(
        f"  {'All':<8}{len(combined):<6}"
        f"{combined['nse'].median():>10.3f}{combined['kge'].median():>10.3f}"
        f"{combined['fhv'].median():>10.1f}{combined['flv'].median():>10.1f}"
    )

    combined.to_csv(config.output_dir / "all_fold_results.csv", index=False)
    print(f"\nResults saved to {config.output_dir}/")
    print(f"Run finished: {datetime.now().isoformat(timespec='seconds')}")


if __name__ == "__main__":
    main()
