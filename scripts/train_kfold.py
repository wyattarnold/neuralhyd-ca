#!/usr/bin/env python3
"""Entry point for k-fold stratified spatial cross-validation.

For each fold ~20 % of basins per tier (T1 rainfall / T2 transitional /
T3 snow) are held out as unseen test watersheds, exercising ungauged-basin
generalisation.  Basins — not timesteps — are the unit of splitting.

Usage
-----
Run with the default config::

    python scripts/train_kfold.py

Pass an alternate TOML file to run a named experiment; the output
directory is derived automatically from the filename::

    python scripts/train_kfold.py scripts/config_dual_lstm_kfold.toml

Outputs (written to ``config.output_dir``):
    all_fold_results.csv           Tier-median NSE/KGE/FHV/FLV per fold
    fold_<n>/best_model.pt         Checkpoint: model weights + norm_stats
    fold_<n>/basin_results.csv     Per-basin metrics for the held-out set
    fold_<n>/timeseries/           Observed vs predicted CSV per basin
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

from src.lstm.config import load_config
from src.lstm.dataset import (
    HydroDataset,
    compute_norm_stats,
    create_folds,
    load_all_data,
)
from src.lstm.evaluate import evaluate_fold
from src.lstm.model import build_model
from src.lstm.train import pick_device, train_model
from src.lstm.subbasin_dataset import (
    SubbasinHydroDataset,
    compute_subbasin_norm_stats,
    create_subbasin_folds,
    load_subbasin_data,
    subbasin_collate,
)
from src.lstm.subbasin_train import evaluate_fold_subbasin

_SCRIPTS_DIR = Path(__file__).resolve().parent


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
    parser = argparse.ArgumentParser(description="Train and evaluate LSTM hydrology model.")
    parser.add_argument(
        "config", nargs="?", default=str(_SCRIPTS_DIR / "config.toml"),
        help="Path to TOML config file (default: scripts/config.toml)",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    # ---- tee stdout to log.txt ----
    tee = _Tee(sys.stdout, config.output_dir / "log.txt")
    _original_stdout = sys.stdout
    sys.stdout = tee

    try:
        _main_body(config, args, device=None)
    finally:
        sys.stdout = _original_stdout
        tee.close()


def _main_body(config, args, *, device=None) -> None:  # noqa: D401
    print(f"Run started: {datetime.now().isoformat(timespec='seconds')}")
    print(f"Config: {args.config}")
    print(f"Output: {config.output_dir}")
    print(f"Model type: {config.model_type}")
    print()

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
    print("Loading data …")
    if getattr(config, "spatial_mode", "gauge") == "subbasin":
        _run_subbasin(config, device)
        return

    basin_ids, climate_data, flow_data, static_df, tier_map = load_all_data(config)
    n_per_tier = {t: sum(1 for v in tier_map.values() if v == t) for t in (1, 2, 3)}
    print(
        f"  {len(basin_ids)} basins  "
        f"(T1={n_per_tier[1]}, T2={n_per_tier[2]}, T3={n_per_tier[3]})"
    )

    # ---- folds ----
    folds = create_folds(
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
        # When pin_dataset_to_device is enabled on a non-CPU device, only
        # the *training* dataset is pre-moved to the GPU.  Validation runs
        # once per epoch and tolerates host→device copies fine, so leaving
        # the val dataset on CPU avoids doubling the dataset's VRAM
        # footprint (which on Windows triggers the WDDM "shared GPU
        # memory" sysmem fallback and tanks throughput).
        train_ds_device = device if (config.pin_dataset_to_device and device.type != "cpu") else None
        print(f"  Building training dataset  ({len(train_ids)} basins) …")
        train_ds = HydroDataset(
            train_ids, climate_data, flow_data, static_df, config, norm,
            device=train_ds_device,
        )
        print(f"    {len(train_ds):,} training samples  "
              f"(batch size {config.batch_size} → {len(train_ds) // config.batch_size:,} batches/epoch)")

        print(f"  Building validation dataset  ({len(val_ids)} held-out basins) …")
        val_ds = HydroDataset(
            val_ids, climate_data, flow_data, static_df, config, norm,
            device=None,
        )
        print(f"    {len(val_ds):,} validation samples")

        # loaders
        # Train loader: if train tensors are on device, workers + pinning
        # must be disabled (CUDA/MPS tensors can't cross worker boundaries).
        if train_ds_device is not None:
            train_workers, train_pin, train_pw = 0, False, False
        else:
            train_workers = config.num_workers
            train_pin = device.type == "cuda"
            train_pw = train_workers > 0
        # Val loader: tensors on CPU — use workers + pinning when on CUDA.
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
        model = build_model(config).to(device)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Model: {config.model_type} — {n_params:,} parameters  (device: {device})")

        # train
        print("  Training …")
        model, history = train_model(
            model, train_loader, val_loader, config, fold_idx, device,
            norm_stats=norm,
        )

        # save loss curve for this fold
        fold_dir = config.output_dir / f"fold_{fold_idx}"
        history_df = pd.DataFrame(history)
        history_df.to_csv(fold_dir / "loss_curve.csv", index=False)

        # evaluate held-out basins
        df = evaluate_fold(model, val_ds, tier_map, config, fold_idx, device)
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


def _run_subbasin(config, device) -> None:
    """Subbasin-mode training path (subbasin inputs, gauge-level loss)."""
    bundle = load_subbasin_data(config)
    gauge_ids = bundle["gauge_ids"]
    tier_map = bundle["gauge_tier"]
    n_per_tier = {t: sum(1 for v in tier_map.values() if v == t) for t in (1, 2, 3)}
    print(
        f"  subbasin mode: {len(gauge_ids)} gauges  "
        f"(T1={n_per_tier[1]}, T2={n_per_tier[2]}, T3={n_per_tier[3]})  "
        f"| max {config.subbasin_level.upper()}s per gauge: {bundle['n_max_subbasins']}  "
        f"| {config.subbasin_level.upper()} input tensors: {len(bundle['subbasin_climate'])}"
    )

    folds = create_subbasin_folds(
        gauge_ids, tier_map, bundle["gauge_flow"],
        bundle["gauge_subbasins"],   # gauge → list[subbasin_id] (nesting detection)
        config.n_folds, config.seed,
    )

    all_results: list[pd.DataFrame] = []

    for fold_idx, (train_ids, val_ids) in enumerate(folds):
        print(f"\n{'=' * 60}")
        print(f"FOLD {fold_idx + 1}/{config.n_folds}  "
              f"(train {len(train_ids)}  / val {len(val_ids)} gauges)")
        print("=" * 60)

        norm = compute_subbasin_norm_stats(train_ids, bundle, config)

        # Pin training tensors to GPU; keep validation on CPU to avoid
        # doubling the dataset's VRAM footprint (Windows WDDM otherwise
        # spills into shared system memory and stalls on PCIe).
        train_ds_device = device if (config.pin_dataset_to_device and device.type != "cpu") else None
        print(f"  Building training dataset  ({len(train_ids)} gauges) …")
        train_ds = SubbasinHydroDataset(
            train_ids, bundle, norm, config, device=train_ds_device,
        )
        print(f"    {len(train_ds):,} training samples  "
              f"(batch size {config.batch_size} → {len(train_ds) // max(config.batch_size, 1):,} batches/epoch)")

        print(f"  Building validation dataset  ({len(val_ids)} gauges) …")
        val_ds = SubbasinHydroDataset(
            val_ids, bundle, norm, config, device=None,
        )
        print(f"    {len(val_ds):,} validation samples")

        # Train loader: device-resident tensors → no workers / no pinning.
        if train_ds_device is not None:
            train_workers, train_pin, train_pw = 0, False, False
        else:
            train_workers = config.num_workers
            train_pin = device.type == "cuda"
            train_pw = train_workers > 0
        # Val loader: tensors on CPU — use workers + pinning when on CUDA.
        val_workers = config.num_workers
        val_pin = device.type == "cuda"
        val_pw = val_workers > 0
        train_loader = torch.utils.data.DataLoader(
            train_ds, batch_size=config.batch_size, shuffle=True,
            num_workers=train_workers, pin_memory=train_pin, persistent_workers=train_pw,
            collate_fn=subbasin_collate,
        )
        val_loader = torch.utils.data.DataLoader(
            val_ds, batch_size=config.batch_size, shuffle=False,
            num_workers=val_workers, pin_memory=val_pin, persistent_workers=val_pw,
            collate_fn=subbasin_collate,
        )

        model = build_model(config).to(device)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Model: {config.model_type} — {n_params:,} parameters  (device: {device})")

        print("  Training …")
        model, history = train_model(
            model, train_loader, val_loader, config, fold_idx, device,
            norm_stats=norm,
        )

        fold_dir = config.output_dir / f"fold_{fold_idx}"
        history_df = pd.DataFrame(history)
        history_df.to_csv(fold_dir / "loss_curve.csv", index=False)

        df = evaluate_fold_subbasin(model, val_ds, tier_map, config, fold_idx, device)
        all_results.append(df)

    combined = pd.concat(all_results, ignore_index=True)
    print(f"\n{'=' * 60}")
    print("AGGREGATE ACROSS ALL FOLDS  [subbasin]")
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
