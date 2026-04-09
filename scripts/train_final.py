#!/usr/bin/env python3
"""Train a final model on all 216 watersheds for deployment.

Unlike ``train_kfold.py`` no basins are withheld; all available data are
used for training so the checkpoint captures the full distribution of
California watershed behaviour.  The saved checkpoint bundles model
weights and normalisation statistics so it can be loaded for inference
on new basins without the original training data.

Usage
-----
    python scripts/train_final.py [config.toml]

Output (written to ``config.output_dir``):
    best_model.pt    Checkpoint: ``model_state_dict`` + ``norm_stats``
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import pandas as pd
import torch

from src.lstm.config import load_config
from src.lstm.dataset import (
    HydroDataset,
    compute_norm_stats,
    load_all_data,
)
from src.lstm.evaluate import evaluate_fold
from src.lstm.model import build_model
from src.lstm.train import train_model

_SCRIPTS_DIR = Path(__file__).resolve().parent


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Train final model on all basins.")
    parser.add_argument(
        "config", nargs="?", default=str(_SCRIPTS_DIR / "config.toml"),
        help="Path to TOML config file (default: scripts/config.toml)",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    config.output_dir = config.output_dir / "dual_lstm_final"
    config.output_dir.mkdir(parents=True, exist_ok=True)

    # ---- device ----
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # ---- load data ----
    print("Loading data …")
    basin_ids, climate_data, flow_data, static_df, tier_map = load_all_data(config)
    n_per_tier = {t: sum(1 for v in tier_map.values() if v == t) for t in (1, 2, 3)}
    print(
        f"  {len(basin_ids)} basins  "
        f"(T1={n_per_tier[1]}, T2={n_per_tier[2]}, T3={n_per_tier[3]})"
    )

    # ---- normalisation from ALL basins ----
    norm = compute_norm_stats(
        climate_data, flow_data, static_df,
        basin_ids, basin_ids, config,
    )

    # ---- single dataset (train = val = all basins) ----
    print(f"  Building dataset  ({len(basin_ids)} basins) …")
    ds = HydroDataset(basin_ids, climate_data, flow_data, static_df, config, norm)
    print(
        f"    {len(ds):,} samples  "
        f"(batch size {config.batch_size} → {len(ds) // config.batch_size:,} batches/epoch)"
    )

    pin = device.type == "cuda"
    pw = config.num_workers > 0
    train_loader = torch.utils.data.DataLoader(
        ds, batch_size=config.batch_size, shuffle=True,
        num_workers=config.num_workers, pin_memory=pin,
        persistent_workers=pw,
    )
    val_loader = torch.utils.data.DataLoader(
        ds, batch_size=config.batch_size, shuffle=False,
        num_workers=config.num_workers, pin_memory=pin,
        persistent_workers=pw,
    )

    # ---- model ----
    model = build_model(config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: {config.model_type} — {n_params:,} parameters  (device: {device})")

    # ---- train ----
    print("  Training …")
    model, history = train_model(
        model, train_loader, val_loader, config, 0, device,
        norm_stats=norm,
    )

    # ---- evaluate all basins ----
    df = evaluate_fold(model, ds, tier_map, config, 0, device)
    df.to_csv(config.output_dir / "all_basin_results.csv", index=False)
    print(f"\nResults saved to {config.output_dir}/")


if __name__ == "__main__":
    main()
