#!/usr/bin/env python3
"""Optuna hyperparameter sweep (single-fold) for neuralhyd-ca LSTM models.

Runs fold-0 only with reduced epochs and SWA off for fast iteration.  The
study is persisted to SQLite so it survives interrupts and can be resumed.
Post-processing (summary, plots, best-config export) is handled separately by
``scripts/sweep_report.py`` so it can be re-run on any sweep folder at any
number of completed trials.

The default search space targets **model-agnostic generalization levers**
(LR, dropout, weight decay, log-loss blend, input noise) so it works for
single, dual, and MoE configs alike.  Edit ``_make_config`` to sweep other
parameters (e.g. fast_window / aux_loss_weight for dual, moe_* for MoE).

Usage::

    python scripts/sweep.py scripts/cfg_single_lstm.toml --n-trials 30
    python scripts/sweep.py scripts/cfg_single_lstm.toml --n-trials 5 --epochs 10

Then, at any point::

    python scripts/sweep_report.py data/training/output/<name>/sweep \\
        --base-config scripts/cfg_single_lstm.toml
"""
from __future__ import annotations

import argparse
import dataclasses
import gc
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import optuna
import torch

from src.lstm.config import Config, load_config
from src.lstm.dataset import (
    HydroDataset,
    compute_norm_stats,
    create_folds,
    load_all_data,
)
from src.lstm.evaluate import evaluate_fold
from src.lstm.model import build_model
from src.lstm.train import pick_device, train_model


def _make_config(trial: optuna.Trial, base: Config) -> Config:
    """Clone *base* and apply trial suggestions.

    Default space = model-agnostic generalization levers.  Everything not
    suggested here is inherited unchanged from the base TOML.
    """
    cfg = dataclasses.replace(base)

    # ── Search space (generalization levers; valid for any model_type) ──
    cfg.learning_rate = trial.suggest_float("learning_rate", 3e-4, 2e-3, log=True)
    cfg.dropout = trial.suggest_float("dropout", 0.0, 0.30, step=0.05)
    cfg.weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    cfg.log_loss_lambda = trial.suggest_float("log_loss_lambda", 0.0, 0.30, step=0.05)
    cfg.input_noise_std = trial.suggest_float("input_noise_std", 0.0, 0.10, step=0.02)

    # Sweep-specific overrides: fewer epochs, no SWA (set in main()).
    if hasattr(base, "_sweep_epochs"):
        cfg.num_epochs = base._sweep_epochs  # type: ignore[attr-defined]
    cfg.use_swa = False

    # Per-trial output dir
    cfg.output_dir = base.output_dir / f"trial_{trial.number}"
    return cfg


def objective(
    trial: optuna.Trial,
    *,
    base_config: Config,
    fold_data: tuple,
    device: torch.device,
) -> float:
    """Train fold 0 and return weighted floored-mean NSE (maximised)."""
    basin_ids, climate_data, flow_data, static_df, tier_map = fold_data

    config = _make_config(trial, base_config)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    folds = create_folds(basin_ids, tier_map, flow_data, config.n_folds, config.seed)
    train_ids, val_ids = folds[0]

    norm = compute_norm_stats(
        climate_data, flow_data, static_df, train_ids, basin_ids, config,
    )

    train_ds = HydroDataset(train_ids, climate_data, flow_data, static_df, config, norm)
    val_ds = HydroDataset(val_ids, climate_data, flow_data, static_df, config, norm)

    pin = device.type == "cuda"
    pw = config.num_workers > 0
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=config.batch_size, shuffle=True,
        num_workers=config.num_workers, pin_memory=pin, persistent_workers=pw,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=config.batch_size, shuffle=False,
        num_workers=config.num_workers, pin_memory=pin, persistent_workers=pw,
    )

    model = build_model(config).to(device)

    # Pruning callback: report val loss each epoch so Optuna can kill bad trials.
    def _prune_callback(epoch: int, val_loss: float) -> None:
        trial.report(val_loss, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

    def _free() -> None:
        gc.collect()
        if device.type == "mps":
            torch.mps.empty_cache()
        elif device.type == "cuda":
            torch.cuda.empty_cache()

    try:
        model, _history = train_model(
            model, train_loader, val_loader, config, 0, device,
            epoch_callback=_prune_callback,
            norm_stats=norm,
        )
    except optuna.TrialPruned:
        del model, train_ds, val_ds, train_loader, val_loader
        _free()
        raise

    df = evaluate_fold(model, val_ds, tier_map, config, 0, device)

    del model, train_ds, val_ds, train_loader, val_loader
    _free()

    # Primary objective: weighted floored-mean NSE across tiers.
    # Floor at -0.5 so catastrophic basins don't dominate, but the tail is
    # still visible to Optuna (unlike median, which ignores it).
    NSE_FLOOR = -0.5

    def _floored_mean(series) -> float:
        vals = series.dropna().clip(lower=NSE_FLOOR)
        return float(vals.mean()) if len(vals) else float("nan")

    t1_nse = _floored_mean(df[df["tier"] == 1]["nse"])
    t2_nse = _floored_mean(df[df["tier"] == 2]["nse"])
    t3_nse = _floored_mean(df[df["tier"] == 3]["nse"])

    if any(np.isnan(v) for v in (t1_nse, t2_nse, t3_nse)):
        return float("-inf")

    weighted_nse = 0.25 * t1_nse + 0.50 * t2_nse + 0.25 * t3_nse

    trial.set_user_attr("tier1_nse", t1_nse)
    trial.set_user_attr("tier2_nse", t2_nse)
    trial.set_user_attr("tier3_nse", t3_nse)
    trial.set_user_attr("weighted_nse", weighted_nse)
    trial.set_user_attr("overall_nse", float(df["nse"].median()))
    trial.set_user_attr("overall_kge", float(df["kge"].median()))
    for tier in (1, 2, 3):
        sub = df[df["tier"] == tier]
        trial.set_user_attr(f"tier{tier}_fhv", float(sub["fhv"].median()))
        trial.set_user_attr(f"tier{tier}_flv", float(sub["flv"].median()))

    return weighted_nse


def main() -> None:
    parser = argparse.ArgumentParser(description="Optuna hyperparameter sweep (fold-0).")
    parser.add_argument("config", help="Base TOML config file.")
    parser.add_argument("--n-trials", type=int, default=30,
                        help="Number of Optuna trials (default: 30).")
    parser.add_argument("--epochs", type=int, default=25,
                        help="Max epochs per trial (default: 25).")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    base_config = load_config(config_path)
    base_config._sweep_epochs = args.epochs  # type: ignore[attr-defined]
    sweep_dir = base_config.output_dir / "sweep"
    base_config.output_dir = sweep_dir
    sweep_dir.mkdir(parents=True, exist_ok=True)

    device = pick_device()
    print(f"Device: {device}")

    print("Loading data ...")
    fold_data = load_all_data(base_config)
    basin_ids = fold_data[0]
    tier_map = fold_data[4]
    n_per_tier = {t: sum(1 for v in tier_map.values() if v == t) for t in (1, 2, 3)}
    print(f"  {len(basin_ids)} basins  "
          f"(T1={n_per_tier[1]}, T2={n_per_tier[2]}, T3={n_per_tier[3]})")

    # Persistent SQLite study so it survives interrupts.
    storage = f"sqlite:///{sweep_dir / 'sweep.db'}"
    study_name = f"neuralhyd-{sweep_dir.parent.name}-sweep"
    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        storage=storage,
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=8,
        ),
    )

    study.optimize(
        lambda trial: objective(
            trial, base_config=base_config, fold_data=fold_data, device=device,
        ),
        n_trials=args.n_trials,
    )

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if completed:
        best = study.best_trial
        print(f"\n{'=' * 60}")
        print(f"BEST TRIAL: #{best.number}  (weighted NSE = {best.value:.4f})")
        for param, val in sorted(best.params.items()):
            if isinstance(val, float):
                print(f"  {param:<24s} = {val:.6f}")
            else:
                print(f"  {param:<24s} = {val}")
        print("=" * 60)
        print(f"\nRun full report:  python scripts/sweep_report.py {sweep_dir} "
              f"--base-config {config_path}")
    else:
        print("No completed trials.")


if __name__ == "__main__":
    main()
