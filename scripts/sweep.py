#!/usr/bin/env python3
"""Optuna hyperparameter sweep for the dual-pathway model.

Single-objective search on fold 0: maximise composite score
  score = tier-weighted KGE − fhv_weight × (tier-weighted |FHV| / 100)
Tier weights: T1=0.25, T2=0.50, T3=0.25.
MedianPruner kills unpromising trials early.

8 search dimensions (5 continuous, 3 categorical; 1 conditional per arm):
  output_type (cat: deterministic | cmal)
  fast_window (int, log), dropout (float), extreme_weight_max (float),
  extreme_threshold (float), learning_rate (float, log),
  static_feature_set (cat)
  ── CMAL arm: cmal_n_components (cat [3, 5])
  ── Deterministic arm: log_loss_lambda (float [0, 0.15])

Usage (from scripts/):
    KMP_DUPLICATE_LIB_OK=TRUE python sweep.py config_dual_lstm_cmal_kfold.toml --n-trials 100

Inspect results:
    python sweep.py config_dual_lstm_cmal_kfold.toml --report
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
from src.lstm.train import train_model

# ── Presets ───────────────────────────────────────────────────────────────

LAYER_SCALES: dict[str, tuple[int, int]] = {
    "small":  (32, 64),
    "medium": (48, 96),
    "large":  (64, 128),
}

# Static feature sets — progressive widening across ALL 4 semantic groups.
# Each level adds depth within every category, not just new categories.
#   Topography : area, elevation, slope, ele_range, hyp_integral, twi_std
#   Network    : river area, time-of-conc, soil-group, drain-density, bifurc, sinuosity
#   Soil/Surface: forest%, clay%, sand%, climate-zone, lithology
#   Climate    : aridity, snow_frac, PET, high_precip_freq, precip, low_precip_dur, high_precip_dur
STATIC_FEATURE_SETS: dict[str, dict] = {
    # 8 features — 2 per group (Physical + Climate CSVs)
    "minimal": {
        "features": [
            # topo (2)
            "total_Shape_Area_km2",
            "ele_mt_uav",
            # network (2)
            "ria_ha_usu",
            "riv_tc_usu",
            # soil (2)
            "for_pc_use",
            "cly_pc_uav",
            # climate (2)
            "aridity_index",
            "snow_fraction",
        ],
        "log_transform": ["total_Shape_Area_km2", "ria_ha_usu"],
        "embed_dim": 6,
        "needs_dem_network": False,
    },
    # 13 features — 3–4 per group (Physical + Climate CSVs)
    "core": {
        "features": [
            # topo (3)
            "total_Shape_Area_km2",
            "ele_mt_uav",
            "slp_dg_uav",
            # network (3)
            "ria_ha_usu",
            "riv_tc_usu",
            "sgr_dk_sav",
            # soil (3)
            "for_pc_use",
            "cly_pc_uav",
            "snd_pc_uav",
            # climate (4)
            "aridity_index",
            "snow_fraction",
            "pet_mean",
            "high_precip_freq",
        ],
        "log_transform": ["total_Shape_Area_km2", "ria_ha_usu"],
        "embed_dim": 8,
        "needs_dem_network": False,
    },
    # 18 features — 3–7 per group, all Physical + Climate columns
    "full": {
        "features": [
            # topo (3)
            "total_Shape_Area_km2",
            "ele_mt_uav",
            "slp_dg_uav",
            # network (3)
            "ria_ha_usu",
            "riv_tc_usu",
            "sgr_dk_sav",
            # soil (5)
            "for_pc_use",
            "cly_pc_uav",
            "snd_pc_uav",
            "clz_cl_smj",
            "lit_cl_smj",
            # climate (7)
            "aridity_index",
            "snow_fraction",
            "pet_mean",
            "high_precip_freq",
            "precip_mean",
            "low_precip_dur",
            "high_precip_dur",
        ],
        "log_transform": ["total_Shape_Area_km2", "ria_ha_usu"],
        "embed_dim": 10,
        "needs_dem_network": False,
    },
    # 24 features — 5–7 per group, adds DEM + Network CSV features
    "extended": {
        "features": [
            # topo (6) — adds DEM-derived shape descriptors
            "total_Shape_Area_km2",
            "ele_mt_uav",
            "slp_dg_uav",
            "ele_range",
            "hyp_integral",
            "twi_std",
            # network (6) — adds Network-derived structure
            "ria_ha_usu",
            "riv_tc_usu",
            "sgr_dk_sav",
            "drain_density_km",
            "bifurc_ratio",
            "main_chan_sinuosity",
            # soil (5)
            "for_pc_use",
            "cly_pc_uav",
            "snd_pc_uav",
            "clz_cl_smj",
            "lit_cl_smj",
            # climate (7)
            "aridity_index",
            "snow_fraction",
            "pet_mean",
            "high_precip_freq",
            "precip_mean",
            "low_precip_dur",
            "high_precip_dur",
        ],
        "log_transform": ["total_Shape_Area_km2", "ria_ha_usu"],
        "embed_dim": 12,
        "needs_dem_network": True,
    },
}


# ── Trial config builder ─────────────────────────────────────────────────

def _make_config(
    trial: optuna.Trial,
    base: Config,
    *,
    force_output_type: str | None = None,
) -> Config:
    """Clone *base* and apply trial suggestions."""
    cfg = dataclasses.replace(base)

    # ── Fixed from prior evidence ──
    cfg.model_type = "dual"
    cfg.static_dropout = 0.10
    cfg.info_gap = False
    cfg.aux_loss_weight = 0.25           # ρ=-0.02 in prior sweep; no signal
    cfg.aux_peak_asymmetry = 4.5
    cfg.baseflow_alpha = 0.925
    cfg.fast_hidden_size, cfg.slow_hidden_size = LAYER_SCALES["medium"]  # 9/11 top-half

    # ── Search space (8 dimensions: 5 continuous + 3 categorical) ──

    # 1. Output type — deterministic vs probabilistic (CMAL)
    if force_output_type:
        cfg.output_type = force_output_type
    else:
        cfg.output_type = trial.suggest_categorical(
            "output_type", ["deterministic", "cmal"],
        )

    if cfg.output_type == "cmal":
        # CMAL fixed params
        cfg.cmal_loss = "crps"
        cfg.cmal_beta_crps = 0.0
        cfg.cmal_entropy_weight = 0.1
        cfg.cmal_scale_reg_weight = 0.0
        cfg.cmal_crps_n_samples = 50
        cfg.cmal_hidden_size = 64
        cfg.log_loss_lambda = 0.0
        # 2a. CMAL components (conditional)
        cfg.cmal_n_components = trial.suggest_categorical(
            "cmal_n_components", [3, 5],
        )
    else:
        # 2b. Log-MSE blend weight (conditional; 0 = pure MSE)
        cfg.log_loss_lambda = trial.suggest_float(
            "log_loss_lambda", 0.0, 0.15,
        )

    # 3. Fast window — log-int lets TPE interpolate across scales
    cfg.fast_window = trial.suggest_int(
        "fast_window", 7, 180, log=True,
    )

    # 4. Dropout — continuous between 0.05 and 0.30
    cfg.dropout = trial.suggest_float(
        "dropout", 0.05, 0.30,
    )

    # 5–6. Extreme weighting — 2 continuous params.
    #   weight_max=1.0 disables extreme weighting (uniform loss).
    #   Derived: ramp = 2×threshold, peak_boost = weight_max.
    cfg.extreme_weight_max = trial.suggest_float(
        "extreme_weight_max", 1.0, 15.0,
    )
    cfg.extreme_threshold = trial.suggest_float(
        "extreme_threshold", 2.0, 7.0,
    )
    cfg.extreme_ramp = 2.0 * cfg.extreme_threshold
    cfg.extreme_peak_boost = cfg.extreme_weight_max

    # 7. Learning rate (log-scale)
    cfg.learning_rate = trial.suggest_float(
        "learning_rate", 2e-4, 1.2e-3, log=True,
    )

    # 8. Static feature set (ordinal — changes column list; must stay categorical)
    feat_name = trial.suggest_categorical(
        "static_feature_set", list(STATIC_FEATURE_SETS.keys()),
    )
    fset = STATIC_FEATURE_SETS[feat_name]
    cfg.static_features = fset["features"]
    cfg.log_transform_static = fset["log_transform"]
    cfg.static_embedding_dim = fset["embed_dim"]

    # Extended feature set requires DEM & Network CSV paths
    if fset.get("needs_dem_network"):
        static_dir = Path(cfg.static_basin_atlas).parent
        cfg.static_dem = static_dir / "DEM_Attributes_Watersheds.csv"
        cfg.static_network = static_dir / "Network_Attributes_Watersheds.csv"
    else:
        cfg.static_dem = None
        cfg.static_network = None

    # Sweep overrides: fewer epochs, no SWA
    cfg.num_epochs = getattr(base, "_sweep_epochs", 25)
    cfg.use_swa = False

    # Per-trial output dir
    cfg.output_dir = base.output_dir / f"trial_{trial.number}"

    return cfg


# ── Objective ─────────────────────────────────────────────────────────────

KGE_FLOOR = -0.5   # clip floor for both KGE and NSE
FHV_CAP = 100.0  # cap |FHV| at 100% for scoring


def _floored_mean(series, floor=-0.5):
    """Mean after clipping to floor, ignoring NaN."""
    vals = series.dropna().clip(lower=floor)
    return float(vals.mean()) if len(vals) else float("nan")


def _capped_abs_mean(series, cap=100.0):
    """Mean of |values| capped at cap, ignoring NaN."""
    vals = series.dropna().abs().clip(upper=cap)
    return float(vals.mean()) if len(vals) else float("nan")


def objective(
    trial: optuna.Trial,
    *,
    base_config: Config,
    fold_data: tuple,
    device: torch.device,
    fhv_weight: float = 0.1,
    force_output_type: str | None = None,
    no_prune: bool = False,
) -> float:
    """Train fold 0 and return composite score (maximised).

    score = tier-weighted KGE − fhv_weight × (tier-weighted |FHV| / 100)
    """
    basin_ids, climate_data, flow_data, static_df, tier_map = fold_data

    config = _make_config(trial, base_config, force_output_type=force_output_type)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    folds = create_folds(
        basin_ids, tier_map, flow_data,
        config.n_folds, config.holdout_fraction, config.seed,
    )
    train_ids, val_ids = folds[0]

    norm = compute_norm_stats(
        climate_data, flow_data, static_df,
        train_ids, basin_ids, config,
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

    # Pruning callback — report val loss each epoch
    def _prune_callback(epoch: int, val_loss: float) -> None:
        trial.report(val_loss, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

    callback = None if no_prune else _prune_callback

    try:
        model, _history = train_model(
            model, train_loader, val_loader, config, 0, device,
            epoch_callback=callback,
            norm_stats=norm,
        )
    except optuna.TrialPruned:
        del model, train_ds, val_ds, train_loader, val_loader
        gc.collect()
        if device.type == "mps":
            torch.mps.empty_cache()
        raise

    df = evaluate_fold(model, val_ds, tier_map, config, 0, device)

    # Free memory
    del model, train_ds, val_ds, train_loader, val_loader
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()

    # ── Per-tier metrics ──
    tier_kge, tier_fhv = {}, {}
    for t in (1, 2, 3):
        sub = df[df["tier"] == t]
        tier_kge[t] = _floored_mean(sub["kge"], floor=KGE_FLOOR)
        tier_fhv[t] = _capped_abs_mean(sub["fhv"], cap=FHV_CAP)
        trial.set_user_attr(f"tier{t}_nse", float(sub["nse"].median()))
        trial.set_user_attr(f"tier{t}_kge", tier_kge[t])
        trial.set_user_attr(f"tier{t}_fhv", float(sub["fhv"].median()))
        trial.set_user_attr(f"tier{t}_flv", float(sub["flv"].median()))
        if "picp_90" in sub.columns:
            trial.set_user_attr(f"tier{t}_picp90", float(sub["picp_90"].median()))

    # Guard NaN
    if any(np.isnan(v) for v in tier_kge.values()):
        return float("-inf")

    # Tier-weighted objective
    w = {1: 0.25, 2: 0.50, 3: 0.25}
    weighted_kge = sum(w[t] * tier_kge[t] for t in (1, 2, 3))
    weighted_abs_fhv = sum(w[t] * tier_fhv[t] for t in (1, 2, 3))

    # Track secondary metrics as attrs
    trial.set_user_attr("weighted_kge", weighted_kge)
    trial.set_user_attr("weighted_abs_fhv", weighted_abs_fhv)
    trial.set_user_attr("overall_nse", float(df["nse"].median()))
    trial.set_user_attr("overall_kge", float(df["kge"].median()))

    # Composite: KGE with FHV penalty (|FHV|/100 normalises to ~KGE scale)
    score = weighted_kge - fhv_weight * (weighted_abs_fhv / 100.0)
    trial.set_user_attr("score", score)
    return score


# ── Report ────────────────────────────────────────────────────────────────

def print_report(study: optuna.study.Study) -> None:
    """Print top trials ranked by weighted KGE."""
    completed = [
        t for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE
    ]
    pruned = [
        t for t in study.trials
        if t.state == optuna.trial.TrialState.PRUNED
    ]
    failed = len(study.trials) - len(completed) - len(pruned)
    print(f"\nTrials: {len(completed)} completed, {len(pruned)} pruned"
          f"{f', {failed} failed' if failed else ''}, "
          f"{len(study.trials)} total")

    if not completed:
        print("No completed trials.")
        return

    ranked = sorted(completed, key=lambda x: x.value, reverse=True)
    best = ranked[0]
    print(f"\nBest trial: #{best.number}  wKGE = {best.value:.4f}")

    top_n = ranked[:min(20, len(ranked))]
    print(f"\n{'=' * 98}")
    print(f"TOP {len(top_n)} TRIALS BY COMPOSITE SCORE  (score = wKGE - \u03bb\u00b7w|FHV|/100)")
    print(f"{'=' * 98}")
    print(f"{'#':>4s}  {'score':>7s}  {'wKGE':>7s}  {'w|FHV|':>7s}  {'ovKGE':>6s}  "
          f"{'ovNSE':>6s}  {'type':>5s}  {'K/log\u03bb':>6s}  {'fw':>4s}  {'drop':>5s}  "
          f"{'wMax':>5s}  {'thr':>5s}  {'LR':>8s}  {'feats':>8s}")
    print("-" * 98)
    for t in top_n:
        p = t.params
        a = t.user_attrs
        otype = p.get('output_type', 'cmal')
        if otype == 'cmal':
            cond_col = f"K={p.get('cmal_n_components', '?')}"
        else:
            cond_col = f"\u03bb={p.get('log_loss_lambda', 0):.2f}"
        print(
            f"{t.number:4d}  "
            f"{t.value:7.4f}  "
            f"{a.get('weighted_kge', t.value):7.4f}  "
            f"{a.get('weighted_abs_fhv', 0):7.2f}  "
            f"{a.get('overall_kge', 0):6.3f}  "
            f"{a.get('overall_nse', 0):6.3f}  "
            f"{otype:>5s}  "
            f"{cond_col:>6s}  "
            f"{p['fast_window']:4d}  "
            f"{p['dropout']:5.3f}  "
            f"{p['extreme_weight_max']:5.1f}  "
            f"{p['extreme_threshold']:5.2f}  "
            f"{p['learning_rate']:8.6f}  "
            f"{p['static_feature_set']:>8s}"
        )

    # Per-tier breakdown of top-3
    top3 = ranked[:3]
    print(f"\n{'=' * 92}")
    print("TOP-3 — PER-TIER DETAIL")
    print(f"{'=' * 92}")
    for t in top3:
        a = t.user_attrs
        p = t.params
        print(f"\n  Trial #{t.number}  (wKGE={t.value:.4f}, w|FHV|={a.get('weighted_abs_fhv', 0):.2f})")
        otype = p.get('output_type', 'cmal')
        cond = (f"K={p['cmal_n_components']}" if otype == 'cmal'
                else f"\u03bb={p.get('log_loss_lambda', 0):.2f}")
        print(f"  Config: {otype}, {cond}, fw={p['fast_window']}, drop={p['dropout']:.3f}, "
              f"wMax={p['extreme_weight_max']:.1f}, thr={p['extreme_threshold']:.2f}, "
              f"lr={p['learning_rate']:.6f}, feats={p['static_feature_set']}")
        for tier in (1, 2, 3):
            nse = a.get(f"tier{tier}_nse", float("nan"))
            kge = a.get(f"tier{tier}_kge", float("nan"))
            fhv = a.get(f"tier{tier}_fhv", float("nan"))
            flv = a.get(f"tier{tier}_flv", float("nan"))
            picp = a.get(f"tier{tier}_picp90", float("nan"))
            print(f"    T{tier}: NSE={nse:.3f}  KGE={kge:.3f}  "
                  f"FHV={fhv:+.1f}  FLV={flv:+.1f}  PICP={picp:.3f}")


# ── Main ──────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Optuna hyperparameter sweep (maximise tier-weighted KGE).",
    )
    parser.add_argument("config", nargs="?", default="config_dual_lstm_cmal_kfold.toml",
                        help="Base TOML config file.")
    parser.add_argument("--n-trials", type=int, default=100,
                        help="Number of Optuna trials (default: 100).")
    parser.add_argument("--epochs", type=int, default=25,
                        help="Max epochs per trial (default: 25).")
    parser.add_argument("--fhv-weight", type=float, default=0.1,
                        help="FHV penalty weight in composite score (default: 0.1).")
    parser.add_argument("--output-type", choices=["cmal", "deterministic"],
                        default=None,
                        help="Force output type (skip suggest_categorical).")
    parser.add_argument("--no-prune", action="store_true",
                        help="Disable MedianPruner (useful for CMAL-only runs).")
    parser.add_argument("--report", action="store_true",
                        help="Print ranked results from existing study and exit.")
    args = parser.parse_args()

    base_config = load_config(args.config)
    base_config._sweep_epochs = args.epochs  # type: ignore[attr-defined]

    # Ensure DEM & Network CSVs are loaded so static_df has all columns
    # needed by the "extended" feature set (even if base TOML omits them).
    static_dir = Path(base_config.static_basin_atlas).parent
    if base_config.static_dem is None:
        base_config.static_dem = static_dir / "DEM_Attributes_Watersheds.csv"
    if base_config.static_network is None:
        base_config.static_network = static_dir / "Network_Attributes_Watersheds.csv"

    sweep_dir = base_config.output_dir / "sweep"
    base_config.output_dir = sweep_dir
    sweep_dir.mkdir(parents=True, exist_ok=True)

    # Optuna study (persistent SQLite)
    storage = f"sqlite:///{sweep_dir / 'sweep.db'}"
    study_name = "dual-sweep"

    if args.report:
        study = optuna.load_study(
            study_name=study_name, storage=storage,
        )
        print_report(study)
        return

    # Device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # Load data once
    print("Loading data …")
    fold_data = load_all_data(base_config)
    basin_ids = fold_data[0]
    tier_map = fold_data[4]
    n_per_tier = {t: sum(1 for v in tier_map.values() if v == t) for t in (1, 2, 3)}
    print(f"  {len(basin_ids)} basins  "
          f"(T1={n_per_tier[1]}, T2={n_per_tier[2]}, T3={n_per_tier[3]})")

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    pruner = (
        optuna.pruners.NopPruner()
        if args.no_prune
        else optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=12)
    )

    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        storage=storage,
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(
            n_startup_trials=10,
            multivariate=True,
            group=True,
            seed=42,
        ),
        pruner=pruner,
    )

    if args.output_type:
        print(f"Forced output_type={args.output_type}")
    if args.no_prune:
        print("Pruning disabled")

    study.optimize(
        lambda trial: objective(
            trial, base_config=base_config, fold_data=fold_data, device=device,
            fhv_weight=args.fhv_weight,
            force_output_type=args.output_type,
            no_prune=args.no_prune,
        ),
        n_trials=args.n_trials,
    )

    print_report(study)


if __name__ == "__main__":
    main()
