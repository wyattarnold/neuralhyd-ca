# Copilot Instructions — neuralhyd-ca

## Project Overview

This is a **daily streamflow prediction** project using LSTM networks trained on 216 California USGS watersheds. Two model variants are available (selected via `Config.model_type`): a dual-pathway LSTM with multiplicative composition and physically interpretable flow decomposition, and a single-LSTM baseline. The model predicts today's streamflow from a lookback window of daily climate forcing (precipitation, tmax, tmin) conditioned on static watershed attributes. It is **not** a forecast model — it uses observed climate inputs, not future predictions.

## Environment

Use the `neuralhyd` environment for all scripts. The project is tested on Python 3.14 with the following key dependencies:

- **Python**: 3.14 via miniforge3 (`/Users/wyatt/miniforge3/envs/neuralhyd/bin/python`)
- **Key deps**: PyTorch ≥ 2.0, pandas, numpy, scikit-learn, matplotlib, tqdm
- **OS**: macOS (Apple Silicon). Set `KMP_DUPLICATE_LIB_OK=TRUE` to avoid duplicate OpenMP crashes.
- **Device preference**: MPS → CUDA → CPU (auto-detected in `scripts/train_kfold.py`)

## Running

```bash
KMP_DUPLICATE_LIB_OK=TRUE /Users/wyatt/miniforge3/envs/neuralhyd/bin/python scripts/train_kfold.py
```

All hyperparameters live in `scripts/config.toml`. Pass an alternate TOML file as the first argument to `scripts/train_kfold.py` to run a named experiment — the output directory is derived automatically from the filename (e.g. `config_single.toml` → `data/training/output/single/`). The `Config` dataclass in `src/lstm/config.py` is the typed container; use `load_config(path)` from that module to instantiate it.

## Repository Structure

```
scripts/
  train_kfold.py              # Entry point: k-fold stratified spatial cross-validation
  train_final.py              # Train on full dataset for deployment
  prepare_data.py             # Data preparation pipeline (steps 1–8 + analysis)
  post_process.py             # Post-training CLI: eval metrics, CDF plots, simulation
  config.toml                 # Default experiment configuration
  config_dual_lstm_kfold.toml # Named experiment config (dual model)
  config_single_lstm_kfold.toml # Named experiment config (single baseline)
src/
  paths.py                   # Top-level path constants (DEFAULT_CONFIG, TRAINING_OUTPUT_DIR)
  lstm/                      # LSTM model package
    config.py                # Config dataclass + load_config() — typed container for TOML values
    dataset.py               # load_all_data(), create_folds(), compute_norm_stats(), HydroDataset
    model.py                 # DualPathwayLSTM, SingleLSTM, StaticEncoder, build_model()
    train.py                 # train_epoch(), validate_epoch(), train_model(), load_checkpoint()
    loss.py                  # mse_loss, blended_loss, pathway_auxiliary_loss, compute_nse/kge/fhv/flv
    evaluate.py              # evaluate_basin(), evaluate_fold()
  data/                      # Data preparation modules called by prepare_data.py
    paths.py                 # Centralised path constants for the data pipeline
  eval/                      # Post-training evaluation helpers
    metrics.py               # Aggregate metric computation
    plots.py                 # CDF and comparison plots
    simulate.py              # Re-run trained models to produce timeseries
data/
  training/          # All final training/evaluation inputs and model outputs
    climate/         # climate_<basin_id>.csv — daily precip_mm, tmax_c, tmin_c (1915–2018)
    flow/            # tier_{1,2,3}/<basin_id>_cleaned.csv — daily flow + climate
    static/          # Physical_Attributes_Watersheds.csv, Climate_Statistics_Watersheds.csv
    watersheds/      # watersheds.geojson, watersheds.csv
    output/          # Created at runtime — per-fold checkpoints, basin results, timeseries
  eval/              # Evaluation CSVs written by post_process.py --eval
  raw/               # Immutable source data (USGS flow downloads, watershed geometry)
  prepare/           # Intermediate pipeline outputs (geo_ops, verify_climate_data, etc.)
  external/          # External comparison data (CEC process-based model results)
```

## Architectures

### Dual-Pathway LSTM (`model_type="dual"`, default)

Two LSTM branches with **multiplicative composition** and an **information gap**:

- **Fast pathway**: LSTM (hidden=64), sees last 18 days. Captures storm runoff, event recession, direct surface response.
- **Slow pathway**: LSTM (hidden=128), sees the full 365-day window (or, when `info_gap=true`, days 1–347 — blind to the last 18 days). Captures baseflow, snowmelt dynamics, seasonal soil moisture.
- **Information gap** (optional, default off): When enabled, slow LSTM does not see the last `fast_window` days, preventing it from learning storm responses. When off, separation is driven by the multiplicative structure and head activations alone.
- **Multiplicative composition**: `q_total = q_slow × (1 + q_fast_raw)`. The slow pathway sets the baseflow level; the fast pathway acts as a dimensionless storm amplifier. Storm contribution scales with antecedent wetness.
- **Softplus activation**: `Softplus(x)` — strictly positive, smooth gradients, well-behaved near zero.
- **Static encoder**: MLP (16 features → 32 hidden → 10-dim embedding) concatenated to each dynamic timestep. Conditions all pathways on watershed properties (elevation, soil texture, snow fraction, aridity, etc.).
- **Auxiliary loss**: Asymmetric MSE supervision of pathway components against Lyne-Hollick baseflow separation targets (α=0.925, peak asymmetry=2.0×).
- **Parameters**: ~101k

### Single LSTM Baseline (`model_type="single"`)

One LSTM (hidden=128) processing the full 365-day lookback with the same static encoder. Simpler baseline — pathway outputs are zero-filled to keep the same 3-tuple interface. ~78k parameters.

### Common Interface

Both models return `(q_total, q_fast, q_slow)` from `forward()`. Use `build_model(config)` to instantiate the correct class — all consumers (train, evaluate, train_kfold) accept `nn.Module`.

## Validation Design

- **5-fold stratified spatial cross-validation** — basins are the unit of splitting, not timesteps
- Each fold holds out ~20% of each tier (Tier 1: rainfall-dominated, Tier 2: transitional, Tier 3: snow-dominated)
- **No watershed appears in both train and val within a fold** — this tests ungauged-basin generalization
- Primary metric: **per-tier median NSE, KGE, FHV, and FLV** on held-out basins

## Normalisation Conventions

- **Flow target**: converted from cfs to **mm/day** (`q × 2.44577 / area_km²`, where 2.44577 = 0.0283168 m³/cfs × 86400 s/day × 1000 mm/m ÷ 1e6 m²/km²) using raw basin area, then divided by per-basin std. This removes area as a confound and makes flow physically comparable across basins.
- **Climate inputs**: z-score normalised globally using training-basin statistics only
- **Static attributes**: z-score normalised globally using training basins; `total_Shape_Area_km2` and `ria_ha_usu` are log10-transformed first (the raw area value is used for mm/day conversion before the log transform)

## Data Details

- **216 basins** across 3 tiers: T1 (88 warm/low-elevation rainfall-dominated), T2 (97 transitional mixed rain-snow), T3 (31 cold/high-elevation snow-dominated)
- Climate records span 1915–2018 (~38k days); streamflow records vary by basin (typically 1950s–1980s or longer)
- The cleaned streamflow CSVs already contain climate columns — the separate `training/climate/` files provide the full 1915–2018 record for any basin
- Basin IDs are USGS site numbers (e.g., `11476500`)
- See `usgs_watersheds_characteristics_report.md` for detailed tier characterisation

## Key Domain Concepts

- **Tier 2 is the hardest generalization target** — mixed rain/snow behavior, most internal heterogeneity. Tier 2 performance is the best single indicator of model quality.
- **Tier 3 requires long memory** — snow accumulation creates multi-month lags between precipitation and runoff. The 365-day lookback and slow pathway are essential for these basins.
- **Static conditioning is critical** — the same precipitation can produce completely different runoff depending on elevation, soil type, and snow fraction. Without static attributes, the model cannot distinguish tier behaviors.
- Flow must be non-negative. The architecture enforces this via Softplus on all pathway outputs.

## Training Schedule

`train_model()` runs in two phases:
1. **Warmup + ReduceLROnPlateau**: `warmup_epochs` ramp-up, then LR halved on plateau. When `patience` exhausts without improvement, phase 2 begins (or training stops if `use_swa=false`).
2. **SWA phase** (when `use_swa=true`): fixed low LR (`swa_lr`), model weights averaged each epoch for `swa_patience` epochs. SWA often produces small consistent gains over the best single checkpoint.

## Checkpoint Format

Checkpoints (`best_model.pt`) are saved as dicts with two keys:
- `model_state_dict`: the `nn.Module` state dict
- `norm_stats`: the normalisation statistics dict (climate mean/std, static mean/std, per-basin flow std) — needed to normalise inputs for new basins at inference time

Use `load_checkpoint(path, model, device)` from `src.lstm.train` to load a checkpoint and recover the norm stats. It also handles legacy bare-state-dict files gracefully.

## Coding Conventions

- All configuration goes through the `Config` dataclass — no magic numbers in other modules.
- Type hints throughout; `from __future__ import annotations` in every module.
- PyTorch conventions: `batch_first=True` for LSTMs, `@torch.no_grad()` for inference.
- Data flows as: `load_all_data()` → `create_folds()` → `compute_norm_stats()` → `HydroDataset` → `DataLoader` → `train_model(norm_stats=norm)` → `evaluate_fold()`.
- Checkpoints bundle model weights + norm stats. Use `load_checkpoint()` from `src.lstm.train` to load them.
- The `HydroDataset.__getitem__` returns a 6-tuple: `(x_dynamic, x_static, y_norm, y_components, basin_id, flow_std)`.
- Evaluation denormalises predictions by multiplying by `flow_std` before computing NSE/KGE/FHV/FLV.

## When Modifying

- To add new dynamic features: update `Config.dynamic_features`, ensure they exist in the climate CSVs. `n_dynamic` in the model is auto-computed as `len(dynamic_features)`.
- To add new static features: update `Config.static_features`, ensure they exist in the static attribute CSVs (joined from BasinATLAS + Climate Statistics tables on `PourPtID`).
- To change the architecture: modify `src/lstm/model.py`. Both models share the `forward()` signature `(x_dynamic, x_static) → (q_total, q_fast, q_slow)` used by train, evaluate, and train_kfold — keep it stable or update all three. Use `build_model(config)` for instantiation.
- To change the loss: modify `src/lstm/loss.py`. By default the training loop uses `blended_loss` (MSE + log-MSE, controlled by `log_loss_lambda`; set to 0 for pure MSE). Swap in an NSE-based loss by replacing the `blended_loss` / `mse_loss` call in `train_epoch`.

## Documentation Alignment

- **`README.md`** contains architecture diagrams (Mermaid), data descriptions, and validation design. When architecture, features, or hyperparameters change, update the README to stay in sync.
- **`data/README.md`** documents the `data/` directory layout. Update when directories are added, renamed, or repurposed.
- **This file** (`copilot-instructions.md`) is the primary context for Copilot. Keep the Architecture section, Config dimensions, and parameter counts current.
