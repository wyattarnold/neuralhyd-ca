# Model Documentation Overview

This directory documents the active modeling approaches in `neuralhyd-ca`. The root [README.md](./../../README.md) stays short; these pages carry implementation-level architecture, data, loss, and training details.

Run-specific metrics are not repeated here because they change as experiments are retrained. For current results, use `data/training/output/<run>/all_fold_results.csv`, `data/training/output/<run>/fold_<n>/basin_results.csv`, or post-processing outputs under `data/eval/`.

## Active Configurations

| Config | Family page | Entry point |
| --- | --- | --- |
| [cfg_single_lstm.toml](./../../scripts/cfg_single_lstm.toml) | [Single LSTM baseline](lstm.md#single-lstm-baseline) | `python scripts/train_kfold.py scripts/cfg_single_lstm.toml` |
| [cfg_dual_lstm.toml](./../../scripts/cfg_dual_lstm.toml) | [Dual-pathway LSTM](lstm.md#dual-pathway-lstm) | `python scripts/train_kfold.py scripts/cfg_dual_lstm.toml` |
| [cfg_single_lstm_cmal.toml](./../../scripts/cfg_single_lstm_cmal.toml) | [Single LSTM with CMAL](lstm.md#single-lstm-with-cmal) | `python scripts/train_kfold.py scripts/cfg_single_lstm_cmal.toml` |
| [cfg_moe_lstm.toml](./../../scripts/cfg_moe_lstm.toml) | [Unsupervised MoE-tau](lstm.md#unsupervised-moe-tau) | `python scripts/train_kfold.py scripts/cfg_moe_lstm.toml` |

### Grouped Static Encoder Variants

Two additional configs swap the default flat static-attribute MLP for a
**grouped** encoder that gives each semantic group of watershed
attributes (topography, routing, soil, land_cover, hydroclimate) its own
small sub-encoder before a fusion layer. A `[static_feature_groups]`
table in the config selects the grouped encoder and declares group
membership; `static_group_dropout` randomly drops whole groups during
training as a regularizer.

| Config | Base architecture |
| --- | --- |
| [cfg_single_lstm_grouped_static.toml](./../../scripts/cfg_single_lstm_grouped_static.toml) | Single LSTM with grouped static encoder |
| [cfg_dual_lstm_grouped_static.toml](./../../scripts/cfg_dual_lstm_grouped_static.toml) | Dual-pathway LSTM with grouped static encoder |

## Modeling Goal

The project predicts daily streamflow for California watersheds from observed daily climate forcing and static watershed attributes. The models are hindcast simulators: they use the observed precipitation and temperature context for the target day, not weather forecasts. The practical question is: given the historical climate record and the basin attributes, what streamflow should this watershed produce today?

The active model family is the LSTM gauge model: these predict gauge-scale flow directly from watershed climate sequences and static watershed attributes. See [lstm.md](./lstm.md).

## Data Preparation Pipeline

The main data-preparation entry point is [prepare_data.py](./../../scripts/prepare_data.py). It builds training/evaluation inputs from raw gauge flow, watershed geometries, gridded climate, BasinATLAS/static products, and HUC overlays.

```mermaid
flowchart TD
    Catalog["Step 0<br/>training watershed catalog<br/>USGS + optional CDEC FNF"]:::seq
    FlowRaw["Step 1<br/>retrieve raw USGS flow"]:::input
    Climate["Step 2<br/>area-weighted climate series"]:::seq
    ClimateQA["Step 3<br/>monthly climate verification"]:::seq
    Static["Step 4<br/>BasinATLAS static attributes"]:::seq
    ClimStats["Step 5<br/>climate statistics"]:::seq
    FlowClean["Step 6<br/>flow cleaning and filtering"]:::seq
    QA["Step 7<br/>comprehensive QA/QC"]:::seq
    Tier["Step 8<br/>flow-precip QA and tier sorting"]:::seq
    TrainingTree["data/training<br/>model-ready zarr/CSV inputs"]:::output
    EvalTree["data/eval<br/>full-domain HUC products and post-process outputs"]:::output

    Catalog --> FlowRaw --> FlowClean
    Catalog --> Climate --> ClimateQA
    Catalog --> Static
    Climate --> ClimStats
    Static --> ClimStats
    FlowClean --> QA --> Tier --> TrainingTree
    ClimateQA --> TrainingTree
    ClimStats --> TrainingTree
    Static --> TrainingTree
    Climate --> EvalTree
    Static --> EvalTree
    ClimStats --> EvalTree

    classDef input fill:#e3f2fd,stroke:#1976d2,color:#0d47a1
    classDef seq fill:#ede7f6,stroke:#5e35b1,color:#311b92
    classDef output fill:#fffde7,stroke:#f9a825,color:#f57f17
```

Important preparation concepts:

- Step 2 reads the gridded meteo archive, so a run that includes it needs `--meteo-dir <store>`; without it the script exits before step 0.
- `--target watersheds` writes `data/training/climate/watersheds.zarr` and `data/training/static/watersheds/`.
- `--target huc8`, `huc10`, or `huc12` writes full-domain, inference-only products to `data/eval/climate/<level>.zarr` and `data/eval/static/<level>/`. There is no HUC training subset; `post_process.py --simulate --target <level>` reads these.
- `--geo-intersect` (a one-time GIS step that writes the `data/prepare/geo_ops/` intersect tables used by steps 2 and 4) runs after step 0 and before step 1.
- Step 0 can include or exclude CDEC full-natural-flow basins; `--exclude-cdec` also skips CDEC flows in steps 6 and 8. Model configs still have `include_cdec_basins` because a run may intentionally exclude them even when prepared data exist.
- Step 8 assigns hydroclimatic tiers used for fold stratification and reporting.

Statewide 1/16° gridded inputs (daily forcing from the same WGEN store as step 2, but with the x10 precip and tmin/tmax corrections applied, plus the AlphaEarth 2017 embedding and its 2017–2025 multi-year mean) come from a separate entry point, [prepare_gridded.py](./../../scripts/prepare_gridded.py). They sit outside this chart and no model reads them yet; see the [dataset card](./../../data/gridded/README.md).

## Training Inputs And Storage Layout

Active training consumes these prepared products:

| Product | Purpose |
| --- | --- |
| `data/training/flow.zarr` | Daily gauge flow and tier metadata. |
| `data/training/climate/watersheds.zarr` | Daily watershed-mean `precip_mm`, `tmax_c`, `tmin_c`. |
| `data/training/static/watersheds/*.csv` | Watershed physical attributes and climate statistics keyed by `PourPtID`. |

All active training scripts are run from the repository root with the `neuralhyd` environment. The unified cross-validation entry point is [train_kfold.py](./../../scripts/train_kfold.py):

```bash
python scripts/train_kfold.py scripts/cfg_dual_lstm.toml
python scripts/train_kfold.py scripts/cfg_single_lstm.toml
```

## Training Basis And Leakage Control

Every active model is trained with spatial cross-validation: basins are split, not timesteps. The goal is ungauged-basin generalization, so no basin appears in both train and validation within a fold.

All active configs use `include_cdec_basins = false`, so training uses the 210 USGS gauge watersheds that survive QA/QC filtering and static-attribute intersection. The 14 CDEC FNF pseudo-gauge basins are still prepared (`flow.zarr` holds 224 basins) and a config can opt back in with `include_cdec_basins = true`; the runs currently under `data/training/output/` were trained that way (224 basins). Normalization statistics are computed from training basins only. Validation basins are held out for statistics, model fitting, and checkpoint selection except for their fixed metadata needed to construct tensors and report metrics.

```mermaid
flowchart LR
    Config["TOML config<br/>model_type"]:::input
    Load["load_all_data()"]:::seq
    Fold["build spatial folds"]:::seq
    Norm["training-only normalization"]:::seq
    Train["train fold model"]:::head
    Select["select checkpoint<br/>loss, NSE, or KGE"]:::head
    Eval["held-out evaluation"]:::output
    Aggregate["all_fold_results.csv"]:::output

    Config --> Load --> Fold --> Norm --> Train --> Select --> Eval --> Aggregate

    classDef input fill:#e3f2fd,stroke:#1976d2,color:#0d47a1
    classDef seq fill:#ede7f6,stroke:#5e35b1,color:#311b92
    classDef head fill:#fff3e0,stroke:#ef6c00,color:#e65100
    classDef output fill:#fffde7,stroke:#f9a825,color:#f57f17
```

## Hydroclimatic Tiers

The tier split is used for fold balance and interpretation:

- Tier 1: warmer, lower-elevation rainfall-dominated basins.
- Tier 2: transitional mixed rain-snow basins. This is usually the hardest generalization target and is often the most useful single diagnostic of model quality.
- Tier 3: colder, higher-elevation snow-influenced basins where long memory matters.

Held-out metrics are reported by tier and across all basins. This avoids one aggregate score hiding systematic failures in snow, rain, or mixed-regime watersheds.

## Normalization And Units

Gauge-mode LSTM targets are converted from cfs to mm/day, then divided by each basin's mean daily precipitation to form a runoff ratio. LSTM evaluation multiplies predictions back by the same precipitation scale before computing metrics.

Climate inputs and static attributes are z-scored with training-basin statistics only. Heavy-tailed static attributes listed in `log_transform_static` are log10-transformed first.

## Losses And Checkpoint Selection

LSTM deterministic configs use weighted MSE/log-MSE blends, optional pathway auxiliary losses, and optional CMAL CRPS/NLL. Details are in [docs/models/lstm.md](lstm.md#shared-deterministic-losses).

Checkpoint selection is config-specific via `validation_selection_metric`: `"loss"` (default), `"nse"`, or `"kge"`.

## Outputs And Post-Processing

Typical training outputs are:

- `data/training/output/<run>/log.txt`
- `data/training/output/<run>/all_fold_results.csv`
- `data/training/output/<run>/fold_<n>/best_model.pt`
- `data/training/output/<run>/fold_<n>/basin_results.csv`
- `data/training/output/<run>/fold_<n>/timeseries/<basin_id>.csv`

Post-processing is handled by [post_process.py](./../../scripts/post_process.py):

```bash
python scripts/post_process.py --eval dual_lstm single_lstm          # per-basin metrics → data/eval/<run>.csv
python scripts/post_process.py --cdf --barplot --runs dual_lstm ...  # CDF + median barplot PNGs
python scripts/post_process.py --simulate dual_lstm --target training_watersheds  # historical sim CSVs
python scripts/post_process.py --cdec-barplot dual_lstm single_lstm  # SAC-SMA vs neural on 14 CDEC basins
```

`--eval` recomputes FHV, FeHV and FLV from each fold's saved timeseries with the current flow-duration-curve definitions, so a run's stored `basin_results.csv` values may differ from its `data/eval/<run>.csv`. `--simulate --target` takes `training_watersheds` (each basin's held-out fold model) or `watersheds`, `huc8`, `huc10`, `huc12` (ensemble of all fold models). `--cdec-barplot` needs runs trained with `include_cdec_basins = true`.

After regenerating sim products, run `python -m app.build_data` to refresh the Streamflow Explorer app's Parquet bundles in `app/data/timeseries/`.

## Code Map

Important implementation files:

- [config.py](./../../src/lstm/config.py): LSTM config dataclass and TOML loading.
- [dataset.py](./../../src/lstm/dataset.py): gauge-mode loading, folds, normalization, and datasets.
- [model.py](./../../src/lstm/model.py): single (with optional CMAL), dual, and MoE LSTM architectures.
- [train.py](./../../src/lstm/train.py): LSTM training loop, SWA, and checkpoint I/O.
- [loss.py](./../../src/lstm/loss.py): LSTM training losses and evaluation metrics.
- [src/data/](../../src/data): data preparation modules called by [prepare_data.py](./../../scripts/prepare_data.py).
- [src/data/gridded/](../../src/data/gridded) and [prepare_gridded.py](./../../scripts/prepare_gridded.py): statewide 1/16° gridded inputs in `data/gridded/`. `lattice.py` defines the grid, `ncio.py` holds the NetCDF, checksum and provenance helpers, `forcing.py` builds the daily NetCDFs (one per variable), and `aef.py` runs the AlphaEarth Earth Engine reduction (2017, and each year of the 2017–2025 mean). Dataset card: [data/gridded/README.md](./../../data/gridded/README.md).
