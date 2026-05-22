# neuralhyd-ca

Daily streamflow prediction for California watersheds using neural hydrology models.

The project is under active development. Model architectures, configs, and results may change as experiments are retrained.

## What This Project Does

Most models predict today's streamflow from observed daily climate forcing and static watershed attributes. They are hindcast or reconstruction models, not streamflow-forecast systems.

A web viewer for results is available at <https://neuralhyd-ca.onrender.com>.

## Installation

Create the conda environment from the included environment file:

```bash
conda env create -f environment.yml
conda activate neuralhyd
```

## Quick Start

Run commands from the repository root.

Prepare data for a fresh setup:

```bash
python scripts/prepare_data.py
```

Train an experiment:

```bash
python scripts/train_kfold.py scripts/cfg_single_lstm.toml
python scripts/train_kfold.py scripts/cfg_dual_lstm.toml
```

Post-process trained runs:

```bash
python scripts/post_process.py --eval single_lstm
python scripts/post_process.py --cdf --barplot --runs single_lstm
python scripts/post_process.py --simulate dual_lstm --target training_watersheds
```

## Model Families

Architecture details are consolidated by family under [docs/models/overview.md](docs/models/overview.md).

| Family | Main configs | Details |
| --- | --- | --- |
| LSTM gauge models | `cfg_single_lstm.toml`, `cfg_dual_lstm.toml`, `cfg_dual_lstm_cmal.toml`, `cfg_moe_lstm.toml` | [docs/models/lstm.md](docs/models/lstm.md) |

## Repository Layout

```text
scripts/     Training, data preparation, and post-processing entry points
src/lstm/    LSTM models, datasets, losses, training, and evaluation
src/data/    Data preparation pipeline
data/        Training inputs, prepared data, model outputs, and eval products
app/         FastAPI + React/Leaflet Streamflow Explorer web app
docs/        Sphinx docs and model-family documentation
```

## Web App

Build it with:

```bash
python -c "import app.build_data as b; 
			b.build_training_watersheds_geojson(0.0001); 
			b.build_static_attrs();
			b.build_obs_parquet();
			b.build_obs_baseflow_parquet();  
			b.build_lstm_parquets();
			b.build_lstm_single_parquets();
			b.build_sacsma_parquet()"

cd app\frontend; npm run build
```

Serve the Streamflow Explorer app locally with:

```bash
python -m app serve
python -m app serve --port 9000
```

The backend is FastAPI and the frontend is React + Leaflet. The app serves prepared GeoJSON layers and timeseries data from `app/data/` and `data/eval/` products.

## Notes For Contributors

Use the `neuralhyd` environment for scripts. All cross-validation model families run through `scripts/train_kfold.py` with an explicit TOML config. Named configs use the `cfg_<model>.toml` convention and derive stable output directories from their filenames. Keep architecture documentation in the model-family pages when model behavior, inputs, losses, or training schedules change.
