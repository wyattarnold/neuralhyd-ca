# neuralhyd-ca

Daily streamflow prediction for California watersheds using neural hydrology models.

> [!WARNING]
> The project is under active development. Model architectures, configs, and results may change as experiments are retrained.

Most models predict today's streamflow from observed daily climate forcing and static watershed attributes. They are hindcast or reconstruction models, not streamflow-forecast systems.

A web viewer for results is available at <https://neuralhyd-ca.onrender.com>.

## Installation

### 1. Git LFS (required before cloning)

This repository stores all of its data — zarr cubes, CSVs, GeoJSON, and model
checkpoints — in **Git LFS**. Install Git LFS *before* cloning, otherwise the
data files are checked out as tiny text pointer stubs instead of real content
and training fails with confusing zarr / reshape / "missing CDEC" errors.

```bash
# install Git LFS once per machine, then clone
git lfs install
git clone https://github.com/wyattarnold/neuralhyd-ca.git
cd neuralhyd-ca

# if you cloned BEFORE installing Git LFS, fetch the real data now
git lfs pull
```

Sanity check — this must print `"shape": [224]`, **not** a
`version https://git-lfs.github.com/...` pointer stub:

```bash
cat data/training/flow.zarr/basin/zarr.json
```

If it shows a pointer stub, or any other shape, the data did not materialize.
Run `git lfs pull`, then confirm `git status` is **clean** — a modified file
under `*.zarr/` or `src/` means a hand-edited / stale metadata file is shadowing
the real LFS content, and must be reverted (`git restore <file>`).

The optional statewide gridded NetCDFs in `data/gridded/` (~3 GB) are the one
exception: `.lfsconfig` excludes them from the default fetch, so they stay
pointer stubs until pulled explicitly (see
[Statewide gridded inputs](#statewide-gridded-inputs-optional)).

### 2. Conda environment

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

### Statewide gridded inputs (optional)

`data/gridded/` holds statewide 1/16° products on the Livneh lattice: daily
precipitation and tmax/tmin for 1915–2018 (one NetCDF per variable, x10 summer
precip spikes corrected, tmin/tmax inversions swapped), the AlphaEarth 2017
satellite embedding (64 bands per cell), and the equal-weight 2017–2025 mean of
the annual AlphaEarth embeddings (with the nine per-year layers alongside).
No model reads them yet. See the
[dataset card](data/gridded/README.md) for contents, lineage, caveats and
licence terms.

Fetch the committed NetCDFs and check them:

```bash
git lfs pull --include="data/gridded/*.nc" --exclude=""
python scripts/prepare_gridded.py verify
```

Or rebuild them. This needs a local copy of the WGEN NonDetrend-Unsplit store,
DWR WGEN Product A for the x10 gate, and an Earth Engine project for the
AlphaEarth steps:

```bash
python scripts/prepare_gridded.py grid      --meteo-dir <store>
python scripts/prepare_gridded.py forcing   --meteo-dir <store>
python scripts/prepare_gridded.py check-x10 --meteo-dir <store> --product-a-dir <Product_A/1>
python scripts/prepare_gridded.py aef --dry-run
python scripts/prepare_gridded.py aef --run --project <ee-project>      # ~49 EECU-h, resumable
python scripts/prepare_gridded.py aef --assemble
python scripts/prepare_gridded.py aef --check 30 --project <ee-project>  # gates partials + NetCDF
# 2017-2025 mean: 8 more years, ~392 EECU-h, resumable
python scripts/prepare_gridded.py aef --run --project <ee-project> --years all
python scripts/prepare_gridded.py aef --assemble-mean
python scripts/prepare_gridded.py aef --check 30 --year 2021 --project <ee-project>
python scripts/prepare_gridded.py verify --meteo-dir <store>
```

## Docs

Architecture details are consolidated by family under [docs/models/overview.md](docs/models/overview.md).

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

The backend is FastAPI and the frontend is React + Leaflet. The app serves prepared GeoJSON layers and timeseries data from `app/data/` and `data/eval/` products.

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
