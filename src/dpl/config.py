"""Typed configuration for the dPL+SAC-SMA model.

Mirrors :mod:`src.lstm.config` in style.  Loaded from a TOML file via
:func:`load_dpl_config`.  All hyperparameters and physical-parameter
ranges live here — no magic numbers in other modules.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from src.paths import PROJECT_ROOT, TRAINING_OUTPUT_DIR

# ---------------------------------------------------------------------------
# Physical parameter table
# ---------------------------------------------------------------------------
# (name, lo, hi)  — sigmoid-squashed output of gA is mapped to [lo, hi].
# Defaults follow Shen-group dPL+HBV conventions, NWS SAC-SMA & SNOW17 manuals,
# and Sungwook Wi's Steinschneider-Lab MATLAB module included in this repo.

# Bounds calibrated against the Steinschneider-Lab 15-CDEC pooled GA
# calibration (sacramento_ga_15cdec_pool.txt) so the GA optima for the
# Sacramento/CDEC basins fall comfortably inside [lo, hi].  Where the GA
# fixed a parameter (SCF, PXTEMP, SIDE) we pin or narrow the range
# accordingly; PRISM precip is already gauge-corrected so SCF is held
# near unity.

SNOW17_PARAMS: list[tuple[str, float, float]] = [
    ("SCF",    0.9,   1.2),    # near-pinned; PRISM precip already corrected
    ("PXTEMP", -1.0,  1.0),    # GA fixes at 0; allow narrow window
    ("MFMAX",  0.5,   3.0),    # widened upper to match GA range
    ("MFMIN",  0.05,  0.5),
    ("UADJ",   0.03,  0.20),
    ("MBASE",  0.0,   2.0),    # base temp is physically >= 0
    ("TIPM",   0.10,  1.0),    # tightened lo to GA range
    ("PLWHC",  0.02,  0.30),
    ("NMF",    0.05,  0.30),   # tightened upper to GA range
    ("DAYGM",  0.0,   0.30),
]

SACSMA_PARAMS: list[tuple[str, float, float]] = [
    ("UZTWM",  1.0,   1000.0),  # widened: GA optimum ~476 mm
    ("UZFWM",  1.0,   1000.0),  # widened: GA optimum ~809 mm
    ("LZTWM",  1.0,   1000.0),  # widened: GA optimum ~88 but allow large
    ("LZFPM",  1.0,   2000.0),  # widened
    ("LZFSM",  1.0,   2000.0),  # widened: GA optimum ~3631 (snow basins)
    ("UZK",    0.05,  0.75),
    ("LZPK",   0.001, 0.25),    # widened: GA optimum ~0.144
    ("LZSK",   0.01,  0.5),
    ("ZPERC",  1.0,   500.0),
    ("REXP",   1.0,   10.0),    # widened: GA optimum ~9.4
    ("PFREE",  0.0,   0.60),
    ("PCTIM",  0.0,   0.10),
    ("ADIMP",  0.0,   0.40),
    ("RIVA",   0.0,   0.40),
    ("SIDE",   0.0,   0.0),     # pinned (GA fixes at 0)
    ("RSERV",  0.0,   0.40),
    # Capillary-rise rate (Song et al. 2024 / Feng et al. 2024).  Fractional
    # daily upward flux from lower-zone free water to upper-zone tension
    # when surface is dry.  Active only when enable_capillary_rise=True.
    ("THETA_C", 0.0,  0.05),
]

PET_PARAMS: list[tuple[str, float, float]] = [
    ("HAMON_COEF", 0.6, 1.6),
]

LOHMANN_PARAMS: list[tuple[str, float, float]] = [
    # HRU (hillslope) gamma UH — daily.  N is shape, TAU is mean lag (days).
    # Saint-Venant river-routing parameters (VELO, DIFF) are deferred to a
    # future revision when per-HUC12 channel travel distance (flowlen) is
    # available; with flowlen=0 the river UH is delta(t=0) and only the
    # hillslope UH applies.
    ("UH_N",   1.5,  5.0),
    ("UH_TAU", 0.25, 5.0),  # mean travel time in days
]

# Dynamic params: only the recession-rate trio.  These vary with seasonal
# vegetation phenology, soil temperature, and frozen-ground state.
# Percolation curve params (ZPERC, REXP) are structural soil properties
# and stay static; this also matches Shen-group dPL precedent
# (Feng et al. 2022, Song et al. 2024) of <= 3 dynamic params.
DEFAULT_DYNAMIC_PARAMS: tuple[str, ...] = ("UZK", "LZPK", "LZSK")


def all_param_names() -> list[str]:
    """Names of every physical parameter, in canonical order."""
    return (
        [n for n, *_ in SNOW17_PARAMS]
        + [n for n, *_ in SACSMA_PARAMS]
        + [n for n, *_ in PET_PARAMS]
        + [n for n, *_ in LOHMANN_PARAMS]
    )


def param_bounds() -> tuple[list[float], list[float]]:
    """Return (lows, highs) for every parameter in canonical order."""
    lows: list[float] = []
    highs: list[float] = []
    for table in (SNOW17_PARAMS, SACSMA_PARAMS, PET_PARAMS, LOHMANN_PARAMS):
        for _, lo, hi in table:
            lows.append(lo)
            highs.append(hi)
    return lows, highs


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------

_PATH_FIELDS = frozenset([
    "data_dir", "huc12_climate_zarr",
    "huc12_physical_csv", "huc12_climate_stats_csv",
    "huc12_manifest_csv", "flow_zarr", "output_dir",
])


@dataclass
class DplConfig:
    # ----- Paths -----
    data_dir: Path
    huc12_climate_zarr: Path          # data/training/climate/huc12.zarr
    huc12_physical_csv: Path          # Physical_Attributes_HUC12.csv (BasinATLAS-derived)
    huc12_climate_stats_csv: Path     # Climate_Statistics_HUC12.csv (PRISM/Daymet stats)
    huc12_manifest_csv: Path          # PourPtID, huc12, gauge_area_km2, huc12_area_km2, ...
    flow_zarr: Path                   # data/training/flow.zarr
    output_dir: Path

    # ----- Sequence / windows -----
    seq_len: int                   # samples drawn during training (post-warmup)
    warmup_steps: int              # state spin-up days excluded from loss
    n_mul: int                     # parallel SAC-SMA components per HUC12 (1 = off)

    # ----- gA architecture -----
    static_embedding_dim: int
    static_hidden_size: int
    lstm_hidden_size: int          # gA dynamic encoder hidden size
    lstm_num_layers: int
    dropout: float
    static_dropout: float

    # ----- Physical parameter strategy -----
    dynamic_params: List[str]      # subset of param names predicted per timestep
    enable_snow17: bool
    enable_routing: bool

    # ----- Loss / training -----
    log_loss_lambda: float
    log_loss_epsilon: float
    batch_size: int                # number of basins per gradient step
    learning_rate: float
    weight_decay: float
    warmup_epochs: int
    num_epochs: int
    patience: int
    min_delta: float
    num_workers: int
    grad_clip: float
    use_swa: bool
    swa_lr: float
    swa_patience: int

    # ----- Validation -----
    n_folds: int
    seed: int

    # ----- Feature lists -----
    dynamic_features: List[str]    # forcings into gA + the physics core
    static_features: List[str]     # HUC12 attributes
    log_transform_static: List[str]

    # ----- Optional / defaulted -----
    basin_loss_weight_exponent: float = 0.25
    grad_smooth_eps: float = 1e-3   # used for soft-relu / smooth clamp helpers
    flow_eps: float = 1e-6
    # Peak-flow weighting on the primary blended loss.  When
    # ``extreme_peak_boost > 1.0`` the squared residual at every timestep
    # is multiplied by a ramp that rises from 1 (below ``q_start``) to
    # ``extreme_peak_boost`` (at and above ``q_top``).  Per-basin
    # ``q_start`` / ``q_top`` are precomputed from observed flow at
    # ``extreme_start_quantile`` and ``extreme_top_quantile`` respectively.
    extreme_peak_boost: float = 1.0
    extreme_start_quantile: float = 0.98
    extreme_top_quantile: float = 0.995
    # SAC-SMA structural switches (Song et al. 2024)
    enable_capillary_rise: bool = True
    implicit_percolation: bool = True
    n_inc: int = 12
    # Auxiliary baseflow supervision
    baseflow_aux_lambda: float = 0.0
    baseflow_alpha: float = 0.925


def _resolve(p: str | Path, root: Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else (root / p).resolve()


def load_dpl_config(toml_path: str | Path) -> DplConfig:
    """Parse a DplConfig TOML file."""
    toml_path = Path(toml_path)
    with open(toml_path, "rb") as fh:
        cfg = tomllib.load(fh)

    flat: dict = {}
    for section in cfg.values():
        if isinstance(section, dict):
            flat.update(section)
        else:
            flat.update(cfg)

    root = toml_path.parent
    for key in _PATH_FIELDS:
        if key in flat:
            flat[key] = _resolve(flat[key], root)

    if "output_dir" not in flat:
        flat["output_dir"] = TRAINING_OUTPUT_DIR / toml_path.stem.replace("config_", "")

    flat.setdefault("dynamic_params", list(DEFAULT_DYNAMIC_PARAMS))

    return DplConfig(**flat)


__all__ = [
    "DplConfig", "load_dpl_config",
    "SNOW17_PARAMS", "SACSMA_PARAMS", "PET_PARAMS", "LOHMANN_PARAMS",
    "DEFAULT_DYNAMIC_PARAMS",
    "all_param_names", "param_bounds",
]
