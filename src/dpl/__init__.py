"""Differentiable physics-based hydrology (dPL + SAC-SMA + Snow17 + Lohmann).

This package mirrors ``src.lstm`` for the DPL family of models. The public
entry point is :func:`build_model` returning a :class:`DPLSacSMA` module.
HUC12 polygons act as the lumped hydrologic unit; SAC-SMA + Snow17 + Hamon
PET runs per HUC12, outputs are routed via Lohmann UH and area-weighted
to the USGS gauge for loss computation.
"""

from __future__ import annotations

from .config import DplConfig, load_dpl_config  # noqa: F401
from .dataset import (  # noqa: F401
    BasinBundle,
    DplDataset,
    collate_basins,
    compute_norm_stats,
    create_folds,
    load_dpl_data,
)
from .evaluate import BasinResult, evaluate_basin, evaluate_fold  # noqa: F401
from .model import DPLSacSMA, build_model  # noqa: F401
from .train import load_checkpoint, train_model  # noqa: F401
