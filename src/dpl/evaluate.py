"""Per-basin evaluation for dPL+SAC-SMA."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from src.lstm.loss import compute_fehv, compute_fhv, compute_flv, compute_kge, compute_nse

from .config import DplConfig
from .dataset import BasinBundle, DplDataset, collate_basins
from .model import DPLSacSMA


@dataclass
class BasinResult:
    basin_id: int
    tier: int
    nse: float
    kge: float
    fhv: float
    fehv: float
    flv: float
    timeseries: dict[str, np.ndarray]


@torch.no_grad()
def evaluate_basin(
    model: DPLSacSMA,
    bundle: BasinBundle,
    norm_stats: dict,
    config: DplConfig,
    device: torch.device,
) -> BasinResult:
    """Run a basin end-to-end and compute the standard metric set."""
    model.eval()
    ds = DplDataset({bundle.basin_id: bundle}, [bundle.basin_id], norm_stats, config, is_train=False)
    sample = ds[0]
    batch = collate_basins([sample])
    batch = {
        k: (v.to(device) if isinstance(v, torch.Tensor) else
            ({kk: vv.to(device) if isinstance(vv, torch.Tensor) else vv for kk, vv in v.items()} if isinstance(v, dict) else v))
        for k, v in batch.items()
    }
    out = model(batch["x_dyn"], batch["x_static"], batch["forcing"], batch["aux"])

    warmup = config.warmup_steps
    # Both prediction and target are already in raw mm/day at the gauge.
    pred_mm = out["q_gauge"][0, warmup:].cpu().numpy()
    obs_mm = batch["y_raw"][0, warmup:].cpu().numpy()

    nse = compute_nse(obs_mm, pred_mm)
    kge = compute_kge(obs_mm, pred_mm)
    fhv = compute_fhv(obs_mm, pred_mm)
    fehv = compute_fehv(obs_mm, pred_mm)
    flv = compute_flv(obs_mm, pred_mm)

    return BasinResult(
        basin_id=bundle.basin_id,
        tier=bundle.tier,
        nse=float(nse), kge=float(kge), fhv=float(fhv), fehv=float(fehv), flv=float(flv),
        timeseries={
            "date": np.array([np.datetime64(d) for d in bundle.dates[warmup:]]),
            "obs_mm": obs_mm.astype(np.float32),
            "pred_mm": pred_mm.astype(np.float32),
            "surf_mm": out["surf"][0, warmup:].cpu().numpy().astype(np.float32) if "surf" in out else None,
            "base_mm": out["base"][0, warmup:].cpu().numpy().astype(np.float32) if "base" in out else None,
        },
    )


def evaluate_fold(
    model: DPLSacSMA,
    bundles: dict[int, BasinBundle],
    val_ids: Iterable[int],
    norm_stats: dict,
    config: DplConfig,
    device: torch.device,
    out_dir: Path | None = None,
) -> list[BasinResult]:
    """Evaluate every held-out basin and optionally write timeseries CSVs."""
    results: list[BasinResult] = []
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
    for bid in val_ids:
        if bid not in bundles:
            continue
        res = evaluate_basin(model, bundles[bid], norm_stats, config, device)
        results.append(res)
        if out_dir is not None:
            ts = res.timeseries
            import pandas as pd
            df = pd.DataFrame({k: v for k, v in ts.items() if v is not None})
            df.to_csv(out_dir / f"{bid}_timeseries.csv", index=False)
    return results
