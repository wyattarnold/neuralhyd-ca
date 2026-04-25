"""Training loop for dPL+SAC-SMA.

Mirrors the structure of :mod:`src.lstm.train`: warmup → ReduceLROnPlateau
→ optional SWA finetune.  Loss is blended MSE + log-MSE on the basin
flow, evaluated only on the post-warmup window so SAC-SMA / Snow-17 state
has time to stabilise.
"""

from __future__ import annotations

import math
import random
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.lstm.loss import (
    compute_fehv, compute_fhv, compute_flv, compute_kge, compute_nse,
    extreme_ramp_weight,
)

from .config import DplConfig
from .dataset import BasinBundle, DplDataset, collate_basins
from .model import DPLSacSMA


def _to_device(batch: dict, device: torch.device) -> dict:
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device)
        elif isinstance(v, dict):
            out[k] = {kk: vv.to(device) if isinstance(vv, torch.Tensor) else vv for kk, vv in v.items()}
        else:
            out[k] = v
    return out


def _basin_loss_weights(targets: torch.Tensor, exponent: float) -> torch.Tensor:
    """Per-basin weight 1 / std(target)**exponent, normalised to mean 1."""
    var = targets.var(dim=-1).clamp_min(0.1)
    w = 1.0 / var.pow(exponent / 2.0)
    w = w / w.mean().clamp_min(1e-6)
    return w


def _blended_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    log_lambda: float,
    log_eps: float,
    sample_weights: torch.Tensor | None = None,
    time_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    diff = (pred - target) ** 2
    if log_lambda > 0:
        log_diff = (torch.log(pred.clamp_min(0.0) + log_eps)
                    - torch.log(target.clamp_min(0.0) + log_eps)) ** 2
        per_step = diff + log_lambda * log_diff
    else:
        per_step = diff
    if time_weights is not None:
        # Normalised weighted mean over time so the loss magnitude is
        # independent of the ramp's overall scale.
        tw = time_weights
        per_basin = (per_step * tw).sum(dim=-1) / tw.sum(dim=-1).clamp_min(1e-8)
    else:
        per_basin = per_step.mean(dim=-1)
    if sample_weights is None:
        return per_basin.mean()
    return (per_basin * sample_weights).sum() / sample_weights.sum().clamp_min(1e-8)


# ---------------------------------------------------------------------------
# Train / eval loops
# ---------------------------------------------------------------------------


def train_epoch(
    model: DPLSacSMA,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    config: DplConfig,
    device: torch.device,
) -> float:
    model.train()
    total = 0.0
    n = 0
    warmup = config.warmup_steps
    bf_lambda = float(getattr(config, "baseflow_aux_lambda", 0.0))
    peak_boost = float(getattr(config, "extreme_peak_boost", 1.0))
    for batch in loader:
        batch = _to_device(batch, device)
        out = model(batch["x_dyn"], batch["x_static"], batch["forcing"], batch["aux"])
        q_pred = out["q_gauge"]                    # (B, T_full)
        y = batch["y"]                              # (B, T_full)
        # Drop warmup window
        q_pred = q_pred[:, warmup:]
        y = y[:, warmup:]
        # Per-basin sample weight
        weights = _basin_loss_weights(y, config.basin_loss_weight_exponent)
        # Per-timestep peak ramp (1.0 → peak_boost across q_start → q_top).
        # Disabled when peak_boost ≤ 1.0 to keep pure-MSE behaviour.
        time_weights: torch.Tensor | None = None
        if peak_boost > 1.0 and "extreme_qs" in batch:
            q_start = batch["extreme_qs"][:, 0:1]   # (B, 1) broadcasts over T
            q_ramp  = batch["extreme_qs"][:, 1:2]
            time_weights = extreme_ramp_weight(y, q_start, q_ramp, peak_boost)
        loss = _blended_loss(
            q_pred, y, config.log_loss_lambda, config.log_loss_epsilon,
            sample_weights=weights, time_weights=time_weights,
        )

        # Auxiliary baseflow supervision (Song et al. 2024).  Compares the
        # SAC-SMA aggregated baseflow to the Lyne–Hollick separated
        # baseflow on observed flow, both in raw mm/day.  Mask out NaN
        # entries that arose from data gaps in the observed flow.
        if bf_lambda > 0.0 and "q_base_gauge" in out and "y_base" in batch:
            q_base = out["q_base_gauge"][:, warmup:]
            y_base = batch["y_base"][:, warmup:]
            valid = torch.isfinite(y_base)
            if valid.any():
                diff_b = (q_base - y_base) ** 2
                diff_b = torch.where(valid, diff_b, torch.zeros_like(diff_b))
                # Weighted mean over valid entries, then per-basin weight
                denom = valid.float().sum(dim=-1).clamp_min(1.0)
                per_basin_b = diff_b.sum(dim=-1) / denom
                bf_loss = (per_basin_b * weights).sum() / weights.sum().clamp_min(1e-8)
                if torch.isfinite(bf_loss):
                    loss = loss + bf_lambda * bf_loss

        if not torch.isfinite(loss):
            continue
        optimizer.zero_grad()
        loss.backward()
        if config.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()
        total += float(loss.detach())
        n += 1
    return total / max(n, 1)


@torch.no_grad()
def validate_epoch(
    model: DPLSacSMA,
    bundles: dict[int, BasinBundle],
    val_ids: list[int],
    norm_stats: dict,
    config: DplConfig,
    device: torch.device,
    val_ds: DplDataset | None = None,
) -> float:
    """Loss on the held-out basins, averaged over their full record."""
    model.eval()
    ds = val_ds if val_ds is not None else DplDataset(
        bundles, val_ids, norm_stats, config, is_train=False
    )
    total = 0.0
    n = 0
    for i in range(len(ds)):
        sample = ds[i]
        batch = collate_basins([sample])
        batch = _to_device(batch, device)
        out = model(batch["x_dyn"], batch["x_static"], batch["forcing"], batch["aux"])
        q_pred = out["q_gauge"][:, config.warmup_steps:]
        y = batch["y"][:, config.warmup_steps:]
        loss = _blended_loss(q_pred, y, config.log_loss_lambda, config.log_loss_epsilon)
        if torch.isfinite(loss):
            total += float(loss)
            n += 1
    return total / max(n, 1)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def train_model(
    config: DplConfig,
    bundles: dict[int, BasinBundle],
    train_ids: list[int],
    val_ids: list[int],
    norm_stats: dict,
    out_dir: Path,
    device: torch.device,
    *,
    n_dynamic: int,
    n_static: int,
    log_fn=print,
) -> dict:
    """Run full training schedule for one fold.  Returns history dict."""
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)

    from .model import build_model
    model = build_model(config, n_dynamic=n_dynamic, n_static=n_static).to(device)

    train_ds = DplDataset(bundles, train_ids, norm_stats, config, is_train=True)
    # Pre-build the val dataset so its per-bundle cache is allocated once,
    # not rebuilt every epoch.
    val_ds = DplDataset(bundles, val_ids, norm_stats, config, is_train=False)
    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=collate_basins,
        drop_last=False,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(config.num_workers > 0),
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5,
        patience=max(1, config.patience // 2),
    )

    best_val = math.inf
    best_state: dict | None = None
    epochs_no_improve = 0
    history = {"train": [], "val": [], "lr": []}

    for epoch in range(1, config.num_epochs + 1):
        # Linear warmup of LR for the first warmup_epochs
        if epoch <= config.warmup_epochs:
            lr_scale = epoch / max(1, config.warmup_epochs)
            for pg in optimizer.param_groups:
                pg["lr"] = config.learning_rate * lr_scale

        t0 = time.time()
        tr = train_epoch(model, train_loader, optimizer, config, device)
        va = validate_epoch(model, bundles, val_ids, norm_stats, config, device, val_ds=val_ds)
        cur_lr = optimizer.param_groups[0]["lr"]
        history["train"].append(tr)
        history["val"].append(va)
        history["lr"].append(cur_lr)
        log_fn(f"epoch {epoch:03d} | train {tr:.4f} | val {va:.4f} | lr {cur_lr:.2e} | {time.time()-t0:.1f}s")

        if epoch > config.warmup_epochs:
            scheduler.step(va)

        if va < best_val - config.min_delta:
            best_val = va
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
            torch.save(
                {"model_state_dict": best_state, "norm_stats": norm_stats, "config": config.__dict__},
                out_dir / "best_model.pt",
            )
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= config.patience:
                log_fn(f"early stopping at epoch {epoch} (best val {best_val:.4f})")
                break

    # Optional SWA fine-tune
    if config.use_swa and best_state is not None:
        log_fn("starting SWA phase")
        from torch.optim.swa_utils import AveragedModel
        model.load_state_dict(best_state)
        swa_model = AveragedModel(model)
        for pg in optimizer.param_groups:
            pg["lr"] = config.swa_lr
        for s_ep in range(1, config.swa_patience + 1):
            tr = train_epoch(model, train_loader, optimizer, config, device)
            swa_model.update_parameters(model)
            log_fn(f"swa epoch {s_ep:03d} | train {tr:.4f}")
        # Save SWA model under separate file
        torch.save(
            {"model_state_dict": swa_model.module.state_dict(), "norm_stats": norm_stats, "config": config.__dict__},
            out_dir / "swa_model.pt",
        )

    return history


def load_checkpoint(path: Path, model: torch.nn.Module, device: torch.device) -> dict:
    obj = torch.load(path, map_location=device, weights_only=False)
    if isinstance(obj, dict) and "model_state_dict" in obj:
        model.load_state_dict(obj["model_state_dict"])
        return obj.get("norm_stats", {})
    model.load_state_dict(obj)
    return {}
