"""Training loop, early stopping, LR scheduling, and checkpoint I/O.

Key exports
-----------
train_epoch(model, loader, optimiser, config)
    One forward + backward pass over all batches; returns mean loss.
validate_epoch(model, loader, config)
    Inference-only pass; returns mean validation loss.
train_model(model, train_loader, val_loader, config, norm_stats)
    Full training run with ``ReduceLROnPlateau`` scheduling and
    patience-based early stopping.  Saves ``best_model.pt`` when
    validation loss improves; bundles ``norm_stats`` into the checkpoint.
load_checkpoint(path, model, device)
    Load a ``best_model.pt`` checkpoint into *model* in-place and return
    the attached ``norm_stats`` dict.  Handles legacy bare state-dicts.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.optim.swa_utils import AveragedModel
from torch.utils.data import DataLoader

from .config import Config
from .loss import mse_loss, blended_loss, pathway_auxiliary_loss, compute_nse, compute_kge


def train_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    config: Config,
) -> tuple[float, float]:
    """Returns (primary_mse, total_loss_with_aux)."""
    model.train()
    sum_primary = torch.tensor(0.0, device=device)
    sum_total = torch.tensor(0.0, device=device)
    n = 0
    use_aux = config.aux_loss_weight > 0 and config.model_type == "dual"
    noise_std = config.input_noise_std
    use_blended = config.log_loss_lambda > 0
    for x_d, x_s, y, y_comp, _bid, _std in loader:
        x_d, x_s, y = x_d.to(device), x_s.to(device), y.to(device)
        if noise_std > 0:
            x_d = x_d + torch.randn_like(x_d) * noise_std
        optimizer.zero_grad(set_to_none=True)
        q_total, q_fast, q_slow = model(x_d, x_s)
        if use_blended:
            primary = blended_loss(
                q_total, y, config.log_loss_lambda, config.log_loss_epsilon,
            )
        else:
            primary = mse_loss(q_total, y)
        loss = primary
        if use_aux:
            y_comp = y_comp.to(device)
            loss = loss + config.aux_loss_weight * pathway_auxiliary_loss(
                q_fast, q_slow,
                y_comp[:, 0], y_comp[:, 1],
                peak_asymmetry=config.aux_peak_asymmetry,
            )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.grad_clip)
        optimizer.step()
        sum_primary += primary.detach()
        sum_total += loss.detach()
        n += 1
    return sum_primary.item() / max(n, 1), sum_total.item() / max(n, 1)


@torch.no_grad()
def validate_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, float, float]:
    """Returns (mse, median_nse, median_kge) computed per basin then aggregated."""
    model.eval()
    total_loss = torch.tensor(0.0, device=device)
    n = 0
    # Accumulate per-basin predictions and observations
    basin_pred: dict[int, list] = {}
    basin_obs: dict[int, list] = {}
    for x_d, x_s, y, _ycomp, bids, fstd in loader:
        x_d, x_s, y = x_d.to(device), x_s.to(device), y.to(device)
        q_total, _, _ = model(x_d, x_s)
        total_loss += mse_loss(q_total, y)
        n += 1
        # denormalise for NSE/KGE
        scale = fstd.to(device)
        pred_de = (q_total * scale).cpu().numpy()
        obs_de = (y * scale).cpu().numpy()
        bid_np = bids.numpy() if isinstance(bids, torch.Tensor) else bids
        for i, bid in enumerate(bid_np):
            bid = int(bid)
            basin_pred.setdefault(bid, []).append(pred_de[i])
            basin_obs.setdefault(bid, []).append(obs_de[i])
    mse = total_loss.item() / max(n, 1)
    # Per-basin NSE/KGE, then take median
    nses, kges = [], []
    for bid in basin_pred:
        p = np.array(basin_pred[bid])
        o = np.array(basin_obs[bid])
        nse_val = compute_nse(o, p)
        kge_val = compute_kge(o, p)
        if not np.isnan(nse_val):
            nses.append(nse_val)
        if not np.isnan(kge_val):
            kges.append(kge_val)
    median_nse = float(np.median(nses)) if nses else float("nan")
    median_kge = float(np.median(kges)) if kges else float("nan")
    return mse, median_nse, median_kge


def _save_checkpoint(
    state_dict: dict,
    path,
    norm_stats: dict | None = None,
) -> None:
    """Save model weights and (optionally) normalisation statistics."""
    payload: dict = {"model_state_dict": state_dict}
    if norm_stats is not None:
        payload["norm_stats"] = norm_stats
    torch.save(payload, path)


def load_checkpoint(
    path,
    model: torch.nn.Module,
    device: torch.device | str = "cpu",
) -> dict | None:
    """Load model weights and return norm_stats (if present in checkpoint).

    Works with both new-style checkpoints (dict with 'model_state_dict' +
    'norm_stats') and legacy bare state_dict files.
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
        return ckpt.get("norm_stats")
    # Legacy: bare state_dict
    model.load_state_dict(ckpt)
    return None


def train_model(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: Config,
    fold_idx: int,
    device: torch.device,
    epoch_callback: callable | None = None,
    norm_stats: dict | None = None,
) -> tuple[torch.nn.Module, dict]:
    """Train with warmup → cosine → optional SWA. Returns (best model, history).

    Two phases controlled by a single patience counter:
      1. Normal training: warmup then cosine LR.  Best checkpoint saved.
         When patience exhausts → if use_swa, activate SWA; else stop.
      2. SWA phase: fixed low LR, weight averaging every epoch.
         When patience exhausts again → stop.  Final averaged model returned.

    If *epoch_callback* is provided, it is called as
    ``epoch_callback(epoch, val_loss)`` after each validation step.
    The callback may raise an exception (e.g. ``optuna.TrialPruned``)
    to terminate training early.

    If *norm_stats* is provided, it is bundled into the checkpoint so
    that normalisation can be recovered at inference time.
    """
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay,
    )

    # Warmup → cosine annealing
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[
            torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=0.1, total_iters=config.warmup_epochs,
            ),
            torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=config.num_epochs - config.warmup_epochs,
                eta_min=1e-5,
            ),
        ],
        milestones=[config.warmup_epochs],
    )

    # SWA (created lazily on activation)
    swa_model: AveragedModel | None = None
    swa_scheduler: torch.optim.lr_scheduler.LRScheduler | None = None

    # torch.compile for graph-level optimisations (skipped on MPS — limited backend support)
    if device.type != "mps" and hasattr(torch, "compile"):
        model = torch.compile(model)

    ckpt_dir = config.output_dir / f"fold_{fold_idx}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "best_model.pt"

    best_val = float("inf")
    wait = 0
    swa_active = False
    history: dict = {"train_loss": [], "val_loss": []}

    for epoch in range(1, config.num_epochs + 1):
        train_mse, train_total = train_epoch(
            model, train_loader, optimizer, device, config,
        )

        if swa_active:
            swa_model.update_parameters(model)
            swa_scheduler.step()
            val_loss, val_nse, val_kge = validate_epoch(swa_model, val_loader, device)
        else:
            scheduler.step()
            val_loss, val_nse, val_kge = validate_epoch(model, val_loader, device)

        history["train_loss"].append(train_mse)
        history["val_loss"].append(val_loss)

        lr_now = optimizer.param_groups[0]["lr"]
        aux_str = f" | total {train_total:.4f}" if train_total != train_mse else ""
        tag = f" [SWA n={swa_model.n_averaged}]" if swa_active else ""
        print(
            f"  Epoch {epoch:>3d}/{config.num_epochs} | "
            f"MSE train {train_mse:.4f}{aux_str} | MSE val {val_loss:.4f} | "
            f"NSE {val_nse:.3f} | KGE {val_kge:.3f} | LR {lr_now:.2e}{tag}"
        )

        if epoch_callback is not None:
            epoch_callback(epoch, val_loss)

        # Track improvement (require min_delta relative improvement)
        threshold = best_val * (1.0 - config.min_delta)
        if val_loss < threshold:
            best_val = val_loss
            wait = 0
            if not swa_active:
                _save_checkpoint(model.state_dict(), ckpt_path, norm_stats)
        else:
            wait += 1

        # Patience exhausted → transition or stop
        current_patience = config.swa_patience if swa_active else config.patience
        if wait >= current_patience:
            if not swa_active and config.use_swa:
                # Reload best pre-SWA weights and start averaging from there
                load_checkpoint(ckpt_path, model, device)
                swa_model = AveragedModel(model, device=device)
                swa_scheduler = torch.optim.swa_utils.SWALR(
                    optimizer, swa_lr=config.swa_lr, anneal_epochs=2,
                )
                swa_active = True
                best_val = float("inf")  # reset for SWA phase
                wait = 0
                print(f"  >> Val plateau at epoch {epoch} — activating SWA")
            else:
                phase = "SWA converged" if swa_active else "Early stopping"
                print(f"  {phase} at epoch {epoch}")
                break

    # Return final model
    if swa_active:
        _save_checkpoint(swa_model.module.state_dict(), ckpt_path, norm_stats)
        load_checkpoint(ckpt_path, model, device)
        print(f"  SWA model saved (averaged over {swa_model.n_averaged} snapshots)")
    else:
        load_checkpoint(ckpt_path, model, device)

    return model, history
