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
    Writes a ``log.txt`` per fold with full epoch-by-epoch metrics.
load_checkpoint(path, model, device)
    Load a ``best_model.pt`` checkpoint into *model* in-place and return
    the attached ``norm_stats`` dict.  Handles legacy bare state-dicts.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch.optim.swa_utils import AveragedModel
from torch.utils.data import DataLoader

from .config import Config
from .loss import (
    mse_loss, blended_loss, pathway_auxiliary_loss, extreme_weighted_mse,
    cmal_nll, cmal_crps, cmal_entropy_reg, cmal_scale_reg,
    compute_nse, compute_kge,
)


def _get_cmal_params(model: torch.nn.Module):
    """Retrieve CMAL distribution params from the last forward pass."""
    m = model
    if hasattr(m, "module"):
        m = m.module
    if hasattr(m, "_orig_mod"):
        m = m._orig_mod
    return getattr(m, "_last_cmal_params", None)


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
    use_extreme = config.extreme_weight_max > 1.0
    use_cmal = config.output_type == "cmal"
    cmal_use_crps = use_cmal and config.cmal_loss == "crps"
    cmal_entropy_w = config.cmal_entropy_weight if use_cmal else 0.0
    cmal_scale_w = config.cmal_scale_reg_weight if use_cmal else 0.0
    for x_d, x_s, y, y_comp, _bid, _std, flow_win in loader:
        x_d, x_s, y = x_d.to(device), x_s.to(device), y.to(device)
        if noise_std > 0:
            x_d = x_d + torch.randn_like(x_d) * noise_std
        optimizer.zero_grad(set_to_none=True)
        q_total, q_fast, q_slow = model(x_d, x_s)

        if use_cmal:
            cmal_params = _get_cmal_params(model)
            # Compute per-sample extreme-flow weights (detached)
            if use_extreme:
                frac = ((y.detach() - config.extreme_threshold) / config.extreme_ramp).clamp(0, 1)
                sw = 1.0 + (config.extreme_weight_max - 1.0) * frac
            else:
                sw = None
            if cmal_use_crps:
                primary = cmal_crps(
                    y, *cmal_params,
                    n_samples=config.cmal_crps_n_samples,
                    beta=config.cmal_beta_crps,
                    sample_weights=sw,
                )
            else:
                primary = cmal_nll(y, *cmal_params, sample_weights=sw)
        elif use_extreme:
            mse_term = extreme_weighted_mse(
                q_total, y,
                threshold=config.extreme_threshold,
                max_weight=config.extreme_weight_max,
                ramp=config.extreme_ramp,
            )
            if use_blended:
                log_term = ((torch.log(q_total + config.log_loss_epsilon)
                             - torch.log(y + config.log_loss_epsilon)) ** 2).mean()
                primary = (1 - config.log_loss_lambda) * mse_term + config.log_loss_lambda * log_term
            else:
                primary = mse_term
        else:
            mse_term = mse_loss(q_total, y)
            if use_blended:
                log_term = ((torch.log(q_total + config.log_loss_epsilon)
                             - torch.log(y + config.log_loss_epsilon)) ** 2).mean()
                primary = (1 - config.log_loss_lambda) * mse_term + config.log_loss_lambda * log_term
            else:
                primary = mse_term

        loss = primary
        if use_cmal and (cmal_entropy_w > 0 or cmal_scale_w > 0):
            pi_c, _mu_c, bl_c, br_c = cmal_params
            if cmal_entropy_w > 0:
                loss = loss + cmal_entropy_w * cmal_entropy_reg(pi_c)
            if cmal_scale_w > 0:
                loss = loss + cmal_scale_w * cmal_scale_reg(bl_c, br_c)
        if use_aux:
            y_comp = y_comp.to(device)
            y_fast_lh = y_comp[:, 0]
            y_slow_lh = y_comp[:, 1]
            loss = loss + config.aux_loss_weight * pathway_auxiliary_loss(
                q_fast, q_slow,
                y_fast_lh, y_slow_lh,
                peak_asymmetry=config.aux_peak_asymmetry,
                y_total_norm=y if use_extreme else None,
                extreme_threshold=config.extreme_threshold,
                extreme_peak_boost=config.extreme_peak_boost,
                extreme_ramp=config.extreme_ramp,
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
    config: Config | None = None,
) -> tuple[float, float, float]:
    """Returns (loss, median_nse, median_kge) computed per basin then aggregated.

    When *config* indicates CMAL output, the loss is CMAL NLL; otherwise MSE.
    """
    model.eval()
    total_loss = torch.tensor(0.0, device=device)
    n = 0
    use_cmal = config is not None and config.output_type == "cmal"
    cmal_use_crps = use_cmal and config is not None and config.cmal_loss == "crps"
    # Accumulate per-basin predictions and observations
    basin_pred: dict[int, list] = {}
    basin_obs: dict[int, list] = {}
    for x_d, x_s, y, _ycomp, bids, fstd, _fw in loader:
        x_d, x_s, y = x_d.to(device), x_s.to(device), y.to(device)
        q_total, _, _ = model(x_d, x_s)
        if use_cmal:
            cmal_params = _get_cmal_params(model)
            if cmal_use_crps:
                total_loss += cmal_crps(
                    y, *cmal_params,
                    n_samples=config.cmal_crps_n_samples,
                    beta=config.cmal_beta_crps,
                )
            else:
                total_loss += cmal_nll(y, *cmal_params)
        else:
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


# ---------------------------------------------------------------------------
# Gate diagnostics (MoE models)
# ---------------------------------------------------------------------------


def _get_gate_stats(model: torch.nn.Module) -> dict | None:
    """Extract tau and last-batch mean π/blend from gated/moe models.

    Returns None for models without a gating network.
    """
    # Unwrap torch.compile / AveragedModel wrappers
    m = model
    if hasattr(m, "module"):
        m = m.module
    if hasattr(m, "_orig_mod"):
        m = m._orig_mod

    n_out = getattr(m, "n_gate_outputs", 0)
    if n_out == 0:
        return None
    stats: dict = {}
    if hasattr(m, "tau"):
        stats["tau"] = m.tau.item()
    if hasattr(m, "_last_pi"):
        pi = m._last_pi
        for i in range(pi.shape[0]):
            stats[f"pi_{i}"] = pi[i].item()
    else:
        # Before the first forward pass, fill with NaN so columns exist
        for i in range(n_out):
            stats[f"pi_{i}"] = float("nan")
    return stats


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

    A ``log.txt`` file is written to the fold directory with per-epoch
    metrics (train loss, val loss, NSE, KGE, LR, and gate diagnostics).
    """
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.learning_rate, weight_decay=config.weight_decay,
    )

    # Warmup → cosine annealing
    # When warmup_epochs=0 (e.g. fine-tuning pretrained weights), skip the
    # warmup and use bare cosine to avoid SequentialLR edge cases.
    if config.warmup_epochs > 0:
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
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.num_epochs,
            eta_min=1e-5,
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
    log_path = ckpt_dir / "log.txt"

    # Determine whether model has a gating network (for extra columns)
    has_gate = _get_gate_stats(model) is not None

    best_val = float("inf")
    wait = 0
    swa_active = False

    if config.output_type == "cmal":
        _loss_tag = config.cmal_loss          # "nll" or "crps"
        _loss_label = config.cmal_loss.upper()
    else:
        _loss_tag = "mse"
        _loss_label = "MSE"

    history: dict = {
        "epoch": [], "phase": [], "lr": [],
        f"train_{_loss_tag}": [], "train_total": [],
        f"val_{_loss_tag}": [], "val_nse": [], "val_kge": [],
    }
    if has_gate:
        _init_gate = _get_gate_stats(model)
        if "tau" in _init_gate:
            history["tau"] = []
        # Pre-populate pi column names from initial stats
        gate_pi_keys = [k for k in sorted(_init_gate) if k.startswith("pi_")]
        for k in gate_pi_keys:
            history[k] = []

    # Open log file — write header, then append each epoch
    log_fh = open(log_path, "w")
    _log_columns = [
        "epoch", "phase", "lr",
        f"train_{_loss_tag}", "train_total",
        f"val_{_loss_tag}", "val_nse", "val_kge",
    ]
    if has_gate:
        if "tau" in _init_gate:
            _log_columns.append("tau")
        _log_columns.extend(gate_pi_keys)

    log_fh.write("\t".join(_log_columns) + "\n")
    log_fh.flush()

    def _record_epoch(epoch: int, phase: str, lr: float,
                      train_mse: float, train_total: float,
                      val_loss: float, val_nse: float, val_kge: float,
                      active_model: torch.nn.Module) -> None:
        """Append one row to history, log file, and console."""
        history["epoch"].append(epoch)
        history["phase"].append(phase)
        history["lr"].append(lr)
        history[f"train_{_loss_tag}"].append(train_mse)
        history["train_total"].append(train_total)
        history[f"val_{_loss_tag}"].append(val_loss)
        history["val_nse"].append(val_nse)
        history["val_kge"].append(val_kge)

        # gate diagnostics
        gate_str = ""
        if has_gate:
            gs = _get_gate_stats(active_model)
            if "tau" in gs:
                history["tau"].append(gs["tau"])
            for k in gate_pi_keys:
                history[k].append(gs.get(k, float("nan")))
            pi_vals = " ".join(f"{gs.get(k, 0):.3f}" for k in gate_pi_keys)
            tau_str = f"τ={gs['tau']:.4f} " if "tau" in gs else ""
            gate_str = f" | {tau_str}π=[{pi_vals}]"

        # console
        has_train = not (math.isnan(train_mse) or math.isnan(train_total))
        aux_str = f" | total {train_total:.4f}" if has_train and train_total != train_mse else ""
        train_str = f"{train_mse:.4f}" if has_train else "   ---"
        swa_tag = f" [SWA n={swa_model.n_averaged}]" if swa_model is not None and phase == "swa" else ""
        label = "init " if epoch == 0 else ""
        print(
            f"  {label}Epoch {epoch:>3d}/{config.num_epochs} | "
            f"{_loss_label} train {train_str}{aux_str} | {_loss_label} val {val_loss:.4f} | "
            f"NSE {val_nse:.3f} | KGE {val_kge:.3f} | "
            f"LR {lr:.2e}{gate_str}{swa_tag}"
        )

        # log file
        row = [
            epoch, phase, f"{lr:.6e}",
            f"{train_mse:.6f}", f"{train_total:.6f}",
            f"{val_loss:.6f}", f"{val_nse:.4f}", f"{val_kge:.4f}",
        ]
        if has_gate:
            if "tau" in gs:
                row.append(f"{gs['tau']:.6f}")
            for k in gate_pi_keys:
                row.append(f"{gs.get(k, float('nan')):.6f}")
        log_fh.write("\t".join(str(v) for v in row) + "\n")
        log_fh.flush()

    try:
        # ---- Epoch 0: cold (random-weight) performance ----
        val_loss_0, val_nse_0, val_kge_0 = validate_epoch(model, val_loader, device, config)
        _record_epoch(0, "init", 0.0, float("nan"), float("nan"),
                       val_loss_0, val_nse_0, val_kge_0, model)

        for epoch in range(1, config.num_epochs + 1):
            train_mse, train_total = train_epoch(
                model, train_loader, optimizer, device, config,
            )

            if swa_active:
                swa_model.update_parameters(model)
                swa_scheduler.step()
                val_loss, val_nse, val_kge = validate_epoch(swa_model, val_loader, device, config)
            else:
                scheduler.step()
                val_loss, val_nse, val_kge = validate_epoch(model, val_loader, device, config)

            lr_now = optimizer.param_groups[0]["lr"]
            phase = "swa" if swa_active else "train"
            active_model = swa_model if swa_active else model

            _record_epoch(epoch, phase, lr_now, train_mse, train_total,
                          val_loss, val_nse, val_kge, active_model)

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
                    log_fh.write(f"# SWA activated at epoch {epoch}\n")
                    log_fh.flush()
                else:
                    phase_label = "SWA converged" if swa_active else "Early stopping"
                    print(f"  {phase_label} at epoch {epoch}")
                    log_fh.write(f"# {phase_label} at epoch {epoch}\n")
                    log_fh.flush()
                    break
    finally:
        log_fh.close()

    # Return final model
    if swa_active:
        _save_checkpoint(swa_model.module.state_dict(), ckpt_path, norm_stats)
        load_checkpoint(ckpt_path, model, device)
        print(f"  SWA model saved (averaged over {swa_model.n_averaged} snapshots)")
    else:
        load_checkpoint(ckpt_path, model, device)

    return model, history
