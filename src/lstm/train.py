"""Training loop, early stopping, LR scheduling, and checkpoint I/O.

Key exports
-----------
train_epoch(model, loader, optimiser, config)
    One forward + backward pass over all batches; returns mean loss.
validate_epoch(model, loader, config)
    Inference-only pass; returns mean validation loss.
train_model(model, train_loader, val_loader, config, norm_stats)
    Full training run with warmup → cosine annealing → optional SWA,
    using patience-based transitions.  Saves ``best_model.pt`` when
    validation loss improves; bundles ``norm_stats`` into the checkpoint.
    Writes a ``log.txt`` per fold with full epoch-by-epoch metrics.
load_checkpoint(path, model, device)
    Load a ``best_model.pt`` checkpoint into *model* in-place and return
    the attached ``norm_stats`` dict.  Handles legacy bare state-dicts.
"""

from __future__ import annotations

import math
from contextlib import nullcontext

import numpy as np
import torch
from torch.optim.swa_utils import AveragedModel
from torch.utils.data import DataLoader

from .config import Config
from .loss import (
    mse_loss, pathway_auxiliary_loss,
    cmal_nll, cmal_crps, cmal_entropy_reg, cmal_scale_reg,
    compute_nse, compute_kge,
)


def pick_device() -> torch.device:
    """Select the best available torch device: MPS → CUDA → CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _amp_context(device: torch.device, config: Config | None):
    """Return an autocast context when BF16 AMP is supported, else nullcontext.

    BF16 autocast only enabled on CUDA.  MPS bf16 support is inconsistent
    across PyTorch versions (especially for LSTMs / log / exp used in
    CMAL), so we deliberately stay in fp32 on MPS to keep numerics
    identical between macOS (M-series) and Windows (CUDA) runs.
    """
    if (
        config is not None
        and getattr(config, "use_amp", False)
        and device.type == "cuda"
        and torch.cuda.is_bf16_supported()
    ):
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    """Strip ``torch.compile`` / ``AveragedModel`` wrappers."""
    m = model
    if hasattr(m, "module"):
        m = m.module
    if hasattr(m, "_orig_mod"):
        m = m._orig_mod
    return m


def _get_cmal_params(model: torch.nn.Module):
    """Retrieve CMAL distribution params from the last forward pass."""
    params = getattr(_unwrap_model(model), "_last_cmal_params", None)
    if params is None:
        raise RuntimeError(
            "CMAL params not found on model. Ensure output_type='cmal' "
            "in config and that forward() has been called."
        )
    return params


def _blended_primary(
    pred: torch.Tensor, target: torch.Tensor, sample_weights: torch.Tensor,
    mse_term: torch.Tensor, config: Config,
) -> torch.Tensor:
    """Combine MSE with log-space MSE using per-sample weights."""
    if config.log_loss_lambda <= 0:
        return mse_term
    log_sq = (torch.log(pred + config.log_loss_epsilon)
              - torch.log(target + config.log_loss_epsilon)) ** 2
    log_term = (log_sq * sample_weights).sum() / (sample_weights.sum() + 1e-8)
    return (1 - config.log_loss_lambda) * mse_term + config.log_loss_lambda * log_term


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
    use_extreme_aux = config.extreme_peak_boost > 1.0 and use_aux
    use_cmal = config.output_type == "cmal"
    cmal_use_crps = use_cmal and config.cmal_loss == "crps"
    cmal_entropy_w = config.cmal_entropy_weight if use_cmal else 0.0
    cmal_scale_w = config.cmal_scale_reg_weight if use_cmal else 0.0
    for x_d, x_s, y, y_comp, _bid, _pmean, basin_w, extreme_qs in loader:
        x_d = x_d.to(device, non_blocking=True)
        x_s = x_s.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        basin_w = basin_w.to(device, non_blocking=True)
        extreme_qs = extreme_qs.to(device, non_blocking=True)
        if noise_std > 0:
            x_d = x_d + torch.randn_like(x_d) * noise_std
        optimizer.zero_grad(set_to_none=True)
        with _amp_context(device, config):
            q_total, q_fast, q_slow = model(x_d, x_s)

            # Primary loss uses basin weights only (no extreme ramp).
            # Extreme-flow sensitivity is handled by:
            #   - log-MSE blend term in _blended_primary (relative errors on small flows)
            #   - aux pathway extreme_peak_boost (dual only)
            if use_cmal:
                cmal_params = _get_cmal_params(model)
                if cmal_use_crps:
                    primary = cmal_crps(
                        y, *cmal_params,
                        n_samples=config.cmal_crps_n_samples,
                        beta=config.cmal_beta_crps,
                        sample_weights=basin_w,
                    )
                else:
                    primary = cmal_nll(y, *cmal_params, sample_weights=basin_w)
            else:
                mse_term = mse_loss(q_total, y, sample_weights=basin_w)
                primary = _blended_primary(q_total, y, basin_w, mse_term, config)

            loss = primary
            if use_cmal and (cmal_entropy_w > 0 or cmal_scale_w > 0):
                pi_c, _mu_c, bl_c, br_c = cmal_params
                if cmal_entropy_w > 0:
                    loss = loss + cmal_entropy_w * cmal_entropy_reg(pi_c)
                if cmal_scale_w > 0:
                    loss = loss + cmal_scale_w * cmal_scale_reg(bl_c, br_c)
            if use_aux:
                y_comp = y_comp.to(device, non_blocking=True)
                y_fast_lh = y_comp[:, 0]
                y_slow_lh = y_comp[:, 1]
                # Per-basin quantile thresholds: [y_q_start, y_q_top]
                # ramp width = y_q_top − y_q_start (floored inside extreme_ramp_weight)
                q_start_b = extreme_qs[:, 0]
                q_top_b = extreme_qs[:, 1]
                ramp_b = (q_top_b - q_start_b)
                loss = loss + config.aux_loss_weight * pathway_auxiliary_loss(
                    q_fast, q_slow,
                    y_fast_lh, y_slow_lh,
                    y_total_norm=y if use_extreme_aux else None,
                    extreme_threshold=q_start_b,
                    extreme_peak_boost=config.extreme_peak_boost,
                    extreme_ramp=ramp_b,
                    sample_weights=basin_w,
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
    for x_d, x_s, y, _ycomp, bids, pmean, basin_w, _extreme_qs in loader:
        x_d = x_d.to(device, non_blocking=True)
        x_s = x_s.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        basin_w = basin_w.to(device, non_blocking=True)
        with _amp_context(device, config):
            q_total, _, _ = model(x_d, x_s)
            if use_cmal:
                cmal_params = _get_cmal_params(model)
                if cmal_use_crps:
                    total_loss += cmal_crps(
                        y, *cmal_params,
                        n_samples=config.cmal_crps_n_samples,
                        beta=config.cmal_beta_crps,
                        sample_weights=basin_w,
                    )
                else:
                    total_loss += cmal_nll(y, *cmal_params, sample_weights=basin_w)
            else:
                total_loss += mse_loss(q_total, y, sample_weights=basin_w)
        n += 1
        # denormalise for NSE/KGE (scale = per-basin precip_mean)
        scale = pmean.to(device, non_blocking=True)
        pred_de = (q_total.float() * scale).cpu().numpy()
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
    m = _unwrap_model(model)

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
    # Dispatch to subbasin-mode epoch functions when requested.
    if getattr(config, "spatial_mode", "gauge") == "subbasin":
        from .subbasin_train import (
            train_epoch_subbasin as _train_epoch_impl,
            validate_epoch_subbasin as _validate_epoch_impl,
        )
    else:
        _train_epoch_impl = train_epoch
        _validate_epoch_impl = validate_epoch

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

    # torch.compile is intentionally not used here: LSTMs already hit a
    # fused cuDNN kernel on CUDA, and Inductor's CUDA backend requires
    # Triton (not shipped with default Windows PyTorch wheels).  On MPS
    # compile support is still limited.  Skipping it keeps behaviour
    # identical across macOS (MPS) and Windows/Linux (CUDA).

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
    log_fh = open(log_path, "w", encoding="utf-8")
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
        val_loss_0, val_nse_0, val_kge_0 = _validate_epoch_impl(model, val_loader, device, config)
        _record_epoch(0, "init", 0.0, float("nan"), float("nan"),
                       val_loss_0, val_nse_0, val_kge_0, model)

        for epoch in range(1, config.num_epochs + 1):
            train_mse, train_total = _train_epoch_impl(
                model, train_loader, optimizer, device, config,
            )

            if swa_active:
                swa_model.update_parameters(model)
                swa_scheduler.step()
                val_loss, val_nse, val_kge = _validate_epoch_impl(swa_model, val_loader, device, config)
            else:
                scheduler.step()
                val_loss, val_nse, val_kge = _validate_epoch_impl(model, val_loader, device, config)

            lr_now = optimizer.param_groups[0]["lr"]
            phase = "swa" if swa_active else "train"
            active_model = swa_model if swa_active else model

            _record_epoch(epoch, phase, lr_now, train_mse, train_total,
                          val_loss, val_nse, val_kge, active_model)

            if epoch_callback is not None:
                epoch_callback(epoch, val_loss)

            # Track improvement using val loss (lower is better).
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
