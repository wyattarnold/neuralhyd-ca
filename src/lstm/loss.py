"""Training loss and scalar evaluation metrics.

Key exports
-----------
mse_loss(q_pred, q_target)
    Mean-squared error used during training.  Operates on normalised
    flow values (divided by per-basin std).
compute_nse(obs, sim)
    Nash-Sutcliffe Efficiency — primary skill score; 1 is perfect,
    values below 0 indicate the model is worse than the mean.
compute_kge(obs, sim)
    Kling-Gupta Efficiency — composite of correlation, bias, and
    variability; 1 is perfect.
compute_fhv(obs, sim)
    Fractional high-flow bias (top 2 % of flows) — penalises peak
    over- or under-prediction.
compute_flv(obs, sim)
    Fractional low-flow bias (bottom 30 % of flows) — penalises
    baseflow over- or under-prediction.

All evaluation functions operate on denormalised mm/day values.
"""

from __future__ import annotations

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Training loss
# ---------------------------------------------------------------------------


def mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean squared error on per-basin-std-normalised flow."""
    return torch.mean((pred - target) ** 2)


def blended_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    lam: float,
    eps: float,
) -> torch.Tensor:
    """Blended MSE + log-space MSE for low-flow sensitivity.

    L = (1-λ)·MSE(Q, Q̂) + λ·MSE(log(Q+ε), log(Q̂+ε))
    """
    sq = (pred - target) ** 2
    if lam > 0:
        log_sq = (torch.log(pred + eps) - torch.log(target + eps)) ** 2
        per_sample = (1 - lam) * sq + lam * log_sq
    else:
        per_sample = sq
    return per_sample.mean()


def pathway_auxiliary_loss(
    q_fast: torch.Tensor, q_slow: torch.Tensor,
    y_fast: torch.Tensor, y_slow: torch.Tensor,
    peak_asymmetry: float = 1.0,
) -> torch.Tensor:
    """Mean of per-pathway MSE losses for component supervision.

    When *peak_asymmetry* > 1, under-prediction of the fast (event)
    pathway is penalised more heavily than over-prediction.  This
    encourages the model to capture peak flows rather than miss them.
    """
    fast_err = (q_fast - y_fast) ** 2
    if peak_asymmetry != 1.0:
        under = q_fast < y_fast  # model missed the peak
        fast_err = torch.where(under, fast_err * peak_asymmetry, fast_err)
    return (torch.mean(fast_err) + torch.mean((q_slow - y_slow) ** 2)) / 2


# ---------------------------------------------------------------------------
# Evaluation metrics  (numpy, on raw / denormalised values)
# ---------------------------------------------------------------------------


def compute_nse(obs: np.ndarray, pred: np.ndarray) -> float:
    """Nash–Sutcliffe Efficiency."""
    obs, pred = np.asarray(obs, dtype=np.float64), np.asarray(pred, dtype=np.float64)
    denom = np.sum((obs - obs.mean()) ** 2)
    if denom < 1e-12:
        return float("nan")
    return float(1.0 - np.sum((obs - pred) ** 2) / denom)


def compute_kge(obs: np.ndarray, pred: np.ndarray) -> float:
    """Kling–Gupta Efficiency (2009 formulation)."""
    obs, pred = np.asarray(obs, dtype=np.float64), np.asarray(pred, dtype=np.float64)
    if len(obs) < 2 or obs.std() < 1e-12:
        return float("nan")
    r = np.corrcoef(obs, pred)[0, 1]
    beta = pred.mean() / (obs.mean() + 1e-10)
    gamma = (pred.std() / (pred.mean() + 1e-10)) / (obs.std() / (obs.mean() + 1e-10))
    return float(1.0 - np.sqrt((r - 1) ** 2 + (beta - 1) ** 2 + (gamma - 1) ** 2))


def compute_fhv(obs: np.ndarray, pred: np.ndarray, h: float = 0.02) -> float:
    """Peak flow bias (%BiasFHV) on the upper *h* fraction of the FDC."""
    obs, pred = np.asarray(obs, dtype=np.float64), np.asarray(pred, dtype=np.float64)
    idx = np.argsort(obs)[::-1]
    n = max(1, int(np.ceil(h * len(obs))))
    obs_h, pred_h = obs[idx[:n]], pred[idx[:n]]
    denom = obs_h.sum()
    if denom < 1e-12:
        return float("nan")
    return float((pred_h.sum() - denom) / denom * 100)


def compute_flv(obs: np.ndarray, pred: np.ndarray, l: float = 0.3) -> float:
    """Low flow bias (%BiasFLV) on the lower *l* fraction of the FDC."""
    obs, pred = np.asarray(obs, dtype=np.float64), np.asarray(pred, dtype=np.float64)
    idx = np.argsort(obs)
    n = max(1, int(np.ceil(l * len(obs))))
    obs_l, pred_l = obs[idx[:n]], pred[idx[:n]]
    # guard against log(0)
    eps = 1e-8
    obs_l = np.maximum(obs_l, eps)
    pred_l = np.maximum(pred_l, eps)
    log_obs = np.log(obs_l)
    log_pred = np.log(pred_l)
    denom = (log_obs - log_obs[-1]).sum()
    if abs(denom) < 1e-12:
        return float("nan")
    numer = (log_pred - log_pred[-1]).sum() - denom
    return float(-1.0 * numer / denom * 100)
