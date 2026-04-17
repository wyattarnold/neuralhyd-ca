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


def extreme_weighted_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 3.0,
    max_weight: float = 30.0,
    ramp: float = 6.0,
) -> torch.Tensor:
    """MSE with sample weights that ramp up for extreme normalised flows.

    Weight is 1.0 for flows below *threshold*, then linearly ramps to
    *max_weight* over the next *ramp* normalised-flow units.  Weights are
    detached (no gradient through the weight computation).
    """
    sq = (pred - target) ** 2
    frac = ((target.detach() - threshold) / ramp).clamp(0, 1)
    w = 1.0 + (max_weight - 1.0) * frac
    return (sq * w).mean()


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
    y_total_norm: torch.Tensor | None = None,
    extreme_threshold: float = 3.0,
    extreme_peak_boost: float = 12.0,
    extreme_ramp: float = 6.0,
) -> torch.Tensor:
    """Mean of per-pathway MSE losses for component supervision.

    When *peak_asymmetry* > 1, under-prediction of the fast (event)
    pathway is penalised more heavily than over-prediction.  This
    encourages the model to capture peak flows rather than miss them.

    When *y_total_norm* is provided, the fast pathway error is further
    boosted during extreme events: the effective under-prediction penalty
    ramps from *peak_asymmetry* to *peak_asymmetry* × *extreme_peak_boost*
    as normalised total flow rises from *extreme_threshold* to
    *extreme_threshold* + *extreme_ramp*.
    """
    fast_err = (q_fast - y_fast) ** 2
    if peak_asymmetry != 1.0:
        under = q_fast < y_fast  # model missed the peak
        fast_err = torch.where(under, fast_err * peak_asymmetry, fast_err)

    # Event-adaptive boost: further amplify fast-pathway errors during extremes
    if y_total_norm is not None:
        frac = ((y_total_norm.detach() - extreme_threshold) / extreme_ramp).clamp(0, 1)
        event_weight = 1.0 + (extreme_peak_boost - 1.0) * frac
        fast_err = fast_err * event_weight

    return (torch.mean(fast_err) + torch.mean((q_slow - y_slow) ** 2)) / 2


# ---------------------------------------------------------------------------
# CMAL (Countable Mixture of Asymmetric Laplacians)
# ---------------------------------------------------------------------------


def cmal_nll(
    target: torch.Tensor,
    pi: torch.Tensor,
    mu: torch.Tensor,
    b_l: torch.Tensor,
    b_r: torch.Tensor,
    sample_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Negative log-likelihood of a CMAL distribution.

    Parameters
    ----------
    target : (B,) normalised flow
    pi     : (B, K) mixture weights  (sum to 1)
    mu     : (B, K) location params  (positive)
    b_l    : (B, K) left scale       (positive)
    b_r    : (B, K) right scale      (positive)
    sample_weights : (B,) optional per-sample weights (e.g. extreme-flow).
    """
    y = target.unsqueeze(-1)                              # (B, 1)
    diff = y - mu                                         # (B, K)

    # log f(y|μ,b_L,b_R) per component
    log_norm = -torch.log(b_l + b_r)                      # (B, K)
    log_comp = torch.where(
        diff < 0,
        log_norm + diff / b_l,                            # left tail
        log_norm - diff / b_r,                            # right tail
    )

    # log p(y) = logsumexp_k [ log πₖ + log fₖ ]
    log_mixture = torch.logsumexp(
        torch.log(pi + 1e-8) + log_comp, dim=-1,
    )                                                     # (B,)

    nll = -log_mixture                                    # (B,)
    if sample_weights is not None:
        nll = nll * sample_weights
    return nll.mean()


def cmal_crps(
    target: torch.Tensor,
    pi: torch.Tensor,
    mu: torch.Tensor,
    b_l: torch.Tensor,
    b_r: torch.Tensor,
    n_samples: int = 50,
    beta: float = 0.0,
    sample_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Energy-score CRPS for a CMAL distribution.

    Term 1 (E_F[|X − y|]) is computed analytically per component.
    Term 2 (E_F[|X − X′|]) uses reparameterised ALD samples split into
    two independent halves so the estimator is unbiased.

    When *beta* > 0, uses the β-weighted (tw-CRPS) formulation that
    penalises overconfidence: each sample's CRPS is divided by σ^β
    where σ is the predicted spread (mean of b_l + b_r).  This
    prevents the model from shrinking intervals to reduce the raw CRPS.

    CRPS = Term1 − 0.5 · Term2.  Lower is better.

    Parameters
    ----------
    target : (B,) normalised flow observations.
    pi     : (B, K) mixture weights (sum to 1).
    mu     : (B, K) location params (positive).
    b_l    : (B, K) left scale (positive).
    b_r    : (B, K) right scale (positive).
    n_samples : samples per component for the spread term.
    """
    S = b_l + b_r                                          # (B, K)

    # --- Term 1: Σ_k π_k · E_k[|X_k − y|]  (closed form) ------------
    y = target.unsqueeze(-1)                               # (B, 1)
    d = y - mu                                             # (B, K)
    # Use -|d| so the exponent is always ≤ 0 in BOTH branches.
    # torch.where evaluates both branches for all elements — without
    # this, the non-selected branch computes exp(+large) = inf whose
    # gradient poisons the backward pass (inf * 0 = NaN in IEEE 754).
    neg_abs_d = -d.abs()
    E_abs_k = torch.where(
        d >= 0,
        d + (b_l ** 2 - b_r ** 2) / S
            + 2 * b_r ** 2 / S * torch.exp((neg_abs_d / b_r).clamp(min=-50)),
        -d + (b_r ** 2 - b_l ** 2) / S
            + 2 * b_l ** 2 / S * torch.exp((neg_abs_d / b_l).clamp(min=-50)),
    )                                                      # (B, K)
    term1 = (pi * E_abs_k).sum(dim=-1)                     # (B,)

    # --- Term 2: Σ_j Σ_k π_j π_k E[|X_j − X_k|]  (sample-based) ----
    half = max(n_samples // 2, 1)
    u = torch.rand(*pi.shape, 2 * half, device=pi.device)  # (B, K, 2H)
    p_left = (b_l / S).unsqueeze(-1)                       # (B, K, 1)
    mu_e = mu.unsqueeze(-1)
    bl_e = b_l.unsqueeze(-1)
    br_e = b_r.unsqueeze(-1)

    samples = torch.where(
        u < p_left,
        mu_e + bl_e * torch.log((u / p_left).clamp(min=1e-8)),
        mu_e - br_e * torch.log(((1 - u) / (1 - p_left)).clamp(min=1e-8)),
    )                                                      # (B, K, 2H)

    s1 = samples[:, :, :half]                              # (B, K, H)
    s2 = samples[:, :, half:]                              # (B, K, H)
    diff2 = s1.unsqueeze(2) - s2.unsqueeze(1)              # (B, K, K, H)
    E_jk = torch.abs(diff2).mean(dim=-1)                   # (B, K, K)

    pi_outer = pi.unsqueeze(-1) * pi.unsqueeze(-2)         # (B, K, K)
    term2 = (pi_outer * E_jk).sum(dim=(-1, -2))            # (B,)

    crps_per_sample = term1 - 0.5 * term2                   # (B,)

    if beta > 0:
        # Predicted spread: weighted mean of (b_l + b_r) across components
        sigma = (pi * S).sum(dim=-1).clamp(min=1e-6)       # (B,)
        crps_per_sample = crps_per_sample / sigma.pow(beta)

    if sample_weights is not None:
        crps_per_sample = crps_per_sample * sample_weights
    return crps_per_sample.mean()


def cmal_entropy_reg(pi: torch.Tensor) -> torch.Tensor:
    """Negative entropy of mixture weights — penalises component collapse.

    Returns mean negative entropy over the batch.  Add to the total loss
    with a positive weight to encourage uniform component usage.
    """
    return (pi * torch.log(pi + 1e-8)).sum(dim=-1).mean()


def cmal_scale_reg(b_l: torch.Tensor, b_r: torch.Tensor) -> torch.Tensor:
    """Negative mean log-scale — penalises overconfident narrow distributions.

    Returns the negative mean of log(b_l + b_r), which grows as scales
    shrink.  Add to the total loss with a positive weight.
    """
    return -torch.log(b_l + b_r).mean()


def cmal_quantiles_np(
    pi: np.ndarray,
    mu: np.ndarray,
    b_l: np.ndarray,
    b_r: np.ndarray,
    levels: list[float] | None = None,
    n_iter: int = 64,
) -> dict[float, np.ndarray]:
    """Compute quantiles of a CMAL mixture via bisection (numpy).

    Parameters
    ----------
    pi, mu, b_l, b_r : (N, K) arrays — mixture parameters.
    levels : quantile levels (default [0.05, 0.25, 0.5, 0.75, 0.95]).
    n_iter : bisection iterations.

    Returns
    -------
    dict mapping quantile level → (N,) array of quantile values.
    """
    if levels is None:
        levels = [0.05, 0.25, 0.5, 0.75, 0.95]

    def _mixture_cdf(y_col: np.ndarray) -> np.ndarray:
        """CDF of the CMAL mixture evaluated at y_col (N,)."""
        y_exp = y_col[:, None]                             # (N, 1)
        diff = y_exp - mu                                  # (N, K)
        norm_l = b_l / (b_l + b_r)
        norm_r = b_r / (b_l + b_r)
        cdf_k = np.where(
            diff < 0,
            norm_l * np.exp(np.clip(diff / b_l, -50, 0)),
            1.0 - norm_r * np.exp(np.clip(-diff / b_r, -50, 0)),
        )
        return (pi * cdf_k).sum(axis=-1)

    results: dict[float, np.ndarray] = {}
    for q in levels:
        lo = np.zeros(len(pi))
        hi = mu.max(axis=-1) + 8.0 * b_r.max(axis=-1)
        for _ in range(n_iter):
            mid = (lo + hi) * 0.5
            cdf_val = _mixture_cdf(mid)
            lo = np.where(cdf_val < q, mid, lo)
            hi = np.where(cdf_val >= q, mid, hi)
        results[q] = (lo + hi) * 0.5
    return results





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
