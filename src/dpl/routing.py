"""Lohmann hillslope unit hydrograph (gamma-distribution form).

The gamma-distribution HRU UH from :file:`rout_lohmann.m` is computed
analytically in daily increments using the regularised incomplete gamma
function (:func:`torch.special.gammainc`).  This is fully differentiable
in both the shape parameter ``N`` and the mean travel time ``tau``.

The Saint-Venant river-routing UH from the original MATLAB depends on
per-HUC12 channel travel distance (``flowlen``), which is deferred to a
follow-up data-prep step.  With ``flowlen=0`` the river UH collapses to
delta(t=0) and only the hillslope UH is applied — implemented here.

Surface runoff and baseflow are routed separately (the original treats
baseflow with an identity HRU UH, so we apply the gamma UH only to the
direct-runoff component).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

UH_DAYS: int = 12  # support window of the daily HRU UH
_INTRA_DAY_STEPS: int = 24  # Riemann-sum density inside each daily bin


def gamma_uh(
    shape: torch.Tensor,
    tau: torch.Tensor,
    n_days: int = UH_DAYS,
    intra_day: int = _INTRA_DAY_STEPS,
) -> torch.Tensor:
    """Daily Lohmann HRU UH from a Gamma(shape, tau/shape) distribution.

    Computed by Riemann-summing the gamma pdf inside each daily bin —
    fully differentiable in both ``shape`` and ``tau``.  This matches
    the integration approach used in :file:`rout_lohmann.m`.

    Parameters
    ----------
    shape : (N,) tensor of UH gamma shape parameters (>0).
    tau   : (N,) tensor of mean travel times in days (>0).
    n_days : int, support window in days.
    intra_day : int, sub-bin grid resolution.

    Returns
    -------
    (N, n_days) tensor — rows sum to (approximately) 1.0.
    """
    device, dtype = shape.device, shape.dtype
    # Cell centres: (n_days * intra_day,) values in (0, n_days)
    n = n_days * intra_day
    dx = 1.0 / intra_day                 # day units
    centres = (torch.arange(n, device=device, dtype=dtype) + 0.5) * dx  # (n,)

    scale = (tau / shape).clamp_min(1e-3).unsqueeze(-1)                 # (N, 1)
    a = shape.unsqueeze(-1)                                             # (N, 1)
    z = centres.unsqueeze(0) / scale                                    # (N, n)

    log_pdf = -torch.lgamma(a) + (a - 1.0) * torch.log(z.clamp_min(1e-12)) - z - torch.log(scale)
    pdf = torch.exp(log_pdf)                                            # (N, n)
    mass = pdf * dx                                                     # (N, n)
    uh = mass.view(-1, n_days, intra_day).sum(dim=-1)                   # (N, n_days)
    uh = uh / uh.sum(dim=-1, keepdim=True).clamp_min(1e-6)
    return uh


def apply_uh(flow: torch.Tensor, uh: torch.Tensor) -> torch.Tensor:
    """Causal convolution of per-unit flow with per-unit UH.

    Parameters
    ----------
    flow : (N, T) tensor of inflow.
    uh   : (N, K) tensor of unit hydrograph.

    Returns
    -------
    (N, T) routed flow tensor.
    """
    N, T = flow.shape
    K = uh.shape[-1]
    # Pad K-1 zeros on the left so output length = T (causal).
    flow_p = F.pad(flow, (K - 1, 0)).unsqueeze(0)            # (1, N, T+K-1)
    # Build a depthwise kernel (groups=N): each unit has its own UH.
    # Conv1d kernel shape: (out=N, in/groups=1, K).  Flip UH for cross-correlation.
    kernel = torch.flip(uh, dims=[-1]).unsqueeze(1)          # (N, 1, K)
    out = F.conv1d(flow_p, kernel, groups=N)                 # (1, N, T)
    return out.squeeze(0)


def route(
    surf: torch.Tensor,
    base: torch.Tensor,
    uh_n: torch.Tensor,
    uh_tau: torch.Tensor,
) -> torch.Tensor:
    """Apply Lohmann hillslope UH to the direct-runoff component.

    Baseflow is passed through unchanged (HRU base UH is delta(t=0) in the
    original MATLAB implementation).
    """
    uh = gamma_uh(uh_n, uh_tau)
    direct_routed = apply_uh(surf, uh)
    return direct_routed + base
