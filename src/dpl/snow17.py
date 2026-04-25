"""Differentiable port of NWS Snow-17 (Anderson 1976).

Algorithm follows Sungwook Wi's MATLAB implementation
(:file:`snow_snow17_fix.m`).  All conditional branches are expressed as
``torch.where`` operations so gradients flow smoothly through state
transitions.

The model runs as a sequential time loop; per-step state is shape
``(N,)`` where ``N`` is the number of lumped units (HUC12 × Nmul) in
the batch, and parameters are broadcast across time.

Parameters (per unit)
---------------------
SCF, PXTEMP, MFMAX, MFMIN, UADJ, MBASE, TIPM, PLWHC, NMF, DAYGM
(see :data:`src.dpl.config.SNOW17_PARAMS`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass
class Snow17State:
    """Per-unit Snow-17 state vector."""

    w_i: torch.Tensor      # accumulated water-equivalent of ice (mm)
    w_q: torch.Tensor      # liquid water held in pack (mm)
    ati: torch.Tensor      # antecedent temperature index (deg C)
    deficit: torch.Tensor  # heat deficit (mm)

    @classmethod
    def zeros(cls, n: int, device: torch.device, dtype: torch.dtype) -> "Snow17State":
        z = torch.zeros(n, device=device, dtype=dtype)
        return cls(w_i=z.clone(), w_q=z.clone(), ati=z.clone(), deficit=z.clone())


def _seasonal_melt_factor(
    doy: torch.Tensor, mfmax: torch.Tensor, mfmin: torch.Tensor
) -> torch.Tensor:
    # Treat year as 365 days (sub-day differences vs. leap years are negligible
    # for daily-timestep snow physics and avoid year-aware indexing).
    n_mar21 = doy - 80.0
    sv = 0.5 * torch.sin((n_mar21 * 2.0 * math.pi) / 365.0) + 0.5
    # dtt/6 = 24/6 = 4 (hourly→daily factor inherited from MATLAB)
    return 4.0 * (sv * (mfmax - mfmin) + mfmin)


def snow17_step(
    state: Snow17State,
    prcp: torch.Tensor,
    tavg: torch.Tensor,
    doy: torch.Tensor,
    elev_m: torch.Tensor,
    p: dict[str, torch.Tensor],
) -> tuple[Snow17State, torch.Tensor]:
    """Advance Snow-17 by one daily timestep.

    Returns
    -------
    new_state : :class:`Snow17State`
    outflow : (N,) tensor of pack outflow (mm/day) — water leaving the pack.
    """
    SCF, PXTEMP = p["SCF"], p["PXTEMP"]
    MFMAX, MFMIN = p["MFMAX"], p["MFMIN"]
    UADJ, MBASE = p["UADJ"], p["MBASE"]
    TIPM, PLWHC = p["TIPM"], p["PLWHC"]
    NMF, DAYGM = p["NMF"], p["DAYGM"]

    w_i, w_q, ati, deficit = state.w_i, state.w_q, state.ati, state.deficit

    # ---------------- precipitation phase ----------------
    is_snow = (tavg <= PXTEMP).to(prcp.dtype)
    snow = is_snow * prcp
    rain = (1.0 - is_snow) * prcp

    pn = snow * SCF
    w_i = w_i + pn

    # ---------------- non-melt energy exchange ----------------
    mf = _seasonal_melt_factor(doy, MFMAX, MFMIN)
    t_snow_new = torch.minimum(tavg, torch.zeros_like(tavg))
    delta_hd_snow = -(t_snow_new * pn) / 160.0  # 80 cal/g / 0.5 cal/g/C = 160
    # dtp/6 = 4
    delta_hd_t = NMF * 4.0 * (mf / MFMAX) * (ati - t_snow_new)

    # Update ATI: if pn>1.5*dtp(=36mm) reset to t_snow_new, else exponential filter.
    tipm_dtt = 1.0 - (1.0 - TIPM) ** 4.0
    ati_new = ati + tipm_dtt * (tavg - ati)
    ati = torch.where(pn > 36.0, t_snow_new, ati_new)
    ati = torch.minimum(ati, torch.zeros_like(ati))

    # ---------------- melt ----------------
    t_rain = tavg.clamp_min(0.0)
    # Rain-on-snow melt
    stefan = 6.12e-10
    e_sat = 2.7489e8 * torch.exp(-4278.63 / (tavg + 242.792))
    elev_h = elev_m / 100.0  # hundreds of metres
    p_atm = 33.86 * (29.9 - 0.335 * elev_h + 0.00022 * elev_h.clamp_min(0.0) ** 2.4)
    # dtp = 24
    term1 = stefan * 24.0 * ((tavg + 273.0) ** 4 - 273.0 ** 4)
    term2 = 0.0125 * rain * t_rain
    term3 = 8.5 * UADJ * 4.0 * ((0.9 * e_sat - 6.11) + 0.00057 * p_atm * tavg)
    melt_ros = (term1 + term2 + term3).clamp_min(0.0)
    melt_nonrain = (mf * (tavg - MBASE) * (24.0 / 24.0) + 0.0125 * rain * t_rain).clamp_min(0.0)

    is_ros = (rain > (0.25 * 24.0)).to(prcp.dtype)
    can_nonrain_melt = (1.0 - is_ros) * (tavg > MBASE).to(prcp.dtype)
    melt = is_ros * melt_ros + can_nonrain_melt * melt_nonrain

    # ---------------- ripeness / liquid-water bookkeeping ----------------
    deficit = (deficit + delta_hd_snow + delta_hd_t).clamp_min(0.0)

    # Branch A: melt < w_i  (pack remains)
    melt_a = torch.minimum(melt, w_i)  # cannot melt more than is there
    w_i_a = w_i - melt_a
    # cap deficit at 0.33*w_i_a
    deficit_a = torch.minimum(deficit, 0.33 * w_i_a)
    qw = melt_a + rain
    w_qx = PLWHC * w_i_a

    ripe_thresh = deficit_a + deficit_a * PLWHC + w_qx
    is_ripe = (qw + w_q > ripe_thresh).to(prcp.dtype)
    is_holding = ((qw >= deficit_a) & (qw + w_q <= deficit_a * (1.0 + PLWHC) + w_qx)).to(prcp.dtype)
    is_below = (1.0 - is_ripe) * (1.0 - is_holding)

    e_ripe = qw + w_q - w_qx - deficit_a * (1.0 + PLWHC)
    w_i_a_ripe = w_i_a + deficit_a
    w_q_a_ripe = w_qx
    deficit_a_ripe = torch.zeros_like(deficit_a)

    w_i_a_hold = w_i_a + deficit_a
    w_q_a_hold = w_q + qw - deficit_a
    deficit_a_hold = torch.zeros_like(deficit_a)

    w_i_a_below = w_i_a + qw
    deficit_a_below = (deficit_a - qw).clamp_min(0.0)

    e_a = is_ripe * e_ripe.clamp_min(0.0)
    w_i_a_new = is_ripe * w_i_a_ripe + is_holding * w_i_a_hold + is_below * w_i_a_below
    w_q_a_new = is_ripe * w_q_a_ripe + is_holding * w_q_a_hold + is_below * w_q
    deficit_a_new = is_ripe * deficit_a_ripe + is_holding * deficit_a_hold + is_below * deficit_a_below

    # Branch B: melt >= w_i  (pack fully melts)
    qw_b = w_i + w_q + rain
    e_b = qw_b
    w_i_b = torch.zeros_like(w_i)
    w_q_b = torch.zeros_like(w_q)
    deficit_b = deficit  # untouched in MATLAB branch

    branch_a = (melt < w_i).to(prcp.dtype)
    branch_b = 1.0 - branch_a

    w_i = branch_a * w_i_a_new + branch_b * w_i_b
    w_q = branch_a * w_q_a_new + branch_b * w_q_b
    deficit = branch_a * deficit_a_new + branch_b * deficit_b
    e = branch_a * e_a + branch_b * e_b

    # If deficit dropped to zero, reset ATI
    ati = torch.where(deficit <= 1e-9, torch.zeros_like(ati), ati)

    # ---------------- constant ground melt ----------------
    has_pack = (w_i > DAYGM).to(prcp.dtype)
    gmwlos = (DAYGM / w_i.clamp_min(1e-6)) * w_q
    gmslos = DAYGM
    gmro_pack = gmwlos + gmslos
    gmro_thaw = w_i + w_q
    w_i_pack = w_i - gmslos
    w_q_pack = w_q - gmwlos
    w_i = has_pack * w_i_pack
    w_q = has_pack * w_q_pack
    e = e + has_pack * gmro_pack + (1.0 - has_pack) * gmro_thaw

    # In the thaw branch swe == 0; w_i and w_q already zero in that branch.
    new_state = Snow17State(w_i=w_i, w_q=w_q, ati=ati, deficit=deficit)
    return new_state, e.clamp_min(0.0)


def run_snow17(
    prcp: torch.Tensor,
    tavg: torch.Tensor,
    doy: torch.Tensor,
    elev_m: torch.Tensor,
    params: dict[str, torch.Tensor],
    state: Snow17State | None = None,
) -> tuple[torch.Tensor, Snow17State]:
    """Run Snow-17 over a full forcing window.

    Parameters
    ----------
    prcp, tavg, doy : (N, T) tensors
    elev_m : (N,) tensor
    params : dict of (N,) (or (N,T) for time-varying) tensors
    state : optional initial Snow17State

    Returns
    -------
    outflow : (N, T) pack outflow tensor
    final_state : Snow17State
    """
    N, T = prcp.shape
    if state is None:
        state = Snow17State.zeros(N, prcp.device, prcp.dtype)
    # Pre-split params; Snow-17 currently has no dynamic params in the
    # default config, so this is a no-op in the common case.
    static_p: dict[str, torch.Tensor] = {}
    dynamic_p: dict[str, torch.Tensor] = {}
    for k, v in params.items():
        if v.ndim >= 2:
            dynamic_p[k] = v
        else:
            static_p[k] = v
    p_cur: dict[str, torch.Tensor] = dict(static_p)

    out = torch.empty_like(prcp)
    for t in range(T):
        for k, v in dynamic_p.items():
            p_cur[k] = v[..., t]
        state, e_t = snow17_step(
            state, prcp[:, t], tavg[:, t], doy[:, t], elev_m, p_cur
        )
        out[:, t] = e_t
    return out, state
