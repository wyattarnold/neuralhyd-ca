"""Differentiable port of the Sacramento Soil Moisture Accounting model.

Algorithm follows :file:`sma_sacramento.m` (Sungwook Wi / Steinschneider
Lab port of the NWS Fortran Sacramento) with all conditional branches
replaced by smooth ``torch.where`` selections so gradients flow through
state transitions.

The sub-daily refinement step in the original (``ninc`` based on
``floor(1+0.2*(uzfwc+twx))``) is fixed at a constant ``N_INC=5`` for
differentiability and reproducibility — this matches typical
implementations in the dPL literature.

Parameters (per unit, in canonical order)
-----------------------------------------
UZTWM, UZFWM, LZTWM, LZFPM, LZFSM,
UZK, LZPK, LZSK,
ZPERC, REXP, PFREE,
PCTIM, ADIMP, RIVA, SIDE, RSERV, THETA_C

When ``enable_capillary_rise=True`` an upward flux moves water from the
lower-zone free stores to the upper-zone tension store at rate
``THETA_C * (lzfpc + lzfsc) * (1 - uztwc/UZTWM)`` (Song et al. 2024).
This reduces zero/low-flow bias in arid basins.

When ``implicit_percolation=True`` percolation uses the analytic
implicit-Euler solution ``perc = uzfwc * (1 - exp(-k*dt))`` instead of
the explicit ``tanh(k)`` saturator; mathematically the exact closed-form
solution for the linear depletion ODE over a substep of length ``dt``
with frozen rate ``k``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

# Default sub-daily refinement steps.  Can be overridden via run_sacsma
# kwargs.  Finer (12) follows the Song et al. 2024 implicit-scheme spirit.
N_INC: int = 12


@dataclass
class SacState:
    """Per-unit SAC-SMA state vector (mm)."""

    uztwc: torch.Tensor
    uzfwc: torch.Tensor
    lztwc: torch.Tensor
    lzfsc: torch.Tensor
    lzfpc: torch.Tensor
    adimc: torch.Tensor

    @classmethod
    def init_capacity(cls, p: dict[str, torch.Tensor]) -> "SacState":
        """Initialise storages at their capacities (Shen et al. convention)."""
        return cls(
            uztwc=p["UZTWM"].clone(),
            uzfwc=p["UZFWM"].clone(),
            lztwc=p["LZTWM"].clone(),
            lzfsc=p["LZFSM"].clone(),
            lzfpc=p["LZFPM"].clone(),
            adimc=(p["UZTWM"] + p["LZTWM"]).clone(),
        )


def _smooth_relu(x: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    """Smooth approximation of relu — keeps gradients well-defined at 0."""
    return 0.5 * (x + torch.sqrt(x * x + eps * eps))


def sacsma_step(
    state: SacState,
    pr: torch.Tensor,    # (N,) effective precipitation reaching soil (mm)
    pet: torch.Tensor,   # (N,) potential ET (mm)
    p: dict[str, torch.Tensor],
    *,
    n_inc: int = N_INC,
    enable_capillary_rise: bool = True,
    implicit_percolation: bool = True,
) -> tuple[SacState, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Advance SAC-SMA by one daily timestep.

    Returns
    -------
    new_state : :class:`SacState`
    surf, base, tet : (N,) tensors of surface flow, baseflow, total ET (mm).
    """
    UZTWM, UZFWM = p["UZTWM"], p["UZFWM"]
    LZTWM, LZFPM, LZFSM = p["LZTWM"], p["LZFPM"], p["LZFSM"]
    UZK, LZPK, LZSK = p["UZK"], p["LZPK"], p["LZSK"]
    ZPERC, REXP, PFREE = p["ZPERC"], p["REXP"], p["PFREE"]
    PCTIM, ADIMP, RIVA = p["PCTIM"], p["ADIMP"], p["RIVA"]
    SIDE, RSERV = p["SIDE"], p["RSERV"]
    THETA_C = p.get("THETA_C", torch.zeros_like(p["UZTWM"]))

    uztwc, uzfwc = state.uztwc, state.uzfwc
    lztwc, lzfsc, lzfpc = state.lztwc, state.lzfsc, state.lzfpc
    adimc = state.adimc

    parea = 1.0 - ADIMP - PCTIM
    edmnd = pet
    zero = torch.zeros_like(uztwc)

    # ---------------- ET(1): from upper-zone tension ----------------
    et1_demand = edmnd * uztwc / UZTWM.clamp_min(1e-6)
    et1 = torch.minimum(et1_demand, uztwc)
    uztwc = uztwc - et1
    red = edmnd - et1

    # ---------------- ET(2): from upper-zone free (only if uztwc=0) ----------------
    uztw_empty = (uztwc <= 1e-6).to(uztwc.dtype)
    et2_full = torch.minimum(red, uzfwc)
    et2 = uztw_empty * et2_full
    uzfwc = uzfwc - et2
    red = red - et2

    # If uztwc had water remaining, rebalance UZ tension/free if free is fuller
    uzrat = (uztwc + uzfwc) / (UZTWM + UZFWM).clamp_min(1e-6)
    rebalance = ((1.0 - uztw_empty) * (uztwc / UZTWM.clamp_min(1e-6) < uzfwc / UZFWM.clamp_min(1e-6)).to(uztwc.dtype))
    uztwc_re = UZTWM * uzrat
    uzfwc_re = UZFWM * uzrat
    uztwc = rebalance * uztwc_re + (1.0 - rebalance) * uztwc
    uzfwc = rebalance * uzfwc_re + (1.0 - rebalance) * uzfwc

    # ---------------- ET(3): from lower-zone tension ----------------
    et3 = red * lztwc / (UZTWM + LZTWM).clamp_min(1e-6)
    et3 = torch.minimum(et3, lztwc)
    lztwc = lztwc - et3

    # Resupply from lower free water to lower tension
    saved = RSERV * (LZFPM + LZFSM)
    denom_lz = (LZTWM + LZFPM + LZFSM - saved).clamp_min(1e-6)
    ratlzt = lztwc / LZTWM.clamp_min(1e-6)
    ratlz = (lztwc + lzfpc + lzfsc - saved) / denom_lz
    need_resupply = (ratlzt < ratlz).to(uztwc.dtype)
    delta = (ratlz - ratlzt) * LZTWM
    delta = need_resupply * delta
    lztwc = lztwc + delta
    lzfsc_after = lzfsc - delta
    overdraw = (-lzfsc_after).clamp_min(0.0)
    lzfsc = lzfsc_after.clamp_min(0.0)
    lzfpc = lzfpc - overdraw  # remainder from primary

    # ---------------- Capillary rise (Song et al. 2024) ----------------
    # Upward flux from lower-zone free water to upper-zone tension store
    # when surface is dry.  Rate scales with both LZ availability and the
    # UZ tension deficit.  Reduces zero/low-flow bias in arid basins.
    if enable_capillary_rise:
        deficit_uztw = (1.0 - uztwc / UZTWM.clamp_min(1e-6)).clamp(0.0, 1.0)
        cr_demand = THETA_C * (lzfpc + lzfsc) * deficit_uztw
        total_lzf = (lzfpc + lzfsc).clamp_min(1e-6)
        cr = torch.minimum(cr_demand, lzfpc + lzfsc)
        # Draw proportionally from primary and supplementary
        cr_p = cr * (lzfpc / total_lzf)
        cr_s = cr - cr_p
        lzfpc = (lzfpc - cr_p).clamp_min(0.0)
        lzfsc = (lzfsc - cr_s).clamp_min(0.0)
        uztwc = uztwc + cr
        # Spillover into UZ free if it overshoots
        over_uz = (uztwc - UZTWM).clamp_min(0.0)
        uztwc = uztwc - over_uz
        uzfwc = uzfwc + over_uz

    # ---------------- ET(5): ADIMP zone ----------------
    et5 = et1 + (red + et2) * (adimc - et1 - uztwc) / (UZTWM + LZTWM).clamp_min(1e-6)
    et5_capped = torch.minimum(et5, adimc.clamp_min(0.0))
    adimc = adimc - et5_capped
    et5 = et5_capped * ADIMP  # area-weighted

    # ---------------- partition rainfall above UZ tension ----------------
    twx = pr + uztwc - UZTWM
    has_excess = (twx > 0).to(uztwc.dtype)
    uztwc = has_excess * UZTWM + (1.0 - has_excess) * (uztwc + pr)
    twx = has_excess * twx  # excess above UZ tension capacity

    adimc = adimc + pr - twx

    # Impervious-area runoff
    roimp = pr * PCTIM

    # ---------------- sub-daily SMA loop ----------------
    sbf = torch.zeros_like(uztwc)
    ssur = torch.zeros_like(uztwc)
    sif = torch.zeros_like(uztwc)
    sperc = torch.zeros_like(uztwc)
    sdro = torch.zeros_like(uztwc)

    dinc = 1.0 / n_inc
    pinc = twx / n_inc

    duz = 1.0 - (1.0 - UZK) ** dinc
    dlzp = 1.0 - (1.0 - LZPK) ** dinc
    dlzs = 1.0 - (1.0 - LZSK) ** dinc

    for _ in range(n_inc):
        # Direct runoff from ADIMP
        ratio = ((adimc - uztwc) / LZTWM.clamp_min(1e-6)).clamp_min(0.0)
        addro = pinc * ratio * ratio

        # Baseflow from primary lower-zone free
        bf_p = lzfpc * dlzp
        lzfpc = (lzfpc - bf_p).clamp_min(0.0)
        sbf = sbf + bf_p

        # Baseflow from supplementary lower-zone free
        bf_s = lzfsc * dlzs
        lzfsc = (lzfsc - bf_s).clamp_min(0.0)
        sbf = sbf + bf_s

        # Percolation.  Original SAC-SMA:
        #   perc = percm * uzfwc/UZFWM * (1 + ZPERC * defr^REXP)
        #   perc = min(perc, uzfwc)
        # The amplifier (1+ZPERC*defr^REXP) can reach ~(1+250) so the linear
        # formula gives perc >> uzfwc; the hard min clips it but the local
        # Jacobian in the unclipped regime explodes (~1e5/day) and the
        # gradient blows to Inf.
        #
        # Two safe forms:
        #   - explicit (tanh):  perc = uzfwc * tanh(k)  (smooth saturator)
        #   - implicit (exp):   perc = uzfwc * (1 - exp(-k*dinc))
        # The implicit form is the analytic implicit-Euler solution to the
        # linear depletion ODE d(uzfwc)/dt = -k * uzfwc with k frozen over
        # the substep, mathematically consistent with the implicit numerical
        # schemes recommended by Song et al. 2024.  Both have bounded
        # gradients in [0, 1].
        percm = LZFPM * dlzp + LZFSM * dlzs
        defr = (1.0 - (lztwc + lzfpc + lzfsc) / (LZTWM + LZFPM + LZFSM).clamp_min(1e-6)).clamp_min(0.0)
        # Add epsilon inside the base to avoid NaN gradient w.r.t. dynamic REXP
        # at defr=0 (lower zone fully saturated): d/dREXP(0**REXP) = 0*log(0)=NaN.
        k_perc = (percm / UZFWM.clamp_min(1e-6)) * (1.0 + ZPERC * ((defr + 1e-6) ** REXP))
        if implicit_percolation:
            # Bound the exponent argument to avoid float overflow; tanh
            # (saturator) and 1-exp(-x) agree for small x and saturate at 1.
            perc = uzfwc * (1.0 - torch.exp(-(k_perc * dinc).clamp_max(50.0)))
        else:
            perc = uzfwc * torch.tanh(k_perc)
        # Cap so lower zone does not exceed total capacity
        deficit_lz = (LZTWM + LZFPM + LZFSM) - (lztwc + lzfpc + lzfsc)
        perc = torch.minimum(perc, deficit_lz.clamp_min(0.0))
        uzfwc = uzfwc - perc

        sperc = sperc + perc

        # Interflow
        del_if = uzfwc * duz
        sif = sif + del_if
        uzfwc = uzfwc - del_if

        # Distribute percolation to lower zones
        perct = perc * (1.0 - PFREE)
        space_lzt = LZTWM - lztwc
        perct_into = torch.minimum(perct, space_lzt)
        lztwc = lztwc + perct_into
        percf = (perct - perct_into) + perc * PFREE

        # Split percf between primary & supplementary (parallel reservoirs)
        hpl = LZFPM / (LZFPM + LZFSM).clamp_min(1e-6)
        ratlp = lzfpc / LZFPM.clamp_min(1e-6)
        ratls = lzfsc / LZFSM.clamp_min(1e-6)
        # Denominator is state-dependent and can approach 0 when both lower
        # zones saturate (ratlp,ratls → 1).  A larger floor (0.1) bounds the
        # div backward so that grad_denom ~ 1/denom² stays < 100 instead of
        # ballooning to 1e12 and blowing up into NaN via 0 * Inf downstream.
        denom = (1.0 - ratlp + 1.0 - ratls).clamp_min(0.1)
        fracp = (hpl * 2.0 * (1.0 - ratlp) / denom).clamp(0.0, 1.0)
        percp = percf * fracp
        percs = percf - percp
        # Supplementary first
        space_s = LZFSM - lzfsc
        percs_into = torch.minimum(percs, space_s.clamp_min(0.0))
        lzfsc = lzfsc + percs_into
        primary_excess = percp + (percs - percs_into)
        space_p = LZFPM - lzfpc
        percp_into = torch.minimum(primary_excess, space_p.clamp_min(0.0))
        lzfpc = lzfpc + percp_into
        # Spillover from primary -> tension
        spill = (primary_excess - percp_into).clamp_min(0.0)
        lztwc = lztwc + spill

        # Distribute pinc between uzfwc and surface runoff
        space_uz = UZFWM - uzfwc
        into_uz = torch.minimum(pinc, space_uz.clamp_min(0.0))
        sur = (pinc - into_uz).clamp_min(0.0)
        uzfwc = uzfwc + into_uz
        ssur = ssur + sur * parea
        # Direct-runoff component from ADIMP region.
        # Since addro = pinc * ratio * ratio upstream, addro/pinc = ratio**2
        # algebraically; rewrite to avoid division by pinc (which can be 0
        # when no precip excess exists and produces NaN gradients).
        ratio_used = (1.0 - ratio * ratio).clamp(0.0, 1.0)
        adsur = sur * ratio_used
        ssur = ssur + adsur * ADIMP

        adimc = adimc + pinc - addro - adsur
        cap_adimc = UZTWM + LZTWM
        over = (adimc - cap_adimc).clamp_min(0.0)
        adimc = adimc - over
        addro = addro + over
        sdro = sdro + addro * ADIMP

    # ---------------- aggregate channel inflow ----------------
    eused = (et1 + et2 + et3) * parea
    sif = sif * parea
    tbf = sbf * parea
    bfcc = tbf / (1.0 + SIDE).clamp_min(1e-6)

    base = bfcc
    surf = roimp + sdro + ssur + sif

    # Riparian ET
    et4 = (edmnd - eused).clamp_min(0.0) * RIVA

    # Subtract et4 from channel inflow
    ch_inflow = surf + base - et4
    enough = (ch_inflow > 0).to(uztwc.dtype)
    half = 0.5 * et4
    surf_a = (surf - half).clamp_min(0.0)
    base_a = (base - half).clamp_min(0.0)
    # If one of them goes negative, the other absorbs the rest
    deficit_surf = (surf - half).clamp_max(0.0).abs()
    deficit_base = (base - half).clamp_max(0.0).abs()
    surf_a = (surf_a - deficit_base).clamp_min(0.0)
    base_a = (base_a - deficit_surf).clamp_min(0.0)
    surf = enough * surf_a
    base = enough * base_a

    # Ensure adimc >= uztwc
    adimc = torch.maximum(adimc, uztwc)

    tet = eused + et4 + et5

    new_state = SacState(uztwc=uztwc, uzfwc=uzfwc, lztwc=lztwc,
                          lzfsc=lzfsc, lzfpc=lzfpc, adimc=adimc)
    return new_state, surf, base, tet


def run_sacsma(
    pr: torch.Tensor,
    pet: torch.Tensor,
    params: dict[str, torch.Tensor],
    state: SacState | None = None,
    *,
    n_inc: int = N_INC,
    enable_capillary_rise: bool = True,
    implicit_percolation: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, SacState]:
    """Run SAC-SMA over a forcing window.

    Parameters
    ----------
    pr, pet : (N, T) tensors of effective precipitation and PET (mm/day).
    params : dict of (N,) (or (N,T)) tensors.

    Returns
    -------
    surf, base, tet : (N, T) tensors (mm/day).
    final_state : :class:`SacState`.
    """
    N, T = pr.shape
    # Pre-split params into static (ndim<2) and dynamic (ndim==2, last dim=T)
    static_p: dict[str, torch.Tensor] = {}
    dynamic_p: dict[str, torch.Tensor] = {}
    for k, v in params.items():
        if v.ndim >= 2:
            dynamic_p[k] = v
        else:
            static_p[k] = v

    if state is None:
        params0 = dict(static_p)
        for k, v in dynamic_p.items():
            params0[k] = v[..., 0]
        state = SacState.init_capacity(params0)

    surf = torch.empty_like(pr)
    base = torch.empty_like(pr)
    tet = torch.empty_like(pr)

    # Reuse a single dict across timesteps; only dynamic entries rewritten
    # each step.  Saves re-hashing + re-allocating 16+ dict slots per day.
    p_cur: dict[str, torch.Tensor] = dict(static_p)

    for t in range(T):
        for k, v in dynamic_p.items():
            p_cur[k] = v[..., t]
        state, s_t, b_t, e_t = sacsma_step(
            state, pr[:, t], pet[:, t], p_cur,
            n_inc=n_inc,
            enable_capillary_rise=enable_capillary_rise,
            implicit_percolation=implicit_percolation,
        )
        surf[:, t] = s_t
        base[:, t] = b_t
        tet[:, t] = e_t

    return surf, base, tet, state
