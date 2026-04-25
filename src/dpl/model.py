"""Top-level differentiable hydrology model: dPL + SAC-SMA + Snow-17 + Lohmann.

The forward pass

1. Predicts physical parameters per HUC12 from gA (LSTM + static encoder).
2. Computes Hamon PET with the learned coefficient.
3. Runs Snow-17 on per-HUC12 forcing → effective P entering the soil.
4. Runs SAC-SMA per HUC12 → surface + base flow components.
5. Routes per-HUC12 hillslope flow with the gamma UH.
6. Aggregates all HUC12s of a basin to the gauge with area weights.

Inputs are organised so that all HUC12s across all basins in a batch are
flattened along a single ``N_units`` dimension.  ``basin_index`` maps
each unit to its parent basin so the final aggregation is a vectorised
``segment_sum`` (here implemented as ``index_add_``).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .config import DplConfig
from .parameter_net import ParameterNet
from .pet import hamon_pet
from .routing import route as route_lohmann
from .sacsma import run_sacsma
from .snow17 import run_snow17


class DPLSacSMA(nn.Module):
    """Differentiable parameter-learning SAC-SMA model.

    Forward signature
    -----------------
    ``forward(x_dyn, x_static, forcing, aux) -> q_gauge``

    All tensors (except ``aux``) have leading dim ``N_units``
    (HUC12 × Nmul), with ``aux`` carrying basin-level metadata and the
    aggregation index used to roll up unit flows to the gauge.
    """

    def __init__(self, config: DplConfig, n_dynamic: int, n_static: int):
        super().__init__()
        self.config = config
        self.parameter_net = ParameterNet(config, n_dynamic=n_dynamic, n_static=n_static)

    def forward(
        self,
        x_dyn: torch.Tensor,        # (N_units, T, n_dynamic_norm) — gA inputs (z-scored)
        x_static: torch.Tensor,     # (N_units, n_static_norm)
        forcing: dict[str, torch.Tensor],
        aux: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Run the full pipeline.

        Parameters
        ----------
        x_dyn, x_static
            Z-scored inputs to gA (LSTM + static encoder).
        forcing : dict
            Raw, *unscaled* per-unit forcings the physics core consumes:
              - ``prcp``  : (N_units, T) precipitation (mm/day)
              - ``tavg``  : (N_units, T) air temperature (deg C)
              - ``doy``   : (N_units, T) day of year (1..366)
        aux : dict
            Per-unit auxiliary quantities:
              - ``elev_m`` : (N_units,) elevation (m)
              - ``lat_rad``: (N_units,) latitude (radians)
              - ``area_weight`` : (N_units,) HUC12 area / parent-basin area
              - ``basin_index`` : (N_units,) long tensor mapping unit→basin id (0..B-1)
              - ``n_basins``    : int
        """
        params = self.parameter_net(x_dyn, x_static)

        prcp = forcing["prcp"]
        tavg = forcing["tavg"]
        doy = forcing["doy"]

        # ---------------- PET (Hamon) ----------------
        pet = hamon_pet(tavg, doy, aux["lat_rad"], params["HAMON_COEF"])

        # ---------------- Snow-17 ----------------
        if self.config.enable_snow17:
            snow_par = {n: params[n] for n in (
                "SCF", "PXTEMP", "MFMAX", "MFMIN", "UADJ",
                "MBASE", "TIPM", "PLWHC", "NMF", "DAYGM",
            )}
            effective_p, _ = run_snow17(prcp, tavg, doy, aux["elev_m"], snow_par)
        else:
            effective_p = prcp

        # ---------------- SAC-SMA ----------------
        sac_par = {n: params[n] for n in (
            "UZTWM", "UZFWM", "LZTWM", "LZFPM", "LZFSM",
            "UZK", "LZPK", "LZSK",
            "ZPERC", "REXP", "PFREE",
            "PCTIM", "ADIMP", "RIVA", "SIDE", "RSERV",
            "THETA_C",
        )}
        surf, base, _tet, _ = run_sacsma(
            effective_p, pet, sac_par,
            n_inc=self.config.n_inc,
            enable_capillary_rise=self.config.enable_capillary_rise,
            implicit_percolation=self.config.implicit_percolation,
        )

        # ---------------- Routing ----------------
        if self.config.enable_routing:
            q_unit = route_lohmann(surf, base, params["UH_N"], params["UH_TAU"])
        else:
            q_unit = surf + base                           # (N_units, T)

        # ---------------- Aggregate to gauge ----------------
        weighted = q_unit * aux["area_weight"].unsqueeze(-1)     # (N_units, T)
        B = int(aux["n_basins"])
        T = weighted.shape[-1]
        q_gauge = weighted.new_zeros((B, T))
        q_gauge.index_add_(0, aux["basin_index"], weighted)

        # Baseflow at gauge — routing leaves baseflow unchanged in this UH,
        # so aggregate the per-unit baseflow with the same area weights.
        base_w = base * aux["area_weight"].unsqueeze(-1)
        q_base_gauge = weighted.new_zeros((B, T))
        q_base_gauge.index_add_(0, aux["basin_index"], base_w)

        return {
            "q_gauge": q_gauge,
            "q_base_gauge": q_base_gauge,
            "q_unit": q_unit,
            "surf": surf,
            "base": base,
            "pet": pet,
            "params": params,
        }


def build_model(config: DplConfig, n_dynamic: int, n_static: int) -> DPLSacSMA:
    """Instantiate :class:`DPLSacSMA` from a :class:`DplConfig`."""
    return DPLSacSMA(config, n_dynamic=n_dynamic, n_static=n_static)
