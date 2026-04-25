"""Hamon potential ET with a learnable adjustment coefficient.

Reference: Lu et al. (2005), Forsythe et al. (1995).  Daylength uses the
CBM model (same as :file:`pet_hamon.m`).

All operations are vectorised over an arbitrary leading batch dimension
(typically ``(N_units,)``) so a single call computes PET for every HUC12
in the batch across ``T`` days.
"""

from __future__ import annotations

import math

import torch


def _theta(doy: torch.Tensor) -> torch.Tensor:
    # Forsythe et al. CBM-model intermediate: doy is 1..366.
    return 0.2163108 + 2.0 * torch.atan(0.9671396 * torch.tan(0.0086 * (doy - 186.0)))


def daylight_hours(doy: torch.Tensor, latitude_rad: torch.Tensor) -> torch.Tensor:
    """Daylight hours per day.

    Parameters
    ----------
    doy : (..., T) tensor
        Julian day of year (1..366).
    latitude_rad : (...,) tensor
        Latitude in **radians**, broadcastable with ``doy``.

    Returns
    -------
    (..., T) tensor of daylight hours.
    """
    theta = _theta(doy)
    var_pi = torch.asin(0.39795 * torch.cos(theta))
    sun_alt = math.sin(0.8333 * math.pi / 180.0)
    lat = latitude_rad.unsqueeze(-1)
    cos_arg = (sun_alt + torch.sin(lat) * torch.sin(var_pi)) / (
        torch.cos(lat) * torch.cos(var_pi)
    )
    cos_arg = cos_arg.clamp(-1.0 + 1e-6, 1.0 - 1e-6)
    return 24.0 - (24.0 / math.pi) * torch.acos(cos_arg)


def hamon_pet(
    tavg_c: torch.Tensor,
    doy: torch.Tensor,
    latitude_rad: torch.Tensor,
    coef: torch.Tensor,
) -> torch.Tensor:
    """Compute PET (mm/day) via the Hamon equation.

    Parameters
    ----------
    tavg_c : (..., T) tensor
        Daily mean air temperature, deg C.
    doy : (..., T) tensor
        Julian day of year (1..366), broadcasts with tavg.
    latitude_rad : (...,) tensor
        Latitude (radians), broadcasts with batch dims.
    coef : (...,) or (..., T) tensor
        Hamon coefficient (dimensionless, typically 0.6–1.6).

    Returns
    -------
    (..., T) tensor of PET in mm/day, clamped non-negative.
    """
    dl = daylight_hours(doy, latitude_rad)
    e_sat = 0.611 * torch.exp(17.27 * tavg_c / (237.3 + tavg_c))
    coef_b = coef if coef.ndim == tavg_c.ndim else coef.unsqueeze(-1)
    pet = coef_b * 29.8 * dl * (e_sat / (tavg_c + 273.2))
    return pet.clamp_min(0.0)
