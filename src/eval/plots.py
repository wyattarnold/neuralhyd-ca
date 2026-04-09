"""Plotting utilities for post-processing evaluation results."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Style: series appearance keyed by substrings in labels
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _Style:
    color: str
    linewidth: float
    linestyle: str
    alpha: float
    zorder: int
    display: str          # clean display name for annotation


# Palette aligned with the web app (app/frontend/src/chartConfig.js).
# LSTM models are hero lines (thick, saturated); VIC recedes (thin, dashed).
_STYLES: dict[str, _Style] = {
    "dual":             _Style("#e89c0f", 2.0, "-",  1.0, 4, "Dual LSTM"),
    "single":           _Style("#9b59b6", 2.0, "-",  1.0, 3, "Single LSTM"),
    "VIC Simulated":    _Style("#668bdc", 1.2, "--", 0.75, 2, "VIC Regionalized"),
    "VIC Regionalized": _Style("#668bdc", 1.2, "--", 0.55, 1, "VIC Regionalized"),
    "VIC Calibrated":   _Style("#3568b8", 1.2, "-.", 0.75, 2, "VIC Calibrated"),
}

_FALLBACK = _Style("#3b3b3d", 1.4, "-", 0.9, 2, "")


def _style_for(label: str) -> _Style:
    for key, style in _STYLES.items():
        if key in label:
            return style
    return _FALLBACK


# ---------------------------------------------------------------------------
# Global rc overrides — small, clean, minimal
# ---------------------------------------------------------------------------

def _apply_rc() -> None:
    plt.rcParams.update({
        "font.family":       "sans-serif",
        "font.sans-serif":   ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size":         8,
        "axes.titlesize":    9,
        "axes.labelsize":    8,
        "xtick.labelsize":   7,
        "ytick.labelsize":   7,
        "axes.linewidth":    0.5,
        "xtick.major.width": 0.4,
        "ytick.major.width": 0.4,
        "xtick.major.size":  3,
        "ytick.major.size":  3,
        "axes.spines.top":   True,
        "axes.spines.right": True,
        "axes.grid":         False,
        "figure.dpi":        300,
    })


# ---------------------------------------------------------------------------
# CDF plot
# ---------------------------------------------------------------------------

def plot_metric_cdf(
    series_dict: Dict[str, np.ndarray],
    metric_name: str,
    *,
    title: Optional[str] = None,
    ylim: Optional[tuple] = None,
    yscale: Optional[str] = None,
    hline_at: Optional[float] = None,
    out_path: Optional[Path] = None,
) -> plt.Figure:
    """Plot CDF curves for one metric across multiple model series.

    Parameters
    ----------
    series_dict : {label: 1-d array of metric values}
    metric_name : str
    title, ylim, yscale, hline_at, out_path : optional overrides

    Returns
    -------
    matplotlib.figure.Figure
    """
    _apply_rc()

    fig, ax = plt.subplots(figsize=(5, 4.5))

    for label, values in series_dict.items():
        vals = np.sort(values[np.isfinite(values)])
        if len(vals) == 0:
            continue
        cdf = np.arange(1, len(vals) + 1) / len(vals)
        sty = _style_for(label)
        display = sty.display or label

        med_idx = np.searchsorted(cdf, 0.5)
        med_idx = min(med_idx, len(vals) - 1)
        med_val = vals[med_idx]

        ax.plot(
            cdf, vals,
            color=sty.color, linewidth=sty.linewidth,
            linestyle=sty.linestyle, alpha=sty.alpha,
            zorder=sty.zorder, solid_capstyle="round",
            label=f"{display}  (med {med_val:.2f}, n={len(vals)})",
        )

        # Median marker
        ax.plot(cdf[med_idx], med_val, "o", color=sty.color,
                markersize=3, zorder=sty.zorder + 5, alpha=sty.alpha)

    # ---- reference line ----
    if hline_at is not None:
        ax.axhline(hline_at, color="#BBBBBB", linewidth=0.5, linestyle="-",
                   zorder=0)

    # ---- ticks ----
    ax.set_xticks(np.linspace(0, 1, 11))

    # ---- grid: fine, dashed, nearly transparent ----
    ax.grid(True, linewidth=0.3, linestyle="--", alpha=0.15, color="#000000")

    # ---- axes ----
    ax.set_xlabel("Non-exceedance probability")
    ax.set_ylabel(metric_name)

    if yscale is not None:
        ax.set_yscale(yscale)
    if ylim is not None:
        ax.set_ylim(ylim)

    if title:
        ax.set_title(title, loc="left", fontsize=9, fontweight="medium", pad=8)

    ax.set_xlim(0, 1.0)

    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#333333")

    # ---- legend ----
    ax.legend(
        loc="lower right", frameon=False,
        fontsize=6.5, handlelength=2.0, labelspacing=0.4,
    )

    fig.tight_layout()

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=300, bbox_inches="tight",
                    facecolor="white", edgecolor="none")
    return fig
