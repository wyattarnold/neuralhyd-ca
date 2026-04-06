"""Plotting utilities for post-processing evaluation results."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Consistent colour palette keyed by substrings in series labels.
_COLORS: Dict[str, str] = {
    "dual":             "#E67E22",   # orange
    "single":           "#27AE60",   # green
    "VIC Simulated":    "#2980B9",   # blue
    "VIC Regionalized": "#2980B9",   # blue
    "VIC Calibrated":   "#C0392B",   # red
}


def _color_for(label: str) -> Optional[str]:
    """Return the palette colour matching *label*, or None (use default cycle)."""
    for key, color in _COLORS.items():
        if key in label:
            return color
    return None


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

    The metric is on the y-axis and non-exceedance probability on x-axis.

    Parameters
    ----------
    series_dict : {label: 1-d array of metric values}
        Each entry becomes one CDF line.
    metric_name : str
        Name of the metric (used for axis label).
    title : str, optional
        Plot title.  Defaults to "CDF of {metric_name}".
    ylim : (lo, hi), optional
        y-axis limits.  Defaults to data range.
    yscale : str, optional
        Axis scale (e.g. "symlog").  Defaults to linear.
    hline_at : float, optional
        Draw a horizontal reference line at this value.
    out_path : Path, optional
        If given, saves the figure to this path.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(7, 5))

    for label, values in series_dict.items():
        vals = np.sort(values[np.isfinite(values)])
        cdf = np.arange(1, len(vals) + 1) / len(vals)
        color = _color_for(label)
        ax.plot(cdf, vals, label=f"{label} (n={len(vals)})", linewidth=1.5,
                **({"color": color} if color else {}))

    if hline_at is not None:
        ax.axhline(hline_at, color="k", linewidth=0.8, linestyle="--", alpha=0.6,
                   label=f"ideal ({hline_at})")

    ax.set_xlabel("Non-exceedance probability")
    ax.set_ylabel(metric_name)
    ax.set_title(title or f"CDF of {metric_name}")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    if yscale is not None:
        ax.set_yscale(yscale)
    if ylim is not None:
        ax.set_ylim(ylim)

    fig.tight_layout()
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
    return fig
