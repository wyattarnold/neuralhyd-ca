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


# ---------------------------------------------------------------------------
# Multipanel barplot — median metrics across models
# ---------------------------------------------------------------------------

# Bar colours keyed by substring in model labels
_BAR_COLORS: dict[str, str] = {
    "VIC":    "#668bdc",
    "single": "#9b59b6",
    "dual":   "#e89c0f",
}

_BAR_FALLBACK = "#3b3b3d"


def _bar_color(label: str) -> str:
    for key, color in _BAR_COLORS.items():
        if key.lower() in label.lower():
            return color
    return _BAR_FALLBACK


def plot_metric_barplot(
    data: Dict[str, Dict[str, float]],
    metrics: list[str],
    *,
    title: Optional[str] = None,
    out_path: Optional[Path] = None,
) -> plt.Figure:
    """Multipanel barplot: one panel per metric, one bar per model.

    Parameters
    ----------
    data : {model_label: {metric_name: median_value, ...}, ...}
    metrics : ordered list of metric names (one panel each)
    title : optional suptitle
    out_path : save path

    Returns
    -------
    matplotlib.figure.Figure
    """
    _apply_rc()

    n_metrics = len(metrics)
    labels = list(data.keys())
    n_models = len(labels)

    fig, axes = plt.subplots(1, n_metrics, figsize=(2.4 * n_metrics, 3.2),
                             sharey=False)
    if n_metrics == 1:
        axes = [axes]

    # Metric display config: (ylabel, zero_line, better_direction)
    _METRIC_CFG: dict[str, tuple[str, bool]] = {
        "nse":  ("NSE",       True),
        "kge":  ("KGE",       True),
        "fhv":  ("FHV (%)",   True),
        "fehv": ("FeHV (%)",  True),
        "flv":  ("FLV (%)",   True),
    }

    colors = [_bar_color(l) for l in labels]
    x = np.arange(n_models)

    # Display name mapping for xtick labels
    _DISPLAY_NAMES: dict[str, str] = {
        "VIC":              "VIC",
        "single_lstm_kfold": "Single LSTM",
        "dual_lstm_kfold":   "Dual LSTM",
    }
    tick_labels = [_DISPLAY_NAMES.get(l, l) for l in labels]

    for ax, metric in zip(axes, metrics):
        values = [data[l].get(metric, float("nan")) for l in labels]
        bars = ax.bar(x, values, width=0.6, color=colors, edgecolor="white",
                      linewidth=0.5, zorder=3)

        # Set axis limits before computing label positions
        finite_vals = [v for v in values if np.isfinite(v)]
        if metric in ("nse", "kge"):
            ax.set_ylim(min(min(finite_vals, default=0) * 1.3, -0.05), 1.015)
        else:
            lo = min(min(finite_vals, default=0), 0)  # always include zero
            hi = max(max(finite_vals, default=1), 0)  # always include zero
            span = max(abs(hi - lo), 1.0)
            ax.set_ylim(lo - 0.15 * span, hi + 0.20 * span)

        # Value labels on bars
        ylo, yhi = ax.get_ylim()
        label_pad = 0.025 * (yhi - ylo)
        for bar, val in zip(bars, values):
            if np.isfinite(val):
                if val >= 0:
                    y_pos = val + label_pad
                    va = "bottom"
                else:
                    y_pos = val - label_pad
                    va = "top"
                ax.text(bar.get_x() + bar.get_width() / 2, y_pos,
                        f"{val:.2f}" if abs(val) < 10 else f"{val:.1f}",
                        ha="center", va=va, fontsize=6, fontweight="medium")

        cfg = _METRIC_CFG.get(metric, (metric.upper(), False))
        ax.set_ylabel(cfg[0], fontsize=8)
        ax.axhline(0, color="#252525", linewidth=0.5, linestyle="--", zorder=5)

        if cfg[1]:
            ax.axhline(0, color="#BBBBBB", linewidth=0.5, zorder=1)

        ax.set_xticks(x)
        ax.set_xticklabels(tick_labels, fontsize=6.5, rotation=25, ha="right")
        ax.grid(axis="y", linewidth=0.3, linestyle="--", alpha=0.15,
                color="#000000")

    if title:
        fig.suptitle(title, fontsize=10, fontweight="medium", y=1.02)

    fig.tight_layout()

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=300, bbox_inches="tight",
                    facecolor="white", edgecolor="none")
    return fig
