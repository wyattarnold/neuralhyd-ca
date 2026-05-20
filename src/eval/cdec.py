"""CDEC watershed evaluation: Conventional SAC-SMA vs neural models.

Evaluation period: 2003-10-01 through 2018-09-30 (post-calibration).

The 15 CDEC reservoir stations map to 14 basins in the neural model training
domain (BND is excluded — no matching PourPtID).  The conventional SAC-SMA
simulations are in mm/day and were calibrated through 9/30/2003; only the
post-calibration period is used for all models to ensure a fair comparison.

Public API
----------
run_cdec_barplot(model_dirs, labels, out_path)
    Evaluate each model and write the comparison barplot.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.lstm.loss import (
    compute_fehv,
    compute_fhv,
    compute_flv,
    compute_kge,
    compute_nse,
)
from src.paths import DATA_DIR, EVAL_DIR, TRAINING_OUTPUT_DIR

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SACSMA_DIR: Path = DATA_DIR / "external/sacsma15cdec/climate_historical"

# Evaluation window (post-calibration, water-year aligned)
EVAL_START = pd.Timestamp("2003-10-01")
EVAL_END   = pd.Timestamp("2018-09-30")

# CDEC station code → training-domain PourPtID (BND excluded)
CDEC_MAP: list[tuple[str, str]] = [
    ("FOL", "990000005"),
    ("ISB", "990000015"),
    ("MIL", "990000011"),
    ("MKM", "990000006"),
    ("MRC", "990000010"),
    ("NHG", "990000007"),
    ("NML", "990000008"),
    ("ORO", "990000003"),
    ("PNF", "990000012"),
    ("SCC", "990000014"),
    ("SHA", "990000001"),
    ("TLG", "990000009"),
    ("TRM", "990000013"),
    ("YRS", "990000004"),
]

METRICS: list[str] = ["nse", "kge", "fhv", "fehv", "flv"]

_METRIC_FUNS = {
    "nse":  compute_nse,
    "kge":  compute_kge,
    "fhv":  compute_fhv,
    "fehv": compute_fehv,
    "flv":  compute_flv,
}

_METRIC_LABELS = {
    "nse":  "NSE",
    "kge":  "KGE",
    "fhv":  "FHV (%)",
    "fehv": "FeHV (%)",
    "flv":  "FLV (%)",
}

# Bar colours keyed by model label substring (case-insensitive)
_BAR_COLORS: dict[str, str] = {
    "sacsma":  "#668bdc",   # steel blue — conventional physics
    "single":  "#9b59b6",   # purple — matches existing barplot palette
    "dual":    "#e89c0f",   # amber — matches existing barplot palette
    "moe":     "#27ae60",   # green — MoE
}
_BAR_FALLBACK = "#3b3b3d"


def _bar_color(label: str) -> str:
    for key, color in _BAR_COLORS.items():
        if key in label.lower():
            return color
    return _BAR_FALLBACK


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _find_ts(model_dir: Path, pourpt_id: str) -> Path:
    """Locate a timeseries CSV for *pourpt_id* across k-fold sub-dirs."""
    for fold_dir in sorted(model_dir.glob("fold_*")):
        p = fold_dir / "timeseries" / f"{pourpt_id}.csv"
        if p.exists():
            return p
    raise FileNotFoundError(
        f"No timeseries file for {pourpt_id} in {model_dir}"
    )


def _read_sacsma(cdec_code: str) -> pd.Series:
    """Parse the whitespace-delimited SAC-SMA file → DatetimeIndex Series (mm/day)."""
    path = SACSMA_DIR / f"simflow_sacsma_{cdec_code}_short.txt"
    rows: list[tuple[pd.Timestamp, float]] = []
    with open(path) as fh:
        for line in fh:
            parts = line.split()
            if len(parts) < 4:
                continue
            y, m, d, val = int(parts[0]), int(parts[1]), int(parts[2]), float(parts[3])
            rows.append((pd.Timestamp(y, m, d), val))
    s = pd.Series(dict(rows))
    s.index = pd.DatetimeIndex(s.index)
    s.index.name = "date"
    return s


def _clip(obj: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    return obj.loc[EVAL_START:EVAL_END]


def _metrics(obs: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    return {m: float(_METRIC_FUNS[m](obs, pred)) for m in METRICS}


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------

def evaluate_cdec(
    model_dirs: Sequence[Path],
    labels: Sequence[str],
    *,
    sacsma_label: str = "SACSMA",
) -> dict[str, dict[str, dict[str, float]]]:
    """Evaluate each model on the 14 common CDEC basins.

    The conventional SAC-SMA is always evaluated first, using the dPL model's
    ``obs_mm`` column as the shared observed reference.  At least one of
    *model_dirs* must be the dPL SAC-SMA output directory (it provides obs).

    Parameters
    ----------
    model_dirs:
        Sequence of training-output directories for the neural models, in the
        same order as *labels*.
    labels:
        Display labels for each directory; the first entry whose directory
        contains a dPL-style ``obs_mm`` column will supply the observed data.
    sacsma_label:
        Display label for the conventional SAC-SMA baseline.

    Returns
    -------
    ``{label: {pourpt_id: {metric: value}}}`` — includes the SAC-SMA baseline
    under *sacsma_label* as the first key.
    """
    all_labels = [sacsma_label] + list(labels)
    results: dict[str, dict[str, dict[str, float]]] = {lbl: {} for lbl in all_labels}

    for cdec_code, pourpt_id in CDEC_MAP:
        # Find the dPL directory to use as the obs reference
        obs_ref_series: pd.Series | None = None
        for lbl, mdir in zip(labels, model_dirs):
            try:
                path = _find_ts(mdir, pourpt_id)
            except FileNotFoundError:
                continue
            df = pd.read_csv(path, parse_dates=["date"], index_col="date")
            if "obs_mm" in df.columns:
                obs_ref_series = _clip(df["obs_mm"]).dropna()
                break

        if obs_ref_series is None:
            print(f"  WARNING: no dPL obs_mm for {cdec_code} ({pourpt_id}) — skipping")
            continue

        # Conventional SAC-SMA vs dPL obs
        sacsma_sim = _clip(_read_sacsma(cdec_code))
        common = sacsma_sim.index.intersection(obs_ref_series.index)
        valid  = obs_ref_series.loc[common].notna() & sacsma_sim.loc[common].notna()
        if valid.sum() < 100:
            print(f"  WARNING: only {valid.sum()} common dates for {cdec_code} — skipping")
            continue
        results[sacsma_label][pourpt_id] = _metrics(
            obs_ref_series.loc[common][valid].to_numpy(),
            sacsma_sim.loc[common][valid].to_numpy(),
        )

        # Neural models
        for lbl, mdir in zip(labels, model_dirs):
            try:
                path = _find_ts(mdir, pourpt_id)
            except FileNotFoundError:
                print(f"  WARNING: {lbl} has no timeseries for {pourpt_id}")
                continue
            df = pd.read_csv(path, parse_dates=["date"], index_col="date")

            # Determine obs/pred column names
            if "obs_mm" in df.columns and "pred_mm" in df.columns:
                obs_col, pred_col = "obs_mm", "pred_mm"
            elif "obs" in df.columns and "pred" in df.columns:
                obs_col, pred_col = "obs", "pred"
            else:
                print(f"  WARNING: unrecognised columns in {path.name} — skipping")
                continue

            sub = _clip(df[[obs_col, pred_col]]).dropna()
            results[lbl][pourpt_id] = _metrics(
                sub[obs_col].to_numpy(), sub[pred_col].to_numpy()
            )

    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _apply_rc() -> None:
    plt.rcParams.update({
        "font.family":        "sans-serif",
        "font.sans-serif":    ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size":          8,
        "axes.titlesize":     9,
        "axes.labelsize":     8,
        "xtick.labelsize":    7,
        "ytick.labelsize":    7,
        "axes.linewidth":     0.5,
        "axes.spines.top":    False,
        "axes.spines.right":  False,
    })


_YLIMS_FIXED: dict[str, tuple[float, float]] = {
    "nse": (-0.15, 1.02),
    "kge": (-0.15, 1.02),
}


def plot_cdec_barplot(
    results: dict[str, dict[str, dict[str, float]]],
    out_path: Path,
    *,
    title: str = "CDEC Watershed Evaluation — 14 Basins, 2003–2018",
) -> plt.Figure:
    """One panel per metric; one boxplot per model with jittered data points.

    Boxes are semi-transparent so individual basin points show through.
    Whiskers extend to the full data range with no caps.

    Parameters
    ----------
    results:
        Output of :func:`evaluate_cdec`.
    out_path:
        Destination PNG path (parent directory is created if needed).
    title:
        Figure suptitle.
    """
    _apply_rc()

    rng          = np.random.default_rng(42)
    model_labels = list(results.keys())
    n_models     = len(model_labels)
    n_metrics    = len(METRICS)
    x            = np.arange(n_models)
    colors       = [_bar_color(lbl) for lbl in model_labels]
    box_width    = 0.50

    fig, axes = plt.subplots(
        1, n_metrics,
        figsize=(2.6 * n_metrics, 3.4),
        sharey=False,
    )

    for ax, metric in zip(axes, METRICS):
        all_vals: list[np.ndarray] = []
        for lbl in model_labels:
            v = np.array([
                d[metric]
                for d in results[lbl].values()
                if np.isfinite(d.get(metric, float("nan")))
            ])
            all_vals.append(v)

        # Draw boxplots (IQR box + full-range whiskers, no caps)
        bp = ax.boxplot(
            all_vals,
            positions=x,
            widths=box_width,
            patch_artist=True,
            showfliers=False,
            whis=[0, 100],       # whiskers to min/max
            showcaps=False,
            medianprops=dict(color="#111111", linewidth=1.4, zorder=6),
            whiskerprops=dict(linewidth=0.8, linestyle="-", zorder=4),
            boxprops=dict(linewidth=0.6, zorder=3),
        )

        # Colour + transparency for each box
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.45)

        # Style whiskers to match box colour
        for i, (w1, w2) in enumerate(zip(bp["whiskers"][0::2], bp["whiskers"][1::2])):
            c = colors[i]
            w1.set_color(c)
            w2.set_color(c)

        # Jittered data points
        jitter_width = 0.18
        for i, (vals, color) in enumerate(zip(all_vals, colors)):
            if len(vals) == 0:
                continue
            jitter = rng.uniform(-jitter_width, jitter_width, size=len(vals))
            ax.scatter(
                x[i] + jitter, vals,
                s=14, color=color,
                edgecolors="white", linewidths=0.3,
                alpha=0.85, zorder=5,
            )

        # Axis limits
        if metric in _YLIMS_FIXED:
            ax.set_ylim(*_YLIMS_FIXED[metric])
        else:
            all_finite = np.concatenate([v for v in all_vals if len(v)])
            all_finite = all_finite[np.isfinite(all_finite)]
            if len(all_finite):
                lo   = min(float(np.nanmin(all_finite)), 0)
                hi   = max(float(np.nanmax(all_finite)), 0)
                span = max(abs(hi - lo), 1.0)
                ax.set_ylim(lo - 0.18 * span, hi + 0.22 * span)

        ax.axhline(0, color="#252525", linewidth=0.5, linestyle="--", zorder=2)

        # Median value labels (above/below the box)
        ylo, yhi = ax.get_ylim()
        pad = 0.03 * (yhi - ylo)
        for i, vals in enumerate(all_vals):
            if len(vals) == 0:
                continue
            med   = float(np.nanmedian(vals))
            y_pos = med + pad if med >= 0 else med - pad
            va    = "bottom"   if med >= 0 else "top"
            ax.text(
                x[i], y_pos,
                f"{med:.2f}" if abs(med) < 10 else f"{med:.1f}",
                ha="center", va=va, fontsize=6, fontweight="medium",
            )

        ax.set_ylabel(_METRIC_LABELS[metric], fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(model_labels, fontsize=6.5, rotation=25, ha="right")
        ax.set_xlim(-0.6, n_models - 0.4)
        ax.grid(axis="y", linewidth=0.3, linestyle="--", alpha=0.15, color="#000000")

    fig.suptitle(title, fontsize=9, fontweight="medium", y=1.03)
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    return fig


# ---------------------------------------------------------------------------
# High-level entry point
# ---------------------------------------------------------------------------

def run_cdec_barplot(
    model_run_names: Sequence[str],
    labels: Sequence[str],
    *,
    out_path: Path | None = None,
    sacsma_label: str = "SACSMA",
) -> None:
    """Evaluate models and write the CDEC comparison barplot.

    Parameters
    ----------
    model_run_names:
        Training-output directory names under ``data/training/output/``.
    labels:
        Display labels in the same order as *model_run_names*.
    out_path:
        Destination PNG (defaults to ``data/eval/cdec_barplot.png``).
    sacsma_label:
        Display label for the conventional SAC-SMA baseline column.
    """
    if out_path is None:
        out_path = EVAL_DIR / "cdec_barplot.png"

    model_dirs = [TRAINING_OUTPUT_DIR / name for name in model_run_names]

    # Sanity check
    for name, mdir in zip(model_run_names, model_dirs):
        if not mdir.exists():
            raise FileNotFoundError(
                f"Training output directory not found: {mdir}\n"
                f"  (looked for run name '{name}')"
            )

    print(f"Evaluating {len(CDEC_MAP)} CDEC basins "
          f"(period: {EVAL_START.date()} – {EVAL_END.date()}) ...")

    results = evaluate_cdec(model_dirs, labels, sacsma_label=sacsma_label)

    # Print summary
    all_labels = list(results.keys())
    print(f"\n{'Metric':<8}" + "".join(f"{l:>12}" for l in all_labels))
    for m in METRICS:
        row = f"{m:<8}"
        for lbl in all_labels:
            vals = [v[m] for v in results[lbl].values() if np.isfinite(v.get(m, float("nan")))]
            med  = float(np.nanmedian(vals)) if vals else float("nan")
            row += f"{med:>12.3f}"
        print(row)

    fig = plot_cdec_barplot(results, out_path)
    plt.close(fig)
    print(f"\nSaved → {out_path}")
