"""Analyse per-basin extreme-flow quantiles for aux-loss calibration.

The aux-loss extreme ramp targets each basin's top (1 - start_q) fraction
of the flow distribution. All analysis is in normalised space
(flow_mm / precip_mean - the same dimensionless target the LSTM sees).

Outputs (written to ``data/prepare/flow_extremes/``):
  - basin_quantiles.csv
  - tier_quantile_summary.csv
  - tier_quantile_bars.png
  - ramp_coverage_heatmap.png
  - per_basin_ramp_curves.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.lstm.config import Config, load_config
from src.lstm.dataset import load_all_data, create_folds
from src.paths import QA_DIR

TIER_LABELS = {1: "T1 (rain)", 2: "T2 (mixed)", 3: "T3 (snow)"}
TIER_COLOURS = {1: "#e66101", 2: "#5e3c99", 3: "#0571b0"}
REPORT_PCTS = [50, 75, 90, 95, 99, 99.5, 99.9, 99.99]


def _precip_mean(climate_data: dict, bid: int) -> float:
    cdf = climate_data[bid]
    vals = cdf["precip_mm"].values if "precip_mm" in cdf.columns else cdf.values[:, 0]
    return max(float(np.mean(vals)), 0.01)


def _ramp_weight(y: np.ndarray, q_start: float, q_top: float, peak: float) -> np.ndarray:
    ramp = max(q_top - q_start, 1e-4)
    frac = np.clip((y - q_start) / ramp, 0.0, 1.0)
    return 1.0 + (peak - 1.0) * frac


def main(config: Config | None = None) -> None:
    if config is None:
        config = load_config()

    out_dir = QA_DIR / "flow_extremes"
    out_dir.mkdir(parents=True, exist_ok=True)

    basin_ids, climate_data, flow_data, static_df, tier_map = load_all_data(config)
    folds = create_folds(basin_ids, tier_map, flow_data,
                         n_folds=config.n_folds, seed=config.seed)
    train_ids, _ = folds[0]
    print(f"\nUsing fold-0 training set: {len(train_ids)} basins")

    basin_norm: dict[int, np.ndarray] = {}
    for bid in train_ids:
        vals = flow_data[bid]["flow"].dropna().values.astype(np.float64)
        if len(vals) < 100:
            continue
        basin_norm[bid] = vals / _precip_mean(climate_data, bid)

    print(f"{len(basin_norm)} basins with >=100 observed days")

    rows = []
    for bid, vals in basin_norm.items():
        r = {"basin_id": bid, "tier": tier_map[bid], "n_days": len(vals),
             "mean": float(vals.mean()), "std": float(vals.std()),
             "max": float(vals.max())}
        for p in REPORT_PCTS:
            r[f"p{p}"] = float(np.percentile(vals, p))
        rows.append(r)
    bp_df = pd.DataFrame(rows).sort_values(["tier", "p99"])
    bp_df.to_csv(out_dir / "basin_quantiles.csv", index=False, float_format="%.4f")
    print(f"Saved: {out_dir / 'basin_quantiles.csv'}")

    summary_cols = [f"p{p}" for p in REPORT_PCTS]
    tsum = (bp_df.groupby("tier")[summary_cols]
                 .agg(["median",
                       lambda s: np.percentile(s, 25),
                       lambda s: np.percentile(s, 75),
                       "min", "max"]))
    tsum.columns = [f"{c}_{a}" for c, a in tsum.columns]
    tsum = tsum.rename(columns=lambda c: c.replace("<lambda_0>", "q25")
                                          .replace("<lambda_1>", "q75"))
    tsum.to_csv(out_dir / "tier_quantile_summary.csv", float_format="%.4f")
    print(f"Saved: {out_dir / 'tier_quantile_summary.csv'}")

    print("\n-- Per-basin quantiles (median [IQR] by tier, normalised flow) --")
    for tier in (1, 2, 3):
        sub = bp_df[bp_df.tier == tier]
        print(f"  {TIER_LABELS[tier]:>14s}  (n={len(sub):>3d})  "
              f"p99={sub.p99.median():5.2f} "
              f"[{np.percentile(sub.p99, 25):4.2f}"
              f"-{np.percentile(sub.p99, 75):5.2f}]  "
              f"p99.9={sub['p99.9'].median():5.2f} "
              f"[{np.percentile(sub['p99.9'], 25):5.2f}"
              f"-{np.percentile(sub['p99.9'], 75):5.2f}]")

    # FIGURE 1: per-tier box plots of p99/p99.9
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)
    for ax, pcol, title in zip(
        axes, ["p99", "p99.9"],
        ["Per-basin p99 (1-in-100 day event)",
         "Per-basin p99.9 (1-in-1000 day event)"],
    ):
        data = [bp_df[bp_df.tier == t][pcol].values for t in (1, 2, 3)]
        bp = ax.boxplot(data, positions=[1, 2, 3], widths=0.6,
                        patch_artist=True, showfliers=True,
                        medianprops=dict(color="black", lw=1.5))
        for patch, t in zip(bp["boxes"], (1, 2, 3)):
            patch.set_facecolor(TIER_COLOURS[t]); patch.set_alpha(0.7)
        ax.set_xticks([1, 2, 3])
        ax.set_xticklabels([TIER_LABELS[t] for t in (1, 2, 3)])
        ax.set_ylabel("Normalised flow (flow / precip_mean)")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Per-basin extreme quantiles - each basin has its own threshold",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / "tier_quantile_bars.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_dir / 'tier_quantile_bars.png'}")

    # FIGURE 2: gradient-share heatmap across config grid
    #
    # For each (start_q, top_q, peak_boost) triple, compute the share of total
    # aux-loss gradient weight that lands on the top (1 − start_q) of each
    # basin's days, averaged across basins.  Under uniform weighting the
    # share equals (1 − start_q); the ramp inflates it.  This single metric
    # (a percentage) is directly comparable across all panels on a shared
    # 0–100% scale, so the user can see at a glance how each config
    # concentrates gradient on extreme events.
    start_qs = np.array([0.90, 0.95, 0.97, 0.98, 0.99, 0.995])
    top_qs = np.array([0.98, 0.99, 0.995, 0.999, 0.9995, 0.9999])
    boost_vals = [5.0, 15.0, 30.0]

    def _pct_label(q: float) -> str:
        # e.g. 0.9999 -> "99.99", 0.9 -> "90"
        s = f"{q * 100:.4f}".rstrip("0").rstrip(".")
        return s

    fig, axes = plt.subplots(1, len(boost_vals),
                             figsize=(4.8 * len(boost_vals), 5.2),
                             sharey=True)
    vmax_share = 100.0
    for ax, boost in zip(axes, boost_vals):
        M = np.full((len(start_qs), len(top_qs)), np.nan)
        for i, sq in enumerate(start_qs):
            for j, tq in enumerate(top_qs):
                if tq <= sq:
                    continue
                shares = []
                for vals in basin_norm.values():
                    q_s = np.percentile(vals, sq * 100)
                    q_t = np.percentile(vals, tq * 100)
                    w = _ramp_weight(vals, q_s, q_t, boost)
                    total = float(w.sum())
                    extreme = float(w[vals >= q_s].sum())
                    if total > 0:
                        shares.append(extreme / total)
                M[i, j] = 100.0 * float(np.mean(shares))
        im = ax.imshow(M, origin="lower", aspect="auto", cmap="magma",
                       vmin=0.0, vmax=vmax_share)
        ax.set_xticks(range(len(top_qs)))
        ax.set_xticklabels([_pct_label(q) for q in top_qs], rotation=45)
        ax.set_yticks(range(len(start_qs)))
        ax.set_yticklabels([_pct_label(q) for q in start_qs])
        ax.set_xlabel("extreme_top_quantile (percentile)")
        if ax is axes[0]:
            ax.set_ylabel("extreme_start_quantile (percentile)")
        ax.set_title(f"peak_boost = {boost:g}")
        for i in range(len(start_qs)):
            for j in range(len(top_qs)):
                if not np.isnan(M[i, j]):
                    ax.text(j, i, f"{M[i, j]:.0f}%", ha="center", va="center",
                            color="w" if M[i, j] < 50 else "k",
                            fontsize=9)
    fig.colorbar(im, ax=axes, shrink=0.85,
                 label="Gradient share on extreme days (%)")
    fig.suptitle(
        "Share of auxiliary-loss gradient concentrated on extreme days\n"
        "(top [1 − start_q] fraction of each basin's days; "
        "baseline without boost = 1 − start_q)",
        fontweight="bold",
    )
    fig.savefig(out_dir / "ramp_coverage_heatmap.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_dir / 'ramp_coverage_heatmap.png'}")

    # FIGURE 3: per-basin ramp curves
    sq = float(config.extreme_start_quantile)
    tq = float(config.extreme_top_quantile)
    boost = float(config.extreme_peak_boost)
    rng = np.random.default_rng(0)
    picks: list[int] = []
    for t in (1, 2, 3):
        t_ids = [b for b in basin_norm if tier_map[b] == t]
        k = min(len(t_ids), 8 if t != 3 else 4)
        picks.extend(rng.choice(t_ids, size=k, replace=False).tolist())

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=False)
    x_grid = np.linspace(0, 15, 400)
    for bid in picks:
        vals = basin_norm[bid]
        q_s = np.percentile(vals, sq * 100)
        q_t = np.percentile(vals, tq * 100)
        w = _ramp_weight(x_grid, q_s, q_t, boost)
        axes[0].plot(x_grid, w, color=TIER_COLOURS[tier_map[bid]],
                     alpha=0.6, lw=1.2)
    axes[0].set_xlabel("Normalised flow (flow / precip_mean)")
    axes[0].set_ylabel("Sample weight")
    axes[0].set_title(f"Per-basin ramp curves "
                      f"(q_start={sq}, q_top={tq}, boost={boost:g})")
    for t in (1, 2, 3):
        axes[0].plot([], [], color=TIER_COLOURS[t], label=TIER_LABELS[t], lw=2)
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    for bid in picks:
        vals = np.sort(basin_norm[bid])
        cdf = np.linspace(0, 1, len(vals))
        axes[1].plot(vals, cdf, color=TIER_COLOURS[tier_map[bid]],
                     alpha=0.5, lw=1)
        q_s = np.percentile(vals, sq * 100)
        q_t = np.percentile(vals, tq * 100)
        axes[1].axvspan(q_s, q_t, color=TIER_COLOURS[tier_map[bid]],
                        alpha=0.05)
    axes[1].axhline(sq, color="gray", ls="--", lw=0.8, label=f"q_start = {sq}")
    axes[1].axhline(tq, color="gray", ls=":", lw=0.8, label=f"q_top = {tq}")
    axes[1].set_xlabel("Normalised flow (flow / precip_mean)")
    axes[1].set_ylabel("CDF")
    axes[1].set_ylim(0.90, 1.001)
    axes[1].set_xlim(0, 15)
    axes[1].set_title("Per-basin ramp window on each basin's upper CDF")
    axes[1].legend(loc="lower right")
    axes[1].grid(alpha=0.3)

    fig.suptitle("Per-basin quantile thresholds: every basin uses its own p-start / p-top",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / "per_basin_ramp_curves.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_dir / 'per_basin_ramp_curves.png'}")

    print(f"""
-- Per-basin extreme weighting (current config) --
  extreme_start_quantile = {sq}
  extreme_top_quantile   = {tq}
  extreme_peak_boost     = {boost:g}

  By design, each basin has (1 - start_q) = {100 * (1 - sq):.2f}% of its days
  in the ramp and (1 - top_q) = {100 * (1 - tq):.3f}% at peak weight - the
  same fraction for every basin regardless of flow regime.

  The absolute threshold value varies widely across basins (see
  tier_quantile_bars.png): T1 basins typically have the highest
  normalised p99 (flashy storm response) while T2 can have the lowest
  (ephemeral streams with many zero-flow days). A single global
  threshold is therefore a poor fit and is replaced by the per-basin
  quantile scheme.

Outputs saved to: {out_dir}
""")


if __name__ == "__main__":
    cfg = load_config(Path(sys.argv[1])) if len(sys.argv) > 1 else None
    main(cfg)
