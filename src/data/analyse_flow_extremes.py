"""Analyse training target-flow distributions for extreme-loss threshold calibration.

Produces percentile tables (physical mm/d and normalised), per-tier upper-tail
histograms, per-basin P99 scatter plots, weight-ramp visualisation, and a
normalised-space upper-tail histogram.  All outputs land in
``<output_dir>/flow_extremes/``.

Called from ``prepare_data.py --analysis flow_extremes`` or directly::

    python -m src.data.analyse_flow_extremes [config.toml]
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
PCTS = [50, 75, 90, 95, 97.5, 99, 99.5, 99.9, 99.95, 99.99]


def main(config: Config | None = None) -> None:
    """Run the full analysis, writing plots and CSVs to data/prepare/flow_extremes/."""
    if config is None:
        config = load_config()

    out_dir = QA_DIR / "flow_extremes"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── load data ──────────────────────────────────────────────────────
    basin_ids, climate_data, flow_data, static_df, tier_map = load_all_data(config)
    folds = create_folds(basin_ids, tier_map, flow_data,
                         n_folds=config.n_folds, seed=config.seed)

    train_ids, _val_ids = folds[0]
    print(f"\nUsing fold-0 training set: {len(train_ids)} basins")

    # ── collect per-basin flow arrays (mm/d, non-NaN only) ─────────────
    basin_flows: dict[int, np.ndarray] = {}
    for bid in train_ids:
        vals = flow_data[bid]["flow"].dropna().values.astype(np.float64)
        if len(vals) > 0:
            basin_flows[bid] = vals

    all_flows = np.concatenate(list(basin_flows.values()))
    print(f"Total training samples: {len(all_flows):,}")
    print(f"Flow range: [{all_flows.min():.3f}, {all_flows.max():.1f}] mm/d")
    print(f"Mean: {all_flows.mean():.3f}, Median: {np.median(all_flows):.3f}, "
          f"Std: {all_flows.std():.3f} mm/d")

    # ── percentile table ───────────────────────────────────────────────
    rows = []
    for tier in (1, 2, 3):
        tier_vals = np.concatenate([basin_flows[b] for b in train_ids
                                    if tier_map[b] == tier and b in basin_flows])
        row: dict = {"tier": TIER_LABELS[tier], "n_days": len(tier_vals)}
        for p in PCTS:
            row[f"p{p}"] = np.percentile(tier_vals, p)
        rows.append(row)

    row_all: dict = {"tier": "All", "n_days": len(all_flows)}
    for p in PCTS:
        row_all[f"p{p}"] = np.percentile(all_flows, p)
    rows.append(row_all)

    pct_df = pd.DataFrame(rows)
    print("\n── Percentile table (mm/d) ──")
    print(pct_df.to_string(index=False, float_format="{:.2f}".format))
    pct_df.to_csv(out_dir / "percentile_table.csv", index=False, float_format="%.3f")

    # ── fraction above candidate thresholds ────────────────────────────
    thresholds = [5, 10, 15, 20, 30, 50, 75, 100, 150, 200]
    print("\n── Fraction of training samples above threshold ──")
    print(f"{'threshold':>10s}  {'frac':>10s}  {'count':>10s}  {'1/freq':>10s}")
    for t in thresholds:
        n_above = int((all_flows > t).sum())
        frac = n_above / len(all_flows)
        inv_freq = len(all_flows) / max(n_above, 1)
        print(f"{t:>10.0f}  {frac:>10.5f}  {n_above:>10,}  {inv_freq:>10.0f}")

    # ── per-basin p99 ──────────────────────────────────────────────────
    basin_p99 = []
    for bid in train_ids:
        if bid not in basin_flows or len(basin_flows[bid]) < 50:
            continue
        vals = basin_flows[bid]
        basin_p99.append({
            "basin_id": bid,
            "tier": tier_map[bid],
            "p99": np.percentile(vals, 99),
            "p999": np.percentile(vals, 99.9),
            "max": vals.max(),
            "std": vals.std(),
            "n": len(vals),
        })

    bp_df = pd.DataFrame(basin_p99).sort_values("p99", ascending=False)
    bp_df.to_csv(out_dir / "basin_p99.csv", index=False, float_format="%.3f")

    # ── FIGURE 1: upper-tail histogram per tier ────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    for ax, tier in zip(axes, (1, 2, 3)):
        tier_vals = np.concatenate([basin_flows[b] for b in train_ids
                                    if tier_map[b] == tier and b in basin_flows])
        upper = tier_vals[tier_vals > np.percentile(tier_vals, 90)]
        ax.hist(upper, bins=100, color=TIER_COLOURS[tier], alpha=0.8, edgecolor="none")
        ax.set_xlabel("Flow (mm/d)")
        ax.set_title(f"{TIER_LABELS[tier]}  (>{np.percentile(tier_vals, 90):.1f} mm/d)")
        ax.set_yscale("log")
        for thr in [20, 50, 100]:
            if thr < upper.max():
                ax.axvline(thr, color="k", ls="--", lw=0.8, alpha=0.6)
                ax.text(thr + 1, ax.get_ylim()[1] * 0.5, f"{thr}", fontsize=8,
                        rotation=90, va="center")

    axes[0].set_ylabel("Count (log scale)")
    fig.suptitle("Upper-tail flow distribution by tier (>90th percentile)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / "upper_tail_hist.png", dpi=150, bbox_inches="tight")
    print(f"\nSaved: {out_dir / 'upper_tail_hist.png'}")

    # ── FIGURE 2: per-basin P99 scatter ────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    for tier in (1, 2, 3):
        mask = bp_df["tier"] == tier
        ax.scatter(bp_df.loc[mask, "std"], bp_df.loc[mask, "p99"],
                   c=TIER_COLOURS[tier], label=TIER_LABELS[tier],
                   alpha=0.7, s=30, edgecolors="none")
    ax.set_xlabel("Per-basin flow std (mm/d)")
    ax.set_ylabel("Per-basin 99th percentile (mm/d)")
    ax.set_title("Basin-level extreme flow characterisation")
    ax.legend()
    for thr in [20, 50, 100]:
        ax.axhline(thr, color="gray", ls=":", lw=0.8)
        ax.text(ax.get_xlim()[1] * 0.95, thr + 1, f"{thr} mm/d", fontsize=8,
                ha="right", color="gray")
    fig.tight_layout()
    fig.savefig(out_dir / "basin_p99_scatter.png", dpi=150, bbox_inches="tight")
    print(f"Saved: {out_dir / 'basin_p99_scatter.png'}")

    # ── FIGURE 3: weight ramp visualisation ────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    flow_range = np.linspace(0, 200, 500)

    ramp_configs = [
        {"threshold": 20, "ramp": 80, "max_w": 25, "label": "Conservative\n(20/80/25×)"},
        {"threshold": 10, "ramp": 40, "max_w": 15, "label": "Moderate\n(10/40/15×)"},
        {"threshold": 30, "ramp": 70, "max_w": 50, "label": "Aggressive\n(30/70/50×)"},
    ]

    for ax, rcfg in zip(axes, ramp_configs):
        frac = np.clip((flow_range - rcfg["threshold"]) / rcfg["ramp"], 0, 1)
        weight = 1.0 + (rcfg["max_w"] - 1.0) * frac
        ax.plot(flow_range, weight, "k-", lw=2)
        ax.fill_between(flow_range, 1, weight, alpha=0.15, color="crimson")
        ax.set_xlabel("Observed flow (mm/d)")
        ax.set_ylabel("Sample weight")
        ax.set_title(rcfg["label"])
        ax.axvline(rcfg["threshold"], color="gray", ls="--", lw=0.8)
        ax.axvline(rcfg["threshold"] + rcfg["ramp"], color="gray", ls="--", lw=0.8)

        ax2 = ax.twinx()
        sorted_f = np.sort(all_flows)
        cdf_y = np.arange(1, len(sorted_f) + 1) / len(sorted_f)
        ax2.plot(sorted_f, cdf_y, color="steelblue", alpha=0.4, lw=1)
        ax2.set_ylabel("CDF (all flows)", color="steelblue")
        ax2.set_ylim(0.9, 1.001)
        ax2.tick_params(axis="y", labelcolor="steelblue")

    fig.suptitle("Extreme-weight ramp candidates with flow CDF overlay", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / "weight_ramp_candidates.png", dpi=150, bbox_inches="tight")
    print(f"Saved: {out_dir / 'weight_ramp_candidates.png'}")

    # ── FIGURE 4: normalised-space view ────────────────────────────────
    print("\n── Normalised-space statistics ──")
    norm_chunks = []
    for bid in train_ids:
        if bid not in basin_flows:
            continue
        vals = basin_flows[bid]
        std = max(vals.std(), 0.01)
        norm_chunks.append(vals / std)

    all_norm = np.concatenate(norm_chunks)
    print(f"Normalised flow range: [{all_norm.min():.3f}, {all_norm.max():.1f}]")
    for p in PCTS:
        print(f"  p{p:>6.2f}: {np.percentile(all_norm, p):.3f}")

    fig, ax = plt.subplots(figsize=(10, 5))
    upper_norm = all_norm[all_norm > np.percentile(all_norm, 90)]
    ax.hist(upper_norm, bins=150, color="steelblue", alpha=0.8, edgecolor="none")
    ax.set_yscale("log")
    ax.set_xlabel("Normalised flow (flow / basin_std)")
    ax.set_ylabel("Count (log scale)")
    ax.set_title("Upper-tail distribution in normalised space (>90th pct)")

    for p, ls in [(95, ":"), (99, "--"), (99.9, "-")]:
        val = np.percentile(all_norm, p)
        ax.axvline(val, color="crimson", ls=ls, lw=1)
        ax.text(val + 0.1, ax.get_ylim()[1] * 0.3, f"p{p}\n({val:.1f})",
                fontsize=8, color="crimson")

    fig.tight_layout()
    fig.savefig(out_dir / "normalised_upper_tail.png", dpi=150, bbox_inches="tight")
    print(f"Saved: {out_dir / 'normalised_upper_tail.png'}")

    # ── calibration summary ────────────────────────────────────────────
    p95 = np.percentile(all_flows, 95)
    p99 = np.percentile(all_flows, 99)
    p999 = np.percentile(all_flows, 99.9)
    p9999 = np.percentile(all_flows, 99.99)

    print(f"""
── Calibration summary ──
Physical space (mm/d):
  p95   = {p95:>8.2f}  (1 in {100 / 5:.0f} days)
  p99   = {p99:>8.2f}  (1 in {100 / 1:.0f} days)
  p99.9 = {p999:>8.2f}  (1 in 1,000 days ≈ 3 years)
  p99.99= {p9999:>8.2f}  (1 in 10,000 days ≈ 27 years)

Suggested threshold_mm:
  Conservative: {p95:.0f} mm/d (≈p95, upweights top 5%)
  Moderate:     {p99:.0f} mm/d (≈p99, upweights top 1%)
  Tight:        {p999:.0f} mm/d (≈p99.9, upweights top 0.1%)

Suggested ramp_mm (threshold → max_weight):
  {p99 - p95:.0f} mm/d (p95 → p99 span)
  {p999 - p99:.0f} mm/d (p99 → p99.9 span)
  {p9999 - p99:.0f} mm/d (p99 → p99.99 span)

Outputs saved to: {out_dir}
""")


if __name__ == "__main__":
    main()
