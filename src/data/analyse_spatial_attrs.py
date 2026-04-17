"""Spatial attribute analysis: redundancy, discriminative power, and feature selection.

Loads all static attribute CSVs (Physical, Climate, DEM, Network, Width Function),
computes pairwise correlations, hierarchical clustering, mutual information with tier,
RF importance, incremental information over a baseline feature set, and flow-target
correlations.  Writes a text report and diagnostic plots to
``data/prepare/spatial_analysis/``.

Called from ``prepare_data.py --analysis spatial_attributes`` or directly::

    python -m src.data.analyse_spatial_attrs
"""
from __future__ import annotations

import sys
import warnings
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.spatial.distance import squareform
from scipy.stats import kruskal, spearmanr
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

from src.paths import FLOW_DIR, QA_DIR, STATIC_DIR

TIER_LABELS = {1: "T1 (rain)", 2: "T2 (mixed)", 3: "T3 (snow)"}
TIER_COLOURS = {1: "#e66101", 2: "#5e3c99", 3: "#0571b0"}
CORR_CUTOFF = 0.85  # |ρ| threshold for "highly correlated"
CLUSTER_DIST = 1 - CORR_CUTOFF  # distance cutoff for clustering


# ── helpers ────────────────────────────────────────────────────────────────


def _load_static(static_dir: Path) -> pd.DataFrame:
    """Merge all static attribute CSVs on PourPtID."""
    ba = pd.read_csv(static_dir / "Physical_Attributes_Watersheds.csv")
    clm = pd.read_csv(static_dir / "Climate_Statistics_Watersheds.csv")
    dem = pd.read_csv(static_dir / "DEM_Attributes_Watersheds.csv")
    net = pd.read_csv(static_dir / "Network_Attributes_Watersheds.csv")
    df = ba.merge(clm, on="PourPtID").merge(dem, on="PourPtID", how="left").merge(
        net, on="PourPtID", how="left"
    )
    return df


def _load_width_functions(static_dir: Path) -> tuple[pd.DataFrame, list[str]]:
    """Load width function CSV; return (df, wf_col_names)."""
    wf = pd.read_csv(static_dir / "Width_Functions_Watersheds.csv")
    wf_cols = [c for c in wf.columns if c != "PourPtID"]
    return wf, wf_cols


def _tier_map(flow_dir: Path) -> dict[int, int]:
    """Build basin → tier mapping from flow directory structure."""
    tmap: dict[int, int] = {}
    for t in [1, 2, 3]:
        for fp in (flow_dir / f"tier_{t}").glob("*_cleaned.csv"):
            bid = int(fp.stem.split("_")[0])
            tmap[bid] = t
    return tmap


def _flow_statistics(flow_dir: Path) -> pd.DataFrame:
    """Compute per-basin flow variability statistics."""
    rows = []
    for t in [1, 2, 3]:
        for fp in sorted((flow_dir / f"tier_{t}").glob("*_cleaned.csv")):
            bid = int(fp.stem.split("_")[0])
            try:
                q = pd.read_csv(fp, usecols=["flow"])["flow"].dropna()
                if len(q) < 365:
                    continue
                rows.append(
                    {
                        "PourPtID": bid,
                        "q_cv": q.std() / q.mean() if q.mean() > 0 else np.nan,
                        "q_skew": q.skew(),
                        "baseflow_idx": q.quantile(0.1) / q.mean() if q.mean() > 0 else np.nan,
                        "flashiness": q.diff().abs().mean() / q.mean() if q.mean() > 0 else np.nan,
                        "zero_flow_frac": (q == 0).mean(),
                    }
                )
            except Exception:
                pass
    return pd.DataFrame(rows)


def _scalar_feat_cols(df: pd.DataFrame) -> list[str]:
    """Return all numeric columns except PourPtID and tier."""
    exclude = {"PourPtID", "tier"}
    return [
        c
        for c in df.columns
        if c not in exclude and df[c].dtype in (np.float64, np.int64, float, int)
    ]


# ── plots ──────────────────────────────────────────────────────────────────


def _plot_correlation_heatmap(
    spear: pd.DataFrame, feat_cols: list[str], out_dir: Path
) -> None:
    """Clustered heatmap of |Spearman ρ|."""
    abs_corr = spear.loc[feat_cols, feat_cols].abs()
    dist = 1 - abs_corr.values
    np.fill_diagonal(dist, 0)
    dist = np.clip((dist + dist.T) / 2, 0, None)
    condensed = squareform(dist, checks=False)
    Z = linkage(condensed, method="average")
    order = dendrogram(Z, no_plot=True)["leaves"]
    ordered_feats = [feat_cols[i] for i in order]
    ordered_corr = abs_corr.loc[ordered_feats, ordered_feats]

    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(ordered_corr.values, vmin=0, vmax=1, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(ordered_feats)))
    ax.set_yticks(range(len(ordered_feats)))
    ax.set_xticklabels(ordered_feats, rotation=90, fontsize=7)
    ax.set_yticklabels(ordered_feats, fontsize=7)
    fig.colorbar(im, ax=ax, label="|Spearman ρ|", shrink=0.7)
    ax.set_title("Clustered |Spearman| correlation – all static features")
    fig.tight_layout()
    fig.savefig(out_dir / "correlation_heatmap.png", dpi=200)
    plt.close(fig)


def _plot_dendrogram(feat_cols: list[str], spear: pd.DataFrame, out_dir: Path) -> None:
    """Dendrogram of hierarchical feature clustering."""
    abs_corr = spear.loc[feat_cols, feat_cols].abs().values
    dist = 1 - abs_corr
    np.fill_diagonal(dist, 0)
    dist = np.clip((dist + dist.T) / 2, 0, None)
    condensed = squareform(dist, checks=False)
    Z = linkage(condensed, method="average")

    fig, ax = plt.subplots(figsize=(14, 6))
    dendrogram(
        Z,
        labels=feat_cols,
        ax=ax,
        leaf_rotation=90,
        leaf_font_size=7,
        color_threshold=CLUSTER_DIST,
    )
    ax.axhline(y=CLUSTER_DIST, ls="--", color="grey", lw=0.8, label=f"|ρ|={CORR_CUTOFF}")
    ax.set_ylabel("1 − |Spearman ρ|")
    ax.set_title("Hierarchical clustering of static features")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "feature_dendrogram.png", dpi=200)
    plt.close(fig)


def _plot_incremental_info(
    residuals: list[tuple[str, float, float]], out_dir: Path
) -> None:
    """Horizontal bar chart of residual variance (unique information)."""
    names = [r[0] for r in residuals]
    resid = [r[2] for r in residuals]

    fig, ax = plt.subplots(figsize=(8, max(4, len(names) * 0.3)))
    colours = ["#2166ac" if v > 0.50 else ("#92c5de" if v > 0.20 else "#d6604d") for v in resid]
    ax.barh(range(len(names)), resid, color=colours)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=7)
    ax.set_xlabel("Residual variance (1 − R² vs existing 16)")
    ax.set_title("Unique information in new features")
    ax.axvline(0.50, ls="--", color="grey", lw=0.7, label="50%")
    ax.axvline(0.20, ls=":", color="grey", lw=0.7, label="20%")
    ax.legend(fontsize=7)
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(out_dir / "incremental_information.png", dpi=200)
    plt.close(fig)


def _plot_mi_importance(
    mi_df: pd.DataFrame, imp_df: pd.DataFrame, out_dir: Path
) -> None:
    """Side-by-side MI(tier) and RF Gini importance."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    top = mi_df.head(25)
    axes[0].barh(range(len(top)), top["mi_tier"].values, color="#5e3c99")
    axes[0].set_yticks(range(len(top)))
    axes[0].set_yticklabels(top["feature"].values, fontsize=7)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Mutual information with tier")
    axes[0].set_title("MI(tier) — top 25")

    top_imp = imp_df.head(25)
    axes[1].barh(range(len(top_imp)), top_imp["importance"].values, color="#e66101")
    axes[1].set_yticks(range(len(top_imp)))
    axes[1].set_yticklabels(top_imp["feature"].values, fontsize=7)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("RF Gini importance")
    axes[1].set_title("Random Forest importance — top 25")

    fig.tight_layout()
    fig.savefig(out_dir / "mi_and_rf_importance.png", dpi=200)
    plt.close(fig)


def _plot_within_tier_corr(
    df_full: pd.DataFrame,
    feat_cols: list[str],
    targets: list[str],
    out_dir: Path,
) -> None:
    """Heatmap of within-tier |Spearman ρ| between features and flow targets."""
    fig, axes = plt.subplots(1, 3, figsize=(18, max(5, len(feat_cols) * 0.22)), sharey=True)
    for idx, t in enumerate([1, 2, 3]):
        tier_sub = df_full[df_full["tier"] == t]
        corr_mat = np.zeros((len(feat_cols), len(targets)))
        for i, fc in enumerate(feat_cols):
            for j, tgt in enumerate(targets):
                mask = tier_sub[fc].notna() & tier_sub[tgt].notna()
                if mask.sum() > 10:
                    rho, _ = spearmanr(tier_sub.loc[mask, fc], tier_sub.loc[mask, tgt])
                    corr_mat[i, j] = abs(rho)
        im = axes[idx].imshow(corr_mat, vmin=0, vmax=0.7, cmap="YlOrRd", aspect="auto")
        axes[idx].set_xticks(range(len(targets)))
        axes[idx].set_xticklabels(targets, rotation=45, ha="right", fontsize=7)
        axes[idx].set_title(f"{TIER_LABELS[t]} ({len(tier_sub)})", fontsize=9)
        if idx == 0:
            axes[idx].set_yticks(range(len(feat_cols)))
            axes[idx].set_yticklabels(feat_cols, fontsize=6)
    fig.colorbar(im, ax=axes, label="|Spearman ρ|", shrink=0.5)
    fig.suptitle("Within-tier |ρ| between static features and flow statistics", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / "within_tier_flow_correlations.png", dpi=200)
    plt.close(fig)


# ── main analysis ──────────────────────────────────────────────────────────


def main() -> None:
    """Run the full spatial attribute analysis."""
    warnings.filterwarnings("ignore")
    out_dir = QA_DIR / "spatial_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "spatial_analysis_report.txt"
    lines: list[str] = []

    def log(msg: str = "") -> None:
        print(msg)
        lines.append(msg)

    # ── load data ──────────────────────────────────────────────────────
    df = _load_static(STATIC_DIR)
    wf, wf_cols = _load_width_functions(STATIC_DIR)
    tmap = _tier_map(FLOW_DIR)
    df["tier"] = df["PourPtID"].map(tmap)
    df = df.dropna(subset=["tier"])
    df["tier"] = df["tier"].astype(int)

    feat_cols = _scalar_feat_cols(df)

    log(f"Basins with tier labels: {len(df)}")
    log(f"Candidate scalar features: {len(feat_cols)}")
    log(f"Width function bins: {len(wf_cols)}")

    # Impute NaN with median for analysis
    for c in feat_cols:
        if df[c].isna().any():
            df[c] = df[c].fillna(df[c].median())

    # ── PART 1: Pairwise Spearman ──────────────────────────────────────
    log("\n" + "=" * 90)
    log(f"PART 1: PAIRWISE |SPEARMAN rho| > {CORR_CUTOFF}")
    log("=" * 90)

    spear = df[feat_cols].corr(method="spearman")
    pairs_high = []
    for i, ci in enumerate(feat_cols):
        for j in range(i + 1, len(feat_cols)):
            cj = feat_cols[j]
            rho = spear.loc[ci, cj]
            if abs(rho) > CORR_CUTOFF:
                pairs_high.append((ci, cj, rho))
    pairs_high.sort(key=lambda x: -abs(x[2]))
    for ci, cj, rho in pairs_high:
        log(f"  {ci:28s} vs {cj:28s}  ρ = {rho:+.4f}")

    # ── PART 2: Hierarchical clustering ────────────────────────────────
    log("\n" + "=" * 90)
    log(f"PART 2: HIERARCHICAL CLUSTERING (distance = 1 − |Spearman|, cut at {CLUSTER_DIST:.2f})")
    log("=" * 90)

    abs_corr = spear.abs().values
    dist_matrix = 1 - abs_corr
    np.fill_diagonal(dist_matrix, 0)
    dist_matrix = np.clip((dist_matrix + dist_matrix.T) / 2, 0, None)
    condensed = squareform(dist_matrix, checks=False)
    Z = linkage(condensed, method="average")
    clusters = fcluster(Z, t=CLUSTER_DIST, criterion="distance")
    cluster_df = pd.DataFrame({"feature": feat_cols, "cluster": clusters}).sort_values("cluster")

    log("\nClusters of highly correlated features:")
    for cid in sorted(cluster_df["cluster"].unique()):
        members = cluster_df[cluster_df["cluster"] == cid]["feature"].tolist()
        if len(members) > 1:
            log(f"\n  Cluster {cid} ({len(members)} features):")
            for m in members:
                log(f"    - {m}")

    log("\n  Singletons:")
    for cid in sorted(cluster_df["cluster"].unique()):
        members = cluster_df[cluster_df["cluster"] == cid]["feature"].tolist()
        if len(members) == 1:
            log(f"    - {members[0]}")

    # ── PART 3: MI with tier ───────────────────────────────────────────
    log("\n" + "=" * 90)
    log("PART 3: MUTUAL INFORMATION WITH TIER")
    log("=" * 90)

    X_scaled = StandardScaler().fit_transform(df[feat_cols])
    mi_tier = mutual_info_classif(X_scaled, df["tier"], discrete_features=False, random_state=42)
    mi_df = pd.DataFrame({"feature": feat_cols, "mi_tier": mi_tier}).sort_values(
        "mi_tier", ascending=False
    )

    log("\n  (higher = more discriminative for T1/T2/T3)")
    for _, row in mi_df.iterrows():
        log(f"  {row['feature']:28s}  MI = {row['mi_tier']:.4f}")

    # ── PART 4: RF importance ──────────────────────────────────────────
    log("\n" + "=" * 90)
    log("PART 4: RANDOM FOREST TIER CLASSIFICATION — feature importance")
    log("=" * 90)

    rf = RandomForestClassifier(
        n_estimators=500, max_depth=8, random_state=42, class_weight="balanced"
    )
    rf.fit(X_scaled, df["tier"])
    imp = rf.feature_importances_
    imp_df = pd.DataFrame({"feature": feat_cols, "importance": imp}).sort_values(
        "importance", ascending=False
    )

    cv_all = cross_val_score(rf, X_scaled, df["tier"], cv=5, scoring="accuracy").mean()
    log(f"\n  5-fold CV accuracy (all {len(feat_cols)} features): {cv_all:.3f}")
    log("\n  Feature importance (Gini):")
    cumul = 0.0
    for _, row in imp_df.iterrows():
        cumul += row["importance"]
        log(f"  {row['feature']:28s}  imp = {row['importance']:.4f}  cumul = {cumul:.3f}")

    # ── PART 5: Cluster-based selection ────────────────────────────────
    log("\n" + "=" * 90)
    log("PART 5: CLUSTER-BASED SELECTION (keep highest-MI member per cluster)")
    log("=" * 90)

    mi_lookup = mi_df.set_index("feature")["mi_tier"]
    cluster_reps: dict[int, dict] = {}
    for cid in sorted(cluster_df["cluster"].unique()):
        members = cluster_df[cluster_df["cluster"] == cid]["feature"].tolist()
        best = max(members, key=lambda f: mi_lookup[f])
        worst = [m for m in members if m != best]
        cluster_reps[cid] = {"keep": best, "drop": worst}
        if len(members) > 1:
            log(f"  Cluster {cid}: KEEP {best}, drop {worst}")

    kept = [v["keep"] for v in cluster_reps.values()]
    dropped = [f for v in cluster_reps.values() for f in v["drop"]]
    log(f"\n  Kept ({len(kept)}): {kept}")
    log(f"  Dropped ({len(dropped)}): {dropped}")

    # VIF on kept set
    log("\n  VIF on kept set:")
    X_kept = df[kept]
    X_kz = (X_kept - X_kept.mean()) / (X_kept.std() + 1e-9)
    vifs = []
    for i, col in enumerate(kept):
        others = [c for j, c in enumerate(kept) if j != i]
        lr = LinearRegression().fit(X_kz[others], X_kz[col])
        r2 = lr.score(X_kz[others], X_kz[col])
        vif = 1.0 / (1.0 - r2) if r2 < 1.0 else float("inf")
        vifs.append((col, vif, r2))
    vifs.sort(key=lambda x: -x[1])
    for col, vif, r2 in vifs:
        flag = " *** HIGH" if vif > 10 else (" * moderate" if vif > 5 else "")
        log(f"    {col:28s}  VIF={vif:8.1f}  R²={r2:.3f}{flag}")

    X_kept_sc = StandardScaler().fit_transform(X_kept)
    cv_kept = cross_val_score(
        RandomForestClassifier(n_estimators=500, max_depth=8, random_state=42, class_weight="balanced"),
        X_kept_sc,
        df["tier"],
        cv=5,
        scoring="accuracy",
    ).mean()
    log(f"\n  5-fold CV (kept {len(kept)}): {cv_kept:.3f}  (was {cv_all:.3f} with all {len(feat_cols)})")

    # ── PART 6: Width function tier analysis ───────────────────────────
    log("\n" + "=" * 90)
    log("PART 6: WIDTH FUNCTION — tier-stratified analysis")
    log("=" * 90)

    df_wf = df[["PourPtID", "tier"]].merge(wf, on="PourPtID")
    for t in [1, 2, 3]:
        vals = df_wf[df_wf["tier"] == t][wf_cols].values
        if len(vals) == 0:
            continue
        mean_wf = vals.mean(axis=0)
        peak_bin = int(np.argmax(mean_wf))
        centroid = float(np.average(np.arange(len(wf_cols)), weights=mean_wf))
        entropy = float(-np.sum(mean_wf * np.log(mean_wf + 1e-12)))
        log(
            f"  {TIER_LABELS[t]} ({len(vals):3d} basins): peak_bin={peak_bin:2d}  "
            f"centroid={centroid:.1f}  entropy={entropy:.3f}"
        )

    wf_all = df_wf[wf_cols].dropna().values
    pca = PCA(n_components=6).fit(wf_all)
    wf_pcs = pca.transform(wf_all)

    log(f"\n  PCA on width function (6 PCs, {pca.explained_variance_ratio_.sum() * 100:.1f}% variance):")
    for k in range(6):
        log(f"    PC{k + 1}: {pca.explained_variance_ratio_[k] * 100:.1f}%")

    mi_wf_tier = mutual_info_classif(
        wf_pcs, df_wf["tier"].values, discrete_features=False, random_state=42
    )
    log("\n  Width function PC MI with tier:")
    for k in range(6):
        log(f"    PC{k + 1}: MI = {mi_wf_tier[k]:.4f}")

    log("\n  Kruskal-Wallis (tier differences in WF PCs):")
    for k in range(6):
        groups = [wf_pcs[df_wf["tier"].values == t, k] for t in [1, 2, 3]]
        stat, p = kruskal(*groups)
        sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
        log(f"    PC{k + 1}: H={stat:8.2f}  p={p:.2e}  {sig}")

    # ── baseline feature set (used by Parts 7c, 8, 9) ─────────────────
    existing_16 = [
        "total_Shape_Area_km2", "ele_mt_uav", "slp_dg_uav",
        "for_pc_use", "cly_pc_uav", "snd_pc_uav",
        "ria_ha_usu", "riv_tc_usu", "sgr_dk_sav",
        "precip_mean", "pet_mean", "aridity_index", "snow_fraction",
        "low_precip_dur", "clz_cl_smj", "lit_cl_smj",
    ]

    # ── PART 7: Flow-target correlations ───────────────────────────────
    log("\n" + "=" * 90)
    log("PART 7: FEATURE CORRELATION WITH HYDROLOGICAL TARGETS (Spearman, all basins)")
    log("=" * 90)

    fstat_df = _flow_statistics(FLOW_DIR)
    df_full = df.merge(fstat_df, on="PourPtID")
    targets = ["q_cv", "q_skew", "baseflow_idx", "flashiness", "zero_flow_frac"]

    log(f"\n  Flow stats computed for {len(fstat_df)} basins")
    for tgt in targets:
        log(f"\n  --- {tgt} ---")
        corrs = []
        for fc in feat_cols:
            if fc in df_full.columns:
                mask = df_full[fc].notna() & df_full[tgt].notna()
                if mask.sum() > 10:
                    rho, p = spearmanr(df_full.loc[mask, fc], df_full.loc[mask, tgt])
                    corrs.append((fc, rho, p))
        corrs.sort(key=lambda x: -abs(x[1]))
        for fc, rho, p in corrs[:10]:
            sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
            log(f"    {fc:28s}  ρ = {rho:+.4f}  {sig}")

    # ── PART 7a: Within-tier flow-target correlations ──────────────────
    log("\n" + "=" * 90)
    log("PART 7a: WITHIN-TIER FLOW-TARGET CORRELATIONS (T2 focus)")
    log("=" * 90)
    log("  Features that distinguish basins *within* a tier matter most for the model.")
    log("  Cross-tier signal (e.g. snow_fraction separating T1 from T3) inflates Part 7.")
    log("  Here we isolate within-tier discriminative power.\n")

    for t in [1, 2, 3]:
        tier_sub = df_full[df_full["tier"] == t]
        n = len(tier_sub)
        log(f"\n  {'=' * 40}")
        log(f"  {TIER_LABELS[t]} — {n} basins")
        log(f"  {'=' * 40}")
        for tgt in targets:
            vals = tier_sub[tgt].dropna()
            if len(vals) < 10:
                continue
            log(f"\n    --- {tgt} (within {TIER_LABELS[t]}, std={vals.std():.3f}) ---")
            corrs = []
            for fc in feat_cols:
                if fc in tier_sub.columns:
                    mask = tier_sub[fc].notna() & tier_sub[tgt].notna()
                    if mask.sum() > 10:
                        rho, p = spearmanr(
                            tier_sub.loc[mask, fc], tier_sub.loc[mask, tgt]
                        )
                        corrs.append((fc, rho, p))
            corrs.sort(key=lambda x: -abs(x[1]))
            for fc, rho, p in corrs[:8]:
                sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
                log(f"      {fc:28s}  ρ = {rho:+.4f}  {sig}")

    # ── PART 7b: Within-tier feature variance ──────────────────────────
    log("\n" + "=" * 90)
    log("PART 7b: WITHIN-TIER FEATURE VARIANCE")
    log("=" * 90)
    log("  Features with low CoV within a tier cannot distinguish basins in that tier.")
    log("  Features with high within-tier CoV relative to total CoV carry intra-tier info.\n")

    total_cv = {}
    for fc in feat_cols:
        s = df[fc].std()
        m = df[fc].mean()
        total_cv[fc] = abs(s / m) if abs(m) > 1e-9 else 0.0

    for t in [2, 1, 3]:  # T2 first — most important
        tier_sub = df[df["tier"] == t]
        log(f"\n  {TIER_LABELS[t]} — {len(tier_sub)} basins:")
        within_cv = []
        for fc in feat_cols:
            s = tier_sub[fc].std()
            m = tier_sub[fc].mean()
            cv = abs(s / m) if abs(m) > 1e-9 else 0.0
            ratio = cv / total_cv[fc] if total_cv[fc] > 1e-9 else 0.0
            within_cv.append((fc, cv, ratio))
        # Show features with highest within-tier CoV fraction (most useful for intra-tier)
        within_cv.sort(key=lambda x: -x[2])
        log("    Top 15 features by within-tier CoV / total CoV (higher = more intra-tier spread):")
        for fc, cv, ratio in within_cv[:15]:
            log(f"      {fc:28s}  within_CoV={cv:.3f}  ratio={ratio:.2f}")
        log("    Bottom 5:")
        for fc, cv, ratio in within_cv[-5:]:
            log(f"      {fc:28s}  within_CoV={cv:.3f}  ratio={ratio:.2f}")

    # ── PART 7c: Within-tier RF regression of flow stats ───────────────
    log("\n" + "=" * 90)
    log("PART 7c: WITHIN-TIER RF REGRESSION — can features predict flow variability?")
    log("=" * 90)
    log("  RF R² for predicting flow statistics from static features, *within each tier*.")
    log("  Positive R² means features explain why basins within a tier differ.\n")

    for t in [2, 1, 3]:
        tier_sub = df_full[df_full["tier"] == t].dropna(subset=targets)
        n = len(tier_sub)
        if n < 15:
            log(f"  {TIER_LABELS[t]}: too few basins ({n}), skipping")
            continue
        log(f"\n  {TIER_LABELS[t]} — {n} basins:")

        for subset_name, subset_cols in [
            ("Existing 16", existing_16),
            ("Recommended 23", existing_16 + [
                "ele_range", "hyp_integral", "twi_std",
                "drain_density_km", "bifurc_ratio", "main_chan_sinuosity",
                "high_precip_dur",
            ]),
        ]:
            valid_cols = [c for c in subset_cols if c in tier_sub.columns]
            X_t = StandardScaler().fit_transform(tier_sub[valid_cols].fillna(0))
            log(f"    {subset_name} ({len(valid_cols)} features):")
            for tgt in targets:
                y_t = tier_sub[tgt].values
                if np.std(y_t) < 1e-9:
                    continue
                # LOO-style CV for small n, else 5-fold
                cv_folds = min(5, n)
                scores = cross_val_score(
                    RandomForestRegressor(
                        n_estimators=200, max_depth=5, random_state=42
                    ),
                    X_t, y_t, cv=cv_folds, scoring="r2",
                )
                log(f"      {tgt:20s}  R²={scores.mean():+.3f} ± {scores.std():.3f}")

    # ── PART 8: Incremental information over existing 16 ──────────────
    log("\n" + "=" * 90)
    log("PART 8: INCREMENTAL INFORMATION — residual variance vs existing 16 features")
    log("=" * 90)

    new_features = [c for c in feat_cols if c not in existing_16]

    log(f"\n  Existing baseline: {len(existing_16)} features")
    log(f"  New candidates: {len(new_features)} features")

    X_exist = StandardScaler().fit_transform(df[existing_16])
    residuals = []
    for nf in new_features:
        y = df[nf].values
        mask = np.isfinite(y)
        if mask.sum() < 50:
            continue
        lr = LinearRegression().fit(X_exist[mask], y[mask])
        r2 = lr.score(X_exist[mask], y[mask])
        resid_var = 1 - r2
        residuals.append((nf, r2, resid_var))
    residuals.sort(key=lambda x: -x[2])

    log("\n  (higher residual = more unique information)\n")
    for nf, r2, rv in residuals:
        label = "  <-- redundant" if rv < 0.10 else ("  <-- UNIQUE" if rv > 0.50 else "")
        log(f"  {nf:28s}  R²={r2:.3f}  residual={rv:.3f}{label}")

    # ── PART 9: Subset comparison — within-tier flow prediction ──────
    log("\n" + "=" * 90)
    log("PART 9: FEATURE SUBSET COMPARISON — within-T2 flow prediction + tier accuracy")
    log("=" * 90)
    log("  Tier accuracy is a sanity check; within-T2 R² is the real test.")
    log("  The model needs features to distinguish basins within each tier.\n")

    subsets: dict[str, list[str]] = {
        "Existing 16 only": existing_16,
        "+ ele_range, hyp_integral": existing_16 + ["ele_range", "hyp_integral"],
        "+ TWI (twi_p50, twi_std)": existing_16 + ["twi_p50", "twi_std"],
        "+ network (dd, bf, sin)": existing_16
        + ["drain_density_km", "bifurc_ratio", "main_chan_sinuosity"],
        "+ high_precip_dur": existing_16 + ["high_precip_dur"],
        "Recommended 23": existing_16
        + [
            "ele_range",
            "hyp_integral",
            "twi_std",
            "drain_density_km",
            "bifurc_ratio",
            "main_chan_sinuosity",
            "high_precip_dur",
        ],
    }

    # Tier classification (sanity check)
    log("  A. Tier classification accuracy (5-fold CV):")
    for name, cols in subsets.items():
        valid_cols = [c for c in cols if c in df.columns]
        X = StandardScaler().fit_transform(df[valid_cols].fillna(df[valid_cols].median()))
        acc = cross_val_score(
            RandomForestClassifier(
                n_estimators=500, max_depth=8, random_state=42, class_weight="balanced"
            ),
            X,
            df["tier"].astype(int),
            cv=5,
            scoring="accuracy",
        ).mean()
        log(f"    {name:40s}  ({len(valid_cols):2d} feat)  acc = {acc:.3f}")

    # Within-T2 flow prediction (the real test)
    t2_sub = df_full[df_full["tier"] == 2].dropna(subset=targets)
    n_t2 = len(t2_sub)
    log(f"\n  B. Within-T2 RF regression ({n_t2} basins, 5-fold CV R²):")
    log(f"     Target: q_cv (flow coefficient of variation — captures basin response diversity)\n")

    for name, cols in subsets.items():
        valid_cols = [c for c in cols if c in t2_sub.columns]
        X_t2 = StandardScaler().fit_transform(t2_sub[valid_cols].fillna(0))
        y_t2 = t2_sub["q_cv"].values
        cv_folds = min(5, n_t2)
        scores = cross_val_score(
            RandomForestRegressor(n_estimators=200, max_depth=5, random_state=42),
            X_t2, y_t2, cv=cv_folds, scoring="r2",
        )
        log(f"    {name:40s}  ({len(valid_cols):2d} feat)  R²={scores.mean():+.3f} ± {scores.std():.3f}")

    # Also test flashiness within T2
    log(f"\n     Target: flashiness\n")
    for name, cols in subsets.items():
        valid_cols = [c for c in cols if c in t2_sub.columns]
        X_t2 = StandardScaler().fit_transform(t2_sub[valid_cols].fillna(0))
        y_t2 = t2_sub["flashiness"].values
        cv_folds = min(5, n_t2)
        scores = cross_val_score(
            RandomForestRegressor(n_estimators=200, max_depth=5, random_state=42),
            X_t2, y_t2, cv=cv_folds, scoring="r2",
        )
        log(f"    {name:40s}  ({len(valid_cols):2d} feat)  R²={scores.mean():+.3f} ± {scores.std():.3f}")

    # Also test baseflow_idx within T2
    log(f"\n     Target: baseflow_idx\n")
    for name, cols in subsets.items():
        valid_cols = [c for c in cols if c in t2_sub.columns]
        X_t2 = StandardScaler().fit_transform(t2_sub[valid_cols].fillna(0))
        y_t2 = t2_sub["baseflow_idx"].values
        cv_folds = min(5, n_t2)
        scores = cross_val_score(
            RandomForestRegressor(n_estimators=200, max_depth=5, random_state=42),
            X_t2, y_t2, cv=cv_folds, scoring="r2",
        )
        log(f"    {name:40s}  ({len(valid_cols):2d} feat)  R²={scores.mean():+.3f} ± {scores.std():.3f}")

    # ── write report ───────────────────────────────────────────────────
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nReport written to {report_path}")

    # ── plots ──────────────────────────────────────────────────────────
    _plot_correlation_heatmap(spear, feat_cols, out_dir)
    _plot_dendrogram(feat_cols, spear, out_dir)
    _plot_incremental_info(residuals, out_dir)
    _plot_mi_importance(mi_df, imp_df, out_dir)
    _plot_within_tier_corr(df_full, feat_cols, targets, out_dir)
    print(f"Plots written to {out_dir}/")


if __name__ == "__main__":
    main()
