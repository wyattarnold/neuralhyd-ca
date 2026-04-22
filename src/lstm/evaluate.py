"""Per-basin evaluation and fold-level result aggregation.

Predictions are denormalised (multiplied by per-basin precip_mean) before
metrics are computed so that NSE/KGE/FHV/FLV are in physical units
(mm/day) and comparable across basins of different size.

Key exports
-----------
evaluate_basin(model, dataset, basin_id, norm_stats, config, device)
    Run inference for a single basin and return a metrics dict plus the
    observed-vs-predicted timeseries DataFrame.
evaluate_fold(model, val_basins, basin_data, norm_stats, config, device)
    Evaluate all held-out basins in a fold; write per-basin CSVs to
    ``output_dir/fold_<n>/timeseries/`` and return a combined
    ``basin_results.csv`` DataFrame with tier-grouped summary rows.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
import torch

from .config import Config
from .dataset import HydroDataset, make_lookback_batch
from .loss import (
    compute_kge, compute_nse, compute_fhv, compute_fehv, compute_flv,
    cmal_quantiles_np,
)
from .train import _unwrap_model

# Quantile levels for CMAL prediction intervals
_CMAL_QUANTILES = [0.05, 0.25, 0.5, 0.75, 0.95]


@torch.no_grad()
def evaluate_basin(
    model: torch.nn.Module,
    basin_data: dict,
    config: Config,
    device: torch.device,
) -> dict | None:
    """Run the model on every valid timestep of a single basin.

    Returns a dict with nse, kge, n_obs, obs, pred, fast, slow –
    or None when the basin has no valid samples.
    """
    model.eval()

    dynamic = basin_data["dynamic"]      # (T, n_dynamic) already normalised
    flow = basin_data["flow"]            # (T,)    raw
    static = basin_data["static"]        # (n_static,) normalised
    precip_mean = basin_data["precip_mean"]
    seq_len = config.seq_len

    valid_idx = np.where(~np.isnan(flow))[0]
    valid_idx = valid_idx[valid_idx >= seq_len - 1]
    if len(valid_idx) == 0:
        return None

    all_pred, all_obs, all_fast, all_slow = [], [], [], []
    # CMAL distribution params (collected per batch for quantile computation)
    cmal_pi, cmal_mu, cmal_bl, cmal_br = [], [], [], []
    use_cmal = config.output_type == "cmal"

    for start in range(0, len(valid_idx), config.batch_size):
        batch_idx = valid_idx[start : start + config.batch_size]

        x_d = make_lookback_batch(dynamic, batch_idx, seq_len).to(device)
        x_s = static.unsqueeze(0).expand(len(batch_idx), -1).to(device)

        q_tot, q_f, q_sl = model(x_d, x_s)

        # denormalise
        scale = float(precip_mean) + 1e-8
        all_pred.append(q_tot.cpu().numpy() * scale)
        all_fast.append(q_f.cpu().numpy() * scale)
        all_slow.append(q_sl.cpu().numpy() * scale)
        all_obs.append(flow[batch_idx])

        # Collect CMAL params (in normalised space — denormalise later)
        if use_cmal:
            params = _unwrap_model(model)._last_cmal_params
            cmal_pi.append(params[0].cpu().numpy())
            cmal_mu.append(params[1].cpu().numpy())
            cmal_bl.append(params[2].cpu().numpy())
            cmal_br.append(params[3].cpu().numpy())

    pred = np.clip(np.concatenate(all_pred), 0, None)
    obs = np.concatenate(all_obs)
    fast = np.concatenate(all_fast)
    slow = np.concatenate(all_slow)

    result = {
        "nse": compute_nse(obs, pred),
        "kge": compute_kge(obs, pred),
        "fhv": compute_fhv(obs, pred),
        "fehv": compute_fehv(obs, pred),
        "flv": compute_flv(obs, pred),
        "n_obs": len(obs),
        "obs": obs,
        "pred": pred,
        "fast": fast,
        "slow": slow,
        "valid_idx": valid_idx,
    }

    # CMAL prediction intervals
    if use_cmal and cmal_pi:
        scale = float(precip_mean) + 1e-8
        pi_all = np.concatenate(cmal_pi)
        mu_all = np.concatenate(cmal_mu)
        bl_all = np.concatenate(cmal_bl)
        br_all = np.concatenate(cmal_br)
        # Compute quantiles in normalised space, then denormalise
        quantiles = cmal_quantiles_np(pi_all, mu_all, bl_all, br_all,
                                      levels=_CMAL_QUANTILES)
        for q, vals in quantiles.items():
            result[f"q{int(q*100):02d}"] = np.clip(vals * scale, 0, None)

        # PICP: proportion of observations within 90% prediction interval
        q05 = result["q05"]
        q95 = result["q95"]
        covered = np.sum((obs >= q05) & (obs <= q95))
        result["picp_90"] = float(covered / len(obs)) if len(obs) > 0 else float("nan")

    # Capture MoE gate weights (per-basin mean π from last batch)
    m = _unwrap_model(model)
    if hasattr(m, "_last_pi"):
        pi = m._last_pi.cpu().numpy()
        for i, v in enumerate(pi):
            result[f"pi_{i}"] = float(v)

    return result


def evaluate_fold(
    model: torch.nn.Module,
    val_dataset: HydroDataset,
    tier_map: Dict[int, int],
    config: Config,
    fold_idx: int,
    device: torch.device,
) -> pd.DataFrame:
    """Evaluate all held-out basins and print a tier-level summary."""

    rows: List[dict] = []
    ts_dir = config.output_dir / f"fold_{fold_idx}" / "timeseries"
    ts_dir.mkdir(parents=True, exist_ok=True)
    use_cmal = config.output_type == "cmal"

    for bid, bdata in val_dataset.basin_data.items():
        res = evaluate_basin(model, bdata, config, device)
        if res is None:
            continue
        row = {
            "basin_id": bid,
            "tier": tier_map[bid],
            "nse": res["nse"],
            "kge": res["kge"],
            "fhv": res["fhv"],
            "fehv": res["fehv"],
            "flv": res["flv"],
            "n_obs": res["n_obs"],
        }
        if use_cmal and "picp_90" in res:
            row["picp_90"] = res["picp_90"]
        # Include per-expert gate weights when available (MoE)
        for k, v in res.items():
            if k.startswith("pi_"):
                row[k] = v
        rows.append(row)

        # Save per-basin timeseries (obs, pred, pathway components, quantiles)
        dates = bdata["dates"]
        ts_data = {
            "date": dates[res["valid_idx"]].strftime("%Y-%m-%d"),
            "obs": res["obs"],
            "pred": res["pred"],
            "q_fast": res["fast"],
            "q_slow": res["slow"],
        }
        if use_cmal:
            for q_level in _CMAL_QUANTILES:
                col = f"q{int(q_level*100):02d}"
                if col in res:
                    ts_data[col] = res[col]
        ts = pd.DataFrame(ts_data)
        ts.to_csv(ts_dir / f"{bid}.csv", index=False)

    df = pd.DataFrame(rows)

    # ---- pretty-print ----
    has_picp = "picp_90" in df.columns

    print(f"\n  Fold {fold_idx + 1} Validation Results")
    hdr = f"  {'Tier':<8}{'N':<6}{'NSE med':>10}{'KGE med':>10}{'FHV med':>10}{'FEHV med':>10}{'FLV med':>10}"
    if has_picp:
        hdr += f"{'PICP90':>10}"
    print(hdr)
    extra_cols = 10 if has_picp else 0
    print(f"  {'-' * (64 + extra_cols)}")
    for tier in sorted(df["tier"].unique()):
        sub = df[df["tier"] == tier]
        line = (
            f"  {tier:<8}{len(sub):<6}"
            f"{sub['nse'].median():>10.3f}{sub['kge'].median():>10.3f}"
            f"{sub['fhv'].median():>10.1f}{sub['fehv'].median():>10.1f}{sub['flv'].median():>10.1f}"
        )
        if has_picp:
            line += f"{sub['picp_90'].median():>10.3f}"
        print(line)
    all_line = (
        f"  {'All':<8}{len(df):<6}"
        f"{df['nse'].median():>10.3f}{df['kge'].median():>10.3f}"
        f"{df['fhv'].median():>10.1f}{df['fehv'].median():>10.1f}{df['flv'].median():>10.1f}"
    )
    if has_picp:
        all_line += f"{df['picp_90'].median():>10.3f}"
    print(all_line)

    # ---- save ----
    out = config.output_dir / f"fold_{fold_idx}"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "basin_results.csv", index=False)

    return df
