"""Subbasin-mode training, validation, and evaluation.

This module implements the training loop and evaluation for
``spatial_mode='subbasin'``: each gauge sample is a padded stack of
sub-basin inputs (HUC10 or HUC12 polygons); the model is applied to the
flattened stack and the per-subbasin predictions are area-weight-
aggregated to the gauge before the loss is computed.

Aggregation (in physical mm/day units):

    Q_gauge_mmday_hat = sum_i  weight_i * y_hat_i_mmday

The model's raw output is mm/day for each subbasin (the target was not
precip-normalised at the dataset level).  Padding rows have mask=0 and
weight=0 so they contribute nothing.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .config import Config
from .loss import (
    mse_loss, pathway_auxiliary_loss,
    compute_kge, compute_nse, compute_fhv, compute_fehv, compute_flv,
)
from .train import _amp_context, _blended_primary


# ---------------------------------------------------------------------------
# Aggregation helper
# ---------------------------------------------------------------------------


def aggregate_subbasins_to_gauge(
    q_sub_mmday: torch.Tensor,    # (B, N_max) mm/day from the model
    weights: torch.Tensor,        # (B, N_max)  A_i / A_gauge (0 for padding)
    mask: torch.Tensor,           # (B, N_max)  1 real, 0 padding
) -> torch.Tensor:
    """Area-weighted aggregation of subbasin predictions → gauge mm/day.

    Returns a (B,) tensor in mm/day.  The model output is already mm/day
    (no per-subbasin precip de-normalisation).  ``mask`` defends against
    any non-zero values the padded rows produce from the LSTM (e.g.
    Softplus(0) > 0).
    """
    return (q_sub_mmday * mask * weights).sum(dim=1)


def _forward_packed(
    model: torch.nn.Module,
    x_dyn: torch.Tensor,   # (B, N, T, D)
    x_stat: torch.Tensor,  # (B, N, n_stat)
    mask: torch.Tensor,    # (B, N)
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the model on real (non-padding) subbasin rows only, scatter back.

    With ``N_max`` up to ~113 but median ~4, most rows in the flattened
    (B·N, T, D) stack are padding.  Running the LSTM only on real rows
    yields ~10–25× speedups for typical batches with no effect on the
    loss (padded rows contribute nothing via mask·weight = 0).

    Returns three (B, N) tensors matching the shape of the original stack,
    with padded positions zero-filled.
    """
    B, N, T, D = x_dyn.shape
    x_d_flat = x_dyn.reshape(B * N, T, D)
    x_s_flat = x_stat.reshape(B * N, -1)
    mask_flat = mask.reshape(B * N) > 0

    if mask_flat.all():
        q_t, q_f, q_s = model(x_d_flat, x_s_flat)
    else:
        idx = mask_flat.nonzero(as_tuple=False).squeeze(1)
        q_t_r, q_f_r, q_s_r = model(x_d_flat[idx], x_s_flat[idx])
        q_t = x_d_flat.new_zeros(B * N)
        q_f = x_d_flat.new_zeros(B * N)
        q_s = x_d_flat.new_zeros(B * N)
        q_t[idx] = q_t_r
        q_f[idx] = q_f_r
        q_s[idx] = q_s_r
    return q_t.view(B, N), q_f.view(B, N), q_s.view(B, N)


# ---------------------------------------------------------------------------
# Training epoch
# ---------------------------------------------------------------------------


def train_epoch_subbasin(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    config: Config,
) -> tuple[float, float]:
    """Subbasin-mode train epoch.  Returns (primary_mse, total_loss_with_aux)."""
    model.train()
    sum_primary = torch.tensor(0.0, device=device)
    sum_total = torch.tensor(0.0, device=device)
    n = 0
    use_aux = config.aux_loss_weight > 0 and config.model_type == "dual"
    noise_std = config.input_noise_std
    use_extreme_aux = config.extreme_peak_boost > 1.0 and use_aux

    for (x_dyn, x_stat, mask, weights,
         y, y_comp, _bid, basin_w, extreme_qs) in loader:
        # x_dyn  : (B, N_max, seq_len, n_dyn)
        # x_stat : (B, N_max, n_stat)
        # mask/weights : (B, N_max)
        # y      : (B,)  mm/day
        # y_comp : (B, 2) mm/day
        x_dyn = x_dyn.to(device, non_blocking=True)
        x_stat = x_stat.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        weights = weights.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        basin_w = basin_w.to(device, non_blocking=True)
        extreme_qs = extreme_qs.to(device, non_blocking=True)

        if noise_std > 0:
            x_dyn = x_dyn + torch.randn_like(x_dyn) * noise_std

        optimizer.zero_grad(set_to_none=True)
        with _amp_context(device, config):
            # Forward only real subbasin rows (padding skipped)
            q_total, q_fast, q_slow = _forward_packed(model, x_dyn, x_stat, mask)

            # Aggregate to gauge (mm/day)
            q_total_g = aggregate_subbasins_to_gauge(q_total, weights, mask)
            q_fast_g  = aggregate_subbasins_to_gauge(q_fast,  weights, mask)
            q_slow_g  = aggregate_subbasins_to_gauge(q_slow,  weights, mask)

            # Primary loss in mm/day space
            mse_term = mse_loss(q_total_g, y, sample_weights=basin_w)
            primary = _blended_primary(q_total_g, y, basin_w, mse_term, config)

            loss = primary
            if use_aux:
                y_comp = y_comp.to(device, non_blocking=True)
                y_fast_lh = y_comp[:, 0]
                y_slow_lh = y_comp[:, 1]
                q_start_b = extreme_qs[:, 0]
                q_top_b = extreme_qs[:, 1]
                ramp_b = (q_top_b - q_start_b)
                loss = loss + config.aux_loss_weight * pathway_auxiliary_loss(
                    q_fast_g, q_slow_g,
                    y_fast_lh, y_slow_lh,
                    y_total_norm=y if use_extreme_aux else None,
                    extreme_threshold=q_start_b,
                    extreme_peak_boost=config.extreme_peak_boost,
                    extreme_ramp=ramp_b,
                    sample_weights=basin_w,
                )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.grad_clip)
        optimizer.step()
        sum_primary += primary.detach()
        sum_total += loss.detach()
        n += 1

    return sum_primary.item() / max(n, 1), sum_total.item() / max(n, 1)


# ---------------------------------------------------------------------------
# Validation epoch
# ---------------------------------------------------------------------------


@torch.no_grad()
def validate_epoch_subbasin(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    config: Config | None = None,
) -> tuple[float, float, float]:
    """Subbasin-mode validate epoch.  Returns (mse, median_nse, median_kge)
    computed per gauge, in mm/day.
    """
    model.eval()
    total_loss = torch.tensor(0.0, device=device)
    n = 0
    basin_pred: Dict[int, list] = {}
    basin_obs: Dict[int, list] = {}

    for (x_dyn, x_stat, mask, weights,
         y, _ycomp, bids, basin_w, _extreme_qs) in loader:
        x_dyn = x_dyn.to(device, non_blocking=True)
        x_stat = x_stat.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        weights = weights.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        basin_w = basin_w.to(device, non_blocking=True)

        with _amp_context(device, config):
            q_total, _, _ = _forward_packed(model, x_dyn, x_stat, mask)
            q_total_g = aggregate_subbasins_to_gauge(
                q_total, weights, mask,
            )
            total_loss += mse_loss(q_total_g, y, sample_weights=basin_w)
        n += 1

        pred_np = q_total_g.float().cpu().numpy()
        obs_np = y.float().cpu().numpy()
        bid_np = bids.numpy() if isinstance(bids, torch.Tensor) else np.asarray(bids)
        for i, b in enumerate(bid_np):
            b = int(b)
            basin_pred.setdefault(b, []).append(pred_np[i])
            basin_obs.setdefault(b, []).append(obs_np[i])

    mse = total_loss.item() / max(n, 1)
    nses, kges = [], []
    for b in basin_pred:
        p = np.array(basin_pred[b])
        o = np.array(basin_obs[b])
        nse_val = compute_nse(o, p)
        kge_val = compute_kge(o, p)
        if not np.isnan(nse_val):
            nses.append(nse_val)
        if not np.isnan(kge_val):
            kges.append(kge_val)
    median_nse = float(np.median(nses)) if nses else float("nan")
    median_kge = float(np.median(kges)) if kges else float("nan")
    return mse, median_nse, median_kge


# ---------------------------------------------------------------------------
# Fold evaluation
# ---------------------------------------------------------------------------


@torch.no_grad()
def evaluate_fold_subbasin(
    model: torch.nn.Module,
    val_dataset,                # SubbasinHydroDataset
    tier_map: Dict[int, int],
    config: Config,
    fold_idx: int,
    device: torch.device,
) -> pd.DataFrame:
    """Re-run the validation dataset batch by batch, aggregate predictions
    per gauge, and compute per-gauge metrics (NSE/KGE/FHV/FEHV/FLV) in
    mm/day.  Write per-gauge timeseries CSVs and a basin_results.csv.
    """
    model.eval()

    ts_dir = config.output_dir / f"fold_{fold_idx}" / "timeseries"
    ts_dir.mkdir(parents=True, exist_ok=True)

    from .subbasin_dataset import subbasin_collate
    loader = DataLoader(
        val_dataset, batch_size=config.batch_size,
        shuffle=False, num_workers=0,
        collate_fn=subbasin_collate,
    )

    # Accumulate per-gauge predictions + observations + pathway components
    pred_by_gauge: Dict[int, list] = {}
    obs_by_gauge: Dict[int, list] = {}
    fast_by_gauge: Dict[int, list] = {}
    slow_by_gauge: Dict[int, list] = {}
    tidx_by_gauge: Dict[int, list] = {}

    # Sample order follows val_dataset.samples so we can recover tidx.
    sample_iter = iter(val_dataset.samples)

    for (x_dyn, x_stat, mask, weights,
         y, _yc, bids, _lw, _qs) in loader:
        x_dyn = x_dyn.to(device, non_blocking=True)
        x_stat = x_stat.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        weights = weights.to(device, non_blocking=True)

        q_total, q_fast, q_slow = _forward_packed(model, x_dyn, x_stat, mask)

        q_tot_g = aggregate_subbasins_to_gauge(q_total, weights, mask)
        q_fast_g = aggregate_subbasins_to_gauge(q_fast, weights, mask)
        q_slow_g = aggregate_subbasins_to_gauge(q_slow, weights, mask)

        q_tot_np = q_tot_g.float().cpu().numpy()
        q_fast_np = q_fast_g.float().cpu().numpy()
        q_slow_np = q_slow_g.float().cpu().numpy()
        obs_np = y.float().cpu().numpy()
        bid_np = bids.numpy() if isinstance(bids, torch.Tensor) else np.asarray(bids)

        for i, b in enumerate(bid_np):
            b = int(b)
            sample = next(sample_iter)
            _, tidx = sample
            pred_by_gauge.setdefault(b, []).append(float(q_tot_np[i]))
            obs_by_gauge.setdefault(b, []).append(float(obs_np[i]))
            fast_by_gauge.setdefault(b, []).append(float(q_fast_np[i]))
            slow_by_gauge.setdefault(b, []).append(float(q_slow_np[i]))
            tidx_by_gauge.setdefault(b, []).append(int(tidx))

    rows: List[dict] = []
    for b, obs_list in obs_by_gauge.items():
        obs = np.array(obs_list)
        pred = np.clip(np.array(pred_by_gauge[b]), 0, None)
        fast = np.clip(np.array(fast_by_gauge[b]), 0, None)
        slow = np.clip(np.array(slow_by_gauge[b]), 0, None)
        tidx = np.array(tidx_by_gauge[b])

        row = {
            "basin_id": b,
            "tier": tier_map[b],
            "nse": compute_nse(obs, pred),
            "kge": compute_kge(obs, pred),
            "fhv": compute_fhv(obs, pred),
            "fehv": compute_fehv(obs, pred),
            "flv": compute_flv(obs, pred),
            "n_obs": len(obs),
        }
        rows.append(row)

        dates = val_dataset.gauge_data[b]["dates"]
        ts = pd.DataFrame({
            "date": dates[tidx].strftime("%Y-%m-%d"),
            "obs": obs,
            "pred": pred,
            "q_fast": fast,
            "q_slow": slow,
        })
        ts.to_csv(ts_dir / f"{b}.csv", index=False)

    df = pd.DataFrame(rows)

    # Pretty-print per-tier summary
    print(f"\n  Fold {fold_idx + 1} Validation Results  [subbasin]")
    hdr = (
        f"  {'Tier':<8}{'N':<6}{'NSE med':>10}{'KGE med':>10}"
        f"{'FHV med':>10}{'FEHV med':>10}{'FLV med':>10}"
    )
    print(hdr)
    print(f"  {'-' * 64}")
    if not df.empty:
        for tier in sorted(df["tier"].unique()):
            sub = df[df["tier"] == tier]
            print(
                f"  {tier:<8}{len(sub):<6}"
                f"{sub['nse'].median():>10.3f}{sub['kge'].median():>10.3f}"
                f"{sub['fhv'].median():>10.1f}{sub['fehv'].median():>10.1f}"
                f"{sub['flv'].median():>10.1f}"
            )
        print(
            f"  {'All':<8}{len(df):<6}"
            f"{df['nse'].median():>10.3f}{df['kge'].median():>10.3f}"
            f"{df['fhv'].median():>10.1f}{df['fehv'].median():>10.1f}"
            f"{df['flv'].median():>10.1f}"
        )

    out = config.output_dir / f"fold_{fold_idx}"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "basin_results.csv", index=False)
    return df
