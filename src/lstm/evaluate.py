"""Per-basin evaluation and result aggregation."""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
import torch

from .config import Config
from .dataset import HydroDataset
from .loss import compute_kge, compute_nse, compute_fhv, compute_flv


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
    flow_std = basin_data["flow_std"]
    seq_len = config.seq_len

    valid_idx = np.where(~np.isnan(flow))[0]
    valid_idx = valid_idx[valid_idx >= seq_len - 1]
    if len(valid_idx) == 0:
        return None

    all_pred, all_obs, all_fast, all_slow = [], [], [], []

    for start in range(0, len(valid_idx), config.batch_size):
        batch_idx = valid_idx[start : start + config.batch_size]

        x_d = torch.stack(
            [dynamic[i - seq_len + 1 : i + 1] for i in batch_idx]
        ).to(device)
        x_s = static.unsqueeze(0).expand(len(batch_idx), -1).to(device)

        q_tot, q_f, q_sl = model(x_d, x_s)

        # denormalise
        scale = float(flow_std) + 1e-8
        all_pred.append(q_tot.cpu().numpy() * scale)
        all_fast.append(q_f.cpu().numpy() * scale)
        all_slow.append(q_sl.cpu().numpy() * scale)
        all_obs.append(flow[batch_idx])

    pred = np.clip(np.concatenate(all_pred), 0, None)
    obs = np.concatenate(all_obs)
    fast = np.concatenate(all_fast)
    slow = np.concatenate(all_slow)

    return {
        "nse": compute_nse(obs, pred),
        "kge": compute_kge(obs, pred),
        "fhv": compute_fhv(obs, pred),
        "flv": compute_flv(obs, pred),
        "n_obs": len(obs),
        "obs": obs,
        "pred": pred,
        "fast": fast,
        "slow": slow,
    }


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

    for bid, bdata in val_dataset.basin_data.items():
        res = evaluate_basin(model, bdata, config, device)
        if res is None:
            continue
        rows.append(
            {
                "basin_id": bid,
                "tier": tier_map[bid],
                "nse": res["nse"],
                "kge": res["kge"],
                "fhv": res["fhv"],
                "flv": res["flv"],
                "n_obs": res["n_obs"],
            }
        )

        # Save per-basin timeseries (obs, pred, pathway components)
        ts = pd.DataFrame(
            {
                "obs": res["obs"],
                "pred": res["pred"],
                "q_fast": res["fast"],
                "q_slow": res["slow"],
            }
        )
        ts.to_csv(ts_dir / f"{bid}.csv", index=False)

    df = pd.DataFrame(rows)

    # ---- pretty-print ----
    print(f"\n  Fold {fold_idx + 1} Validation Results")
    print(f"  {'Tier':<8}{'N':<6}{'NSE med':>10}{'KGE med':>10}{'FHV med':>10}{'FLV med':>10}")
    print(f"  {'-' * 54}")
    for tier in sorted(df["tier"].unique()):
        sub = df[df["tier"] == tier]
        print(
            f"  {tier:<8}{len(sub):<6}"
            f"{sub['nse'].median():>10.3f}{sub['kge'].median():>10.3f}"
            f"{sub['fhv'].median():>10.1f}{sub['flv'].median():>10.1f}"
        )
    print(
        f"  {'All':<8}{len(df):<6}"
        f"{df['nse'].median():>10.3f}{df['kge'].median():>10.3f}"
        f"{df['fhv'].median():>10.1f}{df['flv'].median():>10.1f}"
    )

    # ---- save ----
    out = config.output_dir / f"fold_{fold_idx}"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "basin_results.csv", index=False)

    return df
