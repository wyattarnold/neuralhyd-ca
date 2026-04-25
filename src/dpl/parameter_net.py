"""Parameter-prediction network (``gA``) for the dPL+SAC-SMA model.

Maps per-HUC12 static attributes (and optionally a dynamic forcing
window) to physical parameters of Snow-17, SAC-SMA, Hamon PET, and the
Lohmann UH.  The architecture mirrors Shen-group ``MultiInv_HBVTDModel``:

* A static MLP encoder digests HUC12 attributes.
* An LSTM ingests the dynamic forcing window and concatenates its
  hidden state with the static embedding.
* Two heads emit
    - **static parameters** (one value per unit, time-invariant)
    - **dynamic parameters** (one value per unit per timestep)
  Both pass through a sigmoid and are mapped to physical ranges
  ``[lo, hi]`` from :mod:`src.dpl.config`.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .config import (
    DplConfig,
    LOHMANN_PARAMS,
    PET_PARAMS,
    SACSMA_PARAMS,
    SNOW17_PARAMS,
    all_param_names,
    param_bounds,
)


def _physical_param_tables() -> tuple[list[str], list[float], list[float]]:
    names = all_param_names()
    lows, highs = param_bounds()
    return names, lows, highs


class StaticEncoder(nn.Module):
    """Two-layer MLP from raw static attributes to a fixed embedding."""

    def __init__(self, n_features: int, embedding_dim: int,
                 hidden_size: int = 64, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, embedding_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (N, n_static) -> (N, embed)
        return self.net(x)


class ParameterNet(nn.Module):
    """gA: dynamic LSTM + static MLP → SAC-SMA / Snow-17 / Hamon / UH params."""

    def __init__(self, config: DplConfig, n_dynamic: int, n_static: int):
        super().__init__()
        self.config = config
        names, lows, highs = _physical_param_tables()
        self._names = names
        self._n_params = len(names)
        self._dynamic_set = set(config.dynamic_params)

        self.register_buffer(
            "_lo", torch.tensor(lows, dtype=torch.float32), persistent=False
        )
        self.register_buffer(
            "_hi", torch.tensor(highs, dtype=torch.float32), persistent=False
        )
        # Indices of dynamic vs static params (in canonical order).
        dyn_idx = [i for i, n in enumerate(names) if n in self._dynamic_set]
        sta_idx = [i for i, n in enumerate(names) if n not in self._dynamic_set]
        self.register_buffer("_dyn_idx", torch.tensor(dyn_idx, dtype=torch.long), persistent=False)
        self.register_buffer("_sta_idx", torch.tensor(sta_idx, dtype=torch.long), persistent=False)

        # Static encoder
        self.static_encoder = StaticEncoder(
            n_features=n_static,
            embedding_dim=config.static_embedding_dim,
            hidden_size=config.static_hidden_size,
            dropout=config.static_dropout,
        )

        # Dynamic LSTM
        self.lstm = nn.LSTM(
            input_size=n_dynamic,
            hidden_size=config.lstm_hidden_size,
            num_layers=config.lstm_num_layers,
            batch_first=True,
            dropout=config.dropout if config.lstm_num_layers > 1 else 0.0,
        )

        # Static-parameter head: from final LSTM hidden + static embed -> static params
        head_in = config.lstm_hidden_size + config.static_embedding_dim
        self.static_head = nn.Sequential(
            nn.Linear(head_in, config.lstm_hidden_size),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.lstm_hidden_size, len(sta_idx)),
        )
        # Dynamic-parameter head: per-timestep LSTM output + static embed -> dynamic params
        self.dynamic_head = nn.Sequential(
            nn.Linear(head_in, config.lstm_hidden_size),
            nn.ReLU(),
            nn.Linear(config.lstm_hidden_size, len(dyn_idx)),
        ) if dyn_idx else None

    def forward(
        self, x_dyn: torch.Tensor, x_static: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Return a dict of physical parameters keyed by name.

        Parameters
        ----------
        x_dyn : (N, T, n_dynamic) tensor — forcing window into gA.
        x_static : (N, n_static) tensor — HUC12 static attributes.

        Returns
        -------
        dict[name -> (N,) or (N, T) tensor].  Each parameter is mapped
        into its physical range ``[lo, hi]``.
        """
        N, T, _ = x_dyn.shape
        e_static = self.static_encoder(x_static)                 # (N, embed)
        h_seq, (h_n, _) = self.lstm(x_dyn)                       # (N, T, H), (L, N, H)
        h_last = h_n[-1]                                         # (N, H)

        # Static head
        s_in = torch.cat([h_last, e_static], dim=-1)             # (N, H+embed)
        s_logits = self.static_head(s_in)                        # (N, n_static_params)
        s_norm = torch.sigmoid(s_logits)                         # (N, n_static_params)

        # Dynamic head — only build the larger (N, T, ·) tensor when needed
        if self.dynamic_head is not None:
            e_static_rep = e_static.unsqueeze(1).expand(-1, T, -1)   # (N, T, embed)
            d_in = torch.cat([h_seq, e_static_rep], dim=-1)          # (N, T, H+embed)
            d_logits = self.dynamic_head(d_in)                       # (N, T, n_dyn_params)
            d_norm = torch.sigmoid(d_logits)                         # (N, T, n_dyn_params)
        else:
            d_norm = None

        # Emit a dict of per-parameter tensors directly — avoids allocating
        # an (N, T, P_total) scratch buffer only to re-slice column by column.
        lo = self._lo
        hi = self._hi
        out: dict[str, torch.Tensor] = {}
        if s_norm.numel() > 0:
            sta_idx = self._sta_idx.tolist()
            for j, i in enumerate(sta_idx):
                out[self._names[i]] = lo[i] + (hi[i] - lo[i]) * s_norm[:, j]
        if d_norm is not None and d_norm.numel() > 0:
            dyn_idx = self._dyn_idx.tolist()
            for j, i in enumerate(dyn_idx):
                out[self._names[i]] = lo[i] + (hi[i] - lo[i]) * d_norm[:, :, j]
        return out
