"""LSTM models with static watershed conditioning.

Two architectures share a common forward signature returning
``(q_total, q_fast, q_slow)``:

* **DualPathwayLSTM** – two LSTM branches (fast event-scale / slow
  baseflow) producing a physically interpretable flow decomposition.

* **SingleLSTM** – one LSTM processing the full 365-day lookback.
  Serves as a simpler baseline.  Pathway outputs are zero-filled to
  keep the same 3-tuple interface.

Use ``build_model(config)`` to instantiate the model selected by
``config.model_type`` ("dual" or "single").
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .config import Config


class StaticEncoder(nn.Module):
    """MLP: n_static → hidden → embedding_dim."""

    def __init__(self, n_features: int, embedding_dim: int,
                 hidden_size: int = 32, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, embedding_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DualPathwayLSTM(nn.Module):
    """Two-branch LSTM for rainfall–runoff simulation (fast + slow)."""

    def __init__(self, config: Config):
        super().__init__()
        n_dynamic = len(config.dynamic_features)
        n_static = len(config.effective_static_features)
        self.fast_window = config.fast_window
        self.info_gap = config.info_gap

        # Static encoder
        self.static_encoder = StaticEncoder(
            n_static, config.static_embedding_dim,
            hidden_size=config.static_hidden_size,
            dropout=config.static_dropout,
        )

        input_size = n_dynamic + config.static_embedding_dim

        # Fast LSTM (event-scale window)
        self.fast_lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=config.fast_hidden_size,
            num_layers=1,
            batch_first=True,
        )

        # Slow LSTM (full window — baseflow / seasonal)
        self.slow_lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=config.slow_hidden_size,
            num_layers=1,
            batch_first=True,
        )

        self.dropout = nn.Dropout(config.dropout)

        # Slow head: baseflow in physical units (mm/d), strictly positive
        self.slow_head = nn.Sequential(
            nn.Linear(config.slow_hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Softplus(),
        )
        # Fast head: dimensionless storm amplifier (≥ 0)
        # Multiplied onto q_slow → storm response scales with baseflow
        self.fast_head = nn.Sequential(
            nn.Linear(config.fast_hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Softplus(),
        )

    # ------------------------------------------------------------------

    def forward(
        self,
        x_dynamic: torch.Tensor,
        x_static: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x_dynamic : (B, seq_len, n_dynamic)
        x_static  : (B, n_static)

        Returns
        -------
        q_total, q_fast, q_slow : each (B,)
        """
        B, T, _ = x_dynamic.shape

        # Encode static attributes and tile once across full sequence
        e_s = self.static_encoder(x_static)                   # (B, E)
        e_full = e_s.unsqueeze(1).expand(-1, T, -1)           # (B, T, E)
        x_full = torch.cat([x_dynamic, e_full], dim=-1)       # (B, T, D+E)

        # ----- fast pathway (last fast_window days) -----
        _, (h_fast, _) = self.fast_lstm(x_full[:, -self.fast_window :, :])
        h_fast = self.dropout(h_fast.squeeze(0))              # (B, H_fast)
        fast_ratio = self.fast_head(h_fast)                   # (B, 1) dimensionless ≥ 0

        # ----- slow pathway -----
        if self.info_gap:
            # Blind slow LSTM to the last fast_window days so it cannot
            # learn storm responses — only antecedent / baseflow signal.
            slow_input = x_full[:, :-self.fast_window, :]
        else:
            # Full sequence — separation driven by multiplicative
            # composition and head activations alone.
            slow_input = x_full
        _, (h_slow, _) = self.slow_lstm(slow_input)
        h_slow = self.dropout(h_slow.squeeze(0))              # (B, H_slow)
        q_slow = self.slow_head(h_slow)                       # (B, 1) mm/d baseflow

        # ----- multiplicative composition -----
        # q_total = q_slow × (1 + fast_ratio)
        # Slow sets the baseflow level; fast is a dimensionless storm
        # amplifier so storm contribution scales with antecedent wetness.
        q_fast_contrib = q_slow * fast_ratio                  # (B, 1) mm/d storm runoff
        q_total = q_slow + q_fast_contrib                     # (B, 1) = q_slow × (1 + r)
        return (
            q_total.squeeze(-1),
            q_fast_contrib.squeeze(-1),
            q_slow.squeeze(-1),
        )


class SingleLSTM(nn.Module):
    """Single-branch LSTM baseline — full 365-day lookback."""

    def __init__(self, config: Config):
        super().__init__()
        n_dynamic = len(config.dynamic_features)
        n_static = len(config.effective_static_features)

        self.static_encoder = StaticEncoder(
            n_static, config.static_embedding_dim,
            hidden_size=config.static_hidden_size,
            dropout=config.static_dropout,
        )

        input_size = n_dynamic + config.static_embedding_dim

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=config.single_hidden_size,
            num_layers=1,
            batch_first=True,
        )

        self.dropout = nn.Dropout(config.dropout)

        self.head = nn.Sequential(
            nn.Linear(config.single_hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Softplus(),
        )

    def forward(
        self,
        x_dynamic: torch.Tensor,
        x_static: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, T, _ = x_dynamic.shape

        e_s = self.static_encoder(x_static)
        e_full = e_s.unsqueeze(1).expand(-1, T, -1)
        x_full = torch.cat([x_dynamic, e_full], dim=-1)

        _, (h, _) = self.lstm(x_full)
        h = self.dropout(h.squeeze(0))
        q_total = self.head(h).squeeze(-1)                    # (B,)

        zeros = torch.zeros_like(q_total)
        return q_total, zeros, zeros


def build_model(config: Config) -> nn.Module:
    """Instantiate the model selected by ``config.model_type``."""
    if config.model_type == "dual":
        return DualPathwayLSTM(config)
    if config.model_type == "single":
        return SingleLSTM(config)
    raise ValueError(f"Unknown model_type: {config.model_type!r}")
