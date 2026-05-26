"""LSTM models with static watershed conditioning.

All architectures share a common forward signature returning
``(q_total, q_fast, q_slow)``:

* **DualPathwayLSTM** -- two LSTM branches (fast event-scale / slow
  baseflow) with fixed multiplicative composition.

* **SingleLSTM** -- one LSTM processing the full 365-day lookback.
  Pathway outputs are zero-filled to keep the same 3-tuple interface.

* **MoELSTM** -- K independent LSTM experts with a gate-LSTM (final
    hidden state) and learnable temperature (MoE-tau).  Pathway outputs
    are zero-filled.

Use ``build_model(config)`` to instantiate the model selected by
``config.model_type`` ("dual", "single", "moe").
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .config import Config
from .loss import extract_mixture_quantile


class StaticEncoder(nn.Module):
    """MLP: n_static â†' hidden â†' embedding_dim."""

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
        self.output_dim = embedding_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)



class GroupedStaticEncoder(nn.Module):
    """Encode semantic static groups into a concatenated embedding.

    Each group gets a small two-layer encoder. The output dimension is
    auto-sized to ``max(ceil(2 * sqrt(n)), 4)`` for each group, giving enough
    capacity for categorical-heavy groups without letting statics dominate the
    recurrent input. All group embeddings are concatenated to form the output.

    One shared instance is built per model; all branches receive the same
    embedding directly (Option 1 fully-shared design).

    Parameters
    ----------
    group_sizes : list[int]
        Number of input features per group, **in the order they appear
        in the flat static feature vector**.
    group_names : list[str]
        Semantic group names in the same order as ``group_sizes``.
    dropout : float
        Dropout applied inside each group encoder and to the concatenated output.
    group_dropout : float
        Probability of dropping each semantic group embedding during training.
    """

    def __init__(
        self,
        group_names: list[str],
        group_sizes: list[int],
        dropout: float = 0.2,
        group_dropout: float = 0.0,
    ):
        super().__init__()
        self.group_names = group_names
        self._splits = group_sizes
        self.group_dropout = float(group_dropout)

        encoders: list[nn.Module] = []
        out_dim = 0
        for n_feat in group_sizes:
            # Auto-size: ceil(2 * sqrt(n)) with floor of 4.
            g_out = max(math.ceil(2.0 * math.sqrt(n_feat)), 4)
            g_hidden = max(2 * g_out, min(n_feat, 64))
            encoders.append(nn.Sequential(
                nn.Linear(n_feat, g_hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(g_hidden, g_out),
                nn.GELU(),
            ))
            out_dim += g_out
        self.group_encoders = nn.ModuleList(encoders)
        self.output_dim = out_dim
        self._dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        parts = x.split(self._splits, dim=-1)
        encoded = [enc(p) for enc, p in zip(self.group_encoders, parts)]
        if self.training and self.group_dropout > 0.0:
            keep_prob = 1.0 - self.group_dropout
            encoded = [
                e * e.new_empty(e.shape[0], 1).bernoulli_(keep_prob) / keep_prob
                for e in encoded
            ]
        return self._dropout(torch.cat(encoded, dim=-1))


def _build_static_encoder(config: Config) -> StaticEncoder | GroupedStaticEncoder:
    """Instantiate the correct static encoder based on config."""
    group_sizes = config.static_group_sizes
    if group_sizes is not None:
        group_names = config.static_group_names
        if group_names is None:
            raise ValueError("static_group_sizes were provided without static_group_names")
        return GroupedStaticEncoder(
            group_names=group_names,
            group_sizes=group_sizes,
            dropout=config.static_dropout,
            group_dropout=config.static_group_dropout,
        )
    n_static = len(config.encoded_static_feature_names)
    return StaticEncoder(
        n_static, config.static_embedding_dim,
        hidden_size=config.static_hidden_size,
        dropout=config.static_dropout,
    )


class CMALHead(nn.Module):
    """Countable Mixture of Asymmetric Laplacians output head.

    Produces K mixture components, each parameterised by a weight pi_k,
    location mu_k, left scale b_L,k, and right scale b_R,k.  The
    asymmetric Laplace naturally handles the skewed, heavy-tailed
    nature of streamflow distributions.

    Parameters
    ----------
    input_size : int
        Dimension of the incoming hidden state.
    n_components : int
        K -- number of mixture components (default 3).
    hidden_size : int
        Intermediate dense layer width.
    """

    def __init__(self, input_size: int, n_components: int = 3,
                 hidden_size: int = 32):
        super().__init__()
        self.n_components = n_components
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 4 * n_components),
        )

    def forward(
        self, h: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (pi, mu, b_l, b_r), each (B, K)."""
        out = self.net(h)
        K = self.n_components
        pi = torch.softmax(out[:, :K], dim=-1)
        mu = nn.functional.softplus(out[:, K : 2 * K])
        b_l = nn.functional.softplus(out[:, 2 * K : 3 * K]) + 1e-4
        b_r = nn.functional.softplus(out[:, 3 * K :]) + 1e-4
        return pi, mu, b_l, b_r


class DualPathwayLSTM(nn.Module):
    """Two-branch LSTM for rainfall-runoff simulation (fast + slow).

    In deterministic mode, uses multiplicative composition:
    ``q_total = q_slow * (1 + fast_ratio)``.  The slow pathway sets the
    baseflow level; the fast pathway is a dimensionless storm amplifier,
    so storm contribution scales with antecedent wetness.

    In CMAL mode, a ``CMALHead`` on concatenated ``[h_slow, h_fast]``
    parameterises the predictive distribution.  The pathway heads are
    retained so that the auxiliary loss can still supervise ``q_slow``
    and ``q_fast`` against Lyne-Hollick targets, keeping the pathway
    representations physically grounded.
    """

    def __init__(self, config: Config):
        super().__init__()
        n_dynamic = len(config.dynamic_features)
        self.fast_window = config.fast_window
        self.info_gap = config.info_gap
        self._use_cmal = config.output_type == "cmal"
        self._adaptive_quantile = self._use_cmal and config.cmal_adaptive_quantile

        self._grouped = config.static_group_sizes is not None
        if self._grouped:
            self.shared_static_enc = _build_static_encoder(config)
            static_out_dim = self.shared_static_enc.output_dim
        else:
            self.static_encoder = _build_static_encoder(config)
            static_out_dim = self.static_encoder.output_dim

        fast_input_size = n_dynamic + static_out_dim
        slow_input_size = n_dynamic + static_out_dim

        # Fast LSTM (event-scale window)
        self.fast_lstm = nn.LSTM(
            input_size=fast_input_size,
            hidden_size=config.fast_hidden_size,
            batch_first=True,
        )

        # Slow LSTM (full window -- baseflow / seasonal)
        self.slow_lstm = nn.LSTM(
            input_size=slow_input_size,
            hidden_size=config.slow_hidden_size,
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
        # Fast head: dimensionless storm amplifier (>= 0)
        # Multiplied onto q_slow -> storm response scales with baseflow
        self.fast_head = nn.Sequential(
            nn.Linear(config.fast_hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Softplus(),
        )

        # CMAL probabilistic head (uses both pathway hidden states)
        if self._use_cmal:
            combined_size = config.slow_hidden_size + config.fast_hidden_size
            self.cmal_head = CMALHead(
                combined_size,
                n_components=config.cmal_n_components,
                hidden_size=config.cmal_hidden_size,
            )
            # Adaptive quantile selector: maps detached hidden states -> alpha in (0,1).
            # Separate from CMAL params so the pinball gradient doesn't feed back
            # through (b_l, b_r) and cause alpha -> 1 collapse.
            if self._adaptive_quantile:
                self.alpha_selector = nn.Linear(combined_size, 1)

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
            In CMAL mode, q_total is E[Y] from the mixture distribution.
            q_fast and q_slow are always the deterministic pathway outputs
            (used by the auxiliary loss).
        """
        B, T, _ = x_dynamic.shape

        # Encode static attributes: one shared embedding for all branches
        if self._grouped:
            e = self.shared_static_enc(x_static)
            slow_e = event_e = e
        else:
            e = self.static_encoder(x_static)
            slow_e = event_e = e
        slow_full = slow_e.unsqueeze(1).expand(-1, T, -1)
        event_full = event_e.unsqueeze(1).expand(-1, T, -1)
        x_slow_full = torch.cat([x_dynamic, slow_full], dim=-1)
        x_event_full = torch.cat([x_dynamic, event_full], dim=-1)

        # ----- fast pathway (last fast_window days) -----
        _, (h_fast, _) = self.fast_lstm(x_event_full[:, -self.fast_window :, :])
        h_fast = self.dropout(h_fast.squeeze(0))              # (B, H_fast)
        fast_ratio = self.fast_head(h_fast)                   # (B, 1) dimensionless >= 0

        # ----- slow pathway -----
        if self.info_gap:
            slow_input = x_slow_full[:, :-self.fast_window, :]
        else:
            slow_input = x_slow_full
        _, (h_slow, _) = self.slow_lstm(slow_input)
        h_slow = self.dropout(h_slow.squeeze(0))              # (B, H_slow)
        q_slow = self.slow_head(h_slow)                       # (B, 1) mm/d baseflow

        # ----- pathway outputs for auxiliary loss -----
        # Multiplicative composition: q_total = q_slow * (1 + fast_ratio)
        # q_fast_contrib is the storm amplification above baseflow.
        q_fast_contrib = q_slow * fast_ratio                  # (B, 1) mm/d storm runoff

        if self._use_cmal:
            h_combined = torch.cat([h_slow, h_fast], dim=-1)
            pi, mu, b_l, b_r = self.cmal_head(h_combined)
            self._last_cmal_params = (pi, mu, b_l, b_r)
            if self._adaptive_quantile:
                # Alpha from a separate selector on detached hiddens.
                # q_total uses stopped distribution tensors so the pinball gradient
                # flows ONLY to alpha_selector weights -- CRPS fully owns (pi,mu,b_l,b_r).
                alpha = torch.sigmoid(
                    self.alpha_selector(h_combined.detach()).squeeze(-1)
                )                                                   # (B,) in (0, 1)
                self._last_alpha     = alpha.detach()               # for diagnostics
                self._last_alpha_raw = alpha                        # for pinball loss
                q_total = extract_mixture_quantile(
                    alpha, pi.detach(), mu.detach(), b_l.detach(), b_r.detach(),
                )
            else:
                q_total = (pi * (mu + b_r - b_l)).sum(dim=-1)
            q_total = q_total.clamp(min=0.0)
            q_fast_out = q_fast_contrib.squeeze(-1)
            q_slow_out = q_slow.squeeze(-1)
            return q_total, q_fast_out, q_slow_out

        # ----- deterministic multiplicative composition -----
        q_total = q_slow * (1.0 + fast_ratio)                 # (B, 1)

        q_total = q_total.squeeze(-1)
        q_fast_out = q_fast_contrib.squeeze(-1)
        q_slow_out = q_slow.squeeze(-1)

        return q_total, q_fast_out, q_slow_out


class SingleLSTM(nn.Module):
    """Single-branch LSTM baseline -- full 365-day lookback."""

    def __init__(self, config: Config):
        super().__init__()
        n_dynamic = len(config.dynamic_features)
        self._use_cmal = config.output_type == "cmal"

        self._grouped = config.static_group_sizes is not None
        if self._grouped:
            self.shared_static_enc = _build_static_encoder(config)
            static_out_dim = self.shared_static_enc.output_dim
        else:
            self.static_encoder = _build_static_encoder(config)
            static_out_dim = self.static_encoder.output_dim

        input_size = n_dynamic + static_out_dim

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=config.single_hidden_size,
            batch_first=True,
        )

        self.dropout = nn.Dropout(config.dropout)

        if self._use_cmal:
            self.head = CMALHead(
                config.single_hidden_size,
                n_components=config.cmal_n_components,
                hidden_size=config.cmal_hidden_size,
            )
        else:
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

        if self._grouped:
            e_s = self.shared_static_enc(x_static)
        else:
            e_s = self.static_encoder(x_static)
        e_full = e_s.unsqueeze(1).expand(-1, T, -1)
        x_full = torch.cat([x_dynamic, e_full], dim=-1)

        _, (h, _) = self.lstm(x_full)
        h = self.dropout(h.squeeze(0))

        if self._use_cmal:
            pi, mu, b_l, b_r = self.head(h)
            self._last_cmal_params = (pi, mu, b_l, b_r)
            q_total = (pi * (mu + b_r - b_l)).sum(dim=-1).clamp(min=0.0)
        else:
            q_total = self.head(h).squeeze(-1)                    # (B,)

        zeros = torch.zeros_like(q_total)
        return q_total, zeros, zeros


class MoELSTM(nn.Module):
    """Mixture-of-Experts LSTM with learnable temperature (MoE-tau).

    K independent LSTM experts process the input sequence.  A gating
    LSTM computes sequence-level expert weights from its final hidden
    state via temperature-scaled softmax. The mixture of experts'
    final hidden states is linearly projected to discharge.

    Returns ``(q_total, zeros, zeros)`` to match the 3-tuple interface.
    """

    def __init__(self, config: Config):
        super().__init__()
        n_dynamic = len(config.dynamic_features)
        self.n_experts = config.moe_n_experts

        self._grouped = config.static_group_sizes is not None
        if self._grouped:
            self.shared_static_enc = _build_static_encoder(config)
            static_out_dim = self.shared_static_enc.output_dim
        else:
            self.static_encoder = _build_static_encoder(config)
            static_out_dim = self.static_encoder.output_dim

        input_size = n_dynamic + static_out_dim
        D_h = config.moe_expert_hidden_size
        D_g = config.moe_gate_hidden_size
        K = config.moe_n_experts
        self.tau_min = config.moe_tau_min

        # --- Expert LSTMs ---
        self.experts = nn.ModuleList([
            nn.LSTM(input_size=input_size, hidden_size=D_h, batch_first=True)
            for _ in range(K)
        ])

        # --- Gating network: LSTM (final hidden state -> expert logits) ---
        self.gate_lstm = nn.LSTM(
            input_size=input_size, hidden_size=D_g,
            batch_first=True,
        )
        # Expert logits: z = W_p c
        self.gate_proj = nn.Linear(D_g, K, bias=False)

        # Learnable log-temperature (initialised so tau = moe_tau_init)
        # tau = tau_min + (1 - tau_min) * sigmoid(_log_tau) keeps tau_min <= tau < 1
        # and ensures gradient always flows (no dead-zone from clamping below tau_min).
        _tau_frac_init = (config.moe_tau_init - config.moe_tau_min) / (1.0 - config.moe_tau_min)
        self._log_tau = nn.Parameter(
            torch.tensor(_inv_sigmoid(float(max(1e-6, min(1 - 1e-6, _tau_frac_init)))))
        )

        self.dropout = nn.Dropout(config.dropout)

        # Number of gate outputs (for diagnostic logging)
        self.n_gate_outputs = K

        # Linear probe: y_hat = w^T m + b, with Softplus for non-negative flow
        self.head = nn.Sequential(
            nn.Linear(D_h, 1),
            nn.Softplus(),
        )

    @property
    def tau(self) -> torch.Tensor:
        """Learnable temperature tau_min <= tau < 1 (bounded sigmoid; gradient always flows)."""
        return self.tau_min + (1.0 - self.tau_min) * torch.sigmoid(self._log_tau)

    def forward(
        self,
        x_dynamic: torch.Tensor,
        x_static: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, T, _ = x_dynamic.shape

        if self._grouped:
            e_s = self.shared_static_enc(x_static)
        else:
            e_s = self.static_encoder(x_static)
        e_full = e_s.unsqueeze(1).expand(-1, T, -1)            # (B, T, E)
        x = torch.cat([x_dynamic, e_full], dim=-1)            # (B, T, D+E)

        # --- Expert forward passes: collect final hidden states ---
        # h_experts: (B, K, D_h)
        h_list = []
        for expert in self.experts:
            _, (h_k, _) = expert(x)                            # h_k: (1, B, D_h)
            h_list.append(h_k.squeeze(0))                      # (B, D_h)
        h_experts = torch.stack(h_list, dim=1)                 # (B, K, D_h)

        # --- Gating network: final hidden state of the gate LSTM ---
        _, (h_g, _) = self.gate_lstm(x)                        # h_g: (1, B, D_g)
        c = h_g.squeeze(0)                                     # (B, D_g)

        # Expert logits and temperature-scaled softmax
        z = self.gate_proj(c)                                  # (B, K)
        tau = self.tau                                         # bounded in [tau_min, 1) by construction
        pi = torch.softmax(z / tau, dim=-1)                    # (B, K)
        self._last_tau_eff = tau.detach()

        mean_pi = pi.mean(dim=0)                               # (K,)
        self._last_pi = mean_pi.detach()                       # (K,) detached -- for diagnostics
        self._last_pi_raw = mean_pi                            # (K,) with grad -- for balance loss

        # Per-expert predictions for diagnostics (B, K)
        q_per_expert = self.head(h_experts).squeeze(-1)         # (B, K)
        self._last_q_experts = q_per_expert.detach()

        # Mixture of expert hidden states
        m = (pi.unsqueeze(-1) * h_experts).sum(dim=1)          # (B, D_h)
        m = self.dropout(m)

        q_total = self.head(m).squeeze(-1)                     # (B,)
        zeros = torch.zeros_like(q_total)
        return q_total, zeros, zeros


def _inv_sigmoid(x: float) -> float:
    """Inverse sigmoid: returns y s.t. sigmoid(y) = x."""
    return math.log(x / (1.0 - x))


def build_model(config: Config) -> nn.Module:
    """Instantiate the model selected by ``config.model_type``."""
    if config.output_type == "cmal" and config.model_type not in ("single", "dual"):
        raise ValueError(
            f"CMAL output is only supported for model_type='single' or 'dual', "
            f"got {config.model_type!r}"
        )
    if config.model_type == "dual":
        return DualPathwayLSTM(config)
    if config.model_type == "single":
        return SingleLSTM(config)
    if config.model_type == "moe":
        return MoELSTM(config)
    raise ValueError(f"Unknown model_type: {config.model_type!r}")
