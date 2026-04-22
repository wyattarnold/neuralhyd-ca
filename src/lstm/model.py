"""LSTM models with static watershed conditioning.

All architectures share a common forward signature returning
``(q_total, q_fast, q_slow)``:

* **DualPathwayLSTM** – two LSTM branches (fast event-scale / slow
  baseflow) with fixed multiplicative composition.

* **SingleLSTM** – one LSTM processing the full 365-day lookback.
  Pathway outputs are zero-filled to keep the same 3-tuple interface.

* **MoELSTM** – K independent LSTM experts with LSTM-Attention gating
  and learnable temperature (MoE-τ).  Pathway outputs are zero-filled.

Use ``build_model(config)`` to instantiate the model selected by
``config.model_type`` ("dual", "single", or "moe").
"""

from __future__ import annotations

import math

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


class ScaleHead(nn.Module):
    """Per-basin multiplicative scale head.

    Produces ``log(s_b)`` from the static embedding; the caller multiplies
    pathway outputs by ``exp(log s_b)``.  The final layer is zero-initialised
    so the scale starts at 1.0 for every basin, making training begin
    identically to the ``precip_mean`` baseline.  The scale then drifts
    end-to-end under the main loss, absorbing per-basin amplitude so the
    LSTMs can produce outputs in a compact range.
    """

    def __init__(self, in_dim: int, hidden: int = 32, max_log_scale: float = 4.0):
        super().__init__()
        self.max_log_scale = max_log_scale
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        # Zero-init final layer → log s = 0 → s = 1 at init
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, e_static: torch.Tensor) -> torch.Tensor:
        log_s = self.net(e_static).squeeze(-1)
        # Clamp to prevent runaway scale during early training
        return log_s.clamp(-self.max_log_scale, self.max_log_scale)


class GroupedStaticEncoder(nn.Module):
    """Encode semantic groups of static features, then fuse.

    Each group gets a small linear encoder.  Group embeddings are
    concatenated and projected to the final ``embedding_dim``.

    Parameters
    ----------
    group_sizes : list[int]
        Number of input features per group, **in the order they appear
        in the flat static feature vector**.
    embedding_dim : int
        Final output dimension (same role as ``StaticEncoder``).
    group_hidden : int
        Hidden size per group encoder.  If 0, each group's output dim
        is ``max(n_features_in_group // 2, 2)`` (auto-sized).
    dropout : float
        Dropout applied after the fusion layer.
    """

    def __init__(
        self,
        group_sizes: list[int],
        embedding_dim: int,
        group_hidden: int = 0,
        dropout: float = 0.2,
    ):
        super().__init__()
        self._splits = group_sizes

        # Per-group encoders
        encoders: list[nn.Module] = []
        concat_dim = 0
        for n_feat in group_sizes:
            # Auto-size: ceil(2n/3) with floor of 3 — avoids destructive
            # bottlenecks on small groups while still compressing large ones.
            out_dim = group_hidden if group_hidden > 0 else max(-(-2 * n_feat // 3), 3)
            encoders.append(nn.Sequential(
                nn.Linear(n_feat, out_dim),
                nn.ReLU(),
            ))
            concat_dim += out_dim
        self.group_encoders = nn.ModuleList(encoders)

        # Fusion
        self.fusion = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(concat_dim, embedding_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        parts = x.split(self._splits, dim=-1)
        encoded = [enc(p) for enc, p in zip(self.group_encoders, parts)]
        return self.fusion(torch.cat(encoded, dim=-1))


def _build_static_encoder(config: Config) -> StaticEncoder | GroupedStaticEncoder:
    """Instantiate the correct static encoder based on config."""
    embed_dim = config.static_embedding_dim
    group_sizes = config.static_group_sizes
    if group_sizes is not None:
        return GroupedStaticEncoder(
            group_sizes=group_sizes,
            embedding_dim=embed_dim,
            group_hidden=config.static_group_hidden,
            dropout=config.static_dropout,
        )
    n_static = len(config.effective_static_features)
    return StaticEncoder(
        n_static, embed_dim,
        hidden_size=config.static_hidden_size,
        dropout=config.static_dropout,
    )


class CMALHead(nn.Module):
    """Countable Mixture of Asymmetric Laplacians output head.

    Produces K mixture components, each parameterised by a weight πₖ,
    location μₖ, left scale b_L,k, and right scale b_R,k.  The
    asymmetric Laplace naturally handles the skewed, heavy-tailed
    nature of streamflow distributions.

    Parameters
    ----------
    input_size : int
        Dimension of the incoming hidden state.
    n_components : int
        K — number of mixture components (default 3).
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
    """Two-branch LSTM for rainfall–runoff simulation (fast + slow).

    In deterministic mode, uses multiplicative composition:
    ``q_total = q_slow * (1 + fast_ratio)``.  The slow pathway sets the
    baseflow level; the fast pathway is a dimensionless storm amplifier,
    so storm contribution scales with antecedent wetness.

    In CMAL mode, a ``CMALHead`` on concatenated ``[h_slow, h_fast]``
    parameterises the predictive distribution.  The pathway heads are
    retained so that the auxiliary loss can still supervise ``q_slow``
    and ``q_fast`` against Lyne–Hollick targets, keeping the pathway
    representations physically grounded.
    """

    def __init__(self, config: Config):
        super().__init__()
        n_dynamic = len(config.dynamic_features)
        self.fast_window = config.fast_window
        self.info_gap = config.info_gap
        self._use_cmal = config.output_type == "cmal"

        embed_dim = config.static_embedding_dim

        self.static_encoder = _build_static_encoder(config)

        input_size = n_dynamic + embed_dim

        # Fast LSTM (event-scale window)
        self.fast_lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=config.fast_hidden_size,
            batch_first=True,
        )

        # Slow LSTM (full window — baseflow / seasonal)
        self.slow_lstm = nn.LSTM(
            input_size=input_size,
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
        # Fast head: dimensionless storm amplifier (≥ 0)
        # Multiplied onto q_slow → storm response scales with baseflow
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

        # Learned per-basin scale head (always on).  Zero-init so scale = 1
        # at init; absorbs per-basin amplitude end-to-end during training.
        # In CMAL mode, the scale multiplies mu, b_l, b_r so the full
        # mixture distribution scales correctly.
        self.scale_head = ScaleHead(embed_dim)

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

        # Encode static attributes and tile across sequence
        e = self.static_encoder(x_static)                     # (B, E)
        e_full = e.unsqueeze(1).expand(-1, T, -1)             # (B, T, E)
        x_full = torch.cat([x_dynamic, e_full], dim=-1)       # (B, T, D+E)

        # Per-basin learned scale (B,); scale = 1 at init
        s = torch.exp(self.scale_head(e))                     # (B,)

        # ----- fast pathway (last fast_window days) -----
        _, (h_fast, _) = self.fast_lstm(x_full[:, -self.fast_window :, :])
        h_fast = self.dropout(h_fast.squeeze(0))              # (B, H_fast)
        fast_ratio = self.fast_head(h_fast)                   # (B, 1) dimensionless ≥ 0

        # ----- slow pathway -----
        if self.info_gap:
            slow_input = x_full[:, :-self.fast_window, :]
        else:
            slow_input = x_full
        _, (h_slow, _) = self.slow_lstm(slow_input)
        h_slow = self.dropout(h_slow.squeeze(0))              # (B, H_slow)
        q_slow = self.slow_head(h_slow)                       # (B, 1) mm/d baseflow

        # ----- pathway outputs for auxiliary loss -----
        # Multiplicative composition: q_total = q_slow * (1 + fast_ratio)
        # q_fast_contrib is the storm amplification above baseflow.
        q_fast_contrib = q_slow * fast_ratio                  # (B, 1) mm/d storm runoff

        if self._use_cmal:
            # CMAL distribution from combined hidden states, scaled per basin
            h_combined = torch.cat([h_slow, h_fast], dim=-1)  # (B, H_slow + H_fast)
            pi, mu, b_l, b_r = self.cmal_head(h_combined)
            s_unsq = s.unsqueeze(-1)                          # (B, 1) → broadcast over K
            mu = mu * s_unsq
            b_l = b_l * s_unsq
            b_r = b_r * s_unsq
            self._last_cmal_params = (pi, mu, b_l, b_r)
            # E[Y] = Σ πₖ (μₖ + b_R,k − b_L,k)  — already scaled
            q_total = (pi * (mu + b_r - b_l)).sum(dim=-1).clamp(min=0.0)
            q_fast_out = (q_fast_contrib.squeeze(-1) * s)
            q_slow_out = (q_slow.squeeze(-1) * s)
            return q_total, q_fast_out, q_slow_out

        # ----- deterministic multiplicative composition -----
        q_total = q_slow * (1.0 + fast_ratio)                 # (B, 1)

        q_total = q_total.squeeze(-1) * s
        q_fast_out = q_fast_contrib.squeeze(-1) * s
        q_slow_out = q_slow.squeeze(-1) * s

        return q_total, q_fast_out, q_slow_out


class SingleLSTM(nn.Module):
    """Single-branch LSTM baseline — full 365-day lookback."""

    def __init__(self, config: Config):
        super().__init__()
        n_dynamic = len(config.dynamic_features)
        self._use_cmal = config.output_type == "cmal"

        self.static_encoder = _build_static_encoder(config)

        input_size = n_dynamic + config.static_embedding_dim

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

        # Learned per-basin scale head (always on)
        self.scale_head = ScaleHead(config.static_embedding_dim)

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

        s = torch.exp(self.scale_head(e_s))                       # (B,)

        if self._use_cmal:
            pi, mu, b_l, b_r = self.head(h)
            s_unsq = s.unsqueeze(-1)
            mu = mu * s_unsq
            b_l = b_l * s_unsq
            b_r = b_r * s_unsq
            self._last_cmal_params = (pi, mu, b_l, b_r)
            q_total = (pi * (mu + b_r - b_l)).sum(dim=-1).clamp(min=0.0)
        else:
            q_total = self.head(h).squeeze(-1) * s                # (B,)

        zeros = torch.zeros_like(q_total)
        return q_total, zeros, zeros


class MoELSTM(nn.Module):
    """Mixture-of-Experts LSTM with learnable temperature (MoE-τ).

    K independent LSTM experts process the input sequence.  An
    LSTM-Attention gating network computes sequence-level expert
    weights via temperature-scaled softmax. The mixture of experts'
    final hidden states is linearly projected to discharge.

    Returns ``(q_total, zeros, zeros)`` to match the 3-tuple interface.
    """

    def __init__(self, config: Config):
        super().__init__()
        n_dynamic = len(config.dynamic_features)
        self.n_experts = config.moe_n_experts

        # Static encoder (shared across experts and gate)
        self.static_encoder = _build_static_encoder(config)

        input_size = n_dynamic + config.static_embedding_dim
        D_h = config.moe_expert_hidden_size
        D_g = config.moe_gate_hidden_size
        D_a = config.moe_attention_dim
        K = config.moe_n_experts

        # --- Expert LSTMs ---
        self.experts = nn.ModuleList([
            nn.LSTM(input_size=input_size, hidden_size=D_h, batch_first=True)
            for _ in range(K)
        ])

        # --- Gating network: LSTM + temporal attention ---
        self.gate_lstm = nn.LSTM(
            input_size=input_size, hidden_size=D_g,
            batch_first=True,
        )
        # Attention parameters: u_t = v^T tanh(W g_t)
        self.attn_W = nn.Linear(D_g, D_a, bias=False)
        self.attn_v = nn.Linear(D_a, 1, bias=False)
        # Expert logits: z = W_p c
        self.gate_proj = nn.Linear(D_g, K, bias=False)

        # Learnable log-temperature (initialised so τ = moe_tau_init)
        # τ = sigmoid(log_tau_param) keeps 0 < τ < 1
        self._log_tau = nn.Parameter(
            torch.tensor(_inv_sigmoid(config.moe_tau_init))
        )

        self.dropout = nn.Dropout(config.dropout)

        # Number of gate outputs (for diagnostic logging)
        self.n_gate_outputs = K

        # Linear probe: ŷ = w^T m + b, with Softplus for non-negative flow
        self.head = nn.Sequential(
            nn.Linear(D_h, 1),
            nn.Softplus(),
        )

        # Learned per-basin scale head (always on)
        self.scale_head = ScaleHead(config.static_embedding_dim)

    @property
    def tau(self) -> torch.Tensor:
        """Learnable temperature 0 < τ < 1."""
        return torch.sigmoid(self._log_tau)

    def forward(
        self,
        x_dynamic: torch.Tensor,
        x_static: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, T, _ = x_dynamic.shape

        # Encode static and tile across sequence
        e_s = self.static_encoder(x_static)                    # (B, E)
        e_full = e_s.unsqueeze(1).expand(-1, T, -1)            # (B, T, E)
        x = torch.cat([x_dynamic, e_full], dim=-1)             # (B, T, D+E)

        # --- Expert forward passes: collect final hidden states ---
        # h_experts: (B, K, D_h)
        h_list = []
        for expert in self.experts:
            _, (h_k, _) = expert(x)                            # h_k: (1, B, D_h)
            h_list.append(h_k.squeeze(0))                      # (B, D_h)
        h_experts = torch.stack(h_list, dim=1)                 # (B, K, D_h)

        # --- Gating network ---
        g, _ = self.gate_lstm(x)                               # (B, T, D_g)

        # Temporal attention: α_t = softmax(v^T tanh(W g_t))
        u = self.attn_v(torch.tanh(self.attn_W(g)))            # (B, T, 1)
        alpha = torch.softmax(u, dim=1)                        # (B, T, 1)
        c = (alpha * g).sum(dim=1)                             # (B, D_g)

        # Expert logits and temperature-scaled softmax
        z = self.gate_proj(c)                                  # (B, K)
        tau = self.tau.clamp(min=1e-4)
        pi = torch.softmax(z / tau, dim=-1)                    # (B, K)

        # Store batch-mean gate weights for diagnostics
        self._last_pi = pi.detach().mean(dim=0)                # (K,)

        # Mixture of expert hidden states
        m = (pi.unsqueeze(-1) * h_experts).sum(dim=1)          # (B, D_h)
        m = self.dropout(m)

        q_total = self.head(m).squeeze(-1)                     # (B,)
        q_total = q_total * torch.exp(self.scale_head(e_s))
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
