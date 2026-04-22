Architecture
============

Two model variants are the focus of this documentation, selected via
``model_type`` in the experiment TOML: a **dual-pathway LSTM** and a
**single-LSTM baseline**. A third variant, a mixture-of-experts LSTM
(``model_type="moe"``), also exists in the code but is not covered here.
All hyperparameters (hidden sizes, window lengths, feature lists, etc.)
are configurable — see ``scripts/config_dual_lstm_kfold.toml`` and
``scripts/config_single_lstm_kfold.toml`` for current values.

Both variants return the 3-tuple ``(q_total, q_fast, q_slow)`` from
``forward()``. For the single-LSTM, ``q_fast`` and ``q_slow`` are zeros
so the same downstream code handles both models.

Dual-Pathway LSTM (default)
----------------------------

Two parallel LSTM branches model distinct hydrological response timescales
with **multiplicative composition**:

- **Fast pathway**: LSTM (hidden size ≈ 64), last ``fast_window`` days
  of the input window (default 28). Captures storm runoff, event
  recession, and direct surface response.
- **Slow pathway**: LSTM (hidden size ≈ 108–128), full 365-day window
  (or the first ``seq_len − fast_window`` days when ``info_gap=true``).
  Captures baseflow, snowmelt dynamics, and seasonal soil-moisture
  storage.

Composition: ``q_total = q_slow × (1 + q_fast_raw)``

The slow pathway sets the baseflow level; the fast pathway emits a
dimensionless storm amplifier so that storm contribution scales with
antecedent wetness — physically realistic because wetter catchments
produce more runoff from the same precipitation event.

Both pathway heads use **Softplus** activation — strictly positive, with
smooth gradients and well-behaved output near zero. This guarantees
non-negative flow predictions without the dead-gradient problems of ReLU.

Single-LSTM Baseline
---------------------

One LSTM (hidden size = 128) processing the full 365-day lookback. Same
static encoder, same Softplus head, no pathway decomposition. Returns
zeros for ``q_fast`` and ``q_slow`` to keep the 3-tuple interface
stable.

Static Encoder (shared)
------------------------

Both architectures use the same static encoder — a small MLP that
projects 18 raw watershed attributes into a 10-dimensional embedding
(via a 32-unit hidden layer), which is then tiled across each dynamic
timestep. This conditions the LSTMs on watershed properties so the same
precipitation signal produces appropriately different runoff responses
for different basins. An alternative **grouped encoder** (enabled via
``static_feature_groups``) gives each semantic group (topography,
network, soil/surface, climate) its own small sub-encoder before a
fusion layer.

Static features include watershed geometry (area, mean elevation, slope),
river network characteristics (river area, density, gradient), soil and
surface properties (forest fraction, clay/sand content, climate and
lithology classes), and long-term climate normals (mean precipitation,
PET, aridity index, snow fraction, high/low precipitation frequency and
duration). ``total_Shape_Area_km2`` and ``ria_ha_usu`` are
log10-transformed before z-score normalisation.

Per-Basin Scale Head
--------------------

Both variants carry a small **ScaleHead** on the static embedding that
emits ``log(s_b)``; pathway outputs are multiplied by ``exp(log s_b)``.
The final layer is zero-initialised so training starts with ``s = 1`` for
every basin (identical to the ``precip_mean`` baseline normalisation),
after which the scale drifts end-to-end under the main loss. This lets
the LSTMs operate in a compact output range while per-basin amplitude is
absorbed by the scale head.

Loss Function
-------------

Total training loss: ``L_total = L_primary + w_aux × L_aux``

**Primary loss** supervises the total predicted streamflow (``q_total``)
against the observed target in dimensionless runoff-ratio units (flow
divided by each basin's mean daily precipitation):

- **Blended MSE + log-MSE** (default, with ``log_loss_lambda > 0``):
  ``L = MSE(Q, Q̂) + λ · MSE(log(Q+ε), log(Q̂+ε))``. The log-space term
  amplifies sensitivity to low flows. Set ``log_loss_lambda = 0`` for
  pure MSE.

**Auxiliary loss** (dual-pathway only) supervises each pathway component
against targets from Lyne–Hollick digital baseflow separation (α=0.925):

- Quickflow → target for ``q_fast``, with an **event-adaptive extreme
  ramp**: the fast-pathway squared error is boosted from 1 up to
  ``extreme_peak_boost`` as normalised total flow rises from each
  basin's ``extreme_start_quantile`` to its ``extreme_top_quantile``
  (per-basin thresholds computed at fold-init time).
- Baseflow → target for ``q_slow``.

These are soft targets, not hard constraints — the model can deviate
from the Lyne–Hollick decomposition where the data supports it.

