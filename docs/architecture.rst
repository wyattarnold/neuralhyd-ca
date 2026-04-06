Architecture
============

Two model variants are available, selected via ``model_type`` in
``config.toml``. All hyperparameters (hidden sizes, window lengths, feature
lists, etc.) are configurable — see ``scripts/config.toml`` for current values.

Dual-Pathway LSTM (default)
----------------------------

Two parallel LSTM branches model distinct hydrological response timescales
with **multiplicative composition**:

- **Fast pathway**: LSTM (hidden=64), last 18 days of the input window.
  Captures storm runoff, event recession, and direct surface response.
- **Slow pathway**: LSTM (hidden=128), full 365-day window.
  Captures baseflow, snowmelt dynamics, and seasonal soil-moisture storage.

Composition: ``q_total = q_slow × (1 + q_fast_raw)``

The slow pathway sets the baseflow level; the fast pathway acts as a
dimensionless storm amplifier so that storm contribution scales with
antecedent wetness — physically realistic because wetter catchments
produce more runoff from the same precipitation event.

Both pathway heads use **Softplus** activation — strictly positive, with
smooth gradients and well-behaved output near zero. This guarantees
non-negative flow predictions without the dead-gradient problems of ReLU.

Single LSTM Baseline
---------------------

One LSTM (hidden=128) processing the full 365-day lookback. Simpler
architecture without flow decomposition. Returns zero for pathway
components to maintain the same ``(q_total, q_fast, q_slow)`` interface.

Static Encoder (shared)
------------------------

Both architectures use the same static encoder — a small MLP that projects
16 raw watershed attributes into a 10-dimensional embedding (via a 32-unit
hidden layer), which is then tiled across each dynamic timestep. This
conditions the LSTMs on watershed properties so the same precipitation
signal produces appropriately different runoff responses for different
basins.

Static features include watershed geometry (area, slope), land cover
(forest fraction), soil texture, river network characteristics,
geology/lithology classes, and long-term climate normals (mean
precipitation, PET, aridity index, snow fraction). Features with heavy
right skew (e.g. area) are log-transformed before normalisation.

Loss Function
-------------

The total training loss: ``L_total = L_primary + w_aux × L_aux``

**Primary loss** supervises total predicted streamflow (``q_total``) against
observed flow, both normalised by per-basin std. Two modes are available:

- **MSE** (default): Standard mean squared error.
- **Blended MSE + log-MSE**: ``L = (1−λ)·MSE(Q,Q̂) + λ·MSE(log(Q+ε),log(Q̂+ε))``.
  The log-space term amplifies sensitivity to low flows.

**Auxiliary loss** (dual-pathway only) supervises each pathway component
against targets from Lyne-Hollick digital baseflow separation (α=0.925):

- Quickflow → target for ``q_fast`` (with asymmetric weighting — under-prediction
  of storm peaks is penalised more heavily)
- Baseflow → target for ``q_slow``

These are soft targets, not hard constraints. The model can deviate from
the Lyne-Hollick decomposition where the data supports it.

Training
--------

- **Adam** optimiser with weight decay
- **Warmup → cosine annealing** LR schedule
- **Gaussian input noise** on normalised climate inputs — strong regulariser
  that improves generalisation to unseen basins
- **Stochastic Weight Averaging (SWA)** — activates on the first learning
  plateau and averages weights until a second plateau
- **Gradient clipping** for training stability
- **Early stopping** with a minimum relative improvement threshold
