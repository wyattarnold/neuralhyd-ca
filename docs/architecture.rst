Architecture
============

Two model variants are available via the ``model_type`` configuration:

Dual-Pathway LSTM (default)
----------------------------

Two LSTM branches with multiplicative composition:

- **Fast pathway**: LSTM (hidden=64), 18-day window → storm runoff
- **Slow pathway**: LSTM (hidden=128), 365-day window → baseflow, snowmelt

Composition: ``q_total = q_slow × (1 + q_fast_raw)``

The slow pathway sets the baseflow level; the fast pathway acts as a
dimensionless storm amplifier that scales with antecedent wetness.

Single LSTM Baseline
---------------------

One LSTM (hidden=128) processing the full 365-day lookback. Simpler
architecture used for benchmarking.

Both models use a static encoder MLP that projects 16 watershed attributes
into a 10-dimensional embedding concatenated to each dynamic timestep.
