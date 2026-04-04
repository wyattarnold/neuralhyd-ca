Overview
========

The NeuralHydrology Explorer visualises streamflow predictions produced by
LSTM neural networks trained on 216 California USGS watersheds across three
hydrologic tiers:

- **Tier 1** — warm, low-elevation, rainfall-dominated (88 basins)
- **Tier 2** — transitional, mixed rain-snow (97 basins)
- **Tier 3** — cold, high-elevation, snow-dominated (31 basins)

The model uses a 365-day lookback window of daily precipitation, maximum
temperature, and minimum temperature, conditioned on 16 static watershed
attributes (elevation, soil texture, snow fraction, aridity, etc.).

Performance Metrics
-------------------

- **NSE** — Nash-Sutcliffe Efficiency
- **KGE** — Kling-Gupta Efficiency
- **FHV** — percent bias in high flows (top 2%)
- **FLV** — percent bias in low flows (bottom 30%)
