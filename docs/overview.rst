Overview
========

neuralhyd-ca predicts daily streamflow for 216 California USGS watersheds
using LSTM networks conditioned on static watershed attributes. The model
takes a lookback window of observed daily climate forcing (precipitation,
tmax, tmin) combined with physical watershed properties and predicts
today's streamflow. This is a **hindcast** model — it uses observed climate
inputs, not future predictions.

The Streamflow Explorer web application visualises these predictions
alongside observed flows and process-based model results (VIC) for direct
comparison.

Watershed Tiers
---------------

Basins are grouped into three hydroclimatic tiers based on elevation,
temperature, and snow influence:

- **Tier 1** (88 basins) — warm, low-elevation, rainfall-dominated.
  Runoff responds quickly to precipitation with minimal snow storage.
- **Tier 2** (97 basins) — transitional, mixed rain-snow. The hardest
  tier to generalise — high internal heterogeneity and mixed response
  timescales.
- **Tier 3** (31 basins) — cold, high-elevation, snow-dominated.
  Requires long memory (365-day lookback) to capture multi-month lags
  between precipitation and runoff from snowmelt.

Performance Metrics
-------------------

Model quality is evaluated with four complementary metrics:

- **NSE** (Nash-Sutcliffe Efficiency) — overall fit, sensitive to peaks.
  Perfect score = 1; score > 0 beats the mean-flow baseline.
- **KGE** (Kling-Gupta Efficiency) — decomposes error into correlation,
  bias, and variability. Less peak-dominated than NSE. Perfect = 1.
- **FHV** (percent bias in high flows) — bias in the top 2% of flows.
  Positive = over-prediction; negative = under-prediction of peaks.
- **FLV** (percent bias in low flows) — bias in the bottom 30% of flows.
  Captures baseflow and recession performance.
