Overview
========

.. warning::

   This project is under active development. Model outputs, evaluation metrics,
   and application features may change without notice.

neuralhyd-ca predicts daily streamflow for 210 California USGS watersheds
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

- **Tier 1** (89 basins) — warm, low-elevation, rainfall-dominated.
  Runoff responds quickly to precipitation with minimal snow storage.
- **Tier 2** (92 basins) — transitional, mixed rain-snow. The hardest
  tier to generalise — high internal heterogeneity and mixed response
  timescales.
- **Tier 3** (29 basins) — cold, high-elevation, snow-dominated.
  Requires long memory (365-day lookback) to capture multi-month lags
  between precipitation and runoff from snowmelt.

Performance Metrics
-------------------

Model quality is evaluated with five complementary metrics:

- **NSE** (Nash–Sutcliffe Efficiency) — overall fit, sensitive to peaks.
  Perfect score = 1; score > 0 beats the mean-flow baseline.
- **KGE** (modified Kling–Gupta Efficiency, Kling et al. 2012) —
  decomposes error into correlation, bias, and variability (ratio of
  coefficients of variation). Less peak-dominated than NSE. Perfect = 1.
- **FHV** (percent bias in high flows) — volume bias in the top 2% of
  the flow-duration curve. Positive = over-prediction; negative =
  under-prediction of peaks.
- **FEHV** (percent bias in extreme high flows) — the same bias in the
  top 0.1% of the flow-duration curve; emphasises the rarest peaks.
- **FLV** (percent bias in low flows) — log-space bias in the bottom 30%
  of the flow-duration curve, each curve measured from its own minimum.
  Captures baseflow and recession performance.

FHV, FEHV and FLV follow Yilmaz et al. (2008): observed and simulated
flows are sorted independently, so they compare the two flow-duration
curves (flow magnitudes), not flows on the same days (timing).
