# CDF Distribution Analysis by Tier

## Overview

This report summarizes the distributional characteristics of 216 California USGS watersheds grouped into three tiers based on monthly flow ~ precipitation + temperature regression R²: **Tier 1** (R² > 0.60, 88 watersheds), **Tier 2** (0.20 ≤ R² ≤ 0.60, 97 watersheds), and **Tier 3** (R² < 0.20, 31 watersheds). Higher R² indicates a stronger direct monthly relationship between precipitation/temperature and streamflow.

---

## Record Length

![Observed Streamflow Days](cdf_observed_flow_days.pdf)

Tier 3 watersheds tend to have **longer observational records** — roughly 60% exceed 10,000 days (~27 years), compared to ~40% for Tier 1 and ~45% for Tier 2. Tier 1 watersheds skew shortest, with a median around 8,000 days. Tier 2 sits in between with a median around 9,000 days.

---

## Summary CDFs (Tier Averages)

![Summary Averages](cdf_summary.pdf)

### Daily Streamflow (mm)
Tier 3 watersheds produce **higher daily flows on average**, with their CDF shifted right on the log scale. Both Tier 1 and Tier 2 show a nonzero y-intercept at the lowest flow values, indicating a fraction of near-zero flow days — Tier 2 has the largest offset (~14% of days), followed by Tier 1 (~8%), consistent with intermittent or ephemeral streams. Tier 3 watersheds start closer to 0.1mm/day, reflecting more sustained baseflow.

### Daily Tmax & Tmin
Temperature distributions reveal a clear **elevation/climate gradient** across all three tiers. Tier 1 is **warmest** (Tmax median ~25°C; Tmin median ~8°C), Tier 2 is **intermediate** (Tmax median ~20°C; Tmin median ~3°C), and Tier 3 is **coldest** (Tmax median ~15–18°C; Tmin median ~−3 to −5°C). Temperature is a strong separator between tiers, and Tier 2 occupies a middle position rather than overlapping with either extreme.

### Non-Zero Precipitation (>0.1 mm)
The distribution of non-zero precipitation amounts is remarkably similar across all three tiers on the log scale, indicating that **when it rains, it rains comparably** regardless of tier.

### Non-Zero Precipitation Day Fraction
Tier 3 watersheds have the **highest fraction of wet days** (median ~0.37), clearly separated from Tier 1 and Tier 2 which largely overlap (both with medians ~0.28–0.30). The main distinction is between Tier 3 and the rest, rather than a smooth three-tier gradient. However, Tiers 1 and 2 span a wider range of wet-day fractions, having both the driest and wettest (by frequency) basins.

### Annual Q/P Ratio
Tier 1 ratios cluster in the 0.01–1.0 range, reflecting moderate and relatively consistent aridity — these warm, low-elevation basins lose a meaningful but predictable fraction of precipitation to ET, so annual runoff tracks precipitation with a roughly stable reduction factor. The moderate aridity index range (1–6) translates directly into this bounded, coherent Q/P distribution. Tier 2 spans a far wider range (10⁻⁵ to 10⁰), which is a direct consequence of it being the **most water-limited tier** (aridity index 2–9, the highest and widest of the three). At the dry end, the most arid Tier 2 basins lose nearly all precipitation to ET and deep percolation, driving Q/P toward near-zero; at the wetter end, basins with some snow influence generate runoff well above what their aridity would suggest in a purely rainfall system. Tier 3 is shifted furthest right, with the **highest runoff ratios overall** and some watersheds exceeding Q/P = 1.0 — physically implying snowmelt carryover from prior years contributing to annual discharge beyond what current-year precipitation alone would explain.

---

## Per-Tier Individual Watershed CDFs

![Tier 1 Detail](cdf_detail_tier_1.pdf)
![Tier 2 Detail](cdf_detail_tier_2.pdf)
![Tier 3 Detail](cdf_detail_tier_3.pdf)

### Streamflow
Tier 1 CDFs show **distinct shelves at low flows** — many watersheds have a significant fraction of near-zero flow days, characteristic of ephemeral or intermittent streams. Tier 2 shows equally prominent shelves, with some watersheds having 40–60% of days at minimal flow; this tier actually has the highest average zero-flow fraction. Tier 3 CDFs are smoother and more continuously distributed, consistent with sustained baseflow in snowmelt-dominated basins.

### Tmax & Tmin
All three tiers show tight within-tier clustering for temperature, indicating that **tier membership captures a real climatic grouping**, not just noise. Tier 1 clusters in the warmest range, Tier 2 in the middle, and Tier 3 values the coldest. Tier 2 temperature spread is wider than Tier 1 or 3, consistent with spanning a broader range of elevations.

### Non-Zero Precipitation (>0.1mm/day)
Precipitation CDFs are tightly bundled within each tier and nearly identical across tiers — the three tiers are essentially indistinguishable for precipitation amounts. The spread within each tier is narrowest in Tier 1 and wider in Tiers 1 and 2.

---

## Annual Q/P Ratio by Tier

![Q/P Ratio](cdf_annual_qp_ratio.pdf)

Tier 1 Q/P ratios cluster tightly between 0.01 and 1.0, with most watersheds having years across the 0.05–0.5 range. The moderate and relatively homogeneous aridity index of Tier 1 basins (1–6) is the key driver: ET demand is consistently meaningful but not dominant, so most precipitation that falls is split between a predictable runoff fraction and atmospheric loss. The tight clustering reflects that aridity conditions are fairly uniform within this tier. Tier 2 shows dramatically more spread (10⁻⁵ to 10⁰), and this is a direct expression of its status as the **most arid and most internally heterogeneous tier** (aridity index 2–9). Basins at the high end of Tier 2 aridity are strongly water-limited — almost all annual precipitation is consumed by ET, pushing annual Q/P toward near-zero or below 10⁻³. Basins at the lower end of Tier 2 aridity, often those with partial snowpack, generate enough runoff to approach Q/P ~ 1.0. This spread makes Tier 2 the least predictable tier for runoff estimation. Tier 3 is the most shifted right, with some watersheds showing Q/P > 1.0 in some years, a hallmark of **snow-dominated hydrology** where inter-annual snowpack storage decouples annual flow from same-year precipitation.

---

## Static Attributes

![Static Attributes](cdf_static_attributes.pdf)

The static attribute CDFs confirm that the tiers correspond to **distinct physiographic and climatic regimes**:

| Attribute | Tier 1 | Tier 2 | Tier 3 | Interpretation |
|-----------|--------|--------|--------|----------------|
| **Elevation** | Low (median ~600 m) | Intermediate (median ~1250 m) | High (median ~2250 m) | **Key differentiator** of tier separation |
| **Slope** | milder slopes | widest range with steepest slopes | smallest range of slopes, tends steeper | Highest diversity terrain in Tier 2 |
| **Mean Precip** | Widest range (~1–7 mm/day) | Tends lower (~1-4 mm/day) | Intermediate, smaller range (~2–5 mm/day) | Highest diversity in Tier 1 |
| **PET** | Generally higher (~7–9 mm/day) | Widest range, tends high (~6–9 mm/day) | Lowest (~5–7 mm/day) | Cooler temperatures suppress PET in Tier 3 |
| **Aridity Index** | Intermediate, with wide range (1-6) | Highest, with wide range (2-9) | Lowest, smaller range (1–3) | Tier 2 = most water-limited |
| **Snow Fraction** | Near zero | Low–moderate (0.05–0.8) | Highest (0.2–0.8) | **Key differentiator** — snow storage breaks monthly P–Q link |
| **Forest Cover** | Wide range, generally higher coverage (~20 to >90%) | Lowest coverage, 40% of watersheds <80% coverage | 90% of watersheds >60% coverage | Tier 2 has most watersheds with lower coverage |
| **Clay Fraction** | Highest, median 17% | Intermediate, median 15% | Lowest, median 12% | Finer-textured soils at lower elevations |
| **Silt Fraction** | Highest, median 33% | Intermediate, median 31% | Lowest, median 29% | Finer-textured soils at lower elevations |
| **Sand Fraction** | Lowest, median 48% | Intermediate, median 54% | Highest, median 57% | Coarser soil in cool, high-elevation basins |

---

## Monthly Averages

![Monthly Averages](monthly_averages.pdf)

The monthly average plot reveals the **seasonal hydrologic regimes** that underpin the tier structure:

### Streamflow
Tier 3 (red) shows a dramatic **snowmelt peak in May–June** (~7 mm/day), with very low flows in winter when precipitation is stored as snowpack. Tier 1 (blue) peaks in **winter (Dec–Feb)** at ~3–4 mm/day, directly tracking precipitation — the hallmark of rainfall-dominated runoff. Tier 2 (green) shows a muted, broad peak from February through May (~1.5 mm/day), reflecting a mix of rain and snowmelt contributions.

### Tmax & Tmin
All tiers follow the expected seasonal cycle, but with consistent offsets. Tier 1 is warmest year-round (summer Tmax ~28–30°C), Tier 2 is intermediate (summer Tmax ~28–30°C, nearly matching Tier 1 in summer but cooler in winter), and Tier 3 is coldest (summer Tmax ~23–25°C, winter Tmin ~−8°C). Tier 3’s cold winters confirm sustained below-freezing conditions that support snowpack accumulation.

### Precipitation
All three tiers share the same **Mediterranean precipitation seasonality** — wet winters, dry summers. Tier 3 receives the most winter precipitation (~7 mm/day in Dec–Jan), Tier 1 is similar (~7 mm/day), and Tier 2 is slightly lower (~5 mm/day). Summer precipitation is negligible across all tiers. **Tier 3 receives similar or slightly more precipitation than Tier 1, but the timing of runoff is completely different** because that precipitation falls as snow.

### Per-Tier Individual Watershed Monthly Averages

![Tier 1 Monthly](monthly_detail_tier_1.pdf)
![Tier 2 Monthly](monthly_detail_tier_2.pdf)
![Tier 3 Monthly](monthly_detail_tier_3.pdf)

The individual watershed monthly plots (with tier mean overlaid in black) show that:
- **Tier 1** watersheds are tightly clustered around the winter rain–runoff pattern, with consistent peak flows in Dec–Feb.
- **Tier 2** shows the **widest spread** in streamflow timing — some watersheds peak in winter (rain-dominated), others peak in spring (snow-influenced), confirming this tier’s transitional character.
- **Tier 3** watersheds are tightly clustered around the May–June snowmelt peak, with very little inter-watershed variability in flow timing.

---

## Key Takeaways

1. **Tier membership is primarily a proxy for elevation and snow dominance.** Tier 1 watersheds are warm, low-elevation, rainfall-dominated basins where monthly precipitation directly predicts monthly flow. Tier 3 watersheds are cold, high-elevation, snow-dominated basins where snowpack storage introduces a multi-month lag that a simple monthly regression cannot capture. Tier 2 spans the transition zone with the widest internal variability — some watersheds behave like Tier 1, others like Tier 3, and many shift between regimes year to year.

2. **Tier 3 is not "bad data" — it is a different hydrologic regime.** The long records, physically consistent Q/P ratios, and coherent temperature/precipitation distributions confirm these are real signals. An LSTM must learn to represent snow accumulation and delayed melt as internal state — it will not find a direct same-timestep P→Q mapping in these basins.

3. **Tier 2 is the hardest generalization challenge.** The mixed rain/snow behavior, high heterogeneity, and wide variability in flow timing (some watersheds peak in winter, others in spring) mean Tier 2 cannot be learned by specializing in either rainfall-runoff or snowmelt dynamics alone. A model that handles Tier 2 well has necessarily learned the full continuum — making Tier 2 performance the best single indicator of generalization quality.

4. **Precipitation forcing is similar across tiers** when it falls, but frequency, phase (rain vs. snow), and soil properties differ substantially. Tier 1 soils are finer-textured (more clay and silt, less sand), while Tier 3 soils are coarser (highest sand fraction). Climate inputs alone are insufficient — the LSTM needs snow fraction, temperature, soil texture, and lagged precipitation as features. Without explicit snow state, the model must infer storage from temperature and multi-month precipitation history, which places a direct requirement on **sequence length**.

5. **The ungauged-basin problem is the central challenge.** The 216 watersheds span three distinct hydrologic regimes, but any operational deployment will encounter sites the model was never trained on. The train/test regime must evaluate whether the LSTM learns transferable hydrologic representations — not site-specific parameter fits — by holding out entire watersheds at test time.

---

## Recommended Train/Test Regime

The goal is an LSTM that performs well **out of sample in any tier** — on watersheds it has never seen, across all three hydrologic regimes. The evaluation design must explicitly test this.

### Primary Evaluation: Tier-Stratified Spatial K-Fold

- Split the 216 watersheds into K=5 folds, with **proportional tier representation in each fold** (~18 Tier 1, ~19 Tier 2, ~6 Tier 3 held out per fold).
- The model never sees the held-out watersheds during training.
- Report KGE (and NSE) **per tier per fold**, not just overall. A model that scores well on average by nailing Tier 1 while failing Tier 3 is not acceptable.
- This directly measures ungauged-basin capability: can the LSTM transfer learned hydrologic dynamics to sites with different characteristics?

### Static Attributes as LSTM Conditioning Features

The static attribute analysis shows that **the same climate sequence produces fundamentally different runoff depending on watershed properties**. The LSTM must receive static attributes (elevation, soil texture, snow fraction, aridity index, forest cover, drainage area) as conditioning inputs — either concatenated to each timestep, injected via an embedding layer, or used to modulate hidden state initialization. Without conditioning, a single LSTM cannot distinguish between a Tier 1 watershed at 600 m and a Tier 3 watershed at 2250 m receiving the same precipitation.

Key static features for cross-tier generalization:

| Feature | Role | Critical for |
|---|---|---|
| **Snow fraction** | Distinguishes rain vs. snow storage | Tier 2 ↔ Tier 3 boundary |
| **Elevation** | Primary tier separator | All cross-tier transfer |
| **Aridity index** | Separates water-limited from energy-limited basins | Tier 1 ↔ Tier 2 boundary |
| **Soil texture (clay/silt/sand)** | Modulates infiltration and baseflow | Tier 1 fine soils vs. Tier 3 coarse soils |
| **Forest cover** | Interception and ET partitioning | Tier 2 (lowest/most variable cover) |
| **Drainage area** | Scales response magnitude and timing | All tiers |
