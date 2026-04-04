Data
====

Training Data
-------------

- 216 USGS watersheds across California
- Daily climate forcing: precipitation (mm), tmax (°C), tmin (°C)
- Climate records: 1915–2018
- Streamflow records vary by basin (typically 1950s–1980s onward)

Validation
----------

3-fold stratified spatial cross-validation. Basins are the unit of
splitting — **no watershed appears in both train and validation within
a fold**. This tests ungauged-basin generalisation.

External Comparison
-------------------

Process-based model outputs (VIC, NOAH-MP) from the California Energy
Commission are available for overlapping basins, enabling direct
comparison between data-driven and physics-based approaches.
