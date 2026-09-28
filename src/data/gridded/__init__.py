"""Statewide 1/16° gridded inputs: Livneh daily forcing + AlphaEarth static (2017; 2017-2025 mean).

Modules
-------
lattice   the product grid (cell keys, dense indices, UTM zones, cell list)
ncio      shared NetCDF / checksum / provenance helpers
forcing   WGEN NonDetrend-Unsplit store → one daily NetCDF per variable
aef       Earth Engine AlphaEarth reduction → static embedding NetCDFs (2017; 2017-2025 mean)

Entry point: ``scripts/prepare_gridded.py``.  Dataset card:
``data/gridded/README.md``.
"""
