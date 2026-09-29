# Statewide 1/16° Gridded Inputs

Dense (lat, lon) products for California on the Livneh 1/16° lattice: daily precipitation and temperature, 1915–2018, one NetCDF per variable, and the AlphaEarth Foundations satellite embedding as a static 64-band vector per cell, both for 2017 alone and (optional) as the 2017–2025 multi-year mean. Everything here is built by [`scripts/prepare_gridded.py`](../../scripts/prepare_gridded.py) (modules in [`src/data/gridded/`](../../src/data/gridded)); path constants are the `GRIDDED_*` names in [`src/paths.py`](../../src/paths.py).

These are prepared inputs only. No model config, dataset class, or loader reads them yet. The NetCDFs (~3 GB) are Git LFS objects that are **not fetched by default**; see [Fetching](#fetching-git-lfs).

## Files

| File | Contents | Approx. size | Stored as |
|---|---|---|---|
| `livneh_precip_mm_daily_1915-2018.nc` | `precip_mm(time, lat, lon)`, the x10 correction table, `x10_count(lat, lon)` | 0.55 GB | Git LFS, opt-in |
| `livneh_tmax_c_daily_1915-2018.nc` | `tmax_c(time, lat, lon)`, `swap_count(lat, lon)` | 1.25–1.3 GB | Git LFS, opt-in |
| `livneh_tmin_c_daily_1915-2018.nc` | `tmin_c(time, lat, lon)`, `swap_count(lat, lon)` | 1.3–1.4 GB | Git LFS, opt-in |
| `alphaearth_2017.nc` | `embedding(band, lat, lon)` plus per-cell coverage diagnostics | a few MB (7.4 MB before compression) | Git LFS, opt-in |
| `alphaearth_2017-2025_mean.nc` | Optional. `embedding(band, lat, lon)` (the 2017–2025 mean), `embedding_year(year, band, lat, lon)` (each year's cell means), per-year coverage and between-year diagnostics | ~35 MB (74 MB before compression) | Git LFS, opt-in |
| `grid_cells.csv` | The 13,786 domain cells: `key, lat, lon, ilat, ilon, utm_zone`, sorted by (lat, lon) | 0.7 MB | Git LFS (repo-wide `*.csv` rule), fetched by default |
| `precip_x10_corrections.csv` | Every x10-corrected cell-day: `key, lat, lon, date, raw_mm, corrected_mm` | ~40 kB | Git LFS (`*.csv`), fetched by default |
| `SHA256SUMS` | sha256 of every product (`sha256sum -c` format) | < 1 kB | plain git |
| `provenance.toml` | How each product was made: tables `[forcing]`, `[alphaearth]` and, once the mean is built, `[alphaearth_mean]` | a few kB (~20 kB with the mean) | plain git |
| `aef_parts/` | Banked Earth Engine partials of the AlphaEarth burn, one directory per year (local only) | – | gitignored |
| `*.part` | Products being written; renamed into place on success | – | gitignored |

There is one forcing file per variable because GitHub LFS caps a single file at 2 GB. Sizes are approximate; the exact byte counts are in `provenance.toml`.

## Grid

All the NetCDFs share one CF-1.8 skeleton (`src/data/gridded/ncio.py`) and one grid (`src/data/gridded/lattice.py`).

| Property | Value |
|---|---|
| Coordinates | Geographic lat/lon, WGS 84 (EPSG:4326). The `crs` grid mapping (`latitude_longitude`) carries the EPSG:4326 WKT in `crs_wkt` and `spatial_ref` plus the CF-1.8 name attributes, so GDAL, rioxarray and pyproj recognise it. |
| Cell size | 1/16° (0.0625°) |
| Cell centres | `0.03125 + k/16`: odd multiples of 1/32, exact binary fractions that survive any float round-trip |
| `lat` | 32.59375 … 43.34375, 173 rows, ascending (south → north), float64 |
| `lon` | -124.34375 … -113.90625, 168 columns, ascending (west → east), float64 |
| Extent (cell edges) | 32.5625–43.375 °N, -124.375 to -113.875 °E |
| Bounds | `lat_bnds`, `lon_bnds` (± 1/32°) |
| Domain | 13,786 cells (47.4 % of the 29,064 grid slots): exactly the cells of the WGEN NonDetrend-Unsplit store. The store is land-only: it has no offshore islands and leaves out some Delta open-water cells. |
| `mask(lat, lon)` | uint8, 1 on the domain cells, 0 elsewhere; identical in every file. Float data variables are NaN where `mask == 0`; integer per-cell variables are 0 there. |
| Cell key | `f"{lat:.5f}_{lon:.5f}"`, identical to the store file names (`data_<key>`) |
| UTM zones | 10N west of -120° (6,509 cells), 11N from -120° to -114° (7,274), 12N east of -114° (3). Both zone edges are cell edges, so no cell straddles a zone. |

`ilat`/`ilon` in `grid_cells.csv` are the dense row/column indices: `lat = 32.59375 + ilat/16`, `lon = -124.34375 + ilon/16`. Because the centres are exact binary fractions, `ds.sel(lat=38.90625, lon=-120.78125)` matches exactly; no `method="nearest"` is needed.

## Daily Forcing

| Variable | File | Dims | dtype | Units | `standard_name` | `cell_methods` |
|---|---|---|---|---|---|---|
| `precip_mm` | `livneh_precip_mm_daily_1915-2018.nc` | (time, lat, lon) | float32 | mm | `lwe_thickness_of_precipitation_amount` | `time: sum area: mean` |
| `tmax_c` | `livneh_tmax_c_daily_1915-2018.nc` | (time, lat, lon) | float32 | degC | `air_temperature` | `time: maximum area: mean` |
| `tmin_c` | `livneh_tmin_c_daily_1915-2018.nc` | (time, lat, lon) | float32 | degC | `air_temperature` | `time: minimum area: mean` |

| Property | Value |
|---|---|
| `time` | int32, `days since 1915-01-01`, `standard` calendar; 37,986 days, 1915-01-01 … 2018-12-31, every Feb 29 included |
| Values | The store's ASCII values (precip 3 decimals, temperature 2 decimals) parsed exactly and cast to float32, after the two corrections below. Nothing else is altered. |
| Fill | `_FillValue` NaN; NaN outside the domain |
| Compression | zlib level 4 + shuffle (zstd/blosc filters are not portable across netCDF4 builds) |
| Chunks | (time=1461, lat=16, lon=16). 1461 days is four years and divides the record exactly (26 × 1461). A 16 × 16 chunk balances a per-cell series read (~0.15–0.2 s) against a statewide one-day map (~0.5 s). |
| No `valid_range` | Real extremes exist, e.g. 814.6 mm at 37.59375_-118.21875 on 1967-01-25 |

Extra variables:

| Variable | File | Dims | dtype | Meaning |
|---|---|---|---|---|
| `x10_lat`, `x10_lon` | precip | (x10_pair) | float64 | Cell of each corrected cell-day |
| `x10_time` | precip | (x10_pair) | int32 | Day of each corrected value (same units/calendar as `time`) |
| `x10_raw_mm` | precip | (x10_pair) | float32 | Raw store value before the correction |
| `x10_corrected_mm` | precip | (x10_pair) | float32 | Stored value (equals `precip_mm` at the pair) |
| `x10_count` | precip | (lat, lon) | int16 | Corrected days per cell (0 outside) |
| `swap_count` | tmax, tmin | (lat, lon) | int32 | Days with tmin > tmax that were swapped; identical in both files |

The pair table is sorted by (time, lat, lon), the same order as `precip_x10_corrections.csv`. `precip_mm` carries the attributes `x10_rule`, `x10_months = [6, 7, 8]` and `x10_threshold_mm = 150.0`.

Global attributes (all three files): `title`, `summary`, `source`, `references`, `license`, `history` (command + git commit), `date_created`, `time_coverage_start/end/resolution`, `input_tree_sha256`, `corrections` (both rules with their counts), `temperature_source_note` (the 2016 seam with before/after inversion counts), `comment` (the GDAL band-count note under [Reading the Data](#reading-the-data)), plus the `geospatial_*` and `grid` attributes of the skeleton. A smoke build made with `--rows A:B` also carries `subset_rows = "A:B"`; a committed product never does, and `verify` fails on one in `data/gridded`.

## AlphaEarth 2017 Embedding

| Variable | Dims | dtype | Meaning |
|---|---|---|---|
| `embedding` | (band, lat, lon) | float32 | Mean of the 2017 AlphaEarth pixel vectors over the cell rectangle. Units `1`, `cell_methods = "area: mean"`, NaN fill, zlib 4 + shuffle, one chunk (64, 173, 168). |
| `band` | (band) | int32 | 0 … 63 |
| `band_name` | (band) | string | `A00` … `A63` (auxiliary coordinate) |
| `valid_frac` | (lat, lon) | float32 | Fraction of the cell rectangle covered by valid (unmasked) pixels |
| `norm` | (lat, lon) | float32 | Euclidean norm of the cell-mean embedding |
| `n_tiles` | (lat, lon) | int8 | AlphaEarth UTM tiles that reached the cell (0 outside) |
| `utm_zone` | (lat, lon) | int8 | 10 / 11 / 12: the zone whose tiles the cell was reduced on (0 outside) |
| `aef_flag` | (lat, lon) | uint8 | Bit flags: 1 `partial_coverage` (`valid_frac` < 1 - 1e-6), 2 `no_valid_pixels` (embedding NaN), 4 `low_coverage` (`valid_frac` < 0.5) |

**Method** (`src/data/gridded/aef.py`):

- Source: Earth Engine `GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL`, calendar year **2017 only**, all 64 bands, no temporal averaging. 2017 is the collection's earliest year and the one closest to the forcing record. The burn refuses to start if Earth Engine lists any image before 2017-01-01.
- Per cell, `mean = Σ(v·w) / Σw`, where `w` carries band A00's mask with Earth Engine's fractional boundary weights. `valid_frac` = Σw divided by the same sum for an unmasked constant on the same grid.
- Each cell is reduced on its own UTM zone's tiles in the tile's native UTM projection, and the per-tile sums are combined. No mosaic is used; the zone edges are cell edges.
- Scale 15 m: Earth Engine reads the full-resolution level on a nearest-neighbour lattice that samples 4/9 of the 10 m pixels. Against the full 10 m mean the difference is ≤ 2e-4 per band (cos ≥ 0.9999998) at about 55 % of the compute. Other scales misbehave (16 m aliases; 19 m and coarser read L2-renormalised pyramid levels), so the code accepts only 10 or 15. `aef --check` re-verifies the 15 m values against an independent 10 m zone-mosaic reduction.
- All 13,786 cells are reduced by the burn, with the same method.
- The values are Earth Engine's dequantised floats, averaged in float64 and stored as float32. About four decimal places are meaningful.

Global attributes: `title`, `summary`, `source` (collection, year, tile count, `DATASET_VERSION` / `MODEL_VERSION` / `PROCESSING_SOFTWARE_VERSION`), `attribution`, `license`, `references`, `modifications`, `method`, `year`, `scale_m`, `images_used`, `dataset_version`, `earliest_year_check`, `history`, `date_created`.

## AlphaEarth 2017–2025 Mean

`alphaearth_2017-2025_mean.nc` is optional. It holds every annual AlphaEarth layer, 2017 through 2025, reduced the same way as the 2017 file, plus their multi-year mean.

| Variable | Dims | dtype | Meaning |
|---|---|---|---|
| `embedding` | (band, lat, lon) | float32 | Equal-weight mean of `embedding_year` over the years in which the cell has valid pixels. Not renormalised. NaN where no year has valid pixels. `cell_methods = "area: mean time: mean"`, one chunk (64, 173, 168). |
| `embedding_year` | (year, band, lat, lon) | float32 | Each year's cell mean, computed exactly as `alphaearth_2017.nc` `embedding`; its 2017 layer is that file's `embedding` bit for bit. NaN on cell-years without valid pixels. Chunks (1, 64, 173, 168). |
| `year` | (year) | int32 | 2017 … 2025 |
| `band`, `band_name` | (band) | int32, string | As in the 2017 file |
| `dataset_version` | (year) | string | `DATASET_VERSION` of each year's tiles (one per year) |
| `n_images_used` | (year) | int32 | UTM tile images used in each year |
| `valid_frac` | (year, lat, lon) | float32 | Per year, the fraction of the cell rectangle covered by valid pixels (0 on cell-years without any) |
| `n_years` | (lat, lon) | int8 | Years with valid pixels, i.e. averaged into `embedding` (9 almost everywhere; 0 outside) |
| `norm` | (lat, lon) | float32 | Euclidean norm of `embedding` |
| `year_cos_min` | (lat, lon) | float32 | Minimum over the valid years of the cosine between a year's cell mean and the multi-year mean |
| `utm_zone` | (lat, lon) | int8 | As in the 2017 file |
| `aef_flag` | (lat, lon) | uint8 | Bit flags over the years: 1 `partial_coverage` (`valid_frac` < 1 - 1e-6 in any year), 2 `no_valid_pixels` (no valid pixels in any year; `embedding` NaN), 4 `low_coverage` (`valid_frac` < 0.5 in any year), 8 `missing_years` (`n_years` < 9) |

**Method.** For each calendar year the burn reduces all 13,786 cells from scratch, with the same method, scale and tiles-per-zone rule as the 2017 file (see [above](#alphaearth-2017-embedding)). `embedding` is the plain average of the per-year float64 cell means over the years with Σw > 0: each such year gets the same weight, the average is not renormalised, and a year with no valid pixels is left out rather than counted as zero. The average is taken in float64 over the per-year means and cast to float32 once, so identical partials always give identical float32 values. `norm` < 1 now measures within-cell **and** between-year heterogeneity; `year_cos_min` isolates the between-year part.

**Dataset versions.** `DATASET_VERSION` is recorded per year from the tiles used. In the 2026-09 build, 2017 and 2025 are 1.1 and 2018–2024 are 1.0. Assembly requires one version within each year, not across years. The per-year versions are in the `dataset_version` variable, the `dataset_version` attribute (`2017: 1.1, 2018: 1.0, …`) and `[alphaearth_mean]` in `provenance.toml`.

**Cost.** About 49 EECU-h per year at 15 m, so 2018–2025 add ~392 EECU-h on top of 2017's ~49 (8 × 128 work units; `aef --dry-run --years all` prints the exact figure for what is left). That is 2–3 months of the noncommercial Community tier's 150 EECU-h, or one burn on a larger quota.

Global attributes: as the 2017 file, with `years` in place of `year`, `source_url`, the per-year `dataset_version` and `images_per_year` strings, the 2017 `earliest_year_check`, and a `modifications` statement that adds the equal-weight multi-year mean.

## Provenance and Lineage

### Forcing

| Item | Source |
|---|---|
| Input store | DWR WGEN NonDetrend-Unsplit statewide store: 13,786 TAB-delimited ASCII files `data_<lat>_<lon>`, columns `year month day precip_mm tmax_c tmin_c`, 37,986 rows each. It is the `Historical_Unsplit` series (the non-temperature-detrended historical baseline) of DWR's release *Gridded Weather Generator Perturbations of Historical Detrended and Stochastically Generated Temperature and Precipitation for the State of CA and HUC8s*, <https://data.ca.gov/dataset/gridded-weather-generator-perturbations-of-historical-detrended-and-stochastically-generated-te>. |
| Precipitation | Livneh-lineage unsplit daily precipitation (Pierce et al. 2021) |
| Temperature | Livneh et al. (2013) for 1915–2015; PRISM-based extension for 2016–2018 |
| Relation to the training climate | Same source. `prepare_data.py` step 2 (`src/data/develop_climate.py`) area-weights these same files into `data/training/climate/watersheds.zarr`; recomputing all 233 basins from the store reproduces the zarr bit for bit. The zarr applies **neither** correction below. |
| Input identity | `input_tree_sha256` (file attribute and provenance): sha256 over `"".join(f"{filename} {sha256_hex}\n")` of the store files, sorted by file name |

References:

- Livneh, B., et al. (2013). A long-term hydrologically based dataset of land surface fluxes and states for the conterminous United States: Update and extensions. *J. Climate* 26, 9384–9392.
- Pierce, D. W., et al. (2021). An extreme-preserving long-term gridded daily precipitation dataset for the conterminous United States. *J. Hydrometeorology* 22(7), 1883–1895.
- PRISM Climate Group, Oregon State University.
- California Department of Water Resources. Gridded Weather Generator Perturbations of Historical Detrended and Stochastically Generated Temperature and Precipitation for the State of CA and HUC8s. data.ca.gov (link above).

The forcing files assert no licence of their own: the `license` attribute says the terms of the source datasets apply.

### AlphaEarth

| Item | Source |
|---|---|
| Collection | `GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL`, calendar year 2017 (and 2017–2025 for the mean), bands A00–A63; catalog page <https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_SATELLITE_EMBEDDING_V1_ANNUAL> |
| Versions | Read from the tiles used and recorded in the files and the provenance (`DATASET_VERSION` 1.1 for 2017 as of 2026-09; per year for the mean). Assembly aborts if the tiles used in one year mix dataset versions, and the burn refuses to start a year whose inventory already does. |
| Inventory | `aef_parts/inventory_<year>.json` lists the tiles over the state with their versions. 2017's also holds the earliest-year check (no image before 2017-01-01); every other year's holds a check that the year has images. The inventories stay on the machine that ran the burn (gitignored); the tiles actually used and the check results are also in the file attributes and `provenance.toml`. |
| Reference | Brown, C. F., et al. (2025). AlphaEarth Foundations: An embedding field model for accurate and efficient global mapping from sparse label data. arXiv:2507.22291 |
| Licence | CC-BY-4.0 (<https://creativecommons.org/licenses/by/4.0/>) |

## Corrections (Forcing)

Only two things are changed. Every other value is the store's ASCII value cast to float32.

### 1. x10 misplaced-decimal summer precipitation

The rule is DWR's upstream correction, applied statewide:

```text
month in (6, 7, 8) and raw precip >= 150.0 mm  ->  precip / 10
```

The divide is done in float64 on the parsed value, then cast to float32.

| Statistic | Value |
|---|---|
| Corrected cell-days | 620 |
| Cells | 445 |
| Dates | 44 (1915-06-01 … 2004-06-06) |
| Raw values corrected | 150.041 … 856.039 mm |
| Largest June–August value left alone | 149.904 mm |
| Busiest dates | 1974-07-08/09/10 (275 cell-days), 1933-06-08/09 (52), 1983-08-30/31 (48), 1947-07-26/27 (44), 1937-06-19/20 (25) |

How it was validated:

- `prepare_gridded.py check-x10` gates the rule statewide against DWR WGEN Product A, which applies the same correction upstream and stores `round(raw/10, 2)`: `|PA - expected| ≤ 0.0051 mm` on every day, with no rule day that Product A left raw and no Product A correction that the rule missed. Product A's temperatures are detrended and are ignored. The full statewide run (2026-09-28, all 13,786 cells, 164 s) matched 620 rule days to 620 Product A corrections exactly.
- That run found **one Product A edit that is not the x10 rule**: 37.40625_-122.34375 on 1974-07-08 (San Mateo coast, the 1974-07-08 artefact storm) is 29.037 mm raw but 24.58 mm in Product A. The ratio is 1.18, and every neighbouring cell matches the raw store. This product keeps the raw value; the case is listed in `forcing.PA_KNOWN_NON_RULE` and reported, not failed, by `check-x10`.

Caveats:

- **The rule is a correction convention, not a measurement.** It cannot tell a decimal artefact from a real storm. For example, 1977-08-16/17 (12 cell-days across Southern California, 151–319 mm, the largest near Yuma) plausibly coincides with the remnants of Tropical Storm Doreen and may be real rain. It has not been checked against station records, and the rule divides it anyway.
- **It leaves a discontinuity on the artefact dates.** Values of 100–150 mm on those dates stay raw, so corrected cells sit next to uncorrected neighbours. For example, 149.85 mm at 39.09375_-120.65625 on 1974-07-09 is kept while its neighbours are divided down to 15–85 mm.
- **Non-summer extremes are never touched**, e.g. 814.6 mm at 37.59375_-118.21875 on 1967-01-25 and 739.8 mm at 34.09375_-116.84375 on 1938-03-03.
- **The current training climate is uncorrected.** `data/training/climate/watersheds.zarr` (and the HUC climate from the same step) still carries the spikes. For example, basin 11315000 reads 174.56 mm on 1974-07-09 where the corrected value is 17.46 mm. On the x10 dates the gridded forcing and the watershed climate therefore disagree.

### 2. Inverted temperatures

Where `tmin > tmax`, the two values are swapped (`tmin = min`, `tmax = max`), which leaves the daily mean unchanged.

| Statistic | Value |
|---|---|
| Swapped cell-days | ~181,660 (0.035 %) |
| Cells with at least one swap | ~6,008 |
| Worst cell | 38.15625_-122.96875: 2,327 days (6.1 %); coastal cells dominate |
| Days with `tmin == tmax` | ~1,893 (left as is) |

The counts above come from a statewide scan of the store. The build records its own exact counts in `swap_count`, the `corrections` attribute and `provenance.toml` (`[forcing.temp_swap]`). The watershed climate in `watersheds.zarr` averages the raw, unswapped values.

## Known Caveats

**Temperature source seam at 2016-01-01.** Temperature is Livneh for 1915–2015 and a PRISM-based extension for 2016–2018 (WGEN README: "Livneh temperature (1915-2015) corrected to PRISM observations (2016-2018)"). The inversion rate shows the seam: on a 2,392-cell sample band it runs about 500–2,800 ppm per year through 2015, then 0 ppm in 2016, 5.7 in 2017 and 103 in 2018. Treat 2016–2018 temperature as a different source. The before/after counts are in `temperature_source_note` and in `[forcing.temp_swap]` (`n_cell_days_before_seam`, `n_cell_days_from_seam`).

**x10 rule.** See the caveats under [Corrections](#1-x10-misplaced-decimal-summer-precipitation).

**AlphaEarth is 2017 conditions.** The embedding describes the land surface as observed in 2017 (land cover, water extent such as the Salton Sea, burn scars, development) and is static over the whole 1915–2018 forcing record. The 2017–2025 mean averages nine years instead, which smooths year-specific states (a wet or dry year's water extent, a fire scar) but describes years that mostly lie **after** the forcing record ends in 2018. `year_cos_min` shows where the years disagree.

**Coastal and partial coverage.** Where part of a cell rectangle has no valid pixels (e.g. open ocean), the embedding is the mean over the valid pixels only. `valid_frac` < 1 and `aef_flag` bit 1 are set; bit 4 is added when `valid_frac` < 0.5, where the 15 m lattice subsampling error grows. A cell with no valid pixels at all has a NaN embedding and `valid_frac` 0, so it carries all three bits (`aef_flag = 7`). In the mean, a year without valid pixels is left out of that cell's average (`n_years` < 9, `aef_flag` bit 8). That year's `valid_frac` is 0, so bits 1 and 4 are set as well (`aef_flag = 13` for an otherwise fully covered cell). A cell with no valid pixels in any year is NaN (`aef_flag = 15`). Lakes, reservoirs and bays are embedded rather than masked (`valid_frac` 1 on every Clear Lake and Goose Lake cell).

**Embeddings are not unit length.** Each pixel vector is unit length; their cell mean is not. `norm` < 1 measures within-cell heterogeneity (0.67–0.999, median 0.86 over the 13,786 cells in 2017), and in the multi-year mean also between-year heterogeneity. Do not renormalise without keeping `norm`, and do not dequantise again (Earth Engine already did).

**Zone 12N.** Only three cells lie east of -114° (34.09375_-113.96875, 34.15625_-113.96875, 34.15625_-113.90625). Three cells would rarely turn up in a random sample, so `aef --check` always samples them, and `aef --assemble` warns if any 12N or zone-edge cell has incomplete coverage.

**Attribution (CC-BY 4.0).** Any redistribution of `alphaearth_2017.nc`, `alphaearth_2017-2025_mean.nc` or products derived from them must keep this attribution, which both files carry verbatim in their `attribution` attribute:

> The AlphaEarth Foundations Satellite Embedding dataset is produced by Google and Google DeepMind.

The licensed material is the Earth Engine collection at <https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_SATELLITE_EMBEDDING_V1_ANNUAL>. The files' `source` attribute (and the mean's `source_url` attribute) and `source_url` in the `[alphaearth]` and `[alphaearth_mean]` tables of `provenance.toml` carry the same link, and `license` names CC-BY-4.0 with its URL.

The modification statement required by CC-BY 4.0 §3(a)(1)(B) is each file's `modifications` attribute. For the default 15 m build of the 2017 file it reads:

> Modified from the original (CC-BY-4.0 section 3(a)(1)(B)): the 10 m pixel embeddings were aggregated to 1/16-degree cell means over each cell rectangle; each cell was reduced on its own UTM zone's tiles in native UTM on a 15 m nearest-neighbour lattice of the full-resolution level (4/9 of the 10 m pixels; <= 2e-4 per band, cos >= 0.9999998 vs the full 10 m mean); the cell means are not unit length.

The mean's statement says the same for each calendar year 2017–2025 and adds that "the per-year cell means were then averaged with equal weight over the years in which the cell has valid pixels; neither the per-year nor the multi-year means are unit length."

## Fetching (Git LFS)

`.lfsconfig` sets `fetchexclude = data/gridded/*.nc`, so a clone or a plain `git lfs pull` leaves the NetCDFs (four, five with the AlphaEarth mean) as small text pointer stubs (starting `version https://git-lfs.github.com/spec/v1`). This keeps clones, CI and the Render deploy light. The CSVs are LFS objects too, but they are not excluded and arrive with the normal fetch. `SHA256SUMS`, `provenance.toml` and this README are plain git.

Fetch the NetCDFs from the repo root; `--exclude=""` overrides the configured `fetchexclude`:

```bash
git lfs pull --include="data/gridded/*.nc" --exclude=""

# or a single file
git lfs pull --include="data/gridded/alphaearth_2017.nc" --exclude=""
```

Then check the bytes:

```bash
cd data/gridded && sha256sum -c SHA256SUMS
```

`.gitattributes` keeps `SHA256SUMS` and `provenance.toml` at LF line endings on every checkout. In a Windows checkout made before that rule existed, `core.autocrlf` may have given them CRLF, and `sha256sum -c` then reports every file as unreadable. Use `tr -d '\r' < SHA256SUMS | sha256sum -c` there. `prepare_gridded.py verify` accepts either line ending.

## Rebuilding

Everything runs from the repo root through `scripts/prepare_gridded.py`. Each subcommand takes `--out-dir` (default `data/gridded`); point it at a scratch directory for trial builds so `data/gridded` is never touched. `python scripts/prepare_gridded.py COMMAND --help` lists every option.

### Prerequisites

- **Python environment.** Use the `neuralhyd` env, which has netCDF4 (`environment.yml`). Run the commands from an activated env (`conda activate neuralhyd`), not with a bare `python` from another install. `conda run` also works, but only as `conda run --no-capture-output -n neuralhyd python -s ...`. Without `--no-capture-output` it holds back all output until the process exits, which hides the progress, retry and resume messages of `forcing`, `check-x10` and `aef --run`. On Windows, a numpy in the user site-packages (`%APPDATA%\Python`) can shadow the env's copy. If `python -c "import numpy; print(numpy.__file__)"` points outside the env, add `-s` (ignore user site-packages) to every command.
- **Forcing store.** A local copy of the WGEN NonDetrend-Unsplit statewide store (13,786 `data_<lat>_<lon>` files). It is not in the repo.
- **Product A** (for `check-x10`). A DWR WGEN Product A realisation directory (`.../WGEN/Product_A/1`, files `meteo_<key>`). If it lives in OneDrive as cloud-only placeholders, a full check downloads every file (~15.7 GB).
- **Earth Engine** (for `aef --run` and `aef --check`). `earthengine-api` is in the pip section of `environment.yml`, so an env created from it already has the package. For an older `neuralhyd` env, or any other conda env with netCDF4, numpy and pandas (the `aef` path imports no zarr or torch), run `pip install earthengine-api`. Run `earthengine authenticate` once and pass `--project <ee-project>`, an Earth Engine–registered cloud project; there is no default. Behind the DWR TLS proxy the Windows trust store is injected automatically.
- **Long runs.** Launch the full `forcing` build and the `aef --run` burn from your own terminal.

### Commands, in order

```bash
STORE=/path/to/WGEN_NonDetrend_Unsplit_Statewide
PA=/path/to/WGEN/Product_A/1

# 1. Cell list -> grid_cells.csv + SHA256SUMS
python scripts/prepare_gridded.py grid --meteo-dir "$STORE"

# 2. Forcing NetCDFs, x10 CSV, SHA256SUMS, provenance [forcing]
#    (skips when all three exist and match SHA256SUMS/provenance; --force to
#    regenerate; LFS pointer stubs from a default clone count as missing)
python scripts/prepare_gridded.py forcing --meteo-dir "$STORE"

# 3. Gate the x10 rule against Product A (exit 1 on any mismatch)
python scripts/prepare_gridded.py check-x10 --meteo-dir "$STORE" --product-a-dir "$PA" --out-csv x10_mismatches.csv

# 4. AlphaEarth plan: units, cells, EECU estimate (no Earth Engine calls)
python scripts/prepare_gridded.py aef --dry-run

# 5. The burn: spends Earth Engine quota, resumable
python scripts/prepare_gridded.py aef --run --project <ee-project>
python scripts/prepare_gridded.py aef --status            # progress, any time

# 6. Banked partials -> alphaearth_2017.nc, SHA256SUMS, provenance [alphaearth]
python scripts/prepare_gridded.py aef --assemble

# 7. Independent 10 m re-reduction of 30 stratified cells; gates the banked
#    partials and the assembled NetCDF (exit 1 on failure)
python scripts/prepare_gridded.py aef --check 30 --project <ee-project>

# 8. Optional, offline: compare the bank with an external reference bank in the
#    same format (read only); --compare-mean also compares the assembled mean
python scripts/prepare_gridded.py aef --compare-parts /path/to/reference/aef_parts

# 9. Offline QA of everything; --meteo-dir also re-reads store cells bit for bit
python scripts/prepare_gridded.py verify --meteo-dir "$STORE"
```

The optional 2017–2025 mean adds four steps. They reuse the 2017 partials, so run them after step 5 (in any order relative to steps 6–9):

```bash
# M1. Plan for every year: 2017 is already banked, 2018-2025 are left (no Earth Engine calls)
python scripts/prepare_gridded.py aef --dry-run --years all

# M2. The burn for the other eight years: ~392 EECU-h, resumable, same bank as 2017
python scripts/prepare_gridded.py aef --run --project <ee-project> --years all
python scripts/prepare_gridded.py aef --status            # per-year progress, any time

# M3. Every year's partials -> alphaearth_2017-2025_mean.nc, SHA256SUMS, provenance [alphaearth_mean]
python scripts/prepare_gridded.py aef --assemble-mean

# M4. Independent 10 m check of one year; gates its partials and its layer of the mean
python scripts/prepare_gridded.py aef --check 30 --year 2021 --project <ee-project>
```

Then re-run step 8 with `--compare-mean /path/to/reference_mean.npz` (it compares every year both banks hold and the mean on the overlapping cells) and step 9 (it verifies the mean too).

| Step | Cost | Notes |
|---|---|---|
| `grid` | ~1 s | Refuses to replace a `grid_cells.csv` that disagrees with the store, and refuses to write fewer than 13,786 cells into `data/gridded`. |
| `forcing` | ~3 min (estimated from smoke builds), ~2.5 GB RAM | Works in 16-row bands; the next band is parsed while the current one is compressed and written, and writing dominates. Smoke test: `--rows 96:112 --out-dir <scratch>` (1,144 cells, ~25 s). |
| `check-x10` | ~3 min for all cells (estimate) | `--sample 40` checks every rule cell plus 40 random cells. It still takes ~1.5 min because it parses the whole store first to find the rule cells. |
| `aef --dry-run` | instant | 128 work units (10N 58, 11N 69, 12N 1); ~176,000 EECU-s ≈ **49 EECU-h** at 15 m (~89 at 10 m). |
| `aef --run` | ~49 EECU-h, 15–40 min wall-clock at 8 workers | About a third of the noncommercial Community tier's 150 EECU-h per month. An over-quota project is throttled; the retry path copes, slowly. `--max-units 2` is a smoke test. |
| `aef --assemble` | seconds | Aborts on any cell not banked exactly once, on `valid_frac` > 1 + 1e-6 (double counting), or on mixed dataset versions. |
| `aef --check 30` | ~23.3 EECU-s per cell (~0.2 EECU-h) | Always includes the 3 12N cells and the 3 11N cells at -114.03125; the rest are split across both sides of -120°, partially covered cells and random cells. Gate per cell: max \|d\| < 2e-4 per band and \|\|m\| - \|m10\|\| < 1e-3 on fully covered cells. Partial cells with `valid_frac` ≥ 0.5 get both tolerances × 1/√`valid_frac`, because the 15 m lattice error grows as the valid pixel count shrinks. Cells below 0.5 (`low_coverage`) are reported, not gated. One line per stratum shows its max \|d\|. It reads the NetCDF before any Earth Engine call and gates it too, so run it after `--assemble`. |
| `aef --compare-parts DIR` | seconds | Offline, against an external reference bank in the same format: `DIR/<year>/*.npz` partials with `keys`, `S`, `W`, `imgs` and `scale` (an optional `DIR/run.json` must pin the same scale). Every year both banks hold: PASS when S/W and W agree to 1e-9 on the overlapping cells (identical reductions of the same pixels agree to rounding). With `--compare-mean NPZ`, a reference mean holding `lat`, `lon`, `emb`, `n_years`, `years`, `dataset_version`, `valid_frac`, `year_cos_min` and `norm`, the assembled mean's `embedding` must also equal its `emb` on the overlapping cells within 1e-7 (identical partials give identical float32 values; partials that agree to ~1e-13 can differ by one float32 ulp), with equal `n_years` and the same `DATASET_VERSION` per year; without it only the partials are compared. Never modifies the reference directory. |
| `aef --dry-run --years all` | instant | 8 × 128 units left once 2017 is banked; ~392 EECU-h at 15 m. `--years 2018 2019` plans selected years. |
| `aef --run --years all` | ~392 EECU-h, a few hours at 8 workers | One pool over every (year, unit) job, year by year; banked per year, resumable. Queries each new year's inventory first. |
| `aef --assemble-mean` | ~5 s | Aborts unless all nine years are complete, listing every year with nothing banked and every year with cells missing, banked twice or without an inventory. Also aborts on `valid_frac` > 1 + 1e-6 or mixed dataset versions within a year. |
| `aef --check 30 --year Y` | ~0.2 EECU-h | As `aef --check`, for year Y's partials and Y's layer of the mean (and `alphaearth_2017.nc` when Y is 2017). |
| `verify` | ~1 min; `--full` 3–5 min (estimates) | Exit 1 on any failure. |

The forcing can be verified on its own before spending any Earth Engine quota: `verify --skip-aef --meteo-dir "$STORE"`.

**How the burn stays safe.** Each work unit (one zone's 1-degree block, at most `--chunk` 150 cells; the same plan every year) is banked atomically as `aef_parts/<year>/<unit>.npz` holding its cell keys, S, W, tile counts, tile ids and scale; a re-run skips banked units. `aef_parts/run.json` pins chunk, scale and grid for the whole bank and lists the years runs were started for (`years`). A bank started before the mean existed carries `"year": 2017` instead; it is accepted as is, a 2017 run leaves it untouched, and `--years` adds the `years` list while keeping `year`. `run.lock` stops two runs from spending quota on the same units. Before the first unit, the burn writes `inventory_<year>.json` for each requested year and `expected_15m.npz` (shared by all years). Ctrl+C cancels the queued units; in-flight ones finish and are banked. A cell that no tile reached is a hard error, and its unit is not banked.

**Committing a rebuild.** `.gitattributes` routes `data/gridded/*.nc` and the CSVs through LFS, and `SHA256SUMS` and `provenance.toml` stay in plain git with LF line endings; `aef_parts/` and `*.part` are gitignored. After staging, `git lfs status` should list the NetCDFs (four, five with the AlphaEarth mean) as LFS objects. Every rebuild produces new NetCDF bytes (see [Checksums](#checksums-and-content-hashes)), and GitHub LFS stores every pushed version, about 3 GB more per forcing rebuild. So commit rebuilt NetCDFs only when a `content_sha256` in `provenance.toml` changed or the metadata change is intended; otherwise restore the committed files. Build from a committed tree, so that `history` and `[forcing] git` name a commit that contains the code. `history`, `command` and `input_dir` keep only the last component of each path, so no local user paths are recorded.

## Verifying

`python scripts/prepare_gridded.py verify` runs offline and prints one PASS/FAIL line per check.

| Product | Checks |
|---|---|
| `grid_cells.csv` | Valid lattice keys, 13,786 cells, sha256 matches `SHA256SUMS` |
| Forcing | The files are products, not Git LFS pointer stubs; coordinates, bounds and time axis exactly the lattice; encoding (float32, chunks, zlib 4 + shuffle, NaN fill); `mask` identical across files and equal to `grid_cells.csv`, all 13,786 cells for a full product, and no `--rows` smoke build in `data/gridded`; values finite exactly on the mask, precip ≥ 0, no June–August precip ≥ 150 mm (the x10 postcondition) and tmin ≤ tmax on each 16-row band's first, last and one random day; x10 table sorted, June–August, ≥ 150 mm, equal to `precip_mm` at the pairs, consistent with `x10_count` and the CSV; `swap_count` identical in both temperature files; `SHA256SUMS` and the provenance file hashes |
| Forcing, `--full` | The same value checks on every day of every band, so any June–August value ≥ 150 mm left in `precip_mm` fails, and each `content_sha256` recomputed against `provenance.toml` |
| Forcing, `--meteo-dir` | 64 random cells (`--sample`) plus up to 64 x10 cells re-read from the store through the same parser and corrections; bitwise equal to the files |
| AlphaEarth | Lattice coordinates; `mask` equals `grid_cells.csv`; 64 bands named A00–A63; embedding finite exactly on cells with valid pixels; `valid_frac` in (0, 1 + 1e-6]; `aef_flag` consistent with `valid_frac`; `norm` equals \|embedding\| and is ≤ 1 + 1e-6; `n_tiles` ≥ 1 and `utm_zone` correct in the domain; attribution, licence and year attributes; `content_sha256` and `SHA256SUMS` |
| AlphaEarth mean (when present; its absence is reported, not failed) | Lattice, `mask`, bands and years 2017–2025, one `dataset_version` per year; `embedding_year` finite exactly on the cell-years with `valid_frac` > 0; `valid_frac` in [0, 1 + 1e-6]; `n_years` equals the count of finite years; `embedding` finite exactly where `n_years` > 0 and equal to the mean of `embedding_year` over those years within 1e-6; `norm` equals \|embedding\|; `year_cos_min` recomputed within 1e-5; `aef_flag` consistent with `valid_frac` and `n_years`; `utm_zone`; attribution, licence, `source_url` and `years` attributes; `embedding_year` for 2017 equal to `alphaearth_2017.nc` `embedding` bit for bit (when that file is present); both content hashes and `SHA256SUMS` |

`--skip-forcing` and `--skip-aef` restrict the run to one kind of product.

### Checksums and content hashes

- **`SHA256SUMS`** covers the bytes of every product (`sha256sum -c SHA256SUMS`). The NetCDF bytes change on **every** rebuild, even with identical values and the same libraries: each file embeds `history` (the command and git commit) and `date_created`, and the bytes also depend on the HDF5/netCDF-C build. Only `content_sha256` is stable across rebuilds.
- **`content_sha256`** (in `provenance.toml`) hashes the decoded values and does not depend on the library build:
  - Forcing, per variable: sha256 over the float32 little-endian values in (lat, lon, time) C order over the whole 173 × 168 grid. That is each cell's full daily series in turn, cells in ascending (lat, lon) order, with NaN cells included as the canonical quiet NaN `0x7fc00000`.
  - AlphaEarth: sha256 over the float32 little-endian `embedding` in stored (band, lat, lon) C order over the whole 64 × 173 × 168 grid, NaN included.
  - AlphaEarth mean: `content_sha256` the same over its `embedding`; `content_sha256_embedding_year` the same over `embedding_year` in stored (year, band, lat, lon) C order (9 × 64 × 173 × 168).
- **`provenance.toml`** is written by the build and should not be edited by hand. `[forcing]` records per-file sha256, byte count and content hash, the input directory, `input_tree_sha256`, the x10 and swap counts (including the seam split), chunks, encoding, library versions, git commit and dirty flag, and the command. `[alphaearth]` records the file's sha256, byte count and content hash plus collection, `source_url`, versions, `images_used`, the earliest-year check, scale, chunk, `grid_sha1`, coverage counts (cells, no-valid, partial, low), the `valid_frac` minimum, the `norm` range, attribution and licence. `[alphaearth_mean]` records the same for the mean (both content hashes, `years`, `dataset_version` and `n_images_used` per year, counts of cells with all, some or no years, the `norm` and `year_cos_min` ranges) plus one `[alphaearth_mean.per_year.<year>]` table per year with its versions, `images_used`, inventory counts, the earthengine-api version that queried the inventory, partials and coverage counts.

To recompute the AlphaEarth content hash by hand:

```python
import hashlib, netCDF4, numpy as np

with netCDF4.Dataset("data/gridded/alphaearth_2017.nc") as ds:
    ds.set_auto_mask(False)
    emb = ds["embedding"][:]
print(hashlib.sha256(np.ascontiguousarray(emb, dtype="<f4").tobytes()).hexdigest())
# compare with [alphaearth] content_sha256 in provenance.toml
```

## Reading the Data

```python
import xarray as xr

pr = xr.open_dataset("data/gridded/livneh_precip_mm_daily_1915-2018.nc")
cell = pr.precip_mm.sel(lat=38.90625, lon=-120.78125)   # one cell, 37,986 days
day = pr.precip_mm.sel(time="1974-07-09")               # statewide map for one day

aef = xr.open_dataset("data/gridded/alphaearth_2017.nc")
vec = aef.embedding.sel(lat=38.90625, lon=-120.78125)   # 64 values; see aef_flag / valid_frac

mean = xr.open_dataset("data/gridded/alphaearth_2017-2025_mean.nc")
vec = mean.embedding.sel(lat=38.90625, lon=-120.78125)             # 2017-2025 mean; see n_years
y21 = mean.embedding_year.sel(year=2021, lat=38.90625, lon=-120.78125)
```

xarray and netCDF4 read every day. **GDAL-based readers (QGIS, `gdalinfo`/`gdal_translate`, rasterio, `rioxarray.open_rasterio`, R terra) turn each day of a forcing file into a band and by default stop at 32,768 bands.** That is 2004-09-17: the last 5,218 days are dropped with only a log warning. Set `GDAL_MAX_BAND_COUNT=65536` first, as an environment variable or a GDAL config option (`with rasterio.Env(GDAL_MAX_BAND_COUNT=65536): ...`). The forcing files repeat this in their `comment` attribute. The AlphaEarth files have 64 bands (the mean's `embedding_year` 9 × 64) and are unaffected.
