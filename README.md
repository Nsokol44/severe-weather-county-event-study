# Sudden strikes and slow burns: how county economies respond to severe weather

Nicholas J. Sokol, Department of Geography & Sustainability, University of Tennessee, Knoxville

Code and data for stacked event studies of county personal income, employment and population after major hazard events (tornado, hail/ice, wind, flood, hurricane, drought), 2001-2019. Fast-onset perils are compared with slow-onset drought, and drought is defined two ways: from Storm Events reports and from SPEI-6.

## Contents

- `code/` Python scripts
  - `run_all.py` runs the estimation and writes tables to `results/`
  - `figures.py` draws figures 1-7 from `results/`
  - `estimator.py` stacked event-study estimator used by `run_all.py`
  - `spei_build.py` builds `data/spei_county_year.csv` from SPEIbase
  - `fetch_data.py` re-fetches the FEMA declarations extract
  - `build_panel.py`, `estimate_event_study.py`, `estimate_contrast.py`, `lead_adjusted_contrast.py` build the panel from raw Storm Events and BEA files and run the baseline estimates (outputs archived in `data/baseline_corrected/`)
  - `test_universe.py`, `test_deflator.py` checks
- `data/` processed inputs, with checksums in `data/DATA_MANIFEST.md`
- `results/` tables, figure inputs and figures written by the scripts
- `figures/` figures 1-7 (PDF and PNG)

## Install and run

```bash
pip install -r requirements.txt
cd code
python run_all.py
python figures.py
```

`run_all.py` takes about 2 minutes. Checks: `python test_universe.py` and `python test_deflator.py`.

## Data

All inputs used by `run_all.py` are in `data/`.

| File | Source |
|---|---|
| `panel_county_year.parquet`, `panel_county_year_drought6.parquet`, `shock_events_*.csv` | Built by `code/build_panel.py` from NOAA/NCEI Storm Events (https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/) and BEA Regional Accounts CAINC1 and CAEMP25N (https://apps.bea.gov/regional/zip/CAINC1.zip) |
| `bp18mr25.dbx` | NWS zone-county correlation file, 18 Mar 2025 (https://www.weather.gov/source/gis/Shapefiles/County/bp18mr25.dbx) |
| `county_adjacency2023.txt` | Census Bureau county adjacency, 2023 |
| `cpi_annual.csv` | FRED CPIAUCSL annual means |
| `fema_declarations.csv` | OpenFEMA DisasterDeclarationsSummaries v2 (https://www.fema.gov/api/open/v2/DisasterDeclarationsSummaries), fetched 2026-10-06 |
| `spei_county_year.csv` | Built by `code/spei_build.py` from SPEIbase v2.10 `spei06.nc` (DOI 10.20350/digitalCSIC/16497) |
| `county_boundaries.geojson` | Census cartographic boundary file cb_2023_us_county_500k, simplified; used only for the Figure 1 maps |

The Storm Events files (about 280 MB) and `spei06.nc` are not included. Per-file hashes for Storm Events are in `data/storm_events_download_manifest.csv`. `data/raw/` holds the small fixtures read by `test_deflator.py`.

## Output in results/

- `table2_primary_per_peril.csv` per-peril event studies
- `table3_onset_contrast.csv` slow minus fast contrast
- `table4_rings_stacked.csv` hit and adjacent-county rings
- `table5_fema_split.csv` declared versus undeclared events
- `table6_spei_drought.csv` SPEI-6 drought
- `table_hurricane_sensitivity.csv`, `fema_share_by_peril.csv`, `cohort_counts.json`, `spei_agreement.json`, `ring_distances.json`
- `paths_*.csv`, `onset_group_summary.csv`, `cohort_locations.csv` inputs to `figures.py`
- `baseline_corrected_summary.json` baseline tables computed from `data/baseline_corrected/`
- `run_manifest.json`, `run_log.txt` run record
- `fig*.png`, `fig*.pdf` figures

## Expected results

Average post-event effects in percentage points (standard error in parentheses), from `results/*.csv`. Standard errors are clustered by state (51 clusters).

- Reporting-based drought, income: -0.86 (0.30), 446 cohorts, Benjamini-Hochberg q = 0.065
- Tornado, employment: -0.84 (0.40), 175 cohorts, q = 0.26
- Drought minus fast-onset pool (499 and 446 cohorts), income: -1.11 (0.41) raw, -1.15 (0.76) lead-adjusted
- SPEI-6 drought, 939 cohorts, controls screened on SPEI shocks: income +0.12 (0.26), employment +0.35 (0.46), population +0.07 (0.29)
- SPEI-6 drought, controls screened on reporting shocks only: income +0.09 (0.19), employment -0.07 (0.12)
- Share of same-state controls that are SPEI-shocked in the cohort year under that screening: 84% (`results/spei_agreement.json`)
- FEMA declared minus undeclared, income: +0.81 (0.36)
- Baseline run in `data/baseline_corrected/`: hurricane income -1.07 (0.34); onset contrast -0.87 (0.24) raw and +0.15 (0.31) lead-adjusted; income pre-trend test rejects for 6 of 6 perils

Cohort counts: tornado 175, hail/ice 111, drought 446, flood 334, wind 162, hurricane 51, SPEI-6 drought 939.
