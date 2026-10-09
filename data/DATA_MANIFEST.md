# Data manifest

The processed inputs read by `code/run_all.py` are in this directory. Source URLs are also listed in the top-level `README.md`, and `code/fetch_data.py` re-fetches the FEMA extract.

| File | SHA-256 | Source and how to rebuild |
|---|---|---|
| panel_county_year.parquet | ca9905c3e032dc5c6a04a79087ff97a9c199bce3d5d6bb59804c9aa124c47aba | County-year panel built by `code/build_panel.py` from NOAA/NCEI Storm Events and BEA regional accounts |
| panel_county_year_drought6.parquet | ba6cb4651c02f3fd8003eb03dcdfce049b57d7ca57bee4a0f3178b68b5404b6b | Same build, drought shock defined as at least 6 months of drought reports (`python build_panel.py 6 drought6`) |
| shock_events_corrected.csv | b09e500dce3da47127037f4f9ca0f7da7349b1e8bb06fc108c8a7bf426b0cff5 | Shock events from the `build_panel.py` run that produced the panel above |
| shock_events_drought6.csv | f5de3094d8f24e017a5fed851b8200202cdbb9a295644cbbcda7c4f6f3425103 | Shock events from the drought6 run |
| cpi_annual.csv | ba4010e7c2e5880082a2cd7f46cefd9bca17b66bad16e279ed394d39a06e2856 | FRED CPIAUCSL annual means |
| fema_declarations.csv | 72b678ecc141b75539fdbc99bd75fea8628aa367a4264e4e9c8be21cbe1e6e18 | OpenFEMA DisasterDeclarationsSummaries v2 API, fetched 2026-10-06; re-fetch with `code/fetch_data.py` |
| spei_county_year.csv | 31e38767da081a17d191b3b71b6a377f0107e692dae22d995d5a9b177f6e6eb7 | Built by `code/spei_build.py` from SPEIbase v2.10 `spei06.nc` (379,643,921 bytes, SHA-256 91f47f21444d83913558dd7f6a637820af4b2ee158a1a5ecaddaca694500c174; DOI 10.20350/digitalCSIC/16497) |
| county_adjacency2023.txt | df62fedd7d27528af7fd870c7ec92232fb2d6bea3ec6cbc9928946b78e128f64 | Census Bureau county adjacency file, 2023 |
| county_boundaries.geojson | c2b2defaf1d61099bb1eef19630a5e36e56d2568f1412cb4d72ec875f68c7abd | Census cartographic boundary file cb_2023_us_county_500k (source zip SHA-256 99d6597b1fc7767deef62e01d28d8b5dcbd578e151855f7dc0d173cbf5bf0868), GEOID, NAME and STATEFP plus geometry simplified at 0.008 degrees; used only by `code/figures.py` for Figure 1 (Albers Equal Area, EPSG:5070) |
| bp18mr25.dbx | 721c00ea7a609d5f06ea16f5c8e588d1ab9a928e58d345be522781c957484f6d | NWS zone-county correlation file, 18 Mar 2025 |
| storm_events_download_manifest.csv | 667b415fe75a44890d2eb82af1dd25a64c38a9ddaef88651c995df633dfdbff6 | File size and SHA-256 for each of the 29 Storm Events bulk files |
| baseline_corrected/ | per-file hashes not listed | Baseline estimation outputs (event-study, contrast and descriptive tables) from the `build_panel.py` and `estimate_*.py` scripts; read by `run_all.py` stage 0 |

The Storm Events files (about 280 MB) are not included. To rebuild the panel from raw files, download them as listed in `storm_events_download_manifest.csv` and run `code/build_panel.py`.

## data/raw/

Small fixtures read by `code/test_deflator.py` (see `data/raw/README.md`).

| File | SHA-256 | Source |
|---|---|---|
| raw/cpi_annual.csv | ba4010e7c2e5880082a2cd7f46cefd9bca17b66bad16e279ed394d39a06e2856 | Identical copy of `data/cpi_annual.csv` |
| raw/storm_events/StormEvents_details-ftp_v1.0_d2008_c20260323.csv.gz | 1952b298a2ce292044f11a68f94aa5f20e431bf725586a342a1232d06ea42ade | One-record extract (EVENT_ID 85070, Autauga County AL tornado) of the 2008 Storm Events details file listed in `storm_events_download_manifest.csv`, all columns unchanged |

## Estimation universe

`estimator.universe()` keeps every county and county-equivalent in the panel (3,149) except nine ingestion pseudo-codes, the Connecticut planning-region codes 9110-9190, which have no outcome or exposure data in 2001-2019. This leaves 3,140 counties across all 51 state and DC FIPS prefixes. `code/test_universe.py` checks the count.
