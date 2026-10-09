# data/raw

Inputs read by `code/test_deflator.py`.

- `cpi_annual.csv` is an identical copy of `data/cpi_annual.csv` (FRED CPIAUCSL annual means), placed at the path the test reads.
- `storm_events/StormEvents_details-ftp_v1.0_d2008_c20260323.csv.gz` is a one-record extract of the 2008 Storm Events details file (its SHA-256 is in `data/storm_events_download_manifest.csv`). The record is EVENT_ID 85070, an Autauga County, AL tornado on 17 Feb 2008, with DAMAGE_PROPERTY 10.00M and DAMAGE_CROPS 0.00K. All original columns are unchanged. The test uses only this record. The full Storm Events files are not included.
