#!/usr/bin/env python3
"""Re-fetch the OpenFEMA disaster declarations extract.

Input: OpenFEMA DisasterDeclarationsSummaries v2 API.
Output: data/fema_declarations.csv. Other data sources are listed in README.md.
"""
import json, urllib.request
from pathlib import Path
import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "data"

def fetch_fema():
    rows, skip = [], 0
    while True:
        url = ("https://www.fema.gov/api/open/v2/DisasterDeclarationsSummaries"
               f"?$top=1000&$skip={skip}&$select=fipsStateCode,fipsCountyCode,"
               "incidentType,incidentBeginDate,declarationType,disasterNumber&$format=json")
        with urllib.request.urlopen(url, timeout=60) as r:
            batch = json.load(r).get("DisasterDeclarationsSummaries", [])
        rows += batch
        if len(batch) < 1000:
            break
        skip += 1000
    pd.DataFrame(rows).drop_duplicates().to_csv(DATA / "fema_declarations.csv", index=False)
    print("fema rows:", len(rows))

if __name__ == "__main__":
    fetch_fema()
