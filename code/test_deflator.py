#!/usr/bin/env python3
"""Test the CPI deflator in build_panel.py (nominal dollars to 2019 dollars).

Inputs: data/raw/cpi_annual.csv and the 2008 Storm Events extract in data/raw/storm_events/.
Checks deflator factors, the Autauga County AL 2008 tornado record (EVENT_ID 85070) and the deflator expression.
Usage: python code/test_deflator.py (exit status 0 on pass)
"""
import importlib.util
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"

spec = importlib.util.spec_from_file_location("build_panel", HERE / "build_panel.py")
bp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bp)


def factor(cpi: pd.DataFrame, year: int) -> float:
    """Factor converting nominal dollars of `year` to 2019 dollars."""
    cpi19 = float(cpi.loc[cpi["year"] == 2019, "cpi"].iloc[0])
    return cpi19 / float(cpi.loc[cpi["year"] == year, "cpi"].iloc[0])


def main() -> int:
    cpi = pd.read_csv(DATA / "raw" / "cpi_annual.csv")
    failures = []

    # deflator factors by year
    expected = {1996: 1.6298, 2008: 1.1877, 2019: 1.0000, 2023: 0.8390}
    for yr, exp in expected.items():
        got = factor(cpi, yr)
        ok = abs(got - exp) < 5e-4
        print(f"factor({yr}) = {got:.4f}  (expected ~{exp})  {'OK' if ok else 'FAIL'}")
        if not ok:
            failures.append(f"factor {yr}")
    for yr in cpi["year"]:
        f = factor(cpi, int(yr))
        c = float(cpi.loc[cpi["year"] == yr, "cpi"].iloc[0])
        c19 = float(cpi.loc[cpi["year"] == 2019, "cpi"].iloc[0])
        if (c < c19 and not f > 1) or (c > c19 and not f < 1):
            failures.append(f"direction {yr}")

    # Autauga Co. AL tornado, EVENT_ID 85070, DAMAGE_PROPERTY 10.00M, DAMAGE_CROPS 0.00K
    raw08 = sorted((DATA / "raw" / "storm_events").glob("StormEvents_details-*_d2008_*.csv.gz"))[0]
    df = pd.read_csv(raw08, dtype=str, compression="gzip", low_memory=False,
                     usecols=["EVENT_ID", "DAMAGE_PROPERTY", "DAMAGE_CROPS"])
    row = df[df["EVENT_ID"] == "85070"].iloc[0]
    nom = float(bp.parse_damage(pd.Series([row["DAMAGE_PROPERTY"]])).iloc[0]) \
        + float(bp.parse_damage(pd.Series([row["DAMAGE_CROPS"]])).iloc[0])
    corrected = nom * factor(cpi, 2008)
    buggy = nom * (float(cpi.loc[cpi["year"] == 2008, "cpi"].iloc[0])
                   / float(cpi.loc[cpi["year"] == 2019, "cpi"].iloc[0]))
    print(f"Autauga 2008 tornado (EVENT_ID 85070): nominal ${nom/1e6:.2f}M -> "
          f"corrected ${corrected/1e6:.2f}M (expect ~$11.87M); "
          f"old inverted formula would give ${buggy/1e6:.2f}M (was $8.42M)")
    if nom != 10_000_000:
        failures.append(f"nominal parse {nom}")
    if abs(corrected - 11_870_000) > 20_000:
        failures.append(f"corrected value {corrected}")
    if abs(buggy - 8_420_000) > 20_000:
        failures.append(f"buggy reference {buggy}")

    # deflator expression in build_panel.py
    src = (HERE / "build_panel.py").read_text()
    if "cpi19 / cpi[\"cpi\"]" not in src or "cpi[\"cpi\"] / cpi19" in src:
        failures.append("build_panel.py deflator expression")

    if failures:
        print("FAIL:", failures)
        return 1
    print("PASS: deflator converts nominal dollars to 2019 dollars correctly.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
