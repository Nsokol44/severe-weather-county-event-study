#!/usr/bin/env python3
"""Check that the estimation universe keeps every county in the source panel except the nine Connecticut planning-region codes.

Input: data/panel_county_year.parquet. Output: printed checks, exit status 0 on pass.
Usage: python code/test_universe.py
"""
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
sys.path.insert(0, str(HERE))
import estimator as E

EXPECTED_STATE_COUNTS = {53: 39, 54: 55, 55: 73, 56: 23}  # WA, WV, WI, WY


def main() -> int:
    src = pd.read_parquet(DATA / "panel_county_year.parquet")
    uni = E.universe(src)
    src_fips = set(src.fips.unique())
    uni_fips = set(uni.fips.unique())
    failures = []

    # state/DC prefixes in source and universe
    src_pref = {int(f) // 1000 for f in src_fips}
    uni_pref = {int(f) // 1000 for f in uni_fips}
    print(f"state/DC prefixes: source {len(src_pref)}, universe {len(uni_pref)}")
    if src_pref != uni_pref or len(uni_pref) != 51:
        failures.append(f"prefix mismatch: missing {sorted(src_pref - uni_pref)}")

    # per-state county sets against the source panel
    pseudo = set(E.INGESTION_PSEUDO_FIPS)
    for pref in sorted(src_pref):
        s = {f for f in src_fips if int(f) // 1000 == pref} - pseudo
        u = {f for f in uni_fips if int(f) // 1000 == pref}
        if s != u:
            failures.append(f"state {pref}: universe differs from source "
                            f"(missing {sorted(s - u)}, extra {sorted(u - s)})")
    for pref, exp in EXPECTED_STATE_COUNTS.items():
        got = sum(1 for f in uni_fips if int(f) // 1000 == pref)
        print(f"state {pref}: {got} counties in universe (expected {exp})")
        if got != exp:
            failures.append(f"state {pref} count {got} != {exp}")

    # dropped codes
    dropped = src_fips - uni_fips
    print(f"dropped codes: {sorted(int(f) for f in dropped)}")
    if dropped != set(E.INGESTION_PSEUDO_FIPS):
        failures.append(f"dropped set {sorted(dropped)} != INGESTION_PSEUDO_FIPS")
    if any(int(f) >= 51900 for f in dropped):
        failures.append("a FIPS >= 51900 code is still being dropped")

    print(f"universe counties: {len(uni_fips)} (source panel {len(src_fips)})")
    if failures:
        print("FAIL:", failures)
        return 1
    print("PASS: universe contains every real county in the source panel; "
          "only the nine documented ingestion pseudo-codes are excluded.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
