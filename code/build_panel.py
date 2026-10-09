#!/usr/bin/env python3
"""Build the county-year panel of outcomes and peril exposure.

Inputs under data/raw/: Storm Events details files, bp18mr25.dbx, BEA CAINC1 and CAEMP25N, cpi_annual.csv.
Outputs: data/panel_county_year.parquet, data/shock_events.csv, data/build_summary.json (suffix added when a tag is given).
Usage: python build_panel.py [drought_min_months [tag]]
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
MAJOR_SHOCK_2019USD = 1_000_000
# months of drought reports needed for a drought shock; optional args: python build_panel.py 6 drought6
DROUGHT_MIN_MONTHS = int(sys.argv[1]) if len(sys.argv) > 1 else 3
BUILD_TAG = sys.argv[2] if len(sys.argv) > 2 else ""

PERIL_MAP = {
    "tornado": "tornado",
    "hurricane (typhoon)": "hurricane",
    "hurricane": "hurricane",
    "tropical storm": "hurricane",
    "tropical depression": "hurricane",
    "storm surge/tide": "hurricane",
    "hail": "hail_ice",
    "ice storm": "hail_ice",
    "winter storm": "hail_ice",
    "winter weather": "hail_ice",
    "blizzard": "hail_ice",
    "heavy snow": "hail_ice",
    "sleet": "hail_ice",
    "drought": "drought",
    "flash flood": "flood",
    "flood": "flood",
    "coastal flood": "flood",
    "lakeshore flood": "flood",
    "debris flow": "flood",
    "high wind": "wind",
    "strong wind": "wind",
    "thunderstorm wind": "wind",
    "thunderstorm winds": "wind",
    "tstm wind": "wind",
    "marine thunderstorm wind": "wind",
    "marine high wind": "wind",
    "marine strong wind": "wind",
    "marine hail": "hail_ice",
}
PERILS = ["tornado", "hurricane", "hail_ice", "drought", "flood", "wind"]
FAST_PERILS = [p for p in PERILS if p != "drought"]

DMG_RE = re.compile(r"^\s*([0-9]*\.?[0-9]+)\s*([KMB]?)\s*$", re.I)
DMG_MULT = {"": 1.0, "K": 1e3, "M": 1e6, "B": 1e9}


def parse_damage(s: pd.Series) -> pd.Series:
    s = s.fillna("").astype(str).str.strip()
    m = s.str.extract(DMG_RE)
    num, suf = m[0], m[1].fillna("").str.upper()
    out = pd.to_numeric(num, errors="coerce").fillna(0.0) * suf.map(DMG_MULT).fillna(1.0)
    plain = pd.to_numeric(s, errors="coerce")
    return out.where(num.notna(), plain.fillna(0.0))


def load_zone_map() -> tuple[pd.DataFrame, dict[int, str]]:
    """Return (zone-to-county rows, state FIPS to abbreviation) from bp18mr25.dbx."""
    cols = ["state_ab", "zone", "cwa", "zname", "state_zone", "county_name",
            "fips", "tz", "fe_area", "lat", "lon"]
    z = pd.read_csv(DATA / "raw" / "bp18mr25.dbx", sep="|", dtype=str,
                    header=None, names=cols, engine="python", on_bad_lines="skip")
    z["fips"] = pd.to_numeric(z["fips"], errors="coerce")
    z = z[z["fips"].notna()].copy()
    z["fips"] = z["fips"].astype(int)
    z["state_fips"] = z["fips"] // 1000
    z["zone3"] = z["zone"].astype(str).str.strip().str.zfill(3)
    smap = (z.drop_duplicates(["state_fips", "state_ab"])
              .set_index("state_fips")["state_ab"].to_dict())
    return z[["state_ab", "zone3", "fips"]].drop_duplicates(), smap


def load_storm_events() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    files = sorted((DATA / "raw" / "storm_events").glob("StormEvents_details-*.csv.gz"))
    assert files, "no storm event files found - run download_storm_events.py"
    usecols = ["BEGIN_YEARMONTH", "EVENT_ID", "STATE_FIPS", "CZ_TYPE", "CZ_FIPS",
               "EVENT_TYPE", "DAMAGE_PROPERTY", "DAMAGE_CROPS"]
    county_frames, zone_frames = [], []
    audit = {"files": len(files), "rows_total": 0, "rows_by_peril_type": {},
             "unmapped_types": {}}
    for fp in files:
        df = pd.read_csv(fp, usecols=lambda c: c in usecols, dtype=str,
                         compression="gzip", low_memory=False)
        audit["rows_total"] += len(df)
        df["year"] = df["BEGIN_YEARMONTH"].astype(str).str[:4].astype(int)
        df["month"] = df["BEGIN_YEARMONTH"].astype(str).str[4:6].astype(int)
        df["peril"] = df["EVENT_TYPE"].astype(str).str.strip().str.lower().map(PERIL_MAP)
        unmapped = df[df["peril"].isna()]["EVENT_TYPE"].value_counts()
        for k, v in unmapped.items():
            audit["unmapped_types"][k] = audit["unmapped_types"].get(k, 0) + int(v)
        tab = (df.dropna(subset=["peril"])
                 .groupby(["peril", "CZ_TYPE"]).size())
        for (p, t), n in tab.items():
            audit["rows_by_peril_type"][f"{p}|{t}"] = \
                audit["rows_by_peril_type"].get(f"{p}|{t}", 0) + int(n)
        df = df[df["peril"].notna()].copy()
        df["damage_nom"] = parse_damage(df["DAMAGE_PROPERTY"]) + parse_damage(df["DAMAGE_CROPS"])
        df["sfips"] = pd.to_numeric(df["STATE_FIPS"], errors="coerce")
        df["czfips"] = pd.to_numeric(df["CZ_FIPS"], errors="coerce")
        c = df[df["CZ_TYPE"].str.upper() == "C"].copy()
        c["fips"] = (c["sfips"] * 1000 + c["czfips"])
        c = c[c["fips"].notna()]
        county_frames.append(c[["year", "month", "fips", "peril", "damage_nom"]]
                             .assign(fips=lambda d: d["fips"].astype(int)))
        zn = df[df["CZ_TYPE"].str.upper() == "Z"].copy()
        zn = zn[zn["sfips"].notna() & zn["czfips"].notna()]
        zone_frames.append(zn[["year", "month", "sfips", "czfips", "peril", "damage_nom"]]
                           .assign(sfips=lambda d: d["sfips"].astype(int),
                                   czfips=lambda d: d["czfips"].astype(int)))
    return (pd.concat(county_frames, ignore_index=True),
            pd.concat(zone_frames, ignore_index=True), audit)


def load_bea() -> pd.DataFrame:
    def melt(path: Path, linecode: int, value_name: str) -> pd.DataFrame:
        df = pd.read_csv(path, dtype=str, low_memory=False, encoding="latin-1")
        df = df[pd.to_numeric(df["LineCode"], errors="coerce") == linecode]
        g = df["GeoFIPS"].str.strip().str.strip('"')
        df = df[g.str.fullmatch(r"\d{5}")]
        df["fips"] = g[g.str.fullmatch(r"\d{5}")].astype(int)
        year_cols = [c for c in df.columns if re.fullmatch(r"(19|20)\d{2}", str(c))]
        long = df.melt(id_vars=["fips"], value_vars=year_cols,
                       var_name="year", value_name=value_name)
        long["year"] = long["year"].astype(int)
        long[value_name] = pd.to_numeric(
            long[value_name].astype(str).str.strip(), errors="coerce")
        return long

    raw = DATA / "raw" / "bea"
    inc = melt(raw / "CAINC1__ALL_AREAS_1969_2024.csv", 1, "persinc_kusd")
    pop = melt(raw / "CAINC1__ALL_AREAS_1969_2024.csv", 2, "population")
    pci = melt(raw / "CAINC1__ALL_AREAS_1969_2024.csv", 3, "pc_income_usd")
    emp = melt(raw / "CAEMP25N__ALL_AREAS_2001_2022.csv", 10, "employment")
    out = inc.merge(pop, on=["fips", "year"], how="outer") \
             .merge(pci, on=["fips", "year"], how="outer") \
             .merge(emp, on=["fips", "year"], how="outer")
    out = out[(out["fips"] % 1000 != 0) & (out["fips"] < 80000)]
    return out


def main() -> None:
    cpi = pd.read_csv(DATA / "raw" / "cpi_annual.csv")
    cpi19 = float(cpi.loc[cpi["year"] == 2019, "cpi"].iloc[0])
    deflator = dict(zip(cpi["year"], cpi19 / cpi["cpi"]))  # nominal $ -> 2019$

    zone_map, smap = load_zone_map()
    cev, zev, audit = load_storm_events()
    for d in (cev, zev):
        d["damage_2019"] = d["damage_nom"] * d["year"].map(deflator)

    # county-reported exposure
    c_agg = (cev.groupby(["fips", "year", "peril"], as_index=False)
                .agg(dmg_county=("damage_2019", "sum"),
                     n_events_county=("damage_2019", "size"),
                     months_county=("month", "nunique")))

    # zone-reported exposure, mapped to counties
    zev["state_ab"] = zev["sfips"].map(smap)
    zev["zone3"] = zev["czfips"].astype(str).str.zfill(3)
    z_agg = (zev.groupby(["state_ab", "zone3", "year", "peril"], as_index=False)
                .agg(dmg_zone=("damage_2019", "sum"),
                     n_events_zone=("damage_2019", "size"),
                     months_zone=("month", "nunique")))
    zone_keys = set(map(tuple, zone_map[["state_ab", "zone3"]]
                        .drop_duplicates().itertuples(index=False, name=None)))
    audit["zone_groups_total"] = int(len(z_agg))
    audit["zone_groups_matched"] = int(sum(
        (a, z) in zone_keys
        for a, z in z_agg[["state_ab", "zone3"]].itertuples(index=False, name=None)))
    z_mapped = z_agg.merge(zone_map, on=["state_ab", "zone3"], how="left")
    # county in several zones: maximum zone value
    z_county = (z_mapped.dropna(subset=["fips"])
                        .groupby(["fips", "year", "peril"], as_index=False)
                        .agg(dmg_zone=("dmg_zone", "max"),
                             n_events_zone=("n_events_zone", "max"),
                             months_zone=("months_zone", "max")))
    z_county["fips"] = z_county["fips"].astype(int)

    # exposure is the maximum of the two channels
    exp = c_agg.merge(z_county, on=["fips", "year", "peril"], how="outer")
    for c in ["dmg_county", "dmg_zone", "n_events_county", "n_events_zone",
              "months_county", "months_zone"]:
        exp[c] = exp[c].fillna(0)
    exp["damage_eff"] = exp[["dmg_county", "dmg_zone"]].max(axis=1)
    exp["months_eff"] = exp[["months_county", "months_zone"]].max(axis=1)
    # shock: damage >= MAJOR_SHOCK_2019USD, or months >= DROUGHT_MIN_MONTHS for drought
    exp["major"] = np.where(
        exp["peril"] == "drought",
        (exp["months_eff"] >= DROUGHT_MIN_MONTHS).astype(int),
        (exp["damage_eff"] >= MAJOR_SHOCK_2019USD).astype(int))
    shocks = exp[exp["major"] == 1].copy()
    suffix = f"_{BUILD_TAG}" if BUILD_TAG else ""
    shocks.to_csv(DATA / f"shock_events{suffix}.csv", index=False)

    wide = exp.pivot_table(index=["fips", "year"], columns="peril",
                           values=["damage_eff", "major", "months_eff"], fill_value=0)
    wide.columns = [f"{v}_{p}" for v, p in wide.columns]
    wide = wide.reset_index()

    bea = load_bea()
    panel = bea.merge(wide, on=["fips", "year"], how="left")
    exp_cols = [c for c in panel.columns
                if any(c.endswith(f"_{p}") for p in PERILS)]
    panel[exp_cols] = panel[exp_cols].fillna(0)
    panel["ln_pc_income"] = np.log(panel["pc_income_usd"])
    panel["ln_employment"] = np.log(panel["employment"])
    panel["ln_population"] = np.log(panel["population"])
    panel = panel.sort_values(["fips", "year"]).reset_index(drop=True)
    panel.to_parquet(DATA / f"panel_county_year{suffix}.parquet", index=False)

    inwin = shocks[(shocks.year >= 2001) & (shocks.year <= 2019)]
    summary = {
        "storm_audit": {k: v for k, v in audit.items()},
        "n_counties_bea": int(bea["fips"].nunique()),
        "panel_rows": int(len(panel)),
        "panel_years": [int(panel["year"].min()), int(panel["year"].max())],
        "shock_records_by_peril_all_years": shocks["peril"].value_counts().to_dict(),
        "shock_records_by_peril_2001_2019": inwin["peril"].value_counts().to_dict(),
        "shock_counties_by_peril_2001_2019": {
            p: int(inwin.loc[inwin.peril == p, "fips"].nunique()) for p in PERILS},
        "shock_share_of_bea_counties_2001_2019": {
            p: int(inwin.loc[inwin.peril == p, "fips"].isin(set(bea.fips.unique())).sum())
            for p in PERILS},
        "zone_map_records": int(len(zone_map)),
        "counties_per_zone_median": float(
            zone_map.groupby(["state_ab", "zone3"])["fips"].nunique().median()),
    }
    with open(DATA / f"build_summary{suffix}.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
