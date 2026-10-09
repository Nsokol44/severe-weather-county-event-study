"""Build the county-year SPEI-6 drought file from SPEIbase spei06.nc.

Input: spei06.nc (path as first argument, default data/spei06.nc) and data/bp18mr25.dbx.
A county is located at the lat/lon of the first NWS zone listed for it.
Output: data/spei_county_year.csv. A shock year has SPEI-6 <= -1.0 in at least 3 months.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"


def county_locations():
    cols = ["state_ab", "zone", "cwa", "zname", "state_zone", "county_name",
            "fips", "tz", "fe_area", "lat", "lon"]
    z = pd.read_csv(DATA / "bp18mr25.dbx", sep="|", dtype=str, header=None,
                    names=cols, engine="python", on_bad_lines="skip")
    z["fips"] = pd.to_numeric(z["fips"], errors="coerce")
    z["lat"] = pd.to_numeric(z["lat"], errors="coerce")
    z["lon"] = pd.to_numeric(z["lon"], errors="coerce")
    z = z.dropna(subset=["fips", "lat", "lon"]).drop_duplicates("fips")
    return z[["fips", "lat", "lon"]].assign(fips=lambda d: d.fips.astype(int))


def main(nc_path):
    loc = county_locations()
    ds = xr.open_dataset(nc_path)
    v = ds["spei"]
    lat = v.lat.values; lon = v.lon.values
    times = pd.to_datetime(v.time.values)
    keep = (times.year >= 1995) & (times.year <= 2023)
    v = v.isel(time=np.where(keep)[0]); times = times[keep]
    li = np.array([np.argmin(np.abs(lat - la)) for la in loc.lat])
    oi = np.array([np.argmin(np.abs(lon - lo)) for lo in loc.lon])
    cell = v.values[:, li, oi]  # time x county
    years = times.year.values
    frames = []
    for y in sorted(set(years)):
        c = cell[years == y]
        frames.append(pd.DataFrame({
            "fips": loc.fips.values, "year": y,
            "spei_months_le_m1": np.nansum(c <= -1.0, axis=0),
            "spei_months_le_m15": np.nansum(c <= -1.5, axis=0),
            "spei_min": np.nanmin(np.where(np.isnan(c), np.inf, c), axis=0)}))
    g = pd.concat(frames, ignore_index=True)
    g.loc[~np.isfinite(g.spei_min), "spei_min"] = np.nan
    g["spei_shock"] = (g.spei_months_le_m1 >= 3).astype(int)
    g.to_csv(DATA / "spei_county_year.csv", index=False)
    print("rows", len(g), "shock county-years 2001-2019:",
          int(g[g.year.between(2001, 2019)].spei_shock.sum()))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else
         HERE.parent / "data" / "spei06.nc")
