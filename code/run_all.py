#!/usr/bin/env python3
"""Run the county event-study pipeline and write the tables in results/.

Inputs: files in data/ (panel, SPEI county-year file, FEMA declarations, adjacency, baseline_corrected/).
Outputs: results/*.csv and *.json, including results/run_manifest.json. Usage: python run_all.py
Stages: 0 baseline tables, 1 per-peril event studies, 2 onset contrasts, 3 hurricane sensitivity,
4 spillover rings, 5 FEMA declared vs undeclared, 6 SPEI-6 drought, 7 figure inputs and manifest.
"""
import json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA, RES = ROOT / "data", ROOT / "results"
sys.path.insert(0, str(HERE))
import estimator as E

RES.mkdir(exist_ok=True)
OUTCOMES = ["ln_pc_income", "ln_employment", "ln_population"]
PERILS = ["tornado", "hurricane", "hail_ice", "drought", "flood", "wind"]
FAST = ["tornado", "hurricane", "hail_ice", "wind"]
SCALE_FRAC = 0.00125
SEED = 20261006
t0 = time.time()
log = lambda *a: print(f"[{time.time()-t0:7.1f}s]", *a, flush=True)

# estimation universe: BEA panel counties except the Connecticut planning-region codes
panel = E.universe(pd.read_parquet(DATA / "panel_county_year.parquet"))
counties = sorted(panel.fips.unique())
cidx = {f: i for i, f in enumerate(counties)}
E.COUNTY_INDEX = cidx
states = np.array([f // 1000 for f in counties])
YEARS = list(range(1995, 2024))
PY = list(range(2001, 2020))

def mat(df, col, years):
    M = np.full((len(counties), len(years)), np.nan)
    sub = df[df.year.isin(years)]
    M[[cidx[f] for f in sub.fips], [years.index(y) for y in sub.year]] = sub[col].to_numpy()
    return M

Y = {o: mat(panel, o, PY) for o in OUTCOMES}
persinc = mat(panel, "persinc_kusd", YEARS) * 1000.0
cpi = pd.read_csv(DATA / "cpi_annual.csv")
cpi19 = float(cpi.loc[cpi.year == 2019, "cpi"].iloc[0])
defl = {int(y): cpi19 / float(cpi.loc[cpi.year == y, "cpi"].iloc[0]) for y in cpi.year}
persinc19 = persinc * np.array([defl[y] for y in YEARS])[None, :]

# shock matrices: damage >= SCALE_FRAC of personal income (2019 dollars); drought if >= 3 months
SH = {}
for p in PERILS:
    if p == "drought":
        months = mat(panel, "months_eff_drought", YEARS)
        SH[p] = np.nan_to_num(months) >= 3
    else:
        dmg = mat(panel, f"damage_eff_{p}", YEARS)
        with np.errstate(invalid="ignore", divide="ignore"):
            share = dmg / persinc19
        SH[p] = np.nan_to_num(share) >= SCALE_FRAC
ANY = np.zeros_like(SH["tornado"])
for p in PERILS:
    ANY |= SH[p]
yi = {y: j for j, y in enumerate(YEARS)}

def cohorts_for(peril, exclusive=True, same_year_only=False, shock_mat=None):
    S = shock_mat if shock_mat is not None else SH[peril]
    rows = []
    for g in range(2004, 2017):
        win = [yi[g + d] for d in range(-3, 4)]
        own = S[:, win].sum(axis=1)
        treated = np.where(S[:, yi[g]] & (own == 1))[0]
        for ti in treated:
            if exclusive:
                if same_year_only:
                    other = any(SH[q][ti, yi[g]] for q in PERILS if q != peril)
                else:
                    other = any(SH[q][ti, win].any() for q in PERILS if q != peril)
                if other:
                    continue
            rows.append((int(ti), g))
    return rows

def control_pool(exclude_mat_by_g):
    return exclude_mat_by_g

pool_by_g = {}
for g in range(2004, 2017):
    win = [yi[g + d] for d in range(-3, 4)]
    pool_by_g[g] = np.where(~ANY[:, win].any(axis=1))[0]

adj_raw = pd.read_csv(DATA / "county_adjacency2023.txt", sep="|", dtype=str)
a = pd.to_numeric(adj_raw.iloc[:, 1], errors="coerce"); b = pd.to_numeric(adj_raw.iloc[:, 3], errors="coerce")
ADJ = {}
for x, y in zip(a, b):
    if pd.notna(x) and pd.notna(y) and x != y:
        ADJ.setdefault(int(x), set()).add(int(y)); ADJ.setdefault(int(y), set()).add(int(x))

def run_design(cohort_rows, outcome, group_of, pool=None):
    coh = pd.DataFrame(cohort_rows, columns=["fips_idx", "cohort"])
    coh["grp"] = [group_of(r) for r in cohort_rows]
    groups = sorted(coh.grp.unique())
    S = E.build_stacks(Y[outcome], np.array(counties), states, coh,
                       pool_by_g if pool is None else pool, ADJ, seed=SEED)
    return E.fit(S, groups), coh

# stage 0: baseline tables from data/baseline_corrected
base = json.loads((DATA / "baseline_corrected" / "estimation_summary.json").read_text())
lead = json.loads((DATA / "baseline_corrected" / "lead_adjusted_contrast_NEWdata.json").read_text())
baseline = dict(stacked=base["stacked"], contrast_raw=base.get("contrast"),
                lead_adjusted_contrast=lead,
                source="baseline tables in data/baseline_corrected")
(RES / "baseline_corrected_summary.json").write_text(json.dumps(baseline, indent=2))
log("stage 0 baseline: cohorts",
    {r["peril"]: r["n_cohorts"] for r in base["stacked"] if r["outcome"] == "ln_pc_income"})

# stage 1: per-peril stacked event studies
counts, rows1, paths1 = {}, [], {}
res_store = {}
cohort_loc_rows = []
for p in [q for q in PERILS if q != "hurricane"]:
    cr = cohorts_for(p)
    counts[p] = len(cr)
    cohort_loc_rows += [dict(spec=p, fips=int(counties[ti]), cohort_year=int(g)) for ti, g in cr]
    for o in OUTCOMES:
        res, _ = run_design(cr, o, lambda r, p=p: p)
        res_store[(p, o)] = res
        sm = E.summarize(res, p)
        hb = E.honest_bounds(res, p, 0.0025); hb10 = E.honest_bounds(res, p, 0.01)
        wp, _se = E.wild_boot_p(res, E._w(res, p, E.POST), seed=SEED)
        row = dict(peril=p, outcome=o, n_cohorts=len(cr), avg_post=sm["avg_post"],
                   se=sm["se"], lead_adj=sm["lead_adj"], lead_adj_se=sm["lead_adj_se"],
                   pretrend_p=sm["pretrend_p"], mde80=sm["mde80"], wild_p=wp,
                   honest_lo_M0025=hb["lo"], honest_hi_M0025=hb["hi"], honest_lo_M01=hb10["lo"], honest_hi_M01=hb10["hi"],
                   breakdown_M=E.breakdown_M(res, p), clusters=sm["clusters"])
        rows1.append(row); paths1[(p, o)] = sm["path"]
        log(f"stage1 {p:9s} {o:16s} n={len(cr):4d} post={sm['avg_post']*100:+.2f}pp (se {sm['se']*100:.2f}) pretrend p={sm['pretrend_p']:.3f}")
tab1 = pd.DataFrame(rows1)
tab1["p_raw"] = 2 * (1 - stats.norm.cdf(np.abs(tab1.avg_post / tab1.se)))
tab1["p_bh"] = E.bh_adjust(tab1.p_raw.values)
tab1.to_csv(RES / "table2_primary_per_peril.csv", index=False)
pd.DataFrame([dict(peril=p, outcome=o, rel_time=k, coef=v[0], se=v[1])
              for (p, o), path in paths1.items() for k, v in path.items()]
             ).to_csv(RES / "paths_primary.csv", index=False)

# stage 2: onset contrasts
contrast_rows = []
contrast_path_rows = []
onset_group_rows = []
for slow_name, slow_perils in [("drought_only", ["drought"]), ("drought_flood", ["drought", "flood"])]:
    cr = []
    for p in FAST:
        cr += [(ti, g, "fast") for ti, g in cohorts_for(p)]
    for p in slow_perils:
        cr += [(ti, g, "slow") for ti, g in cohorts_for(p)]
    coh = pd.DataFrame(cr, columns=["fips_idx", "cohort", "grp"])
    for o in OUTCOMES:
        S = E.build_stacks(Y[o], np.array(counties), states, coh, pool_by_g, ADJ, seed=SEED)
        res = E.fit(S, ["fast", "slow"])
        for gname in ["fast", "slow"]:
            smp = E.summarize(res, gname)
            for k, v in smp["path"].items():
                contrast_path_rows.append(dict(slow_def=slow_name, outcome=o, group=gname, rel_time=k, coef=v[0], se=v[1]))
        for la in (False, True):
            est, se, w = E.contrast(res, "slow", "fast", lead_adjust=la)
            wp, _ = E.wild_boot_p(res, w, seed=SEED) if (o != "ln_population") else (np.nan, np.nan)
            contrast_rows.append(dict(slow_def=slow_name, outcome=o, lead_adjusted=la,
                                      slow_minus_fast=est, se=se, wild_p=wp,
                                      n_fast=int((coh.grp == "fast").sum()),
                                      n_slow=int((coh.grp == "slow").sum())))
            log(f"stage2 {slow_name:14s} {o:16s} lead_adj={la} contrast={est*100:+.2f}pp (se {se*100:.2f})")
        if slow_name == "drought_only" and o in ("ln_pc_income", "ln_employment"):
            for gname in ("fast", "slow"):
                smg = E.summarize(res, gname)
                hb25 = E.honest_bounds(res, gname, 0.0025)
                hb10 = E.honest_bounds(res, gname, 0.01)
                onset_group_rows.append(dict(
                    slow_def=slow_name, outcome=o, group=gname,
                    n_cohorts=int((coh.grp == gname).sum()),
                    avg_post=smg["avg_post"], se=smg["se"],
                    lead_adj=smg["lead_adj"], lead_adj_se=smg["lead_adj_se"],
                    pretrend_p=smg["pretrend_p"], mde80=smg["mde80"],
                    honest_lo_M0025=hb25["lo"], honest_hi_M0025=hb25["hi"],
                    honest_lo_M01=hb10["lo"], honest_hi_M01=hb10["hi"],
                    breakdown_M=E.breakdown_M(res, gname)))
pd.DataFrame(contrast_rows).to_csv(RES / "table3_onset_contrast.csv", index=False)
pd.DataFrame(onset_group_rows).to_csv(RES / "onset_group_summary.csv", index=False)
pd.DataFrame(contrast_path_rows).to_csv(RES / "paths_onset_contrast.csv", index=False)
counts["fast_pool"] = int(sum(counts[p] for p in ["tornado", "hail_ice", "wind"]) + 0)
# pool counts including hurricane
hurr_cr = cohorts_for("hurricane"); counts["hurricane"] = len(hurr_cr)
cohort_loc_rows += [dict(spec="hurricane", fips=int(counties[ti]), cohort_year=int(g)) for ti, g in hurr_cr]
counts["fast_pool"] = counts["tornado"] + counts["hail_ice"] + counts["wind"] + counts["hurricane"]
counts["slow_pool_drought_flood"] = counts["drought"] + counts["flood"]

# stage 3: hurricane sensitivity
hurr_rows = []
for label, cr in [("standalone_exclusive_pm3", hurr_cr),
                  ("same_year_exclusivity", cohorts_for("hurricane", same_year_only=True))]:
    for o in ["ln_pc_income", "ln_employment"]:
        res, _ = run_design(cr, o, lambda r: "hurricane")
        sm = E.summarize(res, "hurricane")
        hurr_rows.append(dict(spec=label, outcome=o, n_cohorts=len(cr),
                              avg_post=sm["avg_post"], se=sm["se"],
                              lead_adj=sm["lead_adj"], lead_adj_se=sm["lead_adj_se"],
                              pretrend_p=sm["pretrend_p"], mde80=sm["mde80"]))
        log(f"stage3 {label:26s} {o:16s} n={len(cr)} post={sm['avg_post']*100:+.2f}pp (se {sm['se']*100:.2f})")
pd.DataFrame(hurr_rows).to_csv(RES / "table_hurricane_sensitivity.csv", index=False)

# stage 4: spillover rings
loc = pd.read_csv(DATA / "bp18mr25.dbx", sep="|", dtype=str, header=None, engine="python", on_bad_lines="skip")
loc.columns = [f"c{i}" for i in range(11)]
loc["fips"] = pd.to_numeric(loc.c6, errors="coerce"); loc["lat"] = pd.to_numeric(loc.c9, errors="coerce"); loc["lon"] = pd.to_numeric(loc.c10, errors="coerce")
loc = loc.dropna(subset=["fips", "lat", "lon"]).drop_duplicates("fips").set_index("fips")
def dist_km(f1, f2):
    if f1 not in loc.index or f2 not in loc.index: return np.nan
    la1, lo1, la2, lo2 = loc.at[f1, "lat"], loc.at[f1, "lon"], loc.at[f2, "lat"], loc.at[f2, "lon"]
    p1, p2 = np.radians(la1), np.radians(la2)
    dp = np.radians(la2 - la1); dl = np.radians(lo2 - lo1)
    h = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(h))
ring_rows, ring_dists = [], []
for pool_name, pool_perils in [("fast", FAST), ("drought", ["drought"])]:
    hits = []
    for p in pool_perils:
        hits += cohorts_for(p)
    hit_set = set(hits)
    ring = {}
    for ti, g in hits:
        hf = counties[ti]
        for nb in ADJ.get(hf, set()):
            if nb not in cidx: continue
            ni = cidx[nb]
            win = [yi[g + d] for d in range(-3, 4)]
            if ANY[ni, win].any(): continue
            ring.setdefault((ni, g), hf)
    ring_dists += [dist_km(counties[ni], hf) for (ni, g), hf in ring.items()]
    cr = [(ti, g, "hit") for ti, g in hits] + [(ni, g, "ring1") for (ni, g) in ring]
    coh = pd.DataFrame(cr, columns=["fips_idx", "cohort", "grp"]).drop_duplicates(["fips_idx", "cohort", "grp"])
    for o in ["ln_pc_income", "ln_employment"]:
        S = E.build_stacks(Y[o], np.array(counties), states, coh, pool_by_g, ADJ, seed=SEED)
        res = E.fit(S, ["hit", "ring1"])
        for gname in ["hit", "ring1"]:
            sm = E.summarize(res, gname)
            ring_rows.append(dict(pool=pool_name, outcome=o, group=gname,
                                  n_units=int((coh.grp == gname).sum()),
                                  avg_post=sm["avg_post"], se=sm["se"],
                                  lead_adj=sm["lead_adj"], lead_adj_se=sm["lead_adj_se"],
                                  pretrend_p=sm["pretrend_p"], mde80=sm["mde80"]))
            log(f"stage4 {pool_name:8s} {o:16s} {gname:6s} post={sm['avg_post']*100:+.2f}pp (se {sm['se']*100:.2f}) MDE={sm['mde80']*100:.2f}pp")
pd.DataFrame(ring_rows).to_csv(RES / "table4_rings_stacked.csv", index=False)
ring_dist = dict(median_km=float(np.nanmedian(ring_dists)), p25_km=float(np.nanpercentile(ring_dists, 25)),
                 p75_km=float(np.nanpercentile(ring_dists, 75)), n_pairs=len(ring_dists))
(RES / "ring_distances.json").write_text(json.dumps(ring_dist, indent=2))

# stage 5: FEMA declared vs undeclared
fema = pd.read_csv(DATA / "fema_declarations.csv")
PERIL_OF_DECL = {"Tornado": ["tornado"], "Hurricane": ["hurricane"], "Tropical Storm": ["hurricane"],
                 "Flood": ["flood"], "Coastal Storm": ["hurricane", "flood"], "Drought": ["drought"],
                 "Severe Ice Storm": ["hail_ice"], "Snowstorm": ["hail_ice"], "Winter Storm": ["hail_ice"],
                 "Severe Storm": ["wind", "flood", "tornado", "hail_ice"], "Severe Storm(s)": ["wind", "flood", "tornado", "hail_ice"]}
fema["fips"] = pd.to_numeric(fema.fipsStateCode, errors="coerce") * 1000 + pd.to_numeric(fema.fipsCountyCode, errors="coerce")
fema = fema[fema.fipsCountyCode > 0]
fema["year"] = pd.to_datetime(fema.incidentBeginDate, errors="coerce").dt.year
dec = set()
for r in fema.itertuples():
    for p in PERIL_OF_DECL.get(r.incidentType, []):
        for k in (-1, 0, 1):
            dec.add((int(r.fips), int(r.year) + k, p))
all_coh = []
for p in PERILS:
    for ti, g in cohorts_for(p):
        all_coh.append((ti, g, p, int((counties[ti], g, p) in dec)))
share = pd.DataFrame(all_coh, columns=["fips_idx", "cohort", "peril", "declared"])
share_tab = share.groupby("peril").declared.agg(["mean", "count"]).rename(columns={"mean": "share_declared", "count": "n_cohorts"})
share_tab.to_csv(RES / "fema_share_by_peril.csv")
cr = [(r.fips_idx, r.cohort, "declared" if r.declared else "undeclared") for r in share.itertuples()]
coh = pd.DataFrame(cr, columns=["fips_idx", "cohort", "grp"])
fema_rows = []
for o in ["ln_pc_income", "ln_employment"]:
    S = E.build_stacks(Y[o], np.array(counties), states, coh, pool_by_g, ADJ, seed=SEED)
    res = E.fit(S, ["declared", "undeclared"])
    est, se, w = E.contrast(res, "declared", "undeclared")
    for gname in ["declared", "undeclared"]:
        sm = E.summarize(res, gname)
        fema_rows.append(dict(outcome=o, group=gname, avg_post=sm["avg_post"], se=sm["se"],
                              lead_adj=sm["lead_adj"], lead_adj_se=sm["lead_adj_se"],
                              pretrend_p=sm["pretrend_p"], mde80=sm["mde80"],
                              n_cohorts=int((coh.grp == gname).sum())))
    fema_rows.append(dict(outcome=o, group="declared_minus_undeclared", avg_post=est, se=se,
                          lead_adj=np.nan, lead_adj_se=np.nan, pretrend_p=np.nan, mde80=2.8 * se,
                          n_cohorts=len(coh)))
    log(f"stage5 {o:16s} declared-undeclared={est*100:+.2f}pp (se {se*100:.2f})")
pd.DataFrame(fema_rows).to_csv(RES / "table5_fema_split.csv", index=False)

# stage 6: SPEI-6 drought, two control screenings
# spei_screened: controls have no reporting shock and no SPEI shock in [g-3, g+3]
# reporting_screened: controls screened on reporting shocks only
spei = pd.read_csv(DATA / "spei_county_year.csv")
SM = np.zeros((len(counties), len(YEARS)), dtype=bool)
sub = spei[spei.fips.isin(cidx)]
SM[[cidx[f] for f in sub.fips], [yi[y] for y in sub.year]] = sub.spei_shock.to_numpy().astype(bool)
rep = SH["drought"]
both = SM & rep; either = SM | rep
agree = dict(spei_shock_cy=int(SM[:, [yi[y] for y in range(2001, 2020)]].sum()),
             reporting_shock_cy=int(rep[:, [yi[y] for y in range(2001, 2020)]].sum()),
             both_cy=int(both[:, [yi[y] for y in range(2001, 2020)]].sum()),
             jaccard=float(both[:, [yi[y] for y in range(2001, 2020)]].sum() / either[:, [yi[y] for y in range(2001, 2020)]].sum()))
pool_by_g_spei = {}
for g in range(2004, 2017):
    win = [yi[g + d] for d in range(-3, 4)]
    pool_by_g_spei[g] = np.where(~ANY[:, win].any(axis=1) & ~SM[:, win].any(axis=1))[0]
cr = []
for g in range(2004, 2017):
    win = [yi[g + d] for d in range(-3, 4)]
    own = SM[:, win].sum(axis=1)
    for ti in np.where(SM[:, yi[g]] & (own == 1))[0]:
        if any(SH[q][ti, win].any() for q in PERILS):  # exclusive of all reporting perils
            continue
        cr.append((int(ti), g))
counts["spei_drought"] = len(cr)
cohort_loc_rows += [dict(spec="drought_spei", fips=int(counties[ti]), cohort_year=int(g)) for ti, g in cr]
# share of control units (all, and in the treated county's state) that are SPEI-shocked
# in the cohort year, for the reporting-screened stacks
_coh_spei = pd.DataFrame(cr, columns=["fips_idx", "cohort"])
_coh_spei["grp"] = "spei_drought"
_Sc = E.build_stacks(Y["ln_pc_income"], np.array(counties), states, _coh_spei,
                     pool_by_g, ADJ, seed=SEED)
_sid = _Sc["ent"] // 10**7
_treated_state = {}
for s in np.unique(_sid):
    m = (_sid == s) & (_Sc["trt"] == 1)
    _treated_state[s] = int(_Sc["state"][m][0])
_ctrl0 = (_Sc["trt"] == 0) & (_Sc["rel"] == 0)
_shocked = np.array([bool(SM[cidx[int(f)], yi[int(y)]])
                     for f, y in zip(_Sc["fips"][_ctrl0], _Sc["year"][_ctrl0])])
_same = np.array([_treated_state[s] == st
                  for s, st in zip(_sid[_ctrl0], _Sc["state"][_ctrl0])])
agree["control_contamination_reporting_screened"] = dict(
    share_all_controls=float(_shocked.mean()),
    share_same_state_controls=float(_shocked[_same].mean()),
    n_control_units_rel0=int(_ctrl0.sum()),
    n_same_state_control_units_rel0=int(_same.sum()))
log("stage6 control contamination (reporting-screened): "
    f"all {agree['control_contamination_reporting_screened']['share_all_controls']:.3f}, "
    f"same-state {agree['control_contamination_reporting_screened']['share_same_state_controls']:.3f}")
spei_rows = []
spei_paths = {}
spei_paths_unscreened = {}
for screening, pool in [("spei_screened", pool_by_g_spei),
                        ("reporting_screened", pool_by_g)]:
    for o in OUTCOMES:
        res, _ = run_design(cr, o, lambda r: "spei_drought", pool=pool)
        sm = E.summarize(res, "spei_drought")
        if screening == "spei_screened":
            spei_paths[o] = sm["path"]
        else:
            spei_paths_unscreened[o] = sm["path"]
        hb = E.honest_bounds(res, "spei_drought", 0.0025)
        hb10s = E.honest_bounds(res, "spei_drought", 0.01)
        wp, _se = E.wild_boot_p(res, E._w(res, "spei_drought", E.POST), seed=SEED)
        spei_rows.append(dict(peril="drought_spei", outcome=o, control_screening=screening,
                              n_cohorts=len(cr),
                              avg_post=sm["avg_post"], se=sm["se"], lead_adj=sm["lead_adj"],
                              lead_adj_se=sm["lead_adj_se"], pretrend_p=sm["pretrend_p"],
                              mde80=sm["mde80"], wild_p=wp, clusters=sm["clusters"],
                              honest_lo_M0025=hb["lo"], honest_hi_M0025=hb["hi"], honest_lo_M01=hb10s["lo"], honest_hi_M01=hb10s["hi"],
                              breakdown_M=E.breakdown_M(res, "spei_drought")))
        log(f"stage6 spei[{screening}] {o:16s} n={len(cr)} post={sm['avg_post']*100:+.2f}pp (se {sm['se']*100:.2f}) pretrend p={sm['pretrend_p']:.3f}")
pd.DataFrame(spei_rows).to_csv(RES / "table6_spei_drought.csv", index=False)
(RES / "spei_agreement.json").write_text(json.dumps(agree, indent=2))
(RES / "cohort_counts.json").write_text(json.dumps({k: int(v) for k, v in counts.items()}, indent=2))
log("counts:", counts)

# stage 7: figure inputs and manifest
pd.DataFrame(cohort_loc_rows).to_csv(RES / "cohort_locations.csv", index=False)
pd.DataFrame([dict(outcome=o, rel_time=k, coef=v[0], se=v[1])
              for o, path in spei_paths.items() for k, v in path.items()]
             ).to_csv(RES / "paths_spei.csv", index=False)
pd.DataFrame([dict(outcome=o, rel_time=k, coef=v[0], se=v[1])
              for o, path in spei_paths_unscreened.items() for k, v in path.items()]
             ).to_csv(RES / "paths_spei_unscreened.csv", index=False)
import hashlib, platform as _platform
from datetime import date as _date
def _sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()
EXPECTED_TABLES = [
    "baseline_corrected_summary.json", "cohort_counts.json", "cohort_locations.csv",
    "fema_share_by_peril.csv", "onset_group_summary.csv", "paths_onset_contrast.csv",
    "paths_primary.csv", "paths_spei.csv", "paths_spei_unscreened.csv",
    "ring_distances.json", "spei_agreement.json",
    "table2_primary_per_peril.csv", "table3_onset_contrast.csv",
    "table4_rings_stacked.csv", "table5_fema_split.csv", "table6_spei_drought.csv",
    "table_hurricane_sensitivity.csv",
]
EXPECTED_FIGURES = [
    "fig1_cohort_maps.png", "fig1_cohort_maps.pdf",
    "fig2_income_paths.png", "fig2_income_paths.pdf",
    "fig3_employment_paths.png", "fig3_employment_paths.pdf",
    "fig4_onset_contrast.png", "fig4_onset_contrast.pdf",
    "fig5_spillover_rings.png", "fig5_spillover_rings.pdf",
    "fig6_declaration_split.png", "fig6_declaration_split.pdf",
    "fig7_drought_regimes.png", "fig7_drought_regimes.pdf",
]
_manifest_path = RES / "run_manifest.json"
_manifest = json.loads(_manifest_path.read_text()) if _manifest_path.exists() else {}
_manifest.update({
    "paper": "Severe-weather county event study",
    "build_date": _date.today().isoformat(),
    "python": _platform.python_version(),
    "platform": _platform.platform(),
    "seed": SEED,
    "control_cap_per_stack": 600,
    "scale_frac_income": SCALE_FRAC,
    "universe_counties": len(counties),
    "spei_primary_control_screening": "spei_screened (no SPEI or reporting shock in [g-3, g+3]); reporting_screened rows in table6_spei_drought.csv are the original construction, retained for transparency",
    "data_hashes": {f: _sha(DATA / f) for f in
                    ("panel_county_year.parquet", "spei_county_year.csv", "fema_declarations.csv")},
    "expected_outputs": {"tables_and_data": EXPECTED_TABLES, "figures": EXPECTED_FIGURES},
    "figures_generated_by": "code/figures.py (run after run_all.py)",
    "runtime_seconds_run_all": round(time.time() - t0, 1),
})
_manifest_path.write_text(json.dumps(_manifest, indent=2))
log("stage 7 figure inputs + run_manifest.json written")
log("done in", round(time.time() - t0, 1), "s")
