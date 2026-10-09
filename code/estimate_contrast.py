#!/usr/bin/env python3
"""Slow-vs-fast onset contrast, drought threshold sensitivity and drought ring pools.

Inputs: data/panel_county_year.parquet, data/shock_events.csv and the drought6 files in data/.
Outputs: runs/run_<date>/contrast_paths.csv, drought_sensitivity_ge6.csv, ring_event_study_extra.csv, contrast_summary.json.
Fast group: tornado, hurricane, hail_ice, wind. Slow group: drought. Controls capped at 400 per stack, seed 20261006.
"""
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from estimate_event_study import (COHORT_Y0, COHORT_Y1, K_MAX, K_MIN, KS,
                                  OUTCOMES, PANEL_Y0, PANEL_Y1, _shock_matrix,
                                  demean_two_way, load_adjacency, ols_cluster,
                                  ring_event_study, stacked_event_study)

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
RUNS = HERE.parent / "runs"
FAST = ["tornado", "hurricane", "hail_ice", "wind"]
SLOW = ["drought"]
CONTROL_CAP = 400
SEED = 20261006


def assemble_group_stacks(panel, shocks, perils, outcome, rng):
    """Assemble stacks with capped controls for several perils, with a fast or slow label per row."""
    counties = sorted(panel["fips"].unique())
    ci = {c: i for i, c in enumerate(counties)}
    panel_years = list(range(PANEL_Y0, PANEL_Y1 + 1))
    yi = {y: j for j, y in enumerate(panel_years)}
    Y = np.full((len(counties), len(panel_years)), np.nan)
    sub = panel[panel.year.between(PANEL_Y0, PANEL_Y1)]
    Y[[ci[f] for f in sub["fips"]], [yi[y] for y in sub["year"]]] = sub[outcome].to_numpy()
    storm_years = list(range(int(shocks.year.min()), int(shocks.year.max()) + 1))
    si = {y: j for j, y in enumerate(storm_years)}

    out = {k: [] for k in ["fips", "rel", "trt", "ent", "tim", "y", "grp"]}
    s_global = 0
    for peril in perils:
        grp = "slow" if peril in SLOW else "fast"
        S = _shock_matrix(shocks, peril, counties, storm_years)
        cohorts = []
        for g in range(COHORT_Y0, COHORT_Y1 + 1):
            win = [si[g + d] for d in range(K_MIN, K_MAX + 1)]
            wsum = S[:, win].sum(axis=1)
            treated_idx = np.where(S[:, si[g]] & (wsum == 1))[0]
            cohorts += [(int(counties[i]), g) for i in treated_idx]
        ctrl_mask = {}
        for g in range(COHORT_Y0, COHORT_Y1 + 1):
            win = [si[g + d] for d in range(K_MIN, K_MAX + 1)]
            ctrl_mask[g] = S[:, win].sum(axis=1) == 0
        for (f_t, g) in cohorts:
            members_all = np.where(ctrl_mask[g])[0]
            if len(members_all) > CONTROL_CAP:
                members = rng.choice(members_all, CONTROL_CAP, replace=False)
            else:
                members = members_all
            members = np.append(members, ci[f_t])
            yrs = np.arange(g + K_MIN, g + K_MAX + 1)
            vals = Y[np.ix_(members, [yi[int(y)] for y in yrs])]
            keep = ~np.isnan(vals)
            kf = keep.ravel()
            n_m = len(members)
            out["fips"].append(np.repeat([counties[i] for i in members], len(yrs))[kf])
            out["rel"].append(np.tile(yrs - g, n_m)[kf])
            out["trt"].append(np.repeat((members == ci[f_t]).astype(int), len(yrs))[kf])
            out["ent"].append(np.repeat(s_global * 100000 + np.array(
                [counties[i] for i in members]), len(yrs))[kf])
            out["tim"].append(np.tile(s_global * 100 + yrs, n_m)[kf])
            out["y"].append(vals[keep])
            out["grp"].append(np.repeat(grp, kf.sum()))
            s_global += 1
    return {k: np.concatenate(v) for k, v in out.items()}, s_global


def pooled_contrast(panel, shocks, outcome):
    rng = np.random.default_rng(SEED)
    A, n_stacks = assemble_group_stacks(panel, shocks, FAST + SLOW, outcome, rng)
    groups = ["fast", "slow"]
    cols = []
    for grp in groups:
        for k in KS:
            cols.append(((A["grp"] == grp) & (A["trt"] == 1) & (A["rel"] == k)).astype(float))
    X = np.column_stack(cols)
    yd, Xd = demean_two_way(A["y"], X, A["ent"], A["tim"])
    res = ols_cluster(yd, Xd, A["fips"],
                      dof_extra=len(np.unique(A["ent"])) + len(np.unique(A["tim"])))
    nk = len(KS)
    rows = []
    for gi, grp in enumerate(groups):
        for j, k in enumerate(KS):
            rows.append({"outcome": outcome, "group": grp, "rel_time": k,
                         "coef": float(res["beta"][gi * nk + j]),
                         "se": float(res["se"][gi * nk + j])})
    contrasts = {}
    for label, idxs in [("avg_post", [0, 1, 2, 3]), ("k0", [0]), ("k3", [3])]:
        w = np.zeros(2 * nk)
        for k in idxs:
            w[KS.index(k)] = -1.0 / len(idxs)         # fast
            w[nk + KS.index(k)] = 1.0 / len(idxs)     # slow
        contrasts[label] = {"slow_minus_fast": float(w @ res["beta"]),
                            "se": float(np.sqrt(w @ res["V"] @ w))}
    return pd.DataFrame(rows), {"outcome": outcome, "n_stacks": n_stacks,
                                "n_obs": int(res["n"]), "contrasts": contrasts}


def main() -> None:
    run_dir = RUNS / f"run_{date.today().isoformat()}"
    run_dir.mkdir(parents=True, exist_ok=True)
    panel = pd.read_parquet(DATA / "panel_county_year.parquet")
    shocks = pd.read_csv(DATA / "shock_events.csv")
    adj = load_adjacency()

    rows_all, metas = [], []
    for outcome in OUTCOMES:
        rows, meta = pooled_contrast(panel, shocks, outcome)
        rows_all.append(rows); metas.append(meta)
        c = meta["contrasts"]["avg_post"]
        print(f"[contrast] {outcome}: slow-fast avg post = {c['slow_minus_fast']:+.4f} "
              f"(se {c['se']:.4f}); stacks={meta['n_stacks']}", flush=True)
    pd.concat(rows_all, ignore_index=True).to_csv(run_dir / "contrast_paths.csv", index=False)

    # drought sensitivity, >= 6 months
    panel6 = pd.read_parquet(DATA / "panel_county_year_drought6.parquet")
    shocks6 = pd.read_csv(DATA / "shock_events_drought6.csv")
    sens_rows, sens_meta = [], []
    for outcome in OUTCOMES:
        rows, meta = stacked_event_study(panel6, shocks6, "drought", outcome)
        rows["rule"] = "drought_months_ge6"
        sens_rows.append(rows); sens_meta.append(meta)
        print(f"[sensitivity >=6mo] drought {outcome}: cohorts={meta['n_cohorts']} "
              f"avg_post={meta['avg_post']:+.4f} (se {meta['avg_post_se']:.4f}) "
              f"lead_adj={meta['avg_post_lead_adjusted']:+.4f}", flush=True)
    pd.concat(sens_rows, ignore_index=True).to_csv(
        run_dir / "drought_sensitivity_ge6.csv", index=False)

    # drought ring pools
    ring_rows, ring_meta = [], []
    for pool in [["drought"], ["drought", "flood"]]:
        for outcome in OUTCOMES:
            rows, meta = ring_event_study(panel, shocks, adj, pool, outcome)
            ring_rows.append(rows); ring_meta.append(meta)
            print(f"[ring+] {'+'.join(pool)} {outcome}: events={meta['n_clean_events']}", flush=True)
    pd.concat(ring_rows, ignore_index=True).to_csv(
        run_dir / "ring_event_study_extra.csv", index=False)

    with open(run_dir / "contrast_summary.json", "w") as f:
        json.dump({"contrast": metas, "drought_ge6": sens_meta, "rings_extra": ring_meta},
                  f, indent=2)
    print(f"outputs -> {run_dir}")


if __name__ == "__main__":
    main()
