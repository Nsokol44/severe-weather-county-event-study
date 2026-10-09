#!/usr/bin/env python3
"""Stacked event study by peril and distance-ring event study on the 2001-2019 county panel.

Inputs: data/panel_county_year.parquet, data/shock_events.csv, data/raw/county_adjacency2023.txt.
Outputs: runs/run_<date>/event_study_stacked.csv, ring_event_study.csv, estimation_summary.json.
"""
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
RUNS = HERE.parent / "runs"
PERILS = ["tornado", "hurricane", "hail_ice", "drought", "flood", "wind"]
OUTCOMES = ["ln_pc_income", "ln_employment"]
PANEL_Y0, PANEL_Y1 = 2001, 2019
COHORT_Y0, COHORT_Y1 = 2004, 2016
K_MIN, K_MAX = -3, 3
KS = [k for k in range(K_MIN, K_MAX + 1) if k != -1]


def demean_two_way(y: np.ndarray, X: np.ndarray, g1: np.ndarray, g2: np.ndarray,
                   tol: float = 1e-9, max_iter: int = 100) -> tuple[np.ndarray, np.ndarray]:
    """Alternating-projections demeaning over two grouping factors."""
    y = y.astype(float).copy()
    X = X.astype(float).copy()
    c1, u1 = pd.factorize(g1)
    c2, u2 = pd.factorize(g2)
    n1, n2 = len(u1), len(u2)

    def sweep(v: np.ndarray, codes: np.ndarray, k: int) -> np.ndarray:
        s = np.bincount(codes, weights=v, minlength=k)
        n = np.bincount(codes, minlength=k)
        return v - (s / np.maximum(n, 1))[codes]

    for _ in range(max_iter):
        y_old = y
        y = sweep(y, c1, n1)
        for j in range(X.shape[1]):
            X[:, j] = sweep(X[:, j], c1, n1)
        y = sweep(y, c2, n2)
        for j in range(X.shape[1]):
            X[:, j] = sweep(X[:, j], c2, n2)
        if np.max(np.abs(y - y_old)) < tol:
            break
    return y, X


def ols_cluster(y: np.ndarray, X: np.ndarray, clusters: np.ndarray,
                dof_extra: int) -> dict:
    """OLS on (already demeaned) y, X with cluster-robust (CR1) covariance."""
    n, k = X.shape
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ (X.T @ y)
    resid = y - X @ beta
    cl, uniq = pd.factorize(clusters)
    G = len(uniq)
    scores = np.zeros((G, k))
    np.add.at(scores, cl, X * resid[:, None])
    meat = scores.T @ scores
    dof = max(n - k - dof_extra, 1)
    scale = (G / max(G - 1, 1)) * (n - 1) / dof
    V = scale * XtX_inv @ meat @ XtX_inv
    se = np.sqrt(np.maximum(np.diag(V), 0))
    return {"beta": beta, "se": se, "V": V, "n": n, "resid": resid}


def wald_leads(res: dict, idx: list[int]) -> tuple[float, float]:
    from scipy import stats
    b = res["beta"][idx]
    V = res["V"][np.ix_(idx, idx)]
    W = float(b @ np.linalg.pinv(V) @ b)
    p = float(1 - stats.chi2.cdf(W, len(idx)))
    return W, p


def clean_cohorts(shocks: pd.DataFrame, peril: str) -> pd.DataFrame:
    """County-years with one major shock of `peril` and none in the 3 years either side."""
    s = shocks[shocks.peril == peril]
    idx = set(map(tuple, s[["fips", "year"]].itertuples(index=False, name=None)))
    keep = [(f, g) for f, g in idx
            if COHORT_Y0 <= g <= COHORT_Y1
            and not any((f, g + d) in idx for d in range(K_MIN, K_MAX + 1) if d != 0)]
    return pd.DataFrame(sorted(keep), columns=["fips", "cohort"])


def _shock_matrix(shocks: pd.DataFrame, peril: str,
                  counties: list[int], years: list[int]) -> np.ndarray:
    """Boolean county x year matrix of major shocks for one peril."""
    ci = {c: i for i, c in enumerate(counties)}
    yi = {y: j for j, y in enumerate(years)}
    S = np.zeros((len(counties), len(years)), dtype=bool)
    sub = shocks[shocks.peril == peril]
    for f, y in sub[["fips", "year"]].itertuples(index=False):
        if f in ci and y in yi:
            S[ci[f], yi[y]] = True
    return S


def stacked_event_study(panel: pd.DataFrame, shocks: pd.DataFrame,
                        peril: str, outcome: str) -> tuple[pd.DataFrame, dict]:
    """Stacked event study for one peril and outcome; returns coefficient rows and a summary dict."""
    counties = sorted(panel["fips"].unique())
    ci = {c: i for i, c in enumerate(counties)}
    panel_years = list(range(PANEL_Y0, PANEL_Y1 + 1))
    yi = {y: j for j, y in enumerate(panel_years)}
    Y = np.full((len(counties), len(panel_years)), np.nan)
    sub = panel[panel.year.between(PANEL_Y0, PANEL_Y1)]
    Y[[ci[f] for f in sub["fips"]], [yi[y] for y in sub["year"]]] = sub[outcome].to_numpy()

    storm_years = list(range(int(shocks.year.min()), int(shocks.year.max()) + 1))
    S = _shock_matrix(shocks, peril, counties, storm_years)
    si = {y: j for j, y in enumerate(storm_years)}

    # clean cohorts
    cohorts: list[tuple[int, int]] = []
    for g in range(COHORT_Y0, COHORT_Y1 + 1):
        win = [si[g + d] for d in range(K_MIN, K_MAX + 1)]
        wsum = S[:, win].sum(axis=1)
        treated_idx = np.where(S[:, si[g]] & (wsum == 1))[0]
        cohorts += [(int(counties[i]), g) for i in treated_idx]
    # control mask per cohort year
    ctrl_mask: dict[int, np.ndarray] = {}
    for g in range(COHORT_Y0, COHORT_Y1 + 1):
        win = [si[g + d] for d in range(K_MIN, K_MAX + 1)]
        ctrl_mask[g] = S[:, win].sum(axis=1) == 0

    fips_l, year_l, rel_l, trt_l, ent_l, tim_l, y_l = [], [], [], [], [], [], []
    for s, (f_t, g) in enumerate(cohorts):
        members = np.where(ctrl_mask[g])[0]
        members = np.append(members, ci[f_t])
        yrs = np.arange(g + K_MIN, g + K_MAX + 1)
        yy, mm = np.meshgrid(yrs, members)
        vals = Y[mm, [yi[int(y)] for y in yrs]]
        keep = ~np.isnan(vals)
        kf = keep.ravel()
        fips_l.append(np.repeat([counties[i] for i in members], len(yrs))[kf])
        year_l.append(np.tile(yrs, len(members))[kf])
        rel_l.append(np.tile(yrs - g, len(members))[kf])
        trt_l.append(np.repeat((members == ci[f_t]).astype(int), len(yrs))[kf])
        ent_l.append(np.repeat(s * 100000 + np.array([counties[i] for i in members]), len(yrs))[kf])
        tim_l.append(np.tile(s * 100 + yrs, len(members))[kf])
        y_l.append(vals[keep])
    fips_a = np.concatenate(fips_l); year_a = np.concatenate(year_l)
    rel_a = np.concatenate(rel_l); trt_a = np.concatenate(trt_l)
    ent_a = np.concatenate(ent_l); tim_a = np.concatenate(tim_l)
    y_a = np.concatenate(y_l)
    X = np.column_stack([(trt_a * (rel_a == k)).astype(float) for k in KS])
    yd, Xd = demean_two_way(y_a, X, ent_a, tim_a)
    n_ent = len(np.unique(ent_a)); n_time = len(np.unique(tim_a))
    res = ols_cluster(yd, Xd, fips_a, dof_extra=n_ent + n_time)
    rows = []
    for j, k in enumerate(KS):
        rows.append({"peril": peril, "outcome": outcome, "rel_time": k,
                     "coef": res["beta"][j], "se": res["se"][j],
                     "t": res["beta"][j] / res["se"][j] if res["se"][j] > 0 else np.nan})
    post_idx = [KS.index(k) for k in [0, 1, 2, 3]]
    lead_idx = [KS.index(-3), KS.index(-2)]
    w = np.zeros(len(KS)); w[post_idx] = 0.25
    avg_post = float(w @ res["beta"])
    avg_post_se = float(np.sqrt(w @ res["V"] @ w))
    # lead-adjusted post average
    w_adj = w.copy(); w_adj[lead_idx] -= 0.5
    avg_post_adj = float(w_adj @ res["beta"])
    avg_post_adj_se = float(np.sqrt(w_adj @ res["V"] @ w_adj))
    W, pv = wald_leads(res, lead_idx)
    np.savez_compressed(
        HERE.parent / "runs" / f"run_{date.today().isoformat()}" / f"V_stacked_{peril}_{outcome}.npz",
        V=res["V"], ks=np.array(KS))
    meta = {"peril": peril, "outcome": outcome, "n_cohorts": int(len(cohorts)),
            "n_obs": int(res["n"]), "avg_post": avg_post, "avg_post_se": avg_post_se,
            "avg_post_lead_adjusted": avg_post_adj,
            "avg_post_lead_adjusted_se": avg_post_adj_se,
            "pretrend_wald": W, "pretrend_p": pv}
    return pd.DataFrame(rows), meta


def load_adjacency() -> dict[int, set[int]]:
    adj: dict[int, set[int]] = {}
    raw = pd.read_csv(DATA / "raw" / "county_adjacency2023.txt", sep="|", dtype=str)
    raw.columns = [c.strip() for c in raw.columns]
    a = pd.to_numeric(raw.iloc[:, 1], errors="coerce")
    b = pd.to_numeric(raw.iloc[:, 3], errors="coerce")
    for x, y in zip(a, b):
        if pd.isna(x) or pd.isna(y) or x == y:
            continue
        adj.setdefault(int(x), set()).add(int(y))
        adj.setdefault(int(y), set()).add(int(x))
    return adj


def ring_event_study(panel: pd.DataFrame, shocks: pd.DataFrame, adj: dict,
                     perils: list[str], outcome: str) -> tuple[pd.DataFrame, dict]:
    """Ring event study: hit, ring1 and ring2 status and relative time from the nearest clean shock."""
    events = []
    for peril in perils:
        c = clean_cohorts(shocks, peril)
        c["peril"] = peril
        events.append(c)
    ev = pd.concat(events, ignore_index=True)
    hit_years: dict[int, list[int]] = {}
    for f, g in ev[["fips", "cohort"]].itertuples(index=False):
        hit_years.setdefault(int(f), []).append(int(g))
    ring1_of: dict[int, set[int]] = {}
    for f, years in hit_years.items():
        ring1_of[f] = adj.get(f, set())
    all_ring1 = set().union(*ring1_of.values()) if ring1_of else set()
    ring2_candidates = set()
    for c in all_ring1:
        ring2_candidates |= adj.get(c, set())
    ring2_candidates -= set(hit_years) | all_ring1

    p = panel[(panel.year >= PANEL_Y0) & (panel.year <= PANEL_Y1)].copy()
    status = np.full(len(p), "", dtype=object)
    rel = np.full(len(p), np.nan)
    fips_arr = p["fips"].to_numpy(); year_arr = p["year"].to_numpy()
    # ring 0 (hit) counties
    for i, (f, y) in enumerate(zip(fips_arr, year_arr)):
        if f in hit_years:
            ds = [y - g for g in hit_years[f] if K_MIN <= y - g <= K_MAX]
            if ds:
                d = min(ds, key=abs)
                status[i], rel[i] = "ring0", d
    # ring 1 and ring 2 counties, relative time from the nearest hit
    for i, (f, y) in enumerate(zip(fips_arr, year_arr)):
        if status[i] != "":
            continue
        cands = []
        for nb in adj.get(f, set()):
            if nb in hit_years:
                for g in hit_years[nb]:
                    if K_MIN <= y - g <= K_MAX:
                        cands.append(("ring1", y - g))
        if not cands and f in ring2_candidates:
            for nb in adj.get(f, set()):
                for nb2 in adj.get(nb, set()):
                    if nb2 in hit_years:
                        for g in hit_years[nb2]:
                            if K_MIN <= y - g <= K_MAX:
                                cands.append(("ring2", y - g))
        if cands:
            # prefer ring1 over ring2, then nearest in time
            cands.sort(key=lambda t: (0 if t[0] == "ring1" else 1, abs(t[1])))
            status[i], rel[i] = cands[0]
    p["status"], p["rel"] = status, rel
    rings = ["ring0", "ring1", "ring2"]
    cols = []
    for r in rings:
        for k in KS:
            col = f"{r}_k{k}"
            p[col] = ((p.status == r) & (p.rel == k)).astype(float)
            cols.append(col)
    dat = p.dropna(subset=[outcome])
    X = dat[cols].to_numpy(); y = dat[outcome].to_numpy()
    yd, Xd = demean_two_way(y, X, dat["fips"].to_numpy(), dat["year"].to_numpy())
    n_ent = dat["fips"].nunique(); n_time = dat["year"].nunique()
    res = ols_cluster(yd, Xd, dat["fips"].to_numpy(), dof_extra=n_ent + n_time)
    rows = []
    for j, col in enumerate(cols):
        r, k = col.split("_k")
        rows.append({"pool": "+".join(perils), "outcome": outcome, "ring": r,
                     "rel_time": int(k), "coef": res["beta"][j], "se": res["se"][j]})
    meta = {"pool": "+".join(perils), "outcome": outcome,
            "n_clean_events": int(len(ev)), "n_obs": int(res["n"]),
            "n_hit_counties": int(len(hit_years))}
    return pd.DataFrame(rows), meta


def main() -> None:
    run_dir = RUNS / f"run_{date.today().isoformat()}"
    run_dir.mkdir(parents=True, exist_ok=True)
    panel = pd.read_parquet(DATA / "panel_county_year.parquet")
    shocks = pd.read_csv(DATA / "shock_events.csv")
    adj = load_adjacency()

    all_rows, metas = [], []
    for peril in PERILS:
        for outcome in OUTCOMES:
            rows, meta = stacked_event_study(panel, shocks, peril, outcome)
            all_rows.append(rows); metas.append(meta)
            print(f"[stacked] {peril:10s} {outcome:16s} cohorts={meta['n_cohorts']:4d} "
                  f"avg_post={meta['avg_post']:+.4f} (se {meta['avg_post_se']:.4f}) "
                  f"pretrend p={meta['pretrend_p']:.3f}", flush=True)
    stacked = pd.concat(all_rows, ignore_index=True)
    stacked.to_csv(run_dir / "event_study_stacked.csv", index=False)

    ring_rows, ring_metas = [], []
    fast = ["tornado", "hurricane", "hail_ice", "flood", "wind"]
    for pool in [fast, PERILS]:
        for outcome in OUTCOMES:
            rows, meta = ring_event_study(panel, shocks, adj, pool, outcome)
            ring_rows.append(rows); ring_metas.append(meta)
            print(f"[ring] {'+'.join(pool)[:28]:28s} {outcome:16s} "
                  f"events={meta['n_clean_events']}", flush=True)
    rings = pd.concat(ring_rows, ignore_index=True)
    rings.to_csv(run_dir / "ring_event_study.csv", index=False)

    with open(run_dir / "estimation_summary.json", "w") as f:
        json.dump({"stacked": metas, "rings": ring_metas}, f, indent=2)
    print(f"outputs -> {run_dir}")


if __name__ == "__main__":
    main()
