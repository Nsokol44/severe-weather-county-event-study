"""Stacked event-study estimator.

Each stack holds one treated county and a seeded sample of clean control counties.
Fits relative-time dummies with stack x county and stack x state x year fixed effects, state-clustered SEs.
Also: smoothness bounds, wild cluster bootstrap, Benjamini-Hochberg adjustment, minimum detectable effects.
"""
import numpy as np
import pandas as pd
from scipy import sparse, stats
from scipy.optimize import linprog

KS = [-3, -2, 0, 1, 2, 3]
POST = [0, 1, 2, 3]
LEAD = [-3, -2]

# Connecticut planning-region codes; no outcome data in 2001-2019
INGESTION_PSEUDO_FIPS = (9110, 9120, 9130, 9140, 9150, 9160, 9170, 9180, 9190)


def universe(panel):
    """Panel rows for every county except the pseudo-codes above."""
    return panel[~panel.fips.isin(INGESTION_PSEUDO_FIPS)].copy()


def _demean(V, codes_list, counts_list, tol=1e-9, max_iter=200):
    """Alternating-projections demeaning of the columns of V over two factors."""
    V = V.astype(float).copy()
    mats = []
    for codes, counts in zip(codes_list, counts_list):
        n = len(V)
        D = sparse.csr_matrix((np.ones(n), (np.arange(n), codes)),
                              shape=(n, len(counts)))
        mats.append((D, counts))
    for _ in range(max_iter):
        old = V[:, 0].copy()
        for D, counts in mats:
            V = V - D @ ((D.T @ V) / np.maximum(counts, 1)[:, None])
        if np.max(np.abs(V[:, 0] - old)) < tol:
            break
    return V


def build_stacks(Y, counties, states, cohorts, control_pool_by_g, adj,
                 control_cap=600, seed=20261006):
    """Assemble the stacked arrays: treated county plus sampled clean controls per cohort."""
    fips_l, yr_l, rel_l, trt_l, ent_l, tim_l, grp_l, st_l, y_l = [], [], [], [], [], [], [], [], []
    years = Y_years_global
    yi = {y: j for j, y in enumerate(years)}
    for s, r in enumerate(cohorts.itertuples()):
        g = int(r.cohort); ti = int(r.fips_idx)
        pool = control_pool_by_g[g]
        nb = adj.get(int(counties[ti]), set())
        nb_idx = {COUNTY_INDEX[n] for n in nb if n in COUNTY_INDEX}
        pool = np.array([c for c in pool if c != ti and c not in nb_idx])
        rng = np.random.default_rng(seed + s)
        if len(pool) > control_cap:
            pool = np.sort(rng.choice(pool, size=control_cap, replace=False))
        mem = np.append(pool, ti)
        yrs = list(range(g - 3, g + 4))
        sub = Y[np.ix_(mem, [yi[y] for y in yrs])]
        ok = ~np.isnan(sub).any(axis=1)
        mem, sub = mem[ok], sub[ok]
        if ti not in mem:
            continue
        n, T = len(mem), len(yrs)
        fips_l.append(np.repeat(counties[mem], T)); yr_l.append(np.tile(yrs, n))
        rel_l.append(np.tile(np.array(yrs) - g, n))
        trt_l.append(np.repeat((mem == ti).astype(int), T))
        ent_l.append(np.repeat(s * 10**7 + mem, T))
        tim_l.append((s * 10**7 + np.repeat(states[mem], T) * 100
                      + np.tile(np.array(yrs) - 2000, n)))
        grp_l.append(np.repeat(r.grp, n * T))
        st_l.append(np.repeat(states[mem], T)); y_l.append(sub.ravel())
    cat = np.concatenate
    return dict(fips=cat(fips_l), year=cat(yr_l), rel=cat(rel_l), trt=cat(trt_l),
                ent=cat(ent_l), tim=cat(tim_l), grp=cat(grp_l),
                state=cat(st_l), y=cat(y_l))


Y_years_global = list(range(2001, 2020))
COUNTY_INDEX = {}


def fit(S, groups):
    cols, names = [], []
    for g in groups:
        for k in KS:
            cols.append(((S["grp"] == g) & (S["trt"] == 1)
                         & (S["rel"] == k)).astype(float))
            names.append((g, k))
    X = np.column_stack(cols)
    V = np.column_stack([S["y"], X])
    c1, u1 = pd.factorize(S["ent"]); c2, u2 = pd.factorize(S["tim"])
    cnt1 = np.bincount(c1); cnt2 = np.bincount(c2)
    Vd = _demean(V, [c1, c2], [cnt1, cnt2])
    yd, Xd = Vd[:, 0], Vd[:, 1:]
    XtX_inv = np.linalg.pinv(Xd.T @ Xd)
    b = XtX_inv @ (Xd.T @ yd)
    e = yd - Xd @ b
    cl, uniq = pd.factorize(S["state"]); G = len(uniq)
    sc = np.zeros((G, X.shape[1])); np.add.at(sc, cl, Xd * e[:, None])
    n, kdim = X.shape
    dof = max(n - kdim - len(u1) - len(u2), 1)
    scale = (G / max(G - 1, 1)) * (n - 1) / dof
    Vcov = scale * XtX_inv @ (sc.T @ sc) @ XtX_inv
    return dict(beta=b, V=Vcov, names=names, n=n, clusters=G, groups=groups,
                scores=sc, XtX_inv=XtX_inv, resid=e, cluster_labels=uniq, scale=scale)


def _w(res, g, ks):
    w = np.zeros(len(res["names"]))
    for k in ks:
        w[res["names"].index((g, k))] = 1.0 / len(ks)
    return w


def combo(res, w):
    return float(w @ res["beta"]), float(np.sqrt(max(w @ res["V"] @ w, 0)))


def summarize(res, g):
    wp, wl = _w(res, g, POST), _w(res, g, LEAD)
    ap, sp = combo(res, wp); al, sl = combo(res, wp - wl)
    idx = [res["names"].index((g, k)) for k in LEAD]
    bl = res["beta"][idx]; Vl = res["V"][np.ix_(idx, idx)]
    W = float(bl @ np.linalg.pinv(Vl) @ bl)
    path = {k: (float(res["beta"][res["names"].index((g, k))]),
                float(np.sqrt(res["V"][res["names"].index((g, k)),
                                        res["names"].index((g, k))])))
            for k in KS}
    return dict(group=g, avg_post=ap, se=sp, lead_adj=al, lead_adj_se=sl,
                pretrend_p=float(1 - stats.chi2.cdf(W, 2)), mde80=float(2.8 * sp),
                clusters=res["clusters"], n=res["n"], path=path)


def contrast(res, g1, g0, lead_adjust=False):
    w = _w(res, g1, POST) - _w(res, g0, POST)
    if lead_adjust:
        w = w - (_w(res, g1, LEAD) - _w(res, g0, LEAD))
    est, se = combo(res, w)
    return est, se, w


def wild_boot_p(res, w, reps=999, seed=20261006):
    """Wild cluster bootstrap (Rademacher) p-value for w'beta = 0, from cluster scores."""
    rng = np.random.default_rng(seed)
    infl = res["scores"] @ (res["XtX_inv"] @ w) * np.sqrt(res.get("scale", 1.0))  # per-cluster influence
    est = float(w @ res["beta"])
    R = rng.choice([-1.0, 1.0], size=(reps, len(infl)))
    draws = est + R @ infl - np.mean(R @ infl)
    # draws under the null
    null_draws = (R @ infl)
    p = float((np.abs(null_draws) >= abs(est)).mean())
    return p, float(np.std(null_draws + est))


def honest_bounds(res, g, M, z=1.96):
    """Smoothness bounds on the average post effect: |second difference| of the trend <= M, widened by z*SE. NaN if infeasible."""
    b = res["beta"]; i = lambda k: res["names"].index((g, k))
    d_m3, d_m2 = b[i(-3)], b[i(-2)]
    est, se = combo(res, _w(res, g, POST))
    A, ub = [], []
    for t in range(1, 6):
        coef = np.zeros(4); const = 0.0
        for pos, s_ in zip([t - 1, t, t + 1], [1, -2, 1]):
            if pos < 3:
                const += s_ * [d_m3, d_m2, 0.0][pos]
            else:
                coef[pos - 3] += s_
        A += [coef, -coef]; ub += [M - const, M + const]
    c = np.ones(4) / 4
    lo = linprog(c, A_ub=np.array(A), b_ub=np.array(ub), bounds=[(None, None)] * 4)
    hi = linprog(-c, A_ub=np.array(A), b_ub=np.array(ub), bounds=[(None, None)] * 4)
    if lo.status != 0 or hi.status != 0:
        return dict(M=M, lo=np.nan, hi=np.nan, est=est)
    return dict(M=M, lo=est - (-hi.fun) - z * se, hi=est - lo.fun + z * se, est=est)


def breakdown_M(res, g, grid=np.linspace(0, 0.02, 41)):
    for M in grid:
        hb = honest_bounds(res, g, float(M))
        if np.isfinite(hb["lo"]) and hb["lo"] <= 0 <= hb["hi"]:
            return float(M)
    return float("inf")


def bh_adjust(p):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p)
    q = p[o] * n / (np.arange(n) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(n); out[o] = np.minimum(q, 1)
    return out
