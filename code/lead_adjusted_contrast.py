#!/usr/bin/env python3
"""Lead-adjusted slow-vs-fast onset contrast.

Same stacks, control cap and seed as estimate_contrast.pooled_contrast, keeping the full covariance matrix.
Usage: lead_adjusted_contrast.py <panel.parquet> <shocks.csv> <out.json>
Output: JSON with raw and lead-adjusted contrasts for ln_pc_income and ln_employment.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))
from estimate_contrast import FAST, SLOW, assemble_group_stacks  # noqa: E402
from estimate_event_study import KS, demean_two_way, ols_cluster  # noqa: E402

SEED = 20261006


def contrast_with_leads(panel, shocks, outcome):
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
    beta, V = res["beta"], res["V"]

    def lin(w):
        return float(w @ beta), float(np.sqrt(w @ V @ w))

    out = {"outcome": outcome, "n_stacks": n_stacks, "n_obs": int(res["n"])}
    # raw average post contrast
    w = np.zeros(2 * nk)
    for k in [0, 1, 2, 3]:
        w[KS.index(k)] = -0.25
        w[nk + KS.index(k)] = 0.25
    v, s = lin(w)
    out["raw_avg_post_contrast"] = {"slow_minus_fast": v, "se": s}
    # lead-adjusted contrast: post mean minus lead mean (k=-3,-2), slow minus fast
    w = np.zeros(2 * nk)
    for k in [0, 1, 2, 3]:
        w[KS.index(k)] = -0.25
        w[nk + KS.index(k)] = 0.25
    for k in [-3, -2]:
        w[KS.index(k)] = 0.5
        w[nk + KS.index(k)] = -0.5
    v, s = lin(w)
    out["lead_adjusted_contrast"] = {"slow_minus_fast": v, "se": s}
    # per-group lead means and lead-adjusted post effects
    for gi, grp in enumerate(groups):
        wl = np.zeros(2 * nk)
        for k in [-3, -2]:
            wl[gi * nk + KS.index(k)] = 0.5
        v, s = lin(wl)
        out[f"{grp}_lead_mean"] = {"coef": v, "se": s}
        wa = np.zeros(2 * nk)
        for k in [0, 1, 2, 3]:
            wa[gi * nk + KS.index(k)] = 0.25
        for k in [-3, -2]:
            wa[gi * nk + KS.index(k)] = -0.5
        v, s = lin(wa)
        out[f"{grp}_post_lead_adjusted"] = {"coef": v, "se": s}
    return out


def main() -> None:
    panel = pd.read_parquet(sys.argv[1])
    shocks = pd.read_csv(sys.argv[2])
    results = [contrast_with_leads(panel, shocks, oc)
               for oc in ["ln_pc_income", "ln_employment"]]
    with open(sys.argv[3], "w") as f:
        json.dump(results, f, indent=2)
    for r in results:
        print(r["outcome"],
              "raw:", r["raw_avg_post_contrast"],
              "lead-adj:", r["lead_adjusted_contrast"], flush=True)


if __name__ == "__main__":
    main()
