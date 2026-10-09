#!/usr/bin/env python3
"""Render figures 1-7 from the tables in results/.

Inputs: results/*.csv and *.json written by run_all.py, data/county_boundaries.geojson.
Outputs: results/fig*.png (300 dpi) and results/fig*.pdf. Usage: python figures.py
"""
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
DATA = ROOT / "data"
KS = [-3, -2, -1, 0, 1, 2, 3]
PP = 100.0  # log points -> percentage points

# Okabe-Ito palette
BLUE = "#0072B2"
SKY = "#56B4E9"
ORANGE = "#E69F00"
VERM = "#D55E00"
GREEN = "#009E73"
PURPLE = "#CC79A7"
GREY = "#6E6E6E"
PERIL_COLORS = {"tornado": BLUE, "hail_ice": SKY, "wind": PURPLE,
                "flood": GREEN, "drought": VERM, "hurricane": ORANGE,
                "drought_spei": "#8C6D2F"}
PERIL_LABELS = {"tornado": "Tornado", "hail_ice": "Hail / ice", "wind": "Wind",
                "flood": "Flood", "drought": "Drought (reporting)",
                "hurricane": "Hurricane", "drought_spei": "Drought (SPEI-6)"}
OUTCOME_LABELS = {"ln_pc_income": "Income", "ln_employment": "Employment",
                  "ln_population": "Population"}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.titlesize": 9.5,
    "axes.titleweight": "bold",
    "axes.labelsize": 9.5,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#444444",
    "figure.facecolor": "white",
})


def save(fig, stem):
    fig.savefig(RES / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(RES / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote", stem)


def header(fig, title, note, note_y=0.005):
    fig.suptitle(title, x=0.008, ha="left", fontsize=11, fontweight="bold")
    fig.text(0.008, note_y, note, fontsize=7, color="#444444", ha="left", va="bottom")


def style_path_ax(ax, ylabel=None):
    ax.axhline(0, color="#888888", lw=0.8, zorder=1)
    ax.axvline(-0.5, color="#888888", lw=0.7, ls="--", zorder=1)
    ax.axvspan(-0.5, 3.45, color="#000000", alpha=0.035, zorder=0)
    ax.set_xticks(KS)
    ax.set_xlim(-3.45, 3.45)
    ax.grid(axis="y", color="#E3E3E3", lw=0.6, zorder=0)
    if ylabel:
        ax.set_ylabel(ylabel)


def plot_path(ax, df, color, label=None, band=True):
    """Plot one path from rows of rel_time, coef, se (log points); k=-1 is the zero base."""
    d = df.set_index("rel_time")
    xs, ys, es = [], [], []
    for k in KS:
        if k == -1:
            xs.append(k); ys.append(0.0); es.append(0.0)
        elif k in d.index:
            xs.append(k); ys.append(d.loc[k, "coef"] * PP); es.append(d.loc[k, "se"] * PP)
    xs, ys, es = np.array(xs), np.array(ys), np.array(es)
    if band:
        ax.fill_between(xs, ys - 1.96 * es, ys + 1.96 * es, color=color, alpha=0.16, zorder=2)
    ax.plot(xs, ys, color=color, lw=1.7, marker="o", ms=4, label=label, zorder=3)


def avg_post_lookup(table, **keys):
    d = table
    for k, v in keys.items():
        d = d[d[k] == v]
    r = d.iloc[0]
    return r["avg_post"] * PP, r["se"] * PP


# load inputs
paths = pd.read_csv(RES / "paths_primary.csv")
paths_spei = pd.read_csv(RES / "paths_spei.csv")
paths_onset = pd.read_csv(RES / "paths_onset_contrast.csv")
tab2 = pd.read_csv(RES / "table2_primary_per_peril.csv")
tab3 = pd.read_csv(RES / "table3_onset_contrast.csv")
tab4 = pd.read_csv(RES / "table4_rings_stacked.csv")
tab5 = pd.read_csv(RES / "table5_fema_split.csv")
tab6 = pd.read_csv(RES / "table6_spei_drought.csv")
if "control_screening" in tab6.columns:
    # use the spei_screened rows
    tab6 = tab6[tab6.control_screening == "spei_screened"].reset_index(drop=True)
onset_grp = pd.read_csv(RES / "onset_group_summary.csv")
fema_share = pd.read_csv(RES / "fema_share_by_peril.csv", index_col=0)
counts = json.loads((RES / "cohort_counts.json").read_text())
agree = json.loads((RES / "spei_agreement.json").read_text())
ringdist = json.loads((RES / "ring_distances.json").read_text())

PANEL_ORDER = ["tornado", "hail_ice", "wind", "flood", "drought", "drought_spei"]


def peril_path(peril, outcome):
    if peril == "drought_spei":
        return paths_spei[paths_spei.outcome == outcome]
    return paths[(paths.peril == peril) & (paths.outcome == outcome)]


def peril_avg(peril, outcome):
    if peril == "drought_spei":
        return avg_post_lookup(tab6, outcome=outcome)
    return avg_post_lookup(tab2, peril=peril, outcome=outcome)


def peril_n(peril):
    if peril == "drought_spei":
        return int(tab6[tab6.outcome == "ln_pc_income"].n_cohorts.iloc[0])
    if peril == "hurricane":
        return counts["hurricane"]
    return counts[peril]


# Figure 1: cohort maps
def fig1_maps():
    import geopandas as gpd

    locs = pd.read_csv(RES / "cohort_locations.csv")
    gdf = gpd.read_file(DATA / "county_boundaries.geojson")
    gdf["fips"] = gdf["GEOID"].astype(int)
    conus_states = ~gdf["STATEFP"].isin(["02", "15", "60", "66", "69", "72", "78"])
    n_nonconus = int((~conus_states & gdf["fips"].isin(set(locs.fips))).any())
    gdf = gdf[conus_states].to_crs("EPSG:5070")
    states = gdf.dissolve(by="STATEFP")

    cnt = (locs.groupby(["spec", "fips"]).size().rename("n_years").reset_index())
    specs = ["tornado", "hurricane", "hail_ice", "wind", "flood", "drought", "drought_spei"]
    cmap = plt.get_cmap("YlGnBu")
    bin_colors = {1: cmap(0.35), 2: cmap(0.62), 3: cmap(0.92)}

    fig, axes = plt.subplots(3, 3, figsize=(7.4, 6.9))
    axes = axes.ravel()
    for ax, spec in zip(axes, specs):
        c = cnt[cnt.spec == spec].set_index("fips")["n_years"]
        d = gdf.copy()
        d["n_years"] = d["fips"].map(c)
        base = d[d["n_years"].isna()]
        base.plot(ax=ax, color="#F2F0EB", edgecolor="#C9C9C9", linewidth=0.12)
        for val, col in bin_colors.items():
            sel = d[d["n_years"] == val] if val < 3 else d[d["n_years"] >= 3]
            if len(sel):
                sel.plot(ax=ax, color=col, edgecolor="#C9C9C9", linewidth=0.12)
        states.plot(ax=ax, facecolor="none", edgecolor="#8A8A8A", linewidth=0.45)
        ax.set_title(f"{PERIL_LABELS[spec]} (n = {peril_n(spec)} cohorts)")
        ax.set_axis_off()
    # scale bar + north arrow on the first panel
    ax0 = axes[0]
    xmin, xmax = ax0.get_xlim(); ymin, ymax = ax0.get_ylim()
    x0 = xmin + 0.05 * (xmax - xmin); y0 = ymin + 0.055 * (ymax - ymin)
    for i, (xa, xb) in enumerate([(x0, x0 + 250_000), (x0 + 250_000, x0 + 500_000)]):
        ax0.add_patch(plt.Rectangle((xa, y0), xb - xa, 0.018 * (ymax - ymin),
                                    facecolor="black" if i == 0 else "white",
                                    edgecolor="black", lw=0.7, zorder=6))
    ax0.text(x0 - 40_000, y0 - 0.035 * (ymax - ymin), "0", ha="right", va="top", fontsize=7)
    ax0.text(x0 + 540_000, y0 - 0.035 * (ymax - ymin), "500 km", ha="left", va="top", fontsize=7)
    ax0.annotate("N", xy=(0.955, 0.97), xytext=(0.955, 0.80), xycoords="axes fraction",
                 ha="center", va="center", fontsize=9, fontweight="bold",
                 arrowprops=dict(arrowstyle="-|>", color="black", lw=1.1))
    # legend cell
    axl = axes[7]
    axl.set_axis_off()
    handles = [Patch(facecolor="#F2F0EB", edgecolor="#C9C9C9", label="No clean cohort"),
               Patch(facecolor=bin_colors[1], edgecolor="#C9C9C9", label="1 cohort year"),
               Patch(facecolor=bin_colors[2], edgecolor="#C9C9C9", label="2 cohort years"),
               Patch(facecolor=bin_colors[3], edgecolor="#C9C9C9", label="3+ cohort years")]
    leg = axl.legend(handles=handles, loc="upper left", frameon=False,
                     title="Cohort years per county", title_fontsize=9)
    leg.get_title().set_fontweight("bold")
    _ak = locs[locs.fips // 1000 == 2]
    _ak_by = _ak.groupby("spec").size().to_dict()
    _ak_txt = ", ".join(f"{int(v)} {PERIL_LABELS[k].lower()}" for k, v in sorted(_ak_by.items()))
    axl.text(0.0, 0.34,
             "Clean cohorts, 2004–2016: scaled shocks\n"
             "(≥0.125% of county income; drought by\n"
             "duration / SPEI-6 rule), exclusive of all\n"
             "other perils within ±3 years. Continental\n"
             f"US only: {len(_ak)} Alaska cohorts ({_ak_txt})\n"
             "are not shown.",
             transform=axl.transAxes, fontsize=8, va="top", color="#333333")
    axes[8].set_axis_off()
    header(fig, "Figure 1. Geography of the clean cohorts, by peril",
           "Source: Cohort locations, results/cohort_locations.csv (code/run_all.py). Boundaries: US Census Bureau\n"
           "cartographic boundary file cb_2023 (500k), Albers Equal Area (EPSG:5070). Scale bar and north arrow on first panel.")
    fig.tight_layout(rect=[0, 0.035, 1, 0.965])
    save(fig, "fig1_cohort_maps")


# Figures 2 and 3: event-study paths
def fig_paths(outcome, stem, title):
    fig, axes = plt.subplots(2, 3, figsize=(7.4, 5.2), sharey=True)
    for ax, peril in zip(axes.ravel(), PANEL_ORDER):
        plot_path(ax, peril_path(peril, outcome), PERIL_COLORS[peril])
        style_path_ax(ax)
        est, se = peril_avg(peril, outcome)
        ax.set_title(f"{PERIL_LABELS[peril]} (n = {peril_n(peril)})")
        ax.text(0.04, 0.94, f"Avg post {est:+.2f} ({se:.2f}) pp", transform=ax.transAxes,
                fontsize=8, va="top", ha="left",
                bbox=dict(facecolor="white", edgecolor="#CCCCCC", boxstyle="round,pad=0.3"))
    fig.text(0.012, 0.54, "Response (percentage points)", rotation=90, va="center", fontsize=9.5)
    fig.text(0.5, 0.102, "Years relative to shock (k = −1 omitted)", ha="center", fontsize=9.5)
    header(fig, title,
           "Notes: Stacked event studies (scaled + exclusive cohorts; stack×state×year effects; state-clustered SEs).\n"
           f"Shading: 95% confidence intervals. One estimate passes the Benjamini–Hochberg leg of the decision rule (smallest q = {tab2.p_bh.min():.3f}); none satisfies the full rule.\n"
           "Source: Author's estimates, results/paths_primary.csv and results/paths_spei.csv (SPEI-6 panel); averages from results/table2_primary_per_peril.csv\n"
           "and results/table6_spei_drought.csv.")
    fig.tight_layout(rect=[0.02, 0.135, 1, 0.95])
    save(fig, stem)


# Figure 4: onset contrast
def fig4_onset():
    fig = plt.figure(figsize=(7.4, 6.8))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.0], hspace=0.42, wspace=0.34,
                          width_ratios=[1.18, 1.0],
                          left=0.09, right=0.98, top=0.90, bottom=0.125)
    for j, outcome in enumerate(["ln_pc_income", "ln_employment"]):
        ax = fig.add_subplot(gs[0, j])
        d = paths_onset[(paths_onset.slow_def == "drought_only") & (paths_onset.outcome == outcome)]
        plot_path(ax, d[d.group == "fast"], BLUE, label=f"Fast pool (n = {counts['fast_pool']})")
        plot_path(ax, d[d.group == "slow"], VERM, label=f"Slow: drought only (n = {counts['drought']})")
        style_path_ax(ax, ylabel="Response (percentage points)" if j == 0 else None)
        ax.set_title(f"{OUTCOME_LABELS[outcome]} paths: fast vs slow")
        ax.set_xlabel("Years relative to shock (k = −1 omitted)")
        if j == 0:
            ax.legend(loc="lower left", frameon=False)
    # contrasts, raw and lead-adjusted
    ax = fig.add_subplot(gs[1, 0])
    rows = []
    for slow_def, lab in [("drought_only", "drought only"), ("drought_flood", "drought + flood")]:
        for outcome in ["ln_pc_income", "ln_employment"]:
            for la, ltag in [(False, "raw"), (True, "lead-adj.")]:
                r = tab3[(tab3.slow_def == slow_def) & (tab3.outcome == outcome)
                         & (tab3.lead_adjusted == la)].iloc[0]
                rows.append((f"{OUTCOME_LABELS[outcome]} ({lab}), {ltag}",
                             r.slow_minus_fast * PP, r.se * PP, la))
    ys = np.arange(len(rows))[::-1]
    for y, (lab, est, se, la) in zip(ys, rows):
        c = VERM if not la else BLUE
        m = "o" if not la else "s"
        ax.errorbar(est, y, xerr=1.96 * se, color=c, marker=m, ms=4.5, capsize=2.5, lw=1.2)
        ax.text(2.62, y, f"{est:+.2f} ({se:.2f})", va="center", ha="left", fontsize=7.8,
                bbox=dict(facecolor="white", edgecolor="none", pad=1.2))
    ax.axvline(0, color="#888888", lw=0.8)
    ax.set_yticks(ys, [r[0] for r in rows], fontsize=8)
    ax.set_xlim(-3.9, 4.4)
    ax.set_xlabel("Slow − fast contrast, average post (percentage points)")
    ax.set_title("Average-post contrasts (pp, SE in parentheses)")
    ax.grid(axis="x", color="#E3E3E3", lw=0.6)
    # smoothness bounds for the drought-only groups, income
    ax = fig.add_subplot(gs[1, 1])
    d = onset_grp[(onset_grp.slow_def == "drought_only") & (onset_grp.outcome == "ln_pc_income")]
    labs = {"fast": "Fast pool", "slow": "Slow (drought)"}
    ys = [1, 0]
    for y, gname in zip(ys, ["fast", "slow"]):
        r = d[d.group == gname].iloc[0]
        est, se = r.avg_post * PP, r.se * PP
        lo, hi = r.honest_lo_M01 * PP, r.honest_hi_M01 * PP
        c = BLUE if gname == "fast" else VERM
        if np.isfinite(lo):
            ax.barh(y, hi - lo, left=lo, height=0.34, color=c, alpha=0.30, zorder=2)
        ax.errorbar(est, y, xerr=1.96 * se, color=c, marker="o", ms=5, capsize=3, lw=1.4, zorder=3)
        ax.text(est, y + 0.44, f"breakdown M = {r.breakdown_M:.4f}",
                fontsize=7.8, va="bottom", ha="center", color="#333333")
    ax.axvline(0, color="#888888", lw=0.8)
    ax.set_yticks(ys, [labs["fast"], labs["slow"]])
    ax.set_ylim(-0.62, 1.95)
    ax.set_xlim(-7.6, 7.6)
    ax.set_xlabel("Income, average post (percentage points)")
    ax.set_title("Group averages: 95% CI (whisker) vs honest\nbounds at M = 0.01 (shaded bar)")
    ax.grid(axis="x", color="#E3E3E3", lw=0.6)
    _c = tab3[(tab3.slow_def == "drought_only") & (tab3.outcome == "ln_pc_income")]
    _craw = _c[~_c.lead_adjusted].iloc[0]; _cladj = _c[_c.lead_adjusted].iloc[0]
    header(fig, "Figure 4. Onset contrast: slow-onset (drought) versus fast-onset shocks",
           f"Notes: Headline (drought-only) lead-adjusted income contrast: {_cladj.slow_minus_fast * PP:+.2f} ({_cladj.se * PP:.2f}) pp, not significant; raw: {_craw.slow_minus_fast * PP:+.2f} ({_craw.se * PP:.2f}) pp. Honest bounds are\n"
           "conservative smoothness bounds in the spirit of Rambachan–Roth; at M = 0.0025 the slow-group bound is infeasible (reported as such).\n"
           "Source: Author's estimates, results/paths_onset_contrast.csv, results/table3_onset_contrast.csv, results/onset_group_summary.csv.")
    save(fig, "fig4_onset_contrast")


# Figure 5: spillover rings
def fig5_rings():
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 4.6), sharey=True)
    row_specs = [("fast", "hit", "Fast — hit counties"), ("fast", "ring1", "Fast — ring-1 neighbours"),
                 ("drought", "hit", "Drought — hit counties"), ("drought", "ring1", "Drought — ring-1 neighbours")]
    xlims = {"ln_pc_income": (-1.8, 1.45), "ln_employment": (-2.1, 2.1)}
    for ax, outcome in zip(axes, ["ln_pc_income", "ln_employment"]):
        d = tab4[tab4.outcome == outcome]
        ys = np.arange(len(row_specs))[::-1]
        for y, (pool, grp, lab) in zip(ys, row_specs):
            r = d[(d.pool == pool) & (d.group == grp)].iloc[0]
            est, se, mde = r.avg_post * PP, r.se * PP, r.mde80 * PP
            ax.barh(y, 2 * mde, left=-mde, height=0.6, color="#9E9E9E", alpha=0.20, zorder=1)
            c = BLUE if grp == "hit" else VERM
            ax.errorbar(est, y, xerr=1.96 * se, color=c, marker="o", ms=5, capsize=3, lw=1.4, zorder=3)
            ax.text(est, y + 0.42, f"{est:+.2f} ({se:.2f})", ha="center", va="bottom", fontsize=7.8)
        ax.axvline(0, color="#888888", lw=0.8)
        ax.set_yticks(ys, [f"{lab} (n = {int(tab4[(tab4.pool == p) & (tab4.group == g) & (tab4.outcome == outcome)].n_units.iloc[0])})"
                           for p, g, lab in row_specs], fontsize=8)
        ax.set_title(f"{OUTCOME_LABELS[outcome]}: hit vs ring-1")
        ax.grid(axis="x", color="#E3E3E3", lw=0.6)
        ax.set_ylim(-0.7, 4.15)
        ax.set_xlim(*xlims[outcome])
    axes[0].legend(handles=[Patch(facecolor="#9E9E9E", alpha=0.35, label="±MDE band (80% power)")],
                   loc="lower right", frameon=False)
    fig.text(0.5, 0.105, "Average post effect (percentage points)", ha="center", fontsize=9.5)
    _fr = tab4[(tab4.pool == "fast") & (tab4.group == "ring1") & (tab4.outcome == "ln_pc_income")].iloc[0]
    _dr = tab4[(tab4.pool == "drought") & (tab4.group == "ring1")]
    _drn = int(_dr[_dr.outcome == "ln_pc_income"].n_units.iloc[0])
    _drm = _dr[_dr.outcome == "ln_employment"].iloc[0]
    header(fig, "Figure 5. Spillover rings: hit counties versus adjacent (ring-1) counties",
           f"Notes: Stacked ring design; ring-1 median centroid distance {ringdist['median_km']:.1f} km (IQR {ringdist['p25_km']:.0f}–{ringdist['p75_km']:.0f} km). Fast ring-1 income: {_fr.avg_post * PP:+.2f} ({_fr.se * PP:.2f}) pp,\n"
           f"MDE {_fr.mde80 * PP:.2f} pp — no detectable spillover at annual frequency; the drought ring ({_drn} ring-1 units) is underpowered (employment MDE {_drm.mde80 * PP:.2f} pp).\n"
           "Source: Author's estimates, results/table4_rings_stacked.csv and results/ring_distances.json.")
    fig.tight_layout(rect=[0, 0.15, 1, 0.93])
    save(fig, "fig5_spillover_rings")


# Figure 6: declaration split
def fig6_fema():
    fig = plt.figure(figsize=(7.4, 4.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.55, 1.0], wspace=0.42,
                          left=0.10, right=0.97, top=0.88, bottom=0.16)
    ax = fig.add_subplot(gs[0, 0])
    rows = []
    for outcome in ["ln_pc_income", "ln_employment"]:
        for grp, lab in [("declared", "Declared"), ("undeclared", "Undeclared"),
                         ("declared_minus_undeclared", "Difference")]:
            r = tab5[(tab5.outcome == outcome) & (tab5.group == grp)].iloc[0]
            rows.append((f"{OUTCOME_LABELS[outcome]} — {lab} (n = {int(r.n_cohorts)})",
                         r.avg_post * PP, r.se * PP, grp == "declared_minus_undeclared"))
    ys = np.arange(len(rows))[::-1]
    for y, (lab, est, se, isdiff) in zip(ys, rows):
        c = VERM if isdiff else BLUE
        ax.errorbar(est, y, xerr=1.96 * se, color=c, marker="D" if isdiff else "o",
                    ms=5 if isdiff else 4.5, capsize=2.5, lw=1.3)
        ax.text(1.95, y, f"{est:+.2f} ({se:.2f})", va="center", ha="left", fontsize=7.8)
    ax.axhline(2.5, color="#CCCCCC", lw=0.8)
    ax.axvline(0, color="#888888", lw=0.8)
    ax.set_yticks(ys, [r[0] for r in rows], fontsize=8)
    ax.set_xlim(-2.6, 3.2)
    ax.set_xlabel("Average post effect (percentage points)")
    ax.set_title("Declared vs undeclared shocks (pp, SE in parentheses)")
    ax.grid(axis="x", color="#E3E3E3", lw=0.6)
    ax2 = fig.add_subplot(gs[0, 1])
    order = ["hurricane", "flood", "tornado", "wind", "hail_ice", "drought"]
    shares = [float(fema_share.loc[p, "share_declared"]) * 100 for p in order]
    y2 = np.arange(len(order))[::-1]
    ax2.barh(y2, shares, color=BLUE, alpha=0.75, height=0.6)
    for y, s in zip(y2, shares):
        ax2.text(s + 2, y, f"{s:.0f}%", va="center", fontsize=8.5)
    ax2.set_yticks(y2, [PERIL_LABELS[p] for p in order], fontsize=8.5)
    ax2.set_xlim(0, 112)
    ax2.set_xlabel("Share of cohorts declared (%)")
    ax2.set_title("Declaration share by peril")
    ax2.grid(axis="x", color="#E3E3E3", lw=0.6)
    _fd = tab5[(tab5.outcome == "ln_pc_income") & (tab5.group == "declared_minus_undeclared")].iloc[0]
    header(fig, "Figure 6. Declaration split: declared versus undeclared shocks",
           f"Notes: Declared − undeclared income difference: {_fd.avg_post * PP:+.2f} ({_fd.se * PP:.2f}) pp, not significant; the split is peril-confounded ({float(fema_share.loc['hurricane', 'share_declared']) * 100:.0f}% of hurricane cohorts\n"
           f"declared, {float(fema_share.loc['drought', 'share_declared']) * 100:.0f}% of drought cohorts), so it does not isolate aid. Source: Author's estimates, results/table5_fema_split.csv;\n"
           "declaration shares, results/fema_share_by_peril.csv (OpenFEMA DisasterDeclarationsSummaries).")
    save(fig, "fig6_declaration_split")


# Figure 7: drought regimes
def fig7_regimes():
    fig = plt.figure(figsize=(7.4, 5.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.3, 1.0], wspace=0.26,
                          left=0.09, right=0.97, top=0.88, bottom=0.185)
    ax = fig.add_subplot(gs[0, 0])
    plot_path(ax, peril_path("drought", "ln_pc_income"), VERM,
              label=f"Reporting drought (n = {counts['drought']})")
    plot_path(ax, peril_path("drought_spei", "ln_pc_income"), BLUE,
              label=f"SPEI-6 drought (n = {counts['spei_drought']})")
    style_path_ax(ax, ylabel="Response (percentage points)")
    ax.set_title("Income paths under the two drought regimes")
    ax.set_xlabel("Years relative to shock (k = −1 omitted)")
    ax.legend(loc="lower left", frameon=False)
    ax2 = fig.add_subplot(gs[0, 1])
    rows = []
    for outcome in ["ln_pc_income", "ln_employment", "ln_population"]:
        for peril, lab in [("drought", "Reporting"), ("drought_spei", "SPEI-6")]:
            est, se = peril_avg(peril, outcome)
            rows.append((f"{OUTCOME_LABELS[outcome]} — {lab}", est, se, peril))
    ys = np.arange(len(rows))[::-1]
    for y, (lab, est, se, peril) in zip(ys, rows):
        c = VERM if peril == "drought" else BLUE
        ax2.errorbar(est, y, xerr=1.96 * se, color=c, marker="o", ms=4.5, capsize=2.5, lw=1.3)
        ax2.text(1.72, y, f"{est:+.2f} ({se:.2f})", va="center", ha="left", fontsize=7.8)
    ax2.axhline(3.5, color="#CCCCCC", lw=0.8); ax2.axhline(1.5, color="#CCCCCC", lw=0.8)
    ax2.axvline(0, color="#888888", lw=0.8)
    ax2.set_yticks(ys, [r[0] for r in rows], fontsize=8)
    ax2.set_xlim(-1.95, 2.85)
    ax2.set_xlabel("Average post effect (percentage points)")
    ax2.set_title("Average post effects (pp, SE in parentheses)")
    ax2.grid(axis="x", color="#E3E3E3", lw=0.6)
    _re, _rse = peril_avg("drought", "ln_pc_income")
    _pe, _pse = peril_avg("drought_spei", "ln_pc_income")
    _pmde = float(tab6[tab6.outcome == "ln_pc_income"].mde80.iloc[0]) * PP
    _excl = "excludes" if _pe - 1.96 * _pse > _re or _pe + 1.96 * _pse < _re else "does not exclude"
    header(fig, "Figure 7. The regime contrast: reporting-based versus physical (SPEI-6) drought",
           f"Notes: Reporting drought income {_re:+.2f} ({_rse:.2f}) pp vs SPEI-6 {_pe:+.2f} ({_pse:.2f}) pp; the SPEI estimate (MDE {_pmde:.2f} pp) {_excl} the reporting estimate.\n"
           f"The two rules agree on only {agree['both_cy']:,} of shock county-years (reporting {agree['reporting_shock_cy']:,}; SPEI {agree['spei_shock_cy']:,}; Jaccard {agree['jaccard']:.2f}).\n"
           "Source: Author's estimates, results/paths_primary.csv, results/paths_spei.csv, results/table2_primary_per_peril.csv,\n"
           "results/table6_spei_drought.csv, results/spei_agreement.json.")
    save(fig, "fig7_drought_regimes")


if __name__ == "__main__":
    fig1_maps()
    fig_paths("ln_pc_income", "fig2_income_paths",
              "Figure 2. Income event-study paths by peril")
    fig_paths("ln_employment", "fig3_employment_paths",
              "Figure 3. Employment event-study paths by peril")
    fig4_onset()
    fig5_rings()
    fig6_fema()
    fig7_regimes()
    print("all figures written")
