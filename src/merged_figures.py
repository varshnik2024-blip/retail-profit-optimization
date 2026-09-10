"""
Two merged four-panel figures, replacing the previous figures 2-5.

Merging halves the number of double-column floats and removes two captions,
which is what lets the paper hold all of this evidence inside six pages. The
panel content is unchanged; only titles are shortened to fit the narrower axes.

Run from the project root with src/ on PYTHONPATH.
"""
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import OUTPUTS_DIR, PROCESSED_DIR

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 7,
    "axes.labelsize": 6.6,
    "axes.titlesize": 7.0,
    "xtick.labelsize": 6.0,
    "ytick.labelsize": 6.0,
    "legend.fontsize": 5.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.5,
    "lines.linewidth": 1.1,
    "figure.dpi": 600,
})

COL2 = 7.16
C_BASE, C_MID, C_OURS = "#9aa5b1", "#5b7c99", "#1f4e79"
C_TRUE, C_ACC = "#c0392b", "#e08214"


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(f"{name}.{ext}", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"  wrote {name}.pdf / .png")


# =====================================================================
# Figure 2 - estimator validation (elasticity + uplift)
# =====================================================================
def fig_validation():
    syn = pd.read_csv(OUTPUTS_DIR / "elasticity_synthetic.csv")
    real = pd.read_csv(OUTPUTS_DIR / "elasticity_real.csv")
    truth = pd.read_csv(PROCESSED_DIR / "ground_truth_products.csv")
    m = syn.merge(truth, on="upc")

    us = pd.read_csv(OUTPUTS_DIR / "uplift_summary_synthetic.csv")
    ur = pd.read_csv(OUTPUTS_DIR / "uplift_summary_real.csv")
    tl = pd.read_csv(PROCESSED_DIR / "ground_truth_lift.csv").iloc[0]
    names = ["feature", "display", "tpr_only"]
    nice = ["mailer", "display", "price\ncut"]
    us = us.set_index("treatment").loc[names]
    ur = ur.set_index("treatment").loc[names]

    fig, ax = plt.subplots(1, 4, figsize=(COL2, 1.52))
    a, b, c, d = ax
    fig.subplots_adjust(wspace=0.44)

    # (a) elasticity recovery
    lim = [-6.2, 0.2]
    a.plot(lim, lim, color="#7f8c8d", lw=0.7, ls="--", zorder=1)
    a.scatter(m.true_elasticity, m.elast_naive, s=7, c=C_BASE,
              edgecolor="none", label="naive OLS", zorder=2)
    a.scatter(m.true_elasticity, m.elast_final, s=7, c=C_OURS, marker="^",
              edgecolor="none", label="2SLS + shrink", zorder=3)
    a.set_xlim(lim); a.set_ylim(lim)
    a.set_xlabel("true elasticity")
    a.set_ylabel("estimated")
    a.set_title("(a) Elasticity recovery")
    a.legend(loc="upper left", frameon=False, handletextpad=0.3,
             borderpad=0.1, labelspacing=0.25)

    # (b) impossible estimates
    labels = ["naive", "store\nFE", "2SLS", "2SLS\n+EB"]
    cols = ["elast_naive", "elast_fe", "elast_iv", "elast_final"]
    wrong = [(real[c_].dropna() > 0).mean() * 100 for c_ in cols]
    b.bar(range(4), wrong, color=[C_BASE, C_MID, C_MID, C_OURS], width=0.62)
    for i, v in enumerate(wrong):
        b.text(i, v + 0.12, f"{v:.1f}", ha="center", fontsize=5.8)
    b.set_xticks(range(4)); b.set_xticklabels(labels)
    b.set_ylabel("positive elasticity (%)")
    b.set_title("(b) Impossible, real")
    b.set_ylim(0, max(max(wrong) * 1.4, 1.0))

    # (c) synthetic uplift
    x = np.arange(3); w = 0.26
    c.bar(x - w, [tl[nm] for nm in names], w, label="truth", color=C_TRUE)
    c.bar(x, us["naive_coef"], w, label="naive", color=C_BASE)
    c.bar(x + w, us["did_coef"], w, label="DiD", color=C_OURS)
    c.set_xticks(x); c.set_xticklabels(nice)
    c.set_ylabel("log-lift")
    c.set_title("(c) Uplift, synthetic")
    c.legend(frameon=False, loc="upper right", handletextpad=0.3,
             borderpad=0.1, labelspacing=0.25)

    # (d) real uplift
    d.bar(x - w / 2, ur["naive_coef"], w, label="naive", color=C_BASE)
    d.bar(x + w / 2, ur["did_coef"], w, label="DiD", color=C_OURS)
    d.axhline(0, color="#33404d", lw=0.6)
    for i, v in enumerate(ur["did_coef"]):
        d.text(i + w / 2, v + (0.04 if v >= 0 else 0.05), f"{v:.2f}",
               ha="center", fontsize=5.8)
    d.set_ylim(min(0, ur["did_coef"].min()) - 0.08, None)
    d.set_xticks(x); d.set_xticklabels(nice)
    d.set_ylabel("log-lift")
    d.set_title("(d) Uplift, real")
    d.legend(frameon=False, loc="upper right", handletextpad=0.3,
             borderpad=0.1, labelspacing=0.25)

    save(fig, "fig2_validation")


# =====================================================================
# Figure 3 - decision modules on held-out weeks
# =====================================================================
def fig_decisions():
    fm = pd.read_csv(OUTPUTS_DIR / "forecast_metrics_real.csv")
    t = fm[fm["split"].eq("test")] if "split" in fm.columns else fm
    order = ["naive (last week)", "seasonal naive (52w)", "moving avg 4w",
             "gradient boosting (test)"]
    t = t.set_index("model").reindex(order).dropna(subset=["WAPE"])

    fc = pd.read_csv(OUTPUTS_DIR / "forecast_test_real.csv")
    cat = (fc.groupby("category")
           .apply(lambda x: np.abs(x.units - x.pred).sum() / x.units.sum())
           .sort_values())

    md = pd.read_csv(OUTPUTS_DIR / "markdown_policy_real.csv")

    fig, ax = plt.subplots(1, 4, figsize=(COL2, 1.52))
    a, b, c, d = ax
    fig.subplots_adjust(wspace=0.50)

    # (a) WAPE vs baselines
    lbl = ["last\nweek", "seas.\nnaive", "mov.\navg", "grad.\nboost"]
    cols = [C_BASE, C_BASE, C_BASE, C_OURS]
    a.bar(range(len(t)), t["WAPE"], color=cols[:len(t)], width=0.62)
    for i, v in enumerate(t["WAPE"]):
        a.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=5.6)
    a.set_xticks(range(len(t))); a.set_xticklabels(lbl[:len(t)])
    a.set_ylabel("WAPE, test weeks")
    a.set_title("(a) Forecast vs naive")
    a.set_ylim(0, t["WAPE"].max() * 1.28)

    # (b) WAPE by category
    b.barh(range(len(cat)), cat.values, color=C_MID, height=0.6)
    b.set_yticks(range(len(cat)))
    b.set_yticklabels([x.title().replace(" ", "\n") for x in cat.index],
                      fontsize=5.4)
    for i, v in enumerate(cat.values):
        b.text(v + 0.008, i, f"{v:.3f}", va="center", fontsize=5.6)
    b.set_xlabel("WAPE")
    b.set_title("(b) By category")
    b.set_xlim(0, cat.max() * 1.32)
    b.grid(axis="y", alpha=0)

    # (c) clearance policies
    keys = ["never", "panic", "fixed", "dp"]
    ml = ["never", "panic\n-40%", "fixed\nladder", "DP"]
    cash = [md[f"{k}_profit"].mean() for k in keys]
    sell = [md[f"{k}_sellthrough"].mean() * 100 for k in keys]
    c.bar(range(4), cash, color=[C_BASE, C_BASE, C_MID, C_OURS], width=0.62)
    c.set_xticks(range(4)); c.set_xticklabels(ml)
    c.set_ylabel("cash recovered")
    c.set_title("(c) Clearance policy")
    c.set_ylim(0, max(cash) * 1.3)
    c2 = c.twinx()
    c2.plot(range(4), sell, color=C_ACC, marker="o", ms=2.6, lw=0.9)
    c2.text(0.04, 0.90, "sell-through", transform=c2.transAxes,
            color=C_ACC, fontsize=5.8, ha="left", va="center")
    c2.tick_params(axis="y", labelcolor=C_ACC, labelsize=5.6)
    c2.set_ylim(0, 112); c2.grid(False)
    c2.spines["right"].set_visible(True); c2.spines["right"].set_color(C_ACC)

    # (d) optimal ladder, averaged over all products along the expected
    # inventory path. One example product is anecdote; the mean across the
    # catalogue is what supports the claim that optimal depth eases.
    import markdown as mdmod
    H = mdmod.HORIZON_WEEKS
    acc = np.zeros(H)
    for _, r in md.iterrows():
        # signature: (demand, price, cost, elasticity, inventory)
        pol, _, _ = mdmod.solve_markdown_dp(
            r.weekly_demand, r.base_price, r.unit_cost,
            r.elasticity, int(r.start_inventory))
        I = min(int(r.start_inventory), pol.shape[1] - 1)
        for tt in range(H):
            disc = float(mdmod.DISCOUNTS[pol[tt, I]])
            acc[tt] += disc
            mu = r.weekly_demand * (1 - disc) ** r.elasticity
            I = max(0, int(round(I - min(mu, I))))
    mean_ladder = acc / len(md) * 100

    fixed = np.zeros(H)
    fixed[H // 3:] = 20.0
    fixed[2 * H // 3:] = 40.0
    wk = np.arange(1, H + 1)
    d.step(wk, fixed, where="mid", color=C_BASE, ls="--", label="fixed")
    d.plot(wk, mean_ladder, color=C_OURS, marker="o", ms=2.4, label="DP mean")
    d.set_xticks(wk[::2])
    d.set_xlabel("clearance week")
    d.set_ylabel("discount (%)")
    d.set_title("(d) Optimal depth eases")
    d.set_ylim(0, 46)
    d.legend(frameon=False, loc="upper left", handletextpad=0.3,
             borderpad=0.1, labelspacing=0.25)

    save(fig, "fig3_decisions")


if __name__ == "__main__":
    print("merged figures ->")
    fig_validation()
    fig_decisions()
