"""
Publication figures for the IEEE conference paper.

Reads only the CSVs already written by phases 3-8 and renders five figures at
600 dpi, in a serif face at 7-8 pt to match the IEEE template. Nothing is
recomputed here, so every number in the paper traces back to a pipeline run.

Outputs land in outputs/figures/ as both PDF (vector, for LaTeX) and PNG.

Run:
    python src/make_figures.py
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

from config import OUTPUTS_DIR, PROCESSED_DIR

warnings.filterwarnings("ignore")

FIG_DIR = OUTPUTS_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8.5,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.5,
    "lines.linewidth": 1.2,
    "figure.dpi": 600,
})

COL1 = 3.45      # single IEEE column, inches
COL2 = 7.16      # full two-column width

C_BASE = "#9aa5b1"     # baselines / naive
C_MID = "#5b7c99"      # intermediate method
C_OURS = "#1f4e79"     # our method
C_TRUE = "#c0392b"     # ground truth
C_ACC = "#e08214"      # accent


def save(fig, name, tight=True):
    for ext in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"{name}.{ext}",
                    bbox_inches="tight" if tight else None, pad_inches=0.02)
    plt.close(fig)
    print(f"    wrote {name}.pdf / .png")


# ==========================================================================
# Fig 1 - system architecture
# ==========================================================================
def fig_architecture():
    """
    Layer-and-rail layout: every inter-layer link runs through a horizontal
    rail rather than a direct diagonal. Diagonals between nine boxes produce
    unreadable crossings; a rail keeps the reading order strictly top-down.
    """
    fig, ax = plt.subplots(figsize=(COL2, 3.4))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 66)
    ax.axis("off")

    RAIL = "#33404d"

    def box(x, y, w, h, text, fc, ec, fs=7.0, bold=False):
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.35,rounding_size=1.2",
            linewidth=0.8, facecolor=fc, edgecolor=ec, zorder=3))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fs, zorder=4,
                fontweight="bold" if bold else "normal", linespacing=1.4)

    def vline(x, y1, y2, head=False):
        ax.add_patch(FancyArrowPatch(
            (x, y1), (x, y2), arrowstyle="-|>" if head else "-",
            mutation_scale=7, linewidth=0.7, color=RAIL,
            shrinkA=0, shrinkB=0, zorder=1))

    def hline(x1, x2, y):
        ax.plot([x1, x2], [y, y], color=RAIL, lw=0.7, zorder=1)

    def layer(y, txt):
        ax.text(1.2, y, txt, fontsize=6.4, color="#7a8894",
                fontweight="bold", va="center", ha="center", rotation=90)

    # ---------------------------------------------------------------- data
    layer(57.5, "DATA")
    box(7, 53.5, 26, 8, "dunnhumby weekly panel\n524,926 store-week-UPC rows",
        "#eef2f6", RAIL, bold=True)
    box(37, 53.5, 26, 8, "Schema contract and loader\ncleaning, cost derivation",
        "#eef2f6", RAIL)
    box(67, 53.5, 26, 8, "Temporal split\ntrain / validation / test",
        "#eef2f6", RAIL)
    ax.add_patch(FancyArrowPatch((33, 57.5), (37, 57.5), arrowstyle="-|>",
                 mutation_scale=7, lw=0.7, color=RAIL, zorder=2))
    ax.add_patch(FancyArrowPatch((63, 57.5), (67, 57.5), arrowstyle="-|>",
                 mutation_scale=7, lw=0.7, color=RAIL, zorder=2))

    # rail 1: data -> estimation
    vline(80, 53.5, 49.5)
    hline(20, 80, 49.5)
    for x in (20, 50, 80):
        vline(x, 49.5, 45.5, head=True)

    # ---------------------------------------------------------- estimation
    layer(41.5, "ESTIMATION")
    box(7, 37.5, 26, 8, "Price elasticity\nHausman 2SLS + shrinkage",
        "#dce7f0", "#1f4e79", fs=6.8)
    box(37, 37.5, 26, 8, "Promotion uplift\ntwo-way fixed-effects DiD",
        "#dce7f0", "#1f4e79", fs=6.8)
    box(67, 37.5, 26, 8, "Demand forecast\ngradient boosting + smearing",
        "#dce7f0", "#1f4e79", fs=6.8)

    # rail 2: estimation -> decision
    for x in (20, 50, 80):
        vline(x, 37.5, 33.5)
    hline(20, 80, 33.5)
    for x in (20, 50, 80):
        vline(x, 33.5, 29.5, head=True)

    # ------------------------------------------------------------ decision
    layer(25.5, "DECISION")
    box(7, 21.5, 26, 8, "Price optimisation\nLerner rule + guardrails",
        "#cfe0d5", "#256b46", fs=6.8)
    box(37, 21.5, 26, 8, "Markdown schedule\nbackward induction (DP)",
        "#cfe0d5", "#256b46", fs=6.8)
    box(67, 21.5, 26, 8, "Replenishment\nforecast-driven order-up-to",
        "#cfe0d5", "#256b46", fs=6.8)

    # rail 3: decision -> evaluation
    for x in (20, 50, 80):
        vline(x, 21.5, 17.5)
    hline(20, 80, 17.5)
    vline(50, 17.5, 13.5, head=True)

    # ---------------------------------------------------------- evaluation
    layer(9.5, "EVALUATION")
    box(15, 5.5, 70, 8,
        "Profit and cost impact measured on held-out test weeks\n"
        "benchmarked against naive, textbook and oracle policies",
        "#f7ead9", "#8a5a1a", fs=7.0, bold=True)

    # ------------------------------------------------- validation envelope
    ax.add_patch(FancyBboxPatch(
        (4.2, 1.4), 92.5, 63.0, boxstyle="round,pad=0.2,rounding_size=1.5",
        linewidth=0.7, facecolor="none", edgecolor="#b0bac4",
        linestyle=(0, (4, 3)), zorder=0))
    ax.text(50, 2.9, "Synthetic replica with known ground-truth elasticity and "
                     "promotion lift validates every estimator",
            ha="center", va="center", fontsize=6.6, style="italic",
            color="#6b7783")
    save(fig, "fig1_architecture")


# ==========================================================================
# Fig 2 - elasticity estimator validation
# ==========================================================================
def fig_elasticity():
    syn = pd.read_csv(OUTPUTS_DIR / "elasticity_synthetic.csv")
    real = pd.read_csv(OUTPUTS_DIR / "elasticity_real.csv")
    truth = pd.read_csv(PROCESSED_DIR / "ground_truth_products.csv")
    m = syn.merge(truth, on="upc")

    fig, (a, b) = plt.subplots(1, 2, figsize=(COL2, 2.5))

    lim = [-6.2, 0.2]
    a.plot(lim, lim, color="#7f8c8d", lw=0.7, ls="--", zorder=1,
           label="perfect recovery")
    a.scatter(m.true_elasticity, m.elast_naive, s=13, c=C_BASE,
              edgecolor="none", label="naive OLS", zorder=2)
    a.scatter(m.true_elasticity, m.elast_final, s=13, c=C_OURS,
              marker="^", edgecolor="none", label="2SLS + shrinkage", zorder=3)
    a.set_xlim(lim)
    a.set_ylim(lim)
    a.set_xlabel("true elasticity (synthetic)")
    a.set_ylabel("estimated elasticity")
    a.set_title("(a) Recovery of known elasticity")
    a.legend(loc="upper left", frameon=False)

    labels = ["naive\nOLS", "store FE\n+ controls", "Hausman\n2SLS",
              "2SLS + EB\nshrinkage"]
    cols = ["elast_naive", "elast_fe", "elast_iv", "elast_final"]
    wrong = [(real[c].dropna() > 0).mean() * 100 for c in cols]
    b.bar(range(4), wrong, color=[C_BASE, C_MID, C_MID, C_OURS], width=0.62)
    for i, v in enumerate(wrong):
        b.text(i, v + 0.12, f"{v:.1f}%", ha="center", fontsize=7)
    b.set_xticks(range(4))
    b.set_xticklabels(labels, fontsize=6.5)
    b.set_ylabel("products with positive elasticity (%)")
    b.set_title("(b) Economically impossible estimates, real data")
    b.set_ylim(0, max(max(wrong) * 1.35, 1.0))
    save(fig, "fig2_elasticity")


# ==========================================================================
# Fig 3 - promotion uplift validation
# ==========================================================================
def fig_uplift():
    syn = pd.read_csv(OUTPUTS_DIR / "uplift_summary_synthetic.csv")
    real = pd.read_csv(OUTPUTS_DIR / "uplift_summary_real.csv")
    truth = pd.read_csv(PROCESSED_DIR / "ground_truth_lift.csv").iloc[0]

    names = ["feature", "display", "tpr_only"]
    nice = ["FEATURE\n(mailer)", "DISPLAY\n(in-store)", "TPR_ONLY\n(price cut)"]
    syn = syn.set_index("treatment").loc[names]
    real = real.set_index("treatment").loc[names]

    fig, (a, b) = plt.subplots(1, 2, figsize=(COL2, 2.4))
    x = np.arange(3)
    w = 0.26

    a.bar(x - w, [truth[n] for n in names], w, label="ground truth",
          color=C_TRUE, edgecolor="none")
    a.bar(x, syn["naive_coef"], w, label="naive comparison",
          color=C_BASE, edgecolor="none")
    a.bar(x + w, syn["did_coef"], w, label="two-way FE DiD",
          color=C_OURS, edgecolor="none")
    a.set_xticks(x)
    a.set_xticklabels(nice, fontsize=6.5)
    a.set_ylabel("log-lift coefficient")
    a.set_title("(a) Synthetic data with known lift")
    a.legend(frameon=False, loc="upper right")

    b.bar(x - w / 2, real["naive_coef"], w, label="naive comparison",
          color=C_BASE, edgecolor="none")
    b.bar(x + w / 2, real["did_coef"], w, label="two-way FE DiD",
          color=C_OURS, edgecolor="none")
    b.axhline(0, color="#33404d", lw=0.6)
    for i, v in enumerate(real["did_coef"]):
        # a near-zero bar has no room below it before the tick labels, so
        # annotate above the axis line instead of under the bar
        off = 0.04 if v >= 0 else 0.05
        b.text(i + w / 2, v + off, f"{v:.2f}", ha="center", fontsize=6.6)
    b.set_ylim(min(0, real["did_coef"].min()) - 0.06, None)
    b.set_xticks(x)
    b.set_xticklabels(nice, fontsize=6.5)
    b.set_ylabel("log-lift coefficient")
    b.set_title("(b) Real data")
    b.legend(frameon=False, loc="upper right")
    save(fig, "fig3_uplift")


# ==========================================================================
# Fig 4 - forecast accuracy
# ==========================================================================
def fig_forecast():
    real = pd.read_csv(OUTPUTS_DIR / "forecast_metrics_real.csv")
    t = real[real["split"].eq("test")] if "split" in real.columns else real
    order = ["naive (last week)", "seasonal naive (52w)", "moving avg 4w",
             "gradient boosting (test)"]
    t = t.set_index("model").reindex(order).dropna(subset=["WAPE"])

    fc = pd.read_csv(OUTPUTS_DIR / "forecast_test_real.csv")
    cat = (fc.groupby("category")
           .apply(lambda d: np.abs(d.units - d.pred).sum() / d.units.sum())
           .sort_values())

    fig, (a, b) = plt.subplots(1, 2, figsize=(COL2, 2.3))

    lbl = ["naive\n(last week)", "seasonal\nnaive (52w)", "moving\navg 4w",
           "gradient\nboosting"]
    cols = [C_BASE, C_BASE, C_BASE, C_OURS]
    a.bar(range(len(t)), t["WAPE"], color=cols[:len(t)], width=0.62)
    for i, v in enumerate(t["WAPE"]):
        a.text(i, v + 0.008, f"{v:.3f}", ha="center", fontsize=7)
    a.set_xticks(range(len(t)))
    a.set_xticklabels(lbl[:len(t)], fontsize=6.5)
    a.set_ylabel("WAPE on test weeks")
    a.set_title("(a) Model vs naive baselines")
    a.set_ylim(0, t["WAPE"].max() * 1.22)

    b.barh(range(len(cat)), cat.values, color=C_MID, height=0.6)
    b.set_yticks(range(len(cat)))
    b.set_yticklabels([c.title() for c in cat.index], fontsize=6.5)
    for i, v in enumerate(cat.values):
        b.text(v + 0.006, i, f"{v:.3f}", va="center", fontsize=6.8)
    b.set_xlabel("WAPE")
    b.set_title("(b) Accuracy by category")
    b.set_xlim(0, cat.max() * 1.25)
    b.grid(axis="y", alpha=0)
    save(fig, "fig4_forecast")


# ==========================================================================
# Fig 5 - markdown policy
# ==========================================================================
def fig_markdown():
    md = pd.read_csv(OUTPUTS_DIR / "markdown_policy_real.csv")

    fig, (a, b) = plt.subplots(1, 2, figsize=(COL2, 2.4))
    # the right-hand twin axis on panel (a) needs its own label margin,
    # otherwise it lands on top of panel (b)'s y-label
    fig.subplots_adjust(wspace=0.42)

    keys = ["never", "panic", "fixed", "dp"]
    lbl = ["never\nmark down", "panic\n-40% now", "fixed\n-20%/-40%", "DP\noptimal"]
    cash = [md[f"{k}_profit"].mean() for k in keys]
    sell = [md[f"{k}_sellthrough"].mean() * 100 for k in keys]
    cols = [C_BASE, C_BASE, C_MID, C_OURS]

    a.bar(range(4), cash, color=cols, width=0.62)
    for i, v in enumerate(cash):
        a.text(i, v + max(cash) * 0.02, f"{v:.0f}", ha="center", fontsize=7)
    a.set_xticks(range(4))
    a.set_xticklabels(lbl, fontsize=6.5)
    a.set_ylabel("mean cash recovered per product")
    a.set_title("(a) Clearance policy comparison")
    a.set_ylim(0, max(cash) * 1.25)

    a2 = a.twinx()
    a2.plot(range(4), sell, color=C_ACC, marker="o", ms=3.2, lw=1.0)
    a2.set_ylabel("sell-through (%)", color=C_ACC, fontsize=7, labelpad=1)
    a2.tick_params(axis="y", labelcolor=C_ACC, labelsize=6.5)
    a2.set_ylim(0, 112)
    a2.grid(False)
    a2.spines["right"].set_visible(True)
    a2.spines["right"].set_color(C_ACC)

    # ladder for the product with the largest gain over the fixed rule,
    # regenerated from the saved policy inputs
    ex = md.nlargest(1, "gain_vs_fixed").iloc[0]
    import markdown as mdmod
    pol, _, _ = mdmod.solve_markdown_dp(
        ex.weekly_demand, ex.base_price, ex.unit_cost,
        ex.elasticity, int(ex.start_inventory))
    inv = int(ex.start_inventory)
    ladder = []
    for tt in range(mdmod.HORIZON_WEEKS):
        d = mdmod.DISCOUNTS[pol[tt, min(inv, pol.shape[1] - 1)]]
        ladder.append(d)
        mu = ex.weekly_demand * ((ex.base_price * (1 - d)) / ex.base_price) ** ex.elasticity
        inv = max(0, inv - int(round(mu)))

    weeks = np.arange(1, mdmod.HORIZON_WEEKS + 1)
    fixed = [0, 0, 0, 0.20, 0.20, 0.20, 0.40, 0.40]
    b.step(weeks, np.array(fixed) * 100, where="mid", color=C_MID,
           ls="--", label="fixed schedule")
    b.step(weeks, np.array(ladder) * 100, where="mid", color=C_OURS,
           label="DP optimal")
    b.set_xlabel("week of clearance window")
    b.set_ylabel("discount depth (%)")
    b.set_title(f"(b) Ladder, {str(ex.description)[:18].strip()}")
    b.set_xticks(weeks)
    b.set_ylim(-3, 50)
    b.legend(frameon=False, loc="upper left")
    save(fig, "fig5_markdown")


if __name__ == "__main__":
    print("Rendering figures into", FIG_DIR)
    fig_architecture()
    fig_elasticity()
    fig_uplift()
    fig_forecast()
    fig_markdown()
    print("done.")
