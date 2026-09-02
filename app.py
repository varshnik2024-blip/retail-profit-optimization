"""
Phase 11 - Interactive dashboard.

WHAT THIS IS
------------
Phases 1 to 10 are command-line scripts that write CSV files. A category
manager cannot run those. This turns the whole system into a web page they can
open in a browser and use.

THE DESIGN DECISION THAT MATTERS
--------------------------------
Most of the dashboard reads results that the pipeline already computed. The
Pricing tab does not: it RECOMPUTES the recommendation live as the user moves
the assumption sliders.

That is possible because the pricing rule is a closed-form formula, not a
trained model. Given an elasticity and a cost we can solve for the optimal
price in microseconds, so the user can drag the gross-margin slider and watch
every recommendation and the whole profit figure move. This is the honest way
to present a result that depends on an assumption we could not observe: rather
than hiding the assumption in a config file, we hand the user the dial.

Run:
    streamlit run app.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from config import (OUTPUTS_DIR, MAX_PRICE_INCREASE_PCT,      # noqa: E402
                    MAX_PRICE_DECREASE_PCT, MIN_GROSS_MARGIN_FLOOR,
                    GROSS_MARGIN_BY_CATEGORY, DEFAULT_GROSS_MARGIN)
from schema import Canon                                       # noqa: E402

st.set_page_config(page_title="Retail Profit Optimisation",
                   page_icon="chart_with_upwards_trend", layout="wide")

NAVY = "#1f4e79"
GREY = "#9aa5b1"
AMBER = "#e08214"


# =========================================================================
# Loading - cached so switching tabs does not re-read the disk
# =========================================================================
@st.cache_data(show_spinner=False)
def load(name, source):
    p = OUTPUTS_DIR / f"{name}_{source}.csv"
    return pd.read_csv(p) if p.exists() else None


@st.cache_data(show_spinner=False)
def product_table(source):
    """
    Returns (rows, products).

    rows     - one row per store-week-product from the held-out weeks, which is
               the level the demand response must be applied at. Price varies
               within a product because of promotions, so averaging price
               before applying elasticity biases the answer.
    products - one row per product, carrying the reference price and the
               competitive band the guardrails need.
    """
    fc = load("forecast_test", source)
    ela = load("elasticity", source)
    if fc is None or ela is None:
        return None, None

    cols = ["upc", "elast_final", "description"]
    if "se_best_raw" in ela.columns:
        cols.append("se_best_raw")
    e = ela[cols].rename(columns={"elast_final": "elasticity",
                                  "se_best_raw": "elasticity_se"})

    rows = fc.merge(e, left_on=Canon.UPC, right_on="upc", how="inner")
    rows = rows.dropna(subset=["elasticity"])

    products = (rows.groupby(Canon.UPC)
                .agg(category=(Canon.CATEGORY, "first"),
                     description=("description", "first"),
                     elasticity=("elasticity", "first"),
                     base_price=(Canon.BASE_PRICE, "mean"),
                     current_price=(Canon.PRICE, "mean"),
                     units=(Canon.UNITS, "sum"),
                     comp_lo=(Canon.PRICE, lambda s: s.quantile(0.05)),
                     comp_hi=(Canon.PRICE, lambda s: s.quantile(0.95)))
                .reset_index())
    products["comp_hi"] = products["comp_hi"] * 1.10
    return rows, products


# =========================================================================
# Live pricing - the same maths as src/pricing.py, recomputed on demand
# =========================================================================
def recompute_prices(rows, products, margin_by_cat, max_up, max_down, floor):
    """
    Re-solve the Lerner condition, re-apply the guardrails, then value the
    result ROW BY ROW so promotional price variation is handled correctly.

        p* = c * e / (1 + e)      for e < -1
        inelastic products have no interior optimum, so the cap decides

    Fully vectorised, so a slider drag re-solves everything instantly.
    """
    p = products.copy()
    margins = p["category"].map(margin_by_cat).fillna(DEFAULT_GROSS_MARGIN)
    p["unit_cost"] = (p["base_price"] * (1 - margins)).clip(lower=0.01)

    e = p["elasticity"].clip(upper=-0.05)
    interior = e < -1.02
    raw = np.where(interior, p["unit_cost"] * e / (1 + e), np.inf)

    cap_up = np.minimum(p["current_price"] * (1 + max_up), p["comp_hi"])
    cap_lo = np.maximum.reduce([
        (p["current_price"] * (1 - max_down)).to_numpy(),
        (p["unit_cost"] * (1 + floor)).to_numpy(),
        p["comp_lo"].to_numpy()])
    cap_lo = np.minimum(cap_lo, cap_up)

    rec = np.clip(raw, cap_lo, cap_up)
    rec = np.where(np.isfinite(rec), rec, cap_up)

    p["unconstrained_price"] = raw
    p["recommended_price"] = rec
    p["change_pct"] = (rec / p["current_price"] - 1) * 100
    p["binding"] = np.where(np.isclose(rec, cap_up), "max increase / comp. band",
                            np.where(np.isclose(rec, cap_lo),
                                     "max discount / margin floor", "none"))

    # ---- value it at row level, as Phase 10 does
    # rows already carries a unit_cost column from the pipeline; ours is the
    # slider-driven one, so rename before merging or pandas makes _x/_y
    cost_map = p[[Canon.UPC, "unit_cost", "recommended_price"]].rename(
        columns={"unit_cost": "cost_assumed"})
    r = rows.merge(cost_map, on=Canon.UPC, how="inner")
    ee = r["elasticity"].clip(upper=-0.05)
    ratio = (r["recommended_price"] / r[Canon.PRICE]).clip(0.4, 2.5)
    cf_units = r[Canon.UNITS] * ratio ** ee
    r["profit_now"] = r[Canon.UNITS] * (r[Canon.PRICE] - r["cost_assumed"])
    r["profit_rec"] = cf_units * (r["recommended_price"] - r["cost_assumed"])

    agg = (r.groupby(Canon.UPC)
           .agg(profit_now=("profit_now", "sum"),
                profit_rec=("profit_rec", "sum"))
           .reset_index())
    p = p.merge(agg, on=Canon.UPC, how="left")
    p["profit_delta"] = p["profit_rec"] - p["profit_now"]
    return p


def kpi(col, label, value, help_text=None):
    col.metric(label, value, help=help_text)


# =========================================================================
# Sidebar
# =========================================================================
st.sidebar.title("Controls")

available = [s for s in ("real", "synthetic")
             if (OUTPUTS_DIR / f"forecast_test_{s}.csv").exists()]
if not available:
    st.error("No pipeline outputs found in outputs/. "
             "Run the phase scripts in src/ first.")
    st.stop()

source = st.sidebar.radio(
    "Dataset", available,
    format_func=lambda s: ("Real dunnhumby panel" if s == "real"
                           else "Synthetic replica (known truth)"))

rows, tbl = product_table(source)
if tbl is None:
    st.error(f"Missing outputs for '{source}'.")
    st.stop()

cats = sorted(tbl["category"].unique())
chosen = st.sidebar.multiselect("Categories", cats, default=cats)

st.sidebar.divider()
st.sidebar.caption("BUSINESS ASSUMPTIONS")
st.sidebar.caption(
    "Unit cost is not in any public retail dataset. Set it here and every "
    "recommendation updates.")

uniform = st.sidebar.checkbox("Use one margin for all categories", value=False)
margin_by_cat = {}
if uniform:
    m = st.sidebar.slider("Gross margin %", 10, 60, 30, 1)
    margin_by_cat = {c: m / 100 for c in cats}
else:
    for c in cats:
        default = GROSS_MARGIN_BY_CATEGORY.get(c.upper(), DEFAULT_GROSS_MARGIN)
        margin_by_cat[c] = st.sidebar.slider(
            f"{c.title()} margin %", 10, 60, int(round(default * 100)), 1) / 100

st.sidebar.divider()
st.sidebar.caption("PRICING GUARDRAILS")
max_up = st.sidebar.slider("Max price increase %", 0, 30,
                           int(round(MAX_PRICE_INCREASE_PCT * 100)), 1) / 100
max_down = st.sidebar.slider("Max discount depth %", 0, 60,
                             int(round(MAX_PRICE_DECREASE_PCT * 100)), 1) / 100
floor = st.sidebar.slider("Minimum gross margin %", 0, 25,
                          int(round(MIN_GROSS_MARGIN_FLOOR * 100)), 1) / 100

view = tbl[tbl["category"].isin(chosen)] if chosen else tbl
if view.empty:
    st.warning("No products match that filter.")
    st.stop()
rows_view = rows[rows[Canon.UPC].isin(view[Canon.UPC])]
priced = recompute_prices(rows_view, view, margin_by_cat,
                          max_up, max_down, floor)


# =========================================================================
# Header
# =========================================================================
st.title("Retail Profit Optimisation")
st.caption(
    f"dunnhumby Breakfast at the Frat  ·  {'real panel' if source == 'real' else 'synthetic replica'}"
    f"  ·  {len(view)} products  ·  held-out weeks 131-156")

tabs = st.tabs(["Overview", "Pricing", "Demand", "Markdown",
                "Inventory", "Evidence"])

# ---------------------------------------------------------------- Overview
with tabs[0]:
    fcm = load("forecast_metrics", source)
    inv = load("inventory_policy", source)
    md = load("markdown_policy", source)

    c1, c2, c3, c4 = st.columns(4)
    if fcm is not None:
        t = fcm[fcm.get("split").eq("test")] if "split" in fcm else fcm
        gb = t[t["model"].str.contains("gradient", case=False, na=False)]
        nv = t[t["model"].str.contains("naive \\(last", case=False, na=False)]
        if len(gb) and len(nv):
            g, n = gb["WAPE"].iloc[0], nv["WAPE"].iloc[0]
            kpi(c1, "Forecast WAPE", f"{g:.3f}",
                f"{(1 - g / n) * 100:.1f}% better than the naive baseline")
    kpi(c2, "Pricing profit", f"{priced['profit_delta'].sum():+,.0f}",
        "Modelled: depends on the elasticity estimate")
    if inv is not None:
        b = inv["static_normal_total_cost"].sum()
        o = inv["forecast_driven_total_cost"].sum()
        kpi(c3, "Inventory cost", f"{(1 - o / b) * 100:.1f}% lower",
            "Replayed against demand that really happened")
    if md is not None:
        kpi(c4, "Markdown cash",
            f"{(md['dp_profit'].sum() / md['fixed_profit'].sum() - 1) * 100:+.1f}%",
            "Versus the standard -20%/-40% ladder")

    st.divider()
    led = load("profit_ledger", source)
    if led is not None:
        st.subheader("Every claim, graded by its evidence")
        st.dataframe(led, use_container_width=True, hide_index=True)
        st.caption(
            "MEASURED both sides observed  ·  REPLAYED policy re-run on real "
            "demand  ·  OBSERVATIONAL real outcomes but not randomised  ·  "
            "MODELLED depends on the estimated demand curve")
    else:
        st.info("Run  python src/measure.py  to populate the evidence ledger.")

# ----------------------------------------------------------------- Pricing
with tabs[1]:
    st.subheader("Price recommendations")
    st.caption("These recompute live. Move a slider in the sidebar and watch "
               "the numbers change.")

    c1, c2, c3, c4 = st.columns(4)
    kpi(c1, "Profit now", f"{priced['profit_now'].sum():,.0f}")
    kpi(c2, "Profit if adopted", f"{priced['profit_rec'].sum():,.0f}")
    kpi(c3, "Change", f"{priced['profit_delta'].sum():+,.0f}",
        f"{priced['profit_delta'].sum() / priced['profit_now'].sum() * 100:+.1f}%")
    kpi(c4, "Products raised",
        f"{(priced['change_pct'] > 0.05).sum()} / {len(priced)}")

    binding = priced["binding"].value_counts()
    if binding.get("max increase / comp. band", 0) / len(priced) > 0.7:
        st.warning(
            f"{binding.get('max increase / comp. band', 0)} of {len(priced)} "
            "products are pinned to the upper guardrail. The formula wants to "
            "go higher, so the cap is doing the work, not the elasticity. "
            "Raise the margin assumption to see this soften.")

    show = (priced[["description", "category", "elasticity", "current_price",
                    "recommended_price", "change_pct", "profit_delta", "binding"]]
            .sort_values("profit_delta", ascending=False)
            .rename(columns={"description": "Product", "category": "Category",
                             "elasticity": "Elasticity",
                             "current_price": "Now", "recommended_price": "Recommended",
                             "change_pct": "Change %", "profit_delta": "Profit change",
                             "binding": "Constraint"}))
    st.dataframe(show.round(2), use_container_width=True, hide_index=True,
                 height=340)

    exp = load("price_explanations", source)
    if exp is not None:
        st.divider()
        st.subheader("Why? (from the saved run, at the default assumptions)")
        pick = st.selectbox("Product", exp["description"].tolist())
        row = exp[exp["description"] == pick].iloc[0]
        a, b = st.columns([2, 1])
        a.info(row["explanation"])
        b.metric("Recommended", f"{row['recommended_price']:.2f}",
                 f"{row['change_pct']:+.1f}%")
        b.caption(f"Confidence: {row['confidence']}")

# ------------------------------------------------------------------ Demand
with tabs[2]:
    st.subheader("Forecast accuracy on held-out weeks")
    fcm = load("forecast_metrics", source)
    if fcm is not None:
        t = fcm[fcm.get("split").eq("test")] if "split" in fcm else fcm
        chart = t.set_index("model")[["WAPE"]]
        st.bar_chart(chart, color=NAVY, height=260)
        st.caption("Lower is better. The three naive baselines are the point: "
                   "an accuracy figure quoted without them cannot be judged.")
        st.dataframe(t[["model", "WAPE", "MAE", "RMSE", "bias_pct"]].round(3),
                     use_container_width=True, hide_index=True)

    shp = load("shap_importance", source)
    if shp is not None:
        st.divider()
        st.subheader("What drives the forecast (SHAP)")
        st.bar_chart(shp.head(10).set_index("feature")[["share_pct"]],
                     color=GREY, height=260)
        st.caption("Share of the total explanation. SHAP approximates the "
                   "model's reasoning; the pricing explanations are exact "
                   "because pricing is a formula.")

# ---------------------------------------------------------------- Markdown
with tabs[3]:
    st.subheader("Clearance policy comparison")
    md = load("markdown_policy", source)
    if md is None:
        st.info("Run  python src/markdown.py  first.")
    else:
        sel = md[md["category"].isin(chosen)] if chosen else md
        pol = pd.DataFrame({
            "policy": ["never mark down", "panic -40% now",
                       "fixed -20%/-40%", "DP optimal"],
            "cash recovered": [sel[f"{k}_profit"].mean()
                               for k in ("never", "panic", "fixed", "dp")],
            "sell-through %": [sel[f"{k}_sellthrough"].mean() * 100
                               for k in ("never", "panic", "fixed", "dp")],
        })
        c1, c2 = st.columns(2)
        c1.bar_chart(pol.set_index("policy")[["cash recovered"]],
                     color=NAVY, height=280)
        c2.bar_chart(pol.set_index("policy")[["sell-through %"]],
                     color=AMBER, height=280)
        st.dataframe(pol.round(1), use_container_width=True, hide_index=True)
        st.caption(
            "Objective is cash recovered, not margin: the stock is already "
            "bought, so its cost is sunk. The optimal ladder eases discount "
            "depth as stock depletes, which a fixed schedule cannot do because "
            "it never looks at remaining inventory.")

# --------------------------------------------------------------- Inventory
with tabs[4]:
    st.subheader("Replenishment policy comparison")
    inv = load("inventory_policy", source)
    if inv is None:
        st.info("Run  python src/inventory.py  first.")
    else:
        sel = inv[inv["category"].isin(chosen)] if chosen else inv
        keys = [("static_normal", "static normal (textbook)"),
                ("static_empirical", "static empirical quantile"),
                ("forecast_driven", "forecast-driven (ours)"),
                ("oracle", "oracle (perfect foresight)")]
        rows_ = [{"policy": lbl,
                  "holding": sel[f"{k}_holding_cost"].sum(),
                  "stockout": sel[f"{k}_stockout_cost"].sum(),
                  "total cost": sel[f"{k}_total_cost"].sum(),
                  "fill rate %": sel[f"{k}_fill_rate"].mean() * 100,
                  "weeks cover": sel[f"{k}_weeks_of_cover"].mean()}
                 for k, lbl in keys]
        pol = pd.DataFrame(rows_)
        st.bar_chart(pol.set_index("policy")[["holding", "stockout"]],
                     height=300)
        st.dataframe(pol.round(2), use_container_width=True, hide_index=True)
        st.caption(
            "Less stock and better service at the same time, because the "
            "forecast indicates when cover is needed rather than carrying it "
            "all year. The oracle row bounds what is achievable.")

# ---------------------------------------------------------------- Evidence
with tabs[5]:
    st.subheader("What this system does and does not know")

    cf = load("counterfactual_pricing", source)
    ne = load("natural_experiment", source)

    if cf is not None:
        st.markdown("**Pricing profit under different assumptions**")
        st.dataframe(cf[["scenario", "delta", "delta_pct"]].round(1),
                     use_container_width=True, hide_index=True)
        st.caption("The spread across these rows is the finding. A single "
                   "number would overstate what the data supports.")

    if ne is not None:
        r = ne.iloc[0]
        st.divider()
        st.markdown("**Check against profit that was actually recorded**")
        c1, c2, c3 = st.columns(3)
        c1.metric("Effect per store-week", f"{r['pct_effect']:+.1f}%")
        c2.metric("t statistic", f"{r['t']:.1f}")
        c3.metric("Store-weeks used", f"{int(r['n_near'] + r['n_far']):,}")
        if abs(r["t"]) < 2:
            st.warning(
                "This check is INCONCLUSIVE. The modelled estimate above and "
                "this observational one disagree, and both are reported. The "
                "modelled figure depends on the demand curve being right; when "
                "we look only at profit that was genuinely recorded, we cannot "
                "detect the effect.")

    st.divider()
    st.markdown("**Stated assumptions**")
    st.markdown(
        "- Unit cost is derived from a gross-margin assumption, adjustable in "
        "the sidebar. No public retail dataset contains COGS.\n"
        "- Lead time, holding cost, stockout penalty and salvage value are "
        "documented assumptions in `config.py`.\n"
        "- Markdown starting stock is a constructed scenario; the demand model "
        "behind it is estimated from real data.\n"
        "- Promotions were not randomised, so the causal estimates rest on the "
        "fixed-effects specification and are supported, not proved, by the "
        "parallel-trends placebo test.\n"
        "- The profit model is item-level and cannot see basket or "
        "store-traffic effects, which is why below-cost promotions are flagged "
        "rather than judged.")