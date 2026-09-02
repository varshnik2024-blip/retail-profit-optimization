"""
Phase 6 - Profit-maximising price optimisation, with guardrails.

WHAT THIS DOES
--------------
Takes the elasticity from Phase 3 and the baseline demand from Phase 5, and
recommends a price per product that maximises gross profit - then refuses to
recommend anything a real retailer could not actually put on a shelf.

THE MATHS
---------
With constant-elasticity demand,

    D(p) = D_base * (p / p_base) ^ e            e = elasticity, negative
    profit(p) = D(p) * (p - c)                  c = unit cost

Differentiate and set to zero:

    d/dp [ (p/p_base)^e * (p - c) ] = 0
    =>  e * (p - c) + p = 0
    =>  p* = c * e / (1 + e)

This is the classic monopoly pricing rule. It is equivalent to the Lerner
condition, that the optimal gross margin equals 1/|e|: a product with
elasticity -2 wants a 50% margin, one with -4 wants 25%.

WHY IT NEEDS GUARDRAILS
-----------------------
The formula only has an interior solution when e < -1. For an INELASTIC
product (-1 < e < 0) the derivative never turns over, so the unconstrained
"optimum" is an infinite price. Naively deploying this would price habitual
staples into the stratosphere.

That is not a bug in the maths, it is the maths correctly telling us that a
single-product profit objective is the wrong objective. Real retailers are
constrained by competitors, by shopper trust, and by law. So we impose:

    - no price rise larger than MAX_PRICE_INCREASE_PCT in one step
    - no discount deeper than MAX_PRICE_DECREASE_PCT
    - never below cost plus MIN_GROSS_MARGIN_FLOOR
    - never outside the band other stores in the chain are already charging

The last one also guards against the algorithmic-collusion problem documented
in recent research: pricing agents left unconstrained can drift into
supra-competitive prices. We report explicitly how much theoretical profit each
guardrail costs, so the trade-off is visible rather than hidden.

CONFIDENCE
----------
The optimal price is very sensitive to elasticity, and elasticity is estimated
with error. We therefore recompute the optimum at e +/- one standard error and
report the spread. A wide spread means "do not act on this one yet".

VALIDATION
----------
The closed-form optimum is checked against a brute-force grid search. If the
formula and the grid disagree, the formula is wrong.

Run:
    python src/pricing.py             # both sources
    python src/pricing.py real
    python src/pricing.py synthetic
"""

import sys
import warnings
import numpy as np
import pandas as pd

from config import (OUTPUTS_DIR, MAX_PRICE_INCREASE_PCT, MAX_PRICE_DECREASE_PCT,
                    MIN_GROSS_MARGIN_FLOOR, MARGIN_SENSITIVITY)
from schema import Canon
from data_loader import load_processed

warnings.filterwarnings("ignore")

ELASTIC_CAP = -1.02      # treat anything above this as "no interior optimum"
CONFIDENCE_BAND = 0.15   # recommended-price spread that counts as "confident"


# ==========================================================================
# Core economics
# ==========================================================================
def demand_at(price, base_demand, base_price, elasticity):
    """Constant-elasticity demand curve."""
    return base_demand * (price / base_price) ** elasticity


def profit_at(price, base_demand, base_price, elasticity, cost):
    return demand_at(price, base_demand, base_price, elasticity) * (price - cost)


def unconstrained_optimum(cost, elasticity):
    """
    p* = c * e / (1 + e), valid only for e < -1.

    For an inelastic product the profit function is increasing in price with no
    turning point, so we return infinity and let the guardrails decide.
    """
    if not np.isfinite(elasticity) or elasticity >= ELASTIC_CAP:
        return np.inf
    return cost * elasticity / (1 + elasticity)


def grid_optimum(base_demand, base_price, elasticity, cost, lo, hi, n=4000):
    """Brute-force search, used to verify the closed form."""
    if not np.isfinite(elasticity) or hi <= lo:
        return np.nan
    grid = np.linspace(lo, hi, n)
    prof = profit_at(grid, base_demand, base_price, elasticity, cost)
    return float(grid[int(np.argmax(prof))])


# ==========================================================================
# Guardrails
# ==========================================================================
def apply_guardrails(raw_price, current_price, cost, comp_lo, comp_hi):
    """
    Clamp the raw optimum into a price a retailer could actually charge.
    Returns (price, name_of_binding_constraint).
    """
    limits = [
        (current_price * (1 + MAX_PRICE_INCREASE_PCT), "max price increase"),
        (comp_hi, "competitive band (upper)"),
    ]
    floors = [
        (current_price * (1 - MAX_PRICE_DECREASE_PCT), "max discount depth"),
        (cost * (1 + MIN_GROSS_MARGIN_FLOOR), "margin floor"),
        (comp_lo, "competitive band (lower)"),
    ]

    price, binding = raw_price, "none (unconstrained)"
    for cap, name in limits:
        if np.isfinite(cap) and price > cap:
            price, binding = cap, name
    for floor, name in floors:
        if np.isfinite(floor) and price < floor:
            price, binding = floor, name
    return float(price), binding


# ==========================================================================
# Build recommendations
# ==========================================================================
def build_recommendations(fc, ela):
    """One recommendation per product, evaluated over the test weeks."""
    emap = dict(zip(ela["upc"], ela["elast_final"]))
    # standard error of the estimate we actually used
    semap = {}
    for _, r in ela.iterrows():
        se = np.nan
        for c in ("se_best_raw", "se_iv", "se_fe"):
            if c in ela.columns and np.isfinite(r.get(c, np.nan)):
                se = r[c]
                break
        semap[r["upc"]] = se

    rows = []
    for upc, g in fc.groupby(Canon.UPC):
        e = emap.get(upc, np.nan)
        if not np.isfinite(e):
            continue
        cost = float(g[Canon.UNIT_COST].mean())
        base_price = float(g[Canon.BASE_PRICE].mean())
        current_price = float(g[Canon.PRICE].mean())
        base_demand = float(g["baseline_demand"].sum())
        if base_demand <= 0 or base_price <= 0 or cost <= 0:
            continue

        # competitive band: what the rest of the chain actually charges
        comp_lo = float(g[Canon.PRICE].quantile(0.05))
        comp_hi = float(g[Canon.PRICE].quantile(0.95)) * 1.10

        raw = unconstrained_optimum(cost, e)
        rec, binding = apply_guardrails(raw, current_price, cost, comp_lo, comp_hi)

        # verify the closed form against a grid over the allowed range
        lo = max(cost * (1 + MIN_GROSS_MARGIN_FLOOR),
                 current_price * (1 - MAX_PRICE_DECREASE_PCT), comp_lo)
        hi = min(current_price * (1 + MAX_PRICE_INCREASE_PCT), comp_hi)
        grid_rec = grid_optimum(base_demand, base_price, e, cost, lo, hi) if hi > lo else np.nan

        p_cur = profit_at(current_price, base_demand, base_price, e, cost)
        p_rec = profit_at(rec, base_demand, base_price, e, cost)

        # confidence: re-solve at elasticity +/- 1 standard error
        se = semap.get(upc, np.nan)
        if np.isfinite(se) and se > 0:
            lo_e, hi_e = e - se, min(e + se, ELASTIC_CAP - 1e-6)
            p_lo = apply_guardrails(unconstrained_optimum(cost, lo_e),
                                    current_price, cost, comp_lo, comp_hi)[0]
            p_hi = apply_guardrails(unconstrained_optimum(cost, hi_e),
                                    current_price, cost, comp_lo, comp_hi)[0]
            spread = abs(p_hi - p_lo) / max(rec, 1e-9)
        else:
            spread = np.nan

        rows.append({
            "upc": upc,
            "category": g[Canon.CATEGORY].iloc[0],
            "description": g["description"].iloc[0] if "description" in g else "",
            "elasticity": e,
            "elasticity_se": se,
            "unit_cost": cost,
            "current_price": current_price,
            "unconstrained_price": raw,
            "recommended_price": rec,
            "grid_check_price": grid_rec,
            "price_change_pct": (rec / current_price - 1) * 100,
            "current_margin_pct": (current_price - cost) / current_price * 100,
            "recommended_margin_pct": (rec - cost) / rec * 100,
            "lerner_target_margin_pct": (100 / abs(e)) if abs(e) > 1 else np.nan,
            "binding_constraint": binding,
            "profit_current": p_cur,
            "profit_recommended": p_rec,
            "profit_uplift": p_rec - p_cur,
            "profit_uplift_pct": (p_rec / p_cur - 1) * 100 if p_cur > 0 else np.nan,
            "confidence_spread": spread,
            "confident": bool(np.isfinite(spread) and spread < CONFIDENCE_BAND),
        })
    return pd.DataFrame(rows)


# ==========================================================================
# Orchestration
# ==========================================================================
def run(source):
    print("=" * 76)
    print(f"PHASE 6 - PRICE OPTIMISATION WITH GUARDRAILS   [{source}]")
    print("=" * 76)

    fc_path = OUTPUTS_DIR / f"forecast_test_{source}.csv"
    ela_path = OUTPUTS_DIR / f"elasticity_{source}.csv"
    for p, msg in [(fc_path, "src/forecast.py"), (ela_path, "src/elasticity.py")]:
        if not p.exists():
            print(f"    MISSING {p.name} - run  python {msg} {source}  first")
            return None

    fc = pd.read_csv(fc_path)
    ela = pd.read_csv(ela_path)

    # description is not in the forecast file; bring it from elasticity.
    # Canon.UPC is literally "upc", so this is a plain same-name join - do NOT
    # drop the key afterwards or the groupby below loses its column.
    if "description" not in fc.columns:
        fc = fc.merge(ela[["upc", "description"]], on=Canon.UPC, how="left")

    print(f"\n[1] Inputs")
    print(f"    forecast rows (test weeks): {len(fc):,}")
    print(f"    products with elasticity  : {ela['elast_final'].notna().sum()}")
    print(f"    guardrails: max +{MAX_PRICE_INCREASE_PCT:.0%} / "
          f"-{MAX_PRICE_DECREASE_PCT:.0%}, margin floor {MIN_GROSS_MARGIN_FLOOR:.0%}")

    rec = build_recommendations(fc, ela)
    print(f"\n[2] Recommendations built for {len(rec)} products")

    # --------------------------------------------- closed form vs grid search
    chk = rec.dropna(subset=["grid_check_price"])
    if len(chk):
        diff = (chk["recommended_price"] - chk["grid_check_price"]).abs()
        rel = diff / chk["recommended_price"]
        print(f"\n[3] Closed form vs brute-force grid search")
        print(f"    max relative difference: {rel.max():.5f}  "
              f"({'AGREE' if rel.max() < 0.01 else 'DISAGREE - investigate'})")

    # ------------------------------------------------------------ elasticity
    n_inelastic = int((rec["elasticity"] >= ELASTIC_CAP).sum())
    print(f"\n[4] Elasticity regimes")
    print(f"    elastic   (e < -1): {len(rec) - n_inelastic:>3}  -> finite optimum exists")
    print(f"    inelastic (e > -1): {n_inelastic:>3}  -> optimum is unbounded, "
          f"guardrail decides")

    # ------------------------------------------------------------ guardrails
    print(f"\n[5] Which constraint ends up binding")
    for name, n in rec["binding_constraint"].value_counts().items():
        print(f"    {name:<28} {n:>3} products ({n / len(rec):>5.0%})")

    # --------------------------------------------------------- profit impact
    tot_cur = rec["profit_current"].sum()
    tot_rec = rec["profit_recommended"].sum()
    conf = rec[rec["confident"]]
    print(f"\n[6] Expected profit impact over the test weeks")
    print(f"    current pricing      {tot_cur:>14,.0f}")
    print(f"    recommended pricing  {tot_rec:>14,.0f}")
    print(f"    uplift               {tot_rec - tot_cur:>14,.0f}  "
          f"({(tot_rec / tot_cur - 1) * 100:+.1f}%)")
    if len(conf):
        cu = conf["profit_recommended"].sum() - conf["profit_current"].sum()
        print(f"    of which high-confidence only: {cu:>+14,.0f}  "
              f"({len(conf)} of {len(rec)} products)")

    # ------------------------------------------------- cost of the guardrails
    print(f"\n[7] What the guardrails cost us")
    unc = rec[np.isfinite(rec["unconstrained_price"])].copy()
    if len(unc):
        unc["profit_unconstrained"] = [
            profit_at(r.unconstrained_price,
                      r.profit_current / max(r.current_price - r.unit_cost, 1e-9),
                      r.current_price, r.elasticity, r.unit_cost)
            for r in unc.itertuples()]
        give_up = unc["profit_unconstrained"].sum() - unc["profit_recommended"].sum()
        print(f"    theoretical profit forgone: {give_up:>12,.0f}")
    print("    We accept that cost deliberately. An unconstrained pricing agent")
    print("    drifts to supra-competitive prices, which is a legal and")
    print("    reputational risk, not a business win.")

    # -------------------------------------------------------- biggest movers
    print(f"\n[8] Largest recommended increases")
    print(f"    {'product':<26}{'elast':>7}{'now':>8}{'rec':>8}{'chg %':>8}{'constraint':>26}")
    print("    " + "-" * 83)
    for r in rec.nlargest(4, "price_change_pct").itertuples():
        print(f"    {str(r.description)[:24]:<26}{r.elasticity:>7.2f}"
              f"{r.current_price:>8.2f}{r.recommended_price:>8.2f}"
              f"{r.price_change_pct:>7.1f}%{r.binding_constraint:>26}")
    print(f"\n[9] Largest recommended decreases")
    print(f"    {'product':<26}{'elast':>7}{'now':>8}{'rec':>8}{'chg %':>8}{'constraint':>26}")
    print("    " + "-" * 83)
    for r in rec.nsmallest(4, "price_change_pct").itertuples():
        print(f"    {str(r.description)[:24]:<26}{r.elasticity:>7.2f}"
              f"{r.current_price:>8.2f}{r.recommended_price:>8.2f}"
              f"{r.price_change_pct:>7.1f}%{r.binding_constraint:>26}")

    # ---------------------------------------------------------- by category
    print(f"\n[10] By category")
    summ = (rec.groupby("category")
            .agg(products=("upc", "count"),
                 mean_elasticity=("elasticity", "mean"),
                 mean_change_pct=("price_change_pct", "mean"),
                 profit_uplift=("profit_uplift", "sum"))
            .round(2))
    print(summ.to_string())

    # ------------------------------------------------- margin sensitivity
    # THIS IS THE MOST IMPORTANT SECTION IN THE PHASE.
    # Unit cost is NOT observed in the data; we derived it from an assumed
    # gross margin in config.py. The Lerner rule says the optimal margin is
    # 1/|e|, so the recommendation is essentially "close the gap between your
    # assumed margin and 1/|e|" - which means our headline number partly
    # reflects that assumption rather than the data. Rather than hide this, we
    # re-solve the whole problem across a band of margin assumptions and report
    # how much the conclusion moves.
    print(f"\n[11] Sensitivity to the gross-margin assumption")
    print(f"    {'margin shift':<14}{'mean chg %':>12}{'profit uplift':>16}{'% up':>8}")
    print("    " + "-" * 52)
    for delta in MARGIN_SENSITIVITY:
        fc_s = fc.copy()
        # cost = base_price * (1 - margin); shifting margin by delta shifts cost
        fc_s[Canon.UNIT_COST] = (fc_s[Canon.UNIT_COST]
                                 - fc_s[Canon.BASE_PRICE] * delta).clip(lower=0.01)
        rs = build_recommendations(fc_s, ela)
        if rs.empty:
            continue
        up = rs["profit_recommended"].sum() - rs["profit_current"].sum()
        share_up = (rs["price_change_pct"] > 0).mean()
        label = f"{delta:+.0%}" if delta else "baseline"
        print(f"    {label:<14}{rs['price_change_pct'].mean():>11.1f}%"
              f"{up:>16,.0f}{share_up:>7.0%}")
    print("    If the sign of the uplift flips across this band, the")
    print("    recommendation is an artefact of the cost assumption, not a")
    print("    finding. Report the band, not just the middle number.")

    out = OUTPUTS_DIR / f"price_recommendations_{source}.csv"
    rec.round(4).to_csv(out, index=False)
    print(f"\n[12] Written: {out.name}")
    print("=" * 76)
    return rec


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else None
    for s in ([which] if which else ["synthetic", "real"]):
        run(s)
        print()