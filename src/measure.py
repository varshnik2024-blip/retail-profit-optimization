"""
Phase 10 - Closed-loop profit measurement.

THE QUESTION THIS PHASE ANSWERS
-------------------------------
Every earlier phase ended with a claim: the forecast is this accurate, the
price should move that much, the markdown recovers this. This phase asks the
only question a retailer actually cares about:

    if we had followed these recommendations, would we have made more money?

THE HONEST DIFFICULTY
---------------------
We cannot simply observe the answer. The retailer charged the price they
charged; we never get to see what would have happened at a different price.
That is the fundamental problem of counterfactual evaluation, and any project
that reports a single confident profit number is quietly ignoring it.

So instead of one number we produce a LEDGER in which every claim is graded by
how much evidence stands behind it:

  MEASURED     compares two things we actually observed. No modelling.
  REPLAYED     a policy re-run against real demand that genuinely happened.
  MODELLED     depends on a demand curve we estimated. Reported as a range.

PART C is the interesting one. Within the 26 held-out weeks the retailer's own
everyday price moved around, so for some store-weeks it landed close to what
our system would have recommended and for others far away. Comparing the gross
profit actually recorded in those two groups uses real observed outcomes rather
than a simulation. Promoted weeks are excluded, because on the full panel
'far from the recommendation' is almost a synonym for 'on clearance' and the
comparison then measures nothing but the margin difference between full price
and markdown. Store-product and week fixed effects are absorbed as in Phase 4.

CONNECTION TO EARLIER PHASES
----------------------------
  forecast_test_*.csv          Phase 5   actual vs predicted units
  elasticity_*.csv             Phase 3   demand response and its std error
  price_recommendations_*.csv  Phase 6   the recommended price per product
  markdown_policy_*.csv        Phase 7   clearance cash recovered
  inventory_policy_*.csv       Phase 8   replenishment cost
  absorb_two_way()             Phase 4   reused for the fixed-effects comparison

Run:
    python src/measure.py             # both sources
    python src/measure.py real
    python src/measure.py synthetic
"""

import sys
import warnings

import numpy as np
import pandas as pd

from config import OUTPUTS_DIR, MARGIN_SENSITIVITY
from schema import Canon
from uplift import absorb_two_way, ols
from pricing import demand_at, profit_at

warnings.filterwarnings("ignore")

NEAR = 0.05      # within 5% of our recommended price counts as "followed"
FAR = 0.15       # more than 15% away counts as "not followed"


# ==========================================================================
# PART A - forecast realisation (MEASURED)
# ==========================================================================
def forecast_realisation(fc):
    """
    Compare predicted against actual on the held-out weeks. Nothing is
    modelled here: both quantities were observed, so this is the strongest
    evidence in the whole project.
    """
    a = fc[Canon.UNITS].to_numpy(float)
    p = fc["pred"].to_numpy(float)
    margin = fc[Canon.PRICE] - fc[Canon.UNIT_COST]

    return {
        "actual_units": a.sum(),
        "predicted_units": p.sum(),
        "unit_error_pct": (p.sum() - a.sum()) / a.sum() * 100,
        "wape": np.abs(a - p).sum() / a.sum(),
        "actual_profit": float((a * margin).sum()),
        "predicted_profit": float((p * margin).sum()),
        "profit_error_pct": float(((p * margin).sum() - (a * margin).sum())
                                  / (a * margin).sum() * 100),
    }


# ==========================================================================
# PART B - counterfactual pricing ledger (MODELLED)
# ==========================================================================
def counterfactual_pricing(fc, rec, elast_shift=0.0, margin_shift=0.0):
    """
    Replay the test weeks charging our recommended price instead of the
    observed one, and value the difference.

    Demand at the counterfactual price comes from the estimated elasticity, so
    this is model-based. elast_shift and margin_shift let the caller re-run the
    whole calculation under different assumptions, which is how we turn a
    single number into an honest range.
    """
    m = fc.merge(rec[["upc", "elasticity", "elasticity_se", "recommended_price",
                      "current_price"]], left_on=Canon.UPC, right_on="upc",
                 how="inner")
    if m.empty:
        return None

    cost = m[Canon.UNIT_COST] - m[Canon.BASE_PRICE] * margin_shift
    cost = cost.clip(lower=0.01)
    e = m["elasticity"] + elast_shift * m["elasticity_se"].fillna(0)
    e = e.clip(upper=-0.05)

    # observed side: what actually happened, valued at the shifted cost
    obs_units = m[Canon.UNITS]
    obs_profit = obs_units * (m[Canon.PRICE] - cost)

    # counterfactual: our price, with demand scaled off the SAME observed
    # units rather than off a fresh forecast. Anchoring on what really sold
    # keeps the comparison as close to the data as the model allows.
    ratio = (m["recommended_price"] / m[Canon.PRICE]).clip(0.4, 2.5)
    cf_units = obs_units * ratio ** e
    cf_profit = cf_units * (m["recommended_price"] - cost)

    return {
        "observed_profit": float(obs_profit.sum()),
        "counterfactual_profit": float(cf_profit.sum()),
        "delta": float(cf_profit.sum() - obs_profit.sum()),
        "delta_pct": float((cf_profit.sum() / obs_profit.sum() - 1) * 100),
        "observed_units": float(obs_units.sum()),
        "counterfactual_units": float(cf_units.sum()),
        "unit_change_pct": float((cf_units.sum() / obs_units.sum() - 1) * 100),
        "rows": len(m),
    }


# ==========================================================================
# PART C - natural experiment on observed profit (OBSERVATIONAL)
# ==========================================================================
def natural_experiment(fc, rec):
    """
    Did store-weeks priced near our recommendation record higher gross profit
    than those priced far from it?

    ONLY NON-PROMOTED WEEKS ARE USED, and this restriction is essential. Our
    recommended price sits above the average shelf price for most products, so
    on the full panel "far from the recommendation" is almost a synonym for
    "deeply discounted": in the real data that group was 69% promoted with 17%
    of rows priced below cost, against 6% promoted in the near group. Comparing
    them measured nothing except that full-price weeks carry more margin than
    clearance weeks, which is true by construction and says nothing about our
    recommendation. Dropping promoted weeks removes the confound completely -
    both groups then contain zero promotions and zero below-cost rows.

    What remains is everyday-price variation. Identification comes from a given
    product in a given store charging different regular prices in different
    weeks, with store-product and week fixed effects absorbed exactly as in
    Phase 4. Both outcomes are real recorded profit; nothing is simulated.
    """
    m = fc.merge(rec[["upc", "recommended_price"]], left_on=Canon.UPC,
                 right_on="upc", how="inner").copy()
    if m.empty:
        return None

    n_all = len(m)
    m = m[m[Canon.ANY_PROMO] == 0].copy()          # everyday prices only
    if len(m) < 500:
        return None

    m["rel_gap"] = ((m[Canon.PRICE] - m["recommended_price"]).abs()
                    / m["recommended_price"])
    m["group"] = np.where(m["rel_gap"] <= NEAR, "near",
                          np.where(m["rel_gap"] >= FAR, "far", "middle"))
    sub = m[m["group"] != "middle"].copy()
    if len(sub) < 500 or sub["group"].nunique() < 2:
        return None

    sub["profit"] = sub[Canon.UNITS] * (sub[Canon.PRICE] - sub[Canon.UNIT_COST])
    sub["log_profit"] = np.log(sub["profit"].clip(lower=0.01))
    sub["near"] = (sub["group"] == "near").astype(float)
    raw = sub.groupby("group")["profit"].mean()

    cell = (sub[Canon.STORE_ID].astype(str) + "_" + sub[Canon.UPC].astype(str))
    codes_c, uniq_c = pd.factorize(cell)
    codes_w, uniq_w = pd.factorize(sub[Canon.WEEK_INDEX])
    X = absorb_two_way(sub[["near"]].to_numpy(float), codes_c, len(uniq_c),
                       codes_w, len(uniq_w))
    y = absorb_two_way(sub["log_profit"].to_numpy(), codes_c, len(uniq_c),
                       codes_w, len(uniq_w))
    if X.std() < 1e-10:
        return None
    beta, se = ols(X, y)

    return {
        "rows_all": n_all,
        "rows_non_promo": len(m),
        "n_near": int((sub["group"] == "near").sum()),
        "n_far": int((sub["group"] == "far").sum()),
        "promo_rate_near": 0.0,
        "promo_rate_far": 0.0,
        "raw_near": float(raw.get("near", np.nan)),
        "raw_far": float(raw.get("far", np.nan)),
        "coef": float(beta[0]),
        "se": float(se[0]),
        "t": float(beta[0] / se[0]) if se[0] > 0 else np.nan,
        "pct_effect": float((np.exp(beta[0]) - 1) * 100),
    }


# ==========================================================================
# Orchestration
# ==========================================================================
def run(source):
    print("=" * 78)
    print(f"PHASE 10 - CLOSED-LOOP PROFIT MEASUREMENT   [{source}]")
    print("=" * 78)

    need = {
        "forecast": OUTPUTS_DIR / f"forecast_test_{source}.csv",
        "pricing": OUTPUTS_DIR / f"price_recommendations_{source}.csv",
    }
    for k, p in need.items():
        if not p.exists():
            print(f"    MISSING {p.name}")
            return None
    fc = pd.read_csv(need["forecast"])
    rec = pd.read_csv(need["pricing"])

    md_path = OUTPUTS_DIR / f"markdown_policy_{source}.csv"
    inv_path = OUTPUTS_DIR / f"inventory_policy_{source}.csv"
    md = pd.read_csv(md_path) if md_path.exists() else None
    inv = pd.read_csv(inv_path) if inv_path.exists() else None

    print(f"\n[1] Evaluation window: held-out weeks only, {len(fc):,} store-weeks")
    print(f"    products with a recommendation: {len(rec)}")

    # ------------------------------------------------------------- PART A
    print("\n[2] PART A - did the forecast come true?   [MEASURED]")
    fa = forecast_realisation(fc)
    print(f"    {'':<26}{'actual':>14}{'predicted':>14}{'error':>10}")
    print("    " + "-" * 64)
    print(f"    {'units sold':<26}{fa['actual_units']:>14,.0f}"
          f"{fa['predicted_units']:>14,.0f}{fa['unit_error_pct']:>9.1f}%")
    print(f"    {'gross profit at observed':<26}{fa['actual_profit']:>14,.0f}"
          f"{fa['predicted_profit']:>14,.0f}{fa['profit_error_pct']:>9.1f}%")
    print(f"    {'prices':<26}")
    print(f"\n    WAPE {fa['wape']:.3f}. Both columns were observed, so this is")
    print("    a direct measurement with no counterfactual reasoning in it.")

    # ------------------------------------------------------------- PART B
    print("\n[3] PART B - profit if our prices had been charged   [MODELLED]")
    base = counterfactual_pricing(fc, rec)
    print(f"    observed gross profit      {base['observed_profit']:>14,.0f}")
    print(f"    under our recommendations  {base['counterfactual_profit']:>14,.0f}")
    print(f"    difference                 {base['delta']:>+14,.0f}  "
          f"({base['delta_pct']:+.1f}%)")
    print(f"    units {base['observed_units']:,.0f} -> "
          f"{base['counterfactual_units']:,.0f} "
          f"({base['unit_change_pct']:+.1f}%)")

    print("\n    Sensitivity - the same calculation under other assumptions:")
    print(f"    {'assumption':<34}{'profit delta':>16}{'vs base':>12}")
    print("    " + "-" * 62)
    scen = [("baseline", 0.0, 0.0),
            ("elasticity -1 std error", -1.0, 0.0),
            ("elasticity +1 std error", +1.0, 0.0)]
    for d in MARGIN_SENSITIVITY:
        if d != 0.0:
            scen.append((f"gross margin {d:+.0%}", 0.0, d))
    lo = hi = base["delta"]
    for name, es, ms in scen:
        r = counterfactual_pricing(fc, rec, es, ms)
        lo, hi = min(lo, r["delta"]), max(hi, r["delta"])
        rel = (r["delta"] - base["delta"]) / abs(base["delta"]) * 100
        print(f"    {name:<34}{r['delta']:>+16,.0f}{rel:>11.0f}%")
    print(f"\n    RANGE across all assumptions: {lo:+,.0f} to {hi:+,.0f}")
    print("    Report the range. The single number is not the finding.")

    # ------------------------------------------------------------- PART C
    print("\n[4] PART C - observed profit where the retailer happened to price")
    print("    near our recommendation   [OBSERVATIONAL, real outcomes]")
    ne = natural_experiment(fc, rec)
    if ne is None:
        print("    not enough store-weeks in both groups to compare")
    else:
        print(f"    non-promoted store-weeks used : {ne['rows_non_promo']:,} "
              f"of {ne['rows_all']:,}")
        print(f"      promoted weeks are excluded: on the full panel the 'far'")
        print(f"      group is mostly clearance, which would confound the test")
        print(f"    priced within {NEAR:.0%} of our price : {ne['n_near']:,}")
        print(f"    priced over  {FAR:.0%} away          : {ne['n_far']:,}")
        print(f"\n    raw mean gross profit  near {ne['raw_near']:>9,.2f}"
              f"   far {ne['raw_far']:>9,.2f}")
        print(f"    after absorbing store-product and week fixed effects:")
        print(f"      coefficient {ne['coef']:+.4f}  (se {ne['se']:.4f}, "
              f"t = {ne['t']:.1f})")
        print(f"      implies {ne['pct_effect']:+.1f}% profit per store-week")
        verdict = ("supports" if ne["coef"] > 0 and abs(ne["t"]) > 2
                   else "does not support" if ne["coef"] < 0 and abs(ne["t"]) > 2
                   else "is inconclusive about")
        print(f"\n    This {verdict} the pricing recommendation, using profit")
        print("    that was actually recorded rather than simulated.")
        print("    Caveat: prices were not set at random, so this is suggestive")
        print("    evidence, not proof.")

    # ------------------------------------------------------------- LEDGER
    print("\n[5] CONSOLIDATED LEDGER - every claim graded by its evidence")
    print(f"    {'module':<22}{'result':>22}{'evidence':>16}")
    print("    " + "-" * 62)
    rows = [("Forecast accuracy", f"WAPE {fa['wape']:.3f}", "MEASURED"),
            ("Forecast bias", f"{fa['unit_error_pct']:+.1f}% units", "MEASURED")]
    if inv is not None:
        base_c = inv["static_normal_total_cost"].sum()
        our_c = inv["forecast_driven_total_cost"].sum()
        rows.append(("Inventory cost",
                     f"{(1 - our_c / base_c) * 100:.1f}% lower", "REPLAYED"))
        rows.append(("Inventory fill rate",
                     f"{inv['static_normal_fill_rate'].mean():.1%} -> "
                     f"{inv['forecast_driven_fill_rate'].mean():.1%}", "REPLAYED"))
    if md is not None:
        rows.append(("Markdown cash",
                     f"{(md['dp_profit'].sum() / md['fixed_profit'].sum() - 1) * 100:+.1f}% "
                     f"vs fixed", "MODELLED"))
    rows.append(("Pricing profit", f"{base['delta_pct']:+.1f}% (range "
                 f"{lo / base['observed_profit'] * 100:+.1f}% to "
                 f"{hi / base['observed_profit'] * 100:+.1f}%)", "MODELLED"))
    if ne is not None:
        rows.append(("Pricing, observed", f"{ne['pct_effect']:+.1f}% per week",
                     "OBSERVATIONAL"))
    for a, b, c in rows:
        print(f"    {a:<22}{b:>22}{c:>16}")

    print("\n    MEASURED      both sides observed, no model")
    print("    REPLAYED      policy re-run against demand that really happened")
    print("    OBSERVATIONAL real outcomes, but treatment not randomised")
    print("    MODELLED      depends on the estimated demand curve")

    # ------------------------------------------------------------- write
    ledger = pd.DataFrame(rows, columns=["module", "result", "evidence_grade"])
    ledger.to_csv(OUTPUTS_DIR / f"profit_ledger_{source}.csv", index=False)
    pd.DataFrame([{**fa, "scenario": "forecast_realisation"}]).round(4).to_csv(
        OUTPUTS_DIR / f"forecast_realisation_{source}.csv", index=False)
    scen_rows = []
    for name, es, ms in scen:
        r = counterfactual_pricing(fc, rec, es, ms)
        scen_rows.append({"scenario": name, **r})
    pd.DataFrame(scen_rows).round(2).to_csv(
        OUTPUTS_DIR / f"counterfactual_pricing_{source}.csv", index=False)
    if ne is not None:
        pd.DataFrame([ne]).round(4).to_csv(
            OUTPUTS_DIR / f"natural_experiment_{source}.csv", index=False)

    print(f"\n[6] Written: profit_ledger_{source}.csv, "
          f"counterfactual_pricing_{source}.csv,")
    print(f"    forecast_realisation_{source}.csv"
          + (f", natural_experiment_{source}.csv" if ne is not None else ""))
    print("=" * 78)
    return ledger


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else None
    for s in ([which] if which else ["synthetic", "real"]):
        run(s)
        print()