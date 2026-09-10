"""
Phase 9 - Explainability and guardrail audit.

WHY THIS PHASE EXISTS
---------------------
Phases 3 to 8 produce numbers: an elasticity, a lift, a forecast, a price, a
markdown ladder, an order quantity. A category manager will not act on a number
that arrives without a reason. This phase answers, for every recommendation,
"why did the system say that?" - and then audits whether the guardrails we
imposed in Phase 6 are actually doing their job.

THREE PARTS, TWO KINDS OF EXPLANATION
-------------------------------------
There is an important distinction here.

  PART A - the forecast is a gradient-boosted tree ensemble. Nobody can read
  400 trees, so we APPROXIMATE its reasoning with SHAP values, which split a
  single prediction into one contribution per feature. SHAP is an estimate of
  the model's behaviour, and we say so.

  PART B - the price recommendation is NOT a black box. It comes from a
  formula: the Lerner condition, then a guardrail clamp. So its explanation can
  be EXACT rather than approximate. We reconstruct, arithmetically, the whole
  chain from elasticity to final price. This is a faithful explanation, not an
  approximation of one, and that difference is worth stating.

  PART C - a guardrail audit. Phase 6 imposed four limits on the recommended
  price. Here we check which one actually binds, how much theoretical profit
  each costs, and whether the recommendations show a pattern that would be
  reputationally unwise: concentrating price rises on the cheapest products,
  which are disproportionately bought by budget-conscious shoppers.

CONNECTION TO EARLIER PHASES
----------------------------
  reads elasticity_*.csv          from Phase 3
  reads forecast_test_*.csv       from Phase 5
  reads price_recommendations_*   from Phase 6
  retrains the Phase 5 model so SHAP has something to explain

Run:
    python src/explain.py             # both sources
    python src/explain.py real
    python src/explain.py synthetic
"""

import sys
import warnings

import numpy as np
import pandas as pd

from config import (OUTPUTS_DIR, MAX_PRICE_INCREASE_PCT,
                    MAX_PRICE_DECREASE_PCT, MIN_GROSS_MARGIN_FLOOR,
                    TRAIN_END_WEEK, VALID_END_WEEK)
from schema import Canon
from data_loader import load_processed
from forecast import build_features, fit_model, FEATURES
from pricing import unconstrained_optimum, ELASTIC_CAP

warnings.filterwarnings("ignore")

SHAP_SAMPLE = 600         # rows explained; SHAP is expensive, this is enough
SHAP_BACKGROUND = 100     # reference rows the explainer compares against
SEED = 42


# ==========================================================================
# PART A - explaining the forecast (approximate, via SHAP)
# ==========================================================================
def explain_forecast(source):
    """
    Fit the Phase 5 model, then use SHAP to attribute each prediction to the
    features that produced it.

    Returns (global_importance_df, local_examples_df) or (None, None) if the
    shap package is unavailable.
    """
    try:
        import shap
    except ImportError:
        print("    shap not installed - skipping Part A")
        print("    install with:  pip install shap")
        return None, None

    df = load_processed(source)
    d = build_features(df).dropna(subset=["lag_1", "roll_mean_4"])
    train = d[d[Canon.WEEK_INDEX] <= VALID_END_WEEK]
    test = d[d[Canon.WEEK_INDEX] > VALID_END_WEEK]

    print(f"    fitting the Phase 5 model on {len(train):,} rows...")
    model = fit_model(train)

    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(test), min(SHAP_SAMPLE, len(test)), replace=False)
    X = test.iloc[idx][FEATURES].reset_index(drop=True)
    bg = train[FEATURES].sample(SHAP_BACKGROUND, random_state=SEED)

    print(f"    computing SHAP values for {len(X)} predictions...")
    # HistGradientBoosting is not supported by the fast TreeExplainer in every
    # shap version, so try it and fall back to the model-agnostic explainer.
    try:
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X)
        method = "TreeExplainer (exact for trees)"
    except Exception:
        explainer = shap.Explainer(model.predict, bg)
        sv = explainer(X).values
        method = "Permutation explainer (model-agnostic)"
    print(f"    method: {method}")

    sv = np.asarray(sv)
    glob = (pd.DataFrame({"feature": FEATURES,
                          "mean_abs_shap": np.abs(sv).mean(axis=0)})
            .sort_values("mean_abs_shap", ascending=False)
            .reset_index(drop=True))
    glob["share_pct"] = glob["mean_abs_shap"] / glob["mean_abs_shap"].sum() * 100

    # a few individual predictions, written out as readable sentences
    meta = test.iloc[idx][[Canon.STORE_ID, Canon.UPC, Canon.WEEK_INDEX,
                           Canon.UNITS, Canon.ANY_PROMO]].reset_index(drop=True)
    rows = []
    interesting = meta.index[meta[Canon.ANY_PROMO] == 1][:3].tolist() + \
        meta.index[meta[Canon.ANY_PROMO] == 0][:2].tolist()
    for i in interesting:
        contrib = pd.Series(sv[i], index=FEATURES).sort_values(
            key=np.abs, ascending=False)
        top = contrib.head(4)
        parts = [f"{k} {'+' if v >= 0 else ''}{v:.2f}" for k, v in top.items()]
        rows.append({
            "store_id": meta.loc[i, Canon.STORE_ID],
            "upc": meta.loc[i, Canon.UPC],
            "week": meta.loc[i, Canon.WEEK_INDEX],
            "actual_units": meta.loc[i, Canon.UNITS],
            "promoted": bool(meta.loc[i, Canon.ANY_PROMO]),
            "top_drivers_log_units": "; ".join(parts),
        })
    return glob, pd.DataFrame(rows)


# ==========================================================================
# PART B - explaining the price recommendation (exact, from the formula)
# ==========================================================================
def explain_prices(rec):
    """
    Rebuild the arithmetic behind every price recommendation.

    Nothing here is approximated. Phase 6 applied a formula and then a clamp,
    so we can restate exactly what happened and check it reproduces the
    recommended price.
    """
    out = []
    for r in rec.itertuples():
        e = r.elasticity
        target_margin = (100 / abs(e)) if abs(e) > 1 else np.nan
        cur_margin = r.current_margin_pct

        if not np.isfinite(e):
            continue
        if e >= ELASTIC_CAP:
            reason = (f"Elasticity {e:.2f} is inelastic (weaker than -1), so "
                      f"profit rises with price without limit. The guardrail, "
                      f"not the formula, sets the price.")
        else:
            reason = (f"Elasticity {e:.2f} implies an optimal margin of "
                      f"{target_margin:.0f}% (Lerner rule). The current margin "
                      f"is {cur_margin:.0f}%, so the unconstrained optimum is "
                      f"{r.unconstrained_price:.2f}.")

        if r.binding_constraint == "none (unconstrained)":
            clamp = "No guardrail was reached; the formula's price was used."
        else:
            clamp = (f"Clamped by '{r.binding_constraint}' to "
                     f"{r.recommended_price:.2f}.")

        direction = ("increase" if r.price_change_pct > 0.05
                     else "decrease" if r.price_change_pct < -0.05 else "hold")
        conf = "high" if r.confident else "low"

        out.append({
            "upc": r.upc,
            "description": r.description,
            "category": r.category,
            "elasticity": round(e, 3),
            "current_price": round(r.current_price, 2),
            "recommended_price": round(r.recommended_price, 2),
            "change_pct": round(r.price_change_pct, 2),
            "direction": direction,
            "current_margin_pct": round(cur_margin, 1),
            "lerner_target_margin_pct": (round(target_margin, 1)
                                         if np.isfinite(target_margin) else None),
            "binding_constraint": r.binding_constraint,
            "confidence": conf,
            "expected_profit_uplift": round(r.profit_uplift, 2),
            "explanation": reason + " " + clamp,
        })
    return pd.DataFrame(out)


# ==========================================================================
# PART C - guardrail audit
# ==========================================================================
def audit_guardrails(rec):
    """Which limits bound, what they cost, and whether the pattern is safe."""
    n = len(rec)
    binding = (rec["binding_constraint"].value_counts()
               .rename_axis("constraint").reset_index(name="products"))
    binding["share_pct"] = binding["products"] / n * 100

    # what would have happened with no guardrails at all
    unc = rec[np.isfinite(rec["unconstrained_price"])].copy()
    if len(unc):
        unc["uncapped_change_pct"] = (unc["unconstrained_price"]
                                      / unc["current_price"] - 1) * 100
        worst = unc.nlargest(3, "uncapped_change_pct")[
            ["description", "elasticity", "current_price",
             "unconstrained_price", "uncapped_change_pct"]]
    else:
        worst = pd.DataFrame()

    # ---- fairness check -------------------------------------------------
    # Raising prices hardest on the cheapest items is regressive: those items
    # are bought disproportionately by price-sensitive shoppers. We check
    # whether the recommendation has that shape.
    r = rec.dropna(subset=["current_price", "price_change_pct"]).copy()
    r["price_quartile"] = pd.qcut(r["current_price"], 4,
                                  labels=["cheapest", "low", "high", "dearest"])
    by_q = (r.groupby("price_quartile", observed=True)
            .agg(products=("upc", "count"),
                 mean_price=("current_price", "mean"),
                 mean_change_pct=("price_change_pct", "mean"),
                 share_increased=("price_change_pct", lambda s: (s > 0).mean() * 100))
            .round(2).reset_index())
    corr = r["current_price"].corr(r["price_change_pct"])

    return binding, worst, by_q, corr


# ==========================================================================
# Orchestration
# ==========================================================================
def run(source):
    print("=" * 76)
    print(f"PHASE 9 - EXPLAINABILITY AND GUARDRAIL AUDIT   [{source}]")
    print("=" * 76)

    rec_path = OUTPUTS_DIR / f"price_recommendations_{source}.csv"
    if not rec_path.exists():
        print(f"    MISSING {rec_path.name} - run  python src/pricing.py {source}")
        return None
    rec = pd.read_csv(rec_path)

    # ------------------------------------------------------------- PART A
    print("\n[1] PART A - what drives the demand forecast (SHAP, approximate)")
    glob, local = explain_forecast(source)
    if glob is not None:
        print(f"\n    {'feature':<20}{'mean |SHAP|':>13}{'share':>9}")
        print("    " + "-" * 42)
        for r in glob.head(8).itertuples():
            bar = "#" * max(1, int(r.share_pct / 2))
            print(f"    {r.feature:<20}{r.mean_abs_shap:>13.4f}"
                  f"{r.share_pct:>8.1f}%  {bar}")
        top3 = glob.head(3)["share_pct"].sum()
        print(f"\n    the top three features carry {top3:.0f}% of the explanation")

        print("\n    Individual predictions (contribution to log units):")
        for r in local.itertuples():
            tag = "promoted" if r.promoted else "no promo"
            print(f"      store {r.store_id}, upc {r.upc}, week {r.week} "
                  f"({tag}, sold {r.actual_units})")
            print(f"        {r.top_drivers_log_units}")

    # ------------------------------------------------------------- PART B
    print("\n[2] PART B - why each price was recommended (exact, from the formula)")
    exp = explain_prices(rec)
    print(f"    explained {len(exp)} of {len(rec)} products")
    print(f"    directions: " + ", ".join(
        f"{k} {v}" for k, v in exp["direction"].value_counts().items()))
    print(f"    confidence: " + ", ".join(
        f"{k} {v}" for k, v in exp["confidence"].value_counts().items()))

    print("\n    Three worked examples:")
    picks = pd.concat([exp.nlargest(1, "change_pct"),
                       exp.nsmallest(1, "change_pct"),
                       exp.nlargest(1, "expected_profit_uplift")]).drop_duplicates("upc")
    for r in picks.itertuples():
        print(f"\n      {str(r.description)[:34]}  ({r.category.title()})")
        print(f"        {r.current_price:.2f} -> {r.recommended_price:.2f}  "
              f"({r.change_pct:+.1f}%, confidence {r.confidence})")
        # wrap the explanation to the terminal
        words, line = r.explanation.split(), "        "
        for w in words:
            if len(line) + len(w) > 74:
                print(line)
                line = "        "
            line += w + " "
        print(line)

    # ------------------------------------------------------------- PART C
    print("\n[3] PART C - guardrail audit")
    binding, worst, by_q, corr = audit_guardrails(rec)

    print(f"\n    Which limit actually binds:")
    print(f"    {'constraint':<30}{'products':>10}{'share':>9}")
    print("    " + "-" * 49)
    for r in binding.itertuples():
        print(f"    {r.constraint:<30}{r.products:>10}{r.share_pct:>8.0f}%")

    if len(worst):
        print(f"\n    What the guardrails prevented (uncapped optimum):")
        print(f"    {'product':<28}{'elast':>7}{'now':>8}{'uncapped':>10}{'change':>10}")
        print("    " + "-" * 63)
        for r in worst.itertuples():
            print(f"    {str(r.description)[:26]:<28}{r.elasticity:>7.2f}"
                  f"{r.current_price:>8.2f}{r.unconstrained_price:>10.2f}"
                  f"{r.uncapped_change_pct:>9.0f}%")
        print("    Without the cap the system would have proposed these rises.")
        print("    They are the reason the guardrail exists.")

    print(f"\n    Fairness check - are rises concentrated on cheap staples?")
    print(f"    {'price band':<14}{'products':>10}{'mean price':>12}"
          f"{'mean change':>13}{'% raised':>10}")
    print("    " + "-" * 59)
    for r in by_q.itertuples():
        print(f"    {str(r.price_quartile):<14}{r.products:>10}"
              f"{r.mean_price:>12.2f}{r.mean_change_pct:>12.1f}%"
              f"{r.share_increased:>9.0f}%")
    print(f"\n    corr(current price, recommended change) = {corr:+.2f}")
    if corr < -0.25:
        print("    NEGATIVE: rises fall harder on cheaper products. This is")
        print("    regressive and should be reviewed before any deployment.")
    elif corr > 0.25:
        print("    POSITIVE: rises fall on dearer products, not on cheap staples.")
    else:
        print("    Near zero: no systematic bias by price level.")

    # ------------------------------------------------------------- write
    if glob is not None:
        glob.round(5).to_csv(OUTPUTS_DIR / f"shap_importance_{source}.csv",
                             index=False)
        local.to_csv(OUTPUTS_DIR / f"shap_examples_{source}.csv", index=False)
    exp.to_csv(OUTPUTS_DIR / f"price_explanations_{source}.csv", index=False)
    binding.round(2).to_csv(OUTPUTS_DIR / f"guardrail_audit_{source}.csv",
                            index=False)
    by_q.to_csv(OUTPUTS_DIR / f"fairness_check_{source}.csv", index=False)

    print(f"\n[4] Written: price_explanations_{source}.csv, "
          f"guardrail_audit_{source}.csv,")
    print(f"    fairness_check_{source}.csv"
          + (f", shap_importance_{source}.csv, shap_examples_{source}.csv"
             if glob is not None else ""))
    print("=" * 76)
    return exp


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else None
    for s in ([which] if which else ["synthetic", "real"]):
        run(s)
        print()