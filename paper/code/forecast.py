"""
Phase 5 - Demand forecasting.

WHY THIS PHASE EXISTS
---------------------
Phases 3 and 4 told us how demand REACTS to price and promotions. This phase
tells us how much will sell in the first place. Everything downstream needs it:

    price optimisation  needs baseline demand to scale by elasticity
    markdown planning   needs to know how much stock will clear on its own
    inventory policy    needs a demand forecast and its uncertainty

TWO FORECASTS, NOT ONE
----------------------
1. ACTUAL-CONDITIONS forecast - predicts units given the price and promotions
   that really ran. This is what we score for accuracy.

2. BASELINE forecast - predicts units assuming NO promotion and full base
   price. This is the clean "what would sell anyway" number that the pricing
   and markdown modules optimise around. Without it we would be optimising on
   top of demand that a promotion already inflated.

HONEST EVALUATION
-----------------
Most student forecasting projects report an accuracy number with nothing to
compare it against, so nobody can tell whether the model is any good. We
therefore run three naive baselines first:

    last week      - repeat last week's units
    same week last year - the seasonal naive, genuinely hard to beat in retail
    4-week average - a simple smoother

A machine learning model that cannot beat "same week last year" is not worth
deploying, and saying so is more useful than quoting an impressive-looking
percentage.

We report WAPE (weighted absolute percentage error), not MAPE. MAPE divides by
actual units, so a single slow-moving product that sold 1 unit can produce a
500% error and dominate the average. WAPE weights by volume and is what
retailers actually use.

NO LEAKAGE
----------
Every lag and rolling feature is shifted so it only ever sees the past. Splits
are strictly by time. The test weeks (131-156) are scored exactly once.

Run:
    python src/forecast.py             # both sources
    python src/forecast.py real
    python src/forecast.py synthetic
"""

import sys
import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance

from config import OUTPUTS_DIR, TRAIN_END_WEEK, VALID_END_WEEK
from schema import Canon
from data_loader import load_processed

warnings.filterwarnings("ignore")

SEED = 42
LAGS = [1, 2, 3, 4, 52]
ROLL = [4, 8, 13]


# ==========================================================================
# Metrics
# ==========================================================================
def wape(actual, pred):
    """Weighted absolute percentage error - the retail standard."""
    a = np.asarray(actual, dtype=float)
    p = np.asarray(pred, dtype=float)
    denom = np.abs(a).sum()
    return float(np.abs(a - p).sum() / denom) if denom > 0 else np.nan


def score(actual, pred, label):
    a = np.asarray(actual, dtype=float)
    p = np.asarray(pred, dtype=float)
    ok = np.isfinite(a) & np.isfinite(p)
    a, p = a[ok], p[ok]
    return {
        "model": label,
        "n": len(a),
        "WAPE": wape(a, p),
        "MAE": float(np.abs(a - p).mean()),
        "RMSE": float(np.sqrt(((a - p) ** 2).mean())),
        "bias_pct": float((p.sum() - a.sum()) / a.sum() * 100) if a.sum() else np.nan,
    }


# ==========================================================================
# Feature engineering
# ==========================================================================
def build_features(df):
    """
    Build the modelling table. All history features are shifted by at least one
    week inside each store-product series, so nothing can see its own present.
    """
    d = df.sort_values([Canon.STORE_ID, Canon.UPC, Canon.WEEK_INDEX]).copy()
    key = [Canon.STORE_ID, Canon.UPC]
    g = d.groupby(key, sort=False)[Canon.UNITS]

    for L in LAGS:
        d[f"lag_{L}"] = g.shift(L)

    # Roll over the ALREADY-SHIFTED series, and re-group before rolling so the
    # window never spans two different store-product series. Rolling straight
    # off groupby().shift() silently leaks one product's history into the next.
    d["_shifted"] = g.shift(1)
    roll = d.groupby(key, sort=False)["_shifted"]
    for R in ROLL:
        d[f"roll_mean_{R}"] = (roll.rolling(R, min_periods=2).mean()
                               .reset_index(level=[0, 1], drop=True))
    d["roll_std_4"] = (roll.rolling(4, min_periods=2).std()
                       .reset_index(level=[0, 1], drop=True))

    # calendar
    woy = (d[Canon.WEEK_INDEX] - 1) % 52
    d["week_of_year"] = woy
    d["sin1"] = np.sin(2 * np.pi * woy / 52)
    d["cos1"] = np.cos(2 * np.pi * woy / 52)
    d["sin2"] = np.sin(4 * np.pi * woy / 52)
    d["cos2"] = np.cos(4 * np.pi * woy / 52)

    # price / promo. These are KNOWN in advance in real planning: retailers
    # commit to next week's price and promo calendar before the week starts,
    # so using them as features is legitimate, not leakage.
    d["price_ratio"] = (d[Canon.PRICE] / d[Canon.BASE_PRICE]).clip(0.05, 1.5)
    d["log_price_ratio"] = np.log(d["price_ratio"])

    # How this week's price compares with the recent norm for this product in
    # this store. Kept as its own column so the counterfactual in
    # predict_baseline_demand can simply re-divide by it.
    d["_hist_price"] = (d.groupby(key, sort=False)[Canon.PRICE].shift(1)
                        .groupby([d[Canon.STORE_ID], d[Canon.UPC]])
                        .rolling(13, min_periods=2).mean()
                        .reset_index(level=[0, 1], drop=True))
    d["price_vs_hist"] = d[Canon.PRICE] / d["_hist_price"].replace(0, np.nan)

    # categorical -> integer codes for the tree model
    for col, name in [(Canon.CATEGORY, "cat_code"),
                      (Canon.SUB_CATEGORY, "sub_code"),
                      (Canon.MANUFACTURER, "mfr_code"),
                      (Canon.PRICE_TIER, "tier_code")]:
        d[name] = pd.factorize(d[col])[0]

    return d


FEATURES = ([f"lag_{L}" for L in LAGS]
            + [f"roll_mean_{R}" for R in ROLL]
            + ["roll_std_4", "week_of_year", "sin1", "cos1", "sin2", "cos2",
               "price_ratio", "log_price_ratio", "price_vs_hist",
               Canon.FEATURE, Canon.DISPLAY, Canon.TPR_ONLY,
               "cat_code", "sub_code", "mfr_code", "tier_code"])


# ==========================================================================
# Baselines
# ==========================================================================
def baselines(d):
    """Simple rules any retailer could run without a data scientist."""
    return {
        "naive (last week)": d["lag_1"],
        "seasonal naive (52w)": d["lag_52"],
        "moving avg 4w": d["roll_mean_4"],
    }


# ==========================================================================
# Model
# ==========================================================================
def fit_model(train):
    """
    Gradient boosting on log units.

    We model log(units) because retail demand is multiplicative and right
    skewed: a promotion doubles sales rather than adding a fixed amount. Fitting
    in logs also stops a handful of very high volume store-weeks from
    dominating the loss.
    """
    X = train[FEATURES]
    y = np.log1p(train[Canon.UNITS])
    m = HistGradientBoostingRegressor(
        max_iter=400, learning_rate=0.06, max_depth=7,
        min_samples_leaf=40, l2_regularization=1.0,
        early_stopping=True, validation_fraction=0.12,
        random_state=SEED,
    )
    m.fit(X, y)

    # DUAN'S SMEARING CORRECTION.
    # Fitting in logs then exponentiating systematically UNDER-predicts, because
    # exp() is convex: the average of exp(errors) is above exp(average error).
    # Left uncorrected this showed up as a -5% bias, which downstream becomes
    # under-ordering and stockouts. The fix is a single multiplier estimated
    # from the training residuals.
    resid = y - m.predict(X)
    m._smear = float(np.mean(np.exp(resid)))
    return m


def predict(model, frame):
    smear = getattr(model, "_smear", 1.0)
    return (np.expm1(model.predict(frame[FEATURES])) * smear).clip(0)


def predict_baseline_demand(model, frame):
    """
    Counterfactual: what would sell with NO promotion at FULL base price?

    This is the number the pricing and markdown modules need. Asking the model
    to score a row where we have switched every promotion off and reset the
    price to base gives us demand stripped of promotional inflation.
    """
    f = frame.copy()
    f[Canon.FEATURE] = 0
    f[Canon.DISPLAY] = 0
    f[Canon.TPR_ONLY] = 0
    f["price_ratio"] = 1.0
    f["log_price_ratio"] = 0.0
    f["price_vs_hist"] = (f[Canon.BASE_PRICE]
                          / f["_hist_price"].replace(0, np.nan)).fillna(1.0)
    smear = getattr(model, "_smear", 1.0)
    return (np.expm1(model.predict(f[FEATURES])) * smear).clip(0)


# ==========================================================================
# Orchestration
# ==========================================================================
def run(source):
    print("=" * 74)
    print(f"PHASE 5 - DEMAND FORECASTING   [{source}]")
    print("=" * 74)

    df = load_processed(source)
    d = build_features(df)

    train = d[d[Canon.WEEK_INDEX] <= TRAIN_END_WEEK]
    valid = d[(d[Canon.WEEK_INDEX] > TRAIN_END_WEEK)
              & (d[Canon.WEEK_INDEX] <= VALID_END_WEEK)]
    test = d[d[Canon.WEEK_INDEX] > VALID_END_WEEK]

    # rows without enough history cannot be modelled or scored fairly
    need = ["lag_1", "roll_mean_4"]
    train_f = train.dropna(subset=need)
    valid_f = valid.dropna(subset=need)
    test_f = test.dropna(subset=need)

    print(f"\n[1] Data")
    print(f"    train  weeks   1-{TRAIN_END_WEEK:<3} {len(train_f):>8,} usable rows")
    print(f"    valid  weeks {TRAIN_END_WEEK+1:>3}-{VALID_END_WEEK:<3} {len(valid_f):>8,} usable rows")
    print(f"    test   weeks {VALID_END_WEEK+1:>3}-156 {len(test_f):>8,} usable rows")
    print(f"    features: {len(FEATURES)}")

    # ------------------------------------------------------------- baselines
    print("\n[2] Baselines on the VALIDATION weeks")
    print(f"    {'model':<24}{'WAPE':>9}{'MAE':>9}{'RMSE':>9}{'bias %':>9}")
    print("    " + "-" * 60)
    results = []
    for name, pred in baselines(valid_f).items():
        ok = pred.notna()
        r = score(valid_f.loc[ok, Canon.UNITS], pred[ok], name)
        results.append(r)
        print(f"    {r['model']:<24}{r['WAPE']:>9.3f}{r['MAE']:>9.2f}"
              f"{r['RMSE']:>9.2f}{r['bias_pct']:>9.1f}")

    # ----------------------------------------------------------------- model
    print("\n[3] Training gradient boosting on log units...")
    model = fit_model(train_f)
    pv = predict(model, valid_f)
    r = score(valid_f[Canon.UNITS], pv, "gradient boosting")
    results.append(r)
    print(f"    {r['model']:<24}{r['WAPE']:>9.3f}{r['MAE']:>9.2f}"
          f"{r['RMSE']:>9.2f}{r['bias_pct']:>9.1f}")

    best_base = min([x for x in results if x["model"] != "gradient boosting"],
                    key=lambda x: x["WAPE"])
    gain = (best_base["WAPE"] - r["WAPE"]) / best_base["WAPE"] * 100
    print(f"\n    best baseline = {best_base['model']} (WAPE {best_base['WAPE']:.3f})")
    if gain > 0:
        print(f"    gradient boosting is {gain:.1f}% better -> worth using")
    else:
        print(f"    gradient boosting is NOT better ({gain:.1f}%) -> keep the baseline")

    # ------------------------------------------------- final refit, score once
    print("\n[4] Refit on train+valid, then score the TEST weeks ONCE")
    final = fit_model(pd.concat([train_f, valid_f]))
    pt = predict(final, test_f)
    rt = score(test_f[Canon.UNITS], pt, "gradient boosting (test)")
    print(f"    {'model':<24}{'WAPE':>9}{'MAE':>9}{'RMSE':>9}{'bias %':>9}")
    print("    " + "-" * 60)
    for name, pred in baselines(test_f).items():
        ok = pred.notna()
        rb = score(test_f.loc[ok, Canon.UNITS], pred[ok], name)
        print(f"    {rb['model']:<24}{rb['WAPE']:>9.3f}{rb['MAE']:>9.2f}"
              f"{rb['RMSE']:>9.2f}{rb['bias_pct']:>9.1f}")
        results.append({**rb, "split": "test"})
    print(f"    {rt['model']:<24}{rt['WAPE']:>9.3f}{rt['MAE']:>9.2f}"
          f"{rt['RMSE']:>9.2f}{rt['bias_pct']:>9.1f}")
    results.append({**rt, "split": "test"})

    # ------------------------------------------------------ accuracy by group
    print("\n[5] Test accuracy by category")
    t = test_f.copy()
    t["pred"] = pt
    print(f"    {'category':<24}{'WAPE':>9}{'mean units':>12}")
    print("    " + "-" * 46)
    for cat, gc in t.groupby(Canon.CATEGORY):
        print(f"    {cat:<24}{wape(gc[Canon.UNITS], gc['pred']):>9.3f}"
              f"{gc[Canon.UNITS].mean():>12.1f}")

    # ------------------------------------------------------------ importance
    print("\n[6] What drives the forecast (permutation importance, sampled)")
    samp = valid_f.sample(min(6000, len(valid_f)), random_state=SEED)
    pi = permutation_importance(
        final, samp[FEATURES], np.log1p(samp[Canon.UNITS]),
        n_repeats=3, random_state=SEED, scoring="neg_mean_absolute_error")
    imp = (pd.DataFrame({"feature": FEATURES, "importance": pi.importances_mean})
           .sort_values("importance", ascending=False))
    for _, row in imp.head(8).iterrows():
        bar = "#" * max(1, int(row.importance / imp.importance.max() * 34))
        print(f"    {row.feature:<18} {row.importance:>7.4f}  {bar}")

    # -------------------------------------------------- baseline demand output
    print("\n[7] Baseline demand (no promo, full price) for downstream phases")
    t["baseline_demand"] = predict_baseline_demand(final, test_f)
    promo_rows = t[t[Canon.ANY_PROMO] == 1]
    if len(promo_rows):
        infl = promo_rows["pred"].sum() / max(promo_rows["baseline_demand"].sum(), 1e-9)
        print(f"    on promoted store-weeks the forecast is {infl:.2f}x the baseline")
        print("    -> that gap is the promotional inflation the pricing and")
        print("       markdown phases must not mistake for organic demand")

    out_cols = [Canon.WEEK_INDEX, Canon.STORE_ID, Canon.UPC, Canon.CATEGORY,
                Canon.UNITS, "pred", "baseline_demand", Canon.PRICE,
                Canon.BASE_PRICE, Canon.UNIT_COST, Canon.ANY_PROMO]
    t[out_cols].round(3).to_csv(OUTPUTS_DIR / f"forecast_test_{source}.csv", index=False)
    pd.DataFrame(results).round(4).to_csv(
        OUTPUTS_DIR / f"forecast_metrics_{source}.csv", index=False)
    imp.round(5).to_csv(OUTPUTS_DIR / f"forecast_importance_{source}.csv", index=False)

    print(f"\n[8] Written: forecast_test_{source}.csv, forecast_metrics_{source}.csv, "
          f"forecast_importance_{source}.csv")
    print("=" * 74)
    return final


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else None
    for s in ([which] if which else ["synthetic", "real"]):
        run(s)
        print()
