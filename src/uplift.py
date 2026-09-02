"""
Phase 4 - Causal promotion uplift engine.

THE QUESTION
------------
The retailer ran three different promotion mechanics:
    FEATURE   - the product appeared in the weekly printed ad / mailer
    DISPLAY   - the product got a special in-store display
    TPR_ONLY  - a temporary price reduction with no other support

Which of them actually CAUSED extra sales, how much, and was the extra volume
worth the margin given away? A naive before/after comparison cannot answer this,
because promotions are not run at random: retailers promote in high season and
push harder in certain stores.

THE METHOD - difference in differences, via two-way fixed effects
----------------------------------------------------------------
For a given product in a given week, some stores are running a display and
others are not. Those non-promoting stores are our control group. We estimate

    log(units) = a_(store x upc) + d_(week) + b_F*FEATURE + b_D*DISPLAY
                 + b_T*TPR_ONLY + g*log(price/base_price) + e

  a_(store x upc)  absorbs everything permanent about a product in a store
                   (shelf position, local popularity, store size)
  d_(week)         absorbs everything common to a week
                   (season, holidays, weather, the economy)
  g*log(price...)  separates the effect of the PRICE CUT from the effect of the
                   MERCHANDISING. Without this, the display coefficient would
                   quietly absorb the discount that usually accompanies it.

So b_D is the lift from the display itself, holding price constant. That
separation is what lets us compare mechanics fairly.

THE PARALLEL TRENDS CHECK
-------------------------
Difference in differences is only valid if treated and control stores were
moving in step BEFORE the promotion. We test this with a placebo: pretend the
promotion happened two weeks earlier than it did, and re-estimate. If we find a
large "effect" before anything happened, the design is broken. A near-zero
placebo supports the assumption.

CONNECTION TO PHASE 3
---------------------
The profit calculation reuses the elasticity estimated in Phase 3 to predict
how much of the observed volume came from the price cut, so we can charge the
promotion for the margin it gave away.

Run:
    python src/uplift.py             # both sources
    python src/uplift.py real
    python src/uplift.py synthetic
"""

import sys
import warnings
import numpy as np
import pandas as pd

from config import PROCESSED_DIR, OUTPUTS_DIR
from schema import Canon
from data_loader import load_processed

warnings.filterwarnings("ignore")

TREATMENTS = [Canon.FEATURE, Canon.DISPLAY, Canon.TPR_ONLY]
NICE = {Canon.FEATURE: "FEATURE (mailer)",
        Canon.DISPLAY: "DISPLAY (in-store)",
        Canon.TPR_ONLY: "TPR_ONLY (price cut)"}
MIN_OBS = 500
MIN_TREATED = 30


# ==========================================================================
# Linear algebra helpers
# ==========================================================================
def _group_demean(M, codes, n_groups):
    """Subtract group means from every column of M."""
    M = np.asarray(M, dtype=float)
    single = M.ndim == 1
    if single:
        M = M[:, None]
    out = np.empty_like(M)
    counts = np.bincount(codes, minlength=n_groups).astype(float)
    counts[counts == 0] = 1.0
    for j in range(M.shape[1]):
        sums = np.bincount(codes, weights=M[:, j], minlength=n_groups)
        out[:, j] = M[:, j] - (sums / counts)[codes]
    return out.ravel() if single else out


def absorb_two_way(M, codes_a, n_a, codes_b, n_b, iters=12):
    """
    Remove two sets of fixed effects by alternating projections.

    Demeaning by A breaks the demeaning by B and vice versa, so we alternate
    until it settles. This is the standard way to fit models with two
    high-dimensional fixed effects without building thousands of dummy columns.
    """
    out = np.asarray(M, dtype=float).copy()
    single = out.ndim == 1
    if single:
        out = out[:, None]
    for _ in range(iters):
        prev = out
        out = _group_demean(out, codes_a, n_a)
        out = _group_demean(out, codes_b, n_b)
        if np.max(np.abs(out - prev)) < 1e-9:
            break
    return out.ravel() if single else out


def ols(X, y):
    """Return (coefficients, standard errors)."""
    XtX = X.T @ X
    try:
        inv = np.linalg.inv(XtX)
    except np.linalg.LinAlgError:
        inv = np.linalg.pinv(XtX)
    beta = inv @ (X.T @ y)
    resid = y - X @ beta
    dof = max(len(y) - X.shape[1], 1)
    se = np.sqrt(np.maximum(np.diag(inv) * (float(resid @ resid) / dof), 0))
    return beta, se


# ==========================================================================
# The DiD estimator
# ==========================================================================
def did_estimate(d, control_price=True, treat_cols=None):
    """
    Two-way fixed effects DiD on one slice of the panel.
    Returns {treatment: (coefficient, standard_error)}.
    """
    treat_cols = treat_cols or TREATMENTS
    codes_cell, uniq_cell = pd.factorize(
        d[Canon.STORE_ID].astype(str) + "_" + d[Canon.UPC].astype(str))
    codes_week, uniq_week = pd.factorize(d[Canon.WEEK_INDEX])

    cols = [d[t].to_numpy(dtype=float) for t in treat_cols]
    names = list(treat_cols)
    if control_price:
        cols.append(d["log_price_ratio"].to_numpy())
        names.append("log_price_ratio")

    X = np.column_stack(cols)
    y = d["log_units"].to_numpy()

    Xa = absorb_two_way(X, codes_cell, len(uniq_cell), codes_week, len(uniq_week))
    ya = absorb_two_way(y, codes_cell, len(uniq_cell), codes_week, len(uniq_week))

    keep = Xa.std(axis=0) > 1e-10
    if not keep.any():
        return {}
    beta, se = ols(Xa[:, keep], ya)

    result, k = {}, 0
    for i, nm in enumerate(names):
        if keep[i]:
            result[nm] = (float(beta[k]), float(se[k]))
            k += 1
    return result


def naive_lift(d, treat):
    """The biased comparison a basic project would report."""
    on = d.loc[d[treat] == 1, Canon.UNITS]
    off = d.loc[d[Canon.ANY_PROMO] == 0, Canon.UNITS]
    if len(on) < MIN_TREATED or len(off) < MIN_TREATED:
        return np.nan
    return float(np.log(on.mean()) - np.log(off.mean()))


# ==========================================================================
# Parallel-trends placebo
# ==========================================================================
def placebo_test(d, lead_weeks=2):
    """
    Shift every treatment flag EARLIER by `lead_weeks` within each store-product
    series, then re-run the DiD. We are asking the model to detect an effect in
    weeks before the promotion actually ran. A real effect there is impossible,
    so a large coefficient means the parallel-trends assumption has failed.
    """
    d = d.sort_values([Canon.STORE_ID, Canon.UPC, Canon.WEEK_INDEX]).copy()
    g = d.groupby([Canon.STORE_ID, Canon.UPC], sort=False)
    fake = []
    for t in TREATMENTS:
        col = f"placebo_{t}"
        d[col] = g[t].shift(-lead_weeks)
        fake.append(col)
    d = d.dropna(subset=fake)
    if len(d) < MIN_OBS:
        return {}
    res = did_estimate(d, control_price=True, treat_cols=fake)
    return {t: res.get(f"placebo_{t}", (np.nan, np.nan)) for t in TREATMENTS}


# ==========================================================================
# Profit consequence of each mechanic
# ==========================================================================
def promo_profit(d, coefs, elasticity_map):
    """
    Was the lift worth the margin given away?

    baseline    : units and price when nothing is running
    with promo  : baseline x exp(merchandising effect) x price effect
                  where the price effect uses the Phase 3 elasticity
    We then compare gross profit per store-week under each scenario.
    """
    rows = []
    base = d[d[Canon.ANY_PROMO] == 0]
    if len(base) < MIN_TREATED:
        return pd.DataFrame()

    for upc, gp in d.groupby(Canon.UPC):
        gbase = gp[gp[Canon.ANY_PROMO] == 0]
        if len(gbase) < MIN_TREATED:
            continue
        base_units = float(gbase[Canon.UNITS].mean())
        base_price = float(gbase[Canon.PRICE].mean())
        cost = float(gp[Canon.UNIT_COST].mean())
        elast = elasticity_map.get(upc, np.nan)
        if not np.isfinite(elast):
            continue

        for t in TREATMENTS:
            gt = gp[gp[t] == 1]
            if len(gt) < MIN_TREATED:
                continue
            b = coefs.get(t, (np.nan, np.nan))[0]
            if not np.isfinite(b):
                continue
            promo_price = float(gt[Canon.PRICE].mean())
            # volume = merchandising lift x price response
            price_effect = (promo_price / base_price) ** elast
            promo_units = base_units * np.exp(b) * price_effect

            profit_base = base_units * (base_price - cost)
            profit_promo = promo_units * (promo_price - cost)
            rows.append({
                "upc": upc,
                "category": gp[Canon.CATEGORY].iloc[0],
                "description": gp[Canon.DESCRIPTION].iloc[0],
                "treatment": t,
                "merch_lift_pct": (np.exp(b) - 1) * 100,
                "discount_pct": (1 - promo_price / base_price) * 100,
                "base_units": base_units,
                "promo_units": promo_units,
                "volume_lift_pct": (promo_units / base_units - 1) * 100,
                "profit_base": profit_base,
                "profit_promo": profit_promo,
                "incremental_profit": profit_promo - profit_base,
                "profitable": profit_promo > profit_base,
                # Sold below cost: no volume lift can ever recover this on the
                # item itself. Retailers do it deliberately as a loss leader to
                # pull shoppers in - a basket effect our item-level model
                # cannot see, so we flag it rather than judge it.
                "below_cost": promo_price < cost,
                "breakeven_lift_pct": (((base_price - cost) / (promo_price - cost) - 1) * 100
                                       if promo_price > cost else np.nan),
            })
    return pd.DataFrame(rows)


# ==========================================================================
# Orchestration
# ==========================================================================
def run(source):
    print("=" * 72)
    print(f"PHASE 4 - CAUSAL PROMOTION UPLIFT   [{source}]")
    print("=" * 72)

    df = load_processed(source)
    d = df[df[Canon.SPLIT] == "train"].copy()
    d["log_units"] = np.log(d[Canon.UNITS].clip(lower=1))
    d["log_price_ratio"] = np.log((d[Canon.PRICE] / d[Canon.BASE_PRICE]).clip(0.05, 1.5))
    print(f"\n[1] Training rows: {len(d):,}")
    for t in TREATMENTS:
        print(f"    {NICE[t]:<22} active in {d[t].mean():>5.1%} of rows")

    # ------------------------------------------------------------ naive vs DiD
    print("\n[2] Naive comparison vs difference-in-differences")
    print(f"    {'mechanic':<22} {'naive':>9} {'DiD':>9} {'std err':>9} {'lift %':>9}")
    print("    " + "-" * 62)
    overall = did_estimate(d)
    for t in TREATMENTS:
        nv = naive_lift(d, t)
        b, se = overall.get(t, (np.nan, np.nan))
        print(f"    {NICE[t]:<22} {nv:>9.3f} {b:>9.3f} {se:>9.4f} "
              f"{(np.exp(b) - 1) * 100:>8.1f}%")
    gp = overall.get("log_price_ratio", (np.nan, np.nan))[0]
    print(f"\n    price coefficient (elasticity control): {gp:.3f}")
    print("    Naive is inflated because it credits the promotion with the")
    print("    season, the store, and the price cut all at once.")

    # ------------------------------------------------------ parallel trends
    print("\n[3] Parallel-trends placebo (effect 2 weeks BEFORE the promotion)")
    pl = placebo_test(d, lead_weeks=2)
    print(f"    {'mechanic':<22} {'placebo':>9} {'real DiD':>10} {'verdict':>12}")
    print("    " + "-" * 56)
    for t in TREATMENTS:
        pb = pl.get(t, (np.nan, np.nan))[0]
        rb = overall.get(t, (np.nan, np.nan))[0]
        # Judge on ABSOLUTE size first. A ratio test alone is meaningless when
        # the real effect is itself near zero - any tiny placebo then looks
        # enormous in relative terms while being economically irrelevant.
        if not np.isfinite(pb) or not np.isfinite(rb):
            verdict = "n/a"
        elif abs(pb) < 0.15:                      # < ~16% pre-trend: negligible
            verdict = "PASS"
        elif abs(pb) < 0.3 * abs(rb):
            verdict = "PASS"
        elif abs(pb) < 0.6 * abs(rb):
            verdict = "WEAK"
        else:
            verdict = "FAIL"
        print(f"    {NICE[t]:<22} {pb:>9.3f} {rb:>10.3f} {verdict:>12}")
    print("    PASS = little happens before the promotion, so the control")
    print("           stores are a fair comparison.")

    # ------------------------------------------------------------ by category
    print("\n[4] Uplift by category (percent lift, merchandising only)")
    hdr = f"    {'category':<24}" + "".join(f"{NICE[t].split()[0]:>12}" for t in TREATMENTS)
    print(hdr)
    print("    " + "-" * (24 + 12 * len(TREATMENTS)))
    cat_rows = []
    for cat, gc in d.groupby(Canon.CATEGORY):
        if len(gc) < MIN_OBS:
            continue
        res = did_estimate(gc)
        line = f"    {cat:<24}"
        rec = {"category": cat}
        for t in TREATMENTS:
            b = res.get(t, (np.nan, np.nan))[0]
            line += f"{(np.exp(b) - 1) * 100:>11.1f}%" if np.isfinite(b) else f"{'n/a':>12}"
            rec[t] = b
        print(line)
        cat_rows.append(rec)

    # ------------------------------------------------------ validate on truth
    truth_path = PROCESSED_DIR / "ground_truth_lift.csv"
    if source == "synthetic" and truth_path.exists():
        print("\n[5] VALIDATION against known ground truth")
        truth = pd.read_csv(truth_path).iloc[0]
        print(f"    {'mechanic':<22} {'true':>8} {'naive':>9} {'DiD':>8} "
              f"{'naive err':>10} {'DiD err':>9}")
        print("    " + "-" * 70)
        tn = te = 0.0
        for t in TREATMENTS:
            tv = float(truth[t])
            nv = naive_lift(d, t)
            b = overall.get(t, (np.nan, np.nan))[0]
            tn += abs(nv - tv)
            te += abs(b - tv)
            print(f"    {NICE[t]:<22} {tv:>8.3f} {nv:>9.3f} {b:>8.3f} "
                  f"{abs(nv - tv):>10.3f} {abs(b - tv):>9.3f}")
        print(f"\n    mean absolute error   naive {tn / 3:.3f}   ->   DiD {te / 3:.3f}"
              f"   ({tn / max(te, 1e-9):.1f}x better)")

    # ------------------------------------------------------------ profitability
    print("\n[6] Is each mechanic worth the margin given away?")
    ela_path = OUTPUTS_DIR / f"elasticity_{source}.csv"
    if not ela_path.exists():
        print(f"    SKIPPED - run  python src/elasticity.py {source}  first")
        prof = pd.DataFrame()
    else:
        ela = pd.read_csv(ela_path)
        emap = dict(zip(ela["upc"], ela["elast_final"]))
        prof = promo_profit(d, overall, emap)
        if prof.empty:
            print("    not enough data")
        else:
            print(f"    {'mechanic':<22} {'prods':>6} {'profitable':>11} "
                  f"{'below cost':>11} {'avg lift':>9} {'needed':>9} {'avg profit':>11}")
            print("    " + "-" * 82)
            for t in TREATMENTS:
                sub = prof[prof.treatment == t]
                if sub.empty:
                    continue
                need = sub["breakeven_lift_pct"].median()
                need_s = f"{need:>8.0f}%" if np.isfinite(need) else f"{'n/a':>9}"
                print(f"    {NICE[t]:<22} {len(sub):>6} {sub.profitable.mean():>10.0%} "
                      f"{sub.below_cost.mean():>10.0%} {sub.volume_lift_pct.mean():>8.1f}% "
                      f"{need_s} {sub.incremental_profit.mean():>11.2f}")
            print("\n    'needed' = volume lift required just to break even on")
            print("    the thinner promo margin. Compare it with 'avg lift'.")
            worst = prof.nsmallest(3, "incremental_profit")
            print("\n    Worst promotions (biggest profit destroyers per store-week):")
            for _, r in worst.iterrows():
                tag = " [BELOW COST]" if r.below_cost else ""
                print(f"      {r.description[:26]:<28} {r.treatment:<9} "
                      f"lift {r.volume_lift_pct:>6.1f}%  disc {r.discount_pct:>5.1f}%  "
                      f"profit {r.incremental_profit:>8.2f}{tag}")
            n_bc = int(prof.below_cost.sum())
            if n_bc:
                print(f"\n    NOTE: {n_bc} product-mechanic pairs priced BELOW COST.")
                print("    These are loss leaders. They lose money on the item by design,")
                print("    to pull shoppers into the store. Our item-level model cannot see")
                print("    that basket effect, so we flag them rather than call them mistakes.")

    # ------------------------------------------------------------------ write
    pd.DataFrame([{"treatment": t,
                   "did_coef": overall.get(t, (np.nan, np.nan))[0],
                   "did_se": overall.get(t, (np.nan, np.nan))[1],
                   "lift_pct": (np.exp(overall.get(t, (np.nan,))[0]) - 1) * 100,
                   "naive_coef": naive_lift(d, t),
                   "placebo_coef": pl.get(t, (np.nan, np.nan))[0]}
                  for t in TREATMENTS]).round(4).to_csv(
        OUTPUTS_DIR / f"uplift_summary_{source}.csv", index=False)

    if cat_rows:
        pd.DataFrame(cat_rows).round(4).to_csv(
            OUTPUTS_DIR / f"uplift_by_category_{source}.csv", index=False)
    if not prof.empty:
        prof.round(4).to_csv(OUTPUTS_DIR / f"promo_profitability_{source}.csv", index=False)

    print(f"\n[7] Written: uplift_summary_{source}.csv, "
          f"uplift_by_category_{source}.csv, promo_profitability_{source}.csv")
    print("=" * 72)
    return overall


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else None
    for s in ([which] if which else ["synthetic", "real"]):
        run(s)
        print()