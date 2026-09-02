"""
Phase 3 - Price elasticity engine.

THE PROBLEM
-----------
Running a naive log-log regression of units on price gives a POSITIVE
elasticity for roughly a fifth of products in this dataset. That is
economically impossible: it says raising the price sells more.

The cause is price endogeneity. Retailers do not set prices at random; they
cut prices in weeks when they already expect strong demand (holidays, paydays,
seasonal peaks) and they promote heavily in high-traffic stores. So price is
correlated with the error term, and OLS is biased.

THE FIX - four estimators, each strictly better than the last
------------------------------------------------------------
1. NAIVE OLS          log(units) ~ log(price)
                      The baseline. Shows the size of the problem.

2. FIXED EFFECTS      + store fixed effects (within transformation)
                      + merchandising controls (feature/display/tpr)
                      + seasonal harmonics and a linear trend
                      Absorbs store quality and seasonality.

3. HAUSMAN 2SLS       Instruments own price with the average price of the SAME
                      product in OTHER stores in the SAME week.
                      Logic: prices in other markets move with shared COST
                      shocks (manufacturer promotions, commodity prices), but
                      are unaffected by THIS store's local demand shock. That
                      makes them a valid instrument, isolating the supply-side
                      variation in price. Standard in empirical IO
                      (Hausman 1996; Nevo 2001).

4. EMPIRICAL BAYES    Shrinks each product's estimate toward its category mean,
                      weighted by its own precision. Products with thin or
                      noisy data borrow strength from their category instead of
                      producing wild estimates.

VALIDATION
----------
On the synthetic data we know the true elasticity, so we can rank the four
estimators by how well each recovers it. That ranking is the justification for
using the final one on the real data, where truth is unobservable.

Run:
    python src/elasticity.py            # both sources
    python src/elasticity.py real
    python src/elasticity.py synthetic
"""

import sys
import warnings
import numpy as np
import pandas as pd

from config import PROCESSED_DIR, OUTPUTS_DIR
from schema import Canon
from data_loader import load_processed

warnings.filterwarnings("ignore")

MIN_OBS = 200            # minimum rows before we trust a per-UPC regression
MIN_PRICE_VAR = 0.01     # minimum sd of log price ratio
ELASTICITY_FLOOR = -8.0  # clip implausible magnitudes
ELASTICITY_CEIL = 0.5


# ==========================================================================
# Feature construction
# ==========================================================================
def build_design(df):
    """Add the regression variables we need, once, for the whole panel."""
    d = df.copy()
    d["log_units"] = np.log(d[Canon.UNITS].clip(lower=1))
    d["log_price"] = np.log(d[Canon.PRICE].clip(lower=0.01))

    # seasonal harmonics (2 cycles) + linear trend, as continuous controls.
    # We deliberately avoid full week fixed effects: they would absorb the
    # cross-store price variation our instrument relies on.
    woy = (d[Canon.WEEK_INDEX] - 1) % 52
    d["sin1"] = np.sin(2 * np.pi * woy / 52)
    d["cos1"] = np.cos(2 * np.pi * woy / 52)
    d["sin2"] = np.sin(4 * np.pi * woy / 52)
    d["cos2"] = np.cos(4 * np.pi * woy / 52)
    d["trend"] = d[Canon.WEEK_INDEX] / d[Canon.WEEK_INDEX].max()

    # ---- Hausman instrument: leave-one-out mean log price of the SAME upc
    #      in OTHER stores during the SAME week.
    grp = d.groupby([Canon.UPC, Canon.WEEK_INDEX])["log_price"]
    tot = grp.transform("sum")
    cnt = grp.transform("count")
    with np.errstate(invalid="ignore", divide="ignore"):
        d["iv_price"] = (tot - d["log_price"]) / (cnt - 1)
    d.loc[cnt <= 1, "iv_price"] = np.nan
    return d


def _demean(arr, codes, n_groups):
    """Within transformation: subtract each row's group mean (store FE)."""
    arr = np.asarray(arr, dtype=float)
    if arr.ndim == 1:
        arr = arr[:, None]
    out = arr.copy()
    for j in range(arr.shape[1]):
        sums = np.bincount(codes, weights=arr[:, j], minlength=n_groups)
        counts = np.bincount(codes, minlength=n_groups).astype(float)
        means = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
        out[:, j] = arr[:, j] - means[codes]
    return out


def _ols(X, y):
    """Return (beta, standard_errors). X must already include any intercept."""
    XtX = X.T @ X
    try:
        XtX_inv = np.linalg.inv(XtX)
    except np.linalg.LinAlgError:
        XtX_inv = np.linalg.pinv(XtX)
    beta = XtX_inv @ (X.T @ y)
    resid = y - X @ beta
    dof = max(len(y) - X.shape[1], 1)
    sigma2 = float(resid @ resid) / dof
    se = np.sqrt(np.maximum(np.diag(XtX_inv) * sigma2, 0))
    return beta, se


# ==========================================================================
# The four estimators, run for one product
# ==========================================================================
CONTROLS = [Canon.FEATURE, Canon.DISPLAY, Canon.TPR_ONLY,
            "sin1", "cos1", "sin2", "cos2", "trend"]


def estimate_one_upc(g):
    """Return naive OLS, fixed-effects and 2SLS elasticity for one product."""
    out = {"n_obs": len(g), "price_sd": float(g["log_price"].std())}

    # ---------------------------------------------------------- 1. naive OLS
    X = np.column_stack([np.ones(len(g)), g["log_price"].to_numpy()])
    beta, se = _ols(X, g["log_units"].to_numpy())
    out["elast_naive"] = float(beta[1])
    out["se_naive"] = float(se[1])

    # ------------------------------------------------- 2. store FE + controls
    codes, uniq = pd.factorize(g[Canon.STORE_ID])
    n_groups = len(uniq)
    y_w = _demean(g["log_units"].to_numpy(), codes, n_groups).ravel()
    Xraw = np.column_stack([g["log_price"].to_numpy()] +
                           [g[c].to_numpy(dtype=float) for c in CONTROLS])
    X_w = _demean(Xraw, codes, n_groups)

    keep = X_w.std(axis=0) > 1e-10          # drop columns with no within variation
    X_w = X_w[:, keep]
    if X_w.shape[1] == 0 or X_w[:, 0].std() < MIN_PRICE_VAR:
        out["elast_fe"] = np.nan
        out["se_fe"] = np.nan
    else:
        beta, se = _ols(X_w, y_w)
        out["elast_fe"] = float(beta[0])
        out["se_fe"] = float(se[0])

    # ------------------------------------------------------- 3. Hausman 2SLS
    gi = g.dropna(subset=["iv_price"])
    if len(gi) < MIN_OBS:
        out["elast_iv"] = np.nan
        out["se_iv"] = np.nan
        out["first_stage_f"] = np.nan
    else:
        codes_i, uniq_i = pd.factorize(gi[Canon.STORE_ID])
        ng_i = len(uniq_i)
        ctrl = np.column_stack([gi[c].to_numpy(dtype=float) for c in CONTROLS])
        ctrl_w = _demean(ctrl, codes_i, ng_i)
        p_w = _demean(gi["log_price"].to_numpy(), codes_i, ng_i).ravel()
        z_w = _demean(gi["iv_price"].to_numpy(), codes_i, ng_i).ravel()
        y_wi = _demean(gi["log_units"].to_numpy(), codes_i, ng_i).ravel()

        ck = ctrl_w.std(axis=0) > 1e-10
        ctrl_w = ctrl_w[:, ck]

        # --- first stage: price on instrument + controls
        Z1 = np.column_stack([z_w, ctrl_w]) if ctrl_w.size else z_w[:, None]
        b1, se1 = _ols(Z1, p_w)
        p_hat = Z1 @ b1
        # first-stage F statistic on the excluded instrument
        f_stat = (b1[0] / se1[0]) ** 2 if se1[0] > 0 else 0.0
        out["first_stage_f"] = float(f_stat)

        if f_stat < 10:                  # weak instrument - refuse to report
            out["elast_iv"] = np.nan
            out["se_iv"] = np.nan
        else:
            X2 = np.column_stack([p_hat, ctrl_w]) if ctrl_w.size else p_hat[:, None]
            b2, se2 = _ols(X2, y_wi)
            out["elast_iv"] = float(b2[0])
            out["se_iv"] = float(se2[0])

    return out


# ==========================================================================
# Empirical-Bayes shrinkage toward the category mean
# ==========================================================================
def shrink(df, est_col, se_col, group_col=Canon.CATEGORY):
    """
    beta_shrunk = w * beta_own + (1 - w) * beta_group
    w = tau^2 / (tau^2 + se^2), where tau^2 is the between-product variance
    within the group, net of sampling noise.
    """
    out = df.copy()
    out["elast_shrunk"] = np.nan
    out["shrink_weight"] = np.nan

    for grp, sub in out.groupby(group_col):
        ok = sub[est_col].notna() & sub[se_col].notna() & (sub[se_col] > 0)
        if ok.sum() < 3:
            out.loc[sub.index[ok], "elast_shrunk"] = sub.loc[ok, est_col]
            out.loc[sub.index[ok], "shrink_weight"] = 1.0
            continue

        est = sub.loc[ok, est_col].to_numpy()
        se = sub.loc[ok, se_col].to_numpy()
        # ROBUST prior: median and MAD, so one bad estimate cannot drag the
        # whole category. Using the mean here let a single clipped outlier
        # corrupt every product in its category.
        prior_mean = float(np.median(est))
        mad = float(np.median(np.abs(est - prior_mean))) * 1.4826
        tau2 = max(mad ** 2 - float(np.median(se ** 2)), 1e-6)
        w = tau2 / (tau2 + se ** 2)
        out.loc[sub.index[ok], "elast_shrunk"] = w * est + (1 - w) * prior_mean
        out.loc[sub.index[ok], "shrink_weight"] = w

    return out


# ==========================================================================
# Orchestration
# ==========================================================================
def run(source):
    print("=" * 70)
    print(f"PHASE 3 - PRICE ELASTICITY ENGINE   [{source}]")
    print("=" * 70)

    df = load_processed(source)
    train = df[df[Canon.SPLIT] == "train"].copy()
    print(f"\n[1] Training rows (weeks 1-110): {len(train):,}")

    d = build_design(train)
    print(f"    instrument available for {d['iv_price'].notna().mean():.1%} of rows")

    # ---------------------------------------------------------------- estimate
    print("\n[2] Estimating per product")
    rows = []
    for upc, g in d.groupby(Canon.UPC):
        if len(g) < MIN_OBS:
            continue
        rec = {"upc": upc,
               Canon.CATEGORY: g[Canon.CATEGORY].iloc[0],
               "description": g[Canon.DESCRIPTION].iloc[0]}
        rec.update(estimate_one_upc(g))
        rows.append(rec)
    est = pd.DataFrame(rows)
    print(f"    {len(est)} products estimated (>= {MIN_OBS} obs each)")

    # clip implausible values before shrinking
    for c in ("elast_naive", "elast_fe", "elast_iv"):
        est[c] = est[c].clip(ELASTICITY_FLOOR, ELASTICITY_CEIL)

    # ---------------------------------------------------------------- shrink
    # Prefer the 2SLS estimate, but ONLY where the instrument is genuinely
    # strong and the estimate is not pinned to the clipping bound. Otherwise
    # fall back to fixed effects. Mixing a weak-instrument estimate into the
    # panel silently poisons the category prior used for shrinkage.
    iv_ok = (est["elast_iv"].notna()
             & (est["first_stage_f"] >= 10)
             & est["se_iv"].notna() & (est["se_iv"] > 0) & (est["se_iv"] < 2)
             & est["elast_iv"].between(ELASTICITY_FLOOR + 0.01, ELASTICITY_CEIL - 0.01))
    est["iv_used"] = iv_ok
    est["elast_best_raw"] = np.where(iv_ok, est["elast_iv"], est["elast_fe"])
    est["se_best_raw"] = np.where(iv_ok, est["se_iv"], est["se_fe"])
    print(f"    2SLS accepted for {int(iv_ok.sum())}/{len(est)} products; "
          f"fixed effects used for the rest")
    est = shrink(est, "elast_best_raw", "se_best_raw")
    est["elast_final"] = est["elast_shrunk"].fillna(est["elast_best_raw"])

    # ---------------------------------------------------------------- report
    print("\n[3] Wrong-sign rate (positive elasticity = economically impossible)")
    for label, col in [("naive OLS", "elast_naive"),
                       ("store FE + controls", "elast_fe"),
                       ("Hausman 2SLS", "elast_iv"),
                       ("2SLS + EB shrinkage", "elast_final")]:
        v = est[col].dropna()
        if len(v) == 0:
            print(f"    {label:<22} n/a")
            continue
        print(f"    {label:<22} {(v > 0).mean():>6.1%} wrong sign   "
              f"median {v.median():>6.2f}   n={len(v)}")

    fs = est["first_stage_f"].dropna()
    if len(fs):
        print(f"\n[4] Instrument strength: median first-stage F = {fs.median():,.0f} "
              f"({(fs >= 10).mean():.0%} of products pass F>=10)")

    print("\n[5] Final elasticity by category")
    summ = (est.groupby(Canon.CATEGORY)["elast_final"]
            .agg(["count", "median", "min", "max"]).round(2))
    print(summ.to_string())

    # ------------------------------------------------- validate vs ground truth
    truth_path = PROCESSED_DIR / "ground_truth_products.csv"
    if source == "synthetic" and truth_path.exists():
        print("\n[6] VALIDATION against known ground truth")
        truth = pd.read_csv(truth_path)
        m = est.merge(truth, on="upc", how="inner")
        print(f"    matched {len(m)} products\n")
        print(f"    {'estimator':<24} {'corr':>7} {'MAE':>7} {'bias':>7}")
        print("    " + "-" * 47)
        best = None
        for label, col in [("naive OLS", "elast_naive"),
                           ("store FE + controls", "elast_fe"),
                           ("Hausman 2SLS", "elast_iv"),
                           ("2SLS + EB shrinkage", "elast_final")]:
            sub = m[[col, "true_elasticity"]].dropna()
            if len(sub) < 3:
                continue
            corr = sub[col].corr(sub["true_elasticity"])
            mae = (sub[col] - sub["true_elasticity"]).abs().mean()
            bias = (sub[col] - sub["true_elasticity"]).mean()
            print(f"    {label:<24} {corr:>7.3f} {mae:>7.3f} {bias:>+7.3f}")
            if best is None or mae < best[1]:
                best = (label, mae)
        if best:
            print(f"\n    --> best estimator: {best[0]} (MAE {best[1]:.3f})")

    # ---------------------------------------------------------------- write
    cols = ["upc", Canon.CATEGORY, "description", "n_obs", "price_sd",
            "elast_naive", "elast_fe", "elast_iv", "first_stage_f", "iv_used",
            "elast_shrunk", "shrink_weight", "elast_final"]
    out = OUTPUTS_DIR / f"elasticity_{source}.csv"
    est[cols].round(4).to_csv(out, index=False)
    print(f"\n[7] Written: {out.name}")
    print("=" * 70)
    return est


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else None
    for s in ([which] if which else ["synthetic", "real"]):
        run(s)
        print()
