"""
Phase 1 verification.

Confirms the environment is correct, the generated data conforms to the schema
contract, and - most importantly - that the signal we will later try to
estimate is actually present and recoverable.

The last check is the interesting one. It demonstrates that a NAIVE comparison
of promoted vs non-promoted weeks gives a badly biased answer, because
promotions are confounded with seasonality and store tier. That bias is the
problem the causal module (a later phase) exists to solve, and because we know
the true value we will be able to prove the causal method fixed it.

Run:
    python src/verify_setup.py
"""

import sys
import numpy as np
import pandas as pd

from config import FRAT_DIR, PROCESSED_DIR, N_WEEKS
from schema import FratRaw

PASS = "[PASS]"
FAIL = "[FAIL]"
WARN = "[WARN]"

failures = []


def check(condition, label, detail=""):
    tag = PASS if condition else FAIL
    print(f"  {tag} {label}" + (f"  ({detail})" if detail else ""))
    if not condition:
        failures.append(label)
    return condition


def main():
    print("=" * 68)
    print("PHASE 1 VERIFICATION")
    print("=" * 68)

    # ---------------------------------------------------------------- env
    print("\n[1] Environment")
    print(f"  python {sys.version.split()[0]}")
    for mod in ("pandas", "numpy", "sklearn", "scipy", "statsmodels",
                "matplotlib", "shap", "streamlit"):
        try:
            m = __import__(mod)
            print(f"  {PASS} {mod:<12} {getattr(m, '__version__', '?')}")
        except ImportError:
            print(f"  {FAIL} {mod:<12} NOT INSTALLED")
            failures.append(f"import {mod}")

    # ---------------------------------------------------------------- files
    print("\n[2] Raw files present")
    paths = {
        "frat_sales.csv": FRAT_DIR / "frat_sales.csv",
        "frat_stores.csv": FRAT_DIR / "frat_stores.csv",
        "frat_products.csv": FRAT_DIR / "frat_products.csv",
    }
    missing_files = False
    for name, p in paths.items():
        ok = check(p.exists(), name,
                   f"{p.stat().st_size / 1e6:.1f} MB" if p.exists() else "missing")
        missing_files = missing_files or (not ok)
    if missing_files:
        print("\nCannot continue - run  python src/make_synthetic_frat.py  first.")
        return 1

    sales = pd.read_csv(paths["frat_sales.csv"], parse_dates=[FratRaw.WEEK_END_DATE])
    stores = pd.read_csv(paths["frat_stores.csv"])
    products = pd.read_csv(paths["frat_products.csv"])

    # ---------------------------------------------------------------- schema
    print("\n[3] Schema contract")
    check(list(sales.columns) == FratRaw.SALES_COLUMNS, "sales columns match contract")
    check(list(stores.columns) == FratRaw.STORE_COLUMNS, "store columns match contract")
    check(list(products.columns) == FratRaw.PRODUCT_COLUMNS, "product columns match contract")

    # ---------------------------------------------------------------- integrity
    print("\n[4] Data integrity")
    check(sales[FratRaw.WEEK_END_DATE].nunique() == N_WEEKS,
          f"{N_WEEKS} distinct weeks", str(sales[FratRaw.WEEK_END_DATE].nunique()))
    check(sales[FratRaw.UNITS].min() > 0, "all units positive")
    check(sales[FratRaw.PRICE].min() > 0, "all prices positive")
    check((sales[FratRaw.PRICE] <= sales[FratRaw.BASE_PRICE] + 1e-9).all(),
          "price never exceeds base_price")
    check(int(sales[FratRaw.SALES_COLUMNS].isna().sum().sum()) == 0, "no nulls in sales")
    for flag in (FratRaw.FEATURE, FratRaw.DISPLAY, FratRaw.TPR_ONLY):
        check(set(sales[flag].unique()) <= {0, 1}, f"{flag} is binary")
    check(set(sales[FratRaw.STORE_NUM].unique()) <= set(stores[FratRaw.STORE_ID]),
          "every store_num exists in store table")
    check(set(sales[FratRaw.UPC].unique()) <= set(products[FratRaw.UPC]),
          "every UPC exists in product table")

    # ---------------------------------------------------------------- variation
    print("\n[5] Signal present (required for the advanced features)")
    disc = 1 - sales[FratRaw.PRICE] / sales[FratRaw.BASE_PRICE]
    price_var = sales.groupby(FratRaw.UPC)[FratRaw.PRICE].std().median()
    promo_share = sales[[FratRaw.FEATURE, FratRaw.DISPLAY, FratRaw.TPR_ONLY]].max(axis=1).mean()

    check(price_var > 0.05, "genuine price variation exists (elasticity is estimable)",
          f"median within-UPC price sd = {price_var:.3f}")
    check(0.05 < promo_share < 0.60, "promotions present but not universal",
          f"{promo_share:.1%} of rows promoted")
    check(disc.max() > 0.20, "meaningful discount depth observed",
          f"max {disc.max():.1%}")

    # untreated control rows must exist in the SAME week -> needed for DiD
    per_week = sales.groupby(FratRaw.WEEK_END_DATE)[FratRaw.DISPLAY].agg(["mean", "count"])
    both_groups = ((per_week["mean"] > 0.01) & (per_week["mean"] < 0.99)).all()
    check(both_groups, "treated AND untreated stores exist in every week (DiD is possible)")

    # ---------------------------------------------------------------- recoverability
    truth_p = PROCESSED_DIR / "ground_truth_products.csv"
    truth_l = PROCESSED_DIR / "ground_truth_lift.csv"
    if truth_p.exists() and truth_l.exists():
        print("\n[6] Ground-truth recoverability (synthetic data only)")
        truth = pd.read_csv(truth_p)
        lift = pd.read_csv(truth_l).iloc[0]

        s = sales.copy()
        s["logq"] = np.log(s[FratRaw.UNITS])
        s["logp"] = np.log(s[FratRaw.PRICE] / s[FratRaw.BASE_PRICE])

        rows = []
        for upc, g in s.groupby(FratRaw.UPC):
            X = np.column_stack([np.ones(len(g)), g["logp"],
                                 g[FratRaw.FEATURE], g[FratRaw.DISPLAY], g[FratRaw.TPR_ONLY]])
            beta = np.linalg.lstsq(X, g["logq"].to_numpy(), rcond=None)[0]
            rows.append({"upc": upc, "est_elasticity": beta[1]})
        est = pd.DataFrame(rows).merge(truth, on="upc")
        corr = est["est_elasticity"].corr(est["true_elasticity"])
        mae = (est["est_elasticity"] - est["true_elasticity"]).abs().mean()

        check(corr > 0.90, "elasticity is recoverable from the data",
              f"corr(est, true) = {corr:.3f}, MAE = {mae:.3f}")

        # the deliberately-planted confounding problem
        naive = (np.log(s.loc[s[FratRaw.DISPLAY] == 1, FratRaw.UNITS].mean())
                 - np.log(s.loc[s[FratRaw.DISPLAY] == 0, FratRaw.UNITS].mean()))
        true_display = float(lift["display"])
        bias_pct = (naive - true_display) / true_display
        print(f"  {WARN} naive display lift = {naive:.3f} vs TRUE {true_display:.3f} "
              f"({bias_pct:+.0%} biased)")
        print("       ^ this bias is intentional. It is the confounding that the")
        print("         causal/uplift phase will correct, and we can prove it.")
    else:
        print("\n[6] Ground truth files not found - skipping (fine for real data)")

    # ---------------------------------------------------------------- verdict
    print("\n" + "=" * 68)
    if failures:
        print(f"RESULT: {len(failures)} CHECK(S) FAILED")
        for f in failures:
            print("  -", f)
        return 1
    print("RESULT: ALL CHECKS PASSED - Phase 1 complete.")
    print(f"  {len(sales):,} sales rows | {len(stores)} stores | "
          f"{len(products)} products | {N_WEEKS} weeks")
    print("=" * 68)
    return 0


if __name__ == "__main__":
    sys.exit(main())