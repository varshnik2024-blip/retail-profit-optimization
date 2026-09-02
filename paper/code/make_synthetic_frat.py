"""
Schema-conformant stand-in for dunnhumby "Breakfast at the Frat".

WHY THIS EXISTS
---------------
The real dunnhumby download may take days to arrive. Rather than idle, we
generate data that obeys EXACTLY the same schema (same column names, same
value ranges), so the entire pipeline can be built and tested today and then
re-pointed at the real files by flipping one flag in config.py.

It is also better than the real data in one specific way: we know the TRUE
elasticity and the TRUE promotional lift for every product, because we chose
them. That gives us a ground-truth benchmark - we can prove our elasticity
estimator and our difference-in-differences uplift estimator recover the real
parameters, which is something you can never demonstrate on real data.

DEMAND MODEL (log-linear, the standard retail specification)
    log E[units] = base_upc + store_effect + seasonality(week)
                   + elasticity * log(price / base_price)
                   + b_feature*FEATURE + b_display*DISPLAY + b_tpr*TPR_ONLY
    units ~ Poisson(E[units])

Promotion assignment is deliberately CONFOUNDED (retailers promote more in
high season and more in value-tier stores), so the causal module has a real
confounding problem to solve rather than a trivial one.

Run:
    python src/make_synthetic_frat.py
"""

import numpy as np
import pandas as pd

from config import FRAT_DIR, PROCESSED_DIR, RANDOM_SEED, N_WEEKS
from schema import FratRaw

RNG = np.random.default_rng(RANDOM_SEED)

N_STORES = 76
START_DATE = pd.Timestamp("2009-01-14")   # matches the real dataset's era

# The four real categories, with their real sub-categories
CATALOG = {
    "COLD CEREAL": ["ALL FAMILY CEREAL", "KIDS CEREAL", "ADULT CEREAL"],
    "FROZEN PIZZA": ["PIZZA/PREMIUM", "PIZZA/ECONOMY"],
    "BAG SNACKS": ["PRETZELS"],
    "ORAL HYGIENE PRODUCTS": ["MOUTHWASHES/ANTISEPTIC", "MOUTHWASH/RINSES AND SPRAYS"],
}
MANUFACTURERS = {
    "COLD CEREAL": ["GENERAL MI", "KELLOGG", "POST FOODS"],
    "FROZEN PIZZA": ["TOMBSTONE", "DIGIORNO", "PRIVATE LABEL"],
    "BAG SNACKS": ["SNYDERS", "ROLD GOLD", "PRIVATE LABEL"],
    "ORAL HYGIENE PRODUCTS": ["LISTERINE", "SCOPE", "PRIVATE LABEL"],
}
PRICE_RANGE = {                     # realistic shelf prices per category
    "COLD CEREAL": (2.50, 4.80),
    "FROZEN PIZZA": (3.20, 7.50),
    "BAG SNACKS": (1.80, 3.60),
    "ORAL HYGIENE PRODUCTS": (2.90, 6.40),
}
# True own-price elasticity band per category (more negative = more elastic).
# Snacks and cereal are promotion-driven and elastic; oral hygiene is habitual.
ELASTICITY_RANGE = {
    "COLD CEREAL": (-3.2, -1.8),
    "FROZEN PIZZA": (-2.8, -1.5),
    "BAG SNACKS": (-3.5, -2.0),
    "ORAL HYGIENE PRODUCTS": (-1.8, -0.7),
}

# TRUE causal lift of each merchandising treatment, as a log-multiplier.
# exp(0.55) ~= 1.73, i.e. an in-store display genuinely lifts units ~73%.
TRUE_LIFT = {"feature": 0.38, "display": 0.55, "tpr_only": 0.22}


def build_products() -> pd.DataFrame:
    rows = []
    upc_counter = 1111111111
    for category, subcats in CATALOG.items():
        for manufacturer in MANUFACTURERS[category]:
            for i in range(5):                      # 5 products per brand
                subcat = subcats[i % len(subcats)]
                lo, hi = PRICE_RANGE[category]
                el_lo, el_hi = ELASTICITY_RANGE[category]
                rows.append({
                    FratRaw.UPC: upc_counter,
                    FratRaw.DESCRIPTION: f"{manufacturer} {subcat} {i + 1}",
                    FratRaw.MANUFACTURER: manufacturer,
                    FratRaw.CATEGORY: category,
                    FratRaw.SUB_CATEGORY: subcat,
                    FratRaw.PRODUCT_SIZE: f"{RNG.choice([10.5, 12.25, 15.0, 18.0, 22.5])} OZ",
                    # ---- ground truth, written to a separate file, not the raw sheet
                    "_true_base_price": round(float(RNG.uniform(lo, hi)), 2),
                    "_true_elasticity": round(float(RNG.uniform(el_lo, el_hi)), 3),
                    "_true_base_demand": float(RNG.uniform(6, 45)),
                })
                upc_counter += int(RNG.integers(97, 9999))
    return pd.DataFrame(rows)


def build_stores() -> pd.DataFrame:
    tiers = RNG.choice(["UPSCALE", "MAINSTREAM", "VALUE"], N_STORES, p=[0.25, 0.45, 0.30])
    states = RNG.choice(["TX", "OH", "KY", "IN"], N_STORES, p=[0.4, 0.3, 0.2, 0.1])
    cities = {"TX": "HOUSTON", "OH": "CINCINNATI", "KY": "LOUISVILLE", "IN": "INDIANAPOLIS"}
    store_ids = np.arange(1, N_STORES + 1) * 7 + 300     # non-contiguous, like real data
    return pd.DataFrame({
        FratRaw.STORE_ID: store_ids,
        FratRaw.STORE_NAME: [f"STORE {s}" for s in store_ids],
        FratRaw.CITY: [cities[s] for s in states],
        FratRaw.STATE: states,
        FratRaw.MSA_CODE: RNG.integers(10000, 49999, N_STORES),
        FratRaw.SEG_VALUE_NAME: tiers,
        FratRaw.PARKING: RNG.choice([np.nan, 60, 90, 120, 175], N_STORES),
        FratRaw.SALES_AREA: RNG.integers(18000, 65000, N_STORES),
        FratRaw.AVG_WEEKLY_BASKETS: RNG.integers(8000, 42000, N_STORES),
        # ---- ground truth
        "_store_effect": RNG.normal(0, 0.25, N_STORES).round(4),
    })


def build_sales(products: pd.DataFrame, stores: pd.DataFrame) -> pd.DataFrame:
    """Fully vectorised over the (week x store x upc) grid."""
    weeks = np.arange(1, N_WEEKS + 1)
    week_dates = [START_DATE + pd.Timedelta(weeks=int(w - 1)) for w in weeks]

    n_w, n_s, n_p = len(weeks), len(stores), len(products)
    print(f"  grid: {n_w} weeks x {n_s} stores x {n_p} products = {n_w * n_s * n_p:,} rows")

    wi, si, pi = np.meshgrid(np.arange(n_w), np.arange(n_s), np.arange(n_p), indexing="ij")
    wi, si, pi = wi.ravel(), si.ravel(), pi.ravel()

    base_demand = products["_true_base_demand"].to_numpy()[pi]
    elasticity = products["_true_elasticity"].to_numpy()[pi]
    base_price = products["_true_base_price"].to_numpy()[pi]
    store_effect = stores["_store_effect"].to_numpy()[si]
    tier = stores[FratRaw.SEG_VALUE_NAME].to_numpy()[si]

    # ---- seasonality: annual cycle + a December lift
    week_of_year = (weeks - 1) % 52
    season_curve = 0.18 * np.sin(2 * np.pi * week_of_year / 52 + 1.1)
    december = ((week_of_year >= 46) & (week_of_year <= 51)).astype(float) * 0.30
    seasonality = (season_curve + december)[wi]

    # ---- CONFOUNDED promotion assignment.
    # Retailers promote more in high season and more aggressively in VALUE stores.
    promo_propensity = 0.10 + 0.35 * np.clip(seasonality / 0.4, 0, 1)
    promo_propensity = promo_propensity + np.where(tier == "VALUE", 0.10, 0.0)
    promo_propensity = promo_propensity - np.where(tier == "UPSCALE", 0.04, 0.0)
    promo_propensity = np.clip(promo_propensity, 0.02, 0.60)

    feature = (RNG.random(len(pi)) < promo_propensity * 0.45).astype(int)
    display = (RNG.random(len(pi)) < promo_propensity * 0.40).astype(int)
    # TPR_ONLY is by definition mutually exclusive with feature/display
    tpr_candidate = (RNG.random(len(pi)) < promo_propensity * 0.50).astype(int)
    tpr_only = np.where((feature == 0) & (display == 0), tpr_candidate, 0)

    # ---- price: discount depth depends on which mechanic is running
    any_mech = (feature == 1) | (display == 1) | (tpr_only == 1)
    depths = RNG.choice([0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35], size=len(pi),
                        p=[0.20, 0.24, 0.20, 0.15, 0.11, 0.06, 0.04])
    discount = np.where(any_mech, depths, 0.0)
    # a little everyday price movement even without a promotion (base drift)
    drift = RNG.normal(0, 0.012, len(pi))
    bp = np.maximum(np.round(base_price * (1 + drift), 2), 0.49)
    price = np.maximum(np.round(bp * (1 - discount), 2), 0.49)

    # ---- demand
    log_mu = (
        np.log(base_demand)
        + store_effect
        + seasonality
        + elasticity * np.log(price / bp)
        + TRUE_LIFT["feature"] * feature
        + TRUE_LIFT["display"] * display
        + TRUE_LIFT["tpr_only"] * tpr_only
    )
    mu = np.exp(np.clip(log_mu, -4, 8))
    units = RNG.poisson(mu)

    keep = units > 0                    # dunnhumby ships only rows with sales
    print(f"  kept {keep.sum():,} rows with units > 0 ({keep.mean():.1%})")

    n_keep = int(keep.sum())
    hhs = np.maximum(1, (units[keep] * RNG.uniform(0.55, 0.85, n_keep)).astype(int))
    visits = np.maximum(hhs, (units[keep] * RNG.uniform(0.75, 0.95, n_keep)).astype(int))

    sales = pd.DataFrame({
        FratRaw.WEEK_END_DATE: pd.to_datetime([week_dates[i] for i in wi[keep]]),
        FratRaw.STORE_NUM: stores[FratRaw.STORE_ID].to_numpy()[si[keep]],
        FratRaw.UPC: products[FratRaw.UPC].to_numpy()[pi[keep]],
        FratRaw.UNITS: units[keep],
        FratRaw.VISITS: visits,
        FratRaw.HHS: hhs,
        FratRaw.SPEND: np.round(units[keep] * price[keep], 2),
        FratRaw.PRICE: price[keep],
        FratRaw.BASE_PRICE: bp[keep],
        FratRaw.FEATURE: feature[keep],
        FratRaw.DISPLAY: display[keep],
        FratRaw.TPR_ONLY: tpr_only[keep],
    })
    return (sales
            .sort_values([FratRaw.WEEK_END_DATE, FratRaw.STORE_NUM, FratRaw.UPC])
            .reset_index(drop=True))


def main():
    print("Building synthetic 'Breakfast at the Frat'...")
    products = build_products()
    stores = build_stores()
    print(f"  {len(products)} products, {len(stores)} stores")
    sales = build_sales(products, stores)

    # ---- write raw files with the REAL column names, into data/raw/frat/
    sales.to_csv(FRAT_DIR / "frat_sales.csv", index=False)
    stores[FratRaw.STORE_COLUMNS].to_csv(FRAT_DIR / "frat_stores.csv", index=False)
    products[FratRaw.PRODUCT_COLUMNS].to_csv(FRAT_DIR / "frat_products.csv", index=False)

    # ---- write ground truth SEPARATELY. No analysis module may read this
    #      except the validation report, where we check estimates against it.
    truth = products[[FratRaw.UPC, "_true_base_price", "_true_elasticity",
                      "_true_base_demand"]].copy()
    truth.columns = ["upc", "true_base_price", "true_elasticity", "true_base_demand"]
    truth.to_csv(PROCESSED_DIR / "ground_truth_products.csv", index=False)
    pd.DataFrame([TRUE_LIFT]).to_csv(PROCESSED_DIR / "ground_truth_lift.csv", index=False)

    promo_share = sales[[FratRaw.FEATURE, FratRaw.DISPLAY, FratRaw.TPR_ONLY]].max(axis=1).mean()
    max_disc = (1 - sales[FratRaw.PRICE] / sales[FratRaw.BASE_PRICE]).max()

    print("\nWritten to", FRAT_DIR)
    print(f"  frat_sales.csv    {len(sales):,} rows")
    print(f"  frat_stores.csv   {len(stores):,} rows")
    print(f"  frat_products.csv {len(products):,} rows")
    print("Ground truth ->", PROCESSED_DIR / "ground_truth_products.csv")
    print("\nSanity checks")
    print("  weeks           :", sales[FratRaw.WEEK_END_DATE].nunique())
    print(f"  promo row share : {promo_share:.1%}")
    print("  mean units      :", round(float(sales[FratRaw.UNITS].mean()), 2))
    print("  price range     :", sales[FratRaw.PRICE].min(), "-", sales[FratRaw.PRICE].max())
    print(f"  max discount    : {max_disc:.1%}")


if __name__ == "__main__":
    main()
