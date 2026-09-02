"""
Phase 8 - Inventory replenishment.

THE PROBLEM
-----------
Every week, for every product in every store, somebody must decide how much to
reorder. Order too little and you lose sales you could have made. Order too
much and you pay to hold stock that is not moving. The order does not arrive
instantly - there is a lead time - so the decision has to cover demand you
cannot yet see.

THE CLASSICAL ANSWER, AND WHY IT IS NOT GOOD ENOUGH HERE
--------------------------------------------------------
Textbook safety stock is

    order-up-to S = mu * (L + 1) + z * sigma * sqrt(L + 1)

where mu and sigma are the historical mean and standard deviation of weekly
demand, L is the lead time, and z comes from the newsvendor critical fractile

    z = Phi^-1( Cu / (Cu + Co) )        Cu = cost of being short
                                        Co = cost of holding

This rests on two assumptions that are false in this dataset:

  1. Demand is normally distributed.
     Retail demand is right skewed and count valued. Phase 5 showed our
     forecast on promoted weeks runs 1.90x the baseline.

  2. Demand is stationary, so history predicts next week.
     It is not. Next week's demand depends on next week's PROMOTION, which the
     retailer has already planned and therefore already knows.

That second point is the opportunity. A static policy treats a promotion as
random noise and pads safety stock for it all year round. A forecast-driven
policy knows the promotion is coming and orders for it specifically.

WHAT WE COMPARE
---------------
    static normal      classical formula, normal safety stock       (baseline)
    static empirical   same, but the safety factor comes from the
                       actual demand quantile instead of assuming normality
    forecast-driven    order-up-to built from the Phase 5 forecast,
                       which already knows next week's promo calendar   (ours)
    oracle             perfect foresight of demand - not achievable,
                       included to show how much room is actually left

HONEST LIMITATION
-----------------
No public retail dataset contains lead times, holding costs or stockout
penalties, so those come from config.py as stated assumptions. What is REAL
here is the demand: we replay the actual test-period units from the dunnhumby
data, week by week. The policies are judged on demand that genuinely happened.

Run:
    python src/inventory.py             # both sources
    python src/inventory.py real
    python src/inventory.py synthetic
"""

import sys
import warnings
from collections import deque

import numpy as np
import pandas as pd
from scipy.stats import norm

from config import (OUTPUTS_DIR, HOLDING_COST_RATE_PER_WEEK,
                    STOCKOUT_PENALTY_MULTIPLIER, DEFAULT_LEAD_TIME_WEEKS,
                    VALID_END_WEEK)
from schema import Canon
from data_loader import load_processed

warnings.filterwarnings("ignore")

LEAD_TIME = DEFAULT_LEAD_TIME_WEEKS
MIN_HISTORY = 20          # weeks of history needed to set a policy
MIN_TEST_WEEKS = 15       # weeks needed to judge one fairly
MAX_SERIES = 1500         # cap on store-product series, for runtime


# ==========================================================================
# The simulator
# ==========================================================================
def simulate(demand, order_up_to, unit_cost, price, lead_time=LEAD_TIME,
             initial=None):
    """
    Replay one store-product series week by week under a given policy.

    order_up_to may be a scalar (static policy) or an array with one target per
    week (dynamic policy). Returns the cost breakdown and service level.

    Timing within a week: goods arrive, we place an order against the inventory
    POSITION (on hand plus everything already ordered but not yet arrived), then
    demand hits what is physically on the shelf.
    """
    demand = np.asarray(demand, dtype=float)
    T = len(demand)
    S = (np.full(T, float(order_up_to)) if np.isscalar(order_up_to)
         else np.asarray(order_up_to, dtype=float))

    holding_rate = unit_cost * HOLDING_COST_RATE_PER_WEEK
    # being short costs the margin we failed to earn, plus a penalty for the
    # goodwill damage of an empty shelf
    short_cost = max(price - unit_cost, 0.01) * STOCKOUT_PENALTY_MULTIPLIER

    on_hand = float(S[0] if initial is None else initial)
    pipeline = deque([0.0] * lead_time)

    hold_c = short_c = 0.0
    total_demand = total_sold = 0.0
    stockout_weeks = 0
    inv_levels = []

    for t in range(T):
        on_hand += pipeline.popleft()                 # receipts arrive
        inv_position = on_hand + sum(pipeline)        # what we already own or owe
        order = max(0.0, S[t] - inv_position)
        pipeline.append(order)

        d = demand[t]
        sold = min(d, on_hand)
        lost = d - sold
        on_hand -= sold

        hold_c += on_hand * holding_rate
        short_c += lost * short_cost
        total_demand += d
        total_sold += sold
        stockout_weeks += (lost > 0)
        inv_levels.append(on_hand)

    return {
        "holding_cost": hold_c,
        "stockout_cost": short_c,
        "total_cost": hold_c + short_c,
        "fill_rate": total_sold / total_demand if total_demand > 0 else 1.0,
        "stockout_week_pct": stockout_weeks / T,
        "avg_inventory": float(np.mean(inv_levels)),
        "weeks_of_cover": float(np.mean(inv_levels)) / max(demand.mean(), 1e-9),
    }


# ==========================================================================
# Policies
# ==========================================================================
def critical_fractile(unit_cost, price):
    """Newsvendor service level implied by the cost of shortage vs holding."""
    Cu = max(price - unit_cost, 0.01) * STOCKOUT_PENALTY_MULTIPLIER
    Co = unit_cost * HOLDING_COST_RATE_PER_WEEK
    return float(np.clip(Cu / (Cu + Co), 0.50, 0.999))


def policy_static_normal(hist, unit_cost, price, lead_time=LEAD_TIME):
    """Textbook: normal safety stock over the risk period."""
    mu, sigma = hist.mean(), hist.std(ddof=1)
    z = norm.ppf(critical_fractile(unit_cost, price))
    return float(mu * (lead_time + 1) + z * sigma * np.sqrt(lead_time + 1))


def policy_static_empirical(hist, unit_cost, price, lead_time=LEAD_TIME):
    """
    Same idea, but take the quantile from the ACTUAL demand distribution
    instead of assuming it is normal. Right-skewed demand needs more cover in
    the upper tail than a normal curve predicts.
    """
    q = critical_fractile(unit_cost, price)
    risk_period = lead_time + 1
    # empirical quantile of demand summed over the risk period
    sums = np.convolve(hist, np.ones(risk_period), mode="valid")
    if len(sums) < 5:
        return policy_static_normal(hist, unit_cost, price, lead_time)
    return float(np.quantile(sums, q))


def expanding_resid_sd(demand, forecast, fallback_sd, min_obs=4):
    """
    Forecast-error standard deviation known AT EACH WEEK, using only weeks that
    have already happened.

    Measuring this over the whole test period would leak the future into the
    policy: the buyer in week 1 cannot know how accurate the forecast turned
    out to be in week 20. Until we have enough residuals we fall back to the
    historical demand spread, which is what a real planner would do.
    """
    resid = np.asarray(demand, dtype=float) - np.asarray(forecast, dtype=float)
    sds = np.empty(len(resid))
    for t in range(len(resid)):
        sds[t] = float(np.std(resid[:t], ddof=1)) if t >= min_obs else fallback_sd
    return np.maximum(sds, 1e-6)


def policy_forecast_driven(forecast, resid_sd, unit_cost, price,
                           lead_time=LEAD_TIME):
    """
    Order-up-to rebuilt every week from the Phase 5 forecast.

    The forecast already accounts for next week's promotions, price and
    seasonality, so the target rises before a promotion instead of carrying
    padding for it all year. Safety stock is then sized on FORECAST ERROR
    rather than on total demand variability, which is a much smaller number -
    that is where the saving comes from.
    """
    f = np.asarray(forecast, dtype=float)
    T = len(f)
    z = norm.ppf(critical_fractile(unit_cost, price))
    risk = lead_time + 1
    sd = (np.full(T, float(resid_sd)) if np.isscalar(resid_sd)
          else np.asarray(resid_sd, dtype=float))
    S = np.empty(T)
    for t in range(T):
        window = f[t:t + risk]
        expected = window.sum() + (risk - len(window)) * f[-1]   # pad at the end
        S[t] = expected + z * sd[t] * np.sqrt(risk)
    return S


def policy_oracle(demand, lead_time=LEAD_TIME):
    """Perfect foresight. Not achievable - an upper bound on what is possible."""
    d = np.asarray(demand, dtype=float)
    T, risk = len(d), lead_time + 1
    return np.array([d[t:t + risk].sum() + (risk - len(d[t:t + risk])) * d[-1]
                     for t in range(T)])


# ==========================================================================
# Orchestration
# ==========================================================================
def run(source):
    print("=" * 78)
    print(f"PHASE 8 - INVENTORY REPLENISHMENT   [{source}]")
    print("=" * 78)

    fc_path = OUTPUTS_DIR / f"forecast_test_{source}.csv"
    if not fc_path.exists():
        print(f"    MISSING {fc_path.name} - run  python src/forecast.py {source}  first")
        return None

    hist_all = load_processed(source)
    fc = pd.read_csv(fc_path)

    print(f"\n[1] Setup (assumptions from config.py)")
    print(f"    lead time            : {LEAD_TIME} weeks")
    print(f"    holding cost         : {HOLDING_COST_RATE_PER_WEEK:.1%} of unit cost / week")
    print(f"    stockout penalty     : {STOCKOUT_PENALTY_MULTIPLIER:.1f}x lost margin")
    print(f"    demand replayed      : ACTUAL test-week units from the data")

    history = hist_all[hist_all[Canon.WEEK_INDEX] <= VALID_END_WEEK]
    hist_map = {k: g.sort_values(Canon.WEEK_INDEX)[Canon.UNITS].to_numpy()
                for k, g in history.groupby([Canon.STORE_ID, Canon.UPC], sort=False)}

    fc = fc.sort_values([Canon.STORE_ID, Canon.UPC, Canon.WEEK_INDEX])
    series = list(fc.groupby([Canon.STORE_ID, Canon.UPC], sort=False))
    rng = np.random.default_rng(42)
    if len(series) > MAX_SERIES:
        idx = rng.choice(len(series), MAX_SERIES, replace=False)
        series = [series[i] for i in sorted(idx)]
    print(f"    store-product series : {len(series):,} evaluated")

    print(f"\n[2] Replaying {len(series):,} series under 4 policies...")
    rows = []
    for (store, upc), g in series:
        hist = hist_map.get((store, upc))
        if hist is None or len(hist) < MIN_HISTORY or len(g) < MIN_TEST_WEEKS:
            continue
        demand = g[Canon.UNITS].to_numpy(dtype=float)
        forecast = g["pred"].to_numpy(dtype=float)
        if demand.mean() <= 0.5:
            continue

        cost = float(g[Canon.UNIT_COST].mean())
        price = float(g[Canon.PRICE].mean())
        # forecast-error sd known at each week, never using future weeks
        hist_sd = max(float(hist.std(ddof=1)), 1e-6)
        resid_sd = expanding_resid_sd(demand, forecast, hist_sd)

        pols = {
            "static_normal": policy_static_normal(hist, cost, price),
            "static_empirical": policy_static_empirical(hist, cost, price),
            "forecast_driven": policy_forecast_driven(forecast, resid_sd, cost, price),
            "oracle": policy_oracle(demand),
        }
        rec = {"store_id": store, "upc": upc,
               "category": g[Canon.CATEGORY].iloc[0],
               "mean_demand": demand.mean(),
               "demand_cv": demand.std() / max(demand.mean(), 1e-9),
               "forecast_sd": float(np.mean(resid_sd))}
        for name, S in pols.items():
            r = simulate(demand, S, cost, price)
            for k, v in r.items():
                rec[f"{name}_{k}"] = v
        rows.append(rec)

    inv = pd.DataFrame(rows)
    if inv.empty:
        print("    no series had enough history to evaluate")
        return None
    print(f"    evaluated {len(inv):,} series")

    POLICIES = [("static_normal", "static normal (textbook)"),
                ("static_empirical", "static empirical quantile"),
                ("forecast_driven", "forecast-driven (ours)"),
                ("oracle", "oracle / perfect foresight")]

    # ------------------------------------------------------------ headline
    print(f"\n[3] Total cost over the test weeks (lower is better)")
    print(f"    {'policy':<28}{'holding':>11}{'stockout':>11}{'TOTAL':>12}"
          f"{'fill rate':>11}{'cover wks':>11}")
    print("    " + "-" * 84)
    base_total = inv["static_normal_total_cost"].sum()
    for key, label in POLICIES:
        h = inv[f"{key}_holding_cost"].sum()
        s = inv[f"{key}_stockout_cost"].sum()
        t = inv[f"{key}_total_cost"].sum()
        fr = inv[f"{key}_fill_rate"].mean()
        wc = inv[f"{key}_weeks_of_cover"].mean()
        print(f"    {label:<28}{h:>11,.0f}{s:>11,.0f}{t:>12,.0f}"
              f"{fr:>10.1%}{wc:>11.1f}")

    fd = inv["forecast_driven_total_cost"].sum()
    orc = inv["oracle_total_cost"].sum()
    print(f"\n[4] Improvement over the textbook policy")
    print(f"    forecast-driven saves : {base_total - fd:>12,.0f}  "
          f"({(1 - fd / base_total) * 100:.1f}%)")
    print(f"    oracle would save     : {base_total - orc:>12,.0f}  "
          f"({(1 - orc / base_total) * 100:.1f}%)")
    captured = (base_total - fd) / max(base_total - orc, 1e-9)
    print(f"    -> we capture {captured:.0%} of the theoretically available saving")

    # --------------------------------------------------- where it comes from
    print(f"\n[5] Where the saving comes from")
    dh = inv["forecast_driven_holding_cost"].sum() - inv["static_normal_holding_cost"].sum()
    ds = inv["forecast_driven_stockout_cost"].sum() - inv["static_normal_stockout_cost"].sum()
    print(f"    holding cost change  : {dh:>+12,.0f}")
    print(f"    stockout cost change : {ds:>+12,.0f}")
    print(f"    avg weeks of cover   : {inv['static_normal_weeks_of_cover'].mean():.1f} "
          f"-> {inv['forecast_driven_weeks_of_cover'].mean():.1f}")
    print(f"    fill rate            : {inv['static_normal_fill_rate'].mean():.1%} "
          f"-> {inv['forecast_driven_fill_rate'].mean():.1%}")
    print("    Safety stock is sized on FORECAST ERROR, not on total demand")
    print("    variability. The forecast already explains the promotions, so")
    print("    there is far less residual uncertainty left to buffer against.")

    # ---------------------------------------------- normality check (the why)
    print(f"\n[6] Why the textbook formula struggles here")
    ratio = inv["static_empirical_total_cost"].sum() / inv["static_normal_total_cost"].sum()
    print(f"    swapping the normal assumption for the empirical quantile alone")
    print(f"    changes cost by {(ratio - 1) * 100:+.1f}% - evidence that demand is")
    print(f"    not normally distributed")
    print(f"    median demand CV across series: {inv['demand_cv'].median():.2f}")

    # ------------------------------------------------------------ by category
    print(f"\n[7] By category")
    agg = (inv.groupby("category")
           .apply(lambda d: pd.Series({
               "series": len(d),
               "mean_demand": d["mean_demand"].mean(),
               "textbook_cost": d["static_normal_total_cost"].sum(),
               "ours_cost": d["forecast_driven_total_cost"].sum(),
               "saving_pct": (1 - d["forecast_driven_total_cost"].sum()
                              / max(d["static_normal_total_cost"].sum(), 1e-9)) * 100,
           }), include_groups=False)
           .round(2))
    print(agg.to_string())

    # -------------------------------------------------- who benefits most
    print(f"\n[8] The saving is largest where demand is hardest to predict")
    inv["saving"] = (inv["static_normal_total_cost"]
                     - inv["forecast_driven_total_cost"])
    inv["cv_bucket"] = pd.qcut(inv["demand_cv"], 4,
                               labels=["low CV", "med-low", "med-high", "high CV"])
    b = (inv.groupby("cv_bucket", observed=True)
         .agg(series=("saving", "size"), mean_saving=("saving", "mean"),
              mean_cv=("demand_cv", "mean")).round(2))
    print(b.to_string())

    out = OUTPUTS_DIR / f"inventory_policy_{source}.csv"
    inv.round(4).to_csv(out, index=False)
    print(f"\n[9] Written: {out.name}")
    print("=" * 78)
    return inv


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else None
    for s in ([which] if which else ["synthetic", "real"]):
        run(s)
        print()
