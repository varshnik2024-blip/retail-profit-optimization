"""
Phase 7 - Markdown / clearance optimisation by dynamic programming.

HOW THIS DIFFERS FROM PHASE 6
-----------------------------
Phase 6 answered "what is the best steady-state price?" - one price, held
forever, unlimited stock. Clearance is a completely different problem:

    - there is a FIXED quantity of stock and no more coming
    - there is a DEADLINE (end of season, expiry, range change)
    - whatever is left at the deadline is dumped at salvage value
    - the decision is a SEQUENCE: how deep to cut, and in which week

Cutting early sells more units at a thinner margin. Cutting late holds margin
but risks dumping stock at salvage. That trade-off changes every week as the
deadline approaches and as stock runs down, so a single price cannot be right.

THE METHOD - backward induction
-------------------------------
We solve it exactly, as a finite-horizon Markov decision process.

    state   (t, I)   week t, with I units left
    action  d        a discount depth from a discrete menu
    demand  D ~ Poisson(mu(d))     mu from Phase 5 baseline x Phase 3 elasticity
    sales   min(D, I)              cannot sell what we do not have

    V(T, I) = I * salvage_value                        (deadline: dump it)
    V(t, I) = max_d  E[ sales*(price_d - cost) + V(t+1, I - sales) ]

We compute V backwards from the deadline, so at every state we already know
the exact value of every future consequence. The result is the provably
optimal policy for this model - not a heuristic.

WHY DP AND NOT REINFORCEMENT LEARNING
-------------------------------------
RL is for when you cannot write down the transition probabilities. Here we
can: demand is Poisson with a mean we estimated. When the model is known and
the state space is small, backward induction gives the exact optimum in
milliseconds, and RL would only approximate it. Using RL here would be
choosing a weaker tool to look more impressive.

WHAT WE COMPARE AGAINST
-----------------------
    never mark down     hold full price, salvage the remainder
    fixed schedule      the standard retail rule: -20% at week 3, -40% at week 6
    panic markdown      one deep cut immediately
    DP optimal          ours

HONEST LIMITATION
-----------------
The dunnhumby data has no inventory levels, lead times or salvage values - no
public retail dataset does. We therefore construct clearance scenarios from the
demand the data DOES give us (starting stock = a chosen number of weeks of
baseline demand) and state the salvage assumption explicitly. The demand model
is estimated from real data; the inventory scenario is a stated assumption.

Run:
    python src/markdown.py             # both sources
    python src/markdown.py real
    python src/markdown.py synthetic
"""

import sys
import warnings
import numpy as np
import pandas as pd
from scipy.stats import poisson

from config import OUTPUTS_DIR
from schema import Canon

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------
# Clearance assumptions. Stated here rather than hidden in the code.
# (Move these to config.py if you prefer all assumptions in one file.)
# --------------------------------------------------------------------------
HORIZON_WEEKS = 8                 # weeks until the deadline
DISCOUNTS = np.array([0.0, 0.10, 0.20, 0.30, 0.40, 0.50])
SALVAGE_FRACTION_OF_COST = 0.20   # leftover stock recovers 20% of unit cost
# Starting stock, expressed in weeks of baseline demand. This MUST exceed the
# horizon, otherwise the stock clears at full price on its own and there is no
# clearance decision to make - the DP would correctly answer "never discount"
# and the whole exercise would be vacuous. 14 weeks of stock with an 8 week
# deadline means roughly 40% of the buy cannot sell at full price: a realistic
# over-buy, and the situation markdown planning actually exists to handle.
WEEKS_OF_STOCK = 14.0
HOLDING_COST_PER_UNIT_WEEK = 0.01  # small carrying cost, as a fraction of cost
# Cap on the DP state space. This must be LARGER than any scenario we actually
# solve: silently clipping the starting stock turns an over-stocked clearance
# problem into a normally-stocked one, and the DP then correctly answers "never
# discount" to a question we did not mean to ask. Products needing more than
# this are skipped and reported, never truncated.
MAX_INVENTORY_GRID = 1500


# ==========================================================================
# Dynamic programme
# ==========================================================================
def solve_markdown_dp(base_demand_wk, base_price, cost, elasticity,
                      start_inventory, horizon=HORIZON_WEEKS,
                      discounts=DISCOUNTS):
    """
    Exact backward induction.

    Returns (policy, value_table, expected_profit) where
        policy[t, I] = index of the best discount in state (t, I)
    """
    I_max = int(min(start_inventory, MAX_INVENTORY_GRID))
    inv = np.arange(I_max + 1)
    salvage = cost * SALVAGE_FRACTION_OF_COST
    holding = cost * HOLDING_COST_PER_UNIT_WEEK

    # demand mean and unit VALUE for each discount option.
    #
    # COST IS SUNK. The stock is already bought and paid for; no decision we
    # make now can un-buy it. So the clearance objective is to maximise CASH
    # RECOVERED, i.e. revenue plus whatever the leftovers salvage for - not
    # (price - cost) margin. Subtracting cost per unit sold is the classic
    # mistake here: it makes salvaging look better than discounting whenever
    # the discounted margin falls below the salvage value, and the model then
    # refuses to ever mark down. We report profit against the sunk cost
    # separately, for interpretation.
    prices = base_price * (1 - discounts)
    mus = base_demand_wk * (prices / base_price) ** elasticity
    margins = prices

    # cap the demand support generously above the largest mean
    D_max = int(max(10, np.ceil(mus.max() + 6 * np.sqrt(mus.max() + 1))))
    ks = np.arange(D_max + 1)

    # sales[I, k] = min(k, I); leftover[I, k] = I - sales
    sales = np.minimum(ks[None, :], inv[:, None])
    leftover = inv[:, None] - sales

    V = inv * salvage                       # terminal value at the deadline
    policy = np.zeros((horizon, I_max + 1), dtype=int)

    for t in range(horizon - 1, -1, -1):
        best_val = np.full(I_max + 1, -np.inf)
        best_act = np.zeros(I_max + 1, dtype=int)
        for a, (mu, margin) in enumerate(zip(mus, margins)):
            pmf = poisson.pmf(ks, mu)
            pmf[-1] += max(0.0, 1.0 - pmf.sum())     # put the tail on the last cell
            # immediate profit + discounted future value, in expectation
            immediate = sales * margin - leftover * holding
            future = V[leftover]
            ev = (pmf[None, :] * (immediate + future)).sum(axis=1)
            better = ev > best_val
            best_val[better] = ev[better]
            best_act[better] = a
        V = best_val
        policy[t] = best_act

    return policy, V, float(V[I_max])


def simulate_policy(policy_fn, base_demand_wk, base_price, cost, elasticity,
                    start_inventory, horizon=HORIZON_WEEKS, n_sims=400, seed=0):
    """
    Monte-Carlo evaluation. Used both to score fixed rules and to CHECK that the
    DP's expected value matches what actually happens when you follow it.
    """
    rng = np.random.default_rng(seed)
    salvage = cost * SALVAGE_FRACTION_OF_COST
    holding = cost * HOLDING_COST_PER_UNIT_WEEK
    totals, sell_through, weeks_disc = [], [], []

    for s in range(n_sims):
        inv = int(min(start_inventory, MAX_INVENTORY_GRID))
        profit = 0.0
        n_disc = 0
        for t in range(horizon):
            d = policy_fn(t, inv)
            price = base_price * (1 - d)
            mu = base_demand_wk * (price / base_price) ** elasticity
            demand = rng.poisson(max(mu, 0))
            sold = int(min(demand, inv))
            inv -= sold
            profit += sold * price - inv * holding      # cost is sunk
            n_disc += (d > 0)
        profit += inv * salvage
        totals.append(profit)   # = cash recovered, not margin
        sell_through.append(1 - inv / max(start_inventory, 1))
        weeks_disc.append(n_disc)

    return {
        "expected_profit": float(np.mean(totals)),
        "profit_std": float(np.std(totals)),
        "sell_through": float(np.mean(sell_through)),
        "weeks_discounted": float(np.mean(weeks_disc)),
    }


# ==========================================================================
# Benchmark policies
# ==========================================================================
def policy_never(t, inv):
    return 0.0


def policy_fixed(t, inv):
    """The common retail rule of thumb."""
    if t >= 6:
        return 0.40
    if t >= 3:
        return 0.20
    return 0.0


def policy_panic(t, inv):
    return 0.40


def make_dp_policy(policy_table, discounts=DISCOUNTS):
    I_max = policy_table.shape[1] - 1

    def f(t, inv):
        t = min(t, policy_table.shape[0] - 1)
        return float(discounts[policy_table[t, int(min(inv, I_max))]])
    return f


# ==========================================================================
# Orchestration
# ==========================================================================
def run(source):
    print("=" * 78)
    print(f"PHASE 7 - MARKDOWN OPTIMISATION (dynamic programming)   [{source}]")
    print("=" * 78)

    fc_path = OUTPUTS_DIR / f"forecast_test_{source}.csv"
    ela_path = OUTPUTS_DIR / f"elasticity_{source}.csv"
    for p, msg in [(fc_path, "src/forecast.py"), (ela_path, "src/elasticity.py")]:
        if not p.exists():
            print(f"    MISSING {p.name} - run  python {msg} {source}  first")
            return None

    fc = pd.read_csv(fc_path)
    ela = pd.read_csv(ela_path)
    emap = dict(zip(ela["upc"], ela["elast_final"]))
    dmap = dict(zip(ela["upc"], ela["description"]))

    print(f"\n[1] Clearance scenario (stated assumptions)")
    print(f"    horizon              : {HORIZON_WEEKS} weeks to the deadline")
    print(f"    discount menu        : {[f'{d:.0%}' for d in DISCOUNTS]}")
    print(f"    starting stock       : {WEEKS_OF_STOCK:.0f} weeks of baseline demand"
          f"  ({WEEKS_OF_STOCK / HORIZON_WEEKS:.1f}x what sells at full price)")
    print(f"    salvage value        : {SALVAGE_FRACTION_OF_COST:.0%} of unit cost")
    print(f"    holding cost         : {HOLDING_COST_PER_UNIT_WEEK:.0%} of cost per unit-week")

    # per-product weekly baseline demand across the chain
    agg = (fc.groupby(Canon.UPC)
           .agg(base_demand=("baseline_demand", "mean"),
                base_price=(Canon.BASE_PRICE, "mean"),
                cost=(Canon.UNIT_COST, "mean"),
                category=(Canon.CATEGORY, "first"),
                n_store_weeks=(Canon.UNITS, "size"))
           .reset_index())
    # scale a single store's weekly demand up to a store-level clearance problem
    agg["weekly_demand"] = agg["base_demand"] * 1.0

    rows, skipped = [], []
    print(f"\n[2] Solving {len(agg)} clearance problems by backward induction...")
    for r in agg.itertuples():
        e = emap.get(r.upc, np.nan)
        if not np.isfinite(e) or e >= -0.05 or r.weekly_demand <= 0.5:
            continue
        start_inv = int(round(r.weekly_demand * WEEKS_OF_STOCK))
        if start_inv < 5:
            continue
        if start_inv > MAX_INVENTORY_GRID:      # skip, never silently truncate
            skipped.append(r.upc)
            continue

        policy, V, dp_expected = solve_markdown_dp(
            r.weekly_demand, r.base_price, r.cost, e, start_inv)
        dp_fn = make_dp_policy(policy)

        res = {}
        for name, fn in [("never", policy_never), ("fixed", policy_fixed),
                         ("panic", policy_panic), ("dp", dp_fn)]:
            res[name] = simulate_policy(fn, r.weekly_demand, r.base_price,
                                        r.cost, e, start_inv, seed=hash(r.upc) % 2**31)

        rows.append({
            "upc": r.upc,
            "description": dmap.get(r.upc, ""),
            "category": r.category,
            "elasticity": e,
            "weekly_demand": r.weekly_demand,
            "start_inventory": start_inv,
            "base_price": r.base_price,
            "unit_cost": r.cost,
            "dp_expected_value": dp_expected,
            **{f"{k}_profit": v["expected_profit"] for k, v in res.items()},
            **{f"{k}_sellthrough": v["sell_through"] for k, v in res.items()},
            "dp_weeks_discounted": res["dp"]["weeks_discounted"],
            "first_week_discount": float(DISCOUNTS[policy[0, min(start_inv,
                                                                policy.shape[1] - 1)]]),
            "gain_vs_fixed": res["dp"]["expected_profit"] - res["fixed"]["expected_profit"],
            "gain_vs_never": res["dp"]["expected_profit"] - res["never"]["expected_profit"],
        })

    md = pd.DataFrame(rows)
    if md.empty:
        print("    no products qualified for a clearance scenario")
        return None
    print(f"    solved {len(md)} products"
          + (f"; skipped {len(skipped)} needing >{MAX_INVENTORY_GRID} units" if skipped else ""))

    # ------------------------------------------------------------- DP check
    print(f"\n[3] Validation: does following the DP policy actually deliver its")
    print(f"    predicted value? (Monte-Carlo, 400 runs per product)")
    err = (md["dp_profit"] - md["dp_expected_value"]).abs() / md["dp_expected_value"].abs()
    print(f"    median relative gap between DP value and simulation: {err.median():.3%}")
    print(f"    {'AGREE - the DP is solved correctly' if err.median() < 0.05 else 'DISAGREE - investigate'}")

    # --------------------------------------------------------- policy compare
    print(f"\n[4] Policy comparison (expected profit per product, averaged)")
    print(f"    {'policy':<22}{'profit':>12}{'sell-through':>15}{'vs DP':>10}")
    print("    " + "-" * 59)
    dp_mean = md["dp_profit"].mean()
    for name, label in [("never", "never mark down"), ("fixed", "fixed -20%/-40%"),
                        ("panic", "panic -40% now"), ("dp", "DP optimal")]:
        p = md[f"{name}_profit"].mean()
        st = md[f"{name}_sellthrough"].mean()
        gap = (p / dp_mean - 1) * 100
        print(f"    {label:<22}{p:>12,.0f}{st:>14.0%}{gap:>9.1f}%")

    tot_dp = md["dp_profit"].sum()
    tot_fixed = md["fixed_profit"].sum()
    tot_never = md["never_profit"].sum()
    print(f"\n[5] Total across all clearance problems")
    print(f"    DP optimal vs fixed schedule : {tot_dp - tot_fixed:>+12,.0f}  "
          f"({(tot_dp / tot_fixed - 1) * 100:+.1f}%)")
    print(f"    DP optimal vs never marking  : {tot_dp - tot_never:>+12,.0f}  "
          f"({(tot_dp / tot_never - 1) * 100:+.1f}%)")

    # ------------------------------------------------------- what DP learned
    print(f"\n[6] What the optimal policy actually does")
    print(f"    mean weeks spent discounting : {md['dp_weeks_discounted'].mean():.1f} "
          f"of {HORIZON_WEEKS}")
    print(f"    starts at full price         : {(md['first_week_discount'] == 0).mean():.0%} "
          f"of products")
    corr = md["elasticity"].corr(md["dp_weeks_discounted"])
    corr_s = f"{corr:+.2f}" if np.isfinite(corr) else "n/a (no variation)"
    print(f"    corr(elasticity, weeks discounted) = {corr_s}")
    print("    -> more elastic products earn their discounts back, so the")
    print("       policy is willing to cut earlier and more often")

    # ----------------------------------------------------- example trajectory
    ex = md.nlargest(1, "gain_vs_fixed").iloc[0]
    print(f"\n[7] Example markdown ladder: {ex.description[:34]}")
    print(f"    elasticity {ex.elasticity:.2f}, start stock {int(ex.start_inventory)} units, "
          f"base price {ex.base_price:.2f}")
    pol, _, _ = solve_markdown_dp(ex.weekly_demand, ex.base_price, ex.unit_cost,
                                  ex.elasticity, int(ex.start_inventory))
    inv = int(ex.start_inventory)
    print(f"    {'week':>5}{'stock':>8}{'discount':>10}{'price':>9}")
    for t in range(HORIZON_WEEKS):
        d = DISCOUNTS[pol[t, min(inv, pol.shape[1] - 1)]]
        price = ex.base_price * (1 - d)
        print(f"    {t + 1:>5}{inv:>8}{d:>9.0%}{price:>9.2f}")
        mu = ex.weekly_demand * (price / ex.base_price) ** ex.elasticity
        inv = max(0, inv - int(round(mu)))
    print("    (stock path shown at expected demand; the policy adapts to")
    print("     whatever stock actually remains each week)")

    # ------------------------------------------------------------ by category
    print(f"\n[8] Gain over the fixed schedule, by category")
    summ = (md.groupby("category")
            .agg(products=("upc", "count"),
                 mean_elasticity=("elasticity", "mean"),
                 dp_gain_vs_fixed=("gain_vs_fixed", "sum"),
                 dp_sellthrough=("dp_sellthrough", "mean"))
            .round(3))
    print(summ.to_string())

    out = OUTPUTS_DIR / f"markdown_policy_{source}.csv"
    md.round(4).to_csv(out, index=False)
    print(f"\n[9] Written: {out.name}")
    print("=" * 78)
    return md


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else None
    for s in ([which] if which else ["synthetic", "real"]):
        run(s)
        print()
