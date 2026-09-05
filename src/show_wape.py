"""
Live demonstration of the WAPE calculation.

This script exists to be run in front of someone. It does not compute
anything new - it re-derives the headline forecast accuracy figure from the
saved predictions, one step at a time, so the arithmetic behind
"WAPE 0.292" is visible rather than asserted.

Run:
    python src/show_wape.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import OUTPUTS_DIR                                   # noqa: E402
from schema import Canon                                         # noqa: E402

SOURCE = "real"
LINE = "-" * 68


def rule(title=""):
    print(f"\n{LINE}")
    if title:
        print(title)
        print(LINE)


# =========================================================================
# 1. The formula, stated before we touch any data
# =========================================================================
rule("STEP 1  -  WHAT WE ARE COMPUTING")
print("""
    WAPE = weighted absolute percentage error

                 sum of |actual - predicted|
        WAPE  =  ---------------------------
                     sum of |actual|

    In words: add up every unit we got wrong, divide by every unit that
    actually sold. WAPE 0.30 means we were off by 30 units for every
    100 units sold.

    Why not MAPE? MAPE averages the percentage error of each row on its
    own. A week where 1 unit sold and we predicted 6 scores a 500% error
    and drowns out a week where 4,000 units sold and we were 2% off.
    Retail data is full of tiny rows, so MAPE measures the noise.
    WAPE weights by volume, so big sellers count for more.
""")

# =========================================================================
# 2. The data - held-out weeks the model never saw
# =========================================================================
rule("STEP 2  -  THE DATA WE ARE SCORING ON")
fc = pd.read_csv(OUTPUTS_DIR / f"forecast_test_{SOURCE}.csv")

actual = fc[Canon.UNITS].to_numpy(dtype=float)
pred = fc["pred"].to_numpy(dtype=float)

print(f"""
    file        outputs/forecast_test_{SOURCE}.csv
    rows        {len(fc):,}   (one per store x product x week)
    weeks       {fc['week_index'].min()} to {fc['week_index'].max()}
    stores      {fc['store_id'].nunique()}
    products    {fc[Canon.UPC].nunique()}

    These are HELD-OUT weeks. The model was trained on weeks 1-110 and
    tuned on 111-130. It has never seen a single row below.
""")

print("    First five rows:\n")
show = fc[["week_index", "store_id", Canon.UPC, Canon.UNITS, "pred"]].head()
print(show.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

# =========================================================================
# 3. Row-level arithmetic, small enough to check by hand
# =========================================================================
rule("STEP 3  -  THE ARITHMETIC, ROW BY ROW")
print(f"\n    {'actual':>10}{'predicted':>12}{'error':>10}{'|error|':>10}")
print("    " + "-" * 42)
for a, p in zip(actual[:8], pred[:8]):
    print(f"    {a:>10.0f}{p:>12.3f}{a - p:>10.3f}{abs(a - p):>10.3f}")
print("    " + "-" * 42)
print("    ... and so on for all "
      f"{len(actual):,} rows.")

# =========================================================================
# 4. The two sums
# =========================================================================
rule("STEP 4  -  ADD UP BOTH COLUMNS")
abs_err_sum = np.abs(actual - pred).sum()
actual_sum = np.abs(actual).sum()

print(f"""
    sum of |actual - predicted|  =  {abs_err_sum:>15,.1f}   units of error
    sum of |actual|              =  {actual_sum:>15,.1f}   units actually sold
""")

# =========================================================================
# 5. Divide
# =========================================================================
rule("STEP 5  -  DIVIDE")
w = abs_err_sum / actual_sum
print(f"""
        {abs_err_sum:,.1f}  /  {actual_sum:,.1f}  =  {w:.4f}

    WAPE = {w:.4f}   ->   rounded, {w:.3f}

    Read it as: for every 100 units that sold, we were off by {w * 100:.1f}.
""")

# =========================================================================
# 6. Prove it matches the saved figure and the dashboard
# =========================================================================
rule("STEP 6  -  DOES THIS MATCH WHAT WE PUBLISHED?")
met = pd.read_csv(OUTPUTS_DIR / f"forecast_metrics_{SOURCE}.csv")
saved = met[(met["split"] == "test")
            & met["model"].str.contains("gradient", case=False)]["WAPE"].iloc[0]

print(f"""
    recomputed here          {w:.4f}
    saved in the pipeline    {saved:.4f}
    shown on the dashboard   {saved:.3f}

    difference               {abs(w - saved):.2e}
""")
print("    MATCH - the dashboard number is this arithmetic.\n"
      if abs(w - saved) < 1e-4 else
      "    MISMATCH - investigate before presenting.\n")

# =========================================================================
# 7. The number only means something next to a baseline
# =========================================================================
rule("STEP 7  -  IS 0.292 ACTUALLY GOOD?")
print("""
    A WAPE on its own means nothing. It has to beat the dumb answer.
    Same rows, same metric, three baselines a shop could use for free:
""")
test = met[met["split"] == "test"].copy()
best_base = test[~test["model"].str.contains("gradient", case=False)] \
    .sort_values("WAPE").iloc[0]

print(f"    {'model':<26}{'WAPE':>8}{'vs our model':>16}")
print("    " + "-" * 50)
for _, r in test.sort_values("WAPE", ascending=False).iterrows():
    mark = "  <-- ours" if "gradient" in r["model"].lower() else ""
    delta = "" if "gradient" in r["model"].lower() \
        else f"{(r['WAPE'] - saved) / r['WAPE'] * 100:>14.1f}%"
    print(f"    {r['model']:<26}{r['WAPE']:>8.4f}{delta}{mark}")

gain = (best_base["WAPE"] - saved) / best_base["WAPE"] * 100
print(f"""
    Toughest baseline: {best_base['model']} at WAPE {best_base['WAPE']:.4f}.
    Our model is {gain:.1f}% better than that.

    That comparison is why the ledger grades forecast accuracy as
    MEASURED. We predicted, then we looked at what really sold. No
    assumption enters the number.
""")
print(LINE)
