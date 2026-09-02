"""
Central configuration for the Retail Profit Optimization project.

Every module imports paths, constants and business assumptions from here so
that nothing is hard-coded twice. Assumptions that are NOT observable in the
raw data (gross margin, holding cost, stockout cost, lead time) are declared
explicitly in this file and carried into the report as stated assumptions.
"""

from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
FRAT_DIR = RAW_DIR / "frat"           # dunnhumby "Breakfast at the Frat"
JOURNEY_DIR = RAW_DIR / "journey"     # dunnhumby "The Complete Journey"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

for _d in (DATA_DIR, RAW_DIR, FRAT_DIR, JOURNEY_DIR, PROCESSED_DIR, OUTPUTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# Canonical processed files produced by Phase 2 and consumed by everything after
SALES_PARQUET = PROCESSED_DIR / "sales_weekly.csv"
STORES_CSV = PROCESSED_DIR / "stores.csv"
PRODUCTS_CSV = PROCESSED_DIR / "products.csv"

# --------------------------------------------------------------------------
# Data source switch
# --------------------------------------------------------------------------
# "synthetic" -> use the schema-conformant simulator (works today, no download)
# "real"      -> use the actual dunnhumby files once they land in data/raw/frat
DATA_SOURCE = "real"

RANDOM_SEED = 42

# --------------------------------------------------------------------------
# Temporal splits (week index, 1-based). NEVER split randomly on time series.
# --------------------------------------------------------------------------
N_WEEKS = 156
TRAIN_END_WEEK = 110      # weeks 1..110
VALID_END_WEEK = 130      # weeks 111..130
# test = weeks 131..156, touched exactly once at the very end

# --------------------------------------------------------------------------
# Business assumptions (NOT in the raw data - documented, and sensitivity-tested)
# --------------------------------------------------------------------------
# Gross margin by category, from published grocery/CPG retail benchmarks.
# Unit cost is derived as: cost = BASE_PRICE * (1 - gross_margin)
GROSS_MARGIN_BY_CATEGORY = {
    "COLD CEREAL": 0.30,
    "FROZEN PIZZA": 0.26,
    "BAG SNACKS": 0.36,
    "ORAL HYGIENE PRODUCTS": 0.42,
}
DEFAULT_GROSS_MARGIN = 0.30

# Sensitivity band used when we re-run the profit calculation to show the
# conclusion is robust to the margin assumption.
MARGIN_SENSITIVITY = (-0.05, 0.0, +0.05)

# Inventory economics (used by the RL replenishment simulator in a later phase)
HOLDING_COST_RATE_PER_WEEK = 0.004     # fraction of unit cost, per unit, per week
STOCKOUT_PENALTY_MULTIPLIER = 1.5      # times lost gross margin
DEFAULT_LEAD_TIME_WEEKS = 2

# Pricing guardrails (Feature F) - prevent runaway / collusive-looking pricing
MAX_PRICE_INCREASE_PCT = 0.10          # never raise price >10% in one step
MAX_PRICE_DECREASE_PCT = 0.40          # never discount deeper than 40%
MIN_GROSS_MARGIN_FLOOR = 0.05          # never price below cost + 5%


def margin_for_category(category: str) -> float:
    """Look up gross margin for a category, case-insensitively."""
    if category is None:
        return DEFAULT_GROSS_MARGIN
    return GROSS_MARGIN_BY_CATEGORY.get(str(category).upper(), DEFAULT_GROSS_MARGIN)


def split_of_week(week_index: int) -> str:
    """Return 'train' | 'valid' | 'test' for a 1-based week index."""
    if week_index <= TRAIN_END_WEEK:
        return "train"
    if week_index <= VALID_END_WEEK:
        return "valid"
    return "test"


if __name__ == "__main__":
    print("PROJECT_ROOT :", PROJECT_ROOT)
    print("FRAT_DIR     :", FRAT_DIR)
    print("PROCESSED_DIR:", PROCESSED_DIR)
    print("OUTPUTS_DIR  :", OUTPUTS_DIR)
    print("DATA_SOURCE  :", DATA_SOURCE)
    print("Splits       : train<=%d, valid<=%d, test<=%d"
          % (TRAIN_END_WEEK, VALID_END_WEEK, N_WEEKS))