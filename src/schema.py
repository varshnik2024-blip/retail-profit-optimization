"""
Canonical schema contract for the project.

This file is the single source of truth for what columns our pipeline expects.
It is derived from the published dunnhumby "Breakfast at the Frat" user guide,
so the synthetic simulator and the real dataset are interchangeable: every
downstream module codes against THIS contract, never against a specific file.

When the real dunnhumby download arrives, only the loader changes.
Nothing else does.
"""

from dataclasses import dataclass


# --------------------------------------------------------------------------
# Raw dunnhumby "Breakfast at the Frat" column names (as shipped)
# --------------------------------------------------------------------------
class FratRaw:
    """Column names exactly as they appear in the dunnhumby workbook."""

    # transaction / sales sheet
    WEEK_END_DATE = "WEEK_END_DATE"
    STORE_NUM = "STORE_NUM"
    UPC = "UPC"
    UNITS = "UNITS"          # units sold          -> our forecasting target
    VISITS = "VISITS"        # transactions
    HHS = "HHS"              # households buying
    SPEND = "SPEND"          # revenue
    PRICE = "PRICE"          # actual shelf price
    BASE_PRICE = "BASE_PRICE"  # non-promoted reference price
    FEATURE = "FEATURE"      # in weekly ad / mailer      (treatment 1)
    DISPLAY = "DISPLAY"      # in-store display           (treatment 2)
    TPR_ONLY = "TPR_ONLY"    # temporary price reduction  (treatment 3)

    SALES_COLUMNS = [
        WEEK_END_DATE, STORE_NUM, UPC, UNITS, VISITS, HHS, SPEND,
        PRICE, BASE_PRICE, FEATURE, DISPLAY, TPR_ONLY,
    ]

    # store sheet
    STORE_ID = "STORE_ID"
    STORE_NAME = "STORE_NAME"
    CITY = "ADDRESS_CITY_NAME"
    STATE = "ADDRESS_STATE_PROV_CODE"
    MSA_CODE = "MSA_CODE"
    SEG_VALUE_NAME = "SEG_VALUE_NAME"       # price tier: UPSCALE / MAINSTREAM / VALUE
    PARKING = "PARKING_SPACE_QTY"
    SALES_AREA = "SALES_AREA_SIZE_NUM"
    AVG_WEEKLY_BASKETS = "AVG_WEEKLY_BASKETS"

    STORE_COLUMNS = [
        STORE_ID, STORE_NAME, CITY, STATE, MSA_CODE, SEG_VALUE_NAME,
        PARKING, SALES_AREA, AVG_WEEKLY_BASKETS,
    ]

    # product sheet
    DESCRIPTION = "DESCRIPTION"
    MANUFACTURER = "MANUFACTURER"
    CATEGORY = "CATEGORY"
    SUB_CATEGORY = "SUB_CATEGORY"
    PRODUCT_SIZE = "PRODUCT_SIZE"

    PRODUCT_COLUMNS = [
        UPC, DESCRIPTION, MANUFACTURER, CATEGORY, SUB_CATEGORY, PRODUCT_SIZE,
    ]


# --------------------------------------------------------------------------
# Canonical (cleaned) schema used by every analysis module
# --------------------------------------------------------------------------
class Canon:
    """Lower-case, tidy column names produced by the Phase 2 loader."""

    WEEK_END_DATE = "week_end_date"
    WEEK_INDEX = "week_index"        # 1..156, integer
    STORE_ID = "store_id"
    UPC = "upc"
    CATEGORY = "category"
    SUB_CATEGORY = "sub_category"
    MANUFACTURER = "manufacturer"
    DESCRIPTION = "description"
    PRICE_TIER = "price_tier"

    UNITS = "units"
    VISITS = "visits"
    HHS = "hhs"
    SPEND = "spend"
    PRICE = "price"
    BASE_PRICE = "base_price"

    FEATURE = "feature"
    DISPLAY = "display"
    TPR_ONLY = "tpr_only"

    # derived
    DISCOUNT_PCT = "discount_pct"    # 1 - price/base_price
    ANY_PROMO = "any_promo"          # feature | display | tpr_only
    UNIT_COST = "unit_cost"          # base_price * (1 - gross_margin)
    GROSS_PROFIT = "gross_profit"    # units * (price - unit_cost)
    SPLIT = "split"                  # train / valid / test

    REQUIRED = [
        WEEK_END_DATE, WEEK_INDEX, STORE_ID, UPC, UNITS, SPEND,
        PRICE, BASE_PRICE, FEATURE, DISPLAY, TPR_ONLY,
    ]


@dataclass
class ValidationResult:
    ok: bool
    messages: list

    def raise_if_bad(self):
        if not self.ok:
            raise ValueError("Schema validation failed:\n  - " +
                             "\n  - ".join(self.messages))
        return self


def validate_canonical(df) -> ValidationResult:
    """Check a canonical sales frame satisfies the contract."""
    msgs = []
    missing = [c for c in Canon.REQUIRED if c not in df.columns]
    if missing:
        msgs.append(f"missing required columns: {missing}")
        return ValidationResult(False, msgs)

    if len(df) == 0:
        msgs.append("frame is empty")

    if df[Canon.UNITS].min() < 0:
        msgs.append("negative units found")
    if df[Canon.PRICE].min() <= 0:
        msgs.append("non-positive price found")
    if df[Canon.BASE_PRICE].min() <= 0:
        msgs.append("non-positive base_price found")

    for flag in (Canon.FEATURE, Canon.DISPLAY, Canon.TPR_ONLY):
        bad = set(df[flag].dropna().unique()) - {0, 1}
        if bad:
            msgs.append(f"{flag} has non-binary values: {sorted(bad)[:5]}")

    n_null = int(df[Canon.REQUIRED].isna().sum().sum())
    if n_null:
        msgs.append(f"{n_null} null values in required columns")

    return ValidationResult(len(msgs) == 0, msgs)


if __name__ == "__main__":
    print("Raw sales columns   :", FratRaw.SALES_COLUMNS)
    print("Raw store columns   :", FratRaw.STORE_COLUMNS)
    print("Raw product columns :", FratRaw.PRODUCT_COLUMNS)
    print("Canonical required  :", Canon.REQUIRED)