"""
Phase 2 - the single entry point for data into this project.

Reads EITHER the real dunnhumby workbook OR the synthetic stand-in, normalises
both to one canonical schema, derives the economics we need (unit cost, gross
profit, discount depth), assigns temporal splits, and writes the processed
files that every later module consumes.

Design notes
------------
1. Sheets in the real workbook are identified by their COLUMN CONTENT, not by
   sheet name. Different distributions of the dunnhumby file name the sheets
   differently, and hard-coding names is the usual reason these loaders break.

2. The header row is auto-detected, because the real workbook often carries a
   title row above the real header.

3. Unit cost is NOT in the data - no retailer publishes COGS. We derive it from
   a documented per-category gross-margin assumption in config.py, and later
   phases re-run the profit numbers across a margin band to show the conclusion
   does not depend on the exact figure.

4. Splits are strictly temporal. Weeks 1-110 train, 111-130 validation,
   131-156 test. A random split would leak the future into the past.

Run:
    python src/data_loader.py
"""

import sys
import numpy as np
import pandas as pd

from config import (FRAT_DIR, PROCESSED_DIR, DATA_SOURCE, N_WEEKS,
                    TRAIN_END_WEEK, VALID_END_WEEK, margin_for_category)
from schema import FratRaw, Canon, validate_canonical


# ==========================================================================
# Source discovery
# ==========================================================================
def find_real_workbook():
    for pattern in ("*.xlsx", "*.xls"):
        hits = sorted(FRAT_DIR.glob(pattern))
        if hits:
            return hits[0]
    return None


def find_synthetic_csvs():
    need = ["frat_sales.csv", "frat_stores.csv", "frat_products.csv"]
    paths = {n: FRAT_DIR / n for n in need}
    return paths if all(p.exists() for p in paths.values()) else None


# ==========================================================================
# Excel reading helpers
# ==========================================================================
def _detect_header_row(xls, sheet, max_scan=8):
    """Find the row index that actually holds the column names."""
    peek = pd.read_excel(xls, sheet_name=sheet, header=None, nrows=max_scan)
    for i in range(len(peek)):
        n_str = sum(isinstance(v, str) and str(v).strip() != "" for v in peek.iloc[i])
        if n_str >= 3:
            return i
    return 0


def _normalise_cols(df):
    """Upper-case, strip, and collapse internal whitespace in column names."""
    df = df.copy()
    df.columns = [str(c).strip().upper().replace(" ", "_") for c in df.columns]
    return df


def _classify_sheet(columns):
    """Decide what a sheet is, by how many known columns it carries."""
    cols = set(columns)
    scores = {
        "sales": len(cols & set(FratRaw.SALES_COLUMNS)),
        "stores": len(cols & set(FratRaw.STORE_COLUMNS)),
        "products": len(cols & set(FratRaw.PRODUCT_COLUMNS)),
    }
    # sales is the only sheet with UNITS + PRICE
    if {"UNITS", "PRICE"} <= cols:
        return "sales", scores["sales"]
    best = max(scores, key=scores.get)
    return (best, scores[best]) if scores[best] >= 3 else (None, 0)


def load_real_workbook(path):
    print(f"  reading real workbook: {path.name}")
    xls = pd.ExcelFile(path)
    print(f"  sheets: {xls.sheet_names}")

    found = {}
    for sheet in xls.sheet_names:
        try:
            hdr = _detect_header_row(xls, sheet)
            df = pd.read_excel(xls, sheet_name=sheet, header=hdr)
            df = df.dropna(axis=1, how="all")
            df = _normalise_cols(df)
            kind, score = _classify_sheet(df.columns)
            print(f"    {sheet!r:<28} header_row={hdr}  rows={len(df):>7,}  "
                  f"-> {kind or 'UNRECOGNISED'} (score {score})")
            if kind and (kind not in found or score > found[kind][1]):
                found[kind] = (df, score)
        except Exception as exc:
            print(f"    {sheet!r}: skipped ({exc})")

    missing = [k for k in ("sales", "stores", "products") if k not in found]
    if "sales" in missing:
        raise ValueError("Could not find a sales sheet (needs UNITS and PRICE columns). "
                         "Run  python src/inspect_raw.py  and send the output.")
    if missing:
        print(f"  NOTE: no sheet found for {missing} - continuing without it")

    return (found["sales"][0],
            found["stores"][0] if "stores" in found else None,
            found["products"][0] if "products" in found else None)


def load_synthetic(paths):
    print("  reading synthetic CSVs")
    sales = _normalise_cols(pd.read_csv(paths["frat_sales.csv"]))
    stores = _normalise_cols(pd.read_csv(paths["frat_stores.csv"]))
    products = _normalise_cols(pd.read_csv(paths["frat_products.csv"]))
    return sales, stores, products


# ==========================================================================
# Normalisation to the canonical schema
# ==========================================================================
def to_canonical(sales, stores, products):
    q = pd.DataFrame()

    q[Canon.WEEK_END_DATE] = pd.to_datetime(sales[FratRaw.WEEK_END_DATE], errors="coerce")
    q[Canon.STORE_ID] = pd.to_numeric(sales[FratRaw.STORE_NUM], errors="coerce")
    q[Canon.UPC] = pd.to_numeric(sales[FratRaw.UPC], errors="coerce")

    for src, dst in [(FratRaw.UNITS, Canon.UNITS), (FratRaw.SPEND, Canon.SPEND),
                     (FratRaw.PRICE, Canon.PRICE), (FratRaw.BASE_PRICE, Canon.BASE_PRICE),
                     (FratRaw.VISITS, Canon.VISITS), (FratRaw.HHS, Canon.HHS)]:
        q[dst] = pd.to_numeric(sales[src], errors="coerce") if src in sales.columns else np.nan

    for src, dst in [(FratRaw.FEATURE, Canon.FEATURE), (FratRaw.DISPLAY, Canon.DISPLAY),
                     (FratRaw.TPR_ONLY, Canon.TPR_ONLY)]:
        col = pd.to_numeric(sales[src], errors="coerce") if src in sales.columns else 0
        q[dst] = (pd.Series(col, index=q.index).fillna(0) > 0).astype(int)

    # ---- attach product attributes
    if products is not None and FratRaw.UPC in products.columns:
        p = products.copy()
        p[FratRaw.UPC] = pd.to_numeric(p[FratRaw.UPC], errors="coerce")
        keep = {FratRaw.CATEGORY: Canon.CATEGORY,
                FratRaw.SUB_CATEGORY: Canon.SUB_CATEGORY,
                FratRaw.MANUFACTURER: Canon.MANUFACTURER,
                FratRaw.DESCRIPTION: Canon.DESCRIPTION}
        cols = {k: v for k, v in keep.items() if k in p.columns}
        p = p[[FratRaw.UPC] + list(cols)].rename(columns={FratRaw.UPC: Canon.UPC, **cols})
        p = p.drop_duplicates(subset=[Canon.UPC])
        q = q.merge(p, on=Canon.UPC, how="left")

    # ---- attach store price tier
    if stores is not None and FratRaw.STORE_ID in stores.columns:
        s = stores.copy()
        s[FratRaw.STORE_ID] = pd.to_numeric(s[FratRaw.STORE_ID], errors="coerce")
        if FratRaw.SEG_VALUE_NAME in s.columns:
            s = s[[FratRaw.STORE_ID, FratRaw.SEG_VALUE_NAME]].rename(
                columns={FratRaw.STORE_ID: Canon.STORE_ID,
                         FratRaw.SEG_VALUE_NAME: Canon.PRICE_TIER})
            s = s.drop_duplicates(subset=[Canon.STORE_ID])
            q = q.merge(s, on=Canon.STORE_ID, how="left")

    for c in (Canon.CATEGORY, Canon.SUB_CATEGORY, Canon.MANUFACTURER,
              Canon.DESCRIPTION, Canon.PRICE_TIER):
        if c not in q.columns:
            q[c] = "UNKNOWN"
        q[c] = q[c].fillna("UNKNOWN").astype(str).str.strip().str.upper()

    return q


def clean(q):
    """Drop unusable rows, repair recoverable ones. Reports everything it does."""
    report = []
    n0 = len(q)

    q = q.dropna(subset=[Canon.WEEK_END_DATE, Canon.STORE_ID, Canon.UPC,
                         Canon.UNITS, Canon.PRICE])
    if len(q) < n0:
        report.append(f"dropped {n0 - len(q):,} rows with null key/price/units")

    n = len(q)
    q = q[(q[Canon.UNITS] > 0) & (q[Canon.PRICE] > 0)]
    if len(q) < n:
        report.append(f"dropped {n - len(q):,} rows with non-positive units or price")

    q = q.copy()

    # base_price missing or below shelf price -> fall back to shelf price
    bad_bp = q[Canon.BASE_PRICE].isna() | (q[Canon.BASE_PRICE] < q[Canon.PRICE])
    if bad_bp.any():
        report.append(f"repaired {int(bad_bp.sum()):,} rows where base_price was "
                      f"missing or below price")
        q.loc[bad_bp, Canon.BASE_PRICE] = q.loc[bad_bp, Canon.PRICE]

    # spend missing -> reconstruct
    bad_spend = q[Canon.SPEND].isna() | (q[Canon.SPEND] <= 0)
    if bad_spend.any():
        report.append(f"reconstructed spend for {int(bad_spend.sum()):,} rows")
        q.loc[bad_spend, Canon.SPEND] = (q.loc[bad_spend, Canon.UNITS]
                                         * q.loc[bad_spend, Canon.PRICE])

    # cap absurd discounts (data errors), keep genuine deep cuts
    disc = 1 - q[Canon.PRICE] / q[Canon.BASE_PRICE]
    extreme = disc > 0.90
    if extreme.any():
        report.append(f"clipped {int(extreme.sum()):,} rows with >90% implied discount")
        q.loc[extreme, Canon.BASE_PRICE] = q.loc[extreme, Canon.PRICE] / 0.10

    q = q.drop_duplicates(subset=[Canon.WEEK_END_DATE, Canon.STORE_ID, Canon.UPC])
    q[Canon.STORE_ID] = q[Canon.STORE_ID].astype("int64")
    q[Canon.UPC] = q[Canon.UPC].astype("int64")
    q[Canon.UNITS] = q[Canon.UNITS].astype("int64")

    return q.reset_index(drop=True), report


def derive(q):
    """Add week_index, discount, promo flag, unit cost, gross profit, split."""
    q = q.copy()

    weeks = np.sort(q[Canon.WEEK_END_DATE].unique())
    week_map = {w: i + 1 for i, w in enumerate(weeks)}
    q[Canon.WEEK_INDEX] = q[Canon.WEEK_END_DATE].map(week_map).astype("int64")

    q[Canon.DISCOUNT_PCT] = (1 - q[Canon.PRICE] / q[Canon.BASE_PRICE]).clip(0, 0.95)
    q[Canon.ANY_PROMO] = ((q[Canon.FEATURE] + q[Canon.DISPLAY]
                           + q[Canon.TPR_ONLY]) > 0).astype(int)

    # --- unit cost from the documented per-category gross-margin assumption
    margins = q[Canon.CATEGORY].map(margin_for_category)
    q[Canon.UNIT_COST] = (q[Canon.BASE_PRICE] * (1 - margins)).round(4)
    q[Canon.GROSS_PROFIT] = (q[Canon.UNITS] *
                             (q[Canon.PRICE] - q[Canon.UNIT_COST])).round(2)

    # --- strictly temporal split, scaled if the file has != N_WEEKS weeks
    n_weeks = len(weeks)
    tr = round(TRAIN_END_WEEK * n_weeks / N_WEEKS)
    va = round(VALID_END_WEEK * n_weeks / N_WEEKS)
    q[Canon.SPLIT] = np.where(q[Canon.WEEK_INDEX] <= tr, "train",
                              np.where(q[Canon.WEEK_INDEX] <= va, "valid", "test"))
    return q, (n_weeks, tr, va)


# ==========================================================================
# Orchestration
# ==========================================================================
def build(prefer=None):
    source = prefer or DATA_SOURCE
    print("=" * 68)
    print("PHASE 2 - DATA LOADER")
    print("=" * 68)

    wb = find_real_workbook()
    csvs = find_synthetic_csvs()

    print("\n[1] Source discovery")
    print(f"  real workbook : {wb.name if wb else 'not found'}")
    print(f"  synthetic CSVs: {'found' if csvs else 'not found'}")
    print(f"  config prefers: {source}")

    if source == "real" and wb is None:
        print("  !! DATA_SOURCE='real' but no workbook found - falling back to synthetic")
        source = "synthetic"
    if source == "synthetic" and csvs is None:
        if wb is None:
            raise FileNotFoundError("No data found. Run src/make_synthetic_frat.py "
                                    "or place the dunnhumby .xlsx in data/raw/frat/")
        print("  !! no synthetic CSVs - using the real workbook instead")
        source = "real"

    print(f"\n[2] Loading ({source})")
    if source == "real":
        sales, stores, products = load_real_workbook(wb)
    else:
        sales, stores, products = load_synthetic(csvs)
    print(f"  raw sales rows: {len(sales):,}")

    print("\n[3] Normalising to canonical schema")
    q = to_canonical(sales, stores, products)
    print(f"  columns: {len(q.columns)}")

    print("\n[4] Cleaning")
    q, report = clean(q)
    if report:
        for line in report:
            print("  -", line)
    else:
        print("  - nothing to repair")
    print(f"  rows after cleaning: {len(q):,}")

    print("\n[5] Deriving economics and splits")
    q, (n_weeks, tr, va) = derive(q)
    print(f"  weeks detected: {n_weeks}  ->  train<= {tr}, valid<= {va}, test> {va}")

    print("\n[6] Schema validation")
    validate_canonical(q).raise_if_bad()
    print("  [PASS] canonical contract satisfied")

    # ---------------------------------------------------------------- write
    # Tag the frame with its provenance, then write BOTH a source-specific file
    # (so real and synthetic can coexist and be compared) and the generic
    # sales_weekly.csv that later phases read by default.
    q["data_source"] = source
    out = PROCESSED_DIR / f"sales_weekly_{source}.csv"
    q.to_csv(out, index=False)
    q.to_csv(PROCESSED_DIR / "sales_weekly.csv", index=False)

    prod_out = (q.groupby(Canon.UPC)
                .agg(category=(Canon.CATEGORY, "first"),
                     sub_category=(Canon.SUB_CATEGORY, "first"),
                     manufacturer=(Canon.MANUFACTURER, "first"),
                     description=(Canon.DESCRIPTION, "first"),
                     mean_base_price=(Canon.BASE_PRICE, "mean"),
                     mean_unit_cost=(Canon.UNIT_COST, "mean"),
                     total_units=(Canon.UNITS, "sum"),
                     total_spend=(Canon.SPEND, "sum"),
                     total_gross_profit=(Canon.GROSS_PROFIT, "sum"))
                .reset_index())
    prod_out.to_csv(PROCESSED_DIR / f"products_{source}.csv", index=False)
    prod_out.to_csv(PROCESSED_DIR / "products.csv", index=False)

    store_out = (q.groupby(Canon.STORE_ID)
                 .agg(price_tier=(Canon.PRICE_TIER, "first"),
                      total_units=(Canon.UNITS, "sum"),
                      total_spend=(Canon.SPEND, "sum"),
                      total_gross_profit=(Canon.GROSS_PROFIT, "sum"),
                      promo_rate=(Canon.ANY_PROMO, "mean"))
                 .reset_index())
    store_out.to_csv(PROCESSED_DIR / f"stores_{source}.csv", index=False)
    store_out.to_csv(PROCESSED_DIR / "stores.csv", index=False)

    # ---------------------------------------------------------------- summary
    print("\n[7] Summary")
    print(f"  rows            : {len(q):,}")
    print(f"  weeks           : {q[Canon.WEEK_INDEX].nunique()}")
    print(f"  stores          : {q[Canon.STORE_ID].nunique()}")
    print(f"  products (UPC)  : {q[Canon.UPC].nunique()}")
    print(f"  categories      : {q[Canon.CATEGORY].nunique()} "
          f"-> {sorted(q[Canon.CATEGORY].unique())[:5]}")
    print(f"  date range      : {q[Canon.WEEK_END_DATE].min().date()} .. "
          f"{q[Canon.WEEK_END_DATE].max().date()}")
    print(f"  promo rate      : {q[Canon.ANY_PROMO].mean():.1%}")
    print(f"  mean discount   : {q.loc[q[Canon.ANY_PROMO] == 1, Canon.DISCOUNT_PCT].mean():.1%} "
          f"(when promoted)")
    print(f"  total revenue   : {q[Canon.SPEND].sum():,.0f}")
    print(f"  total gross prof: {q[Canon.GROSS_PROFIT].sum():,.0f} "
          f"({q[Canon.GROSS_PROFIT].sum() / q[Canon.SPEND].sum():.1%} of revenue)")

    print("\n  split sizes:")
    for name, grp in q.groupby(Canon.SPLIT, sort=False):
        print(f"    {name:<6} {len(grp):>9,} rows   "
              f"weeks {grp[Canon.WEEK_INDEX].min():>3}-{grp[Canon.WEEK_INDEX].max():<3}")

    print("\n  written to", PROCESSED_DIR)
    print("   ", f"sales_weekly_{source}.csv   (+ sales_weekly.csv)")
    print("   ", f"products_{source}.csv       (+ products.csv)")
    print("   ", f"stores_{source}.csv         (+ stores.csv)")
    print("=" * 68)
    return q


def load_processed(source=None):
    """
    Convenience used by every later phase.

    source=None  -> whichever was built last (sales_weekly.csv)
    source="real"/"synthetic" -> that specific build, so results can be compared
    """
    name = "sales_weekly.csv" if source is None else f"sales_weekly_{source}.csv"
    path = PROCESSED_DIR / name
    if not path.exists():
        raise FileNotFoundError(
            f"{name} not found. Run  python src/data_loader.py "
            f"{source or ''}".strip() + "  first.")
    return pd.read_csv(path, parse_dates=[Canon.WEEK_END_DATE])


if __name__ == "__main__":
    prefer = sys.argv[1] if len(sys.argv) > 1 else None
    build(prefer)
