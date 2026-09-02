"""
Raw file inspector.

The real dunnhumby workbook ships as a single .xlsx with several sheets, and
the exact sheet names / header row position vary between distributions of the
file. This tool prints what is actually inside YOUR copy so the loader can be
confirmed against it rather than guessed at.

Run:
    python src/inspect_raw.py
"""

import sys
import pandas as pd

from config import FRAT_DIR


def find_excel():
    candidates = sorted(FRAT_DIR.glob("*.xlsx")) + sorted(FRAT_DIR.glob("*.xls"))
    return candidates[0] if candidates else None


def describe_sheet(xls, sheet_name):
    print(f"\n{'-' * 66}")
    print(f"SHEET: {sheet_name!r}")
    print("-" * 66)

    # read the top rows with NO header, so we can see where the real header is
    peek = pd.read_excel(xls, sheet_name=sheet_name, header=None, nrows=8)
    print("First 8 raw rows (header=None), first 8 columns:")
    with pd.option_context("display.max_columns", 12, "display.width", 200):
        print(peek.iloc[:, :8].to_string())

    # guess the header row: the first row where >=3 cells are non-null strings
    header_row = 0
    for i in range(len(peek)):
        row = peek.iloc[i]
        n_str = sum(isinstance(v, str) and v.strip() != "" for v in row)
        if n_str >= 3:
            header_row = i
            break
    print(f"\n>>> detected header row index: {header_row}")

    df = pd.read_excel(xls, sheet_name=sheet_name, header=header_row, nrows=500)
    df = df.dropna(axis=1, how="all")
    print(f">>> columns ({len(df.columns)}): {list(df.columns)}")
    print(">>> dtypes:")
    print(df.dtypes.to_string())
    print(">>> first 3 data rows:")
    with pd.option_context("display.max_columns", 20, "display.width", 250):
        print(df.head(3).to_string())


def main():
    path = find_excel()
    if path is None:
        print(f"No .xlsx found in {FRAT_DIR}")
        print("Download 'dunnhumby - Breakfast at the Frat.xlsx' and place it there.")
        return 1

    print("=" * 66)
    print("RAW EXCEL INSPECTION")
    print("=" * 66)
    print("File :", path.name)
    print("Size :", f"{path.stat().st_size / 1e6:.2f} MB")

    xls = pd.ExcelFile(path)
    print("Sheets:", xls.sheet_names)

    for sheet in xls.sheet_names:
        try:
            describe_sheet(xls, sheet)
        except Exception as exc:            # keep going; report and move on
            print(f"  !! could not read sheet {sheet!r}: {exc}")

    print("\n" + "=" * 66)
    print("Send this entire output back so the loader can be confirmed.")
    print("=" * 66)
    return 0


if __name__ == "__main__":
    sys.exit(main())