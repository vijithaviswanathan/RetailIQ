
from pathlib import Path

import pandas as pd
from openpyxl import load_workbook


PROJECT_ROOT = Path(__file__).resolve().parents[3]
RAW_DIR = PROJECT_ROOT / "data" / "raw"

HILLSTROM_FILE = RAW_DIR / "hillstrom.csv"
RETAIL_FILE = RAW_DIR / "online_retail_II.xlsx"


def inspect_hillstrom():
    print("\n" + "=" * 65)
    print("HILLSTROM EMAIL MARKETING DATASET")
    print("=" * 65)

    if not HILLSTROM_FILE.exists():
        print(f"File not found: {HILLSTROM_FILE}")
        return

    df = pd.read_csv(HILLSTROM_FILE)

    print(f"\nRows: {df.shape[0]:,}")
    print(f"Columns: {df.shape[1]}")

    print("\nColumn names:")
    print(df.columns.tolist())

    print("\nData types:")
    print(df.dtypes)

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nDuplicate rows:", df.duplicated().sum())

    print("\nFirst 5 records:")
    print(df.head().to_string(index=False))

    if "segment" in df.columns:
        print("\nCampaign group counts:")
        print(df["segment"].value_counts(dropna=False))

    if "conversion" in df.columns:
        print("\nConversion summary:")
        print(df["conversion"].value_counts(dropna=False))

    if "spend" in df.columns:
        print("\nSpend summary:")
        print(df["spend"].describe())


def inspect_online_retail():
    print("\n" + "=" * 65)
    print("ONLINE RETAIL II DATASET")
    print("=" * 65)

    if not RETAIL_FILE.exists():
        print(f"File not found: {RETAIL_FILE}")
        return

    # Read workbook in streaming mode to avoid loading
    # the entire 1-million-plus-row workbook into memory.
    workbook = load_workbook(
        RETAIL_FILE,
        read_only=True,
        data_only=True
    )

    print("\nWorksheets:", workbook.sheetnames)

    for sheet_name in workbook.sheetnames:
        sheet = workbook[sheet_name]

        print(f"\n--- Worksheet: {sheet_name} ---")
        print("Reported dimensions:", sheet.max_row, "rows x",
              sheet.max_column, "columns")

        rows = sheet.iter_rows(values_only=True)

        headers = next(rows, None)

        if headers is None:
            print("Worksheet is empty.")
            continue

        print("\nColumn names:")
        print(list(headers))

        print("\nFirst 5 data records:")
        for _, row in zip(range(5), rows):
            print(row)

    workbook.close()


if __name__ == "__main__":
    inspect_hillstrom()
    inspect_online_retail()
