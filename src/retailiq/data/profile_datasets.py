
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
REPORT_DIR = PROJECT_ROOT / "reports"
REPORT_FILE = REPORT_DIR / "data_quality_report.md"

HILLSTROM_FILE = RAW_DIR / "hillstrom.csv"
RETAIL_FILE = RAW_DIR / "online_retail_II.xlsx"


def section(title):
    return f"\n## {title}\n"


def profile_hillstrom():
    df = pd.read_csv(HILLSTROM_FILE)

    lines = [
        section("1. Hillstrom Email Marketing"),
        f"- Rows: {len(df):,}",
        f"- Columns: {len(df.columns)}",
        f"- Exact duplicate rows: {df.duplicated().sum():,}",
        f"- Missing values: {int(df.isna().sum().sum()):,}",
        "",
        "### Missing values by column",
        df.isna().sum().to_string(),
        "",
        "### Data types",
        df.dtypes.astype(str).to_string(),
        "",
        "### Campaign group distribution",
        df["segment"].value_counts(dropna=False).to_string(),
        "",
        "### Outcome summary",
        df[["visit", "conversion", "spend"]].describe().to_string(),
        "",
        "### Conversion rate by campaign group",
        (df.groupby("segment")["conversion"].mean() * 100)
        .round(3).to_string() + " (%)",
        "",
        "### Visit rate by campaign group",
        (df.groupby("segment")["visit"].mean() * 100)
        .round(3).to_string() + " (%)",
        "",
        "### Mean spend by campaign group",
        df.groupby("segment")["spend"].mean()
        .round(3).to_string(),
        "",
        "### Spend consistency checks",
        f"- Customers with positive spend: {(df['spend'] > 0).sum():,}",
        f"- Customers with conversion = 1: {(df['conversion'] == 1).sum():,}",
        f"- Conversions with zero spend: "
        f"{((df['conversion'] == 1) & (df['spend'] == 0)).sum():,}",
        f"- Positive spend with conversion = 0: "
        f"{((df['spend'] > 0) & (df['conversion'] == 0)).sum():,}",
        f"- Visits with conversion = 0: "
        f"{((df['visit'] == 1) & (df['conversion'] == 0)).sum():,}",
    ]

    return "\n".join(lines)


def profile_retail_sheet(sheet_name):
    # Process one worksheet at a time to limit memory usage.
    df = pd.read_excel(
        RETAIL_FILE,
        sheet_name=sheet_name,
        engine="openpyxl"
    )

    canceled = df["Invoice"].astype(str).str.startswith("C")

    lines = [
        section(f"2. Online Retail II — {sheet_name}"),
        f"- Rows: {len(df):,}",
        f"- Columns: {len(df.columns)}",
        f"- Exact duplicate rows: {df.duplicated().sum():,}",
        f"- Total missing cells: {int(df.isna().sum().sum()):,}",
        "",
        "### Missing values by column",
        df.isna().sum().to_string(),
        "",
        "### Data types",
        df.dtypes.astype(str).to_string(),
        "",
        "### Transaction checks",
        f"- Canceled invoice rows: {canceled.sum():,}",
        f"- Negative quantity rows: {(df['Quantity'] < 0).sum():,}",
        f"- Zero quantity rows: {(df['Quantity'] == 0).sum():,}",
        f"- Negative price rows: {(df['Price'] < 0).sum():,}",
        f"- Zero price rows: {(df['Price'] == 0).sum():,}",
        f"- Rows without Customer ID: {df['Customer ID'].isna().sum():,}",
        "",
        "### Date coverage",
        f"- Earliest invoice: {df['InvoiceDate'].min()}",
        f"- Latest invoice: {df['InvoiceDate'].max()}",
        "",
        "### Quantity and price summary",
        df[["Quantity", "Price"]].describe().to_string(),
        "",
        "### Countries",
        f"- Unique countries: {df['Country'].nunique(dropna=True)}",
        "",
        "### Revenue diagnostics",
        f"- Gross line value (Quantity × Price): "
        f"{(df['Quantity'] * df['Price']).sum():,.2f}",
        f"- Positive-quantity rows: {(df['Quantity'] > 0).sum():,}",
        f"- Negative-quantity rows: {(df['Quantity'] < 0).sum():,}",
    ]

    return "\n".join(lines)


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    report = [
        "# RetailIQ — Data Quality Report",
        "",
        "This report profiles the original downloaded datasets.",
        "No source data is modified by this script.",
        "",
        profile_hillstrom(),
    ]

    for sheet in ["Year 2009-2010", "Year 2010-2011"]:
        report.append(profile_retail_sheet(sheet))

    REPORT_FILE.write_text("\n".join(report), encoding="utf-8")

    print("\nData profiling completed.")
    print(f"Report saved to: {REPORT_FILE}")
    print(f"Report size: {REPORT_FILE.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
