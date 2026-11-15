
from pathlib import Path

import pandas as pd
from openpyxl import load_workbook
from sqlalchemy import create_engine, text
from sqlalchemy.types import (
    DateTime,
    Float,
    Integer,
    Numeric,
    Text,
)

# --------------------------------------------------
# 1. Project configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[3]

HILLSTROM_FILE = PROJECT_ROOT / "data" / "raw" / "hillstrom.csv"
RETAIL_FILE = PROJECT_ROOT / "data" / "raw" / "online_retail_II.xlsx"

from retailiq.config import DATABASE_URL

CHUNK_SIZE = 10_000

engine = create_engine(DATABASE_URL)

# --------------------------------------------------
# 2. Hillstrom table configuration
# --------------------------------------------------

HILLSTROM_DTYPES = {
    "recency": Integer(),
    "history_segment": Text(),
    "history": Float(),
    "mens": Integer(),
    "womens": Integer(),
    "zip_code": Text(),
    "newbie": Integer(),
    "channel": Text(),
    "segment": Text(),
    "visit": Integer(),
    "conversion": Integer(),
    "spend": Float(),
}


def load_hillstrom():
    print("\nLoading Hillstrom dataset...")

    if not HILLSTROM_FILE.exists():
        raise FileNotFoundError(HILLSTROM_FILE)

    total_rows = 0
    first_chunk = True

    with engine.begin() as connection:
        for chunk in pd.read_csv(
            HILLSTROM_FILE,
            chunksize=CHUNK_SIZE,
        ):
            chunk.to_sql(
                name="campaign_data",
                con=connection,
                schema="hillstrom",
                if_exists="replace" if first_chunk else "append",
                index=False,
                dtype=HILLSTROM_DTYPES,
                chunksize=1000,
            )

            total_rows += len(chunk)
            first_chunk = False

            print(f"Hillstrom rows loaded: {total_rows:,}")

    print(f"Hillstrom loading complete: {total_rows:,} rows")


# --------------------------------------------------
# 3. Online Retail II configuration
# --------------------------------------------------

RETAIL_DTYPES = {
    "invoice": Text(),
    "stock_code": Text(),
    "description": Text(),
    "quantity": Integer(),
    "invoice_date": DateTime(),
    "price": Numeric(12, 4),
    "customer_id": Text(),
    "country": Text(),
    "source_sheet": Text(),
}


def normalize_identifier(value):
    """Convert identifiers to strings while preserving missing values."""

    if pd.isna(value):
        return None

    if isinstance(value, float) and value.is_integer():
        return str(int(value))

    return str(value).strip()


def load_retail_sheet(workbook, sheet_name, first_sheet):
    worksheet = workbook[sheet_name]
    rows = worksheet.iter_rows(values_only=True)

    headers = next(rows)

    # Normalize source column names for SQL.
    column_names = [
    str(column).strip().lower().replace(" ", "_")
    for column in headers
    ]
    # Standardize Excel column names
    column_mapping = {
        "stockcode": "stock_code",
        "invoicedate": "invoice_date",
        }
    column_names = [
        column_mapping.get(column, column)
        for column in column_names
        ]

    total_rows = 0
    batch = []

    def insert_batch(records):
        nonlocal total_rows

        if not records:
            return

        dataframe = pd.DataFrame(records, columns=column_names)

        # Normalize identifier columns.
        for column in ["invoice", "stock_code", "customer_id"]:
            dataframe[column] = dataframe[column].map(
                normalize_identifier
            )

        dataframe["invoice_date"] = pd.to_datetime(
            dataframe["invoice_date"],
            errors="coerce",
        )

        dataframe["quantity"] = pd.to_numeric(
            dataframe["quantity"],
            errors="coerce",
        ).astype("Int64")

        dataframe["price"] = pd.to_numeric(
            dataframe["price"],
            errors="coerce",
        )

        dataframe["source_sheet"] = sheet_name

        dataframe.to_sql(
            name="transactions",
            con=connection,
            schema="retail",
            if_exists="replace" if first_sheet and total_rows == 0 else "append",
            index=False,
            dtype=RETAIL_DTYPES,
            chunksize=1000,
        )

        total_rows += len(dataframe)

        print(
            f"{sheet_name}: {total_rows:,} rows loaded"
        )

    # Use a single transaction for this worksheet.
    with engine.begin() as connection:
        for row in rows:
            if all(value is None for value in row):
                continue

            batch.append(row)

            if len(batch) >= CHUNK_SIZE:
                insert_batch(batch)
                batch = []

        if batch:
            insert_batch(batch)

    print(f"{sheet_name} complete: {total_rows:,} rows")


def load_online_retail():
    print("\nLoading Online Retail II dataset...")

    if not RETAIL_FILE.exists():
        raise FileNotFoundError(RETAIL_FILE)

    workbook = load_workbook(
        RETAIL_FILE,
        read_only=True,
        data_only=True,
    )

    try:
        expected_sheets = [
            "Year 2009-2010",
            "Year 2010-2011",
        ]

        for index, sheet_name in enumerate(expected_sheets):
            if sheet_name not in workbook.sheetnames:
                raise ValueError(
                    f"Missing worksheet: {sheet_name}"
                )

            load_retail_sheet(
                workbook,
                sheet_name,
                first_sheet=(index == 0),
            )

    finally:
        workbook.close()

    print("Online Retail II loading complete.")


# --------------------------------------------------
# 4. Verify database row counts
# --------------------------------------------------

def verify_tables():
    print("\nVerifying PostgreSQL tables...")

    queries = {
        "Hillstrom": """
            SELECT COUNT(*)
            FROM hillstrom.campaign_data;
        """,
        "Online Retail II": """
            SELECT COUNT(*)
            FROM retail.transactions;
        """,
    }

    with engine.connect() as connection:
        for name, query in queries.items():
            count = connection.execute(
                text(query)
            ).scalar_one()

            print(f"{name}: {count:,} rows")


# --------------------------------------------------
# 5. Main execution
# --------------------------------------------------

if __name__ == "__main__":
    try:
        load_hillstrom()
        load_online_retail()
        verify_tables()

        print("\nAll datasets loaded successfully!")

    except Exception as error:
        print("\nDataset loading failed:")
        print(error)
        raise

    finally:
        engine.dispose()
