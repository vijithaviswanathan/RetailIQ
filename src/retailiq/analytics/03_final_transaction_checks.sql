-- ============================================================
-- 4. Transaction Classification View
-- ============================================================

CREATE OR REPLACE VIEW retail.transaction_classified AS
SELECT
    *,
    quantity * price AS raw_line_value,

    CASE
        -- Manual accounting adjustments
        WHEN UPPER(TRIM(stock_code)) = 'M'
          OR LOWER(TRIM(description)) = 'manual'
            THEN 'manual_adjustment'

        -- Bad debt adjustments
        WHEN UPPER(TRIM(stock_code)) = 'B'
          OR LOWER(TRIM(description)) = 'adjust bad debt'
            THEN 'bad_debt_adjustment'

        -- Cancellation-prefixed invoices
        WHEN invoice LIKE 'C%'
            THEN 'cancellation'

        -- Zero-price transactions
        WHEN price = 0
            THEN 'zero_price'

        -- Negative quantity without cancellation prefix
        WHEN quantity < 0
            THEN 'negative_quantity_adjustment'

        -- Other negative prices
        WHEN price < 0
            THEN 'negative_price_adjustment'

        -- Standard positive merchandise transactions
        WHEN quantity > 0 AND price > 0
            THEN 'candidate_sale'

        ELSE 'other'
    END AS transaction_type,

    CASE
        WHEN UPPER(TRIM(stock_code)) = 'M'
          OR LOWER(TRIM(description)) = 'manual'
            THEN FALSE
        WHEN UPPER(TRIM(stock_code)) = 'B'
          OR LOWER(TRIM(description)) = 'adjust bad debt'
            THEN FALSE
        WHEN invoice LIKE 'C%'
            THEN FALSE
        WHEN price <= 0 OR quantity <= 0
            THEN FALSE
        ELSE TRUE
    END AS included_in_merchandise_sales

FROM retail.transactions;