
-- ============================================================
-- RetailIQ: Online Retail II - Initial Data Audit
-- ============================================================

-- 1. Overall dataset statistics
SELECT
    COUNT(*) AS total_rows,
    COUNT(DISTINCT invoice) AS unique_invoices,
    COUNT(DISTINCT customer_id) AS unique_customers,
    COUNT(DISTINCT stock_code) AS unique_products,
    MIN(invoice_date) AS first_transaction,
    MAX(invoice_date) AS last_transaction
FROM retail.transactions;


-- 2. Missing value analysis
SELECT
    COUNT(*) FILTER (WHERE invoice IS NULL) AS missing_invoices,
    COUNT(*) FILTER (WHERE stock_code IS NULL) AS missing_stock_codes,
    COUNT(*) FILTER (WHERE description IS NULL) AS missing_descriptions,
    COUNT(*) FILTER (WHERE invoice_date IS NULL) AS missing_dates,
    COUNT(*) FILTER (WHERE customer_id IS NULL) AS missing_customer_ids,
    COUNT(*) FILTER (WHERE country IS NULL) AS missing_countries,
    COUNT(*) FILTER (WHERE price IS NULL) AS missing_prices,
    COUNT(*) FILTER (WHERE quantity IS NULL) AS missing_quantities
FROM retail.transactions;


-- 3. Invoice cancellation analysis
SELECT
    CASE
        WHEN invoice LIKE 'C%' THEN 'Cancelled invoice'
        ELSE 'Regular invoice'
    END AS invoice_status,
    COUNT(*) AS transaction_rows,
    COUNT(DISTINCT invoice) AS invoice_count
FROM retail.transactions
GROUP BY 1
ORDER BY 1;


-- 4. Quantity and price quality checks
SELECT
    COUNT(*) FILTER (WHERE quantity < 0) AS negative_quantity_rows,
    COUNT(*) FILTER (WHERE quantity = 0) AS zero_quantity_rows,
    COUNT(*) FILTER (WHERE price < 0) AS negative_price_rows,
    COUNT(*) FILTER (WHERE price = 0) AS zero_price_rows,
    COUNT(*) FILTER (WHERE quantity > 0 AND price > 0) AS positive_sale_rows
FROM retail.transactions;


-- 5. Monthly transaction overview
SELECT
    DATE_TRUNC('month', invoice_date)::date AS sales_month,
    COUNT(*) AS transaction_rows,
    COUNT(DISTINCT invoice) AS invoice_count,
    COUNT(DISTINCT customer_id) AS identified_customers,
    ROUND(
        SUM(quantity * price),
        2
    ) AS raw_line_value
FROM retail.transactions
GROUP BY 1
ORDER BY 1;


-- 6. Country-level transaction overview
SELECT
    country,
    COUNT(*) AS transaction_rows,
    COUNT(DISTINCT invoice) AS invoice_count,
    COUNT(DISTINCT customer_id) AS identified_customers,
    ROUND(SUM(quantity * price), 2) AS raw_line_value
FROM retail.transactions
GROUP BY country
ORDER BY raw_line_value DESC
LIMIT 15;


-- 7. Top 15 products by transaction quantity
SELECT
    stock_code,
    MAX(description) AS product_description,
    SUM(quantity) AS total_quantity,
    COUNT(DISTINCT invoice) AS invoice_count
FROM retail.transactions
WHERE quantity > 0
  AND stock_code IS NOT NULL
GROUP BY stock_code
ORDER BY total_quantity DESC
LIMIT 15;


-- 8. Duplicate row investigation
SELECT
    COUNT(*) AS duplicated_row_groups,
    SUM(row_count - 1) AS excess_duplicate_rows
FROM (
    SELECT COUNT(*) AS row_count
    FROM retail.transactions
    GROUP BY
        invoice,
        stock_code,
        description,
        quantity,
        invoice_date,
        price,
        customer_id,
        country,
        source_sheet
    HAVING COUNT(*) > 1
) AS duplicates;
