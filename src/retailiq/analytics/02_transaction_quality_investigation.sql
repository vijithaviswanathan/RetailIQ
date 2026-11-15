
-- RetailIQ: Transaction Quality Investigation
-- Dataset: Online Retail II
-- Purpose: Investigate transaction records before cleaning

-- 1. Customer ID completeness by worksheet
SELECT
    source_sheet,
    COUNT(*) AS total_rows,
    COUNT(*) FILTER (WHERE customer_id IS NULL) AS missing_customer_ids,
    COUNT(DISTINCT customer_id) AS identified_customers,
    ROUND(
        100.0 * COUNT(*) FILTER (WHERE customer_id IS NULL) / COUNT(*),
        2
    ) AS missing_customer_percentage
FROM retail.transactions
GROUP BY source_sheet
ORDER BY source_sheet;

-- 2. Cancellation and quantity combinations
SELECT
    CASE WHEN invoice LIKE 'C%' THEN 'Cancellation invoice'
         ELSE 'Regular invoice' END AS invoice_type,
    CASE WHEN quantity < 0 THEN 'Negative quantity'
         WHEN quantity = 0 THEN 'Zero quantity'
         ELSE 'Positive quantity' END AS quantity_type,
    COUNT(*) AS transaction_rows,
    COUNT(DISTINCT invoice) AS invoice_count,
    ROUND(SUM(quantity * price), 2) AS raw_line_value
FROM retail.transactions
GROUP BY 1, 2
ORDER BY 1, 2;

-- 3. Negative quantities without cancellation prefix
SELECT invoice, stock_code, description, quantity,
       invoice_date, price, customer_id, country
FROM retail.transactions
WHERE quantity < 0 AND invoice NOT LIKE 'C%'
ORDER BY invoice_date
LIMIT 25;

-- 4. Cancellation invoices with positive quantities
SELECT invoice, stock_code, description, quantity,
       invoice_date, price, customer_id, country
FROM retail.transactions
WHERE invoice LIKE 'C%' AND quantity > 0
ORDER BY invoice_date
LIMIT 25;

-- 5. Zero and negative prices
SELECT price,
       COUNT(*) AS transaction_rows,
       COUNT(DISTINCT stock_code) AS product_count
FROM retail.transactions
WHERE price <= 0
GROUP BY price
ORDER BY price;

-- 6. Exact duplicate transaction groups
SELECT invoice, stock_code, description, quantity,
       invoice_date, price, customer_id, country,
       source_sheet, COUNT(*) AS occurrences
FROM retail.transactions
GROUP BY invoice, stock_code, description, quantity,
         invoice_date, price, customer_id, country, source_sheet
HAVING COUNT(*) > 1
ORDER BY occurrences DESC
LIMIT 20;

-- 7. Repeated invoice and product combinations
SELECT invoice, stock_code,
       COUNT(*) AS line_count,
       SUM(quantity) AS total_quantity
FROM retail.transactions
GROUP BY invoice, stock_code
HAVING COUNT(*) > 1
ORDER BY line_count DESC
LIMIT 20;

-- 8. Date coverage by worksheet
SELECT source_sheet,
       MIN(invoice_date) AS first_transaction,
       MAX(invoice_date) AS last_transaction,
       COUNT(*) AS total_rows
FROM retail.transactions
GROUP BY source_sheet
ORDER BY source_sheet;
