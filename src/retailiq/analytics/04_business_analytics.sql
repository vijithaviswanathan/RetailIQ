
-- ============================================================
-- RETAILIQ: PHASE 1 - BUSINESS ANALYTICS
-- Source: retail.transaction_classified
-- Revenue basis: candidate merchandise sales
-- Postage excluded from merchandise performance.
-- Raw transaction data remains unchanged.
-- ============================================================


-- 1. MONTHLY SALES PERFORMANCE
-- Revenue, orders, units, and identified customers by month.

CREATE OR REPLACE VIEW retail.monthly_sales AS
SELECT
    DATE_TRUNC('month', invoice_date)::date AS month,
    COUNT(DISTINCT invoice) AS total_orders,
    SUM(quantity) AS units_sold,
    ROUND(SUM(raw_line_value)::numeric, 2) AS merchandise_revenue,
    COUNT(DISTINCT customer_id) AS identified_customers,
    COUNT(*) FILTER (
        WHERE customer_id IS NULL
    ) AS unidentified_customer_rows
FROM retail.transaction_classified
WHERE included_in_merchandise_sales = TRUE
  AND UPPER(TRIM(stock_code)) NOT IN ('DOT', 'POST')
GROUP BY DATE_TRUNC('month', invoice_date)::date;


-- 2. PRODUCT PERFORMANCE
-- Postage excluded. Unusual single-invoice volume is flagged,
-- not automatically deleted.

CREATE OR REPLACE VIEW retail.product_performance AS
WITH product_metrics AS (
    SELECT
        stock_code,
        description,
        COUNT(*) AS transaction_rows,
        COUNT(DISTINCT invoice) AS invoice_count,
        SUM(quantity) AS units_sold,
        ROUND(SUM(raw_line_value)::numeric, 2) AS product_revenue,
        ROUND(AVG(price)::numeric, 2) AS average_unit_price
    FROM retail.transaction_classified
    WHERE included_in_merchandise_sales = TRUE
      AND UPPER(TRIM(stock_code)) NOT IN ('DOT', 'POST')
    GROUP BY stock_code, description
)
SELECT
    *,
    CASE
        WHEN invoice_count = 1 AND units_sold > 10000
            THEN TRUE
        ELSE FALSE
    END AS single_invoice_volume_outlier
FROM product_metrics;


-- 3. COUNTRY PERFORMANCE
-- Revenue and order distribution by country.

CREATE OR REPLACE VIEW retail.country_performance AS
SELECT
    COALESCE(country, 'Unknown') AS country,
    COUNT(DISTINCT invoice) AS total_orders,
    COUNT(DISTINCT customer_id) AS identified_customers,
    SUM(quantity) AS units_sold,
    ROUND(SUM(raw_line_value)::numeric, 2) AS merchandise_revenue
FROM retail.transaction_classified
WHERE included_in_merchandise_sales = TRUE
  AND UPPER(TRIM(stock_code)) NOT IN ('DOT', 'POST')
GROUP BY COALESCE(country, 'Unknown');


-- 4. MONTHLY CUSTOMER ACTIVITY
-- Active, new, and returning identified customers.

CREATE OR REPLACE VIEW retail.monthly_customer_activity AS
WITH customer_first_purchase AS (
    SELECT
        customer_id,
        DATE_TRUNC('month', MIN(invoice_date))::date
            AS first_purchase_month
    FROM retail.transaction_classified
    WHERE included_in_merchandise_sales = TRUE
      AND customer_id IS NOT NULL
      AND UPPER(TRIM(stock_code)) NOT IN ('DOT', 'POST')
    GROUP BY customer_id
),
monthly_activity AS (
    SELECT
        DATE_TRUNC('month', t.invoice_date)::date AS month,
        COUNT(DISTINCT t.customer_id) AS active_customers,
        COUNT(DISTINCT t.customer_id) FILTER (
            WHERE f.first_purchase_month =
                  DATE_TRUNC('month', t.invoice_date)::date
        ) AS new_customers
    FROM retail.transaction_classified t
    LEFT JOIN customer_first_purchase f
        ON t.customer_id = f.customer_id
    WHERE t.included_in_merchandise_sales = TRUE
      AND t.customer_id IS NOT NULL
      AND UPPER(TRIM(t.stock_code)) NOT IN ('DOT', 'POST')
    GROUP BY DATE_TRUNC('month', t.invoice_date)::date
)
SELECT
    month,
    active_customers,
    new_customers,
    active_customers - new_customers AS returning_customers
FROM monthly_activity;


-- 5. EXECUTIVE BUSINESS KPIs
-- Single-row summary for the dashboard.

CREATE OR REPLACE VIEW retail.executive_kpis AS
SELECT
    ROUND(SUM(raw_line_value)::numeric, 2)
        AS total_merchandise_revenue,
    COUNT(DISTINCT invoice)
        AS total_orders,
    COUNT(DISTINCT customer_id)
        AS identified_customers,
    COUNT(DISTINCT stock_code)
        AS distinct_products,
    SUM(quantity)
        AS total_units_sold,
    ROUND(
        SUM(raw_line_value)::numeric /
        NULLIF(COUNT(DISTINCT invoice), 0),
        2
    ) AS average_order_value,
    ROUND(
        SUM(raw_line_value) FILTER (
            WHERE customer_id IS NULL
        )::numeric,
        2
    ) AS revenue_without_customer_id
FROM retail.transaction_classified
WHERE included_in_merchandise_sales = TRUE
  AND UPPER(TRIM(stock_code)) NOT IN ('DOT', 'POST');
