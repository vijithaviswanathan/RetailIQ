# RetailIQ — Data Quality Report

This report profiles the original downloaded datasets.
No source data is modified by this script.


## 1. Hillstrom Email Marketing

- Rows: 64,000
- Columns: 12
- Exact duplicate rows: 6,562
- Missing values: 0

### Missing values by column
recency            0
history_segment    0
history            0
mens               0
womens             0
zip_code           0
newbie             0
channel            0
segment            0
visit              0
conversion         0
spend              0

### Data types
recency              int64
history_segment        str
history            float64
mens                 int64
womens               int64
zip_code               str
newbie               int64
channel                str
segment                str
visit                int64
conversion           int64
spend              float64

### Campaign group distribution
segment
Womens E-Mail    21387
Mens E-Mail      21307
No E-Mail        21306

### Outcome summary
              visit    conversion         spend
count  64000.000000  64000.000000  64000.000000
mean       0.146781      0.009031      1.050908
std        0.353890      0.094604     15.036448
min        0.000000      0.000000      0.000000
25%        0.000000      0.000000      0.000000
50%        0.000000      0.000000      0.000000
75%        0.000000      0.000000      0.000000
max        1.000000      1.000000    499.000000

### Conversion rate by campaign group
segment
Mens E-Mail      1.253
No E-Mail        0.573
Womens E-Mail    0.884 (%)

### Visit rate by campaign group
segment
Mens E-Mail      18.276
No E-Mail        10.617
Womens E-Mail    15.140 (%)

### Mean spend by campaign group
segment
Mens E-Mail      1.423
No E-Mail        0.653
Womens E-Mail    1.077

### Spend consistency checks
- Customers with positive spend: 578
- Customers with conversion = 1: 578
- Conversions with zero spend: 0
- Positive spend with conversion = 0: 0
- Visits with conversion = 0: 8,816

## 2. Online Retail II — Year 2009-2010

- Rows: 525,461
- Columns: 8
- Exact duplicate rows: 6,865
- Total missing cells: 110,855

### Missing values by column
Invoice             0
StockCode           0
Description      2928
Quantity            0
InvoiceDate         0
Price               0
Customer ID    107927
Country             0

### Data types
Invoice                object
StockCode              object
Description            object
Quantity                int64
InvoiceDate    datetime64[us]
Price                 float64
Customer ID           float64
Country                   str

### Transaction checks
- Canceled invoice rows: 10,206
- Negative quantity rows: 12,326
- Zero quantity rows: 0
- Negative price rows: 3
- Zero price rows: 3,687
- Rows without Customer ID: 107,927

### Date coverage
- Earliest invoice: 2009-12-01 07:45:00
- Latest invoice: 2010-12-09 20:01:00

### Quantity and price summary
            Quantity          Price
count  525461.000000  525461.000000
mean       10.337667       4.688834
std       107.424110     146.126914
min     -9600.000000  -53594.360000
25%         1.000000       1.250000
50%         3.000000       2.100000
75%        10.000000       4.210000
max     19152.000000   25111.090000

### Countries
- Unique countries: 40

### Revenue diagnostics
- Gross line value (Quantity × Price): 9,539,484.63
- Positive-quantity rows: 513,135
- Negative-quantity rows: 12,326

## 2. Online Retail II — Year 2010-2011

- Rows: 541,910
- Columns: 8
- Exact duplicate rows: 5,268
- Total missing cells: 136,534

### Missing values by column
Invoice             0
StockCode           0
Description      1454
Quantity            0
InvoiceDate         0
Price               0
Customer ID    135080
Country             0

### Data types
Invoice                object
StockCode              object
Description            object
Quantity                int64
InvoiceDate    datetime64[us]
Price                 float64
Customer ID           float64
Country                   str

### Transaction checks
- Canceled invoice rows: 9,288
- Negative quantity rows: 10,624
- Zero quantity rows: 0
- Negative price rows: 2
- Zero price rows: 2,515
- Rows without Customer ID: 135,080

### Date coverage
- Earliest invoice: 2010-12-01 08:26:00
- Latest invoice: 2011-12-09 12:50:00

### Quantity and price summary
            Quantity          Price
count  541910.000000  541910.000000
mean        9.552234       4.611138
std       218.080957      96.759765
min    -80995.000000  -11062.060000
25%         1.000000       1.250000
50%         3.000000       2.080000
75%        10.000000       4.130000
max     80995.000000   38970.000000

### Countries
- Unique countries: 38

### Revenue diagnostics
- Gross line value (Quantity × Price): 9,747,765.93
- Positive-quantity rows: 531,286
- Negative-quantity rows: 10,624