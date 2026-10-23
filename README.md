# Instamart Customer Intelligence & Support Copilot

## Overview
(write 2-3 sentences about what the project does)

## Reproduce
1. python -m venv .venv
2. .venv\Scripts\Activate.ps1
3. pip install -r requirements.txt
4. python generate_data.py
5. python Part_A/clean.py
6. python Part_A/validate.py
7. python Part_A/run_queries.py
8. python Part_A/analysis.py

## Part A
### Raw data audit
(your audit numbers)

### Cleaning rules
(each rule with the number of rows or values changed)

### SQL results
(a short list of the 7 query results)

### Exploratory analysis
This data covers 11,110 completed Instamart orders across six cities and five product categories, with all amounts in INR. Total revenue is INR 2,998,668.80, led by Hyderabad and Bengaluru.

Order amounts are strongly right-skewed. The mean is 269.91 but the median is only 135.00, and the skewness is 4.61, so a small number of very large orders pull the average up. Most orders are small: Q1 is 64.80 and Q3 is 297.00.

By the 1.5×IQR rule, the upper fence is 645.30 and 897 orders lie above it. The lower fence is negative (−283.50), so there are no low outliers. Since impossible values were fixed during cleaning, these look like genuine large orders rather than errors.

The strongest relationship is between order_amount and discount_pct (−0.095), while order_amount and quantity is similar in size (+0.087). Both are very weak, and a correlation this small does not establish cause and effect.

## Part B
(coming later)

## Part C
(coming later)

## Part D
(coming later)

## Limitations and Next Steps
(coming later)