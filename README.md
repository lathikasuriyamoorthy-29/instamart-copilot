# Instamart Customer Intelligence & Support Copilot

## Overview
This project builds a Customer Intelligence & Support Copilot for synthetic Instamart data, in four connected parts. Part A cleans messy order, customer and product exports and stores them in a SQLite database. Part B segments customers with K-means and predicts which customers will lapse. Part C triages support tickets with an offline mock language model, with privacy guardrails, retries and a token budget. Part D answers policy questions with retrieval and runs a refund agent that uses tools. All data is synthetic, all amounts are in INR, and no paid service or API key is needed.

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
Raw table shapes: orders 13,084 x 9, customers 1,200 x 5, products 40 x 4, tickets 164 x 5.
Missing values: orders has 53 blank order_amount (0.41%) and 2,541 blank delivery_minutes (19.42%); customers has 61 blank age (5.08%) and 30 blank gender (2.50%); products and tickets have none.

| Check | Result |
|---|---|
| Raw order rows | 13,084 |
| Exact duplicate rows | 119 |
| Rows with status pending | 656 |
| Distinct order_id | 12,309 |
| order_amount that cannot be converted to a number | 120 |
| Negative delivery_minutes | 106 |
| Blank delivery_minutes | 2,541 |
| Blank age | 61 |
| Age outside 10-100 | 6 |
| Distinct raw city spellings | 24 |
| Distinct non-blank gender spellings | 10 |
| Blank gender | 30 |

### Cleaning rules
**Customers**
- City: stripped spaces and applied title case, so 24 spellings became 6 cities.
- Gender: mapped all spellings to male, female or unknown. Result: female 610, male 560, unknown 30 (the blanks).
- Age: set 6 ages outside 10-100 to missing, then filled all 67 missing ages (61 blank + 6 impossible) with the median age of the customer's city. Ages now range from 18 to 60.
- signup_date: converted to a datetime.

**Orders** (rules applied in this order)
1. order_ts converted to a datetime.
2. Removed 119 exact duplicate rows (13,084 to 12,965).
3. Kept only the latest record per order_id, which removed 656 earlier pending records (12,965 to 12,309).
4. order_amount: converted to numeric, then filled 112 unparseable or missing amounts with round(price_inr * quantity * (1 - discount_pct / 100), 2).
5. delivery_minutes: set 105 negative values to missing, then filled 774 missing values of completed orders with the median delivery time of the order's product category (completed orders only). The 1,199 cancelled orders were left missing on purpose, because a cancelled order was never delivered.

**Result after cleaning:** 12,309 orders (11,110 completed, 1,199 cancelled), no pending rows, total completed revenue INR 2,998,668.80.

**Storage and validation:** clean tables are saved to data/clean/ as CSV files and to data/instamart.db (tables customers, products, orders). Part_A/validate.py checks the reference values and prints ALL CHECKPOINTS PASSED.

### SQL results
All queries are in sql/queries.sql and are run with Part_A/run_queries.py. Only completed orders count as revenue.

| Query | Technique | Result |
|---|---|---|
| Q1 | WHERE then HAVING | 30 customers with total spend above INR 8,000 (orders above INR 500 only) |
| Q2 | 3-table join | 30 rows (6 cities x 5 categories); revenues add up to INR 2,998,668.80; largest cell is Bengaluru / Electronics at INR 233,909.45 |
| Q3 | Scalar subquery | 11 products priced above the average catalogue price |
| Q4 | LEFT JOIN | 25 customers with no completed order |
| Q5 | CTE | 550 customers whose total spend is above the average customer total |
| Q6 | Window functions | 1,175 customers ranked within their city; top customer in Bengaluru is C0082 (INR 12,856.20) |
| Q7 | Running total | 183 daily rows; the final cumulative revenue is INR 2,998,668.80 |

RANK and ROW_NUMBER give different values for 8 rows. They differ when customers tie on total spend: RANK gives tied customers the same rank, while ROW_NUMBER gives each one a unique number.

The pandas cross-check (Part_A/analysis.py) rebuilds Q2 with a pivot table; its grand total equals the SQL total of INR 2,998,668.80.

### Exploratory analysis
This data covers 11,110 completed Instamart orders across six cities and five product categories, with all amounts in INR. Total revenue is INR 2,998,668.80, led by Hyderabad and Bengaluru.

Order amounts are strongly right-skewed. The mean is 269.91 but the median is only 135.00, and the skewness is 4.61, so a small number of very large orders pull the average up. Most orders are small: Q1 is 64.80 and Q3 is 297.00.

By the 1.5×IQR rule, the upper fence is 645.30 and 897 orders lie above it. The lower fence is negative (−283.50), so there are no low outliers. Since impossible values were fixed during cleaning, these look like genuine large orders rather than errors.

The strongest relationship is between order_amount and discount_pct (−0.095), while order_amount and quantity is similar in size (+0.087). Both are very weak, and a correlation this small does not establish cause and effect.

## Part B
## Part B

### Task 1: RFM table
RFM was built from completed orders only, measured at the snapshot 2025-10-01 00:00:00. Recency is the whole days since a customer's last completed order, frequency is the number of distinct completed orders, and monetary is the sum of completed order_amount. 1,175 customers have at least one completed order (the other 25 have none). Recency has a median of 19 days (range 0 to 180), frequency has a mean of 9.46 (maximum 33), and monetary has a mean of INR 2,552.06 (maximum INR 14,766.55).

### Task 2: K-means selection
The three RFM columns were standardised with StandardScaler, so that monetary (up to INR 14,766) does not dominate recency (up to 180 days). K-means (n_init=10, random_state=42) was run for K = 2 to 8.

| K | Inertia | Silhouette |
|---|---|---|
| 2 | 1820.8 | 0.465 |
| 3 | 1268.2 | 0.445 |
| 4 | 743.1 | 0.526 |
| 5 | 617.4 | 0.480 |
| 6 | 503.2 | 0.482 |
| 7 | 408.5 | 0.416 |
| 8 | 366.5 | 0.412 |

**Chosen K = 4.** Inertia falls sharply up to K=4 (1268.2 to 743.1) and much more slowly after it (743.1 to 617.4), so the elbow is at K=4. Among K = 3, 4 and 5, K=4 also has the highest silhouette score (0.526). Four segments is also a number the business can design campaigns around.

Inertia is the total squared distance from each customer to the centre of their cluster. Adding more clusters always places centres closer to the customers, so inertia keeps falling as K grows, and K=8 will always score lower than K=2 (at the extreme, one cluster per customer gives an inertia of 0). Because of this, inertia alone would always favour a larger K, so K must also be judged with the elbow in the curve, the silhouette score, and whether the segments are useful to the business.

### Task 3: Segment profile, names and PCA
K = 4 segments. Cluster numbers are only labels and carry no order. Profile (mean values per segment):

| Segment | Name | Customers | Mean recency (days) | Mean frequency | Mean monetary (INR) |
|---|---|---|---|---|---|
| 0 | Occasional shoppers | 436 | 28.86 | 3.77 | 1,057.62 |
| 1 | Premium big spenders | 138 | 24.12 | 6.77 | 7,129.70 |
| 2 | Loyal frequent buyers | 419 | 8.77 | 19.40 | 3,410.02 |
| 3 | Drifting customers | 182 | 109.04 | 2.20 | 685.99 |

Recommended action per segment:
- **Occasional shoppers:** offer bundle deals to raise their basket size.
- **Premium big spenders:** offer early access to new electronics and a loyalty reward.
- **Loyal frequent buyers:** offer a free-delivery pass to protect their ordering habit.
- **Drifting customers:** send a win-back reminder with a time-limited discount.

The Premium segment is the one with the highest mean monetary value (segment 1), and `is_premium` is true only for its 138 customers.

PCA with 2 components on the scaled RFM matrix gives explained_variance_ratio_ of 0.634 and 0.218, a total of 0.852. So two dimensions keep about 85% of the variance, and a 2D picture of the segments is a fair summary of the three RFM columns.

The segment table is saved in `outputs/customer_segments.csv` (1,175 rows, 7 columns: customer_id, recency, frequency, monetary, segment_id, segment_name, is_premium).

### Task 4: Lapse problem
**Cohort:** every customer with at least one completed order before the cutoff 2025-09-01 (1,156 customers).

**Label:** `lapsed` is 1 if the customer has no completed order from 2025-09-01 up to and including 2025-09-30, otherwise 0. 443 customers lapsed (38.32%) and 713 did not, so lapsed is the minority class.

**Features (11), built only from orders dated before 2025-09-01:**
- recency: whole days from the last completed order to the cutoff
- frequency: number of distinct completed orders
- monetary: sum of completed order_amount
- avg_discount_pct: mean discount_pct of completed orders
- cancellation_rate: cancelled orders divided by all orders with a final status, before the cutoff
- avg_delivery_minutes: mean delivery time of completed orders
- favourite_category: most frequent product category of completed orders
- city, age, gender: from the clean customer table
- tenure_days: days from signup_date to the cutoff

No feature column has a missing value (checked in code: 0 missing).

**No leakage:** no order dated on or after 2025-09-01 is used to build any feature. The code filters the history to `order_ts < 2025-09-01` and asserts that its latest order is before the cutoff. The segment labels from Task 3 were **not** used as features, because they were built from the whole order history, which includes the September window that defines the label.

**Split:** `train_test_split(test_size=0.2, stratify=y, random_state=42)` gives 924 training rows and 232 test rows, with the same share of lapsed customers in both.

**Baseline:** always predicting "not lapsed" on the test set gives accuracy 0.616 and an F1 of 0.0 for the lapsed class. Accuracy looks acceptable, but this model never finds a single lapsed customer, so the lapsed class is judged with precision, recall and F1 from here on.

## Part C
(coming later)

## Part D
(coming later)

## Limitations and Next Steps
(coming later)