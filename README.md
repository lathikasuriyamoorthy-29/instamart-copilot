# Instamart Customer Intelligence & Support Copilot

## Overview
This project builds a Customer Intelligence & Support Copilot for synthetic Instamart data, in four connected parts. Part A cleans messy order, customer and product exports and stores them in a SQLite database. Part B segments customers with K-means and predicts which customers will lapse. Part C triages support tickets with an offline mock language model, with privacy guardrails, retries and a token budget. Part D answers policy questions with retrieval and runs a refund agent that uses tools. All data is synthetic, all amounts are in INR, and no paid service or API key is needed.

## Reproduce
1. python -m venv .venv
2. .venv\Scripts\Activate.ps1 (on macOS or Linux: source .venv/bin/activate)
3. pip install -r requirements.txt
4. python generate_data.py
5. python Part_A/audit.py
6. python Part_A/clean.py
7. python Part_A/validate.py
8. python Part_A/run_queries.py
9. python Part_A/analysis.py
10. python Part_B/rfm.py
11. python Part_B/kmeans_select.py
12. python Part_B/segments.py
13. python Part_B/lapse_data.py
14. python Part_B/preprocess.py
15. python Part_B/train_models.py
16. python Part_B/balance_experiment.py
17. python Part_B/interpret_and_save.py
18. python Part_C/run_part_c.py
19. python Part_C/compare_and_route.py

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

By the 1.5×IQR rule, the upper fence is 645.30 and 897 orders lie above it. The lower fence is negative (−283.50), so there are no low outliers. These orders were kept in the data and their cause was not investigated.

The strongest relationship is between order_amount and discount_pct (−0.095), while order_amount and quantity is similar in size (+0.087). Both are very weak, and a correlation this small does not establish cause and effect.

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

### Task 5: Preprocessing without leakage
The nominal columns city, gender and favourite_category are one-hot encoded with `OneHotEncoder(drop="first")`, so each variable with k categories gives k-1 columns (5 for city, 2 for gender and 4 for favourite_category). The 8 numeric features are standardised with `StandardScaler`. Both are combined in a `ColumnTransformer` that is fitted on the training rows only and then only applied to the test rows, so no encoder or scaler sees test data. The transformed matrices have 19 columns (train (924, 19), test (232, 19)), and the code asserts that no feature column contains a missing value (0 missing).

The scaler learned a mean frequency of 8.115 from the 924 training rows, while the full-cohort mean is 8.029. They differ because the scaler is fitted on the training rows only; the test rows are kept out of the fitting on purpose, so no test information leaks into preprocessing. After scaling, the training means are about 0 and the test means are small but not exactly 0, which also shows the scaler did not use test data.

### Task 6: Training, tuning and evaluation
Baseline from Task 4: always predicting "not lapsed" gives accuracy 0.616 and a lapsed-class F1 of 0.0.

Each model is a `Pipeline` (preprocessing from Task 5, then the classifier), tuned with `GridSearchCV(cv=5, scoring="f1")` on the 924 training rows only. Because the encoder and scaler sit inside the pipeline, they are re-fitted on the training part of every fold, so no validation rows leak into preprocessing. The test set was used only after tuning was finished.

| Model | Grid | Best parameters | Best cross-validated F1 |
|---|---|---|---|
| Logistic regression (max_iter=1000) | C in {0.01, 0.1, 1, 10, 100} | C = 100 | 0.651 |
| Random forest (n_estimators=100, random_state=42) | max_depth in {3, 5, 8}, min_samples_leaf in {5, 10, 20} | max_depth = 8, min_samples_leaf = 10 | 0.682 |

The best C is the largest value in the grid, so a wider grid might improve logistic regression slightly.

**Test set (232 customers, 89 lapsed), lapsed class as positive:**

| Model | True 0 | False 1 | False 0 | True 1 | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|
| Logistic regression | 111 | 32 | 31 | 58 | 0.644 | 0.652 | 0.648 |
| Random forest | 112 | 31 | 31 | 58 | 0.652 | 0.652 | 0.652 |

(Confusion matrix layout: rows are the true class and columns the predicted class, in the order not lapsed, lapsed. "False 1" is a loyal customer wrongly flagged as lapsing, "False 0" is a lapsing customer that was missed.)

**Selected model: random forest.** It has the higher cross-validated F1 (0.682 against 0.651) and a slightly higher test F1 (0.652 against 0.648), well above the required 0.55. Both models catch the same 58 of the 89 lapsing customers (recall 0.652), but the forest raises one fewer false alarm. Missing a lapsing customer costs all of their future orders, while wrongly targeting a loyal customer only costs a small discount or message, so recall matters more, and the forest does at least as well as logistic regression on both recall and precision. The test difference between the two models is only one customer, so the cross-validated scores on 924 rows are the stronger evidence for the choice.

### Task 7: Class-balance experiment
The lapsed class is the minority (354 of 924 training rows). To test whether this hurts, the lapsed class was over-sampled in the **training set only**, after the split: every original training row was kept, and 216 randomly chosen duplicates of lapsed training rows (drawn with replacement, `random_state=42`) were added, so both classes have 570 rows. The best logistic regression from Task 6 (C = 100) was refitted on this balanced set and compared with the original model on the **original, unchanged test set** (232 customers, 89 lapsed). Over-sampling before the split would have put copies of the same customer into both training and test data, which is leakage.

| Model | Precision | Recall | F1 |
|---|---|---|---|
| Original (unbalanced) | 0.644 | 0.652 | 0.648 |
| Balanced by over-sampling | 0.589 | 0.854 | 0.697 |

Confusion matrices on the test set (rows are the true class, columns the predicted class, in the order not lapsed, lapsed):

| Model | True 0 | False 1 | False 0 | True 1 |
|---|---|---|---|---|
| Original | 111 | 32 | 31 | 58 |
| Balanced | 90 | 53 | 13 | 76 |

**Recall moved most, rising from 0.652 to 0.854**, while precision fell slightly from 0.644 to 0.589 and did not improve. The balanced model sees lapsed customers as just as common as the others, so it is more willing to predict "lapsed". For the retention team this means they would reach many more of the customers who are about to leave (76 of the 89 lapsing test customers instead of 58, and only 13 missed instead of 31), but they would also contact more customers who would have stayed anyway (53 wrongly flagged loyal customers instead of 32, and 129 customers flagged in total instead of 90). That trade is worth making if a retention offer is cheap compared with the value of a lost customer, and costly if offers are expensive. The duplicated rows add no new information, they only change how the model weighs the two classes.

This was an experiment on logistic regression. The model selected in Task 6 (the random forest trained on the original data) is unchanged.

### Task 8: Interpretation, saved model and retention list
**Top 5 features of the selected random forest (feature_importances_):**

| Feature | Importance | Business reading |
|---|---|---|
| frequency | 0.307 | how often a customer orders is the strongest signal of staying |
| monetary | 0.247 | total spend is the second strongest signal |
| recency | 0.115 | how long since the last order |
| cancellation_rate | 0.061 | customers with many cancelled orders may be having a poor experience |
| avg_discount_pct | 0.058 | the average discount may separate deal-driven customers from regular ones |

**Coefficients of three features in the best logistic regression** (numeric features are standardised, so sizes are comparable):

| Feature | Coefficient | Business reading |
|---|---|---|
| frequency | -1.698 | the more often a customer orders, the less likely they are to lapse (a strong effect) |
| recency | +0.040 | a longer gap since the last order slightly raises the chance of lapsing, though the effect is small once frequency is accounted for |
| tenure_days | -0.050 | customers who signed up longer ago are slightly less likely to lapse |

**Saved model:** the selected random forest (preprocessing and classifier in one pipeline) was saved with `joblib.dump` to `outputs/lapse_model.joblib`, reloaded with `joblib.load`, and its predicted labels and probabilities on the test set are exactly equal to those of the original model.

**Retention list:** 40 test-set customers are in the Premium segment. They were ranked by the model's predicted lapse probability, and the top 10 are saved in `outputs/retention_list.csv` (customer_id, lapse_probability, actually_lapsed). The highest probability is 0.5025, and none of the 10 customers actually lapsed (0 of 10). The Premium flag comes from the whole-window RFM segmentation of Task 3, which includes the September window that defines the lapse label, so it is a post-hoc whole-window label and this list is illustrative, not a leakage-free evaluation. The model's own features use only orders before 2025-09-01.

### Part B: leakage controls
- No encoder, scaler or resampler saw the test rows. The encoder and scaler are fitted only on training rows inside the pipeline (and re-fitted on the training part of every cross-validation fold), and the over-sampling in Task 7 was applied to the training set only, after the split.
- No data from 2025-09-01 onwards entered the features: they use orders before the cutoff only, and the code asserts it.
- The segment labels from Task 3 were not used as features.

## Part C

### Guardrails and privacy (Tasks 1 and 2)
Tickets are processed in file order. A ticket whose text is empty or only whitespace gets status `skipped_empty` and is never sent to the model (4 tickets). A ticket whose estimated tokens exceed `MAX_INPUT_TOKENS = 400` gets status `flagged_long` and is never sent (4 tickets, each estimated at 734 tokens).

Before a ticket goes into a prompt, personal data is replaced in this order: emails with `[EMAIL]`, phone numbers with `[PHONE]`, order ids with `[ORDER_ID]`, using the patterns `[\w.+-]+@[\w-]+\.[\w.]+`, `\b[6-9]\d{9}\b` and `ORD-\d{6}`. Over all 164 tickets there were 278 substitutions (55 emails, 91 phone numbers, 132 order ids), and re-applying the three patterns to the redacted texts found 0 matches. The user message sent to the model is exactly the redacted ticket text, and the code asserts that no email, phone number or order id appears in any message sent.

### Prompts (Task 3)
Both prompts contain the word `CLASSIFY`, list the four categories (billing, delivery, technical, fraud) and require a JSON-only reply of the form `{"category": "...", "reason": "..."}`. The zero-shot prompt has no examples (33 estimated tokens). The few-shot prompt adds a block starting with `EXAMPLES:` and two labelled examples, one of them a fraud ticket (82 estimated tokens).

### Retries (Task 4)
Every model call is wrapped in a retry function with at most 3 retries and exponential backoff with jitter: the wait before retry number n is `0.05 * 2**n * random.uniform(0.9, 1.15)` seconds, and each retry prints one log line with the ticket id, the retry number and the wait. `mock_llm.reset()` is called before each full run. A full zero-shot run over the 156 eligible tickets raised 19 `TransientLLMError`s. All 19 were recovered by the first retry (retry number 0, waits between 0.0459 and 0.0571 seconds, inside the allowed range of 0.045 to 0.0575), and no ticket ended with a permanent failure.

### Validation of answers (Task 5)
The model reply is parsed as JSON. If the `category` is one of the four allowed values the ticket gets status `classified_json`. If the reply is not valid JSON, the lowercase words are extracted with `re.findall(r"[a-z]+", text.lower())`, and the ticket gets `classified_keyword` only if exactly one allowed category word is present. In every other case, including valid JSON whose category is outside the list, the ticket gets `flagged_invalid_output` and goes to a human, and no category is guessed. For the 156 eligible tickets: 135 `classified_json`, 12 `classified_keyword`, 9 `flagged_invalid_output`.

### Zero-shot against few-shot (Task 6)
| Prompt | Tickets classified | Correct | Accuracy | Total tokens | Tokens per classified ticket |
|---|---|---|---|---|---|
| Zero-shot | 147 | 126 | 85.7% | 9,556 | 65.0 |
| Few-shot | 147 | 147 | 100% | 17,200 | 117.0 |

Zero-shot confusion table (rows are the true category, columns the predicted one):

| True \ predicted | billing | delivery | fraud | technical |
|---|---|---|---|---|
| billing | 38 | 0 | 0 | 0 |
| delivery | 0 | 36 | 0 | 0 |
| fraud | 11 | 0 | 23 | 0 |
| technical | 10 | 0 | 0 | 29 |

The few-shot prompt is 49 estimated tokens longer than the zero-shot prompt, so each of the 156 model calls costs 49 more tokens (7,644 more in total, which is why the total went from 9,556 to 17,200). The extra tokens were worth paying for in this run: they removed all 21 errors, including the 11 of 34 fraud tickets that zero-shot had sent to billing, where they would never reach the security team. This comparison uses the offline mock model and synthetic tickets, so the 100% figure should not be expected to carry over to a real model.

### Token budget and routing (Task 7)
**Budget demo:** the zero-shot pipeline was run with `TOKEN_BUDGET = 6000` (the budget is checked before every call). The run used 6,017 tokens in total, slightly above 6,000 because the last call that started under the budget pushed it over, and 58 tickets received status `budget_exhausted` and no model call. The result is saved in `outputs/triage_budget_demo.csv`.

**Final run:** the few-shot pipeline was run with `TOKEN_BUDGET = 100000`, and the result was joined with `outputs/customer_segments.csv` from Part B. Routing rules, applied in this order: tickets with status `skipped_empty`, `flagged_long`, `flagged_invalid_output`, `budget_exhausted` or `failed_call` go to `human_review` with priority NORMAL; otherwise a `fraud` ticket goes to `security_team` with priority HIGH; otherwise a ticket from a Premium customer goes to `premium_desk` with priority HIGH; otherwise it goes to `support_queue` with priority NORMAL. A customer with no row in the segment file is treated as not premium. `outputs/triage_results.csv` has 164 rows.

| Route | Priority | Tickets |
|---|---|---|
| security_team | HIGH | 34 |
| premium_desk | HIGH | 11 |
| support_queue | NORMAL | 102 |
| human_review | NORMAL | 17 |

**Rule-validated outputs:** 147 tickets (135 `classified_json` and 12 `classified_keyword`) have a category that passed the validation rules, and were routed by category and segment. **Sent to humans:** 17 tickets (9 `flagged_invalid_output`, 4 `skipped_empty` and 4 `flagged_long`). The final run used 17,200 tokens, and the few-shot prompt costs 117.0 tokens per classified ticket (see Task 6).

## Part D
(coming later)

## Limitations and Next Steps

**Part A**
- The data is synthetic and fictional, so the results describe this generated dataset only.
- Missing values were filled with simple rules, so the filled values are estimates and not recorded values: 67 ages (61 blank and 6 impossible) with the median age of the customer's city, 774 delivery times of completed orders with the median of the product category, and 112 order amounts with the formula price_inr x quantity x (1 - discount_pct / 100). Every filled value in a group is the same number, and no column records which values were filled.
- Cleaning removed only exact duplicate rows (119) and earlier pending records of the same order_id (656).
- The 897 orders above the 1.5 x IQR upper fence were kept in the data, and their cause was not investigated.
- The order amounts are strongly right-skewed (skewness 4.61). The correlations are all close to 0, and correlation measures only straight-line relationships and does not establish cause and effect.
- `validate.py` checks the clean tables against the fixed reference numbers of this dataset, so it would need new expected values for a different dataset.
- Dates are stored in SQLite as text (YYYY-MM-DD HH:MM:SS), and the daily revenue query depends on that format.
- The project ran on Python 3.10.11, while the brief recommends Python 3.11 or newer.
- Ideas for next steps (not done): add a column that flags every filled value, and compare mean-age filling with median-age filling.

**Part B**
- The Premium flag used for the retention list is a post-hoc whole-window label (the Task 3 segments use orders from the September window that defines the lapse label), so the retention list is illustrative and not a leakage-free evaluation. The model's own features use only orders before 2025-09-01.
- The test set is small (232 customers, 89 lapsed), so scores such as F1 can move by a few points with a different split. The two models differed by one customer on the test set.
- The best logistic regression C was 100, the largest value in the grid, so a wider grid might help.
- Next steps: build the segments from pre-cutoff orders only, tune the decision threshold for recall, and compare mean-age against median-age filling in the cleaning step.

**Part C**
- All results come from the offline mock model and synthetic tickets, so the 85.7% (zero-shot) and 100% (few-shot) accuracies describe this mock, not a real language model.
- Token counts are estimates (the number of words x 1.3, rounded up), not billed tokens.
- The budget is checked before each call, so a run can end slightly over the budget (6,017 tokens against 6,000 in the demo).
- Redaction covers only emails, 10-digit phone numbers starting with 6 to 9, and order ids of the form ORD- followed by six digits. Any other personal data in a ticket would not be hidden.
- Rule-validated outputs were checked only for format and for an allowed category. The accuracy against `true_category` was measured for the comparison only and is not part of the routing.
- A customer with no row in the segment file is treated as not premium, and the Premium flag has the same whole-window caveat as in Part B.