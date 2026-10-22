import sqlite3
import pandas as pd

conn = sqlite3.connect("data/instamart.db")
customers = pd.read_sql_query("SELECT * FROM customers", conn)
orders = pd.read_sql_query("SELECT * FROM orders", conn)

# 1. city counts
cities = customers["city"].value_counts().to_dict()
assert cities == {"Pune": 227, "Delhi": 210, "Bengaluru": 197,
                  "Chennai": 191, "Hyderabad": 191, "Mumbai": 184}, "city counts wrong"

# 2. gender counts
genders = customers["gender"].value_counts().to_dict()
assert genders == {"female": 610, "male": 560, "unknown": 30}, "gender counts wrong"

# 3. ages
assert customers["age"].isna().sum() == 0, "missing ages"
assert customers["age"].between(18, 60).all(), "age outside 18-60"

# 4. order rows and unique ids
assert len(orders) == 12309, "wrong order row count"
assert orders["order_id"].is_unique, "duplicate order_id"

# 5. no pending rows
assert (orders["status"] == "pending").sum() == 0, "pending rows remain"

# 6. status counts
assert (orders["status"] == "completed").sum() == 11110, "completed count wrong"
assert (orders["status"] == "cancelled").sum() == 1199, "cancelled count wrong"

# 7. missing values
done = orders["status"] == "completed"
assert orders["order_amount"].isna().sum() == 0, "missing order_amount"
assert orders.loc[done, "delivery_minutes"].isna().sum() == 0, "completed missing delivery"
assert orders.loc[~done, "delivery_minutes"].isna().sum() == 1199, "cancelled delivery wrong"

# 8. total revenue
assert round(orders.loc[done, "order_amount"].sum(), 2) == 2998668.80, "revenue total wrong"

# 9. timestamp format YYYY-MM-DD HH:MM:SS
assert orders["order_ts"].str.match(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$").all(), "bad ts format"

print("ALL CHECKPOINTS PASSED")