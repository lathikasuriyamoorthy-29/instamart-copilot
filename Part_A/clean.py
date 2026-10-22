import pandas as pd
import numpy as np

customers = pd.read_csv("data/raw/customers.csv")

# (a) City: remove spaces at the ends, make "chennai"/"CHENNAI" -> "Chennai"
customers["city"] = customers["city"].str.strip().str.title()

# (b) Gender: map every spelling to male / female / unknown
gender_map = {"male": "male", "m": "male", "female": "female", "f": "female"}
g = customers["gender"].fillna("").str.strip().str.lower()
customers["gender"] = g.map(gender_map).fillna("unknown")

# (c) Age: impossible ages become missing, then fill with the city's median age
bad_age = (customers["age"] < 10) | (customers["age"] > 100)
customers.loc[bad_age, "age"] = np.nan
city_median = customers.groupby("city")["age"].transform("median")
customers["age"] = customers["age"].fillna(city_median)

# (d) signup_date: text -> real date
customers["signup_date"] = pd.to_datetime(customers["signup_date"])

# ---- checks ----
print(customers["city"].value_counts())
print(customers["gender"].value_counts())
print("missing ages:", customers["age"].isna().sum())
print("age min/max:", customers["age"].min(), customers["age"].max())
print(customers.dtypes)

# ================= ORDERS =================
orders = pd.read_csv("data/raw/orders.csv")

# Rule i: order_ts text -> real date and time
orders["order_ts"] = pd.to_datetime(orders["order_ts"])

# Rule ii: remove rows that are exact copies
orders = orders.drop_duplicates()

# Rule iii: if an order_id appears more than once, keep only the latest record
orders = orders.sort_values("order_ts").drop_duplicates("order_id", keep="last")

products = pd.read_csv("data/raw/products.csv")

# Rule iv: order_amount -> numbers, fill missing from the product price
orders["order_amount"] = pd.to_numeric(orders["order_amount"], errors="coerce")
price = orders["product_id"].map(products.set_index("product_id")["price_inr"])
calc_amount = (price * orders["quantity"] * (1 - orders["discount_pct"] / 100)).round(2)
print("amounts filled:", orders["order_amount"].isna().sum())
orders["order_amount"] = orders["order_amount"].fillna(calc_amount)

# Rule v: negative delivery times -> missing, then fill for completed orders
print("negative delivery times:", (orders["delivery_minutes"] < 0).sum())
orders.loc[orders["delivery_minutes"] < 0, "delivery_minutes"] = np.nan

orders["category"] = orders["product_id"].map(products.set_index("product_id")["category"])
completed = orders["status"] == "completed"
cat_median = (orders["delivery_minutes"].where(completed)
              .groupby(orders["category"]).transform("median"))
fill_mask = completed & orders["delivery_minutes"].isna()
print("delivery times filled:", fill_mask.sum())
orders.loc[fill_mask, "delivery_minutes"] = cat_median[fill_mask]
orders = orders.drop(columns="category")   # helper column, no longer needed

# ---- checks ----
completed = orders["status"] == "completed"
print("rows:", len(orders))
print("completed:", completed.sum(), "| cancelled:", (orders["status"] == "cancelled").sum())
print("missing order_amount:", orders["order_amount"].isna().sum())
print("missing delivery (completed):", orders.loc[completed, "delivery_minutes"].isna().sum())
print("missing delivery (cancelled):", orders.loc[~completed, "delivery_minutes"].isna().sum())
print("sum of completed amounts:", round(orders.loc[completed, "order_amount"].sum(), 2))

# ================= SAVE =================
import os, sqlite3
os.makedirs("data/clean", exist_ok=True)

# dates become plain text, because SQLite has no real date type
orders_out = orders.copy()
orders_out["order_ts"] = orders_out["order_ts"].dt.strftime("%Y-%m-%d %H:%M:%S")
customers_out = customers.copy()
customers_out["signup_date"] = customers_out["signup_date"].dt.strftime("%Y-%m-%d")

# save as CSV files
customers_out.to_csv("data/clean/customers.csv", index=False)
products.to_csv("data/clean/products.csv", index=False)
orders_out.to_csv("data/clean/orders.csv", index=False)

# save into the SQLite database
conn = sqlite3.connect("data/instamart.db")
customers_out.to_sql("customers", conn, if_exists="replace", index=False)
products.to_sql("products", conn, if_exists="replace", index=False)
orders_out.to_sql("orders", conn, if_exists="replace", index=False)
conn.close()
print("saved clean CSVs and data/instamart.db")