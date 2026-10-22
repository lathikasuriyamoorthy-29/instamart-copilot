import pandas as pd

# Load the three raw tables.
# keep_default_na=False for tickets: empty text stays "" instead of becoming NaN.
orders = pd.read_csv("data/raw/orders.csv")
customers = pd.read_csv("data/raw/customers.csv")
products = pd.read_csv("data/raw/products.csv")
tickets = pd.read_csv("data/raw/tickets.csv", keep_default_na=False)

tables = {"orders": orders, "customers": customers,
          "products": products, "tickets": tickets}

# Part 1: shape, data types, missing values for every table
def audit_table(name, df):
    print(f"\n=== {name} ===")
    print("shape:", df.shape)
    print(df.dtypes)
    missing = pd.DataFrame({
        "missing_count": df.isna().sum(),
        "missing_pct": (df.isna().mean() * 100).round(2),
    })
    print(missing)

audit_table("orders", orders)
audit_table("customers", customers)
audit_table("products", products)
audit_table("tickets", tickets)

# Part 2: specific orders audit
print("\n--- orders audit ---")
print("rows:", len(orders))
print("exact duplicate rows:", orders.duplicated().sum())
print("pending rows:", (orders["status"] == "pending").sum())
print("distinct order_id:", orders["order_id"].nunique())

amount_num = pd.to_numeric(orders["order_amount"], errors="coerce")
print("non-numeric order_amount:", amount_num.isna().sum())
print("negative delivery_minutes:", (orders["delivery_minutes"] < 0).sum())
print("blank delivery_minutes:", orders["delivery_minutes"].isna().sum())

# Part 3: specific customers audit
print("\n--- customers audit ---")
print("blank age:", customers["age"].isna().sum())
print("age outside 10-100:", ((customers["age"] < 10) | (customers["age"] > 100)).sum())
print("distinct raw city spellings:", customers["city"].nunique())
print("distinct non-blank gender spellings:", customers["gender"].nunique())
print("blank gender:", customers["gender"].isna().sum())