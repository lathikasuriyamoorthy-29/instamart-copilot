import numpy as np
import pandas as pd

orders = pd.read_csv("data/clean/orders.csv")
customers = pd.read_csv("data/clean/customers.csv")
products = pd.read_csv("data/clean/products.csv")

# completed orders only, with city and category attached
done = orders[orders["status"] == "completed"]
df = (done.merge(customers[["customer_id", "city"]], on="customer_id")
          .merge(products[["product_id", "category"]], on="product_id"))

# ---------- (a) Q2 again with pivot_table ----------
pivot = pd.pivot_table(df, values="order_amount", index="city",
                       columns="category", aggfunc="sum", margins=True)
print(pivot.round(2))
grand = round(pivot.loc["All", "All"], 2)
print("grand total:", grand)
assert grand == 2998668.80, "pivot total does not match SQL"

report = df.groupby("city").agg(total=("order_amount", "sum"),
                                average=("order_amount", "mean"),
                                orders=("order_amount", "count")).round(2)
print(report)

# ---------- (b) distribution of order_amount ----------
amt = done["order_amount"]
q1, q3 = amt.quantile(0.25), amt.quantile(0.75)
iqr = q3 - q1
lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
n_out = ((amt < lower) | (amt > upper)).sum()

print("\nmean:", round(amt.mean(), 2), "| median:", round(amt.median(), 2))
print("skewness:", round(amt.skew(), 2))
print("Q1:", round(q1, 2), "| Q3:", round(q3, 2))
print("fences:", round(lower, 2), "to", round(upper, 2))
print("outliers above upper fence:", n_out)

# correlation between numeric columns
corr = done[["order_amount", "discount_pct", "quantity", "delivery_minutes"]].corr()
print("\n", corr.round(3))
pairs = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool)).stack()
best = pairs.abs().idxmax()
print("largest pair:", best, "->", round(pairs[best], 3))