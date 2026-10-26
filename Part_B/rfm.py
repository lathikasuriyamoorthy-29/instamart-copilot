import pandas as pd

SNAPSHOT = pd.Timestamp("2025-10-01 00:00:00")

def build_rfm():
    orders = pd.read_csv("data/clean/orders.csv")
    orders["order_ts"] = pd.to_datetime(orders["order_ts"])

    done = orders[orders["status"] == "completed"]

    rfm = done.groupby("customer_id").agg(
        last_order=("order_ts", "max"),
        frequency=("order_id", "nunique"),
        monetary=("order_amount", "sum"),
    )
    rfm["recency"] = (SNAPSHOT - rfm["last_order"]).dt.days

    rfm = rfm[["recency", "frequency", "monetary"]].reset_index()
    return rfm

if __name__ == "__main__":
    rfm = build_rfm()
    print("customers:", len(rfm))
    print(rfm[["recency", "frequency", "monetary"]].describe().round(2))