import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

CUTOFF = pd.Timestamp("2025-09-01")
OUTCOME_END = pd.Timestamp("2025-10-01")

FEATURES = ["recency", "frequency", "monetary", "avg_discount_pct",
            "cancellation_rate", "avg_delivery_minutes",
            "favourite_category", "city", "age", "gender", "tenure_days"]


def build_lapse_dataset():
    orders = pd.read_csv("data/clean/orders.csv")
    customers = pd.read_csv("data/clean/customers.csv")
    products = pd.read_csv("data/clean/products.csv")

    orders["order_ts"] = pd.to_datetime(orders["order_ts"])
    customers["signup_date"] = pd.to_datetime(customers["signup_date"])

    # ----- label: who ordered during September? -----
    september = orders[(orders["order_ts"] >= CUTOFF)
                       & (orders["order_ts"] < OUTCOME_END)
                       & (orders["status"] == "completed")]
    active_in_september = set(september["customer_id"])

    # ----- features: ONLY orders before the cutoff -----
    history = orders[orders["order_ts"] < CUTOFF]
    assert history["order_ts"].max() < CUTOFF, "an order on/after the cutoff leaked in"
    history = history.merge(products[["product_id", "category"]], on="product_id")
    done = history[history["status"] == "completed"]

    feats = done.groupby("customer_id").agg(
        last_order=("order_ts", "max"),
        frequency=("order_id", "nunique"),
        monetary=("order_amount", "sum"),
        avg_discount_pct=("discount_pct", "mean"),
        avg_delivery_minutes=("delivery_minutes", "mean"),
        favourite_category=("category", lambda s: s.mode().iloc[0]),
    )
    feats["recency"] = (CUTOFF - feats["last_order"]).dt.days
    feats["cancellation_rate"] = (history["status"].eq("cancelled")
                                  .groupby(history["customer_id"]).mean())

    # ----- add customer details and the label -----
    feats = feats.reset_index()
    df = feats.merge(customers[["customer_id", "city", "age", "gender", "signup_date"]],
                     on="customer_id")
    df["tenure_days"] = (CUTOFF - df["signup_date"]).dt.days
    df["lapsed"] = (~df["customer_id"].isin(active_in_september)).astype(int)

    return df[["customer_id"] + FEATURES + ["lapsed"]]


def load_split():
    df = build_lapse_dataset()
    X = df[FEATURES]
    y = df["lapsed"]
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)


if __name__ == "__main__":
    df = build_lapse_dataset()
    print("cohort:", len(df))
    print("lapsed:", df["lapsed"].sum(), f"({df['lapsed'].mean() * 100:.2f}%)")
    print("missing values in features:", df[FEATURES].isna().sum().sum())

    X_train, X_test, y_train, y_test = load_split()
    print("train rows:", len(X_train), "| test rows:", len(X_test))

    baseline = np.zeros(len(y_test), dtype=int)
    print("baseline accuracy:", round(accuracy_score(y_test, baseline), 3))
    print("baseline lapsed-class F1:", f1_score(y_test, baseline, zero_division=0))