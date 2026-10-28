import os
import joblib
import numpy as np
import pandas as pd

from lapse_data import build_lapse_dataset, load_split
from train_models import tune_models

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_split()
    models = tune_models()
    forest = models["Random forest"].best_estimator_
    logreg = models["Logistic regression"].best_estimator_

    # (a) importances and coefficients
    names = forest.named_steps["pre"].get_feature_names_out()
    importances = pd.Series(forest.named_steps["model"].feature_importances_, index=names)
    print("top 5 random forest features:")
    print(importances.sort_values(ascending=False).head(5).round(3))

    coefs = pd.Series(logreg.named_steps["model"].coef_[0],
                      index=logreg.named_steps["pre"].get_feature_names_out())
    print("\nlogistic regression coefficients (numeric features):")
    print(coefs[coefs.index.str.startswith("num__")].round(3))

    # (b) save, reload, compare
    os.makedirs("outputs", exist_ok=True)
    joblib.dump(forest, "outputs/lapse_model.joblib")
    reloaded = joblib.load("outputs/lapse_model.joblib")
    same_labels = (reloaded.predict(X_test) == forest.predict(X_test)).all()
    same_probs = np.array_equal(reloaded.predict_proba(X_test), forest.predict_proba(X_test))
    print("\nreloaded model reproduces predictions:", bool(same_labels and same_probs))

    # (c) retention list for Premium customers in the test set
    df = build_lapse_dataset()
    test_info = df.loc[X_test.index, ["customer_id"]].copy()
    test_info["lapse_probability"] = forest.predict_proba(X_test)[:, 1].round(4)
    test_info["actually_lapsed"] = y_test.values

    segments = pd.read_csv("outputs/customer_segments.csv")[["customer_id", "is_premium"]]
    test_info = test_info.merge(segments, on="customer_id", how="left")
    test_info["is_premium"] = test_info["is_premium"].fillna(False).astype(bool)
    premium = test_info[test_info["is_premium"]]

    top = (premium.sort_values("lapse_probability", ascending=False)
                  .head(10)[["customer_id", "lapse_probability", "actually_lapsed"]])
    top.to_csv("outputs/retention_list.csv", index=False)

    print("\nPremium customers in the test set:", len(premium))
    print(top.to_string(index=False))
    print("truly lapsed among the list:", int(top["actually_lapsed"].sum()), "of", len(top))