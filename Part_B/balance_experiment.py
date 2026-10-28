import pandas as pd
from sklearn.base import clone
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

from lapse_data import FEATURES, load_split
from train_models import tune_models


def scores(model, X_test, y_test):
    pred = model.predict(X_test)
    return {
        "precision": round(precision_score(y_test, pred), 3),
        "recall": round(recall_score(y_test, pred), 3),
        "F1": round(f1_score(y_test, pred), 3),
    }


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_split()
    best_logreg = tune_models()["Logistic regression"].best_estimator_

    # oversample the lapsed class in the TRAINING set only
    train_df = X_train.copy()
    train_df["lapsed"] = y_train
    lapsed_df = train_df[train_df["lapsed"] == 1]
    n_extra = int((y_train == 0).sum() - (y_train == 1).sum())
    extra = lapsed_df.sample(n=n_extra, replace=True, random_state=42)
    balanced = pd.concat([train_df, extra])

    print("training class counts before:", y_train.value_counts().to_dict())
    print("training class counts after:", balanced["lapsed"].value_counts().to_dict())
    print("test rows (unchanged):", len(X_test))

    balanced_model = clone(best_logreg)
    balanced_model.fit(balanced[FEATURES], balanced["lapsed"])

    results = pd.DataFrame({
        "original": scores(best_logreg, X_test, y_test),
        "balanced": scores(balanced_model, X_test, y_test),
    }).T
    print(results)

    for label, model in [("original", best_logreg), ("balanced", balanced_model)]:
        print(f"\nconfusion matrix, {label} model (rows = true 0/1, columns = predicted 0/1):")
        print(confusion_matrix(y_test, model.predict(X_test)))