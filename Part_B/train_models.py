from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

from lapse_data import load_split
from preprocess import build_preprocessor


def tune_models():
    X_train, X_test, y_train, y_test = load_split()

    logreg = GridSearchCV(
        Pipeline([("pre", build_preprocessor()),
                  ("model", LogisticRegression(max_iter=1000))]),
        param_grid={"model__C": [0.01, 0.1, 1, 10, 100]},
        cv=5,
        scoring="f1",
    )
    logreg.fit(X_train, y_train)

    forest = GridSearchCV(
        Pipeline([("pre", build_preprocessor()),
                  ("model", RandomForestClassifier(n_estimators=100, random_state=42))]),
        param_grid={"model__max_depth": [3, 5, 8],
                    "model__min_samples_leaf": [5, 10, 20]},
        cv=5,
        scoring="f1",
    )
    forest.fit(X_train, y_train)

    return {"Logistic regression": logreg, "Random forest": forest}


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_split()
    models = tune_models()

    for name, search in models.items():
        print(f"\n=== {name} ===")
        print("best params:", search.best_params_)
        print("best cross-validated F1:", round(search.best_score_, 3))

        pred = search.predict(X_test)
        print("confusion matrix (rows = true 0/1, columns = predicted 0/1):")
        print(confusion_matrix(y_test, pred))
        print("precision (lapsed):", round(precision_score(y_test, pred), 3))
        print("recall (lapsed):", round(recall_score(y_test, pred), 3))
        print("test F1 (lapsed):", round(f1_score(y_test, pred), 3))