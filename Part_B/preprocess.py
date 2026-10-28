from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from lapse_data import FEATURES, build_lapse_dataset, load_split

CATEGORICAL = ["city", "gender", "favourite_category"]
NUMERIC = [c for c in FEATURES if c not in CATEGORICAL]


def build_preprocessor():
    return ColumnTransformer(
        [
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), CATEGORICAL),
            ("num", StandardScaler(), NUMERIC),
        ],
        sparse_threshold=0,
    )


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_split()

    # no feature column may contain a missing value
    assert X_train.isna().sum().sum() == 0, "missing values in training features"
    assert X_test.isna().sum().sum() == 0, "missing values in test features"
    print("missing values in features: 0")

    pre = build_preprocessor()
    X_train_t = pre.fit_transform(X_train)
    X_test_t = pre.transform(X_test)
    print("train matrix:", X_train_t.shape, "| test matrix:", X_test_t.shape)

    scaler = pre.named_transformers_["num"]
    i = NUMERIC.index("frequency")
    full_mean = build_lapse_dataset()["frequency"].mean()
    print("frequency mean learned by the scaler (train only):", round(scaler.mean_[i], 3))
    print("frequency mean of the full cohort:", round(full_mean, 3))

    n = len(NUMERIC)
    print("train numeric means after scaling (about 0):", X_train_t[:, -n:].mean(axis=0).round(2))
    print("test numeric means after scaling (not exactly 0):", X_test_t[:, -n:].mean(axis=0).round(2))