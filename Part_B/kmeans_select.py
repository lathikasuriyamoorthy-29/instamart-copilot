import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from rfm import build_rfm

rfm = build_rfm()
cols = ["recency", "frequency", "monetary"]

scaler = StandardScaler()
X = scaler.fit_transform(rfm[cols])

rows = []
for k in range(2, 9):
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = km.fit_predict(X)
    rows.append({
        "K": k,
        "inertia": round(km.inertia_, 1),
        "silhouette": round(silhouette_score(X, labels), 3),
    })

table = pd.DataFrame(rows)
print(table.to_string(index=False))