import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from rfm import build_rfm

K = 4

rfm = build_rfm()
cols = ["recency", "frequency", "monetary"]
X = StandardScaler().fit_transform(rfm[cols])

# (a) final clustering and profile
km = KMeans(n_clusters=K, n_init=10, random_state=42)
rfm["segment_id"] = km.fit_predict(X)

profile = (rfm.groupby("segment_id")
              .agg(customers=("customer_id", "count"),
                   recency=("recency", "mean"),
                   frequency=("frequency", "mean"),
                   monetary=("monetary", "mean"))
              .round(2))
print(profile)

# (b) names and actions, chosen from the profile numbers
premium_id = int(profile["monetary"].idxmax())
rest = profile.drop(index=premium_id)
loyal_id = int(rest["frequency"].idxmax())
rest = rest.drop(index=loyal_id)
drifting_id = int(rest["recency"].idxmax())
occasional_id = int(rest["recency"].idxmin())

names = {
    premium_id: "Premium big spenders",
    loyal_id: "Loyal frequent buyers",
    drifting_id: "Drifting customers",
    occasional_id: "Occasional shoppers",
}
actions = {
    "Premium big spenders": "Offer early access to new electronics and a loyalty reward.",
    "Loyal frequent buyers": "Offer a free-delivery pass to protect their ordering habit.",
    "Drifting customers": "Send a win-back reminder with a time-limited discount.",
    "Occasional shoppers": "Offer bundle deals to raise their basket size.",
}

rfm["segment_name"] = rfm["segment_id"].map(names)
rfm["is_premium"] = rfm["segment_id"] == premium_id

for seg_id, name in names.items():
    print(f"segment {seg_id}: {name} -> {actions[name]}")

# (c) PCA
pca = PCA(n_components=2)
pca.fit(X)
ratio = pca.explained_variance_ratio_
print("explained variance:", ratio.round(3), "total:", round(ratio.sum(), 3))

# save
os.makedirs("outputs", exist_ok=True)
out = rfm[["customer_id", "recency", "frequency", "monetary",
           "segment_id", "segment_name", "is_premium"]]
out.to_csv("outputs/customer_segments.csv", index=False)
print("saved outputs/customer_segments.csv:", out.shape)