"""
Clustering de ciclistes amb KMeans
"""

import pickle
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import homogeneity_score, completeness_score, v_measure_score


df = pd.read_csv("data\\ciclistes.csv")

labels_reals = df["tipus"]
X = df.drop(columns=["id", "tt", "tipus"])

model = KMeans(n_clusters=4, random_state=42)
labels = model.fit_predict(X)

with open("model\\clustering_model.pkl", "wb") as f:
    pickle.dump(model, f)

scores = {
    "homogeneity": homogeneity_score(labels_reals, labels),
    "completeness": completeness_score(labels_reals, labels),
    "v_measure": v_measure_score(labels_reals, labels)
}

with open("model\\scores.pkl", "wb") as f:
    pickle.dump(scores, f)

df["label"] = labels

tipus_dict = {}
for lbl in df["label"].unique():
    tipus_dict[lbl] = df[df["label"] == lbl]["tipus"].mode()[0]

with open("model\\tipus_dict.pkl", "wb") as f:
    pickle.dump(tipus_dict, f)

for t in df["tipus"].unique():
    with open(f"informes\\{t}.txt", "w") as f:
        f.write(str(df[df["tipus"] == t].describe()))

print("Clustering finalitzat correctament")

nous = [
    [3230, 1430],
    [3300, 2120],
    [4010, 1510],
    [4350, 2200]
]

print(model.predict(nous))