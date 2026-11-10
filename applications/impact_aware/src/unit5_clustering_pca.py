"""Unit V: K-Means, hierarchical clustering, DBSCAN and PCA."""
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score

def run(random_state=42):
    X, _ = load_breast_cancer(return_X_y=True)
    Xs = StandardScaler().fit_transform(X)
    # PCA is used to make clustering computationally manageable and comparable.
    reduced = PCA(n_components=5, random_state=random_state).fit_transform(Xs)
    models = {
        "K-Means": KMeans(n_clusters=2, n_init=10, random_state=random_state),
        "Hierarchical": AgglomerativeClustering(n_clusters=2),
        "DBSCAN": DBSCAN(eps=3.0, min_samples=5),
    }
    for name, model in models.items():
        labels = model.fit_predict(reduced)
        unique = set(labels)
        score = silhouette_score(reduced, labels) if len(unique) > 1 and len(unique) < len(labels) else float("nan")
        print(f"{name}: clusters={len(unique - {-1})}, noise={(labels == -1).sum()}, silhouette={score:.4f}")
    print("PCA explained variance ratio:", PCA(n_components=5).fit(Xs).explained_variance_ratio_)

if __name__ == "__main__":
    run()
