"""
Unsupervised clustering for discovering novel/zero-day failure patterns in unclassified logs.
"""

from typing import List, Dict, Any
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

class NovelFailureClusterer:
    """Groups unclassified or low-confidence failure logs into distinct signature clusters."""

    def __init__(self, n_clusters: int = 7, random_state: int = 42):
        self.n_clusters = n_clusters
        self.model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)

    def fit_predict(self, X_embeddings) -> np.ndarray:
        return self.model.fit_predict(X_embeddings)

    def compute_silhouette(self, X_embeddings) -> float:
        labels = self.model.predict(X_embeddings)
        if len(set(labels)) > 1:
            return float(silhouette_score(X_embeddings, labels))
        return 0.0
