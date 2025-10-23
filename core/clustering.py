from typing import Optional
import numpy as np
from sklearn.cluster import HDBSCAN, KMeans


def perform_clustering(embeddings: np.ndarray, method: str, n_clusters: Optional[int] = None, min_cluster_size: Optional[int] = None) -> np.ndarray:
    """
    Performs clustering on the given embeddings using the specified method.

    Args:
        embeddings (np.ndarray): The embeddings to be clustered.
        method (str): The clustering method to use, either 'HDBSCAN (Automatic)' or 'KMeans (Fixed)'.
        n_clusters (Optional[int]): The number of clusters for KMeans.
        min_cluster_size (Optional[int]): The minimum cluster size for HDBSCAN.

    Returns:
        np.ndarray: An array of cluster labels for each embedding.
    """
    if method == "HDBSCAN (Automatic)":
        clusterer = HDBSCAN(
            min_cluster_size=min_cluster_size or 10,
            min_samples=10,
            metric="euclidean",
        )
        return clusterer.fit_predict(embeddings)
    else:
        kmeans = KMeans(n_clusters=n_clusters or 10, random_state=42, n_init=10)
        return kmeans.fit_predict(embeddings)
