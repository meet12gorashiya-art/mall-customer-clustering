"""Feature selection and cluster-quality evaluation utilities.

Mirrors the original notebook, which clusters mall customers on
``Annual_Income`` and ``Spending_Score``, and evaluates candidate values
of k using both the elbow method (within-cluster sum of squares) and the
silhouette score.
"""

from typing import List, Tuple

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from src.utils.logger import get_logger

logger = get_logger(__name__)

CLUSTER_FEATURES = ["Annual_Income", "Spending_Score"]


def select_features(df: pd.DataFrame, columns: List[str] = None) -> pd.DataFrame:
    """Select the numeric columns used for clustering."""
    columns = columns or CLUSTER_FEATURES
    features = df[columns].copy()
    logger.info("Selected clustering features %s -> shape %s", columns, features.shape)
    return features


def evaluate_k_range(features: pd.DataFrame, k_min: int = 3, k_max: int = 8, random_state: int = 42) -> pd.DataFrame:
    """Compute WCSS (elbow) and silhouette score for each k in [k_min, k_max].

    Returns
    -------
    pd.DataFrame with columns: cluster, WCSS_Score, Silhouette_Score
    """
    ks: List[int] = []
    wcss: List[float] = []
    sil: List[float] = []

    for k in range(k_min, k_max + 1):
        model = KMeans(n_clusters=k, n_init="auto", random_state=random_state).fit(features)
        ks.append(k)
        wcss.append(model.inertia_)
        sil.append(silhouette_score(features, model.labels_))

    results = pd.DataFrame({"cluster": ks, "WCSS_Score": wcss, "Silhouette_Score": sil})
    logger.info("Evaluated k=%d..%d -> %s", k_min, k_max, results.to_dict("records"))
    return results


def best_k_by_silhouette(results: pd.DataFrame) -> int:
    """Pick the k with the highest silhouette score."""
    best_row = results.loc[results["Silhouette_Score"].idxmax()]
    best_k = int(best_row["cluster"])
    logger.info("Best k by silhouette score: %d (score=%.4f)", best_k, best_row["Silhouette_Score"])
    return best_k
