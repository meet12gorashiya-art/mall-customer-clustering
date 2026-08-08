"""Model training pipeline for mall customer segmentation.

Evaluates candidate cluster counts (k=3..8) on Annual_Income and
Spending_Score using the elbow method and silhouette score, selects the
k with the best silhouette score (matching the original notebook's
conclusion), fits the final KMeans model, and persists it.

Run directly to (re)train the model:

    python -m src.train
"""

import argparse
import os
import pickle
import time

from sklearn.cluster import KMeans

from src.data_loader import load_data
from src.exceptions import DataLoadError
from src.preprocessing import CLUSTER_FEATURES, best_k_by_silhouette, evaluate_k_range, select_features
from src.utils.logger import get_logger

logger = get_logger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DATA_PATH = os.path.join(BASE_DIR, "data", "mall_customers.csv")
DEFAULT_MODEL_PATH = os.path.join(BASE_DIR, "models", "cluster_model.pkl")


def train_final_model(features, n_clusters: int, random_state: int = 42) -> KMeans:
    logger.info("Training final KMeans with k=%d", n_clusters)
    model = KMeans(n_clusters=n_clusters, init="k-means++", n_init="auto", random_state=random_state)
    model.fit(features)
    return model


def save_model(model, path: str = DEFAULT_MODEL_PATH) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    logger.info("Saved clustering model to %s", path)


def run_pipeline(data_path: str = DEFAULT_DATA_PATH, model_path: str = DEFAULT_MODEL_PATH, k_min: int = 3, k_max: int = 8):
    """End-to-end pipeline: load -> select features -> evaluate k -> train final model -> save."""
    start = time.time()
    try:
        df = load_data(data_path)
    except DataLoadError:
        logger.exception("Training aborted: could not load data.")
        raise

    features = select_features(df)
    evaluation = evaluate_k_range(features, k_min=k_min, k_max=k_max)
    best_k = best_k_by_silhouette(evaluation)

    model = train_final_model(features, n_clusters=best_k)
    save_model(model, model_path)

    elapsed = time.time() - start
    logger.info("Training pipeline finished in %.2fs", elapsed)
    return {
        "best_k": best_k,
        "evaluation": evaluation.to_dict("records"),
        "elapsed_seconds": elapsed,
        "cluster_centers": model.cluster_centers_.tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description="Train the mall customer clustering model.")
    parser.add_argument("--data-path", default=DEFAULT_DATA_PATH)
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--k-min", type=int, default=3)
    parser.add_argument("--k-max", type=int, default=8)
    args = parser.parse_args()

    try:
        result = run_pipeline(args.data_path, args.model_path, args.k_min, args.k_max)
        print(f"Best k: {result['best_k']}")
        print(f"Cluster centers ({CLUSTER_FEATURES}): {result['cluster_centers']}")
    except Exception as exc:  # noqa: BLE001
        logger.error("Training failed: %s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
