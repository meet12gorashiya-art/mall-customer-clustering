"""Prediction utilities: load the trained clustering model and assign new customers to a segment."""

import os
import pickle
from typing import Dict

from src.exceptions import InvalidInputError, ModelNotFoundError
from src.preprocessing import CLUSTER_FEATURES
from src.utils.logger import get_logger

logger = get_logger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MODEL_PATH = os.path.join(BASE_DIR, "models", "cluster_model.pkl")


def load_model(model_path: str = DEFAULT_MODEL_PATH):
    """Load the pickled KMeans model from disk.

    Raises
    ------
    ModelNotFoundError
        If no trained model exists at ``model_path`` yet.
    """
    if not os.path.exists(model_path):
        logger.error("No trained model found at %s", model_path)
        raise ModelNotFoundError(
            f"No trained model at {model_path}. Run `python -m src.train` first."
        )
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    logger.info("Loaded clustering model from %s", model_path)
    return model


def validate_customer(customer: Dict[str, float]) -> None:
    """Ensure both required clustering features are present."""
    missing = [f for f in CLUSTER_FEATURES if f not in customer]
    if missing:
        logger.error("Missing required customer fields: %s", missing)
        raise InvalidInputError(f"Missing required fields: {missing}")


def describe_segment(model, cluster_id: int) -> str:
    """Return a human-readable label for a cluster based on its center."""
    center = model.cluster_centers_[cluster_id]
    income, spending = center[0], center[1]
    income_level = "high" if income >= 65 else "mid" if income >= 40 else "low"
    spending_level = "high" if spending >= 65 else "mid" if spending >= 40 else "low"
    return f"{income_level.capitalize()} income, {spending_level} spending"


def predict_segment(model, customer: Dict[str, float]) -> Dict:
    """Assign a new customer to a cluster.

    Parameters
    ----------
    model : fitted sklearn KMeans estimator
    customer : dict
        Must contain ``Annual_Income`` and ``Spending_Score``.

    Returns
    -------
    dict with keys ``cluster`` (int) and ``description`` (str).
    """
    validate_customer(customer)
    row = [[customer[col] for col in CLUSTER_FEATURES]]
    try:
        cluster_id = int(model.predict(row)[0])
    except Exception as exc:
        logger.error("Cluster prediction failed: %s", exc)
        raise
    description = describe_segment(model, cluster_id)
    logger.info("Predicted cluster=%d (%s)", cluster_id, description)
    return {"cluster": cluster_id, "description": description}
