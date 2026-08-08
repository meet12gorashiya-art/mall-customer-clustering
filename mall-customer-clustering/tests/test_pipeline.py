"""Unit tests for the mall customer clustering pipeline.

Run with:  pytest
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import load_data
from src.exceptions import DataLoadError, InvalidInputError, ModelNotFoundError
from src.predict import describe_segment, load_model, predict_segment, validate_customer
from src.preprocessing import best_k_by_silhouette, evaluate_k_range, select_features
from src.train import run_pipeline

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "mall_customers.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "cluster_model.pkl")


def test_load_data_success():
    df = load_data(DATA_PATH)
    assert not df.empty
    assert "Annual_Income" in df.columns


def test_load_data_missing_file():
    with pytest.raises(DataLoadError):
        load_data("does_not_exist.csv")


def test_select_features_shape():
    df = load_data(DATA_PATH)
    features = select_features(df)
    assert list(features.columns) == ["Annual_Income", "Spending_Score"]
    assert len(features) == len(df)


def test_evaluate_k_range_and_best_k():
    df = load_data(DATA_PATH)
    features = select_features(df)
    results = evaluate_k_range(features, k_min=3, k_max=6)
    assert len(results) == 4
    assert set(results.columns) == {"cluster", "WCSS_Score", "Silhouette_Score"}
    best_k = best_k_by_silhouette(results)
    assert 3 <= best_k <= 6


def test_full_pipeline_and_predict():
    result = run_pipeline(DATA_PATH, MODEL_PATH, k_min=3, k_max=6)
    assert 3 <= result["best_k"] <= 6
    assert os.path.exists(MODEL_PATH)

    model = load_model(MODEL_PATH)
    prediction = predict_segment(model, {"Annual_Income": 80, "Spending_Score": 90})
    assert isinstance(prediction["cluster"], int)
    assert isinstance(prediction["description"], str)


def test_predict_missing_field_raises():
    with pytest.raises(InvalidInputError):
        validate_customer({"Annual_Income": 50})


def test_load_model_missing_raises():
    with pytest.raises(ModelNotFoundError):
        load_model("no_such_model.pkl")
