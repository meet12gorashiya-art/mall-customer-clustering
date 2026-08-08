"""Data loading utilities for the mall customer clustering app."""

import os

import pandas as pd

from src.exceptions import DataLoadError
from src.utils.logger import get_logger

logger = get_logger(__name__)

EXPECTED_COLUMNS = {"Customer_ID", "Gender", "Age", "Annual_Income", "Spending_Score"}


def load_data(csv_path: str) -> pd.DataFrame:
    """Load and lightly validate the mall customers dataset.

    Raises
    ------
    DataLoadError
        If the file is missing, empty, or missing expected columns.
    """
    if not os.path.exists(csv_path):
        logger.error("Data file not found at %s", csv_path)
        raise DataLoadError(f"Data file not found: {csv_path}")

    try:
        df = pd.read_csv(csv_path)
    except pd.errors.EmptyDataError as exc:
        logger.error("Data file is empty: %s", csv_path)
        raise DataLoadError(f"Data file is empty: {csv_path}") from exc
    except pd.errors.ParserError as exc:
        logger.error("Could not parse CSV %s: %s", csv_path, exc)
        raise DataLoadError(f"Could not parse CSV: {exc}") from exc

    if df.empty:
        logger.error("Loaded dataframe is empty for %s", csv_path)
        raise DataLoadError(f"Loaded dataframe is empty: {csv_path}")

    missing = EXPECTED_COLUMNS - set(df.columns)
    if missing:
        logger.error("Dataset missing expected columns: %s", missing)
        raise DataLoadError(f"Dataset missing expected columns: {missing}")

    logger.info("Loaded dataset with shape %s from %s", df.shape, csv_path)
    return df
