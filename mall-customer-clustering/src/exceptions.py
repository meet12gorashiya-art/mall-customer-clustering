"""Custom exceptions for the mall customer clustering app."""


class DataLoadError(Exception):
    """Raised when the source dataset cannot be loaded or is malformed."""


class ModelNotFoundError(Exception):
    """Raised when a prediction is requested but no trained clustering model exists."""


class InvalidInputError(Exception):
    """Raised when input features for prediction fail validation."""
