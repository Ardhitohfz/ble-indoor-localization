"""Custom exceptions for the ML pipeline."""


class BLEPipelineError(Exception):
    """Base exception for BLE pipeline errors."""
    pass


class PreprocessingError(BLEPipelineError):
    """Exception raised during data preprocessing."""
    pass


class ModelError(BLEPipelineError):
    """Exception raised during model operations."""
    pass


class PredictionError(BLEPipelineError):
    """Exception raised during prediction."""
    pass


class DataValidationError(BLEPipelineError):
    """Exception raised during data validation."""
    pass


class ConfigurationError(BLEPipelineError):
    """Exception raised for configuration issues."""
    pass


def create_validation_error(message: str, details: dict = None) -> DataValidationError:
    """Create a validation error with optional details."""
    error = DataValidationError(message)
    if details:
        error.details = details
    return error
