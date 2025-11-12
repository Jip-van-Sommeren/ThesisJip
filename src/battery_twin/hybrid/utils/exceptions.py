"""
Custom exceptions for the embedded hybrid digital twin components.
"""


class DigitalTwinError(Exception):
    """Base exception for hybrid digital twin errors."""

    pass


class ModelError(DigitalTwinError):
    """Exception raised for model-related errors."""

    pass


class ModelNotTrainedError(ModelError):
    """Raised when attempting to use an untrained model."""

    pass


class InvalidDataError(DigitalTwinError):
    """Raised for invalid or malformed input data."""

    pass


class InvalidParameterError(DigitalTwinError):
    """Raised for invalid parameter values."""

    pass


class ConfigurationError(DigitalTwinError):
    """Raised for configuration-related errors."""

    pass


class DataLoaderError(DigitalTwinError):
    """Raised for data loading and processing errors."""

    pass


class VisualizationError(DigitalTwinError):
    """Raised for visualization-related errors."""

    pass

