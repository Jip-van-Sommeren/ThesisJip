"""
Embedded hybrid digital twin components used by the multi-agent system.

This package contains a local copy of the original HybridDigitalTwin modules
so the agent implementation does not depend on the external
Digital-Twin-in-python project.
"""

from .core.digital_twin import HybridDigitalTwin, PredictionResult

__all__ = ["HybridDigitalTwin", "PredictionResult"]

