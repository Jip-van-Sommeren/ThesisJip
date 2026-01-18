"""
Battery Storage Manager

Extends the base MultiAgentStorageManager with battery-specific
storage operations for telemetry, predictions, state estimates, and faults.
"""

import time
import logging
from typing import Dict, List, Any, Optional
from src.battery_twin.storage.storage_manager import MultiAgentStorageManager
from src.battery_twin.storage.battery_storage_config import (
    BatteryStorageConfig,
)

logger = logging.getLogger(__name__)


class BatteryStorageManager(MultiAgentStorageManager):
    """
    Extended storage manager for battery digital twin.
    Provides battery-specific methods for storing and retrieving
    telemetry, predictions, state estimates, and fault events.
    """

    def __init__(self, config: BatteryStorageConfig):
        """Initialize battery storage manager."""
        super().__init__(config)
        self.battery_config = config
        self.battery_ids = config.battery_ids

    def initialize_battery_storage(
        self, battery_id: str, metadata: Dict[str, Any]
    ):
        """
        Initialize storage for a new battery.

        Args:
            battery_id: Battery identifier (e.g., "B0005")
            metadata: Battery metadata (type, capacity, chemistry, etc.)
        """
        if self.document_store:
            # Store battery metadata
            battery_doc = {
                "battery_id": battery_id,
                "battery_type": metadata.get("battery_type", "Li-ion"),
                "nominal_capacity": metadata.get("nominal_capacity", 2.0),
                "nominal_voltage": metadata.get("nominal_voltage", 3.7),
                "chemistry": metadata.get("chemistry", "LiCoO2"),
                "manufacturing_date": metadata.get(
                    "manufacturing_date", "unknown"
                ),
                "first_cycle_date": time.time(),
                "metadata": metadata,
            }

            # Insert or update
            collection = self.document_store.db["battery_metadata"]
            collection.update_one(
                {"battery_id": battery_id}, {"$set": battery_doc}, upsert=True
            )

        # Initialize cache for latest values
        if self.cache:
            cache_key = f"battery:{battery_id}:initialized"
            self.cache.client.set(cache_key, "true", ex=86400)  # 24 hours

        logger.info(f"Initialized storage for battery {battery_id}")

    def record_telemetry(
        self,
        battery_id: str,
        voltage: float,
        current: float,
        temperature: float,
        cycle: int,
        timestamp: Optional[float] = None,
        **kwargs,
    ):
        """
        Record battery telemetry data.

        Args:
            battery_id: Battery identifier
            voltage: Battery voltage (V)
            current: Battery current (A)
            temperature: Cell temperature (°C)
            cycle: Cycle number
            timestamp: Measurement timestamp (defaults to now)
            **kwargs: Additional fields (ambient_temperature, etc.)
        """
        if timestamp is None:
            timestamp = time.time()

        if not self.time_series:
            return

        # Create measurement points
        points = []

        # Voltage measurement
        points.append(
            {
                "measurement": "battery_voltage",
                "tags": {
                    "battery_id": battery_id,
                    "measurement_type": "measured",
                },
                "fields": {"voltage": float(voltage), "cycle": int(cycle)},
                "time": int(timestamp * 1e9),  # nanoseconds
            }
        )

        # Current measurement
        points.append(
            {
                "measurement": "battery_current",
                "tags": {
                    "battery_id": battery_id,
                    "measurement_type": "measured",
                },
                "fields": {"current": float(current), "cycle": int(cycle)},
                "time": int(timestamp * 1e9),
            }
        )

        # Temperature measurement
        points.append(
            {
                "measurement": "battery_temperature",
                "tags": {
                    "battery_id": battery_id,
                    "measurement_type": "measured",
                },
                "fields": {
                    "temperature": float(temperature),
                    "ambient_temperature": float(
                        kwargs.get("ambient_temperature", temperature)
                    ),
                    "cycle": int(cycle),
                },
                "time": int(timestamp * 1e9),
            }
        )

        # Write points
        try:
            self.time_series.write_points(points)
        except Exception as e:
            logger.error(f"Failed to record telemetry: {e}")

        # Update cache with latest values
        if self.cache:
            cache_data = {
                "voltage": voltage,
                "current": current,
                "temperature": temperature,
                "cycle": cycle,
                "timestamp": timestamp,
            }
            cache_key = f"battery:{battery_id}:latest_telemetry"
            self.cache.client.hset(
                cache_key, mapping={k: str(v) for k, v in cache_data.items()}
            )
            self.cache.client.expire(
                cache_key, self.battery_config.cache_ttl_seconds
            )

    def record_capacity(
        self,
        battery_id: str,
        capacity: float,
        cycle: int,
        timestamp: Optional[float] = None,
    ):
        """Record battery capacity measurement."""
        if timestamp is None:
            timestamp = time.time()

        if self.time_series:
            point = {
                "measurement": "battery_capacity",
                "tags": {"battery_id": battery_id, "cycle": str(cycle)},
                "fields": {"capacity": float(capacity)},
                "time": int(timestamp * 1e9),
            }

            try:
                self.time_series.write_points([point])
            except Exception as e:
                logger.error(f"Failed to record capacity: {e}")

    def record_prediction(
        self,
        battery_id: str,
        agent_id: str,
        prediction_type: str,
        predicted_capacity: float,
        uncertainty: Optional[float] = None,
        horizon: int = 0,
        timestamp: Optional[float] = None,
        cycle: Optional[int] = None,
    ):
        """
        Record battery capacity prediction.

        Args:
            battery_id: Battery identifier
            agent_id: Agent that made the prediction
            prediction_type: "physics", "ml", or "hybrid"
            predicted_capacity: Predicted capacity value
            uncertainty: Prediction uncertainty (optional)
            horizon: Prediction horizon in seconds
            timestamp: Prediction timestamp
            cycle: Current cycle number
        """
        if timestamp is None:
            timestamp = time.time()

        # Store in InfluxDB
        if self.time_series:
            point = {
                "measurement": "battery_predictions",
                "tags": {
                    "battery_id": battery_id,
                    "prediction_type": prediction_type,
                    "agent_id": agent_id,
                },
                "fields": {
                    "predicted_capacity": float(predicted_capacity),
                    "uncertainty": (
                        float(uncertainty) if uncertainty is not None else 0.0
                    ),
                    "horizon": int(horizon),
                },
                "time": int(timestamp * 1e9),
            }

            if cycle is not None:
                point["fields"]["cycle"] = int(cycle)

            try:
                self.time_series.write_points([point])
            except Exception as e:
                logger.error(f"Failed to record prediction: {e}")

        # Update cache
        if self.cache and prediction_type == "hybrid":
            cache_key = f"battery:{battery_id}:latest_prediction"
            cache_data = {
                "predicted_capacity": str(predicted_capacity),
                "uncertainty": str(uncertainty) if uncertainty else "0.0",
                "timestamp": str(timestamp),
            }
            self.cache.client.hset(cache_key, mapping=cache_data)
            self.cache.client.expire(
                cache_key, self.battery_config.cache_ttl_seconds
            )

    def record_hybrid_training_sample(
        self,
        battery_id: str,
        cycle: int,
        temperature: float,
        duration: float,
        capacity: float,
        source: str = "telemetry",
        timestamp: Optional[float] = None,
    ):
        """Persist hybrid training samples for traceability."""
        if timestamp is None:
            timestamp = time.time()

        if not self.document_store:
            return

        try:
            collection = self.document_store.db["hybrid_training_samples"]
            collection.insert_one(
                {
                    "battery_id": battery_id,
                    "cycle": int(cycle),
                    "temperature": float(temperature),
                    "duration": float(duration),
                    "capacity": float(capacity),
                    "source": source,
                    "timestamp": float(timestamp),
                }
            )
        except Exception as exc:
            logger.error(f"Failed to record hybrid training sample: {exc}")

    def record_state_estimate(
        self,
        battery_id: str,
        agent_id: str,
        soc: float,
        soh: float,
        internal_resistance: Dict[str, float],
        timestamp: Optional[float] = None,
        covariance: Optional[List[List[float]]] = None,
    ):
        """
        Record battery state estimate (SoC, SoH, resistance).

        Args:
            battery_id: Battery identifier
            agent_id: State estimator agent ID
            soc: State of Charge (0-1)
            soh: State of Health (0-1)
            internal_resistance: Dict with R0, R1, C1 values
            timestamp: Estimate timestamp
            covariance: Covariance matrix (optional)
        """
        if timestamp is None:
            timestamp = time.time()

        if self.time_series:
            point = {
                "measurement": "battery_state_estimates",
                "tags": {"battery_id": battery_id, "agent_id": agent_id},
                "fields": {
                    "soc": float(soc),
                    "soh": float(soh),
                    "r0": float(internal_resistance.get("R0", 0.0)),
                    "r1": float(internal_resistance.get("R1", 0.0)),
                    "c1": float(internal_resistance.get("C1", 0.0)),
                },
                "time": int(timestamp * 1e9),
            }

            try:
                self.time_series.write_points([point])
            except Exception as e:
                logger.error(f"Failed to record state estimate: {e}")

        # Update cache
        if self.cache:
            cache_key = f"battery:{battery_id}:latest_state"
            cache_data = {
                "soc": str(soc),
                "soh": str(soh),
                "r0": str(internal_resistance.get("R0", 0.0)),
                "timestamp": str(timestamp),
            }
            self.cache.client.hset(cache_key, mapping=cache_data)
            self.cache.client.expire(
                cache_key, self.battery_config.cache_ttl_seconds
            )

    def record_fault_event(
        self,
        battery_id: str,
        agent_id: str,
        severity: str,
        fault_type: str,
        cause: str,
        residual_magnitude: float,
        timestamp: Optional[float] = None,
    ):
        """Record battery fault event."""
        if timestamp is None:
            timestamp = time.time()

        # Store in InfluxDB for time series
        if self.time_series:
            point = {
                "measurement": "battery_faults",
                "tags": {
                    "battery_id": battery_id,
                    "fault_type": fault_type,
                    "severity": severity,
                },
                "fields": {
                    "residual_magnitude": float(residual_magnitude),
                    "description": cause,
                },
                "time": int(timestamp * 1e9),
            }

            try:
                self.time_series.write_points([point])
            except Exception as e:
                logger.error(f"Failed to record fault event: {e}")

        # Store in MongoDB for detailed records
        if self.document_store:
            fault_doc = {
                "battery_id": battery_id,
                "agent_id": agent_id,
                "timestamp": timestamp,
                "severity": severity,
                "fault_type": fault_type,
                "cause": cause,
                "residual_magnitude": residual_magnitude,
            }

            collection = self.document_store.db["fault_events"]
            collection.insert_one(fault_doc)

    def record_parameters(
        self,
        battery_id: str,
        agent_id: str,
        parameters: Dict[str, float],
        confidence: float,
        cycle: int,
        timestamp: Optional[float] = None,
    ):
        """Record updated battery model parameters."""
        if timestamp is None:
            timestamp = time.time()

        # Store in InfluxDB
        if self.time_series:
            point = {
                "measurement": "battery_parameters",
                "tags": {"battery_id": battery_id, "agent_id": agent_id},
                "fields": {
                    "k": float(parameters.get("k", 0.0)),
                    "c0": float(parameters.get("C0", 0.0)),
                    "r0": float(parameters.get("R0", 0.0)),
                    "r1": float(parameters.get("R1", 0.0)),
                    "c1": float(parameters.get("C1", 0.0)),
                    "confidence": float(confidence),
                    "cycle": int(cycle),
                },
                "time": int(timestamp * 1e9),
            }

            try:
                self.time_series.write_points([point])
            except Exception as e:
                logger.error(f"Failed to record parameters: {e}")

        # Store in MongoDB for parameter history
        if self.document_store:
            param_doc = {
                "battery_id": battery_id,
                "agent_id": agent_id,
                "timestamp": timestamp,
                "cycle": cycle,
                "parameters": parameters,
                "confidence": confidence,
                "fit_quality": {},
            }

            collection = self.document_store.db["parameter_history"]
            collection.insert_one(param_doc)

    def get_latest_state(self, battery_id: str) -> Optional[Dict[str, Any]]:
        """Get latest battery state from cache."""
        if not self.cache:
            return None

        cache_key = f"battery:{battery_id}:latest_state"
        try:
            data = self.cache.client.hgetall(cache_key)
            if data:
                return {
                    "soc": float(data.get(b"soc", b"0").decode()),
                    "soh": float(data.get(b"soh", b"0").decode()),
                    "r0": float(data.get(b"r0", b"0").decode()),
                    "timestamp": float(data.get(b"timestamp", b"0").decode()),
                }
        except Exception as e:
            logger.error(f"Failed to get latest state: {e}")

        return None

    def get_latest_telemetry(
        self, battery_id: str
    ) -> Optional[Dict[str, Any]]:
        """Get latest telemetry from cache."""
        if not self.cache:
            return None

        cache_key = f"battery:{battery_id}:latest_telemetry"
        try:
            data = self.cache.client.hgetall(cache_key)
            if data:
                return {
                    "voltage": float(data.get(b"voltage", b"0").decode()),
                    "current": float(data.get(b"current", b"0").decode()),
                    "temperature": float(
                        data.get(b"temperature", b"0").decode()
                    ),
                    "cycle": int(data.get(b"cycle", b"0").decode()),
                    "timestamp": float(data.get(b"timestamp", b"0").decode()),
                }
        except Exception as e:
            logger.error(f"Failed to get latest telemetry: {e}")

        return None

    def store_trained_model(
        self,
        agent_id: str,
        battery_id: str,
        model_type: str,
        model_data: bytes,
        version: str,
        metadata: Dict[str, Any],
        performance_metrics: Dict[str, float],
    ):
        """Store trained model in MongoDB."""
        if not self.document_store:
            return

        model_doc = {
            "agent_id": agent_id,
            "battery_id": battery_id,
            "model_type": model_type,
            "version": version,
            "model_data": model_data,
            "metadata": metadata,
            "timestamp": time.time(),
            "performance_metrics": performance_metrics,
        }

        collection = self.document_store.db["trained_models"]
        collection.insert_one(model_doc)
        logger.info(f"Stored trained model for {agent_id} version {version}")

    def load_latest_model(
        self, agent_id: str, battery_id: str, model_type: str
    ) -> Optional[Dict[str, Any]]:
        """Load latest trained model from MongoDB."""
        if not self.document_store:
            return None

        collection = self.document_store.db["trained_models"]
        model_doc = collection.find_one(
            {
                "agent_id": agent_id,
                "battery_id": battery_id,
                "model_type": model_type,
            },
            sort=[("timestamp", -1)],
        )

        return model_doc
