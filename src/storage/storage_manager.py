"""
Multi-Agent Storage Manager
Orchestrates all storage backends and provides unified interface.
"""

import time
import logging
from typing import Dict, List, Any, Optional, Set
from datetime import datetime
import threading
from concurrent.futures import ThreadPoolExecutor
import queue

from ...config.storage_config import StorageConfig
from .time_series_storage import InfluxDBStorage
from .document_storage import MongoDBStorage
from .graph_storage import Neo4jStorage
from .cache_storage import RedisStorage

logger = logging.getLogger(__name__)


class StorageHealthMonitor:
    """Monitors health of all storage backends."""

    def __init__(self, storage_manager: "MultiAgentStorageManager"):
        self.storage_manager = storage_manager
        self.health_status: Dict[str, bool] = {}
        self.monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.check_interval = 30  # seconds

    def start_monitoring(self):
        """Start health monitoring."""
        if self.monitoring:
            return

        self.monitoring = True
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop, daemon=True
        )
        self.monitor_thread.start()
        logger.info("Started storage health monitoring")

    def stop_monitoring(self):
        """Stop health monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
        logger.info("Stopped storage health monitoring")

    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.monitoring:
            try:
                self._check_all_health()
                time.sleep(self.check_interval)
            except Exception as e:
                logger.error(f"Error in health monitoring: {e}")
                time.sleep(5)  # Short delay on error

    def _check_all_health(self):
        """Check health of all storage backends."""
        backends = {
            "time_series": self.storage_manager.time_series,
            "document": self.storage_manager.document_store,
            "graph": self.storage_manager.graph_store,
            "cache": self.storage_manager.cache,
        }

        for name, backend in backends.items():
            if backend:
                try:
                    health = backend.health_check()
                    self.health_status[name] = health

                    if not health:
                        logger.warning(
                            f"Storage backend {name} health check failed"
                        )
                        # Attempt reconnection
                        if backend.connect():
                            logger.info(f"Successfully reconnected to {name}")

                except Exception as e:
                    logger.error(f"Error checking {name} health: {e}")
                    self.health_status[name] = False

    def get_health_status(self) -> Dict[str, bool]:
        """Get current health status of all backends."""
        return self.health_status.copy()


class MultiAgentStorageManager:
    """
    Centralized storage management for multi-agent system.
    Coordinates time series, document, graph, and cache storage.
    """

    def __init__(self, config: StorageConfig):
        self.config = config

        # Storage backends
        self.time_series: Optional[InfluxDBStorage] = None
        self.document_store: Optional[MongoDBStorage] = None
        self.graph_store: Optional[Neo4jStorage] = None
        self.cache: Optional[RedisStorage] = None

        # Batch processing
        self.batch_queue: queue.Queue = queue.Queue()
        self.batch_thread: Optional[threading.Thread] = None
        self.batch_executor = ThreadPoolExecutor(max_workers=4)
        self._batch_processing = False

        # Health monitoring
        self.health_monitor = StorageHealthMonitor(self)

        # Initialize backends
        self._initialize_backends()

    def _initialize_backends(self):
        """Initialize storage backends based on configuration."""

        # Initialize time series storage
        if self.config.enable_time_series:
            if self.config.time_series_backend == "influxdb":
                self.time_series = InfluxDBStorage(
                    self.config.influx_config.__dict__
                )
                logger.info("Initialized InfluxDB time series storage")

        # Initialize document storage
        if self.config.enable_document_store:
            if self.config.document_backend == "mongodb":
                self.document_store = MongoDBStorage(
                    self.config.mongo_config.__dict__
                )
                logger.info("Initialized MongoDB document storage")

        # Initialize graph storage
        if self.config.enable_graph_store:
            if self.config.graph_backend == "neo4j":
                self.graph_store = Neo4jStorage(
                    self.config.neo4j_config.__dict__
                )
                logger.info("Initialized Neo4j graph storage")

        # Initialize cache storage
        if self.config.enable_cache:
            if self.config.cache_backend == "redis":
                self.cache = RedisStorage(self.config.redis_config.__dict__)
                logger.info("Initialized Redis cache storage")

    def connect_all(self) -> bool:
        """Connect to all enabled storage backends."""
        success = True

        if self.time_series:
            if not self.time_series.connect():
                logger.error("Failed to connect to time series storage")
                success = False

        if self.document_store:
            if not self.document_store.connect():
                logger.error("Failed to connect to document storage")
                success = False
            else:
                # Setup indexes
                self.document_store.setup_indexes()

        if self.graph_store:
            if not self.graph_store.connect():
                logger.error("Failed to connect to graph storage")
                success = False

        if self.cache:
            if not self.cache.connect():
                logger.error("Failed to connect to cache storage")
                success = False

        if success:
            self.start_batch_processing()
            self.health_monitor.start_monitoring()
            logger.info("Successfully connected to all storage backends")

        return success

    def disconnect_all(self) -> bool:
        """Disconnect from all storage backends."""
        self.stop_batch_processing()
        self.health_monitor.stop_monitoring()

        success = True

        if self.time_series:
            if not self.time_series.disconnect():
                success = False

        if self.document_store:
            if not self.document_store.disconnect():
                success = False

        if self.graph_store:
            if not self.graph_store.disconnect():
                success = False

        if self.cache:
            if not self.cache.disconnect():
                success = False

        self.batch_executor.shutdown(wait=True)
        logger.info("Disconnected from all storage backends")
        return success

    def start_batch_processing(self):
        """Start batch processing for bulk operations."""
        if self._batch_processing:
            return

        self._batch_processing = True
        self.batch_thread = threading.Thread(
            target=self._batch_processor, daemon=True
        )
        self.batch_thread.start()
        logger.info("Started batch processing")

    def stop_batch_processing(self):
        """Stop batch processing."""
        self._batch_processing = False
        if self.batch_thread:
            self.batch_thread.join()
        logger.info("Stopped batch processing")

    def _batch_processor(self):
        """Process batched operations."""
        batch_data = []
        last_flush = time.time()

        while self._batch_processing:
            try:
                # Try to get item with timeout
                try:
                    item = self.batch_queue.get(timeout=1.0)
                    batch_data.append(item)
                except queue.Empty:
                    pass

                # Check if we should flush
                current_time = time.time()
                should_flush = len(batch_data) >= self.config.batch_size or (
                    batch_data
                    and current_time - last_flush >= self.config.flush_interval
                )

                if should_flush and batch_data:
                    self._flush_batch(batch_data)
                    batch_data.clear()
                    last_flush = current_time

            except Exception as e:
                logger.error(f"Error in batch processor: {e}")
                time.sleep(1)

    def _flush_batch(self, batch_data: List[Dict[str, Any]]):
        """Flush batch of operations to storage."""
        if not batch_data:
            return

        # Group by storage type
        time_series_points = []

        for item in batch_data:
            if item["type"] == "time_series":
                time_series_points.append(item["data"])

        # Write time series batch
        if time_series_points and self.time_series:
            self.time_series.write_points(time_series_points)
            logger.debug(
                f"Flushed {len(time_series_points)} time series points"
            )

    # Agent-specific storage methods

    def initialize_agent_storage(
        self, agent_id: str, agent_type: str, roles: Set[str], groups: Set[str]
    ):
        """Initialize storage for a new agent."""

        # Store agent profile in document store
        if self.document_store:
            profile = {
                "agent_id": agent_id,
                "agent_type": agent_type,
                "roles": list(roles),
                "groups": list(groups),
                "created_at": datetime.utcnow(),
                "status": "active",
            }
            self.document_store.store_agent_profile(profile)

        # Create agent node in graph store
        if self.graph_store:
            self.graph_store.create_agent_node(
                agent_id,
                agent_type,
                {"roles": list(roles), "groups": list(groups)},
            )

        # Initialize cache entries
        if self.cache:
            self.cache.cache_agent_state(
                agent_id, {"initialized": True}, expire=3600
            )

        logger.info(f"Initialized storage for agent {agent_id}")

    def record_belief_update(
        self,
        agent_id: str,
        belief_key: str,
        proposition: str,
        confidence: float,
        timestamp: Optional[float] = None,
    ):
        """Record agent belief update."""
        if not self.config.auto_persist_beliefs:
            return

        if self.time_series:
            # Add to batch queue for bulk processing
            point = {
                "measurement": "agent_beliefs",
                "tags": {"agent_id": agent_id, "belief_key": belief_key},
                "fields": {
                    "proposition": proposition,
                    "confidence": confidence,
                },
                "timestamp": timestamp or time.time(),
            }

            self.batch_queue.put({"type": "time_series", "data": point})

    def record_goal_event(
        self,
        agent_id: str,
        goal_id: str,
        event_type: str,
        priority: float,
        goal_type: str,
        timestamp: Optional[float] = None,
    ):
        """Record agent goal event."""
        if not self.config.auto_persist_goals:
            return

        if self.time_series:
            point = {
                "measurement": "agent_goals",
                "tags": {
                    "agent_id": agent_id,
                    "goal_id": goal_id,
                    "event_type": event_type,
                    "goal_type": goal_type,
                },
                "fields": {"priority": priority},
                "timestamp": timestamp or time.time(),
            }

            self.batch_queue.put({"type": "time_series", "data": point})

    def record_action_execution(
        self,
        agent_id: str,
        action_id: str,
        action_type: str,
        success: bool,
        execution_time: float,
        timestamp: Optional[float] = None,
    ):
        """Record agent action execution."""
        if self.time_series:
            point = {
                "measurement": "agent_actions",
                "tags": {
                    "agent_id": agent_id,
                    "action_id": action_id,
                    "action_type": action_type,
                },
                "fields": {
                    "success": success,
                    "execution_time": execution_time,
                },
                "timestamp": timestamp or time.time(),
            }

            self.batch_queue.put({"type": "time_series", "data": point})

    def record_hierarchy_message(
        self,
        sender: str,
        receiver: str,
        message_type: str,
        priority: float,
        response_required: bool,
        timestamp: Optional[float] = None,
    ):
        """Record hierarchy communication."""
        if not self.config.auto_persist_messages:
            return

        if self.time_series:
            point = {
                "measurement": "hierarchy_messages",
                "tags": {
                    "sender": sender,
                    "receiver": receiver,
                    "message_type": message_type,
                },
                "fields": {
                    "priority": priority,
                    "response_required": response_required,
                },
                "timestamp": timestamp or time.time(),
            }

            self.batch_queue.put({"type": "time_series", "data": point})

    def record_hierarchy_change(
        self,
        agent_id: str,
        supervisor: Optional[str],
        change_type: str,
        timestamp: Optional[float] = None,
    ):
        """Record hierarchy structure change."""
        if self.graph_store and supervisor:
            # Create or update relationship
            if change_type == "assigned":
                self.graph_store.create_hierarchy_relationship(
                    agent_id, supervisor
                )

        # Also record in time series for analytics
        if self.time_series:
            point = {
                "measurement": "hierarchy_changes",
                "tags": {
                    "agent_id": agent_id,
                    "supervisor": supervisor or "none",
                    "change_type": change_type,
                },
                "fields": {"timestamp": timestamp or time.time()},
                "timestamp": timestamp or time.time(),
            }

            self.batch_queue.put({"type": "time_series", "data": point})

    def get_agent_timeline(
        self, agent_id: str, start_time: float, end_time: float
    ) -> Dict[str, List]:
        """Get comprehensive timeline for agent."""
        timeline = {"beliefs": [], "goals": [], "actions": [], "messages": []}

        if not self.time_series:
            return timeline

        # Query beliefs
        beliefs_query = (
            f'SELECT * FROM "agent_beliefs" WHERE "agent_id" = \'{agent_id}\' '
            f"AND time >= {int(start_time)}s AND time <= {int(end_time)}s"
        )
        timeline["beliefs"] = self.time_series.query(beliefs_query)

        # Query goals
        goals_query = (
            f'SELECT * FROM "agent_goals" WHERE "agent_id" = \'{agent_id}\' '
            f"AND time >= {int(start_time)}s AND time <= {int(end_time)}s"
        )
        timeline["goals"] = self.time_series.query(goals_query)

        # Query actions
        actions_query = (
            f'SELECT * FROM "agent_actions" WHERE "agent_id" = \'{agent_id}\' '
            f"AND time >= {int(start_time)}s AND time <= {int(end_time)}s"
        )
        timeline["actions"] = self.time_series.query(actions_query)

        # Query messages (sent and received)
        messages_query = (
            f'SELECT * FROM "hierarchy_messages" WHERE '
            f"(\"sender\" = '{agent_id}' OR \"receiver\" = '{agent_id}') "
            f"AND time >= {int(start_time)}s AND time <= {int(end_time)}s"
        )
        timeline["messages"] = self.time_series.query(messages_query)

        return timeline

    def get_system_metrics(self) -> Dict[str, Any]:
        """Get overall system metrics."""
        metrics = {
            "storage_health": self.health_monitor.get_health_status(),
            "backend_metrics": {},
            "system_stats": {},
        }

        # Get backend metrics
        if self.time_series:
            metrics["backend_metrics"][
                "time_series"
            ] = self.time_series.get_metrics().__dict__
        if self.document_store:
            metrics["backend_metrics"][
                "document"
            ] = self.document_store.get_metrics().__dict__
        if self.graph_store:
            metrics["backend_metrics"][
                "graph"
            ] = self.graph_store.get_metrics().__dict__
        if self.cache:
            metrics["backend_metrics"][
                "cache"
            ] = self.cache.get_metrics().__dict__

        # System-wide statistics
        if self.time_series:
            # Count total agents, messages, etc.
            agents_query = 'SELECT COUNT(DISTINCT("agent_id")) \
            FROM "agent_actions" WHERE time > now() - 24h'
            try:
                result = self.time_series.query(agents_query)
                if result:
                    metrics["system_stats"]["active_agents_24h"] = result[
                        0
                    ].get("count", 0)
            except:
                ...

        return metrics

    def cleanup_old_data(self):
        """Clean up old data based on retention policies."""
        if not self.time_series:
            return

        retention = self.config.retention
        current_time = time.time()

        # Clean up beliefs
        if retention.belief_retention_days > 0:
            cutoff_time = current_time - (
                retention.belief_retention_days * 24 * 3600
            )
            self.time_series.delete_series(
                "agent_beliefs", end_time=cutoff_time
            )

        # Clean up goals
        if retention.goal_retention_days > 0:
            cutoff_time = current_time - (
                retention.goal_retention_days * 24 * 3600
            )
            self.time_series.delete_series("agent_goals", end_time=cutoff_time)

        # Clean up messages
        if retention.message_retention_days > 0:
            cutoff_time = current_time - (
                retention.message_retention_days * 24 * 3600
            )
            self.time_series.delete_series(
                "hierarchy_messages", end_time=cutoff_time
            )

        logger.info("Completed data cleanup based on retention policies")

    def __enter__(self):
        """Context manager entry."""
        self.connect_all()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect_all()
