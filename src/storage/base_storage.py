"""
Base storage interfaces and abstract classes.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple
import time
from dataclasses import dataclass
from enum import Enum


class StorageStatus(Enum):
    """Storage connection status."""

    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    ERROR = "error"
    CONNECTING = "connecting"


@dataclass
class StorageMetrics:
    """Storage performance metrics."""

    operations_count: int = 0
    total_latency: float = 0.0
    error_count: int = 0
    last_operation_time: float = 0.0
    connection_status: StorageStatus = StorageStatus.DISCONNECTED

    @property
    def average_latency(self) -> float:
        return self.total_latency / max(1, self.operations_count)

    def record_operation(self, latency: float, success: bool = True):
        """Record operation metrics."""
        self.operations_count += 1
        self.total_latency += latency
        self.last_operation_time = time.time()
        if not success:
            self.error_count += 1


class BaseStorage(ABC):
    """Abstract base class for all storage backends."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.metrics = StorageMetrics()
        self._connected = False

    @abstractmethod
    def connect(self) -> bool:
        """Establish connection to storage backend."""
        pass

    @abstractmethod
    def disconnect(self) -> bool:
        """Close connection to storage backend."""
        pass

    @abstractmethod
    def health_check(self) -> bool:
        """Check if storage backend is healthy."""
        pass

    def is_connected(self) -> bool:
        """Check if connected to storage backend."""
        return self._connected

    def get_metrics(self) -> StorageMetrics:
        """Get storage performance metrics."""
        return self.metrics

    def _record_operation(
        self, operation_name: str, start_time: float, success: bool = True
    ):
        """Record operation for metrics."""
        latency = time.time() - start_time
        self.metrics.record_operation(latency, success)


class TimeSeriesStorage(BaseStorage):
    """Abstract interface for time series storage backends."""

    @abstractmethod
    def write_point(
        self,
        measurement: str,
        tags: Dict[str, str],
        fields: Dict[str, Any],
        timestamp: Optional[float] = None,
    ) -> bool:
        """Write a single data point."""
        pass

    @abstractmethod
    def write_points(self, points: List[Dict[str, Any]]) -> bool:
        """Write multiple data points."""
        pass

    @abstractmethod
    def query(
        self, query_string: str, database: Optional[str] = None
    ) -> List[Dict]:
        """Execute query and return results."""
        pass

    @abstractmethod
    def create_database(self, database: str) -> bool:
        """Create database if not exists."""
        pass

    @abstractmethod
    def delete_series(
        self,
        measurement: str,
        tags: Optional[Dict[str, str]] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
    ) -> bool:
        """Delete time series data."""
        pass


class DocumentStorage(BaseStorage):
    """Abstract interface for document storage backends."""

    @abstractmethod
    def insert_one(self, collection: str, document: Dict[str, Any]) -> str:
        """Insert single document."""
        pass

    @abstractmethod
    def insert_many(
        self, collection: str, documents: List[Dict[str, Any]]
    ) -> List[str]:
        """Insert multiple documents."""
        pass

    @abstractmethod
    def find_one(
        self, collection: str, filter_dict: Dict[str, Any]
    ) -> Optional[Dict]:
        """Find single document."""
        pass

    @abstractmethod
    def find_many(
        self,
        collection: str,
        filter_dict: Dict[str, Any],
        limit: Optional[int] = None,
        skip: Optional[int] = None,
        sort: Optional[List[Tuple[str, int]]] = None,
    ) -> List[Dict]:
        """Find multiple documents."""
        pass

    @abstractmethod
    def update_one(
        self,
        collection: str,
        filter_dict: Dict[str, Any],
        update: Dict[str, Any],
    ) -> bool:
        """Update single document."""
        pass

    @abstractmethod
    def update_many(
        self,
        collection: str,
        filter_dict: Dict[str, Any],
        update: Dict[str, Any],
    ) -> int:
        """Update multiple documents."""
        pass

    @abstractmethod
    def delete_one(self, collection: str, filter_dict: Dict[str, Any]) -> bool:
        """Delete single document."""
        pass

    @abstractmethod
    def delete_many(self, collection: str, filter_dict: Dict[str, Any]) -> int:
        """Delete multiple documents."""
        pass

    @abstractmethod
    def create_index(
        self, collection: str, keys: List[Tuple[str, int]]
    ) -> bool:
        """Create index on collection."""
        pass


class GraphStorage(BaseStorage):
    """Abstract interface for graph storage backends."""

    @abstractmethod
    def create_node(self, label: str, properties: Dict[str, Any]) -> str:
        """Create node with properties."""
        pass

    @abstractmethod
    def create_relationship(
        self,
        from_node_id: str,
        to_node_id: str,
        relationship_type: str,
        properties: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Create relationship between nodes."""
        pass

    @abstractmethod
    def find_node(
        self, label: str, properties: Dict[str, Any]
    ) -> Optional[Dict]:
        """Find single node by properties."""
        pass

    @abstractmethod
    def find_nodes(
        self,
        label: str,
        properties: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None,
    ) -> List[Dict]:
        """Find multiple nodes."""
        pass

    @abstractmethod
    def update_node(self, node_id: str, properties: Dict[str, Any]) -> bool:
        """Update node properties."""
        pass

    @abstractmethod
    def delete_node(self, node_id: str) -> bool:
        """Delete node and its relationships."""
        pass

    @abstractmethod
    def execute_query(
        self, query: str, parameters: Optional[Dict[str, Any]] = None
    ) -> List[Dict]:
        """Execute custom query."""
        pass

    @abstractmethod
    def get_shortest_path(
        self, from_node_id: str, to_node_id: str
    ) -> List[Dict]:
        """Find shortest path between nodes."""
        pass


class CacheStorage(BaseStorage):
    """Abstract interface for cache storage backends."""

    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        """Get value by key."""
        pass

    @abstractmethod
    def set(self, key: str, value: Any, expire: Optional[int] = None) -> bool:
        """Set key-value pair with optional expiration."""
        pass

    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete key."""
        pass

    @abstractmethod
    def exists(self, key: str) -> bool:
        """Check if key exists."""
        pass

    @abstractmethod
    def expire(self, key: str, seconds: int) -> bool:
        """Set expiration on key."""
        pass

    @abstractmethod
    def increment(self, key: str, amount: int = 1) -> int:
        """Increment numeric value."""
        pass

    @abstractmethod
    def get_many(self, keys: List[str]) -> Dict[str, Any]:
        """Get multiple values."""
        pass

    @abstractmethod
    def set_many(
        self, mapping: Dict[str, Any], expire: Optional[int] = None
    ) -> bool:
        """Set multiple key-value pairs."""
        pass

    @abstractmethod
    def flush_all(self) -> bool:
        """Clear all data."""
        pass
