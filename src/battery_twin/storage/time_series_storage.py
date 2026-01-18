"""
Time series storage implementation for agent metrics and historical data.
"""

import time
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone
from influxdb import InfluxDBClient

from .base_storage import TimeSeriesStorage, StorageStatus


logger = logging.getLogger(__name__)


class InfluxDBStorage(TimeSeriesStorage):
    """InfluxDB implementation of time series storage."""

    def __init__(self, config: Dict[str, Any]):

        super().__init__(config)
        self.client: Optional[InfluxDBClient] = None
        self.database = config.get("database", "agent_metrics")

    def connect(self) -> bool:
        """Connect to InfluxDB."""
        try:
            start_time = time.time()
            self.metrics.connection_status = StorageStatus.CONNECTING

            self.client = InfluxDBClient(
                host=self.config.get("host", "localhost"),
                port=self.config.get("port", 8086),
                username=self.config.get("username"),
                password=self.config.get("password"),
                database=self.database,
                ssl=self.config.get("ssl", False),
                verify_ssl=self.config.get("verify_ssl", False),
                timeout=self.config.get("timeout", 30),
            )

            # Test connection
            self.client.ping()

            # Create database if not exists
            self.create_database(self.database)

            self._connected = True
            self.metrics.connection_status = StorageStatus.CONNECTED
            self._record_operation("connect", start_time, True)

            logger.info(
                f"Connected to InfluxDB: \
                    {self.config.get('host')}:{self.config.get('port')}"
            )
            return True

        except Exception as e:
            self._connected = False
            self.metrics.connection_status = StorageStatus.ERROR
            self._record_operation("connect", start_time, False)
            logger.error(f"Failed to connect to InfluxDB: {e}")
            return False

    def disconnect(self) -> bool:
        """Disconnect from InfluxDB."""
        try:
            if self.client:
                self.client.close()
                self.client = None

            self._connected = False
            self.metrics.connection_status = StorageStatus.DISCONNECTED
            logger.info("Disconnected from InfluxDB")
            return True

        except Exception as e:
            logger.error(f"Error disconnecting from InfluxDB: {e}")
            return False

    def health_check(self) -> bool:
        """Check InfluxDB health."""
        try:
            if not self.client:
                return False

            start_time = time.time()
            self.client.ping()
            self._record_operation("health_check", start_time, True)
            return True

        except Exception as e:
            self._record_operation("health_check", time.time(), False)
            logger.warning(f"InfluxDB health check failed: {e}")
            return False

    def write_point(
        self,
        measurement: str,
        tags: Dict[str, str],
        fields: Dict[str, Any],
        timestamp: Optional[float] = None,
    ) -> bool:
        """Write single data point to InfluxDB."""
        if not self.client:
            logger.error("InfluxDB client not connected")
            return False

        try:
            start_time = time.time()

            # Convert timestamp if provided
            if timestamp:
                time_str = datetime.fromtimestamp(
                    timestamp, tz=timezone.utc
                ).isoformat()
            else:
                time_str = datetime.utcnow().isoformat() + "Z"

            point = {
                "measurement": measurement,
                "tags": tags,
                "fields": fields,
                "time": time_str,
            }

            success = self.client.write_points([point])
            self._record_operation("write_point", start_time, success)

            if not success:
                logger.error(
                    f"Failed to write point to measurement: {measurement}"
                )

            return success

        except Exception as e:
            self._record_operation("write_point", time.time(), False)
            logger.error(f"Error writing point to InfluxDB: {e}")
            return False

    def write_points(self, points: List[Dict[str, Any]]) -> bool:
        """Write multiple data points to InfluxDB."""
        if not self.client:
            logger.error("InfluxDB client not connected")
            return False

        try:
            start_time = time.time()

            # Process points to ensure proper format
            formatted_points = []
            for point in points:
                if "time" not in point and "timestamp" in point:
                    timestamp = point.pop("timestamp")
                    point["time"] = datetime.fromtimestamp(
                        timestamp, tz=timezone.utc
                    ).isoformat()
                elif "time" not in point:
                    point["time"] = datetime.utcnow().isoformat() + "Z"

                formatted_points.append(point)

            success = self.client.write_points(formatted_points)
            self._record_operation("write_points", start_time, success)

            if not success:
                logger.error(
                    f"Failed to write {len(points)} points to InfluxDB"
                )
            else:
                logger.debug(
                    f"Successfully wrote {len(points)} points to InfluxDB"
                )

            return success

        except Exception as e:
            self._record_operation("write_points", time.time(), False)
            logger.error(f"Error writing points to InfluxDB: {e}")
            return False

    def query(
        self, query_string: str, database: Optional[str] = None
    ) -> List[Dict]:
        """Execute query and return results."""
        if not self.client:
            logger.error("InfluxDB client not connected")
            return []

        try:
            start_time = time.time()

            # Switch database if specified
            if database and database != self.database:
                self.client.switch_database(database)

            result = self.client.query(query_string)

            # Convert result to list of dictionaries
            points = []
            for series in result:
                for point in series:
                    points.append(dict(point))

            self._record_operation("query", start_time, True)
            logger.debug(f"Query returned {len(points)} points")

            return points

        except Exception as e:
            self._record_operation("query", time.time(), False)
            logger.error(f"Error executing InfluxDB query: {e}")
            return []

    def create_database(self, database: str) -> bool:
        """Create database if not exists."""
        if not self.client:
            logger.error("InfluxDB client not connected")
            return False

        try:
            start_time = time.time()

            # Check if database exists
            databases = self.client.get_list_database()
            if any(db["name"] == database for db in databases):
                self._record_operation("create_database", start_time, True)
                return True

            # Create database
            self.client.create_database(database)
            self._record_operation("create_database", start_time, True)

            logger.info(f"Created InfluxDB database: {database}")
            return True

        except Exception as e:
            self._record_operation("create_database", time.time(), False)
            logger.error(f"Error creating InfluxDB database {database}: {e}")
            return False

    def delete_series(
        self,
        measurement: str,
        tags: Optional[Dict[str, str]] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
    ) -> bool:
        """Delete time series data."""
        if not self.client:
            logger.error("InfluxDB client not connected")
            return False

        try:
            operation_start = time.time()

            # Build WHERE clause
            where_conditions = []

            if tags:
                for key, value in tags.items():
                    where_conditions.append(f"\"{key}\" = '{value}'")

            if start_time:
                start_iso = datetime.fromtimestamp(
                    start_time, tz=timezone.utc
                ).isoformat()
                where_conditions.append(f"time >= '{start_iso}'")

            if end_time:
                end_iso = datetime.fromtimestamp(
                    end_time, tz=timezone.utc
                ).isoformat()
                where_conditions.append(f"time <= '{end_iso}'")

            where_clause = (
                " AND ".join(where_conditions) if where_conditions else ""
            )

            # Build DELETE query
            if where_clause:
                query = f'DELETE FROM "{measurement}" WHERE {where_clause}'
            else:
                query = f'DELETE FROM "{measurement}"'

            _ = self.client.query(query)
            self._record_operation("delete_series", operation_start, True)

            logger.info(f"Deleted series from measurement: {measurement}")
            return True

        except Exception as e:
            self._record_operation("delete_series", time.time(), False)
            logger.error(f"Error deleting series from InfluxDB: {e}")
            return False

    def record_agent_belief(
        self,
        agent_id: str,
        belief_key: str,
        proposition: str,
        confidence: float,
        timestamp: Optional[float] = None,
    ) -> bool:
        """Record agent belief update."""
        return self.write_point(
            measurement="agent_beliefs",
            tags={"agent_id": agent_id, "belief_key": belief_key},
            fields={"proposition": proposition, "confidence": confidence},
            timestamp=timestamp,
        )

    def record_agent_goal(
        self,
        agent_id: str,
        goal_id: str,
        event_type: str,
        priority: float,
        goal_type: str,
        timestamp: Optional[float] = None,
    ) -> bool:
        """Record agent goal event."""
        return self.write_point(
            measurement="agent_goals",
            tags={
                "agent_id": agent_id,
                "goal_id": goal_id,
                "event_type": event_type,
                "goal_type": goal_type,
            },
            fields={"priority": priority},
            timestamp=timestamp,
        )

    def record_agent_action(
        self,
        agent_id: str,
        action_id: str,
        action_type: str,
        success: bool,
        execution_time: float,
        timestamp: Optional[float] = None,
    ) -> bool:
        """Record agent action execution."""
        return self.write_point(
            measurement="agent_actions",
            tags={
                "agent_id": agent_id,
                "action_id": action_id,
                "action_type": action_type,
            },
            fields={"success": success, "execution_time": execution_time},
            timestamp=timestamp,
        )

    def record_hierarchy_message(
        self,
        sender: str,
        receiver: str,
        message_type: str,
        priority: float,
        response_required: bool,
        timestamp: Optional[float] = None,
    ) -> bool:
        """Record hierarchy communication."""
        return self.write_point(
            measurement="hierarchy_messages",
            tags={
                "sender": sender,
                "receiver": receiver,
                "message_type": message_type,
            },
            fields={
                "priority": priority,
                "response_required": response_required,
            },
            timestamp=timestamp,
        )
