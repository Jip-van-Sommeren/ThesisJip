"""
Cache storage implementation for fast access to agent states and messages.
"""

import time
import logging
import json
from typing import Dict, List, Any, Optional
import redis

from .base_storage import CacheStorage, StorageStatus


logger = logging.getLogger(__name__)


class RedisStorage(CacheStorage):
    """Redis implementation of cache storage."""

    def __init__(self, config: Dict[str, Any]):

        super().__init__(config)
        self.client: Optional[redis.Redis] = None
        self.key_prefix = config.get("key_prefix", "agent_system:")

    def connect(self) -> bool:
        """Connect to Redis."""
        try:
            start_time = time.time()
            self.metrics.connection_status = StorageStatus.CONNECTING

            connection_kwargs = {
                "host": self.config.get("host", "localhost"),
                "port": self.config.get("port", 6379),
                "db": self.config.get("db", 0),
                "decode_responses": self.config.get("decode_responses", True),
                "socket_timeout": self.config.get("socket_timeout", 30),
                "socket_connect_timeout": self.config.get(
                    "socket_connect_timeout", 30
                ),
            }

            if self.config.get("password"):
                connection_kwargs["password"] = self.config["password"]

            if self.config.get("ssl", False):
                connection_kwargs["ssl"] = True
                connection_kwargs["ssl_cert_reqs"] = self.config.get(
                    "ssl_cert_reqs", "required"
                )

            self.client = redis.Redis(**connection_kwargs)

            # Test connection
            self.client.ping()

            self._connected = True
            self.metrics.connection_status = StorageStatus.CONNECTED
            self._record_operation("connect", start_time, True)

            logger.info(
                f"Connected to Redis:\
                    {connection_kwargs['host']}:{connection_kwargs['port']}"
            )
            return True

        except Exception as e:
            self._connected = False
            self.metrics.connection_status = StorageStatus.ERROR
            self._record_operation("connect", start_time, False)
            logger.error(f"Failed to connect to Redis: {e}")
            return False

    def disconnect(self) -> bool:
        """Disconnect from Redis."""
        try:
            if self.client:
                self.client.close()
                self.client = None

            self._connected = False
            self.metrics.connection_status = StorageStatus.DISCONNECTED
            logger.info("Disconnected from Redis")
            return True

        except Exception as e:
            logger.error(f"Error disconnecting from Redis: {e}")
            return False

    def health_check(self) -> bool:
        """Check Redis health."""
        try:
            if not self.client:
                return False

            start_time = time.time()
            self.client.ping()
            self._record_operation("health_check", start_time, True)
            return True

        except Exception as e:
            self._record_operation("health_check", time.time(), False)
            logger.warning(f"Redis health check failed: {e}")
            return False

    def _get_key(self, key: str) -> str:
        """Get prefixed key."""
        return f"{self.key_prefix}{key}"

    def _serialize_value(self, value: Any) -> str:
        """Serialize value for Redis storage."""
        if isinstance(value, (str, int, float, bool)):
            return str(value)
        else:
            return json.dumps(value, default=str)

    def _deserialize_value(self, value: str) -> Any:
        """Deserialize value from Redis storage."""
        try:
            return json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return value

    def get(self, key: str) -> Optional[Any]:
        """Get value by key."""
        if not self.client:
            logger.error("Redis client not connected")
            return None

        try:
            start_time = time.time()

            redis_key = self._get_key(key)
            value = self.client.get(redis_key)

            self._record_operation("get", start_time, True)

            if value is None:
                return None

            return self._deserialize_value(value)

        except Exception as e:
            self._record_operation("get", time.time(), False)
            logger.error(f"Error getting key from Redis: {e}")
            return None

    def set(self, key: str, value: Any, expire: Optional[int] = None) -> bool:
        """Set key-value pair with optional expiration."""
        if not self.client:
            logger.error("Redis client not connected")
            return False

        try:
            start_time = time.time()

            redis_key = self._get_key(key)
            serialized_value = self._serialize_value(value)

            if expire:
                success = self.client.setex(
                    redis_key, expire, serialized_value
                )
            else:
                success = self.client.set(redis_key, serialized_value)

            self._record_operation("set", start_time, success)

            if not success:
                logger.error(f"Failed to set key in Redis: {key}")

            return success

        except Exception as e:
            self._record_operation("set", time.time(), False)
            logger.error(f"Error setting key in Redis: {e}")
            return False

    def delete(self, key: str) -> bool:
        """Delete key."""
        if not self.client:
            logger.error("Redis client not connected")
            return False

        try:
            start_time = time.time()

            redis_key = self._get_key(key)
            deleted_count = self.client.delete(redis_key)

            success = deleted_count > 0
            self._record_operation("delete", start_time, success)

            return success

        except Exception as e:
            self._record_operation("delete", time.time(), False)
            logger.error(f"Error deleting key from Redis: {e}")
            return False

    def exists(self, key: str) -> bool:
        """Check if key exists."""
        if not self.client:
            logger.error("Redis client not connected")
            return False

        try:
            start_time = time.time()

            redis_key = self._get_key(key)
            exists = self.client.exists(redis_key)

            self._record_operation("exists", start_time, True)
            return bool(exists)

        except Exception as e:
            self._record_operation("exists", time.time(), False)
            logger.error(f"Error checking key existence in Redis: {e}")
            return False

    def expire(self, key: str, seconds: int) -> bool:
        """Set expiration on key."""
        if not self.client:
            logger.error("Redis client not connected")
            return False

        try:
            start_time = time.time()

            redis_key = self._get_key(key)
            success = self.client.expire(redis_key, seconds)

            self._record_operation("expire", start_time, success)
            return success

        except Exception as e:
            self._record_operation("expire", time.time(), False)
            logger.error(f"Error setting expiration on Redis key: {e}")
            return False

    def increment(self, key: str, amount: int = 1) -> int:
        """Increment numeric value."""
        if not self.client:
            logger.error("Redis client not connected")
            return 0

        try:
            start_time = time.time()

            redis_key = self._get_key(key)
            result = self.client.incr(redis_key, amount)

            self._record_operation("increment", start_time, True)
            return result

        except Exception as e:
            self._record_operation("increment", time.time(), False)
            logger.error(f"Error incrementing Redis key: {e}")
            return 0

    def get_many(self, keys: List[str]) -> Dict[str, Any]:
        """Get multiple values."""
        if not self.client:
            logger.error("Redis client not connected")
            return {}

        try:
            start_time = time.time()

            redis_keys = [self._get_key(key) for key in keys]
            values = self.client.mget(redis_keys)

            result = {}
            for i, key in enumerate(keys):
                if values[i] is not None:
                    result[key] = self._deserialize_value(values[i])

            self._record_operation("get_many", start_time, True)
            return result

        except Exception as e:
            self._record_operation("get_many", time.time(), False)
            logger.error(f"Error getting multiple keys from Redis: {e}")
            return {}

    def set_many(
        self, mapping: Dict[str, Any], expire: Optional[int] = None
    ) -> bool:
        """Set multiple key-value pairs."""
        if not self.client:
            logger.error("Redis client not connected")
            return False

        try:
            start_time = time.time()

            # Prepare mapping with prefixed keys and serialized values
            redis_mapping = {}
            for key, value in mapping.items():
                redis_key = self._get_key(key)
                serialized_value = self._serialize_value(value)
                redis_mapping[redis_key] = serialized_value

            # Use pipeline for better performance
            pipe = self.client.pipeline()
            pipe.mset(redis_mapping)

            # Set expiration if provided
            if expire:
                for redis_key in redis_mapping.keys():
                    pipe.expire(redis_key, expire)

            results = pipe.execute()
            success = all(results)

            self._record_operation("set_many", start_time, success)
            return success

        except Exception as e:
            self._record_operation("set_many", time.time(), False)
            logger.error(f"Error setting multiple keys in Redis: {e}")
            return False

    def flush_all(self) -> bool:
        """Clear all data."""
        if not self.client:
            logger.error("Redis client not connected")
            return False

        try:
            start_time = time.time()

            # Only flush keys with our prefix
            pattern = f"{self.key_prefix}*"
            keys = self.client.keys(pattern)

            if keys:
                self.client.delete(*keys)

            self._record_operation("flush_all", start_time, True)
            logger.info(f"Flushed {len(keys)} keys from Redis")

            return True

        except Exception as e:
            self._record_operation("flush_all", time.time(), False)
            logger.error(f"Error flushing Redis: {e}")
            return False

    # Agent-specific cache methods

    def cache_agent_state(
        self, agent_id: str, state_data: Dict[str, Any], expire: int = 300
    ) -> bool:
        """Cache agent state with expiration."""
        key = f"agent_state:{agent_id}"
        return self.set(key, state_data, expire)

    def get_cached_agent_state(self, agent_id: str) -> Optional[Dict]:
        """Get cached agent state."""
        key = f"agent_state:{agent_id}"
        return self.get(key)

    def cache_message_queue(
        self, agent_id: str, messages: List[Dict], expire: int = 600
    ) -> bool:
        """Cache agent message queue."""
        key = f"message_queue:{agent_id}"
        return self.set(key, messages, expire)

    def get_cached_message_queue(self, agent_id: str) -> List[Dict]:
        """Get cached message queue."""
        key = f"message_queue:{agent_id}"
        messages = self.get(key)
        return messages if messages else []

    def increment_message_counter(self, agent_id: str) -> int:
        """Increment message counter for agent."""
        key = f"message_count:{agent_id}"
        return self.increment(key)

    def cache_hierarchy_position(
        self, agent_id: str, position_data: Dict[str, Any], expire: int = 3600
    ) -> bool:
        """Cache agent hierarchy position."""
        key = f"hierarchy_position:{agent_id}"
        return self.set(key, position_data, expire)

    def get_cached_hierarchy_position(self, agent_id: str) -> Optional[Dict]:
        """Get cached hierarchy position."""
        key = f"hierarchy_position:{agent_id}"
        return self.get(key)
