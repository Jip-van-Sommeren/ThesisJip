"""
Document storage implementation for agent profiles and configurations.
"""

import time
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from pymongo import MongoClient, ASCENDING, DESCENDING

from .base_storage import DocumentStorage, StorageStatus


logger = logging.getLogger(__name__)


class MongoDBStorage(DocumentStorage):
    """MongoDB implementation of document storage."""

    def __init__(self, config: Dict[str, Any]):

        super().__init__(config)
        self.client: Optional[MongoClient] = None
        self.database = None
        self.database_name = config.get("database", "agent_system")

    def connect(self) -> bool:
        """Connect to MongoDB."""
        try:
            start_time = time.time()
            self.metrics.connection_status = StorageStatus.CONNECTING

            # Build connection string
            host = self.config.get("host", "localhost")
            port = self.config.get("port", 27017)
            username = self.config.get("username")
            password = self.config.get("password")
            auth_source = self.config.get("auth_source", "admin")

            if username and password:
                connection_string = (
                    f"mongodb://{username}:{password}@{host}:{port}/"
                    f"{self.database_name}?authSource={auth_source}"
                )
            else:
                connection_string = (
                    f"mongodb://{host}:{port}/{self.database_name}"
                )

            self.client = MongoClient(
                connection_string,
                serverSelectionTimeoutMS=self.config.get(
                    "server_selection_timeout", 30000
                ),
                connectTimeoutMS=self.config.get("connection_timeout", 30000),
            )

            # Test connection
            self.client.admin.command("ping")

            # Get database
            self.database = self.client[self.database_name]

            self._connected = True
            self.metrics.connection_status = StorageStatus.CONNECTED
            self._record_operation("connect", start_time, True)

            logger.info(
                f"Connected to MongoDB: {host}:{port}/{self.database_name}"
            )
            return True

        except Exception as e:
            self._connected = False
            self.metrics.connection_status = StorageStatus.ERROR
            self._record_operation("connect", start_time, False)
            logger.error(f"Failed to connect to MongoDB: {e}")
            return False

    def disconnect(self) -> bool:
        """Disconnect from MongoDB."""
        try:
            if self.client:
                self.client.close()
                self.client = None
                self.database = None

            self._connected = False
            self.metrics.connection_status = StorageStatus.DISCONNECTED
            logger.info("Disconnected from MongoDB")
            return True

        except Exception as e:
            logger.error(f"Error disconnecting from MongoDB: {e}")
            return False

    def health_check(self) -> bool:
        """Check MongoDB health."""
        try:
            if not self.client or not self.database:
                return False

            start_time = time.time()
            self.client.admin.command("ping")
            self._record_operation("health_check", start_time, True)
            return True

        except Exception as e:
            self._record_operation("health_check", time.time(), False)
            logger.warning(f"MongoDB health check failed: {e}")
            return False

    def insert_one(self, collection: str, document: Dict[str, Any]) -> str:
        """Insert single document."""
        if not self.database:
            logger.error("MongoDB not connected")
            return ""

        try:
            start_time = time.time()

            # Add timestamp if not present
            if "created_at" not in document:
                document["created_at"] = datetime.utcnow()

            collection_obj = self.database[collection]
            result = collection_obj.insert_one(document)

            self._record_operation("insert_one", start_time, True)
            logger.debug(
                f"Inserted document in collection {collection}:\
                    {result.inserted_id}"
            )

            return str(result.inserted_id)

        except Exception as e:
            self._record_operation("insert_one", time.time(), False)
            logger.error(f"Error inserting document in MongoDB: {e}")
            return ""

    def insert_many(
        self, collection: str, documents: List[Dict[str, Any]]
    ) -> List[str]:
        """Insert multiple documents."""
        if not self.database:
            logger.error("MongoDB not connected")
            return []

        try:
            start_time = time.time()

            # Add timestamps if not present
            for doc in documents:
                if "created_at" not in doc:
                    doc["created_at"] = datetime.utcnow()

            collection_obj = self.database[collection]
            result = collection_obj.insert_many(documents)

            self._record_operation("insert_many", start_time, True)
            logger.debug(
                f"Inserted {len(documents)} documents in \
                    collection {collection}"
            )

            return [str(oid) for oid in result.inserted_ids]

        except Exception as e:
            self._record_operation("insert_many", time.time(), False)
            logger.error(f"Error inserting documents in MongoDB: {e}")
            return []

    def find_one(
        self, collection: str, filter_dict: Dict[str, Any]
    ) -> Optional[Dict]:
        """Find single document."""
        if not self.database:
            logger.error("MongoDB not connected")
            return None

        try:
            start_time = time.time()

            collection_obj = self.database[collection]
            result = collection_obj.find_one(filter_dict)

            self._record_operation("find_one", start_time, True)

            if result:
                # Convert ObjectId to string
                result["_id"] = str(result["_id"])

            return result

        except Exception as e:
            self._record_operation("find_one", time.time(), False)
            logger.error(f"Error finding document in MongoDB: {e}")
            return None

    def find_many(
        self,
        collection: str,
        filter_dict: Dict[str, Any],
        limit: Optional[int] = None,
        skip: Optional[int] = None,
        sort: Optional[List[Tuple[str, int]]] = None,
    ) -> List[Dict]:
        """Find multiple documents."""
        if not self.database:
            logger.error("MongoDB not connected")
            return []

        try:
            start_time = time.time()

            collection_obj = self.database[collection]
            cursor = collection_obj.find(filter_dict)

            if skip:
                cursor = cursor.skip(skip)
            if limit:
                cursor = cursor.limit(limit)
            if sort:
                cursor = cursor.sort(sort)

            results = []
            for doc in cursor:
                doc["_id"] = str(doc["_id"])
                results.append(doc)

            self._record_operation("find_many", start_time, True)
            logger.debug(
                f"Found {len(results)} documents in collection {collection}"
            )

            return results

        except Exception as e:
            self._record_operation("find_many", time.time(), False)
            logger.error(f"Error finding documents in MongoDB: {e}")
            return []

    def update_one(
        self,
        collection: str,
        filter_dict: Dict[str, Any],
        update: Dict[str, Any],
    ) -> bool:
        """Update single document."""
        if not self.database:
            logger.error("MongoDB not connected")
            return False

        try:
            start_time = time.time()

            # Add updated timestamp
            if "$set" not in update:
                update = {"$set": update}
            if "$set" in update:
                update["$set"]["updated_at"] = datetime.utcnow()

            collection_obj = self.database[collection]
            result = collection_obj.update_one(filter_dict, update)

            success = result.modified_count > 0
            self._record_operation("update_one", start_time, success)

            if success:
                logger.debug(f"Updated document in collection {collection}")
            else:
                logger.warning(
                    f"No document updated in collection {collection}"
                )

            return success

        except Exception as e:
            self._record_operation("update_one", time.time(), False)
            logger.error(f"Error updating document in MongoDB: {e}")
            return False

    def update_many(
        self,
        collection: str,
        filter_dict: Dict[str, Any],
        update: Dict[str, Any],
    ) -> int:
        """Update multiple documents."""
        if not self.database:
            logger.error("MongoDB not connected")
            return 0

        try:
            start_time = time.time()

            # Add updated timestamp
            if "$set" not in update:
                update = {"$set": update}
            if "$set" in update:
                update["$set"]["updated_at"] = datetime.utcnow()

            collection_obj = self.database[collection]
            result = collection_obj.update_many(filter_dict, update)

            modified_count = result.modified_count
            self._record_operation(
                "update_many", start_time, modified_count > 0
            )

            logger.debug(
                f"Updated {modified_count} documents in\
                    collection {collection}"
            )
            return modified_count

        except Exception as e:
            self._record_operation("update_many", time.time(), False)
            logger.error(f"Error updating documents in MongoDB: {e}")
            return 0

    def delete_one(self, collection: str, filter_dict: Dict[str, Any]) -> bool:
        """Delete single document."""
        if not self.database:
            logger.error("MongoDB not connected")
            return False

        try:
            start_time = time.time()

            collection_obj = self.database[collection]
            result = collection_obj.delete_one(filter_dict)

            success = result.deleted_count > 0
            self._record_operation("delete_one", start_time, success)

            if success:
                logger.debug(f"Deleted document from collection {collection}")

            return success

        except Exception as e:
            self._record_operation("delete_one", time.time(), False)
            logger.error(f"Error deleting document from MongoDB: {e}")
            return False

    def delete_many(self, collection: str, filter_dict: Dict[str, Any]) -> int:
        """Delete multiple documents."""
        if not self.database:
            logger.error("MongoDB not connected")
            return 0

        try:
            start_time = time.time()

            collection_obj = self.database[collection]
            result = collection_obj.delete_many(filter_dict)

            deleted_count = result.deleted_count
            self._record_operation(
                "delete_many", start_time, deleted_count > 0
            )

            logger.debug(
                f"Deleted {deleted_count} documents from\
                    collection {collection}"
            )
            return deleted_count

        except Exception as e:
            self._record_operation("delete_many", time.time(), False)
            logger.error(f"Error deleting documents from MongoDB: {e}")
            return 0

    def create_index(
        self, collection: str, keys: List[Tuple[str, int]]
    ) -> bool:
        """Create index on collection."""
        if not self.database:
            logger.error("MongoDB not connected")
            return False

        try:
            start_time = time.time()

            collection_obj = self.database[collection]
            result = collection_obj.create_index(keys)

            self._record_operation("create_index", start_time, True)
            logger.info(f"Created index on collection {collection}: {result}")

            return True

        except Exception as e:
            self._record_operation("create_index", time.time(), False)
            logger.error(f"Error creating index in MongoDB: {e}")
            return False

    def store_agent_profile(self, agent_profile: Dict[str, Any]) -> str:
        """Store agent profile."""
        return self.insert_one("agent_profiles", agent_profile)

    def get_agent_profile(self, agent_id: str) -> Optional[Dict]:
        """Get agent profile by ID."""
        return self.find_one("agent_profiles", {"agent_id": agent_id})

    def store_role_definition(self, role_data: Dict[str, Any]) -> str:
        """Store role definition."""
        return self.insert_one("role_definitions", role_data)

    def get_role_definition(self, role_name: str) -> Optional[Dict]:
        """Get role definition by name."""
        return self.find_one("role_definitions", {"name": role_name})

    def store_group_configuration(self, group_data: Dict[str, Any]) -> str:
        """Store group configuration."""
        return self.insert_one("group_configurations", group_data)

    def get_group_configuration(self, group_name: str) -> Optional[Dict]:
        """Get group configuration by name."""
        return self.find_one("group_configurations", {"name": group_name})

    def setup_indexes(self):
        """Set up common indexes."""
        indexes = [
            ("agent_profiles", [("agent_id", ASCENDING)]),
            ("role_definitions", [("name", ASCENDING)]),
            ("group_configurations", [("name", ASCENDING)]),
            ("agent_profiles", [("created_at", DESCENDING)]),
        ]

        for collection, keys in indexes:
            self.create_index(collection, keys)
