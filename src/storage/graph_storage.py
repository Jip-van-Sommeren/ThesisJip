"""
Graph storage implementation for hierarchy relationships and network analysis.
"""

import time
import logging
from typing import Dict, List, Any, Optional
from neo4j import GraphDatabase, basic_auth

from .base_storage import GraphStorage, StorageStatus


logger = logging.getLogger(__name__)


class Neo4jStorage(GraphStorage):
    """Neo4j implementation of graph storage."""

    def __init__(self, config: Dict[str, Any]):

        super().__init__(config)
        self.driver = None
        self.database_name = config.get("database", "agent_hierarchy")

    def connect(self) -> bool:
        """Connect to Neo4j."""
        try:
            start_time = time.time()
            self.metrics.connection_status = StorageStatus.CONNECTING

            uri = self.config.get("uri", "bolt://localhost:7687")
            username = self.config.get("username", "neo4j")
            password = self.config.get("password", "password")

            self.driver = GraphDatabase.driver(
                uri,
                auth=basic_auth(username, password),
                encrypted=self.config.get("encrypted", False),
                trust=self.config.get(
                    "trust", "TRUST_SYSTEM_CA_SIGNED_CERTIFICATES"
                ),
                max_connection_lifetime=self.config.get(
                    "max_connection_lifetime", 3600
                ),
                max_connection_pool_size=self.config.get(
                    "max_connection_pool_size", 50
                ),
                connection_acquisition_timeout=self.config.get(
                    "connection_acquisition_timeout", 60
                ),
            )

            # Test connection
            with self.driver.session(database=self.database_name) as session:
                session.run("RETURN 1")

            self._connected = True
            self.metrics.connection_status = StorageStatus.CONNECTED
            self._record_operation("connect", start_time, True)

            logger.info(f"Connected to Neo4j: {uri}")
            return True

        except Exception as e:
            self._connected = False
            self.metrics.connection_status = StorageStatus.ERROR
            self._record_operation("connect", start_time, False)
            logger.error(f"Failed to connect to Neo4j: {e}")
            return False

    def disconnect(self) -> bool:
        """Disconnect from Neo4j."""
        try:
            if self.driver:
                self.driver.close()
                self.driver = None

            self._connected = False
            self.metrics.connection_status = StorageStatus.DISCONNECTED
            logger.info("Disconnected from Neo4j")
            return True

        except Exception as e:
            logger.error(f"Error disconnecting from Neo4j: {e}")
            return False

    def health_check(self) -> bool:
        """Check Neo4j health."""
        try:
            if not self.driver:
                return False

            start_time = time.time()
            with self.driver.session(database=self.database_name) as session:
                session.run("RETURN 1")

            self._record_operation("health_check", start_time, True)
            return True

        except Exception as e:
            self._record_operation("health_check", time.time(), False)
            logger.warning(f"Neo4j health check failed: {e}")
            return False

    def create_node(self, label: str, properties: Dict[str, Any]) -> str:
        """Create node with properties."""
        if not self.driver:
            logger.error("Neo4j driver not connected")
            return ""

        try:
            start_time = time.time()

            with self.driver.session(database=self.database_name) as session:
                # Add creation timestamp
                properties["created_at"] = time.time()

                # Build property string
                prop_items = []
                for key, value in properties.items():
                    if isinstance(value, str):
                        prop_items.append(f"{key}: '{value}'")
                    else:
                        prop_items.append(f"{key}: {value}")

                prop_string = ", ".join(prop_items)

                query = f"CREATE (n:{label} {{{prop_string}}}) RETURN id(n) AS\
                    node_id"
                result = session.run(query)

                node_id = result.single()["node_id"]

                self._record_operation("create_node", start_time, True)
                logger.debug(f"Created node {label} with ID: {node_id}")

                return str(node_id)

        except Exception as e:
            self._record_operation("create_node", time.time(), False)
            logger.error(f"Error creating node in Neo4j: {e}")
            return ""

    def create_relationship(
        self,
        from_node_id: str,
        to_node_id: str,
        relationship_type: str,
        properties: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Create relationship between nodes."""
        if not self.driver:
            logger.error("Neo4j driver not connected")
            return ""

        try:
            start_time = time.time()

            with self.driver.session(database=self.database_name) as session:
                # Add creation timestamp to relationship properties
                if properties is None:
                    properties = {}
                properties["created_at"] = time.time()

                # Build property string
                if properties:
                    prop_items = []
                    for key, value in properties.items():
                        if isinstance(value, str):
                            prop_items.append(f"{key}: '{value}'")
                        else:
                            prop_items.append(f"{key}: {value}")

                    prop_string = "{" + ", ".join(prop_items) + "}"
                else:
                    prop_string = ""

                query = (
                    f"MATCH (a), (b) WHERE id(a) = {from_node_id} AND id(b)\
                        = {to_node_id} "
                    f"CREATE (a)-[r:{relationship_type} {prop_string}]->(b) "
                    f"RETURN id(r) AS rel_id"
                )

                result = session.run(query)
                rel_id = result.single()["rel_id"]

                self._record_operation("create_relationship", start_time, True)
                logger.debug(
                    f"Created relationship {relationship_type} with ID:\
                        {rel_id}"
                )

                return str(rel_id)

        except Exception as e:
            self._record_operation("create_relationship", time.time(), False)
            logger.error(f"Error creating relationship in Neo4j: {e}")
            return ""

    def find_node(
        self, label: str, properties: Dict[str, Any]
    ) -> Optional[Dict]:
        """Find single node by properties."""
        if not self.driver:
            logger.error("Neo4j driver not connected")
            return None

        try:
            start_time = time.time()

            with self.driver.session(database=self.database_name) as session:
                # Build WHERE clause
                where_conditions = []
                for key, value in properties.items():
                    if isinstance(value, str):
                        where_conditions.append(f"n.{key} = '{value}'")
                    else:
                        where_conditions.append(f"n.{key} = {value}")

                where_clause = " AND ".join(where_conditions)

                query = f"MATCH (n:{label}) WHERE {where_clause} RETURN n,\
                    id(n) AS node_id LIMIT 1"
                result = session.run(query)

                record = result.single()
                if record:
                    node_data = dict(record["n"])
                    node_data["_id"] = record["node_id"]

                    self._record_operation("find_node", start_time, True)
                    return node_data

                self._record_operation("find_node", start_time, True)
                return None

        except Exception as e:
            self._record_operation("find_node", time.time(), False)
            logger.error(f"Error finding node in Neo4j: {e}")
            return None

    def find_nodes(
        self,
        label: str,
        properties: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None,
    ) -> List[Dict]:
        """Find multiple nodes."""
        if not self.driver:
            logger.error("Neo4j driver not connected")
            return []

        try:
            start_time = time.time()

            with self.driver.session(database=self.database_name) as session:
                query = f"MATCH (n:{label})"

                if properties:
                    where_conditions = []
                    for key, value in properties.items():
                        if isinstance(value, str):
                            where_conditions.append(f"n.{key} = '{value}'")
                        else:
                            where_conditions.append(f"n.{key} = {value}")

                    where_clause = " AND ".join(where_conditions)
                    query += f" WHERE {where_clause}"

                query += " RETURN n, id(n) AS node_id"

                if limit:
                    query += f" LIMIT {limit}"

                result = session.run(query)

                nodes = []
                for record in result:
                    node_data = dict(record["n"])
                    node_data["_id"] = record["node_id"]
                    nodes.append(node_data)

                self._record_operation("find_nodes", start_time, True)
                logger.debug(f"Found {len(nodes)} nodes with label {label}")

                return nodes

        except Exception as e:
            self._record_operation("find_nodes", time.time(), False)
            logger.error(f"Error finding nodes in Neo4j: {e}")
            return []

    def update_node(self, node_id: str, properties: Dict[str, Any]) -> bool:
        """Update node properties."""
        if not self.driver:
            logger.error("Neo4j driver not connected")
            return False

        try:
            start_time = time.time()

            with self.driver.session(database=self.database_name) as session:
                # Add updated timestamp
                properties["updated_at"] = time.time()

                # Build SET clause
                set_items = []
                for key, value in properties.items():
                    if isinstance(value, str):
                        set_items.append(f"n.{key} = '{value}'")
                    else:
                        set_items.append(f"n.{key} = {value}")

                set_clause = ", ".join(set_items)

                query = f"MATCH (n) WHERE id(n) = {node_id} SET {set_clause}\
                    RETURN n"
                result = session.run(query)

                success = result.single() is not None
                self._record_operation("update_node", start_time, success)

                if success:
                    logger.debug(f"Updated node with ID: {node_id}")

                return success

        except Exception as e:
            self._record_operation("update_node", time.time(), False)
            logger.error(f"Error updating node in Neo4j: {e}")
            return False

    def delete_node(self, node_id: str) -> bool:
        """Delete node and its relationships."""
        if not self.driver:
            logger.error("Neo4j driver not connected")
            return False

        try:
            start_time = time.time()

            with self.driver.session(database=self.database_name) as session:
                query = f"MATCH (n) WHERE id(n) = {node_id} DETACH DELETE n"
                result = session.run(query)

                # Check if any nodes were deleted
                summary = result.consume()
                success = summary.counters.nodes_deleted > 0

                self._record_operation("delete_node", start_time, success)

                if success:
                    logger.debug(f"Deleted node with ID: {node_id}")

                return success

        except Exception as e:
            self._record_operation("delete_node", time.time(), False)
            logger.error(f"Error deleting node from Neo4j: {e}")
            return False

    def execute_query(
        self, query: str, parameters: Optional[Dict[str, Any]] = None
    ) -> List[Dict]:
        """Execute custom query."""
        if not self.driver:
            logger.error("Neo4j driver not connected")
            return []

        try:
            start_time = time.time()

            with self.driver.session(database=self.database_name) as session:
                result = session.run(query, parameters or {})

                records = []
                for record in result:
                    records.append(dict(record))

                self._record_operation("execute_query", start_time, True)
                logger.debug(f"Query returned {len(records)} records")

                return records

        except Exception as e:
            self._record_operation("execute_query", time.time(), False)
            logger.error(f"Error executing query in Neo4j: {e}")
            return []

    def get_shortest_path(
        self, from_node_id: str, to_node_id: str
    ) -> List[Dict]:
        """Find shortest path between nodes."""
        if not self.driver:
            logger.error("Neo4j driver not connected")
            return []

        try:
            start_time = time.time()

            with self.driver.session(database=self.database_name) as session:
                query = (
                    f"MATCH (start), (end), "
                    f"p = shortestPath((start)-[*]-(end)) "
                    f"WHERE id(start) = {from_node_id} AND id(end) =\
                        {to_node_id} "
                    f"RETURN p"
                )

                result = session.run(query)
                record = result.single()

                if record:
                    path = record["p"]
                    path_data = {
                        "nodes": [dict(node) for node in path.nodes],
                        "relationships": [
                            dict(rel) for rel in path.relationships
                        ],
                        "length": len(path.relationships),
                    }

                    self._record_operation(
                        "get_shortest_path", start_time, True
                    )
                    return [path_data]

                self._record_operation("get_shortest_path", start_time, True)
                return []

        except Exception as e:
            self._record_operation("get_shortest_path", time.time(), False)
            logger.error(f"Error finding shortest path in Neo4j: {e}")
            return []

    def create_agent_node(
        self,
        agent_id: str,
        agent_type: str,
        properties: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Create agent node."""
        node_props = {"agent_id": agent_id, "agent_type": agent_type}
        if properties:
            node_props.update(properties)

        return self.create_node("Agent", node_props)

    def create_hierarchy_relationship(
        self, subordinate_id: str, supervisor_id: str
    ) -> str:
        """Create REPORTS_TO relationship."""
        return self.create_relationship(
            subordinate_id,
            supervisor_id,
            "REPORTS_TO",
            {"relationship_type": "hierarchy", "created_at": time.time()},
        )

    def get_hierarchy_tree(self, root_agent_id: str) -> List[Dict]:
        """Get complete hierarchy tree from root."""
        query = """
        MATCH (root:Agent {agent_id: $root_agent_id})
        OPTIONAL MATCH path = (root)<-[:REPORTS_TO*]-(subordinate)
        RETURN root, collect(distinct subordinate) as subordinates,\
            collect(distinct path) as paths
        """

        return self.execute_query(query, {"root_agent_id": root_agent_id})
