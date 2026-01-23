"""
Persistent components that integrate storage with agent framework.
Replaces agent components with storage-aware versions.
"""

import time
from typing import Dict, Set, Optional, Any
from mas.core import State, Goal
from mas.organization import OrganizationalPosition, HierarchyMessage, MessageType
from .storage_manager import MultiAgentStorageManager


class PersistentState(State):
    """
    State component with automatic persistence to storage backends.
    Extends base State to automatically record belief updates.
    """

    def __init__(
        self, agent_id: str, storage_manager: MultiAgentStorageManager
    ):
        super().__init__()
        self.agent_id = agent_id
        self.storage = storage_manager
        self.auto_persist = True

    def update_belief(
        self,
        key: str,
        proposition: str,
        confidence: float = 1.0,
        is_internal: bool = False,
    ):
        """Update belief with automatic storage persistence."""
        # Update local state
        super().update_belief(key, proposition, confidence, is_internal)

        # Persist to storage if enabled
        if self.auto_persist and self.storage:
            self.storage.record_belief_update(
                agent_id=self.agent_id,
                belief_key=key,
                proposition=proposition,
                confidence=confidence,
                timestamp=time.time(),
            )

    def add_internal_belief(self, key: str, belief):
        """Add internal belief with persistence."""
        super().add_internal_belief(key, belief)

        if self.auto_persist and self.storage:
            self.storage.record_belief_update(
                agent_id=self.agent_id,
                belief_key=f"internal:{key}",
                proposition=belief.proposition,
                confidence=belief.confidence,
                timestamp=time.time(),
            )

    def add_external_belief(self, key: str, belief):
        """Add external belief with persistence."""
        super().add_external_belief(key, belief)

        if self.auto_persist and self.storage:
            self.storage.record_belief_update(
                agent_id=self.agent_id,
                belief_key=f"external:{key}",
                proposition=belief.proposition,
                confidence=belief.confidence,
                timestamp=time.time(),
            )

    def restore_from_storage(self, start_time: Optional[float] = None):
        """Restore state from storage."""
        if not self.storage or not self.storage.time_series:
            return

        # Get recent beliefs if no start time specified
        if start_time is None:
            start_time = time.time() - 3600  # Last hour

        end_time = time.time()

        # Query beliefs from storage
        query = (
            f'SELECT LAST("proposition"), "belief_key", "confidence" '
            f'FROM "agent_beliefs" WHERE "agent_id" = \'{self.agent_id}\' '
            f"AND time >= {int(start_time)}s AND time <= {int(end_time)}s "
            f'GROUP BY "belief_key"'
        )

        try:
            results = self.storage.time_series.query(query)

            for result in results:
                belief_key = result.get("belief_key", "")
                proposition = result.get("last", "")
                confidence = result.get("confidence", 1.0)

                if belief_key.startswith("internal:"):
                    key = belief_key[9:]  # Remove "internal:" prefix
                    self.update_belief(
                        key, proposition, confidence, is_internal=True
                    )
                elif belief_key.startswith("external:"):
                    key = belief_key[9:]  # Remove "external:" prefix
                    self.update_belief(
                        key, proposition, confidence, is_internal=False
                    )
                else:
                    self.update_belief(belief_key, proposition, confidence)

        except Exception as e:
            print(f"Error restoring state from storage: {e}")


class PersistentGoalManager:
    """
    Manages goal lifecycle with storage persistence.
    Tracks goal creation, updates, and achievement.
    """

    def __init__(
        self, agent_id: str, storage_manager: MultiAgentStorageManager
    ):
        self.agent_id = agent_id
        self.storage = storage_manager
        self.goals: Set[Goal] = set()

    def add_goal(self, goal: Goal):
        """Add goal with storage persistence."""
        self.goals.add(goal)

        if self.storage:
            self.storage.record_goal_event(
                agent_id=self.agent_id,
                goal_id=goal.condition,
                event_type="created",
                priority=goal.priority,
                goal_type=goal.goal_type.value,
                timestamp=time.time(),
            )

    def remove_goal(self, goal: Goal):
        """Remove goal with storage persistence."""
        self.goals.discard(goal)

        if self.storage:
            self.storage.record_goal_event(
                agent_id=self.agent_id,
                goal_id=goal.condition,
                event_type="removed",
                priority=goal.priority,
                goal_type=goal.goal_type.value,
                timestamp=time.time(),
            )

    def mark_goal_achieved(self, goal: Goal):
        """Mark goal as achieved with storage persistence."""
        goal.active = False

        if self.storage:
            self.storage.record_goal_event(
                agent_id=self.agent_id,
                goal_id=goal.condition,
                event_type="achieved",
                priority=goal.priority,
                goal_type=goal.goal_type.value,
                timestamp=time.time(),
            )

    def update_goal_priority(self, goal: Goal, new_priority: float):
        """Update goal priority with storage persistence."""
        old_priority = goal.priority
        goal.priority = new_priority

        if self.storage:
            self.storage.record_goal_event(
                agent_id=self.agent_id,
                goal_id=goal.condition,
                event_type="priority_updated",
                priority=new_priority,
                goal_type=goal.goal_type.value,
                timestamp=time.time(),
            )

    def get_active_goals(self) -> Set[Goal]:
        """Get all active goals."""
        return {g for g in self.goals if g.active}

    def get_goal_history(
        self, start_time: float, end_time: float
    ) -> Dict[str, Any]:
        """Get goal history from storage."""
        if not self.storage or not self.storage.time_series:
            return {"goals": [], "events": []}

        query = (
            f'SELECT * FROM "agent_goals" WHERE "agent_id" = \'{self.agent_id}\' '
            f"AND time >= {int(start_time)}s AND time <= {int(end_time)}s "
            f"ORDER BY time DESC"
        )

        try:
            events = self.storage.time_series.query(query)
            return {
                "agent_id": self.agent_id,
                "events": events,
                "total_events": len(events),
            }
        except Exception as e:
            print(f"Error getting goal history: {e}")
            return {"goals": [], "events": []}


class PersistentMessageQueue:
    """
    Message queue with storage persistence and caching.
    Handles hierarchy messages with automatic storage.
    """

    def __init__(
        self, agent_id: str, storage_manager: MultiAgentStorageManager
    ):
        self.agent_id = agent_id
        self.storage = storage_manager
        self.local_queue: list[HierarchyMessage] = []
        self.message_cache_ttl = 600  # 10 minutes

    def add_message(self, message: HierarchyMessage):
        """Add message to queue with storage persistence."""
        self.local_queue.append(message)

        # Cache message for fast access
        if self.storage and self.storage.cache:
            cached_queue = self.storage.cache.get_cached_message_queue(
                self.agent_id
            )
            cached_queue.append(message.__dict__)
            self.storage.cache.cache_message_queue(
                self.agent_id, cached_queue, self.message_cache_ttl
            )

        # Record communication in time series
        if self.storage:
            self.storage.record_hierarchy_message(
                sender=message.sender,
                receiver=message.receiver,
                message_type=message.message_type.value,
                priority=message.priority,
                response_required=message.requires_response,
                timestamp=message.timestamp,
            )

    def get_messages(
        self,
        message_type: Optional[MessageType] = None,
        min_priority: float = 0.0,
    ) -> list[HierarchyMessage]:
        """Get messages from queue with optional filtering."""
        messages = []

        for msg in self.local_queue:
            if message_type and msg.message_type != message_type:
                continue
            if msg.priority < min_priority:
                continue
            messages.append(msg)

        return messages

    def remove_message(self, message: HierarchyMessage):
        """Remove message from queue."""
        if message in self.local_queue:
            self.local_queue.remove(message)

            # Update cache
            if self.storage and self.storage.cache:
                remaining = [msg.__dict__ for msg in self.local_queue]
                self.storage.cache.cache_message_queue(
                    self.agent_id, remaining, self.message_cache_ttl
                )

    def clear_messages(self):
        """Clear all messages from queue."""
        self.local_queue.clear()

        if self.storage and self.storage.cache:
            self.storage.cache.cache_message_queue(
                self.agent_id, [], self.message_cache_ttl
            )

    def get_message_count(self) -> int:
        """Get total message count."""
        return len(self.local_queue)

    def restore_from_cache(self):
        """Restore messages from cache."""
        if not self.storage or not self.storage.cache:
            return

        try:
            cached_messages = self.storage.cache.get_cached_message_queue(
                self.agent_id
            )

            self.local_queue = []
            for msg_data in cached_messages:
                # Reconstruct message object
                message = HierarchyMessage(
                    message_type=MessageType(msg_data["message_type"]),
                    sender=msg_data["sender"],
                    receiver=msg_data["receiver"],
                    content=msg_data["content"],
                    priority=msg_data["priority"],
                    timestamp=msg_data["timestamp"],
                    requires_response=msg_data.get("requires_response", False),
                    escalation_path=msg_data.get("escalation_path", []),
                )
                self.local_queue.append(message)

        except Exception as e:
            print(f"Error restoring message queue from cache: {e}")


class PersistentOrganizationalPosition(OrganizationalPosition):
    """
    Organizational position with storage persistence.
    Tracks hierarchy changes and maintains consistency.
    """

    def __init__(
        self,
        agent_id: str,
        storage_manager: MultiAgentStorageManager,
        roles: Set[str] = None,
        supervisor: Optional[str] = None,
        subordinates: Set[str] = None,
        groups: Set[str] = None,
    ):
        super().__init__(
            roles=roles or set(),
            supervisor=supervisor,
            subordinates=subordinates or set(),
            groups=groups or set(),
        )

        self.agent_id = agent_id
        self.storage = storage_manager
        self._cache_ttl = 3600  # 1 hour

        # Cache current position
        self._cache_position()

    def set_supervisor(self, supervisor_id: Optional[str]):
        """Set supervisor with storage persistence."""
        old_supervisor = self.supervisor
        self.supervisor = supervisor_id

        # Record hierarchy change
        if self.storage:
            change_type = "assigned" if supervisor_id else "removed"
            self.storage.record_hierarchy_change(
                agent_id=self.agent_id,
                supervisor=supervisor_id,
                change_type=change_type,
                timestamp=time.time(),
            )

        # Update cache
        self._cache_position()

    def add_subordinate(self, subordinate_id: str):
        """Add subordinate with storage persistence."""
        self.subordinates.add(subordinate_id)

        if self.storage:
            self.storage.record_hierarchy_change(
                agent_id=subordinate_id,
                supervisor=self.agent_id,
                change_type="assigned",
                timestamp=time.time(),
            )

        self._cache_position()

    def remove_subordinate(self, subordinate_id: str):
        """Remove subordinate with storage persistence."""
        self.subordinates.discard(subordinate_id)

        if self.storage:
            self.storage.record_hierarchy_change(
                agent_id=subordinate_id,
                supervisor=None,
                change_type="removed",
                timestamp=time.time(),
            )

        self._cache_position()

    def add_role(self, role_name: str):
        """Add role with cache update."""
        self.roles.add(role_name)
        self._cache_position()

    def remove_role(self, role_name: str):
        """Remove role with cache update."""
        self.roles.discard(role_name)
        self._cache_position()

    def add_group(self, group_name: str):
        """Add group membership with cache update."""
        self.groups.add(group_name)
        self._cache_position()

    def remove_group(self, group_name: str):
        """Remove group membership with cache update."""
        self.groups.discard(group_name)
        self._cache_position()

    def _cache_position(self):
        """Cache current position data."""
        if not self.storage or not self.storage.cache:
            return

        position_data = {
            "roles": list(self.roles),
            "supervisor": self.supervisor,
            "subordinates": list(self.subordinates),
            "groups": list(self.groups),
            "hierarchy_level": self.hierarchy_level,
            "last_updated": time.time(),
        }

        self.storage.cache.cache_hierarchy_position(
            self.agent_id, position_data, self._cache_ttl
        )

    def restore_from_cache(self):
        """Restore position from cache."""
        if not self.storage or not self.storage.cache:
            return

        try:
            cached_data = self.storage.cache.get_cached_hierarchy_position(
                self.agent_id
            )

            if cached_data:
                self.roles = set(cached_data.get("roles", []))
                self.supervisor = cached_data.get("supervisor")
                self.subordinates = set(cached_data.get("subordinates", []))
                self.groups = set(cached_data.get("groups", []))
                self.hierarchy_level = cached_data.get("hierarchy_level", 0)

        except Exception as e:
            print(f"Error restoring organizational position from cache: {e}")


class ActionLogger:
    """
    Logs agent action executions to storage.
    Provides performance tracking and analysis.
    """

    def __init__(
        self, agent_id: str, storage_manager: MultiAgentStorageManager
    ):
        self.agent_id = agent_id
        self.storage = storage_manager

    def log_action_start(self, action_id: str, action_type: str) -> float:
        """Log action start and return start time."""
        start_time = time.time()

        # Could log action starts for detailed timing analysis
        return start_time

    def log_action_complete(
        self,
        action_id: str,
        action_type: str,
        start_time: float,
        success: bool = True,
    ):
        """Log action completion with timing."""
        end_time = time.time()
        execution_time = end_time - start_time

        if self.storage:
            self.storage.record_action_execution(
                agent_id=self.agent_id,
                action_id=action_id,
                action_type=action_type,
                success=success,
                execution_time=execution_time,
                timestamp=end_time,
            )

    def get_action_stats(
        self, start_time: float, end_time: float
    ) -> Dict[str, Any]:
        """Get action execution statistics."""
        if not self.storage or not self.storage.time_series:
            return {}

        query = (
            f'SELECT COUNT("success"), MEAN("execution_time"), '
            f'SUM(case when "success"=true then 1 else 0 end) as success_count '
            f'FROM "agent_actions" WHERE "agent_id" = \'{self.agent_id}\' '
            f"AND time >= {int(start_time)}s AND time <= {int(end_time)}s "
            f'GROUP BY "action_type"'
        )

        try:
            results = self.storage.time_series.query(query)
            stats = {}

            for result in results:
                action_type = result.get("action_type", "unknown")
                stats[action_type] = {
                    "total_executions": result.get("count", 0),
                    "average_execution_time": result.get("mean", 0.0),
                    "success_rate": result.get("success_count", 0)
                    / max(1, result.get("count", 1)),
                }

            return stats

        except Exception as e:
            print(f"Error getting action statistics: {e}")
            return {}
