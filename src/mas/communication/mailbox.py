"""
Agent Mailbox

Message buffer for agent perception, implementing MB_i from the formal model.
Thread-safe queue for buffering incoming messages before agent processing.
"""

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Optional, Dict, Any


@dataclass
class Message:
    """
    A message received by an agent.

    Represents an entry in the mailbox MB_i.
    """
    topic: str
    payload: str
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = time.time()


class Mailbox:
    """
    Agent mailbox: MB_i ⊆ M

    Thread-safe message buffer that integrates with agent perception.
    Messages are added by transport callbacks and consumed by the
    agent's perception phase.

    Features:
    - Thread-safe add/get operations
    - Configurable max size with overflow handling
    - Message filtering and peeking
    - Statistics tracking

    Example:
        mailbox = Mailbox("agent_001", max_size=1000)

        # In transport callback
        mailbox.add(Message(topic="battery/B0005/raw", payload=json_str))

        # In agent perception phase
        messages = mailbox.get_all()
        for msg in messages:
            process_message(msg)
    """

    def __init__(self, agent_id: str, max_size: int = 1000):
        """
        Initialize mailbox.

        Args:
            agent_id: Owner agent's identifier
            max_size: Maximum messages to buffer (oldest dropped on overflow)
        """
        self.agent_id = agent_id
        self.max_size = max_size
        self._messages: Deque[Message] = deque(maxlen=max_size)
        self._lock = threading.Lock()

        # Statistics
        self._total_received = 0
        self._total_dropped = 0

    def add(self, message: Message) -> bool:
        """
        Add a message to the mailbox.

        Thread-safe. If mailbox is full, oldest message is dropped.

        Args:
            message: Message to add

        Returns:
            True if added successfully
        """
        with self._lock:
            was_full = len(self._messages) >= self.max_size
            self._messages.append(message)
            self._total_received += 1

            if was_full:
                self._total_dropped += 1

            return True

    def add_raw(self, topic: str, payload: str, **metadata) -> bool:
        """
        Add a message from raw topic and payload.

        Convenience method for transport callbacks.

        Args:
            topic: MQTT topic
            payload: Message payload (JSON string)
            **metadata: Additional metadata

        Returns:
            True if added successfully
        """
        message = Message(
            topic=topic,
            payload=payload,
            metadata=metadata,
        )
        return self.add(message)

    def get_all(self, clear: bool = True) -> List[Message]:
        """
        Get all messages from the mailbox.

        Args:
            clear: If True, remove messages after getting

        Returns:
            List of messages
        """
        with self._lock:
            messages = list(self._messages)
            if clear:
                self._messages.clear()
            return messages

    def get(self, count: int = 1, clear: bool = True) -> List[Message]:
        """
        Get up to count messages from the mailbox.

        Args:
            count: Maximum number of messages to get
            clear: If True, remove returned messages

        Returns:
            List of messages (may be fewer than count)
        """
        with self._lock:
            messages = []
            for _ in range(min(count, len(self._messages))):
                if clear:
                    messages.append(self._messages.popleft())
                else:
                    messages.append(self._messages[len(messages)])
            return messages

    def peek(self, count: Optional[int] = None) -> List[Message]:
        """
        View messages without removing them.

        Args:
            count: Number of messages to peek (None for all)

        Returns:
            List of messages
        """
        with self._lock:
            if count is None:
                return list(self._messages)
            return list(self._messages)[:count]

    def filter_by_topic(
        self,
        topic_pattern: str,
        clear: bool = True
    ) -> List[Message]:
        """
        Get messages matching a topic pattern.

        Args:
            topic_pattern: Topic pattern (supports * wildcard)
            clear: If True, remove matching messages

        Returns:
            List of matching messages
        """
        import fnmatch

        with self._lock:
            matching = []
            remaining = deque(maxlen=self.max_size)

            for msg in self._messages:
                if fnmatch.fnmatch(msg.topic, topic_pattern.replace("+", "*")):
                    matching.append(msg)
                else:
                    remaining.append(msg)

            if clear:
                self._messages = remaining

            return matching

    def is_empty(self) -> bool:
        """Check if mailbox is empty."""
        with self._lock:
            return len(self._messages) == 0

    def size(self) -> int:
        """Get current number of messages."""
        with self._lock:
            return len(self._messages)

    def clear(self) -> int:
        """
        Clear all messages.

        Returns:
            Number of messages cleared
        """
        with self._lock:
            count = len(self._messages)
            self._messages.clear()
            return count

    def get_stats(self) -> Dict[str, Any]:
        """
        Get mailbox statistics.

        Returns:
            Dictionary with statistics
        """
        with self._lock:
            return {
                "agent_id": self.agent_id,
                "current_size": len(self._messages),
                "max_size": self.max_size,
                "total_received": self._total_received,
                "total_dropped": self._total_dropped,
                "utilization": len(self._messages) / self.max_size,
            }

    def reset_stats(self) -> None:
        """Reset statistics counters."""
        with self._lock:
            self._total_received = 0
            self._total_dropped = 0


__all__ = ["Mailbox", "Message"]
