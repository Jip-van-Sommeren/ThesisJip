# Multi-Agent Framework Architecture Specification

## Overview

This document specifies the abstract multi-agent framework architecture for digital twin applications. The framework is based on the formal definitions in `sec_definitions.md` and uses MQTT as the primary communication protocol.

## Current Architecture Issues

| Issue | Location | Problem |
|-------|----------|---------|
| Fragmentation | Multiple base classes | `abstract_agent.py`, `CommAgentBase`, `BatteryAgentBase` scattered |
| Coupling | `battery_twin/` | Tightly coupled to MQTT bridge implementation |
| Duplication | `src/communication/`, `src/battery_twin/communication/` | TopicManager, Transport exist twice |
| Import hacks | Various | `sys.path.insert()` used in multiple files |
| Mixed concerns | Agent bases | Communication, organization, and domain logic interleaved |

---

## Proposed Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           APPLICATION LAYER                                  │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │  Digital Twin Applications (e.g., BatteryTwin, FactoryTwin)             ││
│  │  - Domain-specific agents (TelemetryAgent, PhysicsAgent, etc.)          ││
│  │  - Domain message schemas (TelemetryMessage, PredictionMessage)         ││
│  │  - Domain topic configurations                                           ││
│  └─────────────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          ORGANIZATION LAYER                                  │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────────┐ │
│  │   RoleManager   │  │  GroupManager   │  │    HierarchyManager         │ │
│  │  - Permissions  │  │  - Membership   │  │    - Sup/Sub relations      │ │
│  │  - Expectations │  │  - Purpose      │  │    - Escalation paths       │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         COMMUNICATION LAYER                                  │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │                    Transport Interface                                   ││
│  │   connect() | disconnect() | publish() | subscribe() | is_connected()   ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│          │                                               │                   │
│          ▼                                               ▼                   │
│  ┌───────────────────────────────┐        ┌───────────────────────────────┐ │
│  │        MqttTransport          │        │       MockTransport           │ │
│  │  (paho-mqtt, QoS 0/1/2, TLS)  │        │    (in-memory, testing)       │ │
│  └───────────────────────────────┘        └───────────────────────────────┘ │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │                    TopicManager                                          ││
│  │   - Template loading (YAML)                                              ││
│  │   - Variable substitution: {entity_id} → actual_id                       ││
│  │   - Subscription patterns: {entity_id} → +                               ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │                    Mailbox (MB_i)                                        ││
│  │   - Message buffering                                                    ││
│  │   - Thread-safe queue                                                    ││
│  │   - Integration with agent perception                                    ││
│  └─────────────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            CORE LAYER                                        │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │                    Agent = (Id, State, Goal, Perception, Action, Decision)│
│  └─────────────────────────────────────────────────────────────────────────┘│
│          │                      │                        │                   │
│          ▼                      ▼                        ▼                   │
│  ┌───────────────┐    ┌───────────────┐        ┌───────────────────┐       │
│  │ ReactiveAgent │    │   BDIAgent    │        │   HybridAgent     │       │
│  │ (stimulus→act)│    │ (BDI cycle)   │        │ (layered)         │       │
│  └───────────────┘    └───────────────┘        └───────────────────┘       │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Component Specifications

### 1. Core Layer (`src/mas/core/`)

#### 1.1 Agent Identity

```python
@dataclass
class AgentId:
    """
    Hierarchical agent identity: id_A = (app, type, instance)
    Provides global uniqueness and context.
    """
    app: str           # Application/domain (e.g., "battery_twin")
    type: str          # Agent type (e.g., "telemetry_ingestor")
    instance: str      # Unique instance ID
```

#### 1.2 Abstract Agent

```python
class Agent(ABC):
    """
    A = (Id, State, Goal, Perception, Action, Decision)

    Protocol-agnostic base. Communication is injected via Transport.
    """
    id: AgentId
    state: State           # Internal/external beliefs
    goals: Set[Goal]       # Active goals
    perception: Perception # Observation function Ω_A
    actions: Dict[str, Action]
    decision: Decision     # δ_A: State × Percepts → Action

    # Injected dependencies
    transport: Optional[Transport] = None
    mailbox: Optional[Mailbox] = None

    @abstractmethod
    def step(self, env_state: Dict) -> Tuple[str, Dict]:
        """Perceive → Decide → Act cycle."""
        pass
```

#### 1.3 Agent Types

| Type | Class | Decision Mechanism | Use Case |
|------|-------|-------------------|----------|
| Reactive | `ReactiveAgent` | Stimulus → Response rules | Fast control, anomaly detection |
| BDI | `BDIAgent` | Beliefs, Desires, Intentions | Planning, optimization |
| Hybrid | `HybridAgent` | Layered: reactive + deliberative | Digital twins (default) |

### 2. Communication Layer (`src/mas/communication/`)

#### 2.1 Transport Interface

```python
class Transport(ABC):
    """
    Transport interface for MQTT-based agent communication.
    Implementations: MqttTransport (production), MockTransport (testing).
    """

    @abstractmethod
    def connect(self) -> bool:
        """Connect to message broker."""

    @abstractmethod
    def disconnect(self) -> None:
        """Disconnect from broker."""

    @abstractmethod
    def is_connected(self) -> bool:
        """Check connection status."""

    @abstractmethod
    def publish(self, topic: str, message: BaseModel, qos: int = 1) -> bool:
        """Publish validated message to topic."""

    @abstractmethod
    def subscribe(
        self,
        topic_pattern: str,
        callback: Callable[[str, str], None],
        qos: int = 1
    ) -> bool:
        """Subscribe to topic pattern with callback."""
```

#### 2.2 MQTT Transport Implementation

```python
class MqttTransport(Transport):
    """
    MQTT implementation using paho-mqtt.
    Supports QoS 0/1/2, TLS, authentication.
    """

    def __init__(
        self,
        config: MqttConfig,
        topic_manager: TopicManager
    ):
        self.config = config
        self.topic_manager = topic_manager
        self.client: mqtt.Client = None

    def publish_to_topic(
        self,
        topic_name: str,     # Logical name from YAML
        message: BaseModel,
        **topic_vars         # e.g., battery_id="B0005"
    ) -> bool:
        """Publish using topic manager for resolution."""
        topic = self.topic_manager.get_topic(topic_name, **topic_vars)
        return self.publish(topic, message)
```

#### 2.3 Topic Manager

```python
class TopicManager:
    """
    Manages topic templates from YAML configuration.
    Twin-agnostic: works with any topic schema.
    """

    def __init__(self, config_path: str):
        """Load topics from YAML."""

    def get_topic(self, topic_name: str, **kwargs) -> str:
        """
        Format topic with variables.

        Example:
            get_topic("raw_telemetry", battery_id="B0005")
            → "battery/B0005/raw"
        """

    def get_subscription_pattern(self, topic_name: str, **kwargs) -> str:
        """
        Get MQTT wildcard pattern.

        Example:
            get_subscription_pattern("raw_telemetry", battery_id=None)
            → "battery/+/raw"
        """
```

#### 2.4 Mailbox

```python
class Mailbox:
    """
    MB_i: Message buffer for agent i.
    Part of agent's perception input.
    """

    agent_id: str
    messages: Deque[Message]
    max_size: int = 1000

    def add(self, message: Message) -> None:
        """Add message (thread-safe)."""

    def get_all(self, clear: bool = True) -> List[Message]:
        """Get all messages, optionally clearing."""

    def peek(self) -> List[Message]:
        """View without removing."""
```

#### 2.5 Message Schema Base

```python
class AgentMessage(BaseModel):
    """Base message type for all agent communication."""

    message_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    sender_id: str
    timestamp: float = Field(default_factory=time.time)

    class Config:
        extra = "allow"  # Allow domain-specific fields
```

### 3. Organization Layer (`src/mas/organization/`)

#### 3.1 Role

```python
@dataclass
class Role:
    """
    R = (responsibilities, permissions, requirements, expectations)
    """
    name: str
    responsibilities: Set[Responsibility]
    permissions: Set[Permission]
    requirements: Set[Requirement]
    expectations: Set[Expectation]
```

#### 3.2 Group

```python
@dataclass
class Group:
    """
    G = (Name, Roles_G, Purpose)
    """
    name: str
    roles: Set[str]    # Role names in this group
    purpose: str
    agents: Set[str]   # Agent IDs currently in group
```

#### 3.3 Hierarchy

```python
class HierarchyManager:
    """
    Manages Sup: A → A ∪ {⊥} and Sub: A → 2^A relations.
    """

    def get_supervisor(self, agent_id: str) -> Optional[str]:
        """Sup(agent_id)"""

    def get_subordinates(self, agent_id: str) -> Set[str]:
        """Sub(agent_id)"""

    def get_escalation_path(self, agent_id: str, level: int) -> List[str]:
        """Path to supervisor at given level."""
```

### 4. Application Layer (`src/battery_twin/`)

#### 4.1 Battery Agent Base

```python
class BatteryAgentBase(HybridAgent):
    """
    Battery-specific agent combining:
    - HybridAgent (reactive + deliberative)
    - MqttTransport for communication
    - BatteryStorageManager for persistence
    """

    def __init__(
        self,
        agent_id: AgentId,
        transport: MqttTransport,
        storage: Optional[BatteryStorageManager] = None
    ):
        super().__init__(agent_id, self._observable_properties())
        self.transport = transport
        self.storage = storage

    @abstractmethod
    def _observable_properties(self) -> Set[str]:
        """Domain-specific observables."""
```

#### 4.2 Battery Message Schemas

```python
class TelemetryMessage(AgentMessage):
    """Battery telemetry from sensors."""
    battery_id: str
    cycle: int
    voltage: float
    current: float
    temperature: float

class PredictionMessage(AgentMessage):
    """Capacity prediction from model."""
    battery_id: str
    prediction_type: Literal["physics", "ml", "hybrid"]
    predicted_capacity: float
    uncertainty: Optional[float]
```

---

## Directory Structure

```
src/
├── mas/                           # Multi-Agent System framework (NEW)
│   ├── __init__.py
│   ├── core/                      # Core agent abstractions
│   │   ├── __init__.py
│   │   ├── agent.py               # Agent, AgentId, State, Goal
│   │   ├── perception.py          # Perception, Belief
│   │   ├── action.py              # Action, ActionType
│   │   ├── decision.py            # Decision, ReactiveRule
│   │   ├── reactive_agent.py      # ReactiveAgent
│   │   ├── bdi_agent.py           # BDIAgent
│   │   └── hybrid_agent.py        # HybridAgent
│   │
│   ├── communication/             # Communication layer
│   │   ├── __init__.py
│   │   ├── transport.py           # Transport ABC
│   │   ├── mqtt_transport.py      # MqttTransport implementation
│   │   ├── topic_manager.py       # TopicManager
│   │   ├── mailbox.py             # Mailbox
│   │   └── message.py             # AgentMessage base
│   │
│   └── organization/              # Organizational structure
│       ├── __init__.py
│       ├── role.py                # Role, RoleManager
│       ├── group.py               # Group, GroupManager
│       └── hierarchy.py           # HierarchyManager
│
├── battery_twin/                  # Battery twin application
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── battery_agent_base.py  # Extends HybridAgent
│   │   ├── telemetry_ingestor_agent.py
│   │   ├── physics_model_agent.py
│   │   ├── ml_residual_agent.py
│   │   ├── state_estimator_agent.py
│   │   └── health_monitor_agent.py
│   │
│   ├── communication/
│   │   ├── __init__.py
│   │   ├── message_schemas.py     # Battery-specific messages
│   │   └── topic_config.yaml      # Battery topic templates
│   │
│   ├── models/                    # Physics/ML models
│   ├── storage/                   # InfluxDB/MongoDB adapters
│   └── config/
│
└── [legacy files - to be migrated]
    ├── abstract_agent.py          → mas/core/agent.py
    ├── reactive_agent.py          → mas/core/reactive_agent.py
    ├── bdi_agent.py               → mas/core/bdi_agent.py
    ├── hybrid_agent.py            → mas/core/hybrid_agent.py
    ├── role.py                    → mas/organization/role.py
    ├── group.py                   → mas/organization/group.py
    └── hierarchy.py               → mas/organization/hierarchy.py
```

---

## Key Design Decisions

### 1. Dependency Injection for Communication

**Before:**
```python
class BatteryAgentBase(CommAgentBase):
    def __init__(self, mqtt_bridge=None, ...):
        # Creates its own MQTT client if not provided
        if mqtt_bridge is None:
            self.mqtt_bridge = MqttBridge(...)
```

**After:**
```python
class BatteryAgentBase(HybridAgent):
    def __init__(self, transport: Transport, ...):
        # Transport MUST be provided (enables testing with MockTransport)
        self.transport = transport
```

### 2. Single TopicManager Instance

All agents in an application share one `TopicManager`:

```python
# Application setup
topic_manager = TopicManager("config/battery_topics.yaml")
transport = MqttTransport(mqtt_config, topic_manager)

# Pass same transport to all agents
telemetry_agent = TelemetryIngestorAgent(transport=transport)
physics_agent = PhysicsModelAgent(transport=transport)
```

### 3. Clean Import Structure

```python
# Application code imports
from mas.core import HybridAgent, AgentId
from mas.communication import MqttTransport, TopicManager
from mas.organization import RoleManager

# No sys.path hacks needed
```

### 4. Environment Integration

The formal environment model `ENV = (S, A_i, τ, O_i)` maps to:

| Formal | Implementation |
|--------|----------------|
| S (state space) | MQTT message flow + storage state |
| A_i (actions) | Agent's registered actions |
| τ (transition) | Publish messages → storage writes → state updates |
| O_i (observation) | Subscribe callbacks → Mailbox → Perception |

---

## Migration Plan

### Phase 1: Create `src/mas/` Package Structure

**Step 1.1: Create directories**
```bash
mkdir -p src/mas/core
mkdir -p src/mas/communication
mkdir -p src/mas/organization
touch src/mas/__init__.py
touch src/mas/core/__init__.py
touch src/mas/communication/__init__.py
touch src/mas/organization/__init__.py
```

**Step 1.2: Migrate Core Layer**

| Source | Destination | Changes |
|--------|-------------|---------|
| `src/abstract_agent.py` | `src/mas/core/agent.py` | Split into agent.py, perception.py, action.py, decision.py |
| `src/reactive_agent.py` | `src/mas/core/reactive_agent.py` | Update imports to use `mas.core.agent` |
| `src/bdi_agent.py` | `src/mas/core/bdi_agent.py` | Update imports to use `mas.core.agent` |
| `src/hybrid_agent.py` | `src/mas/core/hybrid_agent.py` | Update imports to use `mas.core.agent` |

**Step 1.3: Migrate Communication Layer**

| Source | Destination | Changes |
|--------|-------------|---------|
| `src/communication/transport.py` | `src/mas/communication/transport.py` | Keep as-is (already clean) |
| `src/communication/topic_manager.py` | `src/mas/communication/topic_manager.py` | Keep as-is (already clean) |
| NEW | `src/mas/communication/mqtt_transport.py` | Refactor from `MqttBridge` |
| NEW | `src/mas/communication/mailbox.py` | Extract from `base_communication.py` |
| NEW | `src/mas/communication/message.py` | Base `AgentMessage` class |

**Step 1.4: Migrate Organization Layer**

| Source | Destination | Changes |
|--------|-------------|---------|
| `src/role.py` | `src/mas/organization/role.py` | Update imports |
| `src/group.py` | `src/mas/organization/group.py` | Update imports |
| `src/hierarchy.py` | `src/mas/organization/hierarchy.py` | Update imports, remove agent type imports |

---

### Phase 2: Refactor Communication Layer

**Step 2.1: Create `MqttTransport` (merge and simplify)**

Current state:
- `src/communication/mqtt/mqtt_communication.py` - MqttCommunicationService (800+ lines)
- `src/battery_twin/communication/mqtt_bridge.py` - MqttBridge (500+ lines)
- `src/communication/agent_base.py` - CommAgentBase (300+ lines)

Target: Single `MqttTransport` class (~200 lines)

```python
# src/mas/communication/mqtt_transport.py
class MqttTransport(Transport):
    """
    MQTT transport implementation.
    Wraps paho-mqtt with TopicManager integration.
    """
    def __init__(self, config: MqttConfig, topic_manager: TopicManager): ...
    def connect(self) -> bool: ...
    def disconnect(self) -> None: ...
    def is_connected(self) -> bool: ...
    def publish(self, topic: str, payload: str, qos: int = 1) -> bool: ...
    def publish_to_topic(self, topic_name: str, message: BaseModel, **vars) -> bool: ...
    def subscribe(self, pattern: str, callback: Callable, qos: int = 1) -> bool: ...
    def subscribe_to_topic(self, topic_name: str, callback: Callable, **vars) -> bool: ...
```

**Step 2.2: Create `MockTransport` for testing**

```python
# src/mas/communication/mock_transport.py
class MockTransport(Transport):
    """In-memory transport for unit testing."""
    published: List[Tuple[str, str]]
    subscriptions: Dict[str, List[Callable]]

    def simulate_message(self, topic: str, payload: str): ...
```

**Step 2.3: Update `__init__.py` exports**

```python
# src/mas/communication/__init__.py
from .transport import Transport
from .mqtt_transport import MqttTransport, MqttConfig
from .mock_transport import MockTransport
from .topic_manager import TopicManager
from .mailbox import Mailbox
from .message import AgentMessage

__all__ = [
    "Transport", "MqttTransport", "MqttConfig", "MockTransport",
    "TopicManager", "Mailbox", "AgentMessage"
]
```

---

### Phase 3: Update Battery Twin Agents

**Step 3.1: Refactor `BatteryAgentBase`**

Current inheritance chain:
```
BatteryAgentBase → CommAgentBase → (nothing)
BatteryHybridAgent → HybridAgent + BatteryAgentBase (diamond)
```

New inheritance chain:
```
BatteryAgentBase → HybridAgent → Agent
```

Changes to `src/battery_twin/agents/battery_agent_base.py`:
- Remove `CommAgentBase` inheritance
- Inject `MqttTransport` via constructor
- Remove `mqtt_bridge` / `own_mqtt` logic
- Keep `BatteryStorageManager` integration

```python
# BEFORE
class BatteryAgentBase(CommAgentBase):
    def __init__(self, mqtt_bridge=None, storage_manager=None, mqtt_config=None, ...):
        super().__init__(mqtt_bridge=mqtt_bridge, mqtt_config=mqtt_config, ...)

# AFTER
class BatteryAgentBase(HybridAgent):
    def __init__(self, agent_id: AgentId, transport: MqttTransport,
                 storage: Optional[BatteryStorageManager] = None):
        super().__init__(agent_id, self._get_observable_properties())
        self.transport = transport
        self.storage = storage
```

**Step 3.2: Simplify `battery_agent_types.py`**

Current: Complex multiple inheritance wrappers
```python
class BatteryHybridAgent(HybridAgent, BatteryAgentBase):  # Diamond problem
```

New: Remove file entirely, use `BatteryAgentBase` directly

**Step 3.3: Update concrete agents**

Files to update:
- `src/battery_twin/agents/telemetry_ingestor_agent.py`
- `src/battery_twin/agents/physics_model_agent.py`
- `src/battery_twin/agents/ml_residual_agent.py`
- `src/battery_twin/agents/state_estimator_agent.py`
- `src/battery_twin/agents/health_monitor_agent.py`
- `src/battery_twin/agents/registry_agent.py`

Changes per file:
1. Update imports: `from mas.core import AgentId`
2. Update imports: `from mas.communication import MqttTransport`
3. Change constructor to accept `transport: MqttTransport`
4. Replace `self.mqtt_bridge.publish(...)` → `self.transport.publish_to_topic(...)`
5. Replace `self.mqtt_bridge.subscribe(...)` → `self.transport.subscribe_to_topic(...)`

---

### Phase 4: Update Tests

**Step 4.1: Create test fixtures with MockTransport**

```python
# src/battery_twin/tests/conftest.py
import pytest
from mas.communication import MockTransport, TopicManager

@pytest.fixture
def mock_transport():
    topic_manager = TopicManager("src/battery_twin/config/mqtt_topics.yaml")
    return MockTransport(topic_manager)

@pytest.fixture
def telemetry_agent(mock_transport):
    from battery_twin.agents import TelemetryIngestorAgent
    from mas.core import AgentId
    agent = TelemetryIngestorAgent(
        agent_id=AgentId("battery_twin", "telemetry", "test"),
        transport=mock_transport
    )
    agent.setup()
    return agent
```

**Step 4.2: Update existing tests**

| Test File | Changes |
|-----------|---------|
| `test_step3_mqtt.py` | Use MockTransport instead of real broker |
| `test_step5_base_agent.py` | Update imports, use MockTransport |
| `test_step6_telemetry_ingestor.py` | Update agent construction |
| `test_step9_physics_agent.py` | Update agent construction |
| `test_step11_ml_agent.py` | Update agent construction |
| `test_step13_state_agent.py` | Update agent construction |
| `test_step14_health_agent.py` | Update agent construction |

---

### Phase 5: Clean Up Legacy Code

**Step 5.1: Files to DELETE**

```
src/communication/agent_base.py          # Replaced by mas/communication/
src/communication/transport.py           # Moved to mas/communication/
src/communication/topic_manager.py       # Moved to mas/communication/
src/communication/mqtt/                  # Replaced by mas/communication/mqtt_transport.py
src/abstract_agent.py                    # Moved to mas/core/
src/reactive_agent.py                    # Moved to mas/core/
src/bdi_agent.py                         # Moved to mas/core/
src/hybrid_agent.py                      # Moved to mas/core/
src/role.py                              # Moved to mas/organization/
src/group.py                             # Moved to mas/organization/
src/hierarchy.py                         # Moved to mas/organization/
src/battery_twin/agents/battery_agent_types.py  # No longer needed
```

**Step 5.2: Remove `sys.path` hacks**

Files with `sys.path.insert()` to clean:
- `src/battery_twin/communication/mqtt_bridge.py`
- `src/battery_twin/agents/battery_agent_types.py`

**Step 5.3: Update `__init__.py` files**

```python
# src/mas/__init__.py
from .core import Agent, AgentId, HybridAgent, ReactiveAgent, BDIAgent
from .communication import MqttTransport, TopicManager, MockTransport
from .organization import RoleManager, GroupManager, HierarchyManager
```

---

### Phase 6: Validation

**Step 6.1: Run all tests**
```bash
cd /home/jip/Documents/thesis
source venv/bin/activate
python3 -m pytest src/battery_twin/tests/ -v
```

**Step 6.2: Verify imports work**
```python
# Test script
from mas.core import AgentId, HybridAgent
from mas.communication import MqttTransport, TopicManager, MqttConfig
from mas.organization import RoleManager

print("All imports successful!")
```

**Step 6.3: Integration test with real broker**
```bash
# Start MQTT broker
docker run -d -p 1883:1883 eclipse-mosquitto

# Run integration test
python3 -m pytest src/battery_twin/tests/test_step3_mqtt.py -v
```

---

### Execution Order Summary

```
Phase 1: Create mas/ structure          [~2 hours]
  └─ 1.1 Create directories
  └─ 1.2 Migrate core (agent types)
  └─ 1.3 Migrate communication
  └─ 1.4 Migrate organization

Phase 2: Refactor communication         [~3 hours]
  └─ 2.1 Create MqttTransport
  └─ 2.2 Create MockTransport
  └─ 2.3 Update exports

Phase 3: Update battery_twin agents     [~2 hours]
  └─ 3.1 Refactor BatteryAgentBase
  └─ 3.2 Remove battery_agent_types.py
  └─ 3.3 Update concrete agents

Phase 4: Update tests                   [~1 hour]
  └─ 4.1 Create test fixtures
  └─ 4.2 Update existing tests

Phase 5: Clean up legacy                [~1 hour]
  └─ 5.1 Delete old files
  └─ 5.2 Remove sys.path hacks
  └─ 5.3 Update __init__.py

Phase 6: Validation                     [~30 min]
  └─ 6.1 Run tests
  └─ 6.2 Verify imports
  └─ 6.3 Integration test
```

---

## Example Usage

### Creating a Battery Twin Application

```python
from mas.core import AgentId, HybridAgent
from mas.communication import MqttTransport, TopicManager, MqttConfig
from battery_twin.agents import TelemetryIngestorAgent, PhysicsModelAgent
from battery_twin.communication.message_schemas import TelemetryMessage

# 1. Setup communication
mqtt_config = MqttConfig(broker="localhost", port=1883)
topic_manager = TopicManager("battery_twin/config/mqtt_topics.yaml")
transport = MqttTransport(mqtt_config, topic_manager)
transport.connect()

# 2. Create agents
telemetry_agent = TelemetryIngestorAgent(
    agent_id=AgentId("battery_twin", "telemetry_ingestor", "001"),
    transport=transport,
)

physics_agent = PhysicsModelAgent(
    agent_id=AgentId("battery_twin", "physics_model", "001"),
    transport=transport,
)

# 3. Setup communication between agents
telemetry_agent.setup()  # Subscribes to raw telemetry
physics_agent.setup()    # Subscribes to clean telemetry

# 4. Run (agents process messages via callbacks)
# ... application loop ...

# 5. Cleanup
transport.disconnect()
```

### Subscribing to Messages

```python
class PhysicsModelAgent(BatteryAgentBase):

    def _agent_setup(self) -> bool:
        # Subscribe to clean telemetry from all batteries
        self.transport.subscribe_to_topic(
            "clean_telemetry",
            self._on_telemetry,
            battery_id=None  # Wildcard: battery/+/telemetry/clean
        )
        return True

    def _on_telemetry(self, topic: str, payload: str):
        # Parse and process
        msg = TelemetryMessage.model_validate_json(payload)
        prediction = self._run_model(msg)

        # Publish result
        self.transport.publish_to_topic(
            "physics_prediction",
            prediction,
            battery_id=msg.battery_id
        )
```

---

## Testing Strategy

### Unit Tests with MockTransport

```python
class MockTransport(Transport):
    """In-memory transport for testing."""

    def __init__(self):
        self.published: List[Tuple[str, str]] = []
        self.subscriptions: Dict[str, Callable] = {}

    def publish(self, topic: str, message: BaseModel, qos: int = 1) -> bool:
        self.published.append((topic, message.model_dump_json()))
        return True

    def simulate_message(self, topic: str, payload: str):
        """Simulate incoming message for tests."""
        for pattern, callback in self.subscriptions.items():
            if mqtt.topic_matches_sub(pattern, topic):
                callback(topic, payload)
```

```python
def test_physics_agent_publishes_prediction():
    # Arrange
    mock_transport = MockTransport()
    agent = PhysicsModelAgent(transport=mock_transport)
    agent.setup()

    # Act
    mock_transport.simulate_message(
        "battery/B0005/telemetry/clean",
        TelemetryMessage(battery_id="B0005", voltage=3.8, ...).model_dump_json()
    )

    # Assert
    assert len(mock_transport.published) == 1
    topic, payload = mock_transport.published[0]
    assert "prediction/physics" in topic
```

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2025-12-08 | Initial specification |
