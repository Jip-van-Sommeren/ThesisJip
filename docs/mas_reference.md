# MAS core reference

This document summarizes the generic MAS toolkit in `src/mas` and how it is used by the battery digital twin (see `sections/sec_battery_twin_mas.tex`). It is intended as a quick reference for the core MAS abstractions, module layout, and extension points.

## Formal model mapping

The MAS implementation follows the tuple:

A = (id, State, Goal, Perception, Action, Decision)

Mapping to code in `src/mas/core`:

- id -> `AgentId`
- State -> `State` and `Belief`
- Goal -> `Goal` and `GoalType`
- Perception -> `Perception.observe` and `Perception.update_state`
- Action -> `Action` and `ActionType`
- Decision -> `Decision` and `ReactiveRule`

Every concrete agent implements a sense-decide-act loop via `Agent.step`.

## Package layout

- `src/mas/core`: formal agent tuple, base `Agent`, and agent styles (Reactive, BDI, Hybrid).
- `src/mas/communication`: transport abstraction, MQTT implementation, mock transport, mailbox, and message schemas.
- `src/mas/organization`: roles, groups, and hierarchy management.
- `src/mas/__init__.py`: re-exports the public MAS API.

## Core layer (`src/mas/core`)

### `agent.py`

Key types:

- `AgentId(app, type, instance)`: hierarchical identity string `app.type.instance` (UUID if instance missing).
- `Belief(proposition, confidence, timestamp)`: belief with confidence in [0, 1].
- `State`: internal and external belief stores plus limited history.
- `Goal(condition, goal_type, priority, deadline, active)`.
- `Action(action_id, action_type, preconditions, effects)`.
- `Perception(observable_properties)`: filters environment to observations and updates beliefs.
- `Decision`: reactive rules plus simple deliberation (`deliberate` picks first available action).
- `Agent`: base class implementing `perceive`, `decide_action`, `execute_action`, `step`.

### `reactive_agent.py`

- `ReactiveAgent`: stimulus-response agent.
- Uses only reactive rules, no goals. Default action is `noop`.
- `add_stimulus_response` is a convenience helper for rule registration.

### `bdi_agent.py`

- `BDIAgent`: beliefs-desires-intentions agent.
- `Intention` holds a goal and an action plan (list of action IDs).
- Plans are selected from a simple `plan_library` and executed step-by-step.
- `select_intentions` caps concurrent intentions (`max_intentions = 3`).

### `hybrid_agent.py`

- `HybridAgent`: layered reactive plus deliberative agent.
- Emergency reactive rules are evaluated first (default `emergency_stop`).
- Deliberative layer uses BDI-style intentions and plan library.
- Layer stats track emergency, reactive, and deliberative actions.

## Communication layer (`src/mas/communication`)

### Transport and MQTT

- `Transport`: abstract publish/subscribe interface.
- `MqttConfig`: MQTT connection and QoS settings.
- `MqttTransport`: paho-mqtt based transport with reconnect support and optional TLS.
- `MockTransport`: in-memory transport for tests; captures published messages and can simulate incoming messages.

### Topic manager

- `TopicManager`: loads topic templates from YAML with a required `topics:` key.
- Supports template formatting (`get_topic`), wildcard subscription (`get_subscription_pattern`), and parsing (`parse_topic`).

### Mailbox

- `Mailbox`: thread-safe message buffer for perception, with max size and drop stats.
- Supports `get_all`, `get`, `peek`, and `filter_by_topic`.

### Message schemas

- `AgentMessage`: base Pydantic model with id, sender, timestamp, and extension fields.
- `AgentStatusMessage`, `AgentHeartbeatMessage`, `CommandMessage`, `ResponseMessage`.

## Organization layer (`src/mas/organization`)

### Roles

- `Role`: responsibilities, permissions, requirements, expectations.
- `RoleManager`: registers roles and assigns them to agents.

### Groups

- `Group`: name, purpose, role types, agent membership, optional max size.
- `GroupManager`: creates groups, assigns agents, validates role compatibility.

### Hierarchy

- `HierarchyManager`: tree structure with supervisor/subordinate relations.
- `OrganizationalPosition`: roles, supervisor, subordinates, groups, hierarchy level.
- `HierarchyMessage` and `MessageType` for command, report, escalation, delegation, query, response.

## Battery twin usage (from `sections/sec_battery_twin_mas.tex`)

The battery digital twin builds on the generic MAS toolkit:

- Uses `Transport` with `MqttTransport` and a YAML topic map (`TopicManager`).
- Uses `Mailbox` for decoupled perception where needed.
- Implements role-specialized agents (Telemetry Ingestor, State Estimator, Health Monitor, Physics Model, ML Residual) by subclassing reactive, BDI, or hybrid agents.
- Coordination is primarily via topics; hierarchy and roles are available for structured control if needed.

## Minimal usage example

```python
from mas.core import AgentId, HybridAgent
from mas.communication import MqttTransport, MqttConfig, TopicManager

config = MqttConfig(broker="localhost", port=1883)
topic_manager = TopicManager("src/battery_twin/config/mqtt_topics.yaml")
transport = MqttTransport(config, topic_manager)
transport.connect()

agent_id = AgentId("battery_twin", "telemetry_ingestor", "B0005")
agent = HybridAgent(agent_id, observable_properties={"voltage", "current"})
```

## Extension points

- Add new agent types by subclassing `Agent` or extending `ReactiveAgent`, `BDIAgent`, or `HybridAgent`.
- Define new message schemas by extending `AgentMessage`.
- Add new topics in the YAML file used by `TopicManager`.
- Use `MockTransport` for deterministic unit tests without a broker.
