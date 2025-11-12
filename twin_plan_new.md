# Battery Digital Twin Integration Plan

In-depth roadmap to align the existing hybrid digital twin implementation (`Digital-Twin-in-python/`) with the multi-agent scaffold under `src/battery_twin/`, producing a coherent, testable, and extensible multi-agent battery twin.

---

## 1. Objectives

- Reuse the proven physics + ML hybrid models, data pipeline, and validation utilities from `Digital-Twin-in-python`.
- Ground the multi-agent layer (orchestrator + agents + MQTT + storage) on the original library’s APIs and data contracts.
- Deliver a runnable system that supports both programmatic access (current `HybridDigitalTwin` usage) and an agent-based runtime coordinating ingestion, estimation, prediction, and monitoring.
- Establish reliable configuration, storage, and messaging standards shared across both code paths.
- Provide automated tests and documentation that demonstrate the integrated multi-agent workflow.

---

## 2. Current-State Gap Summary

| Area | Original Hybrid Twin | Multi-Agent Scaffold | Gap |
| --- | --- | --- | --- |
| Core modeling | `HybridDigitalTwin` orchestrates physics + ML (`hybrid_digital_twin/models`) | Duplicated physics/EKF/residual implementations | No integration; redundant logic |
| Data handling | `BatteryDataLoader`, NASA pipelines | Telemetry agent expects MQTT + raw data | Different formats + no shared loader |
| Configuration | `pyproject` + `config/` YAML consumed by hybrid twin | `BatteryTwinConfig` expects flat fields; `battery_twin_config.yaml` nested | Incompatible schemas |
| Storage | Optional persistence via configs | Storage manager requires `BatteryStorageConfig`; orchestrator instantiates without config | Runtime failure if storage enabled |
| Messaging | None (single-process API) | MQTT topics defined but not connected to models | Need bridging layer |
| Testing | pytest suites in original repo | Multi-agent tests mostly structure/mocks | Lack of integration coverage |

---

## 3. Integration Strategy Overview

1. **Unify configuration**: define shared YAML schemas and loaders so both the multi-agent orchestrator and the original twin consume the same source of truth.
2. **Expose hybrid services**: wrap `HybridDigitalTwin` training/prediction entrypoints in agent-friendly interfaces.
3. **Normalize data flow**: ensure telemetry ingestion creates the datasets expected by the hybrid twin (feature engineering, batching).
4. **Refactor redundant models**: multi-agent modules should import the existing physics/ML components instead of maintaining separate copies.
5. **Align storage + messaging**: define how predictions, residuals, and state estimates are published, persisted, and consumed.
6. **Testing + validation**: orchestrate end-to-end tests combining MQTT simulation with hybrid predictions compared against baseline scripts.
7. **Documentation + operations**: document architecture, configs, and runbooks reflecting the integrated system.

---

## 4. Detailed Workstreams

### 4.1 Architecture Harmonization
- Map multi-agent responsibilities to hybrid twin capabilities (e.g., State Estimator ↔ EKF pipeline, Physics & ML agents ↔ hybrid model).
- Decide on deployment topology (single process with asyncio/mqtt loop vs. multi-process).
- Define shared domain objects (telemetry message schema, prediction payload) and ensure both codebases rely on them.
- Create component diagram capturing orchestrator, agents, hybrid services, storage, and external systems (MQTT broker, data sources).

### 4.2 Configuration & Dependency Alignment
- Design a unified configuration schema (YAML/JSON) covering:
  - System metadata, MQTT, storage, data sources, agent toggles, model hyperparameters.
- Implement loader utilities:
  - Respect nested structure (`system`, `mqtt`, `storage`, `agents`, etc.).
  - Provide adapters that materialize `BatteryTwinConfig` and the hybrid twin configs from the same document.
- Update orchestrator and tests to use the new schema; remove conflicting defaults.
- Ensure dependency versions (loguru, pydantic, numpy, torch, etc.) match to avoid runtime conflicts; consolidate requirements.

### 4.3 Agent Implementation Updates
- **Telemetry Ingestor**:
  - Replace ad-hoc validation with reusable utilities from `BatteryDataLoader`.
  - Support ingestion from MQTT or direct dataset replay using the shared loader.
  - Produce cleaned telemetry frames stored in a buffer for the state estimator.
- **State Estimator Agent**:
  - Integrate existing EKF implementation (import from `models/extended_kalman_filter` or reuse hybrid twin hooks if available).
  - Align output schema with `HybridDigitalTwin` expectations for downstream processors.
- **Physics & ML Residual Agents**:
  - Remove duplicate model code; instantiate `HybridDigitalTwin` or its subcomponents.
  - Define clear lifecycle: load pre-trained artifacts, subscribe for retraining triggers, publish predictions.
- **Health Monitor Agent**:
  - Consume predictions/state estimates to produce alerts consistent with hybrid twin metrics.
  - Optionally leverage existing evaluation utilities for SoH/RUL estimation.
- **Registry/Orchestrator**:
  - Ensure orchestrator initializes shared services (data loader, hybrid twin model registry).
  - Support dynamic agent enabling/disabling via unified config.

### 4.4 Storage Integration
- Refactor `BatteryStorageManager` to accept a properly instantiated `BatteryStorageConfig`; supply defaults derived from unified config.
- Provide adapters between agent message schemas and storage persistence (Influx, Mongo, Redis).
- Implement serialization for model artifacts (e.g., `HybridDigitalTwin` joblib files) and ensure versioning/metadata align with existing storage schemas.
- Document fallback behaviour when storage backends are disabled (no-op adapters or in-memory stores).

### 4.5 Communication & Topic Management
- Finalize MQTT topics for telemetry, state estimates, physics predictions, ML residuals, and hybrid outputs.
- Implement standardized payloads (reuse Pydantic models), including metadata (battery_id, cycle, timestamps, uncertainty).
- Ensure the hybrid twin training/prediction cycle can operate with MQTT as trigger:
  - e.g., Telemetry → State Estimator → publish features → Trigger Hybrid Agent -> publish predictions.
- Introduce topic-based access control and QoS configuration derived from config.
- Provide local loopback/testing utilities (mock MQTT broker or in-process pub/sub) for deterministic tests.

### 4.6 Data Pipeline & Feature Engineering
- Align telemetry schema with `BatteryDataLoader` expectations (column names, units).
- Build an intermediate feature store/buffer the hybrid model can query for cycles previously ingested.
- Support historical replay (NASA dataset) by streaming through the same telemetry path used in production (ensures parity).
- Validate that hybrid predictions remain numerically consistent with the original single-process pipeline; document acceptable tolerances.

### 4.7 Testing & Validation
- Create integration tests that:
  - Spin up orchestrator with mocked MQTT broker.
  - Feed sample telemetry and assert predictions match baseline `HybridDigitalTwin.predict` outputs.
  - Verify storage writes (use temporary Mongo/Influx fixtures or stubbed clients).
  - Exercise agent lifecycle: start, pause, resume, shutdown.
- Refine unit tests to target shared utilities rather than duplicated logic.
- Extend `Digital-Twin-in-python/tests` to include multi-agent scenarios, ensuring backward compatibility.
- Enable CI jobs running both unit + integration suites, with optional slow tests for dataset replay.

### 4.8 Documentation & Developer Enablement
- Update READMEs to explain:
  - Unified architecture and how agents map to hybrid twin functions.
  - Configuration instructions and environment setup.
  - Commands to run orchestrator (default/full/test modes) and compare with direct hybrid twin usage.
- Provide diagrams (sequence, component, deployment) reflecting the integrated system.
- Document troubleshooting steps (MQTT connection, storage availability, model artifacts).
- Offer quickstart notebook demonstrating telemetry replay through MQTT with live prediction charts.

### 4.9 Operational Considerations
- Define strategy for model lifecycle management:
  - How agents load/save models, retrain schedules, and rollback.
  - Integration with artifact storage (files, Mongo GridFS, S3).
- Establish monitoring hooks:
  - Metrics (agent uptime, message throughput, prediction latency) exposed via MQTT or Prometheus.
  - Health checks for MQTT and storage backends.
- Plan deployment packaging:
  - Containerize orchestrator + agents with dependencies resolved.
  - Provide docker-compose stack for broker + storages + orchestrator.

---

## 5. Phased Implementation Timeline

| Phase | Scope | Deliverables |
| --- | --- | --- |
| **Phase 0 – Foundations (Week 1)** | Inventory models/utilities; design unified configuration schema; dependency consolidation | Config spec draft, dependency matrix |
| **Phase 1 – Shared Services (Weeks 2-3)** | Implement config loaders, refactor storage manager constructors, integrate data loader utilities | `BatteryTwinConfig` v2, working storage initialization, validated data ingestion |
| **Phase 2 – Agent-Model Integration (Weeks 4-6)** | Adapt agents to use hybrid twin components, remove redundant models, establish message schemas | Updated agent classes, integrated hybrid predictions via agents |
| **Phase 3 – End-to-End Pipeline (Weeks 7-8)** | Wire MQTT flow, implement feature buffers, validate predictions vs baseline, enable storage writes | Demo pipeline, comparison test results |
| **Phase 4 – Testing & Ops (Weeks 9-10)** | Add integration tests, documentation, CI updates, operational scripts | Test suite coverage report, updated READMEs, docker-compose stack |
| **Phase 5 – Harden & Release (Weeks 11-12)** | Resolve residual bugs, performance tuning, release notes | GA multi-agent twin release |

---

### Phase 0 Detailed Tasks

> Goal: produce approved configuration schema draft and dependency inventory ready for implementation phases.

- **P0-T1 – Config Schema Workshop**: enumerate required sections/fields, map to current `BatteryTwinConfig` and `HybridDigitalTwin` options, capture open questions (MQTT, storage credentials, agent toggles).
- **P0-T2 – Schema Prototype**: draft a sample YAML document plus JSON-schema/Pydantic model skeleton that reflects consensus from P0-T1.
- **P0-T3 – Dependency Matrix**: list Python packages used in both codebases (versions, optional extras), highlight conflicts and upgrade needs.
- **P0-T4 – Service Inventory**: document expected external services (MQTT broker, Influx, Mongo, Redis) including local dev defaults vs production endpoints.
- **P0-T5 – Review Session**: schedule sign-off meeting (or async review) to approve schema prototype and dependency plan before coding.

---

## 6. Risk Mitigation

- **Model parity drift**: Regularly compare multi-agent predictions against baseline; enforce thresholds in CI.
- **Configuration sprawl**: Maintain single schema and generate typed objects programmatically to avoid divergence.
- **Third-party service availability**: Provide local mocks/stubs for MQTT and storage during development/testing.
- **Performance regressions**: Instrument agents with profiling hooks; run benchmarks after integration (leverage existing validation scripts).
- **Change management**: Introduce feature flags to toggle multi-agent features while maintaining backward compatibility.

---

## 7. Success Criteria

- Multi-agent orchestrator runs using the hybrid twin models with no duplicated logic.
- Configuration, storage, and messaging layers are shared and validated across both code paths.
- End-to-end telemetry replay through MQTT yields predictions matching the original `HybridDigitalTwin` within defined margins.
- Automated tests cover at least: config loading, agent lifecycle, MQTT flow, storage writes, prediction parity.
- Documentation enables new contributors to set up, run, and extend the integrated system confidently.

---

## 8. Next Immediate Actions

1. Review/approve unified config proposal and dependency alignment.
2. Create spike branch to prototype `BatteryTwinConfig` loader that reads existing `Digital-Twin-in-python` configs.
3. Identify redundant model modules in `src/battery_twin/models` and plan migrations to shared imports.
4. Schedule design review with stakeholders to confirm MQTT topic schemas and storage expectations before implementation.
