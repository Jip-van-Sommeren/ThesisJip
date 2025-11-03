# Battery Digital Twin - Multi-Agent Migration Plan

## Overview

This document covers the migration of the monolithic Hybrid Digital Twin for Li-ion batteries into a multi-agent architecture while preserving the original feature set: dataset ingestion, physics-based degradation modeling, machine-learning residual correction, and training/evaluation flows.

**Source**: `Digital-Twin-in-python/` (monolithic implementation)  
**Target**: `src/battery_twin/` (multi-agent implementation)  
**Dataset**: NASA Battery Aging Dataset (`randomized_discharge_RUL_data_2_noNAN_new.csv`, 21 MB)

## Architecture

### Multi-Agent System

```
                ┌────────────────────────────┐
                │  HybridCoordinatorAgent    │
                │     (hybrid.coordinator)   │
                └─────────────┬──────────────┘
                              │
          ┌───────────────────┴─────────────┐
          │                                 │
   ┌──────▼─────────┐               ┌───────▼─────────┐
   │ PhysicsModel   │               │ MLCorrection    │
   │ Agent          │               │ Agent           │
   │ (physics.1)    │               │ (ml.1)          │
   └──────┬─────────┘               └───────┬─────────┘
          │                                 │
   ┌──────▼─────────┐               ┌───────▼──────────┐
   │ DataLoader     │               │ EvaluationAgent  │
   │ Agent          │               │ (evaluation.1)   │
   │ (data.1)       │               └──────────────────┘
   └────────────────┘
```

The coordinator orchestrates training and inference by exchanging MQTT messages with specialized agents that mirror the original modules. Each agent subscribes/publishes to well-defined topics so the system can be distributed or embedded without code changes.

### Agent Specifications

#### 1. DataLoaderAgent (Reactive)
```
A_data = ⟨
  Id: "data.loader.{N}",
  State: {dataset_cache, preprocessing_config},
  Goals: ∅,
  Perception: {event: REQUEST_DATASET},
  Action: {load_dataset(), split_data(), publish_batches()},
  Decision: rule_based
⟩
```
**Responsibilities**:
- Load NASA battery dataset or user-provided CSV via `BatteryDataLoader`
- Apply existing preprocessing/validation rules
- Provide train/validation/test splits and batched iterators
- Surface metadata (feature names, capacity ranges) to coordinator

**Refactored Code**:
- Reuse `hybrid_digital_twin/data/data_loader.py`
- Keep `validate_input_data` usage from `utils/validators.py`

---

#### 2. PhysicsModelAgent (Goal-driven)
```
A_physics = ⟨
  Id: "model.physics.{N}",
  State: {physics_model, fitted_params, training_metrics},
  Goals: {fit_physics_model, serve_predictions},
  Perception: {event: DATASET_AVAILABLE, event: REQUEST_PHYSICS_PREDICTION},
  Action: {fit(), predict(), export_state()},
  Decision: goal_driven
⟩
```
**Responsibilities**:
- Train and evaluate the exponential degradation model (`PhysicsBasedModel`)
- Serve physics-only predictions for both training and inference flows
- Persist fitted parameters for reuse

**Refactored Code**:
- Extract logic from `models/physics_model.py`
- Preserve configuration hooks defined in monolith config files

---

#### 3. MLCorrectionAgent (Goal-driven)
```
A_ml = ⟨
  Id: "model.ml.{N}",
  State: {ml_model, training_history, residual_cache},
  Goals: {fit_residual_model, minimize_error},
  Perception: {event: RESIDUALS_READY, event: REQUEST_ML_CORRECTION},
  Action: {fit(), predict(), track_metrics()},
  Decision: goal_driven
⟩
```
**Responsibilities**:
- Train residual neural network using existing architecture from `models/ml_model.py`
- Produce correction predictions given physics outputs and engineered features
- Monitor validation metrics to support coordinator decisions

**Refactored Code**:
- Reuse `MLCorrectionModel` and its training utilities
- Retain dropout/ensemble options already present in monolith

---

#### 4. HybridCoordinatorAgent (Hybrid)
```
A_coord = ⟨
  Id: "hybrid.coordinator",
  State: {run_config, training_plan, hybrid_metrics, persistence_paths},
  Goals: {orchestrate_training, deliver_hybrid_predictions},
  Perception: {event: DATASET_READY, PHYSICS_PREDICTED, ML_CORRECTED, METRICS_UPDATED},
  Action: {compute_residuals(), request_training(), assemble_predictions(), save_models()},
  Decision: rule_based + goal_monitoring
⟩
```
**Responsibilities**:
- Manage training/inference lifecycle analogous to `HybridDigitalTwin`
- Compute residuals and prepare feature tensors for ML agent
- Combine physics and ML outputs into hybrid predictions
- Serialize/deserialize model bundle (`joblib`) and emit status updates

**Refactored Code**:
- Wrap `HybridDigitalTwin` orchestration in agent interface
- Reuse `ModelMetrics` for tracking combined performance
- Keep CLI integration (`cli.py`) by delegating to this agent

---

#### 5. EvaluationAgent (Reactive)
```
A_eval = ⟨
  Id: "evaluation.{N}",
  State: {metrics_config, report_store},
  Goals: ∅,
  Perception: {event: HYBRID_RESULTS_READY},
  Action: {calculate_metrics(), generate_reports()},
  Decision: rule_based
⟩
```
**Responsibilities**:
- Compute RMSE/MAE/R²/MAPE via `ModelMetrics`
- Generate optional plots using `visualization/plotters.py`
- Provide evaluation summaries back to coordinator/CLI

**Refactored Code**:
- Reuse `utils/metrics.py` and optional `visualization` helpers
- Persist evaluation reports alongside saved model artifacts

---

## Communication Pattern

- **Transport**: MQTT 3.1.1 broker (e.g., Mosquitto) running locally via Docker Compose. Agents connect using async clients and leverage QoS 1 for reliable delivery.
- **Topic Namespace**:
  - `twin/data/request` – Coordinator requests dataset load/slice.
  - `twin/data/ready` – DataLoaderAgent publishes dataset handles/metadata.
  - `twin/physics/train` / `twin/physics/prediction` – PhysicsModelAgent coordination and outputs.
  - `twin/ml/train` / `twin/ml/prediction` – MLCorrectionAgent coordination and outputs.
  - `twin/hybrid/results` – Coordinator publishes combined predictions.
  - `twin/evaluation/metrics` – EvaluationAgent posts metric summaries.
  - `twin/system/status` – Optional heartbeat/status updates for observability.
- **Message Schemas** mirror the monolith’s data structures (NumPy arrays serialized to lists, metadata dicts). Example residual payload:
```json
{
  "batch_id": 12,
  "physics_prediction": [2.81, 2.79, 2.75],
  "actual_capacity": [2.80, 2.77, 2.74],
  "features": {...},
  "metadata": {"battery_id": "B0005", "split": "train"}
}
```
- **Configuration**: YAML/JSON configs reused; broker URI/credentials added alongside existing training parameters so deployments can switch transports without code changes.

## Pipeline Flows

### Training

1. **Coordinator** publishes `twin/data/request` → DataLoaderAgent loads NASA CSV, applies preprocessing, and publishes `twin/data/ready` with split handles.
2. **PhysicsModelAgent** subscribes to `twin/physics/train`, executes `fit()`, and publishes `twin/physics/prediction` messages for train/validation sets.
3. **Coordinator** consumes physics predictions, computes residuals, and publishes feature tensors to `twin/ml/train`.
4. **MLCorrectionAgent** trains residual network, reports progress on `twin/system/status`, and publishes correction outputs to `twin/ml/prediction`.
5. **Coordinator** combines physics + ML outputs, posts hybrid predictions on `twin/hybrid/results`.
6. **EvaluationAgent** listens on `twin/hybrid/results`, calculates metrics, and publishes summaries to `twin/evaluation/metrics`.
7. **Coordinator** receives metrics, persists artifacts (`joblib`), and emits final status updates.

### Inference

1. CLI/Coordinator publishes data request; DataLoaderAgent responds with batch payloads on `twin/data/ready`.
2. PhysicsModelAgent streams predictions on `twin/physics/prediction`.
3. Coordinator assembles ML features, requests corrections via `twin/ml/prediction`.
4. EvaluationAgent optionally validates outputs and reports via `twin/evaluation/metrics`.
5. Coordinator forwards hybrid predictions back to CLI/API client.

## Implementation Phases

### Phase 1: Foundations (8–10 hours)
- Step 1: Audit monolith modules to confirm reusable entry points (`HybridDigitalTwin`, `BatteryDataLoader`, models, metrics).
- Step 2: Provision Docker-based Mosquitto broker, implement base `Agent` interface, and load shared YAML configs (including broker credentials).
- Step 3: Set up testing scaffolding mirroring monolith unit tests plus MQTT connection smoke tests.

### Phase 2: Data & Physics Integration (10–12 hours)
- Step 4: Build DataLoaderAgent around `BatteryDataLoader`, add validation unit tests, and publish dataset payloads over MQTT.
- Step 5: Encapsulate `PhysicsBasedModel` inside PhysicsModelAgent with training/prediction handlers and MQTT topic wiring.
- Step 6: Wire coordinator to manage physics training via MQTT messages and persistence parity checks.

### Phase 3: ML & Hybrid Orchestration (12–14 hours)
- Step 7: Implement residual feature pipeline and message serialization consistent with monolith `_extract_ml_features`.
- Step 8: Wrap `MLCorrectionModel` with MLCorrectionAgent, including training history capture and MQTT streaming.
- Step 9: Finalize HybridCoordinatorAgent to merge predictions, compute metrics, and expose CLI endpoints backed by MQTT flows.

### Phase 4: Evaluation & Tooling (8–10 hours)
- Step 10: Add EvaluationAgent integrating `ModelMetrics`, optional plotting helpers, and metric publication topics.
- Step 11: Update CLI commands to delegate to coordinator while managing MQTT lifecycle (connect/disconnect).
- Step 12: Document agent APIs, configuration examples, and run end-to-end parity tests with NASA dataset over MQTT.

**Total Estimated Time**: 40–48 hours (5–6 working days)

## Success Criteria

### Functional
- Agents reproduce monolith training and inference outputs within tolerance (capacity RMSE variation ≤ ±1%).
- CLI commands (`train`, `predict`, `evaluate`) continue to operate using the new multi-agent backend.
- Model artifacts saved by coordinator are loadable via the monolith’s `load_model` routines.

### Non-Functional
- MQTT round-trip latency ≤ 100 ms on local broker.
- End-to-end training runtime within ±10% of monolith baseline.
- Codebase retains existing logging and error-handling behaviors; MQTT errors surfaced with actionable context.

### Documentation
- Updated developer guide describing agent lifecycle and configuration.
- Sequence diagrams or tables illustrating event flow for training/inference.
- Parity report comparing monolith vs multi-agent metrics on NASA dataset.

## Risks & Mitigations

- **Risk**: MQTT orchestration diverges from monolith execution order, leading to metric drift.  
  **Mitigation**: Mirror `HybridDigitalTwin.fit/predict` sequence and validate topic ordering in integration tests.
- **Risk**: Shared state duplication across agents increases memory footprint.  
  **Mitigation**: Use lightweight data references (NumPy views, iterators) and centralized cache in DataLoaderAgent.
- **Risk**: Broker connectivity issues impact CLI workflows.  
  **Mitigation**: Provide local Mosquitto docker-compose service, retry logic, and offline fallback mode for tests.

## Next Steps

- ✅ Foundations ready for kickoff.
- 🔜 Begin Phase 1, Step 1 upon approval.

## References

1. Xu, J., et al. (2016). "Modeling of Lithium-Ion Battery Degradation for Cell Life Assessment".
2. NASA PCoE Battery Aging Dataset: https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/
3. MQTT v3.1.1 Specification: https://mqtt.org/
4. Original Monolithic Implementation: `Digital-Twin-in-python/`

---

**Document Version**: 2.0  
**Last Updated**: 2025-03-01  
**Status**: Planning (0/12 steps completed)
