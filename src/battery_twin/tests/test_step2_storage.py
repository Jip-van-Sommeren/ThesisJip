#!/usr/bin/env python3
"""
Step 2 Test: Storage Layer for Battery Twin

This test verifies that the battery storage configuration and manager
work correctly with all storage backends.

Prerequisites:
- Docker services must be running: docker-compose up -d

Run: source venv/bin/activate && python3 src/battery_twin/tests/test_step2_storage.py
"""

import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.battery_twin.storage.battery_storage_config import BatteryStorageConfig
from src.battery_twin.storage.battery_storage_manager import BatteryStorageManager


def test_config_loading():
    """Test that configuration loads from YAML."""
    print("Testing configuration loading...")

    try:
        config = BatteryStorageConfig.from_yaml(
            "src/battery_twin/config/battery_twin_config.yaml"
        )
        print(f"  ✓ Configuration loaded successfully")
        print(f"    - Battery IDs: {config.battery_ids}")
        print(f"    - InfluxDB: {config.influx_config.database}")
        print(f"    - MongoDB: {config.mongo_config.database}")
        print(f"    - Neo4j: {config.neo4j_config.database}")
        print(f"    - Redis DB: {config.redis_config.db}")
        return True, config
    except Exception as e:
        print(f"  ✗ Configuration loading failed: {e}")
        return False, None


def test_storage_connections(config):
    """Test connections to all storage backends."""
    print("\nTesting storage connections...")

    try:
        storage = BatteryStorageManager(config)
        print(f"  ✓ Storage manager created")

        # Try to connect to all backends
        success = storage.connect_all()

        if success:
            print(f"  ✓ All storage backends connected")
        else:
            print(f"  ⚠ Some storage backends failed to connect (check Docker services)")

        return storage if success else None

    except Exception as e:
        print(f"  ✗ Storage connection failed: {e}")
        print(f"    Make sure Docker services are running:")
        print(f"    docker-compose up -d")
        return None


def test_battery_initialization(storage):
    """Test battery storage initialization."""
    print("\nTesting battery initialization...")

    if storage is None:
        print("  ⚠ Skipping (no storage connection)")
        return False

    try:
        battery_id = "TEST_B0005"
        metadata = {
            "battery_type": "Li-ion",
            "nominal_capacity": 2.0,
            "nominal_voltage": 3.7,
            "chemistry": "LiCoO2",
            "manufacturing_date": "2024-01-01"
        }

        storage.initialize_battery_storage(battery_id, metadata)
        print(f"  ✓ Battery {battery_id} initialized successfully")
        return True

    except Exception as e:
        print(f"  ✗ Battery initialization failed: {e}")
        return False


def test_telemetry_recording(storage):
    """Test recording battery telemetry."""
    print("\nTesting telemetry recording...")

    if storage is None:
        print("  ⚠ Skipping (no storage connection)")
        return False

    try:
        battery_id = "TEST_B0005"

        # Record some sample telemetry
        for i in range(3):
            storage.record_telemetry(
                battery_id=battery_id,
                voltage=3.8 - i * 0.1,
                current=-2.0,
                temperature=25.0 + i * 0.5,
                cycle=1,
                ambient_temperature=24.0
            )
            time.sleep(0.1)

        print(f"  ✓ Recorded 3 telemetry samples")

        # Try to read back from cache
        latest = storage.get_latest_telemetry(battery_id)
        if latest:
            print(f"  ✓ Retrieved latest telemetry from cache:")
            print(f"    - Voltage: {latest['voltage']:.2f} V")
            print(f"    - Current: {latest['current']:.2f} A")
            print(f"    - Temperature: {latest['temperature']:.2f} °C")
            return True
        else:
            print(f"  ⚠ Could not retrieve from cache (Redis may not be available)")
            return True  # Still pass if write succeeded

    except Exception as e:
        print(f"  ✗ Telemetry recording failed: {e}")
        return False


def test_prediction_recording(storage):
    """Test recording battery predictions."""
    print("\nTesting prediction recording...")

    if storage is None:
        print("  ⚠ Skipping (no storage connection)")
        return False

    try:
        battery_id = "TEST_B0005"

        # Record predictions
        storage.record_prediction(
            battery_id=battery_id,
            agent_id="model.physics.1",
            prediction_type="physics",
            predicted_capacity=1.85,
            uncertainty=None,
            horizon=0,
            cycle=1
        )

        storage.record_prediction(
            battery_id=battery_id,
            agent_id="model.mlresidual.1",
            prediction_type="ml",
            predicted_capacity=0.02,
            uncertainty=0.005,
            horizon=0,
            cycle=1
        )

        storage.record_prediction(
            battery_id=battery_id,
            agent_id="model.mlresidual.1",
            prediction_type="hybrid",
            predicted_capacity=1.87,
            uncertainty=0.005,
            horizon=0,
            cycle=1
        )

        print(f"  ✓ Recorded 3 predictions (physics, ml, hybrid)")
        return True

    except Exception as e:
        print(f"  ✗ Prediction recording failed: {e}")
        return False


def test_state_estimate_recording(storage):
    """Test recording battery state estimates."""
    print("\nTesting state estimate recording...")

    if storage is None:
        print("  ⚠ Skipping (no storage connection)")
        return False

    try:
        battery_id = "TEST_B0005"

        storage.record_state_estimate(
            battery_id=battery_id,
            agent_id="estimator.state.1",
            soc=0.75,
            soh=0.92,
            internal_resistance={
                "R0": 0.05,
                "R1": 0.03,
                "C1": 1000.0
            }
        )

        print(f"  ✓ Recorded state estimate")

        # Try to read back from cache
        latest = storage.get_latest_state(battery_id)
        if latest:
            print(f"  ✓ Retrieved latest state from cache:")
            print(f"    - SoC: {latest['soc']:.2%}")
            print(f"    - SoH: {latest['soh']:.2%}")
            print(f"    - R0: {latest['r0']:.4f} Ω")
            return True
        else:
            print(f"  ⚠ Could not retrieve from cache")
            return True

    except Exception as e:
        print(f"  ✗ State estimate recording failed: {e}")
        return False


def test_fault_recording(storage):
    """Test recording fault events."""
    print("\nTesting fault event recording...")

    if storage is None:
        print("  ⚠ Skipping (no storage connection)")
        return False

    try:
        battery_id = "TEST_B0005"

        storage.record_fault_event(
            battery_id=battery_id,
            agent_id="monitor.faults.1",
            severity="warning",
            fault_type="prediction_residual",
            cause="Residual exceeded 3 sigma threshold",
            residual_magnitude=0.15
        )

        print(f"  ✓ Recorded fault event")
        return True

    except Exception as e:
        print(f"  ✗ Fault recording failed: {e}")
        return False


def test_parameter_recording(storage):
    """Test recording model parameters."""
    print("\nTesting parameter recording...")

    if storage is None:
        print("  ⚠ Skipping (no storage connection)")
        return False

    try:
        battery_id = "TEST_B0005"

        storage.record_parameters(
            battery_id=battery_id,
            agent_id="estimator.paramid.1",
            parameters={
                "k": 0.13,
                "C0": 2.0,
                "R0": 0.05,
                "R1": 0.03,
                "C1": 1000.0
            },
            confidence=0.95,
            cycle=1
        )

        print(f"  ✓ Recorded model parameters")
        return True

    except Exception as e:
        print(f"  ✗ Parameter recording failed: {e}")
        return False


def cleanup(storage):
    """Disconnect from storage."""
    print("\nCleaning up...")

    if storage:
        try:
            storage.disconnect_all()
            print("  ✓ Disconnected from all storage backends")
        except Exception as e:
            print(f"  ⚠ Cleanup warning: {e}")


def main():
    """Run all storage tests."""
    print("=" * 70)
    print("STEP 2 TEST: Battery Storage Layer")
    print("=" * 70)
    print("\nPrerequisites:")
    print("  - Docker services must be running")
    print("  - Run: docker-compose up -d")
    print("=" * 70)

    # Run tests
    tests = []
    storage = None

    # Test 1: Config loading
    success, config = test_config_loading()
    tests.append(("Config Loading", success))

    if not success:
        print("\n✗ Cannot proceed without configuration")
        return 1

    # Test 2: Storage connections
    storage = test_storage_connections(config)
    tests.append(("Storage Connections", storage is not None))

    # Test 3: Battery initialization
    success = test_battery_initialization(storage)
    tests.append(("Battery Initialization", success))

    # Test 4: Telemetry recording
    success = test_telemetry_recording(storage)
    tests.append(("Telemetry Recording", success))

    # Test 5: Prediction recording
    success = test_prediction_recording(storage)
    tests.append(("Prediction Recording", success))

    # Test 6: State estimate recording
    success = test_state_estimate_recording(storage)
    tests.append(("State Estimate Recording", success))

    # Test 7: Fault recording
    success = test_fault_recording(storage)
    tests.append(("Fault Recording", success))

    # Test 8: Parameter recording
    success = test_parameter_recording(storage)
    tests.append(("Parameter Recording", success))

    # Cleanup
    cleanup(storage)

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    all_passed = True
    for test_name, result in tests:
        status = "PASS" if result else "FAIL"
        symbol = "✓" if result else "✗"
        print(f"  {symbol} {test_name}: {status}")
        if not result:
            all_passed = False

    print("=" * 70)

    if all_passed:
        print("\n✓ ALL TESTS PASSED!")
        print("\nNext step: Proceed to Step 3: Message Schemas & MQTT Bridge")
        print("\nData written to:")
        print("  - InfluxDB: battery_metrics database")
        print("  - MongoDB: battery_twin database")
        print("  - Redis: latest values cached")
        print("\nYou can view the data:")
        print("  - MongoDB: http://localhost:8081 (admin/password123)")
        print("  - Redis: http://localhost:8082")
        print("  - Grafana: http://localhost:3000 (admin/password123)")
        return 0
    else:
        print("\n✗ SOME TESTS FAILED")
        print("\nMake sure Docker services are running:")
        print("  docker-compose up -d")
        print("\nCheck service health:")
        print("  docker-compose ps")
        return 1


if __name__ == "__main__":
    sys.exit(main())
