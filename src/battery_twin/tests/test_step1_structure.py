#!/usr/bin/env python3
"""
Step 1 Test: Verify Project Structure and Configuration

This test verifies that all directories, configuration files,
and Docker setup are correctly created.

Run: python src/battery_twin/tests/test_step1_structure.py
"""

import os
import sys
import yaml
from pathlib import Path


def test_directory_structure():
    """Test that all required directories exist."""
    print("Testing directory structure...")

    base_path = Path("src/battery_twin")
    required_dirs = [
        base_path,
        base_path / "agents",
        base_path / "models",
        base_path / "communication",
        base_path / "storage",
        base_path / "data",
        base_path / "config",
        base_path / "tests",
    ]

    all_exist = True
    for directory in required_dirs:
        if directory.exists() and directory.is_dir():
            print(f"  ✓ {directory}")
        else:
            print(f"  ✗ {directory} - MISSING")
            all_exist = False

    return all_exist


def test_init_files():
    """Test that __init__.py files exist in all packages."""
    print("\nTesting __init__.py files...")

    required_inits = [
        "src/battery_twin/__init__.py",
        "src/battery_twin/agents/__init__.py",
        "src/battery_twin/models/__init__.py",
        "src/battery_twin/communication/__init__.py",
        "src/battery_twin/storage/__init__.py",
        "src/battery_twin/data/__init__.py",
    ]

    all_exist = True
    for init_file in required_inits:
        if Path(init_file).exists():
            print(f"  ✓ {init_file}")
        else:
            print(f"  ✗ {init_file} - MISSING")
            all_exist = False

    return all_exist


def test_config_files():
    """Test that configuration files exist and are valid YAML."""
    print("\nTesting configuration files...")

    config_files = {
        "src/battery_twin/config/battery_twin_config.yaml": "main config",
        "src/battery_twin/config/mqtt_topics.yaml": "MQTT topics",
        "src/battery_twin/config/hierarchy_config.yaml": "hierarchy config",
        "config/mosquitto.conf": "Mosquitto config",
    }

    all_valid = True
    for config_path, description in config_files.items():
        path = Path(config_path)
        if not path.exists():
            print(f"  ✗ {description} ({config_path}) - MISSING")
            all_valid = False
            continue

        # Try to parse YAML files
        if config_path.endswith('.yaml'):
            try:
                with open(path, 'r') as f:
                    config = yaml.safe_load(f)
                print(f"  ✓ {description} ({config_path}) - valid YAML")
            except yaml.YAMLError as e:
                print(f"  ✗ {description} ({config_path}) - INVALID YAML: {e}")
                all_valid = False
        else:
            # Just check existence for non-YAML files
            print(f"  ✓ {description} ({config_path}) - exists")

    return all_valid


def test_config_content():
    """Test that configuration files contain expected keys."""
    print("\nTesting configuration content...")

    # Test main config
    with open("src/battery_twin/config/battery_twin_config.yaml") as f:
        main_config = yaml.safe_load(f)

    required_keys = ["system", "mqtt", "storage", "data", "agents"]
    missing_keys = [k for k in required_keys if k not in main_config]

    if missing_keys:
        print(f"  ✗ Main config missing keys: {missing_keys}")
        return False
    else:
        print(f"  ✓ Main config has all required keys")

    # Test MQTT topics config
    with open("src/battery_twin/config/mqtt_topics.yaml") as f:
        mqtt_config = yaml.safe_load(f)

    if "topics" not in mqtt_config:
        print(f"  ✗ MQTT topics config missing 'topics' key")
        return False

    required_topics = [
        "raw_telemetry", "clean_telemetry", "physics_prediction",
        "ml_correction", "hybrid_prediction", "state_estimate",
        "parameters", "faults", "control_command"
    ]
    missing_topics = [t for t in required_topics if t not in mqtt_config["topics"]]

    if missing_topics:
        print(f"  ✗ MQTT topics config missing: {missing_topics}")
        return False
    else:
        print(f"  ✓ MQTT topics config has all required topics")

    # Test hierarchy config
    with open("src/battery_twin/config/hierarchy_config.yaml") as f:
        hierarchy_config = yaml.safe_load(f)

    if "hierarchy" not in hierarchy_config:
        print(f"  ✗ Hierarchy config missing 'hierarchy' key")
        return False

    if "agents" not in hierarchy_config["hierarchy"]:
        print(f"  ✗ Hierarchy config missing 'agents' key")
        return False

    expected_agents = [
        "orchestrator.1", "registry.1", "telemetry.ingestor.1",
        "model.physics.1", "model.mlresidual.1", "estimator.state.1",
        "estimator.paramid.1", "monitor.faults.1", "controller.charge.1"
    ]

    agent_ids = list(hierarchy_config["hierarchy"]["agents"].keys())
    missing_agents = [a for a in expected_agents if a not in agent_ids]

    if missing_agents:
        print(f"  ✗ Hierarchy config missing agents: {missing_agents}")
        return False
    else:
        print(f"  ✓ Hierarchy config has all 9 expected agents")

    # Verify orchestrator is root
    if hierarchy_config["hierarchy"]["root"] != "orchestrator.1":
        print(f"  ✗ Hierarchy root is not orchestrator.1")
        return False
    else:
        print(f"  ✓ Hierarchy root is correctly set to orchestrator.1")

    return True


def test_docker_compose():
    """Test that docker-compose.yml includes Mosquitto."""
    print("\nTesting Docker Compose configuration...")

    docker_compose_path = Path("docker-compose.yml")
    if not docker_compose_path.exists():
        print("  ✗ docker-compose.yml not found")
        return False

    with open(docker_compose_path, 'r') as f:
        content = f.read()

    # Check for Mosquitto service
    if "mosquitto:" in content:
        print("  ✓ Mosquitto service found in docker-compose.yml")
    else:
        print("  ✗ Mosquitto service NOT found in docker-compose.yml")
        return False

    # Check for MQTT ports
    if "1883:1883" in content:
        print("  ✓ MQTT port 1883 configured")
    else:
        print("  ✗ MQTT port 1883 NOT configured")
        return False

    # Check for Mosquitto volumes
    if "mosquitto_data:" in content:
        print("  ✓ Mosquitto data volume configured")
    else:
        print("  ✗ Mosquitto data volume NOT configured")
        return False

    return True


def main():
    """Run all tests."""
    print("=" * 70)
    print("STEP 1 TEST: Project Structure and Configuration")
    print("=" * 70)

    tests = [
        ("Directory Structure", test_directory_structure),
        ("Init Files", test_init_files),
        ("Config Files", test_config_files),
        ("Config Content", test_config_content),
        ("Docker Compose", test_docker_compose),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n  ✗ {test_name} FAILED with exception: {e}")
            results.append((test_name, False))

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    all_passed = True
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        symbol = "✓" if result else "✗"
        print(f"  {symbol} {test_name}: {status}")
        if not result:
            all_passed = False

    print("=" * 70)

    if all_passed:
        print("\n✓ ALL TESTS PASSED!")
        print("\nNext step: Run 'docker-compose up -d' to start services,")
        print("then proceed to Step 2: Storage Layer Setup")
        return 0
    else:
        print("\n✗ SOME TESTS FAILED")
        print("\nPlease fix the issues above before proceeding.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
