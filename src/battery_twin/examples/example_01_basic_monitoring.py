"""
Example 1: Basic Battery Monitoring

This example demonstrates how to run a basic battery monitoring system
with telemetry ingestion, state estimation, and health monitoring.

Usage:
    python3 src/battery_twin/examples/example_01_basic_monitoring.py
"""

import asyncio
from loguru import logger

from src.battery_twin.orchestrator import BatteryTwinConfig, BatteryTwinOrchestrator


async def main():
    """
    Run basic battery monitoring for 5 minutes.

    This example:
    1. Creates a configuration with core agents
    2. Initializes the orchestrator
    3. Starts all agents
    4. Runs for 5 minutes
    5. Gracefully shuts down
    """
    logger.info("=" * 80)
    logger.info("EXAMPLE 1: Basic Battery Monitoring")
    logger.info("=" * 80)

    # Step 1: Create configuration
    logger.info("Creating configuration...")
    config = BatteryTwinConfig(
        battery_id='B0005',
        enable_telemetry_ingestor=True,
        enable_state_estimator=True,
        enable_health_monitor=True,
        enable_physics_model=False,  # Disable for basic monitoring
        enable_ml_residual=False,     # Disable for basic monitoring
        enable_storage=False,         # Disable storage for simplicity
        log_level='INFO',
        enable_metrics=True,
        metrics_interval=10.0
    )
    logger.info(f"✓ Configuration created for battery {config.battery_id}")

    # Step 2: Create orchestrator
    logger.info("Creating orchestrator...")
    orchestrator = BatteryTwinOrchestrator(config)

    # Step 3: Initialize
    logger.info("Initializing system...")
    await orchestrator.initialize()
    logger.info("✓ System initialized")

    # Step 4: Start agents
    logger.info("Starting agents...")
    await orchestrator.start()
    logger.info("✓ All agents running")

    # Print system status
    status = orchestrator.get_status()
    logger.info(f"System State: {status['system_state']}")
    logger.info(f"Battery ID: {status['battery_id']}")
    logger.info(f"Agents Running: {len([a for a in status['agents'].values() if a['status'] == 'RUNNING'])}")
    for agent_name, agent_status in status['agents'].items():
        logger.info(f"  - {agent_name}: {agent_status['status']}")

    # Step 5: Run for 5 minutes (300 seconds)
    logger.info("Running system for 5 minutes...")
    logger.info("Press Ctrl+C to stop early")

    try:
        await orchestrator.run(duration=300.0)
    except KeyboardInterrupt:
        logger.warning("Interrupted by user")

    # Step 6: Shutdown
    logger.info("Shutting down...")
    await orchestrator.shutdown()
    logger.success("✓ Shutdown complete")

    logger.info("=" * 80)
    logger.info("EXAMPLE COMPLETE")
    logger.info("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
