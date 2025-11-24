"""
Example 2: Full System with All Agents

This example demonstrates running the complete battery digital twin system
with all agents enabled, including physics model and ML residual correction.

Usage:
    python3 src/battery_twin/examples/example_02_full_system.py
"""

import asyncio
from loguru import logger

from src.battery_twin.orchestrator import (
    BatteryTwinConfig,
    BatteryTwinOrchestrator,
)


async def main():
    """
    Run full battery digital twin system.

    This example demonstrates:
    1. All agents enabled (telemetry, state, health, physics, ML)
    2. Hybrid prediction (physics + ML)
    3. Complete monitoring and prediction pipeline
    """
    logger.info("=" * 80)
    logger.info("EXAMPLE 2: Full System with All Agents")
    logger.info("=" * 80)

    # Create configuration with all agents enabled
    logger.info("Creating full system configuration...")
    config = BatteryTwinConfig(
        battery_id="B0005",
        enable_telemetry_ingestor=True,
        enable_state_estimator=True,
        enable_health_monitor=True,
        enable_physics_model=True,  # Enable physics predictions
        enable_ml_residual=True,  # Enable ML corrections
        enable_storage=False,  # Disable storage for simplicity
        log_level="INFO",
        enable_metrics=True,
        metrics_interval=15.0,
    )
    logger.info("✓ Full system configuration created")

    # Create orchestrator
    orchestrator = BatteryTwinOrchestrator(config)

    # Initialize
    logger.info("Initializing full system...")
    await orchestrator.initialize()
    logger.info("✓ System initialized")

    # Start all agents
    logger.info("Starting all agents...")
    await orchestrator.start()

    # Print detailed status
    status = orchestrator.get_status()
    running_agents = len(
        [a for a in status["agents"].values() if a["status"] == "RUNNING"]
    )
    total_agents = len(status["agents"])
    logger.info(f"\nSystem Status:")
    logger.info(f"  State: {status['system_state']}")
    logger.info(f"  Battery ID: {status['battery_id']}")
    logger.info(f"  Agents Running: {running_agents}/{total_agents}")
    logger.info(f"  Uptime: {status['uptime_seconds']:.1f}s")
    logger.info(f"\nAgent Details:")
    for agent_name, agent_status in status["agents"].items():
        logger.info(f"  - {agent_name}:")
        logger.info(f"      Status: {agent_status['status']}")
        logger.info(f"      Type: {agent_status['type']}")
        logger.info(f"      Messages: {agent_status['message_count']}")

    # Run for 10 minutes
    logger.info("\nRunning full system for 10 minutes...")
    logger.info("Press Ctrl+C to stop early")

    try:
        await orchestrator.run(duration=600.0)
    except KeyboardInterrupt:
        logger.warning("Interrupted by user")

    # Final status
    final_status = orchestrator.get_status()
    logger.info(f"\nFinal Statistics:")
    logger.info(f"  Total Runtime: {final_status['uptime_seconds']:.1f}s")
    logger.info(f"  Total Messages Processed:")
    for agent_name, agent_status in final_status["agents"].items():
        logger.info(f"    - {agent_name}: {agent_status['message_count']}")

    # Shutdown
    logger.info("\nShutting down...")
    await orchestrator.shutdown()
    logger.success("✓ Shutdown complete")

    logger.info("=" * 80)
    logger.info("EXAMPLE COMPLETE")
    logger.info("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
