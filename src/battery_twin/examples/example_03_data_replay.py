"""
Example 3: Data Replay from NASA Dataset

This example demonstrates how to replay historical NASA battery data
through the digital twin system for testing and validation.

Usage:
    python3 src/battery_twin/examples/example_03_data_replay.py
"""

import asyncio
from loguru import logger

from src.battery_twin.data.nasa_loader import NASABatteryLoader
from src.battery_twin.data.replay_engine import ReplayEngine
from src.battery_twin.orchestrator import (
    BatteryTwinConfig,
    BatteryTwinOrchestrator,
)


async def main():
    """
    Replay NASA battery data through the system.

    This example:
    1. Loads NASA battery dataset
    2. Initializes the digital twin system
    3. Replays 50 cycles through the system at 10x speed
    4. Monitors state estimates and predictions
    """
    logger.info("=" * 80)
    logger.info("EXAMPLE 3: Data Replay from NASA Dataset")
    logger.info("=" * 80)

    # Configuration
    battery_id = "B0005"
    num_cycles = 5  # Reduced from 50 for reasonable demo time
    replay_speed = 100.0  # 100x real-time speed (or use BATCH mode)

    # Step 1: Load NASA dataset
    logger.info(f"Loading NASA dataset for battery {battery_id}...")
    try:
        loader = NASABatteryLoader()
        cycles = loader.load_battery(battery_id)
        logger.info(f"✓ Loaded {len(cycles)} cycles")

        # Show dataset info
        info = loader.get_dataset_info(battery_id)
        logger.info(f"\nDataset Info:")
        logger.info(f"  Battery: {info.battery_id}")
        logger.info(f"  Cycles: {info.n_cycles}")
        logger.info(f"  Measurements: {info.n_total_samples:,}")
        logger.info(f"  Cycle Range: {info.cycle_range[0]} to {info.cycle_range[1]}")
        logger.info(
            f"  Capacity Range: {info.capacity_range[0]:.3f} - {info.capacity_range[1]:.3f} Ah"
        )

    except FileNotFoundError:
        logger.error("NASA dataset not found!")
        logger.error(
            "Please ensure discharge.csv is at: Digital-Twin-in-python/data/raw/discharge.csv"
        )
        return

    # Step 2: Create and initialize orchestrator
    logger.info("\nInitializing digital twin system...")
    config = BatteryTwinConfig(
        battery_id=battery_id,
        enable_telemetry_ingestor=True,
        enable_state_estimator=True,
        enable_health_monitor=True,
        enable_physics_model=True,
        enable_ml_residual=False,  # Disable ML for faster replay
        enable_storage=False,
        log_level="INFO",
    )

    orchestrator = BatteryTwinOrchestrator(config)
    await orchestrator.initialize()
    await orchestrator.start()
    logger.info("✓ System running")

    # Step 3: Create replay engine
    logger.info(f"\nPreparing to replay {num_cycles} cycles...")
    replay_engine = ReplayEngine(
        loader=loader,
        mqtt_bridge=orchestrator.mqtt_bridge,
    )

    # Step 4: Replay data
    from src.battery_twin.data.replay_engine import ReplayMode

    # Note: Use BATCH mode for fastest replay (no delays between samples)
    # Or use FAST mode with high speed_multiplier for time-scaled replay
    use_batch_mode = True  # Set to False to use FAST mode with speed_multiplier

    if use_batch_mode:
        logger.info("Starting data replay in BATCH mode (maximum speed)...")
        replay_mode = ReplayMode.BATCH
        speed = 1.0
    else:
        logger.info(f"Starting data replay at {replay_speed}x speed...")
        replay_mode = ReplayMode.FAST
        speed = replay_speed

    try:
        # Run replay in background thread to avoid blocking async event loop
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(
                replay_engine.replay_battery,
                battery_id=battery_id,
                mode=replay_mode,
                speed_multiplier=speed,
                start_cycle=0,
                end_cycle=num_cycles - 1,
                blocking=True,
            )

            # Wait for completion
            success = await asyncio.get_event_loop().run_in_executor(
                None, future.result
            )

        if success:
            logger.success(f"✓ Replayed {num_cycles} cycles successfully")
        else:
            logger.error("Replay failed to start")

    except Exception as e:
        logger.error(f"Replay failed: {e}")
        import traceback

        traceback.print_exc()

    # Wait for final processing
    logger.info("Waiting for final message processing...")
    await asyncio.sleep(5.0)

    # Step 5: Show final statistics
    status = orchestrator.get_status()
    logger.info("\nFinal Statistics:")
    logger.info(f"  Runtime: {status['uptime_seconds']:.1f}s")
    logger.info(f"  Messages Processed:")
    for agent_name, agent_status in status["agents"].items():
        logger.info(f"    - {agent_name}: {agent_status['message_count']}")

    # Step 6: Shutdown
    logger.info("\nShutting down...")
    await orchestrator.shutdown()
    logger.success("✓ Shutdown complete")

    logger.info("=" * 80)
    logger.info("EXAMPLE COMPLETE")
    logger.info("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
