"""
Example 4: Subscribe to MQTT Messages

This example demonstrates how to subscribe to state estimates, health reports,
and predictions from the battery digital twin system.

Usage:
    python3 src/battery_twin/examples/example_04_subscribe_to_messages.py
"""

import asyncio
import json
from datetime import datetime
from loguru import logger

from src.battery_twin.communication.mqtt_bridge import MqttBridge, MqttConfig


async def main():
    """
    Subscribe to battery digital twin messages.

    This example shows how to:
    1. Connect to MQTT broker
    2. Subscribe to multiple topics
    3. Process incoming messages
    4. Display real-time updates
    """
    logger.info("=" * 80)
    logger.info("EXAMPLE 4: Subscribe to MQTT Messages")
    logger.info("=" * 80)

    battery_id = 'B0005'

    # Message counters
    telemetry_count = 0
    state_estimate_count = 0
    health_report_count = 0
    prediction_count = 0

    # Step 1: Connect to MQTT
    logger.info("Connecting to MQTT broker...")
    mqtt_config = MqttConfig(
        broker="localhost",
        port=1883,
        client_id_prefix="example_subscriber_"
    )
    mqtt_bridge = MqttBridge(mqtt_config)
    await mqtt_bridge.connect()
    logger.info("✓ Connected to MQTT")

    # Step 2: Define message handlers
    def on_telemetry(message: str):
        """Handle incoming telemetry messages."""
        nonlocal telemetry_count
        telemetry_count += 1

        data = json.loads(message)
        logger.info(f"[TELEMETRY #{telemetry_count}] "
                   f"Cycle {data['cycle']}: "
                   f"V={data['voltage']:.3f}V, "
                   f"I={data['current']:.3f}A, "
                   f"T={data['temperature']:.1f}°C")

    def on_state_estimate(message: str):
        """Handle incoming state estimate messages."""
        nonlocal state_estimate_count
        state_estimate_count += 1

        data = json.loads(message)
        logger.success(f"[STATE #{state_estimate_count}] "
                      f"Cycle {data['cycle']}: "
                      f"SoC={data['soc']:.1%}, "
                      f"SoH={data['soh']:.1%}, "
                      f"R0={data['r0']:.4f}Ω, "
                      f"Confidence={data['confidence_level']}")

    def on_health_report(message: str):
        """Handle incoming health report messages."""
        nonlocal health_report_count
        health_report_count += 1

        data = json.loads(message)
        logger.warning(f"[HEALTH #{health_report_count}] "
                      f"Status={data['health_status']}, "
                      f"Risk={data['risk_level']}, "
                      f"RUL={data.get('rul_cycles', 'N/A')} cycles")

    def on_prediction(message: str):
        """Handle incoming prediction messages."""
        nonlocal prediction_count
        prediction_count += 1

        data = json.loads(message)
        logger.info(f"[PREDICTION #{prediction_count}] "
                   f"Capacity={data.get('predicted_capacity', 'N/A'):.3f}Ah")

    # Step 3: Subscribe to topics
    logger.info("\nSubscribing to topics...")

    topics = {
        f"battery/{battery_id}/telemetry/clean": on_telemetry,
        f"battery/{battery_id}/state/estimate": on_state_estimate,
        f"battery/{battery_id}/health/report": on_health_report,
        f"battery/{battery_id}/prediction/hybrid": on_prediction,
    }

    for topic, handler in topics.items():
        await mqtt_bridge.subscribe(topic, handler)
        logger.info(f"  ✓ Subscribed to {topic}")

    # Step 4: Listen for messages
    logger.info("\nListening for messages (60 seconds)...")
    logger.info("Start the orchestrator in another terminal to see messages:")
    logger.info("  python3 -m src.battery_twin.orchestrator")
    logger.info("\nPress Ctrl+C to stop\n")

    try:
        # Listen for 60 seconds
        await asyncio.sleep(60.0)

    except KeyboardInterrupt:
        logger.warning("\nInterrupted by user")

    # Step 5: Show statistics
    logger.info("\n" + "=" * 80)
    logger.info("MESSAGE STATISTICS")
    logger.info("=" * 80)
    logger.info(f"Telemetry Messages: {telemetry_count}")
    logger.info(f"State Estimates: {state_estimate_count}")
    logger.info(f"Health Reports: {health_report_count}")
    logger.info(f"Predictions: {prediction_count}")
    logger.info(f"Total Messages: {telemetry_count + state_estimate_count + health_report_count + prediction_count}")

    # Step 6: Disconnect
    logger.info("\nDisconnecting...")
    await mqtt_bridge.disconnect()
    logger.success("✓ Disconnected")

    logger.info("=" * 80)
    logger.info("EXAMPLE COMPLETE")
    logger.info("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
