"""
Kafka Communication Benchmark Scenarios
Performance testing scenarios for Kafka-based agent communication.
"""

import time
import threading
from typing import Dict, Any
from communication.kafka.kafka_communication import KafkaMessageType
from communication.kafka.kafka_communication_agent import (
    KafkaCommunicationEnvironment,
)


def setup_basic_scenario(agent_count: int = 2) -> Dict[str, Any]:
    """
    Basic Kafka communication scenario.
    Two agents exchanging simple messages.
    """
    env = KafkaCommunicationEnvironment()
    env.setup({})

    # Create agents
    agents = []
    for i in range(agent_count):
        agent = env.create_agent(f"kafka_agent_{i}")
        agents.append(agent)

    # Setup topology
    env.setup_topology(agents, "fully_connected")

    config = {
        "scenario_name": "kafka_basic",
        "message_types": [KafkaMessageType.INFORM, KafkaMessageType.REQUEST],
        "test_duration": 5.0,
    }

    return {
        "environment": env,
        "agents": agents,
        "config": config,
        "agent_count": agent_count,
        "topology_density": 1.0,
    }


def setup_broadcast_scenario(agent_count: int = 5) -> Dict[str, Any]:
    """
    Kafka broadcast performance scenario.
    One agent broadcasting to multiple receivers.
    """
    env = KafkaCommunicationEnvironment()
    env.setup({})

    # Create agents
    agents = []
    for i in range(agent_count):
        agent = env.create_agent(f"kafka_broadcast_agent_{i}")
        agents.append(agent)

    # Setup star topology (agent_0 is broadcaster)
    env.setup_topology(agents, "star")

    config = {
        "scenario_name": "kafka_broadcast",
        "broadcaster_id": "kafka_broadcast_agent_0",
        "message_types": [KafkaMessageType.BROADCAST],
        "test_duration": 10.0,
    }

    return {
        "environment": env,
        "agents": agents,
        "config": config,
        "agent_count": agent_count,
        "topology_density": 0.4,
    }


def setup_high_throughput_scenario(agent_count: int = 10) -> Dict[str, Any]:
    """
    High throughput Kafka scenario.
    Multiple agents sending many messages rapidly.
    """
    env = KafkaCommunicationEnvironment()
    env.setup({})

    # Create agents
    agents = []
    for i in range(agent_count):
        agent = env.create_agent(f"kafka_throughput_agent_{i}")
        agents.append(agent)

    # Fully connected topology
    env.setup_topology(agents, "fully_connected")

    config = {
        "scenario_name": "kafka_high_throughput",
        "messages_per_agent": 100,
        "message_types": [KafkaMessageType.INFORM],
        "test_duration": 15.0,
    }

    return {
        "environment": env,
        "agents": agents,
        "config": config,
        "agent_count": agent_count,
        "topology_density": 1.0,
    }


def setup_request_reply_scenario(agent_count: int = 4) -> Dict[str, Any]:
    """
    Request-reply pattern scenario using Kafka.
    Agents sending requests and waiting for replies.
    """
    env = KafkaCommunicationEnvironment()
    env.setup({})

    # Create agents
    agents = []
    for i in range(agent_count):
        agent = env.create_agent(f"kafka_reqrep_agent_{i}")
        agents.append(agent)

    # Setup topology
    env.setup_topology(agents, "fully_connected")

    config = {
        "scenario_name": "kafka_request_reply",
        "request_pairs": agent_count // 2,
        "message_types": [KafkaMessageType.REQUEST, KafkaMessageType.REPLY],
        "test_duration": 8.0,
    }

    return {
        "environment": env,
        "agents": agents,
        "config": config,
        "agent_count": agent_count,
        "topology_density": 1.0,
    }


def run_basic_message_test(
    scenario_data: Dict[str, Any], duration: float
) -> Dict[str, Any]:
    """Run basic Kafka message exchange test."""
    agents = scenario_data["agents"]

    if len(agents) < 2:
        return {"error": "Need at least 2 agents for basic test"}

    sender = agents[0]
    receiver = agents[1]

    # Send messages
    messages_sent = 0
    start_time = time.time()

    while time.time() - start_time < duration:
        success = sender.send_message(
            receiver.agent_id,
            KafkaMessageType.INFORM,
            {
                "test_data": f"message_{messages_sent}",
                "timestamp": time.time(),
            },
        )
        if success:
            messages_sent += 1
        time.sleep(0.1)

    # Allow time for message delivery
    time.sleep(1.0)

    # Check received messages
    received_messages = receiver.receive_messages()
    messages_received = len(received_messages)

    return {
        "messages_sent": messages_sent,
        "messages_received": messages_received,
        "success_rate": messages_received / max(messages_sent, 1),
    }


def run_broadcast_test(
    scenario_data: Dict[str, Any], duration: float
) -> Dict[str, Any]:
    """Run Kafka broadcast performance test."""
    agents = scenario_data["agents"]

    broadcaster = agents[0]  # First agent is broadcaster
    receivers = agents[1:]

    broadcasts_sent = 0
    start_time = time.time()

    while time.time() - start_time < duration:
        result = broadcaster.broadcast_message(
            KafkaMessageType.BROADCAST,
            {
                "broadcast_data": f"broadcast_{broadcasts_sent}",
                "timestamp": time.time(),
            },
        )
        if result["status"] == "completed":
            broadcasts_sent += 1
        time.sleep(0.2)

    # Allow time for message delivery
    time.sleep(1.5)

    # Check messages received by each receiver
    total_received = 0
    for receiver in receivers:
        messages = receiver.receive_messages()
        total_received += len(messages)

    expected_total = broadcasts_sent * len(receivers)

    return {
        "broadcasts_sent": broadcasts_sent,
        "total_messages_received": total_received,
        "expected_total": expected_total,
        "broadcast_efficiency": total_received / max(expected_total, 1),
    }


def run_throughput_test(
    scenario_data: Dict[str, Any], duration: float
) -> Dict[str, Any]:
    """Run high throughput Kafka test."""
    agents = scenario_data["agents"]
    config = scenario_data["config"]

    results = {"messages_sent_per_agent": {}, "total_messages_sent": 0}

    def agent_sender(agent, target_agents, messages_to_send):
        """Thread function for each agent to send messages."""
        sent_count = 0
        for i in range(messages_to_send):
            for target in target_agents:
                if target.agent_id != agent.agent_id:
                    success = agent.send_message(
                        target.agent_id,
                        KafkaMessageType.INFORM,
                        {"data": f"msg_{i}_from_{agent.agent_id}"},
                    )
                    if success:
                        sent_count += 1
            time.sleep(0.01)  # Small delay
        results["messages_sent_per_agent"][agent.agent_id] = sent_count

    # Start sender threads
    threads = []
    messages_per_agent = config.get("messages_per_agent", 50)

    for agent in agents:
        thread = threading.Thread(
            target=agent_sender, args=(agent, agents, messages_per_agent)
        )
        threads.append(thread)
        thread.start()

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

    # Calculate total sent
    results["total_messages_sent"] = sum(
        results["messages_sent_per_agent"].values()
    )

    # Allow delivery time
    time.sleep(2.0)

    # Count received messages
    total_received = 0
    for agent in agents:
        messages = agent.receive_messages()
        total_received += len(messages)

    results["total_messages_received"] = total_received
    results["throughput_rate"] = results["total_messages_sent"] / duration

    return results


def run_request_reply_test(
    scenario_data: Dict[str, Any], duration: float
) -> Dict[str, Any]:
    """Run request-reply pattern test with Kafka."""
    agents = scenario_data["agents"]

    if len(agents) < 2:
        return {"error": "Need at least 2 agents for request-reply test"}

    # Pair agents for request-reply
    requesters = agents[::2]  # Even indexed agents
    responders = agents[1::2]  # Odd indexed agents

    requests_sent = 0
    replies_received = 0

    # Send requests
    for requester, responder in zip(requesters, responders):
        for i in range(5):  # 5 requests per pair
            success = requester.send_message(
                responder.agent_id,
                KafkaMessageType.REQUEST,
                {"request_id": f"req_{i}", "requester": requester.agent_id},
            )
            if success:
                requests_sent += 1
            time.sleep(0.1)

    # Allow processing time
    time.sleep(1.0)

    # Process requests and send replies
    for responder in responders:
        requests = responder.receive_messages()
        for request in requests:
            if request.message_type == KafkaMessageType.REQUEST:
                responder.reply_to_message(
                    request,
                    {
                        "reply_to": request.content.get("request_id"),
                        "responder": responder.agent_id,
                    },
                )

    # Allow reply delivery time
    time.sleep(1.5)

    # Count replies received
    for requester in requesters:
        replies = requester.receive_messages()
        replies_received += len(
            [
                msg
                for msg in replies
                if msg.message_type == KafkaMessageType.REPLY
            ]
        )

    return {
        "requests_sent": requests_sent,
        "replies_received": replies_received,
        "reply_rate": replies_received / max(requests_sent, 1),
    }


# Test scenario mapping
KAFKA_TEST_SCENARIOS = {
    "test_basic": run_basic_message_test,
    "test_broadcast": run_broadcast_test,
    "test_throughput": run_throughput_test,
    "test_request_reply": run_request_reply_test,
}

# Setup scenario mapping
KAFKA_SETUP_SCENARIOS = {
    "test_basic": setup_basic_scenario,
    "test_broadcast": setup_broadcast_scenario,
    "test_throughput": setup_high_throughput_scenario,
    "test_request_reply": setup_request_reply_scenario,
}
