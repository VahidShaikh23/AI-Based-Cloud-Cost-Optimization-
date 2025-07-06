#!/usr/bin/env python3
# rl_autoscaling_agent.py - Reinforcement Learning Agent for Auto-Scaling

import numpy as np
import pandas as pd
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import random
from collections import deque
import joblib
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ServerType(Enum):
    MICRO = {"name": "t3.micro", "capacity": 100, "cost": 0.0104}
    SMALL = {"name": "t3.small", "capacity": 200, "cost": 0.0208}
    MEDIUM = {"name": "t3.medium", "capacity": 400, "cost": 0.0416}
    LARGE = {"name": "t3.large", "capacity": 800, "cost": 0.0832}
    XLARGE = {"name": "t3.xlarge", "capacity": 1600, "cost": 0.1664}


@dataclass
class ServerInstance:
    instance_id: str
    server_type: ServerType
    capacity: int
    current_load: float
    start_time: datetime
    cost_per_hour: float
    status: str = "running"  # running, starting, stopping, stopped


@dataclass
class TrafficState:
    current_traffic: float
    predicted_traffic: float
    trend: float  # positive = increasing, negative = decreasing
    volatility: float
    time_of_day: int
    day_of_week: int
    is_weekend: bool
    season: str


@dataclass
class ScalingAction:
    action_type: str  # "scale_up", "scale_down", "scale_out", "scale_in", "no_action"
    target_instances: int
    target_server_type: ServerType


class AutoScalingEnvironment:
    """Simulated cloud environment for auto-scaling"""

    def __init__(self, initial_traffic: float = 1000):
        self.current_traffic = initial_traffic
        self.servers: List[ServerInstance] = []
        self.traffic_history = deque(maxlen=168)  # 1 week of hourly data
        self.action_history = deque(maxlen=100)
        self.cost_history = deque(maxlen=168)
        self.response_time_history = deque(maxlen=168)

        # Performance metrics
        self.total_cost = 0.0
        self.total_downtime = 0.0
        self.total_over_provisioning = 0.0
        self.sla_violations = 0

        # Environment parameters
        self.max_response_time = 2.0  # seconds
        self.target_utilization = 0.7
        self.scaling_cooldown = 5  # minutes
        self.last_scaling_action = None

        # Initialize with one micro instance
        self.add_server(ServerType.MICRO)

    def add_server(self, server_type: ServerType) -> ServerInstance:
        """Add a new server instance"""
        server = ServerInstance(
            instance_id=f"i-{len(self.servers):04d}",
            server_type=server_type,
            capacity=server_type.value["capacity"],
            current_load=0.0,
            start_time=datetime.now(),
            cost_per_hour=server_type.value["cost"]
        )
        self.servers.append(server)
        logger.info(f"Added server {server.instance_id} ({server_type.value['name']})")
        return server

    def remove_server(self, server_id: str) -> bool:
        """Remove a server instance"""
        for i, server in enumerate(self.servers):
            if server.instance_id == server_id:
                self.servers.pop(i)
                logger.info(f"Removed server {server_id}")
                return True
        return False

    def calculate_total_capacity(self) -> int:
        """Calculate total capacity of all running servers"""
        return sum(server.capacity for server in self.servers if server.status == "running")

    def calculate_current_utilization(self) -> float:
        """Calculate current utilization percentage"""
        total_capacity = self.calculate_total_capacity()
        return (self.current_traffic / total_capacity) if total_capacity > 0 else 1.0

    def calculate_response_time(self) -> float:
        """Calculate response time based on utilization"""
        utilization = self.calculate_current_utilization()
        if utilization <= 0.5:
            return 0.5
        elif utilization <= 0.8:
            return 0.5 + (utilization - 0.5) * 3.0  # Linear increase
        else:
            return 2.0 + (utilization - 0.8) * 10.0  # Exponential increase

    def calculate_hourly_cost(self) -> float:
        """Calculate current hourly cost"""
        return sum(server.cost_per_hour for server in self.servers if server.status == "running")

    def update_traffic(self, new_traffic: float):
        """Update current traffic and redistribute load"""
        self.current_traffic = new_traffic
        self.traffic_history.append(new_traffic)

        # Redistribute load across servers
        running_servers = [s for s in self.servers if s.status == "running"]
        if running_servers:
            load_per_server = new_traffic / len(running_servers)
            for server in running_servers:
                server.current_load = min(load_per_server, server.capacity)

    def get_state(self) -> Dict:
        """Get current environment state"""
        utilization = self.calculate_current_utilization()
        response_time = self.calculate_response_time()
        hourly_cost = self.calculate_hourly_cost()

        return {
            "current_traffic": self.current_traffic,
            "total_capacity": self.calculate_total_capacity(),
            "utilization": utilization,
            "response_time": response_time,
            "hourly_cost": hourly_cost,
            "num_servers": len(self.servers),
            "server_types": [s.server_type.name for s in self.servers],
            "sla_violation": response_time > self.max_response_time
        }

    def step(self, action: ScalingAction) -> Tuple[Dict, float, bool]:
        """Execute scaling action and return new state, reward, and done flag"""
        state_before = self.get_state()

        # Execute scaling action
        if action.action_type == "scale_up":
            self.add_server(action.target_server_type)
        elif action.action_type == "scale_down" and len(self.servers) > 1:
            # Remove the least utilized server
            server_to_remove = min(self.servers, key=lambda s: s.current_load)
            self.remove_server(server_to_remove.instance_id)
        elif action.action_type == "scale_out":
            # Add multiple instances
            for _ in range(action.target_instances):
                self.add_server(action.target_server_type)
        elif action.action_type == "scale_in" and len(self.servers) > action.target_instances:
            # Remove multiple instances
            servers_to_remove = len(self.servers) - action.target_instances
            for _ in range(servers_to_remove):
                if len(self.servers) > 1:
                    server_to_remove = min(self.servers, key=lambda s: s.current_load)
                    self.remove_server(server_to_remove.instance_id)

        # Update environment state
        state_after = self.get_state()

        # Calculate reward
        reward = self.calculate_reward(state_before, state_after, action)

        # Update metrics
        self.total_cost += state_after["hourly_cost"]
        if state_after["sla_violation"]:
            self.sla_violations += 1

        # Store metrics
        self.cost_history.append(state_after["hourly_cost"])
        self.response_time_history.append(state_after["response_time"])

        return state_after, reward, False  # Never done in continuous environment

    def calculate_reward(self, state_before: Dict, state_after: Dict, action: ScalingAction) -> float:
        """Calculate reward for the scaling action"""
        reward = 0.0

        # Penalize SLA violations heavily
        if state_after["sla_violation"]:
            reward -= 100.0

        # Reward maintaining good utilization (60-80%)
        utilization = state_after["utilization"]
        if 0.6 <= utilization <= 0.8:
            reward += 50.0
        elif utilization < 0.6:
            reward -= (0.6 - utilization) * 50.0  # Penalize under-utilization
        elif utilization > 0.8:
            reward -= (utilization - 0.8) * 100.0  # Penalize over-utilization

        # Penalize cost
        cost_penalty = state_after["hourly_cost"] * 10.0
        reward -= cost_penalty

        # Reward stable response times
        if state_after["response_time"] <= 1.0:
            reward += 20.0

        # Penalize unnecessary scaling actions
        if action.action_type == "no_action":
            reward += 5.0  # Small reward for not taking action when not needed

        return reward


class DQNAgent:
    """Deep Q-Network Agent for auto-scaling decisions"""

    def __init__(self, state_size: int, action_size: int, learning_rate: float = 0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.memory = deque(maxlen=10000)
        self.batch_size = 32

        # Q-table for simplified approach (instead of neural network)
        self.q_table = {}
        self.action_space = [
            ("no_action", 0, ServerType.MICRO),
            ("scale_up", 1, ServerType.MICRO),
            ("scale_up", 1, ServerType.SMALL),
            ("scale_up", 1, ServerType.MEDIUM),
            ("scale_down", 1, ServerType.MICRO),
            ("scale_out", 2, ServerType.MICRO),
            ("scale_in", 1, ServerType.MICRO),
        ]

    def get_state_key(self, state: Dict) -> str:
        """Convert state to string key for Q-table"""
        return f"{int(state['utilization'] * 10)}_{int(state['response_time'] * 10)}_{state['num_servers']}"

    def choose_action(self, state: Dict) -> ScalingAction:
        """Choose action using epsilon-greedy policy"""
        state_key = self.get_state_key(state)

        if random.random() < self.epsilon:
            # Random action (exploration)
            action_idx = random.randint(0, len(self.action_space) - 1)
        else:
            # Best action (exploitation)
            if state_key not in self.q_table:
                self.q_table[state_key] = [0.0] * len(self.action_space)
            action_idx = np.argmax(self.q_table[state_key])

        action_type, target_instances, target_server_type = self.action_space[action_idx]

        return ScalingAction(
            action_type=action_type,
            target_instances=target_instances,
            target_server_type=target_server_type
        )

    def remember(self, state: Dict, action_idx: int, reward: float, next_state: Dict):
        """Store experience in memory"""
        self.memory.append((state, action_idx, reward, next_state))

    def replay(self):
        """Train the agent using experiences from memory"""
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)

        for state, action_idx, reward, next_state in batch:
            state_key = self.get_state_key(state)
            next_state_key = self.get_state_key(next_state)

            if state_key not in self.q_table:
                self.q_table[state_key] = [0.0] * len(self.action_space)
            if next_state_key not in self.q_table:
                self.q_table[next_state_key] = [0.0] * len(self.action_space)

            # Q-learning update
            current_q = self.q_table[state_key][action_idx]
            max_next_q = max(self.q_table[next_state_key])
            new_q = current_q + self.learning_rate * (reward + 0.95 * max_next_q - current_q)
            self.q_table[state_key][action_idx] = new_q

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save_model(self, filepath: str):
        """Save the Q-table"""
        with open(filepath, 'w') as f:
            json.dump(self.q_table, f, indent=2)
        logger.info(f"Model saved to {filepath}")

    def load_model(self, filepath: str):
        """Load the Q-table"""
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                self.q_table = json.load(f)
            logger.info(f"Model loaded from {filepath}")
        else:
            logger.warning(f"Model file not found: {filepath}")


class AutoScalingSimulator:
    """Main simulator for auto-scaling with RL agent"""

    def __init__(self, traffic_model_path: str = "models/best_traffic_model.pkl"):
        self.environment = AutoScalingEnvironment()
        self.agent = DQNAgent(state_size=8, action_size=7)
        self.traffic_model = None
        self.simulation_results = []

        # Load traffic prediction model
        try:
            self.traffic_model = joblib.load(traffic_model_path)
            logger.info("Traffic prediction model loaded successfully")
        except Exception as e:
            logger.warning(f"Could not load traffic model: {e}")

    def generate_traffic_pattern(self, hours: int = 24) -> List[float]:
        """Generate realistic traffic pattern"""
        traffic_pattern = []
        base_traffic = 1000

        for hour in range(hours):
            # Simulate daily pattern
            time_factor = 0.5 + 0.5 * np.sin(2 * np.pi * hour / 24)

            # Add some randomness
            noise = np.random.normal(0, 0.1)

            # Simulate traffic spikes
            if random.random() < 0.1:  # 10% chance of traffic spike
                spike_factor = random.uniform(2.0, 5.0)
                traffic = base_traffic * time_factor * spike_factor
            else:
                traffic = base_traffic * time_factor * (1 + noise)

            traffic_pattern.append(max(100, traffic))  # Minimum 100 requests

        return traffic_pattern

    def run_simulation(self, hours: int = 168, save_results: bool = True) -> Dict:
        """Run auto-scaling simulation"""
        logger.info(f"Starting auto-scaling simulation for {hours} hours")

        # Generate traffic pattern
        traffic_pattern = self.generate_traffic_pattern(hours)

        # Simulation metrics
        total_cost = 0.0
        total_sla_violations = 0
        episode_rewards = []

        for hour in range(hours):
            # Update traffic
            current_traffic = traffic_pattern[hour]
            self.environment.update_traffic(current_traffic)

            # Get current state
            state = self.environment.get_state()

            # Agent chooses action
            action = self.agent.choose_action(state)

            # Execute action and get reward
            next_state, reward, done = self.environment.step(action)

            # Store experience
            action_idx = self.agent.action_space.index(
                (action.action_type, action.target_instances, action.target_server_type)
            )
            self.agent.remember(state, action_idx, reward, next_state)

            # Train agent
            if hour % 10 == 0:  # Train every 10 hours
                self.agent.replay()

            # Update metrics
            total_cost += state["hourly_cost"]
            if state["sla_violation"]:
                total_sla_violations += 1

            episode_rewards.append(reward)

            # Store simulation step
            step_result = {
                "hour": hour,
                "traffic": current_traffic,
                "num_servers": state["num_servers"],
                "utilization": state["utilization"],
                "response_time": state["response_time"],
                "hourly_cost": state["hourly_cost"],
                "action": action.action_type,
                "reward": reward,
                "sla_violation": state["sla_violation"]
            }
            self.simulation_results.append(step_result)

            if hour % 24 == 0:  # Log progress every day
                logger.info(f"Day {hour // 24}: Traffic={current_traffic:.0f}, "
                            f"Servers={state['num_servers']}, "
                            f"Utilization={state['utilization']:.2f}, "
                            f"Cost=${state['hourly_cost']:.2f}/hr")

        # Calculate final metrics
        avg_utilization = np.mean([r["utilization"] for r in self.simulation_results])
        avg_response_time = np.mean([r["response_time"] for r in self.simulation_results])
        avg_cost = np.mean([r["hourly_cost"] for r in self.simulation_results])
        sla_compliance = (hours - total_sla_violations) / hours * 100

        results = {
            "simulation_duration_hours": hours,
            "total_cost": total_cost,
            "average_hourly_cost": avg_cost,
            "average_utilization": avg_utilization,
            "average_response_time": avg_response_time,
            "sla_compliance_percentage": sla_compliance,
            "total_sla_violations": total_sla_violations,
            "total_scaling_actions": len([r for r in self.simulation_results if r["action"] != "no_action"]),
            "final_epsilon": self.agent.epsilon,
            "simulation_results": self.simulation_results
        }

        if save_results:
            self.save_simulation_results(results)

        logger.info(f"Simulation completed! Average cost: ${avg_cost:.2f}/hr, "
                    f"SLA compliance: {sla_compliance:.1f}%")

        return results

    def save_simulation_results(self, results: Dict):
        """Save simulation results to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"simulation_results_{timestamp}.json"

        # Create directory if it doesn't exist
        os.makedirs("simulation_results", exist_ok=True)

        filepath = os.path.join("simulation_results", filename)
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        logger.info(f"Simulation results saved to {filepath}")

        # Save agent model
        model_filepath = os.path.join("models", "autoscaling_agent.json")
        self.agent.save_model(model_filepath)

    def load_agent_model(self, filepath: str = "models/autoscaling_agent.json"):
        """Load pre-trained agent model"""
        self.agent.load_model(filepath)

    def get_real_time_recommendation(self, current_traffic: float) -> Dict:
        """Get real-time scaling recommendation"""
        self.environment.update_traffic(current_traffic)
        state = self.environment.get_state()
        action = self.agent.choose_action(state)

        return {
            "current_state": state,
            "recommended_action": {
                "action_type": action.action_type,
                "target_instances": action.target_instances,
                "target_server_type": action.target_server_type.value["name"]
            },
            "timestamp": datetime.now().isoformat()
        }


# Example usage and testing
if __name__ == "__main__":
    # Initialize simulator
    simulator = AutoScalingSimulator()

    # Run simulation
    results = simulator.run_simulation(hours=168)  # 1 week simulation

    # Print summary
    print("\n" + "=" * 50)
    print("AUTO-SCALING SIMULATION SUMMARY")
    print("=" * 50)
    print(f"Duration: {results['simulation_duration_hours']} hours")
    print(f"Total Cost: ${results['total_cost']:.2f}")
    print(f"Average Hourly Cost: ${results['average_hourly_cost']:.2f}")
    print(f"Average Utilization: {results['average_utilization']:.2f}")
    print(f"Average Response Time: {results['average_response_time']:.2f}s")
    print(f"SLA Compliance: {results['sla_compliance_percentage']:.1f}%")
    print(f"Total Scaling Actions: {results['total_scaling_actions']}")
    print("=" * 50)

    # Test real-time recommendation
    recommendation = simulator.get_real_time_recommendation(1500.0)
    print("\nReal-time Recommendation for 1500 requests:")
    print(f"Action: {recommendation['recommended_action']['action_type']}")
    print(f"Target Instances: {recommendation['recommended_action']['target_instances']}")
    print(f"Server Type: {recommendation['recommended_action']['target_server_type']}")