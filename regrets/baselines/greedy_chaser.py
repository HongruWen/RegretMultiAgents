import numpy as np
from typing import List, Optional
from regrets.baselines.base import BaseBaseline

class GreedyTargetChaser(BaseBaseline):
    """A baseline policy that greedily moves toward the nearest food particle using sensor information."""
    
    def __init__(self, action_dim: int = 2, max_accel: float = 1.0, n_sensors: int = 30):
        """
        Initialize the greedy target chaser.
        
        Args:
            action_dim (int): Dimension of the action space
            max_accel (float): Maximum acceleration (thrust) in any direction
            n_sensors (int): Number of sensors on the agent
        """
        super().__init__(action_dim)
        self.max_accel = max_accel  # Now using 1.0 as default to match action space
        self.n_sensors = n_sensors
        
        # Calculate indices for different sensor readings
        # According to Waterworld documentation:
        # [0:n_sensors] - Obstacle distances
        # [n_sensors:2*n_sensors] - Barrier distances
        # [2*n_sensors:3*n_sensors] - Food distances
        # [3*n_sensors:4*n_sensors] - Food speeds
        # [4*n_sensors:5*n_sensors] - Poison distances
        # [5*n_sensors:6*n_sensors] - Poison speeds
        # [6*n_sensors:7*n_sensors] - Pursuer distances
        # [7*n_sensors:8*n_sensors] - Pursuer speeds
        # [8*n_sensors] - Food collision indicator
        # [8*n_sensors + 1] - Poison collision indicator
        self.food_dist_start = 2 * n_sensors
        self.food_speed_start = 3 * n_sensors
        self.poison_dist_start = 4 * n_sensors
        self.poison_speed_start = 5 * n_sensors
    
    def get_action(
        self,
        state: np.ndarray,
        other_agents_states: Optional[List[np.ndarray]] = None
    ) -> np.ndarray:
        """
        Get the action that moves toward the nearest food particle using sensor information.
        
        Args:
            state (np.ndarray): Current state containing sensor readings
            other_agents_states (Optional[List[np.ndarray]]): States of other agents (not used)
            
        Returns:
            np.ndarray: Action vector [horizontal_thrust, vertical_thrust]
        """
        # Get food distances and speeds from sensors
        food_dists = state[self.food_dist_start:self.food_speed_start]
        food_speeds = state[self.food_speed_start:self.poison_dist_start]
        
        # Find sensor with closest food
        closest_sensor_idx = np.argmin(food_dists)
        closest_food_dist = food_dists[closest_sensor_idx]
        closest_food_speed = food_speeds[closest_sensor_idx]
        
        if closest_food_dist < 1.0:  # If food is detected (distance < 1.0 means food is in range)
            # Calculate angle based on sensor index
            angle = (2 * np.pi * closest_sensor_idx) / self.n_sensors
            
            # Create direction vector pointing toward food
            direction = np.array([np.cos(angle), np.sin(angle)])
            
            # Calculate food's movement vector
            food_movement = np.array([
                np.cos(angle) * closest_food_speed,
                np.sin(angle) * closest_food_speed
            ])
            
            # Combine direction and food movement for interception
            # Scale food movement by distance (more influence when closer)
            intercept_vector = direction + food_movement * (1.0 - closest_food_dist)
            
            # Normalize and scale by max_accel
            intercept_norm = np.linalg.norm(intercept_vector)
            if intercept_norm > 0:
                action = (intercept_vector / intercept_norm) * self.max_accel
            else:
                action = direction * self.max_accel
            
            # Add very small noise to prevent getting stuck
            noise = np.random.normal(0, 0.01, self.action_dim)  # Reduced noise to 1% of action space
            action = action + noise
            
            # Ensure action is within valid range [-1.0, 1.0]
            action = np.clip(action, -1.0, 1.0)
            return action
        else:
            # If no food detected, move in a random direction
            random_angle = np.random.uniform(0, 2 * np.pi)
            direction = np.array([np.cos(random_angle), np.sin(random_angle)])
            return direction * 0.5  # Use half thrust when exploring
    
    def compute_reward(
        self,
        action: np.ndarray,
        state: np.ndarray,
        other_agents_states: Optional[List[np.ndarray]] = None
    ) -> float:
        """
        Compute the reward for moving toward the nearest food.
        
        Args:
            action (np.ndarray): Action taken
            state (np.ndarray): Current state
            other_agents_states (Optional[List[np.ndarray]]): States of other agents (not used)
            
        Returns:
            float: Reward based on distance to nearest food
        """
        # Get food distances from sensors
        food_dists = state[self.food_dist_start:self.food_speed_start]
        
        # Find closest food
        min_dist = np.min(food_dists)
        
        # Reward is negative distance to nearest food
        # This encourages the agent to minimize distance to food
        return -min_dist
    
    def get_expected_reward(
        self,
        state: np.ndarray,
        other_agents_states: Optional[List[np.ndarray]] = None,
        num_samples: int = 100
    ) -> float:
        """
        Get the expected reward for the current state.
        
        Args:
            state (np.ndarray): Current state
            other_agents_states (Optional[List[np.ndarray]]): States of other agents (not used)
            num_samples (int): Number of samples to use for expectation
            
        Returns:
            float: Expected reward (same as current reward for deterministic policy)
        """
        # For this deterministic policy, the expected reward is the same as the current reward
        return self.compute_reward(self.get_action(state), state) 