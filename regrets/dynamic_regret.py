import numpy as np
from pettingzoo.sisl import waterworld_v4
from typing import Dict, List, Tuple
import gymnasium as gym

class DynamicRegretCalculator:
    def __init__(self, env_config: Dict = None):
        """Initialize the Dynamic Regret Calculator.
        
        Args:
            env_config: Configuration for the Waterworld environment
        """
        self.env_config = env_config or {"render_mode": None}
        self.env = waterworld_v4.env(**self.env_config)
        self.action_space = self.env.action_space("pursuer_0")  # Same for all agents
        
        # Number of discrete actions to sample for finding best action
        self.n_action_samples = 5  # Reduced for computational efficiency
        
    def _get_discretized_actions(self) -> List[np.ndarray]:
        """Generate a set of discrete actions to evaluate."""
        actions = []
        # Include zero action
        actions.append(np.array([0.0, 0.0]))
        # Add cardinal directions
        for direction in [(1,0), (-1,0), (0,1), (0,-1)]:
            actions.append(np.array(direction))
        return actions
        
    def find_best_action(self, env: gym.Env, agent: str) -> Tuple[np.ndarray, float]:
        """Find the action that would maximize immediate reward."""
        discrete_actions = self._get_discretized_actions()
        best_reward = float('-inf')
        best_action = None
        
        # Save current state
        env_state = env.save_state()
        
        # Try each action and find the one with highest reward
        for action in discrete_actions:
            # Restore state before trying new action
            env.restore_state(env_state)
            
            # Take action and get reward
            env.step(action)
            _, reward, _, _, _ = env.last()
            
            if reward > best_reward:
                best_reward = reward
                best_action = action
        
        # Restore original state
        env.restore_state(env_state)
        return best_action, best_reward

    def calculate_dynamic_regret(self, env: gym.Env, agent_id: str, 
                               action_history: List[np.ndarray],
                               reward_history: List[float]) -> float:
        """Calculate the Dynamic Regret over a trajectory.
        
        For the Waterworld environment, we define dynamic regret as the sum of
        negative rewards (penalties) received. This represents how much better
        the agent could have done by avoiding poison and collecting food.
        """
        total_regret = 0.0
        rewards = [float(r) for r in reward_history]  # Convert all rewards to Python floats
        
        # Calculate total negative rewards (penalties)
        negative_rewards = [abs(r) for r in rewards if r < 0]
        total_regret = sum(negative_rewards)
            
        return float(total_regret)  # Ensure we return a Python float

def calculate_episode_dynamic_regret(env, episode_data: Dict) -> Dict[str, float]:
    """Calculate dynamic regret for each agent in an episode.
    
    Args:
        env: The Waterworld environment
        episode_data: Dictionary containing state, action, and reward histories
        
    Returns:
        Dictionary mapping agent IDs to their dynamic regret values
    """
    regret_calculator = DynamicRegretCalculator()
    agent_regrets = {}
    
    for agent_id in env.agents:
        if agent_id in episode_data:
            agent_data = episode_data[agent_id]
            dynamic_regret = regret_calculator.calculate_dynamic_regret(
                env,
                agent_id,
                agent_data['actions'],
                agent_data['rewards']
            )
            agent_regrets[agent_id] = float(dynamic_regret)  # Ensure we return Python float
            
    return agent_regrets 