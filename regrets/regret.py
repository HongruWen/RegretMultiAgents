from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, Any
import numpy as np
from .baselines import RLPolicyBaseline
from pettingzoo.sisl import waterworld_v4

class Regret(ABC):
    """Abstract base class for regret calculation in Waterworld environment."""
    
    def __init__(self, n_agents: int = 5):
        """
        Initialize the regret calculator.
        
        Args:
            n_agents (int): Number of agents in the Waterworld environment
        """
        self.cumulative_agent_rewards = np.zeros(n_agents)  # Track total rewards for each agent
        self.timestep = 0
        self.n_agents = n_agents
        
    @abstractmethod
    def update(self, agent_id: str, agent_reward: float, **kwargs) -> float:
        """
        Update the regret calculation with new rewards.
        
        Args:
            agent_id (str): ID of the agent (e.g., 'pursuer_0')
            agent_reward (float): The reward received by the agent
            **kwargs: Additional arguments needed for regret calculation
            
        Returns:
            float: Current regret for this agent
        """
        pass
    
    def get_regret(self, agent_id: Optional[str] = None) -> Union[float, Dict[str, float]]:
        """
        Get the current regret (baseline_total - agent_total).
        
        Args:
            agent_id (Optional[str]): If provided, returns regret for specific agent
            
        Returns:
            Union[float, Dict[str, float]]: Regret for specified agent or all agents
        """
        if agent_id is not None:
            agent_idx = int(agent_id.split('_')[1])
            return self._calculate_regret(self.cumulative_agent_rewards[agent_idx])
        
        return {
            f"pursuer_{i}": self._calculate_regret(self.cumulative_agent_rewards[i])
            for i in range(self.n_agents)
        }
    
    def _calculate_regret(self, agent_total: float) -> float:
        """Helper method to calculate regret given agent's total reward."""
        pass

    def reset(self):
        """Reset the regret calculator."""
        self.cumulative_agent_rewards = np.zeros(self.n_agents)
        self.timestep = 0

class BaselineRegret(Regret):
    """
    Regret calculator that compares against a baseline policy for Waterworld.
    Regret(T) = Σ(R_t^baseline) - Σ(R_t^agent)
    """
    
    def __init__(self, mode: str = "rl", n_agents: int = 5, **kwargs):
        """
        Initialize the baseline regret calculator.
        
        Args:
            mode (str): The type of baseline to use ('rl' for RL policy)
            n_agents (int): Number of agents in the Waterworld environment
            **kwargs: Additional arguments for the baseline
                For RL baseline:
                    - action_dim (int): Dimension of action space (2 for Waterworld)
                    - algorithm (str): RL algorithm to use
                    - model_path (str): Path to saved model
                    - env_config (dict): Environment configuration
                    - device (str): Device to use for inference
        """
        super().__init__(n_agents=n_agents)
        self.mode = mode
        
        # Set default action_dim for Waterworld if not provided
        if 'action_dim' not in kwargs:
            kwargs['action_dim'] = 2  # Waterworld has 2D continuous action space
            
        if mode == "rl":
            self.baseline = RLPolicyBaseline(**kwargs)
        else:
            raise ValueError(f"Unsupported baseline mode: {mode}")
            
        # Fixed baseline total reward for RL (33.6052)
        self.baseline_total = 33.6052
        
        # Store agent rewards for analysis
        self.agent_rewards = {f"pursuer_{i}": [] for i in range(n_agents)}
        
    def update(self, agent_id: str, agent_reward: float, state: np.ndarray, **kwargs) -> float:
        """
        Update regret calculation with new rewards.
        
        Args:
            agent_id (str): ID of the agent (e.g., 'pursuer_0')
            agent_reward (float): Reward received by the agent
            state (np.ndarray): Current state for baseline policy (242-dim for Waterworld)
            **kwargs: Additional arguments (e.g., other_agents_states)
            
        Returns:
            float: Current regret for this agent
        """
        # Verify state dimension matches Waterworld's observation space
        if state.shape[0] != 242:  # Waterworld has 242-dimensional observation space
            raise ValueError(f"Expected state dimension of 242, got {state.shape[0]}")
            
        # Get baseline's expected reward for this state
        baseline_reward = self.baseline.get_expected_reward(
            state,
            other_agents_states=kwargs.get('other_agents_states')
        )
        
        # Store baseline reward for this agent
        self.agent_rewards[agent_id].append(baseline_reward)
        
        # Update cumulative reward for this agent with actual reward
        agent_idx = int(agent_id.split('_')[1])
        self.cumulative_agent_rewards[agent_idx] += agent_reward
        
        # Update timestep only once per environment step
        if agent_id == f"pursuer_{self.n_agents - 1}":  # Last agent in the step
            self.timestep += 1
        
        # Return current regret for this agent
        return self._calculate_regret(self.cumulative_agent_rewards[agent_idx])
    
    def _calculate_regret(self, agent_total: float) -> float:
        """
        Calculate regret as baseline_total - agent_total.
        
        Args:
            agent_total (float): Agent's cumulative reward
            
        Returns:
            float: Current regret
        """
        return self.baseline_total - agent_total
    
    def get_performance_stats(self, agent_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get statistics about the performance.
        
        Args:
            agent_id (Optional[str]): If provided, returns stats for specific agent
            
        Returns:
            Dict[str, Any]: Dictionary containing baseline and agent performance statistics
        """
        if agent_id is not None:
            agent_idx = int(agent_id.split('_')[1])
            return {
                "baseline_total": self.baseline_total,
                "agent_total": self.cumulative_agent_rewards[agent_idx],
                "agent_rewards": {
                    "mean": np.mean(self.agent_rewards[agent_id]),
                    "std": np.std(self.agent_rewards[agent_id]),
                    "timesteps": len(self.agent_rewards[agent_id])
                },
                "regret": self._calculate_regret(self.cumulative_agent_rewards[agent_idx])
            }
        
        return {
            "baseline": {
                "total": self.baseline_total
            },
            "agents": {
                agent_id: {
                    "total_reward": self.cumulative_agent_rewards[int(agent_id.split('_')[1])],
                    "mean_reward": np.mean(rewards),
                    "std_reward": np.std(rewards),
                    "timesteps": len(rewards),
                    "regret": self._calculate_regret(
                        self.cumulative_agent_rewards[int(agent_id.split('_')[1])]
                    )
                }
                for agent_id, rewards in self.agent_rewards.items()
            }
        }
    
    def reset(self):
        """Reset the regret calculator and stored rewards."""
        super().reset()
        self.agent_rewards = {f"pursuer_{i}": [] for i in range(self.n_agents)} 