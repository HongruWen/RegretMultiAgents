from abc import ABC, abstractmethod
import numpy as np
from typing import List, Optional, Dict

class BaseBaseline(ABC):
    """Base class for all baseline policies."""
    
    def __init__(self, action_dim: int = 2):
        """
        Initialize the baseline policy.
        
        Args:
            action_dim (int): Dimension of the action space
        """
        self.action_dim = action_dim
    
    @abstractmethod
    def get_action(
        self,
        state: np.ndarray,
        other_agents_states: Optional[List[np.ndarray]] = None
    ) -> np.ndarray:
        """
        Get the action from the baseline policy.
        
        Args:
            state (np.ndarray): Current state
            other_agents_states (Optional[List[np.ndarray]]): States of other agents
            
        Returns:
            np.ndarray: Action from baseline policy
        """
        pass
    
    @abstractmethod
    def compute_reward(
        self,
        action: np.ndarray,
        state: np.ndarray,
        other_agents_states: Optional[List[np.ndarray]] = None
    ) -> float:
        """
        Compute the reward for a baseline action.
        
        Args:
            action (np.ndarray): Action to evaluate
            state (np.ndarray): Current state
            other_agents_states (Optional[List[np.ndarray]]): States of other agents
            
        Returns:
            float: Reward for the baseline action
        """
        pass
    
    @abstractmethod
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
            other_agents_states (Optional[List[np.ndarray]]): States of other agents
            num_samples (int): Number of samples to use for expectation
            
        Returns:
            float: Expected reward
        """
        pass 