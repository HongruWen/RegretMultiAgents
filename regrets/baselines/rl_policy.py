import numpy as np
import torch
import os
from typing import List, Optional, Dict
from stable_baselines3 import PPO, DDPG
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.utils import get_linear_fn
from pettingzoo.sisl import waterworld_v4
from pettingzoo.utils.conversions import aec_to_parallel_wrapper
from pettingzoo.utils.wrappers import OrderEnforcingWrapper
import gymnasium as gym

from regrets.baselines.base import BaseBaseline

class WaterworldWrapper(gym.Env):
    """Wrapper to convert PettingZoo environment to Gymnasium format."""
    def __init__(self, env):
        super().__init__()
        self.env = env
        # Initialize environment to get spaces
        self.env.reset()
        self.observation_space = self.env.observation_space(self.env.agents[0])
        self.action_space = self.env.action_space(self.env.agents[0])
        self.metadata = {"render_modes": ["human", "rgb_array"]}
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.env.reset(seed=seed)
        obs = self.env.observe(self.env.agents[0])
        return obs, {}
        
    def step(self, action):
        self.env.step(action)
        obs = self.env.observe(self.env.agents[0])
        reward = self.env.rewards[self.env.agents[0]]
        done = self.env.terminations[self.env.agents[0]]
        truncated = self.env.truncations[self.env.agents[0]]
        info = {}
        return obs, reward, done, truncated, info
        
    def render(self, mode="human"):
        return self.env.render()
        
    def close(self):
        self.env.close()

class RLPolicyBaseline(BaseBaseline):
    """RL policy baseline using Stable-Baselines3."""
    
    def __init__(
        self,
        action_dim: int = 2,
        algorithm: str = "PPO",
        model_path: Optional[str] = None,
        env_config: Optional[Dict] = None,
        device: str = "auto"
    ):
        """
        Initialize the RL policy baseline.
        
        Args:
            action_dim (int): Dimension of the action space
            algorithm (str): RL algorithm to use ("PPO" or "DDPG")
            model_path (Optional[str]): Path to saved model
            env_config (Optional[Dict]): Configuration for the environment
            device (str): Device to use for training ("auto", "cuda", or "cpu")
        """
        super().__init__(action_dim)
        self.algorithm = algorithm
        self.model_path = model_path
        self.env_config = env_config or {}
        self.device = device
        
        # Initialize the environment
        self.env = waterworld_v4.env(**self.env_config)
        self.env = OrderEnforcingWrapper(self.env)
        self.env = WaterworldWrapper(self.env)
        self.env = DummyVecEnv([lambda: self.env])
        
        # Initialize the model
        if os.path.exists(f"{self.model_path}.zip"):
            self._load_model()
        else:
            self.model = None
    
    def _load_model(self):
        """Load a saved model."""
        if self.algorithm == "PPO":
            self.model = PPO.load(self.model_path)
        elif self.algorithm == "DDPG":
            self.model = DDPG.load(self.model_path)
        else:
            raise ValueError(f"Unsupported algorithm: {self.algorithm}")
    
    def train(
        self,
        total_timesteps: int = 1_000_000,
        learning_rate: float = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        eval_freq: int = 10000,
        n_eval_episodes: int = 10
    ):
        """
        Train the RL policy.
        
        Args:
            total_timesteps (int): Total number of timesteps to train for
            learning_rate (float): Learning rate for the policy
            n_steps (int): Number of steps to run for each environment per update
            batch_size (int): Minibatch size for training
            n_epochs (int): Number of epochs when optimizing the surrogate loss
            gamma (float): Discount factor
            gae_lambda (float): Factor for trade-off of bias vs variance for GAE
            clip_range (float): Clipping parameter for PPO
            eval_freq (int): Evaluate the model every X timesteps
            n_eval_episodes (int): Number of episodes to evaluate the model
        """
        # Create model directory if it doesn't exist
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        
        # Create evaluation environment
        eval_env = waterworld_v4.env(**self.env_config)
        eval_env = OrderEnforcingWrapper(eval_env)
        eval_env = WaterworldWrapper(eval_env)
        eval_env = DummyVecEnv([lambda: eval_env])
        
        # Create callbacks
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=os.path.dirname(self.model_path),
            log_path=os.path.dirname(self.model_path),
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            deterministic=True,
            render=False
        )
        
        # Create checkpoint callback
        checkpoint_callback = CheckpointCallback(
            save_freq=50000,
            save_path=f"{os.path.dirname(self.model_path)}/checkpoints",
            name_prefix="model"
        )
        
        # Policy kwargs for better performance
        policy_kwargs = dict(
            net_arch=dict(
                pi=[64, 64],  # Policy network
                vf=[64, 64]   # Value network
            ),
            activation_fn=torch.nn.ReLU
        )
        
        # Initialize model based on algorithm
        if self.algorithm == "PPO":
            # Learning rate schedule
            learning_rate_fn = get_linear_fn(
                learning_rate,
                learning_rate * 0.1,
                total_timesteps
            )
            
            self.model = PPO(
                "MlpPolicy",
                self.env,
                learning_rate=learning_rate_fn,
                n_steps=n_steps,
                batch_size=batch_size,
                n_epochs=n_epochs,
                gamma=gamma,
                gae_lambda=gae_lambda,
                clip_range=clip_range,
                policy_kwargs=policy_kwargs,
                device=self.device,
                verbose=1
            )
        elif self.algorithm == "DDPG":
            self.model = DDPG(
                "MlpPolicy",
                self.env,
                learning_rate=learning_rate,
                buffer_size=1000000,
                learning_starts=1000,
                batch_size=batch_size,
                gamma=gamma,
                policy_kwargs=policy_kwargs,
                device=self.device,
                verbose=1
            )
        
        # Train the model
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=[eval_callback, checkpoint_callback],
            progress_bar=True
        )
        
        # Save the final model
        self.model.save(self.model_path)
        
        # Close evaluation environment
        eval_env.close()
    
    def get_action(
        self,
        state: np.ndarray,
        other_agents_states: Optional[List[np.ndarray]] = None
    ) -> np.ndarray:
        """
        Get the action from the RL policy.
        
        Args:
            state (np.ndarray): Current state
            other_agents_states (Optional[List[np.ndarray]]): States of other agents
            
        Returns:
            np.ndarray: Action from RL policy
        """
        if self.model is None:
            raise ValueError("Model not initialized. Call train() first.")
            
        # Combine state with other agents' states if provided
        if other_agents_states is not None:
            state = np.concatenate([state] + other_agents_states)
            
        # Get action from model
        action, _ = self.model.predict(state, deterministic=True)
        
        # Clip action to valid range (assuming [-0.01, 0.01] for Waterworld)
        action = np.clip(action, -0.01, 0.01)
        
        return action
    
    def compute_reward(
        self,
        action: np.ndarray,
        state: np.ndarray,
        other_agents_states: Optional[List[np.ndarray]] = None
    ) -> float:
        """
        Compute the reward for a baseline action using the Waterworld environment.
        
        Args:
            action (np.ndarray): Action to evaluate
            state (np.ndarray): Current state
            other_agents_states (Optional[List[np.ndarray]]): States of other agents
            
        Returns:
            float: Reward for the baseline action
        """
        if self.model is None:
            raise ValueError("Model not initialized. Call train() first.")
            
        # Get the current agent from the environment
        current_agent = self.env.env.env.agents[0]
        
        # Store the current environment state
        current_state = self.env.env.env.state()
        
        # Apply the action and get the reward
        self.env.env.env.step(action)
        reward = self.env.env.env.rewards[current_agent]
        
        # Restore the environment state
        self.env.env.env.reset()
        self.env.env.env.state(current_state)
        
        return reward
    
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
        if self.model is None:
            raise ValueError("Model not initialized. Call train() first.")
            
        # Combine state with other agents' states if provided
        if other_agents_states is not None:
            state = np.concatenate([state] + other_agents_states)
            
        # Sample multiple actions and compute average reward
        total_reward = 0.0
        for _ in range(num_samples):
            action, _ = self.model.predict(state, deterministic=False)
            action = np.clip(action, -0.01, 0.01)
            reward = self.compute_reward(action, state, other_agents_states)
            total_reward += reward
            
        return total_reward / num_samples

if __name__ == "__main__":
    # Example usage
    os.makedirs("models", exist_ok=True)
    
    # Create and train the baseline
    baseline = RLPolicyBaseline(
        action_dim=2,
        algorithm="PPO",
        model_path="models/waterworld_ppo",
        env_config={
            "render_mode": None,  # No rendering during training for speed
            "max_cycles": 1000,   # Maximum number of steps per episode
        },
        device="auto"  # Will use GPU if available
    )
    
    print("Starting RL policy training...")
    baseline.train(
        total_timesteps=1_000_000,  # 1M timesteps
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        eval_freq=10000,
        n_eval_episodes=10
    )
    print("Training completed!") 