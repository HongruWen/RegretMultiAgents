import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import time
from typing import Dict, List, Optional, Union

class GPUTimer:
    """Timer for GPU operations using CUDA events."""
    def __init__(self):
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)
        self.times = []  # Store multiple timing measurements
        
    def start(self):
        """Start timing."""
        torch.cuda.synchronize()
        self.start_event.record()
        
    def stop(self) -> float:
        """Stop timing and return elapsed time in milliseconds."""
        self.end_event.record()
        torch.cuda.synchronize()
        elapsed_time = self.start_event.elapsed_time(self.end_event)
        self.times.append(elapsed_time)
        return elapsed_time
        
    def reset(self):
        """Clear stored timing measurements."""
        self.times.clear()
        
    def average_time(self) -> float:
        """Return average time in milliseconds."""
        return np.mean(self.times) if self.times else 0.0

class BaseBaseline:
    """Base class for baseline policies (for extensibility)."""
    def __init__(self, env=None):
        self.env = env
        # Default max thrust (for continuous actions like WaterWorld)
        self.max_thrust = 0.01
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # If environment provided, use its action space to set max thrust
        if env is not None:
            try:
                if hasattr(env, 'action_spaces'):
                    # PettingZoo parallel API: action_spaces is a dict of agent spaces
                    first_agent = list(env.action_spaces.keys())[0] if len(env.action_spaces) > 0 else None
                    space = env.action_spaces[first_agent] if first_agent is not None else None
                elif hasattr(env, 'action_space'):
                    # Gym API or PettingZoo AEC API
                    space_attr = env.action_space
                    if callable(space_attr):
                        # Some envs have action_space(agent) method
                        agent_list = getattr(env, 'agents', None)
                        agent0 = agent_list[0] if agent_list else None
                        space = env.action_space(agent0) if agent0 is not None else env.action_space()
                    else:
                        space = space_attr
                else:
                    space = None
                if space is not None and hasattr(space, 'high'):
                    # For continuous Box spaces, take the max absolute value as max thrust
                    high_vals = torch.tensor(space.high, device=self.device)
                    self.max_thrust = float(torch.max(torch.abs(high_vals)))
            except Exception:
                # If any issue, default max_thrust remains 0.01
                pass

    def compute_action(self, obs, agent_id=None):
        """Override this method to define the baseline's action given an observation."""
        raise NotImplementedError

class RandomBaseline(BaseBaseline):
    """Baseline policy that takes random actions."""
    def compute_action(self, obs, agent_id=None):
        # For continuous action spaces: sample uniformly in the allowed range
        if hasattr(self.env, 'action_spaces'):
            space = self.env.action_spaces.get(agent_id, None)
            if space is not None and hasattr(space, 'low'):
                low, high = space.low, space.high
                return np.random.uniform(low, high)
        # Fallback: sample in [-max_thrust, max_thrust] for each dimension
        return np.random.uniform(-self.max_thrust, self.max_thrust, size=2)

class GreedyChaserBaseline(BaseBaseline):
    """Baseline that greedily chases the nearest food target (specific to WaterWorld)."""
    def compute_action(self, obs, agent_id=None):
        # Convert observation to numpy array for indexing
        obs = np.array(obs)
        if obs.size <= 2:
            # Observation too small (unexpected in WaterWorld) -> no action
            return np.array([0.0, 0.0])
        # Exclude the last two entries (collision indicators for food/poison)
        obs_core = obs[:-2]
        # Determine number of sensors and features per sensor from observation length
        n_sensors = None
        if obs_core.size % 8 == 0:
            # Speed features enabled (8 features per sensor)&#8203;:contentReference[oaicite:2]{index=2}
            n_sensors = obs_core.size // 8
        elif obs_core.size % 5 == 0:
            # Speed features disabled (5 features per sensor)
            n_sensors = obs_core.size // 5
        else:
            # Unexpected observation shape
            return np.array([0.0, 0.0])
        if n_sensors is None or n_sensors == 0:
            return np.array([0.0, 0.0])
        # Food distances are in indices [2*n_sensors : 3*n_sensors) of obs_core&#8203;:contentReference[oaicite:3]{index=3}
        start = 2 * n_sensors
        end = 3 * n_sensors
        if end > obs_core.size:
            return np.array([0.0, 0.0])
        food_distances = obs_core[start:end]
        if food_distances.size == 0:
            return np.array([0.0, 0.0])
        # Find the sensor with the smallest distance (closest food)
        sensor_index = int(np.argmin(food_distances))
        min_dist = food_distances[sensor_index]
        if min_dist >= 1.0:
            # No food in range (sensors report distance=1 if nothing detected)
            return np.array([0.0, 0.0])
        # Compute the direction vector of that sensor (evenly spaced around 360°)
        angle = 2 * np.pi * (sensor_index / n_sensors)
        direction = np.array([np.cos(angle), np.sin(angle)])
        # Normalize direction and scale by max thrust
        if np.linalg.norm(direction) > 0:
            direction = direction / np.linalg.norm(direction)
        action = direction * self.max_thrust
        return action

class OnlineRegretTracker:
    """
    Tracks online regret for specified agents in the WaterWorld environment by 
    comparing their actions to baseline policy actions. Includes GPU timing.
    """
    def __init__(self, env, baseline_policies):
        """
        :param env: WaterWorld environment instance (with parallel or AEC API).
        :param baseline_policies: dict mapping agent_id -> baseline policy.
               Baseline policy can be an instance of a Baseline class (GreedyChaserBaseline, RandomBaseline, etc.)
               or a string ("greedy", "random") to use a default baseline.
        """
        self.env = env
        self.baseline_policies = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gpu_timer = GPUTimer() if torch.cuda.is_available() else None
        self.compute_times = []  # Store computation times
        
        for agent_id, policy in baseline_policies.items():
            if isinstance(policy, str):
                policy_name = policy.lower()
                if policy_name == "greedy":
                    self.baseline_policies[agent_id] = GreedyChaserBaseline(env)
                elif policy_name == "random":
                    self.baseline_policies[agent_id] = RandomBaseline(env)
                else:
                    raise ValueError(f"Unknown baseline policy: {policy}")
            elif hasattr(policy, "compute_action"):
                # If an object with compute_action is provided, use it
                if isinstance(policy, BaseBaseline):
                    policy.env = env  # ensure env is set for baseline objects
                self.baseline_policies[agent_id] = policy
            else:
                raise ValueError("Baseline policy must be a Baseline instance or a string identifier.")
        # List of agents for which we will compute regret
        self.target_agents = list(self.baseline_policies.keys())
        # Data storage for one episode
        self.episode_actions = []  # list of dicts: each dict maps agent_id -> action at a timestep
        self.episode_rewards = []  # list of dicts: each dict maps agent_id -> reward at that timestep
        self.initial_seed = None   # store initial seed for reproducibility (if used)
        self.timing_stats = {
            'policy_compute_time': [],
            'env_step_time': [],
            'total_time': []
        }

    def start_episode(self, seed=None):
        """
        Begin tracking a new episode. Optionally resets the environment with a given seed.
        :param seed: (optional) Random seed for env.reset() to make the episode deterministic.
        :return: Initial observation from env.reset().
        """
        self.episode_actions.clear()
        self.episode_rewards.clear()
        self.initial_seed = seed
        # Reset the environment (with seed if provided) to start the episode
        if seed is not None:
            try:
                obs = self.env.reset(seed=seed)
            except TypeError:
                # If env.reset doesn't accept seed as parameter, try alternative seeding
                if hasattr(self.env, 'seed'):
                    self.env.seed(seed)
                obs = self.env.reset()
        else:
            obs = self.env.reset()
        return obs

    def record_step(self, actions, rewards):
        """
        Record the actions taken and rewards obtained at the current timestep.
        Call this after env.step() in the main loop.
        :param actions: dict of {agent_id: action} for all agents at this timestep.
        :param rewards: dict of {agent_id: reward} for all agents at this timestep.
        """
        # Store a copy to avoid mutation issues
        self.episode_actions.append(actions.copy())
        self.episode_rewards.append(rewards.copy())

    def compute_regrets(self):
        """
        Compute the regret trajectories with GPU timing measurements.
        """
        if len(self.episode_actions) == 0:
            print("No data recorded for this episode.")
            return None
            
        if self.gpu_timer:
            self.gpu_timer.reset()
            
        start_time = time.time()
        T = len(self.episode_actions)
        
        # Move data to GPU if available
        if torch.cuda.is_available():
            rewards_tensor = {agent: torch.tensor([r.get(agent, 0.0) for r in self.episode_rewards], 
                                                device=self.device) 
                            for agent in self.target_agents}
        
        actual_cum_rewards = {agent: [] for agent in self.target_agents}
        total_actual = {agent: 0.0 for agent in self.target_agents}
        
        # Time the main computation loop
        if self.gpu_timer:
            self.gpu_timer.start()
            
        for t in range(T):
            for agent in self.target_agents:
                reward_t = self.episode_rewards[t].get(agent, 0.0)
                total_actual[agent] += reward_t
                actual_cum_rewards[agent].append(total_actual[agent])
                
        if self.gpu_timer:
            compute_time = self.gpu_timer.stop()
            self.timing_stats['policy_compute_time'].append(compute_time)
            
        # Compute baseline rewards with timing
        baseline_cum_rewards = {agent: [] for agent in self.target_agents}
        total_baseline = {agent: 0.0 for agent in self.target_agents}
        
        for agent in self.target_agents:
            if self.gpu_timer:
                self.gpu_timer.start()
                
            if self.initial_seed is not None:
                obs = self.env.reset(seed=self.initial_seed)
            else:
                obs = self.env.reset()

            if isinstance(obs, tuple):
               obs = obs[0]
                
            total_baseline_reward = 0.0
            baseline_cum = []
            
            for t, action_dict in enumerate(self.episode_actions):
                step_actions = {}
                for ag, act in action_dict.items():
                    if ag == agent:
                        if self.gpu_timer:
                            self.gpu_timer.start()
                        step_actions[ag] = self.baseline_policies[ag].compute_action(obs[ag], agent_id=ag)
                        if self.gpu_timer:
                            self.timing_stats['policy_compute_time'].append(self.gpu_timer.stop())
                    else:
                        step_actions[ag] = act

                if self.gpu_timer:
                    self.gpu_timer.start()
                result = self.env.step(step_actions)
                if self.gpu_timer:
                    self.timing_stats['env_step_time'].append(self.gpu_timer.stop())

                if isinstance(result, tuple) and len(result) == 5:
                    new_obs, rewards, terminations, truncations, infos = result
                else:
                    new_obs, reward, done, info = result
                    rewards = reward if isinstance(reward, dict) else {agent: reward}
                    terminations = {ag: done for ag in step_actions.keys()}
                    truncations = {ag: False for ag in step_actions.keys()}

                reward_val = rewards.get(agent, 0.0) if isinstance(rewards, dict) else float(rewards)
                total_baseline_reward += reward_val
                baseline_cum.append(total_baseline_reward)

                done_flag = False
                if isinstance(terminations, dict):
                    done_flag = terminations.get(agent, False) or truncations.get(agent, False)
                    done_flag = done_flag or terminations.get("__all__", False) or truncations.get("__all__", False)
                elif isinstance(terminations, bool):
                    done_flag = terminations

                if done_flag:
                    while len(baseline_cum) < T:
                        baseline_cum.append(total_baseline_reward)
                    break
                obs = new_obs

            baseline_cum_rewards[agent] = baseline_cum
            total_baseline[agent] = total_baseline_reward

        # Compute final regret
        regret_data = {}
        for agent in self.target_agents:
            bl_cum = baseline_cum_rewards[agent]
            act_cum = actual_cum_rewards[agent]
            if len(bl_cum) < T:
                bl_cum += [bl_cum[-1]] * (T - len(bl_cum))
            if len(act_cum) < T:
                act_cum += [act_cum[-1]] * (T - len(act_cum))
            regret_data[agent] = [bl_cum[i] - act_cum[i] for i in range(T)]

        # Record total computation time
        total_time = time.time() - start_time
        self.timing_stats['total_time'].append(total_time)

        # Create DataFrames and print summaries
        regret_df = pd.DataFrame(regret_data)
        regret_df.index = range(1, T+1)

        summary_data = {
            'ActualReturn': [total_actual[ag] for ag in self.target_agents],
            'BaselineReturn': [total_baseline[ag] for ag in self.target_agents],
            'TotalRegret': [total_baseline[ag] - total_actual[ag] for ag in self.target_agents]
        }
        summary_df = pd.DataFrame(summary_data, index=self.target_agents)

        # Print timing statistics
        print("\nComputation Time Statistics:")
        print(f"Total computation time: {total_time:.4f} seconds")
        if self.gpu_timer:
            print(f"Average policy computation time: {np.mean(self.timing_stats['policy_compute_time']):.4f} ms")
            print(f"Average environment step time: {np.mean(self.timing_stats['env_step_time']):.4f} ms")

        print("\nRegret Summary (per agent):")
        print(summary_df)

        self.regret_df = regret_df
        self.summary_df = summary_df
        return regret_df

    def plot_regret_trajectories(self, show=True, save_path=None):
        """
        Plot cumulative regret over time for each tracked agent.
        :param show: If True, display the plot on screen.
        :param save_path: If provided, save the plot image to this file.
        """
        if not hasattr(self, 'regret_df'):
            print("No regret data to plot. Please run compute_regrets() first.")
            return
        plt.figure(figsize=(8, 6))
        for agent in self.target_agents:
            plt.plot(self.regret_df.index, self.regret_df[agent], label=f"{agent} Regret")
        plt.title("Cumulative Regret Trajectories")
        plt.xlabel("Time Step")
        plt.ylabel("Cumulative Regret (Baseline - Actual)")
        plt.legend()
        plt.grid(True)
        if save_path:
            plt.savefig(save_path)
            print(f"Saved regret plot to {save_path}")
        if show:
            plt.show()
