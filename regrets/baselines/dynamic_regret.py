import numpy as np
import os
import matplotlib.pyplot as plt
from typing import List, Tuple
from datetime import datetime

# PettingZoo import (older style parallel_env -> returns (obs, rews, dones, infos) as lists)
from pettingzoo.sisl import waterworld_v4

# Import your baseline classes
from regrets.baselines.rl_policy import RLPolicyBaseline
from regrets.baselines.greedy_chaser import GreedyTargetChaser


def extract_observation(raw_obs):
    """
    Convert the environment's raw observation (which might be a list, tuple, or dict)
    into a 1D NumPy array so that slicing like state[a:b] works.
    """
    if isinstance(raw_obs, dict):
        # Some older PettingZoo wrappers return obs["observation"]
        # If that's the case, pick that out and convert to NumPy array
        if "observation" in raw_obs:
            return np.array(raw_obs["observation"], dtype=np.float32)
        else:
            # If no "observation" key, try converting the entire dict
            # (Though usually you'd adapt to your environment's structure)
            return np.array(list(raw_obs.values()), dtype=np.float32)
    elif isinstance(raw_obs, (list, tuple)):
        # Convert list or tuple of floats
        return np.array(raw_obs, dtype=np.float32)
    elif isinstance(raw_obs, np.ndarray):
        # Already a NumPy array
        return raw_obs.astype(np.float32)
    else:
        # Last resort: just try converting
        return np.array(raw_obs, dtype=np.float32)


def run_dynamic_regret_experiment(
    seed: int,
    max_steps: int = 500,
    candidate_type: str = "greedy",   # "rl" or "greedy"
    baseline_type: str = "greedy",    # "rl" or "greedy"
    render: bool = False
) -> Tuple[List[float], List[float], List[float]]:
    """
    1) Creates two parallel WaterWorld environments (older style) with the same seed:
         - Env A: pursuer_0 uses candidate policy, pursuer_1 random
         - Env B: pursuer_0 uses baseline  policy, pursuer_1 random
    2) Each step:
       - Convert obs for pursuer_0 to a NumPy array
       - Get action from candidate or baseline
       - Random action for pursuer_1
       - Step environment
       - Extract pursuer_0's reward
       - Accumulate dynamic regret = sum of (baseline - candidate)

    Returns:
      candidate_rewards, baseline_rewards, dynamic_regret
    """

    # ============ 1) Create parallel envs with same seed ============
    env_config = {
        "render_mode": "human" if render else None,
        "max_cycles": max_steps
    }

    envA = waterworld_v4.parallel_env(**env_config)
    envB = waterworld_v4.parallel_env(**env_config)

    obsA = envA.reset(seed=seed)  # returns a tuple/list of observations
    obsB = envB.reset(seed=seed)

    # List of agent names in each environment, e.g. ["pursuer_0", "pursuer_1"]
    agentsA = envA.agents
    agentsB = envB.agents

    # Indices for pursuer_0
    if "pursuer_0" not in agentsA or "pursuer_0" not in agentsB:
        raise ValueError("pursuer_0 not found in environment's agent list.")
    p0_idxA = agentsA.index("pursuer_0")
    p0_idxB = agentsB.index("pursuer_0")

    # ============ 2) Candidate & Baseline policies ============
    if candidate_type == "rl":
        candidate_policy = RLPolicyBaseline(
            action_dim=2,
            algorithm="PPO",
            model_path="models/rl_baseline/waterworld_ppo",  # or your path
            env_config=env_config
        )
    elif candidate_type == "greedy":
        candidate_policy = GreedyTargetChaser(action_dim=2, max_accel=1.0, n_sensors=30)
    else:
        raise ValueError(f"Unknown candidate_type: {candidate_type}")

    if baseline_type == "rl":
        baseline_policy = RLPolicyBaseline(
            action_dim=2,
            algorithm="PPO",
            model_path="models/rl_baseline/waterworld_ppo",
            env_config=env_config
        )
    elif baseline_type == "greedy":
        baseline_policy = GreedyTargetChaser(action_dim=2, max_accel=1.0, n_sensors=30)
    else:
        raise ValueError(f"Unknown baseline_type: {baseline_type}")

    # ============ 3) Step loop & compute dynamic regret ============
    candidate_rewards = []
    baseline_rewards = []
    dynamic_regret = []

    for step_i in range(max_steps):
        # If either environment is fully done (no agents left), we stop
        if len(envA.agents) == 0 or len(envB.agents) == 0:
            break

        # 3a) Build actions for envA
        actionsA = [None] * len(envA.agents)
        for i, agent in enumerate(envA.agents):
            if agent == "pursuer_0":
                # Convert obs to np.array, get candidate policy action
                obs_array = extract_observation(obsA[i])
                actionsA[i] = candidate_policy.get_action(obs_array)
            else:
                # random
                actionsA[i] = envA.action_space(agent).sample()

        # Step envA
        obsA, rewsA, donesA, infosA = envA.step(actionsA)

        # candidate reward: index p0_idxA
        cand_r = rewsA[p0_idxA] if len(rewsA) > p0_idxA else 0.0
        candidate_rewards.append(cand_r)

        # 3b) Build actions for envB
        actionsB = [None] * len(envB.agents)
        for i, agent in enumerate(envB.agents):
            if agent == "pursuer_0":
                obs_array = extract_observation(obsB[i])
                actionsB[i] = baseline_policy.get_action(obs_array)
            else:
                actionsB[i] = envB.action_space(agent).sample()

        obsB, rewsB, donesB, infosB = envB.step(actionsB)

        # baseline reward
        base_r = rewsB[p0_idxB] if len(rewsB) > p0_idxB else 0.0
        baseline_rewards.append(base_r)

        # immediate regret
        diff = base_r - cand_r
        if len(dynamic_regret) == 0:
            dynamic_regret.append(diff)
        else:
            dynamic_regret.append(dynamic_regret[-1] + diff)

        # if all done in either environment, stop
        if all(donesA) or len(envA.agents) == 0:
            break
        if all(donesB) or len(envB.agents) == 0:
            break

    return candidate_rewards, baseline_rewards, dynamic_regret


def demo_dynamic_regret_plot(
    candidate_type: str = "greedy",
    baseline_type: str = "greedy",
    num_episodes: int = 3,
    max_steps: int = 300
):
    """
    Runs multiple seeds, collects dynamic regret, and plots an example.
    """
    all_cum_regrets = []

    for seed in range(num_episodes):
        cand_rewards, base_rewards, dyn_regret = run_dynamic_regret_experiment(
            seed=seed,
            max_steps=max_steps,
            candidate_type=candidate_type,
            baseline_type=baseline_type,
            render=False
        )
        all_cum_regrets.append(dyn_regret)

    if len(all_cum_regrets) == 0:
        print("No data collected; environment ended immediately or error occurred.")
        return

    # Pad different run lengths to the same size
    max_len = max(len(reg) for reg in all_cum_regrets)
    padded = []
    for reg in all_cum_regrets:
        if len(reg) < max_len:
            final_val = reg[-1] if len(reg) > 0 else 0.0
            reg += [final_val] * (max_len - len(reg))
        padded.append(reg)

    # Mean / std of regrets across seeds
    mean_regret = np.mean(padded, axis=0)
    std_regret = np.std(padded, axis=0)

    x = np.arange(max_len)

    plt.figure(figsize=(10, 6))
    plt.plot(x, mean_regret, label="Mean Dynamic Regret")
    plt.fill_between(x, mean_regret - std_regret, mean_regret + std_regret, alpha=0.2, label="Std Dev")

    plt.title(f"Dynamic Regret across {num_episodes} seeds\nCandidate={candidate_type}, Baseline={baseline_type}")
    plt.xlabel("Step")
    plt.ylabel("Cumulative Regret (Baseline - Candidate)")
    plt.legend()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = f"dynamic_regret_{candidate_type}_vs_{baseline_type}_{timestamp}.png"
    plt.savefig(out_file)
    print(f"Saved plot to: {out_file}")
    plt.show()


if __name__ == "__main__":
    """
    Example usage (greedy vs. greedy):
      cd RegretMultiAgents
      export KMP_DUPLICATE_LIB_OK=TRUE
      python -m regrets.baselines.dynamic_regret
    """
    # Quick test: 3 episodes, 300 max steps each, "greedy" vs "greedy"
    demo_dynamic_regret_plot(candidate_type="greedy", baseline_type="greedy", num_episodes=3, max_steps=300)
