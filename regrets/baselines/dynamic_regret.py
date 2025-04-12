import numpy as np
import os
import matplotlib.pyplot as plt
from pettingzoo.sisl import waterworld_v4
from typing import Optional, Dict, List, Tuple
from datetime import datetime

# Import your baseline classes
from regrets.baselines.rl_policy import RLPolicyBaseline
from regrets.baselines.greedy_chaser import GreedyTargetChaser


def run_dynamic_regret_experiment(
    seed: int,
    max_steps: int = 500,
    candidate_type: str = "rl",      # "rl" or "greedy"
    baseline_type: str = "greedy",   # "rl" or "greedy"
    render: bool = False
) -> Tuple[List[float], List[float], List[float]]:
    """
    Run two parallel WaterWorld environments to estimate dynamic regret.
    Environment A uses candidate policy (vs. random), Environment B uses baseline policy (vs. random).
    Both share the same seed. We record the reward at each step for pursuer_0 in each environment
    and compute dynamic regret.

    Args:
        seed (int): Random seed for both environments.
        max_steps (int): Maximum steps (agent-iterations).
        candidate_type (str): "rl" or "greedy" for the candidate policy.
        baseline_type (str): "rl" or "greedy" for the baseline policy.
        render (bool): If True, render the environment (only recommended for debugging).

    Returns:
        (candidate_rewards, baseline_rewards, dynamic_regret):
            - candidate_rewards[t] = reward for candidate agent at step t
            - baseline_rewards[t] = reward for baseline agent at step t
            - dynamic_regret[t] = sum_{tau=1 to t} [ baseline_rewards[tau] - candidate_rewards[tau] ]
    """
    # ===== 1) Create two environments with the same config & seed =====
    env_config = {
        "render_mode": "human" if render else None,
        "max_cycles": max_steps
    }

    envA = waterworld_v4.env(**env_config)
    envB = waterworld_v4.env(**env_config)

    # Set random seeds
    envA.reset(seed=seed)
    envB.reset(seed=seed)

    # ===== 2) Define the baseline objects for each environment =====
    # Candidate policy for Environment A
    if candidate_type == "rl":
        candidate_policy = RLPolicyBaseline(
            action_dim=2,
            algorithm="PPO",
            model_path="models/rl_baseline/waterworld_ppo",
            env_config=env_config
        )
    elif candidate_type == "greedy":
        candidate_policy = GreedyTargetChaser(action_dim=2, max_accel=1.0, n_sensors=30)
    else:
        raise ValueError(f"Unknown candidate_type: {candidate_type}")

    # Baseline policy for Environment B
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

    # ===== 3) Simulation Loop =====
    # We'll track environment steps in lock-step. 
    # Because WaterWorld is an AEC (agent-iter) environment, we iterate agent by agent.
    # We'll only record pursuer_0's reward at the moment *it* acts in each environment.
    candidate_rewards = []
    baseline_rewards = []
    
    # We also track cumulative dynamic regret = sum_{tau=1 to t} (baseline - candidate).
    dynamic_regret = []

    # A small helper to see if we've ended an entire round in each environment
    stepsA = 0
    stepsB = 0

    # Convert each environment to an agent iterator
    agent_iterA = envA.agent_iter()
    agent_iterB = envB.agent_iter()

    # Some logic to know if an environment is "finished"
    doneA = False
    doneB = False

    # Because these are turn-based, we track who is "active" in each environment.
    # We want to line up steps for pursuer_0 in both envs as best as possible.
    while (not doneA or not doneB) and (stepsA < max_steps or stepsB < max_steps):
        
        # --- Environment A step ---
        if not doneA and stepsA < max_steps:
            agentA = next(agent_iterA, None)  # Which agent is next to act?
            if agentA is None:
                # This means environmentA is exhausted
                doneA = True
            else:
                obsA, rewA, termA, truncA, infoA = envA.last()
                
                # If pursuer_0 is about to move, that is the candidate agent
                if termA or truncA:
                    actionA = None  # no-op
                else:
                    if agentA == "pursuer_0":
                        # Candidate policy
                        actionA = candidate_policy.get_action(obsA)
                    else:
                        # Random action
                        actionA = envA.action_space(agentA).sample()

                # Step environment A
                envA.step(actionA)

                # If the agent was pursuer_0, record reward
                if agentA == "pursuer_0":
                    candidate_rewards.append(rewA)

                # Check if environment is done for this agent
                if termA or truncA:
                    if agentA == "pursuer_0":
                        # We still store the final reward for the candidate
                        # (In many tasks, the final "term" step has a reward)
                        pass
                
                # If we've collectively gone max_steps agent moves, we consider done
                stepsA += 1
                if stepsA >= max_steps:
                    doneA = True
        else:
            doneA = True

        # --- Environment B step ---
        if not doneB and stepsB < max_steps:
            agentB = next(agent_iterB, None)
            if agentB is None:
                doneB = True
            else:
                obsB, rewB, termB, truncB, infoB = envB.last()

                if termB or truncB:
                    actionB = None
                else:
                    if agentB == "pursuer_0":
                        # Baseline policy
                        actionB = baseline_policy.get_action(obsB)
                    else:
                        # Random action
                        actionB = envB.action_space(agentB).sample()

                envB.step(actionB)

                # Record reward if agentB == pursuer_0
                if agentB == "pursuer_0":
                    baseline_rewards.append(rewB)

                stepsB += 1
                if stepsB >= max_steps:
                    doneB = True
        else:
            doneB = True

        # --- Now compute dynamic regret if we have a matched time-step ---
        # We only compute regret at a “common index” if both candidate and baseline
        # have acted the same number of times. So we do it at min(len(candidate_rewards), len(baseline_rewards)).
        common_length = min(len(candidate_rewards), len(baseline_rewards))
        if common_length > 0:
            # immediate difference = baseline - candidate for the *most recent* matched step
            # If they have the same length, let's take the last index = common_length-1
            i = common_length - 1
            immediate_regret = baseline_rewards[i] - candidate_rewards[i]

            # cumulative sum
            if len(dynamic_regret) == 0:
                dynamic_regret.append(immediate_regret)
            else:
                dynamic_regret.append(dynamic_regret[-1] + immediate_regret)

    # Pad the regrets if one environment had fewer steps
    # Typically, we'll consider dynamic regret only up to the min steps both had for pursuer_0
    # but you can adjust as needed.
    return candidate_rewards, baseline_rewards, dynamic_regret


def demo_dynamic_regret_plot(
    candidate_type: str = "rl",
    baseline_type: str = "greedy",
    num_episodes: int = 3,
    max_steps: int = 500
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
        
        # We store dynamic_regret(t) for each t
        all_cum_regrets.append(dyn_regret)

    # Now plot the average dynamic regret over seeds (aligned by step index)
    max_len = max(len(r) for r in all_cum_regrets)
    # If different seeds produce different lengths, we can pad with last value
    aligned_regrets = []
    for regrets in all_cum_regrets:
        if len(regrets) < max_len:
            last_val = regrets[-1] if len(regrets) > 0 else 0.0
            regrets = regrets + [last_val]*(max_len - len(regrets))
        aligned_regrets.append(regrets)
    mean_regret = np.mean(aligned_regrets, axis=0)
    std_regret = np.std(aligned_regrets, axis=0)

    x = np.arange(max_len)

    plt.figure(figsize=(10, 6))
    plt.plot(x, mean_regret, label="Mean Dynamic Regret")
    plt.fill_between(x, mean_regret - std_regret, mean_regret + std_regret, alpha=0.2, label="Std Dev")
    plt.title(f"Dynamic Regret over {num_episodes} runs\nCandidate = {candidate_type}, Baseline = {baseline_type}")
    plt.xlabel("Number of Steps (pursuer_0 moves)")
    plt.ylabel("Cumulative Regret (Baseline - Candidate)")
    plt.legend()
    
    # Save or show
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outfile = f"dynamic_regret_{candidate_type}_vs_{baseline_type}_{timestamp}.png"
    plt.savefig(outfile)
    print(f"Saved plot to: {outfile}")
    plt.show()


if __name__ == "__main__":
    """
    Example usage:
      python dynamic_regret.py

    This script will run 3 short episodes to demonstrate dynamic regret
    for 'rl' (PPO) vs 'greedy', then produce a plot. Adjust as needed.
    """
    # Quick test or demo
    # Make sure you have a trained RL model under "models/rl_baseline/waterworld_ppo.zip" 
    # if you're using candidate_type="rl".
    demo_dynamic_regret_plot(candidate_type="rl", baseline_type="greedy", num_episodes=3, max_steps=300)
