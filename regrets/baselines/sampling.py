import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
from tqdm import tqdm
import os
from datetime import datetime
import torch
import argparse

from regrets.baselines.rl_policy import RLPolicyBaseline
from regrets.baselines.greedy_chaser import GreedyTargetChaser
from pettingzoo.sisl import waterworld_v4

def run_experiment(
    seed: int,
    max_steps: int = 500,
    render: bool = False,
    baseline_type: str = "rl",
    baseline_agent_id: str = "pursuer_0"
) -> Dict[str, float]:
    """
    Run a single experiment with the specified baseline.
    
    Args:
        seed (int): Random seed for reproducibility
        max_steps (int): Maximum number of steps per episode
        render (bool): Whether to render the environment
        baseline_type (str): Type of baseline to use ("rl" or "greedy")
        baseline_agent_id (str): ID of the agent using the baseline policy
        
    Returns:
        Dict[str, float]: Final rewards for each agent
    """
    # Create the environment with default config
    env_config = {
        "render_mode": "human" if render else None,
        "max_cycles": max_steps
    }
    env = waterworld_v4.env(**env_config)
    
    # Initialize the appropriate baseline
    if baseline_type == "rl":
        baseline = RLPolicyBaseline(
            action_dim=2,
            algorithm="PPO",
            model_path="models/rl_baseline/waterworld_ppo",
            env_config=env_config,
            device="auto"
        )
    elif baseline_type == "greedy":
        baseline = GreedyTargetChaser(
            action_dim=2,
            max_accel=1.0,
            n_sensors=30
        )
    else:
        raise ValueError(f"Unknown baseline type: {baseline_type}")
    
    # Reset environment with seed
    env.reset(seed=seed)
    
    # Main simulation loop
    episode_rewards = {agent: 0 for agent in env.agents}
    step_count = 0
    
    try:
        for agent in env.agent_iter():
            # Get the current observation, reward, termination status, and info
            observation, reward, termination, truncation, info = env.last()
            
            # Update episode rewards
            episode_rewards[agent] += reward
            
            # If the agent is done or we've reached max steps, skip action
            if termination or truncation or step_count >= max_steps:
                action = None
            else:
                # Use baseline policy for the specified agent, random actions for others
                if agent == baseline_agent_id:
                    action = baseline.get_action(observation)
                else:
                    action = env.action_space(agent).sample()
            
            # Step the environment
            env.step(action)
            
            # Increment step counter
            step_count += 1
            
            # Print progress every 100 steps
            if step_count % 100 == 0:
                print(f"Step {step_count}/{max_steps}, Current rewards: {episode_rewards}")
    
    except Exception as e:
        print(f"Error during experiment: {str(e)}")
    finally:
        # Close the environment
        env.close()
    
    return episode_rewards

def run_multiple_experiments(
    num_experiments: int = 10,
    max_steps: int = 500,
    render: bool = False,
    baseline_type: str = "rl",
    baseline_agent_id: str = "pursuer_0"
) -> Dict[str, List[float]]:
    """
    Run multiple experiments sequentially.
    
    Args:
        num_experiments (int): Number of experiments to run
        max_steps (int): Maximum number of steps per episode
        render (bool): Whether to render the environment
        baseline_type (str): Type of baseline to use ("rl" or "greedy")
        baseline_agent_id (str): ID of the agent using the baseline policy
        
    Returns:
        Dict[str, List[float]]: List of final rewards for each agent
    """
    results = {agent: [] for agent in ["pursuer_0", "pursuer_1"]}
    
    for seed in tqdm(range(num_experiments), desc="Running experiments"):
        try:
            episode_rewards = run_experiment(seed, max_steps, render, baseline_type, baseline_agent_id)
            for agent in results:
                results[agent].append(episode_rewards[agent])
        except Exception as e:
            print(f"Error in experiment {seed}: {str(e)}")
            continue
    
    return results

def plot_results(
    results: Dict[str, List[float]],
    save_dir: str = "results",
    prefix: str = "rl"
):
    """
    Plot the results of multiple experiments.
    
    Args:
        results (Dict[str, List[float]]): Results from multiple experiments
        save_dir (str): Directory to save the plots
        prefix (str): Prefix for saved files
    """
    # Create results directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Create timestamp for unique filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Plot 1: Box plot of rewards
    plt.figure(figsize=(12, 6))
    plt.boxplot([results[agent] for agent in results.keys()], labels=list(results.keys()))
    plt.title("Distribution of Final Rewards")
    plt.ylabel("Final Reward")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{prefix}_rewards_boxplot_{timestamp}.png"))
    plt.close()
    
    # Plot 2: Line plot of rewards over experiments
    plt.figure(figsize=(12, 6))
    for agent in results:
        plt.plot(results[agent], label=agent)
    plt.title("Final Rewards Over Experiments")
    plt.xlabel("Experiment Number")
    plt.ylabel("Final Reward")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{prefix}_rewards_lineplot_{timestamp}.png"))
    plt.close()
    
    # Save numerical results
    np.savez(
        os.path.join(save_dir, f"{prefix}_rewards_{timestamp}.npz"),
        **{agent: np.array(rewards) for agent, rewards in results.items()}
    )
    
    # Print and save detailed statistics
    stats_file = os.path.join(save_dir, f"{prefix}_statistics_{timestamp}.txt")
    with open(stats_file, 'w') as f:
        f.write(f"Detailed Statistics for {prefix.upper()} Experiments\n")
        f.write("=====================================\n\n")
        
        for agent in results:
            rewards = np.array(results[agent])
            if len(rewards) > 0:
                stats = {
                    "Mean": np.mean(rewards),
                    "Std": np.std(rewards),
                    "Min": np.min(rewards),
                    "Max": np.max(rewards),
                    "Median": np.median(rewards),
                    "Q1": np.percentile(rewards, 25),
                    "Q3": np.percentile(rewards, 75),
                    "Skewness": np.mean(((rewards - np.mean(rewards)) / np.std(rewards)) ** 3),
                    "Kurtosis": np.mean(((rewards - np.mean(rewards)) / np.std(rewards)) ** 4) - 3
                }
                
                # Write to file
                f.write(f"\n{agent}:\n")
                for stat_name, stat_value in stats.items():
                    f.write(f"  {stat_name}: {stat_value:.4f}\n")
                
                # Print to console
                print(f"\n{agent}:")
                for stat_name, stat_value in stats.items():
                    print(f"  {stat_name}: {stat_value:.4f}")
            else:
                f.write(f"\n{agent}: No valid results\n")
                print(f"\n{agent}: No valid results")

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run experiments with different baseline policies")
    parser.add_argument("--baseline", type=str, default="rl", choices=["rl", "greedy"],
                      help="Type of baseline to use (rl or greedy)")
    parser.add_argument("--num_experiments", type=int, default=100,
                      help="Number of experiments to run")
    parser.add_argument("--max_steps", type=int, default=500,
                      help="Maximum number of steps per episode")
    parser.add_argument("--render", action="store_true",
                      help="Whether to render the environment")
    args = parser.parse_args()
    
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, "results")
    
    print("Starting sampling script...")
    print(f"Script directory: {script_dir}")
    print(f"Results directory: {results_dir}")
    print(f"Using baseline: {args.baseline}")
    
    # Check if model file exists for RL baseline
    if args.baseline == "rl":
        model_path = "models/rl_baseline/waterworld_ppo"
        if os.path.exists(f"{model_path}.zip"):
            print(f"Found model file at {model_path}")
        else:
            print(f"Warning: Model file not found at {model_path}")
            print("Please ensure the model is trained before running experiments")
            exit(1)
    
    print("\nStarting experiments...")
    # Run multiple experiments
    results = run_multiple_experiments(
        num_experiments=args.num_experiments,
        max_steps=args.max_steps,
        render=args.render,
        baseline_type=args.baseline,
        baseline_agent_id="pursuer_0"  # Only pursuer_0 uses the baseline policy
    )
    
    print("\nPlotting and saving results...")
    # Plot and save results
    plot_results(results, save_dir=results_dir, prefix=args.baseline)
    print("Done!") 