import numpy as np
from pettingzoo.sisl import waterworld_v4
from regrets.baselines.rl_policy import RLPolicyBaseline
from regrets.baselines.greedy_chaser import GreedyTargetChaser
import time
import argparse

def simulate_baselines(baseline_type: str = "rl", render: bool = True, max_steps: int = 500):
    """
    Simulate the Waterworld environment with different baseline policies.
    
    Args:
        baseline_type (str): Type of baseline to use ("rl" or "greedy")
        render (bool): Whether to render the environment
        max_steps (int): Maximum number of steps per episode
    """
    # Create the environment with human rendering mode if requested
    env = waterworld_v4.env(render_mode="human" if render else None)
    
    # Reset the environment with a fixed seed for reproducibility
    env.reset(seed=42)
    
    # Initialize the appropriate baseline
    if baseline_type == "rl":
        baseline = RLPolicyBaseline(
            action_dim=2,
            algorithm="PPO",
            model_path="models/rl_baseline/waterworld_ppo",
            env_config={
                "render_mode": "human" if render else None,
                "max_cycles": max_steps,
            },
            device="auto"
        )
        baseline_name = "RL Policy"
    else:  # greedy
        baseline = GreedyTargetChaser(action_dim=2, max_accel=0.01)
        baseline_name = "Greedy Target Chaser"
    
    # Main simulation loop
    episode_rewards = {agent: 0 for agent in env.agents}
    step_count = 0
    
    print("\nStarting simulation...")
    print(f"pursuer_0: Using {baseline_name}")
    print("pursuer_1: Using random actions")
    print("Press Ctrl+C to stop the simulation")
    
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
                # Use baseline for pursuer_0, random actions for pursuer_1
                if agent == "pursuer_0":
                    action = baseline.get_action(observation)
                else:  # pursuer_1
                    action = env.action_space(agent).sample()
            
            # Step the environment
            env.step(action)
            
            # Increment step counter
            step_count += 1
            
            # Print progress every 100 steps
            if step_count % 100 == 0:
                print(f"\nStep {step_count}/{max_steps}")
                for agent_id, total_reward in episode_rewards.items():
                    policy = baseline_name if agent_id == "pursuer_0" else "Random"
                    print(f"{agent_id} ({policy}) - Total Reward: {total_reward:.4f}")
            
            # Add a small delay to make the visualization more watchable
            if render:
                time.sleep(0.01)
            
    except KeyboardInterrupt:
        print("\nSimulation stopped by user")
    
    # Close the environment when done
    env.close()
    
    # Final statistics
    print("\nFinal Statistics:")
    for agent_id, total_reward in episode_rewards.items():
        policy = baseline_name if agent_id == "pursuer_0" else "Random"
        print(f"{agent_id} ({policy}) - Final Total Reward: {total_reward:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulate Waterworld with different baseline policies")
    parser.add_argument("--baseline", choices=["rl", "greedy"], default="rl",
                      help="Type of baseline to use (rl or greedy)")
    parser.add_argument("--no-render", action="store_true",
                      help="Disable rendering")
    parser.add_argument("--max-steps", type=int, default=500,
                      help="Maximum number of steps per episode")
    
    args = parser.parse_args()
    
    simulate_baselines(
        baseline_type=args.baseline,
        render=not args.no_render,
        max_steps=args.max_steps
    ) 