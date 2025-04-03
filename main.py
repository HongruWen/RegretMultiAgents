import numpy as np
from pettingzoo.sisl import waterworld_v4
from regrets.regret import BaselineRegret
from regrets.baselines.rl_policy import RLPolicyBaseline

def run_simulation(max_steps: int = 500, render: bool = True, seed: int = 42):
    """
    Run a simulation with one RL agent and one random agent, tracking regret for both.
    
    Args:
        max_steps (int): Maximum number of steps per episode
        render (bool): Whether to render the environment
        seed (int): Random seed for reproducibility
    """
    # Create the environment with default config
    env = waterworld_v4.env(render_mode="human" if render else None)
    
    # Initialize the RL policy baseline for pursuer_0
    rl_baseline = RLPolicyBaseline(
        action_dim=2,
        algorithm="PPO",
        model_path="models/rl_baseline/waterworld_ppo",
        device="auto"
    )
    
    # Initialize regret trackers for both agents
    regret_trackers = {
        "pursuer_0": BaselineRegret(
            mode="rl",
            n_agents=2,
            model_path="models/rl_baseline/waterworld_ppo"
        ),
        "pursuer_1": BaselineRegret(
            mode="rl",
            n_agents=2,
            model_path="models/rl_baseline/waterworld_ppo"
        )
    }
    
    # Reset environment with seed
    env.reset(seed=seed)
    
    # Main simulation loop
    episode_rewards = {agent: 0 for agent in env.agents}
    step_count = 0
    
    print("\nStarting simulation...")
    print("pursuer_0: Using RL policy")
    print("pursuer_1: Using random actions")
    
    try:
        for agent in env.agent_iter():
            # Get the current observation, reward, termination status, and info
            observation, reward, termination, truncation, info = env.last()
            
            # Update episode rewards and regret
            episode_rewards[agent] += reward
            
            # Update regret with current state and reward
            regret = regret_trackers[agent].update(
                agent_id=agent,
                agent_reward=reward,
                state=observation
            )
            
            # If the agent is done or we've reached max steps, skip action
            if termination or truncation or step_count >= max_steps:
                action = None
            else:
                # Use RL policy for pursuer_0, random actions for pursuer_1
                if agent == "pursuer_0":
                    action = rl_baseline.get_action(observation)
                else:
                    action = env.action_space(agent).sample()
            
            # Step the environment
            env.step(action)
            
            # Increment step counter and print progress
            if agent == "pursuer_1":  # Only increment after both agents have acted
                step_count += 1
                if step_count % 100 == 0:
                    print(f"\nStep {step_count}/{max_steps}")
                    for agent_id in env.agents:
                        stats = regret_trackers[agent_id].get_performance_stats(agent_id)
                        print(f"{agent_id}:")
                        print(f"  Total Reward: {episode_rewards[agent_id]:.4f}")
                        print(f"  Current Regret: {stats['regret']:.4f}")
                        print(f"  Mean Reward: {stats['agent_rewards']['mean']:.4f}")
    
    except Exception as e:
        print(f"Error during simulation: {str(e)}")
    
    finally:
        # Close the environment
        env.close()
        
        # Print final statistics
        print("\nFinal Statistics:")
        for agent_id in env.agents:
            stats = regret_trackers[agent_id].get_performance_stats(agent_id)
            print(f"\n{agent_id}:")
            print(f"  Final Total Reward: {episode_rewards[agent_id]:.4f}")
            print(f"  Final Regret: {stats['regret']:.4f}")
            print(f"  Mean Reward: {stats['agent_rewards']['mean']:.4f}")
            print(f"  Std Reward: {stats['agent_rewards']['std']:.4f}")

if __name__ == "__main__":
    # Run the simulation
    run_simulation(
        max_steps=500,
        render=True,  # Set to False for faster execution
        seed=42
    )
