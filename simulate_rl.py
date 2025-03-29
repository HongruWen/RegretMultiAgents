import numpy as np
from pettingzoo.sisl import waterworld_v4
from regrets.baselines.rl_policy import RLPolicyBaseline
import time

def simulate_rl_policy():
    # Create the environment with human rendering mode
    env = waterworld_v4.env(render_mode="human")
    
    # Reset the environment with a fixed seed for reproducibility
    env.reset(seed=42)
    
    # Initialize the RL policy baseline with the trained model
    baseline = RLPolicyBaseline(
        action_dim=2,
        algorithm="PPO",
        model_path="models/rl_baseline/waterworld_ppo",
        env_config={
            "render_mode": "human",
            "max_cycles": 1000,
        },
        device="auto"
    )
    
    # Main simulation loop
    episode_rewards = {agent: 0 for agent in env.agents}
    step_count = 0
    max_steps = 1000
    
    print("Starting simulation...")
    print("pursuer_0: Using trained RL policy")
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
                # Use RL policy for pursuer_0, random actions for pursuer_1
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
                print(f"\nStep {step_count}")
                for agent_id, total_reward in episode_rewards.items():
                    policy = "RL Policy" if agent_id == "pursuer_0" else "Random"
                    print(f"{agent_id} ({policy}) - Total Reward: {total_reward:.4f}")
            
            # Add a small delay to make the visualization more watchable
            time.sleep(0.01)
            
    except KeyboardInterrupt:
        print("\nSimulation stopped by user")
    
    # Close the environment when done
    env.close()
    
    # Final statistics
    print("\nFinal Statistics:")
    for agent_id, total_reward in episode_rewards.items():
        policy = "RL Policy" if agent_id == "pursuer_0" else "Random"
        print(f"{agent_id} ({policy}) - Final Total Reward: {total_reward:.4f}")

if __name__ == "__main__":
    simulate_rl_policy() 