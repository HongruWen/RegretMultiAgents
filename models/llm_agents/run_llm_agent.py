import numpy as np
from pettingzoo.sisl import waterworld_v4
import sys
import os
from huggingface_hub import login
from typing import Optional

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from regrets.regret import BaselineRegret
from regrets.baselines.rl_policy import RLPolicyBaseline
from models.llm_agents.llm_agent import WaterworldLLMAgent

def run_simulation(
    max_steps: int = 500,
    render: bool = True,
    seed: int = 42,
    model_name: str = "mistralai/Mistral-7B-v0.1",
    hf_token: Optional[str] = None
):
    """
    Run a simulation with one LLM agent and one random agent, tracking regret for both.
    
    Args:
        max_steps (int): Maximum number of steps per episode
        render (bool): Whether to render the environment
        seed (int): Random seed for reproducibility
        model_name (str): Name of the HuggingFace model to use
        hf_token (Optional[str]): HuggingFace API token for accessing the Inference API
    """
    # Set the HuggingFace API token in the environment
    if hf_token:
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_token
    
    # Create the environment with default config
    env = waterworld_v4.env(render_mode="human" if render else None)
    
    # Initialize the LLM agent for pursuer_0
    llm_agent = WaterworldLLMAgent(
        agent_id="pursuer_0",
        model_name=model_name,
        temperature=0.2
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
    print("pursuer_0: Using LLM agent (via HuggingFace Inference API)")
    print("pursuer_1: Using random actions")
    print(f"Model: {model_name}")
    print(f"Log file: {llm_agent.log_file}")
    
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
                # Use LLM agent for pursuer_0, random actions for pursuer_1
                if agent == "pursuer_0":
                    # Get observation message and generate action
                    obs_message = llm_agent.observe(
                        observation,
                        reward,
                        termination,
                        truncation,
                        info
                    )
                    action = llm_agent.act()
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
    # Get HuggingFace API token from environment variable
    hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    
    if not hf_token:
        print("Warning: HUGGINGFACEHUB_API_TOKEN environment variable not set.")
        print("Please set it to your HuggingFace API token to use the Inference API.")
        print("You can get your token from: https://huggingface.co/settings/tokens")
        sys.exit(1)
    
    # Run the simulation
    run_simulation(
        max_steps=500,
        render=True,  # Set to False for faster execution
        seed=42,
        model_name="mistralai/Mistral-7B-v0.1",
        hf_token=hf_token
    ) 