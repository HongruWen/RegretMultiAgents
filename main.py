import numpy as np
from pettingzoo.sisl import waterworld_v4
from regret import RegretTracker

# Create the environment with human rendering mode
env = waterworld_v4.env(render_mode="human")

# Reset the environment with a fixed seed for reproducibility
env.reset(seed=42)

# Create a regret tracker for each agent
# Since each agent has a 2D continuous action space, we'll discretize it into 9 actions
# (combinations of [left, stay, right] x [up, stay, down])
NUM_DISCRETE_ACTIONS = 9
agent_regret_trackers = {agent: RegretTracker(num_actions=NUM_DISCRETE_ACTIONS) for agent in env.agents}

# Function to convert continuous action to discrete
def get_discrete_action(action_idx):
    """Convert discrete action index to continuous action vector."""
    # Define 9 possible actions: combinations of [-1, 0, 1] for both dimensions
    actions = [
        [-0.01, -0.01],  # down-left
        [0, -0.01],      # down
        [0.01, -0.01],   # down-right
        [-0.01, 0],      # left
        [0, 0],          # stay
        [0.01, 0],       # right
        [-0.01, 0.01],   # up-left
        [0, 0.01],       # up
        [0.01, 0.01]     # up-right
    ]
    return np.array(actions[action_idx])

# Main simulation loop
episode_rewards = {agent: 0 for agent in env.agents}
step_count = 0
max_steps = 1000

for agent in env.agent_iter():
    # Get the current observation, reward, termination status, and info
    observation, reward, termination, truncation, info = env.last()
    
    # Update episode rewards
    episode_rewards[agent] += reward
    
    # If the agent is done or we've reached max steps, skip action
    if termination or truncation or step_count >= max_steps:
        action = None
    else:
        # Get the strategy from regret matching
        strategy = agent_regret_trackers[agent].get_strategy()
        
        # Choose action based on the strategy
        action_idx = np.random.choice(NUM_DISCRETE_ACTIONS, p=strategy)
        action = get_discrete_action(action_idx)
        
        # Update regret tracker with the reward
        agent_regret_trackers[agent].update(action_idx, reward)
    
    # Step the environment
    env.step(action)
    
    # Increment step counter
    step_count += 1
    
    # Print progress every 100 steps
    if step_count % 100 == 0:
        print(f"\nStep {step_count}")
        for agent_id, regret_tracker in agent_regret_trackers.items():
            avg_regret = regret_tracker.get_average_regret()
            total_reward = episode_rewards[agent_id]
            print(f"{agent_id} - Average Regret: {avg_regret:.4f}, Total Reward: {total_reward:.4f}")

# Close the environment when done
env.close()

# Final statistics
print("\nFinal Statistics:")
for agent_id, regret_tracker in agent_regret_trackers.items():
    avg_regret = regret_tracker.get_average_regret()
    total_reward = episode_rewards[agent_id]
    print(f"{agent_id} - Final Average Regret: {avg_regret:.4f}, Total Reward: {total_reward:.4f}")
