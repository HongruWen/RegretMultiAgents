import numpy as np
from pettingzoo.sisl import waterworld_v4

# Create the environment with human rendering mode
env = waterworld_v4.env(render_mode="human")

# Reset the environment with a fixed seed for reproducibility
env.reset(seed=42)

# Main simulation loop
for agent in env.agent_iter():
    # Get the current observation, reward, termination status, and info
    observation, reward, termination, truncation, info = env.last()

    # If the agent is done, skip action
    if termination or truncation:
        action = None
    else:
        # For now, we'll use random actions
        action = env.action_space(agent).sample()

    # Step the environment
    env.step(action)

# Close the environment when done
env.close()
