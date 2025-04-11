import numpy as np
from pettingzoo.sisl import waterworld_v4
import json
import os
from regrets.dynamic_regret import calculate_episode_dynamic_regret

def run_simulation(n_steps=1000, render_mode=None):
    """Run simulation with random actions and track performance."""
    env = waterworld_v4.env(render_mode=render_mode)
    env.reset(seed=42)

    # Initialize tracking dictionaries for each agent
    episode_data = {agent: {
        'rewards': [],
        'actions': [],
        'total_reward': 0.0,
        'food_collected': 0,
        'poison_collisions': 0,
        'negative_rewards': []  # Track negative rewards for regret calculation
    } for agent in env.agents}

    # Track rewards at checkpoints
    checkpoints = {}
    checkpoint_interval = n_steps // 10
    step_count = 0

    for _ in range(n_steps):
        for agent in env.agent_iter():
            observation, reward, termination, truncation, info = env.last()
            reward = float(reward)  # Convert to Python float
            
            # Track reward
            episode_data[agent]['rewards'].append(reward)
            episode_data[agent]['total_reward'] += reward
            
            # Track negative rewards for regret calculation
            if reward < 0:
                episode_data[agent]['negative_rewards'].append(abs(reward))
            
            # Track food and poison stats from info
            if info.get('food_collected', 0) > 0:
                episode_data[agent]['food_collected'] += 1
            if info.get('poison_collision', 0) > 0:
                episode_data[agent]['poison_collisions'] += 1

            # Random action
            if termination or truncation:
                action = None
            else:
                action = env.action_space(agent).sample()
                episode_data[agent]['actions'].append(action)
            
            env.step(action)
            step_count += 1

            # Save checkpoint data after all agents have acted
            if step_count % checkpoint_interval == 0:
                checkpoint = {}
                for agent_id in env.agents:
                    checkpoint[agent_id] = {
                        'total_reward': float(episode_data[agent_id]['total_reward']),
                        'food_collected': episode_data[agent_id]['food_collected'],
                        'poison_collisions': episode_data[agent_id]['poison_collisions'],
                        'regret_so_far': float(sum(episode_data[agent_id]['negative_rewards']))
                    }
                checkpoints[f'step_{step_count}'] = checkpoint

    # Calculate final regrets
    final_regrets = {
        agent: float(sum(data['negative_rewards']))
        for agent, data in episode_data.items()
    }

    # Prepare final results
    results = {
        'checkpoints': checkpoints,
        'final_stats': {
            agent: {
                'total_reward': float(data['total_reward']),
                'food_collected': data['food_collected'],
                'poison_collisions': data['poison_collisions'],
                'dynamic_regret': final_regrets[agent]
            }
            for agent, data in episode_data.items()
        }
    }

    # Save results
    os.makedirs('results', exist_ok=True)
    with open('results/simulation_results.json', 'w') as f:
        json.dump(results, f, indent=4)

    env.close()
    return results

if __name__ == "__main__":
    results = run_simulation()
    print("\nSimulation completed. Results saved to results/simulation_results.json")
    print("\nFinal Statistics:")
    for agent, stats in results['final_stats'].items():
        print(f"\n{agent}:")
        print(f"Total Reward: {stats['total_reward']:.4f}")
        print(f"Food Collected: {stats['food_collected']}")
        print(f"Poison Collisions: {stats['poison_collisions']}")
        print(f"Dynamic Regret: {stats['dynamic_regret']:.4f}") 