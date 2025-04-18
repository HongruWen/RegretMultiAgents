from model_manager import ModelManager
from LanguageAgent import WaterworldAgent, RawAgent



def main(agents, env, planning_horizon=5):
    observations, infos = env.reset()

    # Initialize agents with environment docs and instructions
    for name, agent in agents.items():
        agent.reset()

    warm_up = True

    while env.agents:
        if not warm_up:
            # Get plans for all agents
            plans = {agent: agent.plan() for agent in env.agents}
            
            # Execute the planning horizon steps
            for step in range(planning_horizon):
                # Get current actions for all agents
                actions = {agent: plans[agent][step] for agent in env.agents}
                
                # Add actions to history before executing them
                for agent in env.agents:
                    agent.add_act_to_history()
                
                # Execute the step
                observations, rewards, terminations, truncations, infos = env.step(actions)
                
                # Add observations to history
                for agent in env.agents:
                    agent.observe(
                        observations[agent], 
                        rewards[agent], 
                        terminations[agent], 
                        truncations[agent], 
                        infos[agent]
                    )
                
                # Print current state
                print(f"Step {step + 1}:")
                for agent in env.agents:
                    print(f"  {agent}: Action={actions[agent]}, Reward={rewards[agent]}")
            
            # Reset planning state for next planning horizon
            for agent in env.agents:
                agent.reset_plan()
        else:
            # Warm-up phase
            for agent in env.agents:
                agent.observe(observations[agent])
            print("Warm-up phase completed")
            warm_up = False
    
    env.close()


def waterworld():
    from pettingzoo.sisl import waterworld_v4
    env = waterworld_v4.env(
        n_pursuers=2,
        n_evaders=5,
        n_poisons=10,
        n_coop=1,
        n_sensors=20,
        sensor_range=0.2,
        radius=0.015,
        obstacle_radius=0.2,
        n_obstacles=1,
        obstacle_coord=[(0.5, 0.5)],
        pursuer_max_accel=0.01,
        evader_speed=0.01,
        poison_speed=0.01,
        poison_reward=-1.0,
        food_reward=10.0,
        encounter_reward=0.01,
        thrust_penalty=-0.5,
        local_ratio=1.0,
        speed_features=True,
        max_cycles=10, # default is 100, set to 10 for testing
        render_mode='human'
    )

    # Initialize model manager
    model_manager = ModelManager()
    
    # Add models (you can add more models as needed)
    model_manager.add_model(
        name="mistral-7b",
        model_id="mistralai/Mistral-7B-v0.1",
        temperature=0.2
    )
    
    model_manager.add_model(
        name="llama2-7b",
        model_id="meta-llama/Llama-2-7b-chat-hf",
        temperature=0.2
    )
    
    model_manager.add_model(
        name="zephyr-7b",
        model_id="HuggingFaceH4/zephyr-7b-beta",
        temperature=0.2
    )
    
    # Create agents using a round-robin assignment of models
    planning_horizon = 5
    agents = {}
    
    for agent_name in env.possible_agents:
        # Get the next model in the round-robin sequence
        model = model_manager.get_next_model()
        
        # Create agent with the assigned model
        agents[agent_name] = WaterworldAgent(
            name=agent_name,
            model=model,
            env=env,
            planning_horizon=planning_horizon
        )
        
        # Print model assignment for debugging
        for model_name, model_obj in model_manager.models.items():
            if model_obj == model:
                print(f"Agent {agent_name} using model: {model_name}")
                break
    
    main(agents, env, planning_horizon)

if __name__ == "__main__":
    waterworld()
