import numpy as np
from model_manager import ModelManager
from LanguageAgent import WaterworldAgent, RawAgent



def main(agents, env, planning_horizon=5):
    print("=== Starting Simulation ===")
    env.reset()
    
    # Initialize agents with environment docs and instructions
    for name, agent in agents.items():
        agent.reset()
    
    # Main loop
    for agent_name in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()
        agents[agent_name].observe(
            observation, reward, termination, truncation, info
        )
        
        if termination or truncation:
            action = None
        else:
            action = agents[agent_name].act()
        
        env.step(action)
    
    print("=== Simulation Completed ===")
    env.close()


def waterworld():
    print("Setting up WaterWorld environment...")
    from pettingzoo.sisl import waterworld_v4
    env = waterworld_v4.env(
        n_pursuers=1,
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
        max_cycles=5,  # default is 100, set to 10 for testing
        render_mode='human'
    )
    print("Environment created successfully")

    # Initialize model manager
    print("\nInitializing Model Manager...")
    model_manager = ModelManager()
    
    # Add models
    print("\nAdding models to manager...")
    model_manager.add_model(
        name="deepseek-qwen-32b",
        model_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        temperature=0.2,
        max_length=256
    )
    print("Added DeepSeek-Qwen-32B model")
    
    # model_manager.add_model(
    #     name="llama2-7b",
    #     model_id="meta-llama/Llama-2-7b-chat-hf",
    #     temperature=0.2
    # )
    # print("Added Llama2-7B model")
    
    # model_manager.add_model(
    #     name="zephyr-7b",
    #     model_id="HuggingFaceH4/zephyr-7b-beta",
    #     temperature=0.2
    # )
    # print("Added Zephyr-7B model")
    
    # Create agents using a round-robin assignment of models
    print("\nCreating agents...")
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
    
    print("\nStarting main simulation loop...")
    main(agents, env, planning_horizon)

if __name__ == "__main__":
    waterworld()
