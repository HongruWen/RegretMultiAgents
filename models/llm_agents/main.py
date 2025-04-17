from langchain.chat_models import ChatOpenAI
from pettingzoo_agent import PettingZooAgent

from action_masking_agent import ActionMaskAgent  # isort: skip


def main(agents, env):
    observations, infos = env.reset()

    for name, agent in agents.items():
        agent.reset()

    # for agent_name in env.agent_iter():
    #     observation, reward, termination, truncation, info = env.last()
    #     obs_message = agents[agent_name].observe(
    #         observation, reward, termination, truncation, info
    #     )
    #     print(obs_message)
    #     if termination or truncation:
    #         action = None
    #     else:
    #         action = agents[agent_name].act()
    #     print(f"Action: {action}")
    #     env.step(action)
    # env.close()

    while env.agents:
        # this is where you would insert your policy
        actions = {agent: agent.act() for agent in env.agents}

        observations, rewards, terminations, truncations, infos = env.step(actions)
    env.close()

# Example usage
def rock_paper_scissors():
    from pettingzoo.classic import rps_v2

    env = rps_v2.env(max_cycles=3, render_mode="human")
    agents = {
        name: PettingZooAgent(name=name, model=ChatOpenAI(temperature=1), env=env)
        for name in env.possible_agents
    }
    main(agents, env)



def waterworld():
    from pettingzoo.sisl import waterworld_v4
    env = waterworld_v4.env(render_mode='human')

    # Replace the model with an Open source model, like Mistral-7B-v0.1 or Llama-3.1-8B-Instruct
    # And 
    agents = {
        name: WaterworldAgent(name=name, model=ChatOpenAI(temperature=0.2), env=env)
        for name in env.possible_agents
    }
    main(agents, env)

if __name__ == "__main__":
    # rock_paper_scissors()
    waterworld()
