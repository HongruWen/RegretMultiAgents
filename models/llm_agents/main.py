from langchain.chat_models import ChatOpenAI
from pettingzoo_agent import PettingZooAgent



def main(agents, env, planning_horizon=5):
    observations, infos = env.reset()

    for name, agent in agents.items():
        agent.reset()

    warm_up = True

    while env.agents:
        if not warm_up:
            plans = {agent: agent.plan() for agent in env.agents}
            for i in range(planning_horizon):
                actions = {agent: plans[agent][i] for agent in env.agents}
                observations, rewards, terminations, truncations, infos = env.step(actions)
                # TODO: Add actions to message history
                
                obs_messages = {agent: agent.observe(observations[agent], rewards[agent], terminations[agent], truncations[agent], infos[agent]) for agent in env.agents}
                print(obs_messages)
        else:
            obs_messages = {agent: agent.observe(observations[agent]) for agent in env.agents}
            print(obs_messages)
            warm_up = False
    env.close()


def waterworld():
    from pettingzoo.sisl import waterworld_v4
    env = waterworld_v4.env(render_mode='human')

    # Replace the model with an Open source model, like Mistral-7B-v0.1 or Llama-3.1-8B-Instruct
    planning_horizon=5
    agents = {
        name: WaterworldAgent(name=name, model=ChatOpenAI(temperature=0.2), env=env, planning_horizon=planning_horizon)
        for name in env.possible_agents
    }
    main(agents, env, planning_horizon)

if __name__ == "__main__":
    # rock_paper_scissors()
    waterworld()
