# Regret Multi-Agent Simulation

This project implements a multi-agent simulation using the Waterworld environment from PettingZoo. The simulation demonstrates agent interactions in a 2D space where agents (pursuers) attempt to consume food while avoiding poison.

## Environment Overview

The Waterworld environment features:
- Multiple pursuer agents (purple circles)
- Food targets (red circles)
- Poison targets (green circles)
- Sensor-based perception (black lines)

Each agent has:
- Continuous action space (2D movement)
- Sensor-based observation space
- Reward system based on interactions with food and poison

## LLM Agent

The project includes an LLM-based agent that uses the Hugging Face Inference API to make decisions in the Waterworld environment. The agent processes observations and generates continuous actions using language models.





## Setup

1. Clone the repository:
```bash
git clone git@github.com:HongruWen/RegretMultiAgents.git
cd RegretMultiAgents
```

2. Create and activate a virtual environment:
```bash
python -m venv env
# On Windows:
env\Scripts\activate
# On Unix or MacOS:
source env/bin/activate
```


### Running the LLM Agent

To run a simulation with the LLM agent:

```bash
python models/llm_agents/main.py
```

This will:
- Initialize a Waterworld environment with two agents
- Use the LLM agent for pursuer_0 and random actions for pursuer_1
- Track regret for both agents

The agent uses the Hugging Face Inference API to generate actions based on the current state of the environment. All interactions are logged for analysis.


## Project Structure

- `simulate_rl.py`: Simulation with random policies
- `simulate_baselines.py`: Simulation with baseline policies (RL and greedy)
- `models/llm_agents/main.py`: Main script for LLM agent simulation
- `regrets/`: Implementations of regret-based algorithms and baselines
- `models/`: Model implementations including LLM agents
- `requirements.txt`: Project dependencies
- `.gitignore`: Git ignore rules


## License

This project is open source and available under the MIT License. 