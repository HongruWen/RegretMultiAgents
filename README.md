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

### Setup

1. Install the required packages:
```bash
pip install -r requirements.txt
```

2. Set up your Hugging Face API token:
- Get your API token from https://huggingface.co/settings/tokens
- Set the environment variable:
```bash
# PowerShell
$env:HUGGINGFACEHUB_API_TOKEN="your_token_here"

# Command Prompt
set HUGGINGFACEHUB_API_TOKEN=your_token_here

# Linux/Mac
export HUGGINGFACEHUB_API_TOKEN=your_token_here
```

### Running the LLM Agent

To run a simulation with the LLM agent:

```bash
python models/llm_agents/run_llm_agent.py
```

This will:
- Initialize a Waterworld environment with two agents
- Use the LLM agent for pursuer_0 and random actions for pursuer_1
- Track regret for both agents
- Log all LLM interactions to `models/llm_agents/logs/`

The agent uses the Hugging Face Inference API to generate actions based on the current state of the environment. All interactions are logged for analysis.

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

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the Simulation

To run the basic simulation with random agent movements:
```bash
python main.py
```

The simulation will open a window showing:
- Purple circles: Pursuer agents
- Red circles: Food targets
- Green circles: Poison targets
- Black lines: Agent sensors

## Project Structure

- `main.py`: Main simulation script
- `requirements.txt`: Project dependencies
- `.gitignore`: Git ignore rules

## Dependencies

- numpy>=1.21.0
- matplotlib>=3.4.0
- pettingzoo>=1.22.0
- pygame>=2.5.0
- pymunk>=6.0.0
- scipy>=1.7.0

## Future Enhancements

1. Implement regret-based learning algorithms
2. Add visualization of agent performance metrics
3. Implement cooperative strategies
4. Add configuration options for environment parameters

## License

This project is open source and available under the MIT License. 