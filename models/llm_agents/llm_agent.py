from typing import Dict, Optional, Any
import numpy as np
from langchain_community.chat_models import ChatHuggingFace
from langchain.prompts import PromptTemplate
from langchain.output_parsers import RegexParser
from langchain.schema import HumanMessage, SystemMessage
import json
import os
from datetime import datetime
from huggingface_hub import InferenceClient

class WaterworldLLMAgent:
    """LLM agent for Waterworld environment using LangChain and HuggingFace models."""
    
    def __init__(
        self,
        agent_id: str,
        model_name: str = "mistralai/Mistral-7B-v0.1",  # or "meta-llama/Llama-2-7b-chat-hf"
        temperature: float = 0.2,
        max_tokens: int = 100,
        log_dir: str = "models/llm_agents/logs"
    ):
        """
        Initialize the LLM agent.
        
        Args:
            agent_id (str): ID of the agent (e.g., 'pursuer_0')
            model_name (str): Name of the HuggingFace model to use
            temperature (float): Temperature for sampling
            max_tokens (int): Maximum number of tokens to generate
            log_dir (str): Directory to store interaction logs
        """
        self.agent_id = agent_id
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.model_name = model_name
        
        # Create log directory
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        # Create a new log file for this session
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(log_dir, f"llm_interactions_{timestamp}.json")
        
        # Initialize the HuggingFace client
        self.client = InferenceClient()
        
        # Create the prompt template
        self.prompt_template = PromptTemplate(
            input_variables=["observation", "reward", "termination", "truncation", "total_return"],
            template="""
You are an agent in the Waterworld environment. Your goal is to maximize your return by catching food particles while avoiding poison particles.

Current state:
Observation: {observation}
Reward: {reward}
Termination: {termination}
Truncation: {truncation}
Return: {total_return}

The observation is a 242-dimensional vector containing:
- Positions and velocities of all agents
- Positions and types of food and poison particles
- Sensor readings for nearby particles

Your action should be a 2D continuous vector in the range [-0.01, 0.01] for both dimensions.
Respond with your action in the format: Action: [x, y]

For example:
Action: [0.01, -0.005]
"""
        )
        
        # Create the action parser
        self.action_parser = RegexParser(
            regex=r"Action: \[([-\d.]+),\s*([-\d.]+)\]",
            output_keys=["x", "y"],
            default_output_key="action"
        )
        
        # Initialize message history and rewards
        self.message_history = []
        self.reward_history = []
        
        # Initialize interaction log
        self.interaction_log = {
            "model_name": model_name,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "interactions": []
        }
        
    def observe(
        self,
        observation: np.ndarray,
        reward: float,
        termination: bool,
        truncation: bool,
        info: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Process the observation and return a message for the LLM.
        
        Args:
            observation (np.ndarray): Current observation
            reward (float): Current reward
            termination (bool): Whether the episode has terminated
            truncation (bool): Whether the episode has been truncated
            info (Optional[Dict[str, Any]]): Additional information
            
        Returns:
            str: Formatted message for the LLM
        """
        # Store reward
        self.reward_history.append(reward)
        
        # Format the observation for better readability
        obs_str = f"Agent {self.agent_id} observation:\n"
        obs_str += f"Position: [{observation[0]:.3f}, {observation[1]:.3f}]\n"
        obs_str += f"Velocity: [{observation[2]:.3f}, {observation[3]:.3f}]\n"
        obs_str += f"Nearest food: [{observation[-2]:.3f}]\n"
        obs_str += f"Nearest poison: [{observation[-1]:.3f}]"
        
        # Create the prompt
        prompt = self.prompt_template.format(
            observation=obs_str,
            reward=reward,
            termination=termination,
            truncation=truncation,
            total_return=sum(self.reward_history)
        )
        
        # Store the prompt
        self.message_history.append(prompt)
        
        # Log the interaction
        self.current_interaction = {
            "timestamp": datetime.now().isoformat(),
            "observation": obs_str,
            "reward": reward,
            "termination": termination,
            "truncation": truncation,
            "total_return": sum(self.reward_history),
            "prompt": prompt
        }
        
        return prompt
    
    def act(self) -> np.ndarray:
        """
        Generate an action using the LLM.
        
        Returns:
            np.ndarray: Action vector [x, y]
        """
        # Get the last message
        if not self.message_history:
            return np.zeros(2)
            
        last_message = self.message_history[-1]
        
        try:
            # Generate response using text generation
            response = self.client.text_generation(
                prompt=last_message,
                model=self.model_name,
                temperature=self.temperature,
                max_new_tokens=self.max_tokens,
                do_sample=True
            )
            
            # Parse the action
            parsed = self.action_parser.parse(response)
            action = np.array([float(parsed["x"]), float(parsed["y"])])
            # Clip to valid range
            action = np.clip(action, -0.01, 0.01)
            
            # Log successful interaction
            self.current_interaction.update({
                "response": response,
                "parsed_action": action.tolist(),
                "success": True
            })
        except Exception as e:
            # If parsing fails, return a random action
            action = np.random.uniform(-0.01, 0.01, 2)
            
            # Log failed interaction
            self.current_interaction.update({
                "response": str(e),
                "parsed_action": action.tolist(),
                "success": False,
                "error": str(e)
            })
        
        # Add interaction to log
        self.interaction_log["interactions"].append(self.current_interaction)
        
        # Save log to file
        with open(self.log_file, 'w') as f:
            json.dump(self.interaction_log, f, indent=2)
        
        return action
    
    def reset(self):
        """Reset the agent's message history and rewards."""
        self.message_history = []
        self.reward_history = [] 