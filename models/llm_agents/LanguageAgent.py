import tenacity
from langchain.output_parsers import RegexParser
from langchain.schema import HumanMessage, SystemMessage, AIMessage

import inspect
import numpy as np

class RawAgent:
    """
    A language model agent that can act in the Waterworld environment.
    """
    @classmethod
    def get_docs(cls, env):
        return env.unwrapped.__doc__

    def __init__(self, model, env, planning_horizon=5):
        self.model = model
        self.env = env
        self.docs = self.get_docs(env)
        self.planning_horizon = planning_horizon
        self.plan_sequence = []
        self.current_step = 0
        self.reward_total = 0

        self.instructions = f"""
Your goal is to maximize your return, i.e. the sum of the rewards you receive.
I will give you an observation, reward, termination flag, truncation flag, and the return so far, formatted as:

Observation: <observation>
Reward: <reward>
Termination: <termination>
Truncation: <truncation>
Return: <sum_of_rewards>

You will respond with a sequence of {planning_horizon} actions, formatted as:

Action Sequence: [[x1, y1], [x2, y2], ..., [x{planning_horizon}, y{planning_horizon}]]

where each action is a 2D vector with values between -1 and 1.
x controls horizontal movement (-1 is left, 1 is right)
y controls vertical movement (-1 is up, 1 is down)

Your goal is to catch green circles (food +10) and avoid red circles (poison -1).
Do nothing else but return the action sequence.
"""
        self.action_parser = RegexParser(
            regex=r"Action Sequence: \[(.*)\]", 
            output_keys=["action_sequence"], 
            default_output_key="action_sequence"
        )

        self.message_history = []

    def random_action(self):
        # Generate a random 2D action vector
        return np.random.uniform(-1.0, 1.0, (2,)).astype(np.float32)

    def reset(self):
        """Initialize the agent with environment docs and instructions. 
        This should only be called once at the beginning of the experiment."""
        self.message_history = [
            SystemMessage(content=self.docs),
            SystemMessage(content=self.instructions),
        ]

    def observe(self, obs, reward=0, term=False, trunc=False, info=None):
        self.reward_total += reward

        obs_message = f"""
Observation: {obs}
Reward: {reward}
Termination: {term}
Truncation: {trunc}
Return: {self.reward_total}
        """
        self.message_history.append(HumanMessage(content=obs_message))
        return obs_message

    def _format_messages_to_prompt(self):
        """Convert message history to a single prompt string."""
        formatted_messages = []
        
        for message in self.message_history:
            if isinstance(message, SystemMessage):
                formatted_messages.append(f"System: {message.content}")
            elif isinstance(message, HumanMessage):
                formatted_messages.append(f"Human: {message.content}")
            elif isinstance(message, AIMessage):
                formatted_messages.append(f"AI: {message.content}")
        
        return "\n\n".join(formatted_messages)

    def _act(self):
        # Convert message history to a single prompt string
        prompt = self._format_messages_to_prompt()
        
        # Get completion from the model
        response = self.model(prompt)
        
        # Parse the action sequence from the response
        action_sequence_str = self.action_parser.parse(response)["action_sequence"]
        
        # Process the string to extract 2D actions
        # First, replace whitespace and clean up the string
        cleaned_str = action_sequence_str.replace(" ", "").replace("\n", "")
        
        # Split the string into individual action pairs
        action_pairs = cleaned_str.split("],[")
        
        # Clean up the brackets
        action_pairs = [pair.replace("[", "").replace("]", "") for pair in action_pairs]
        
        # Convert to numpy arrays
        action_sequence = []
        
        for pair in action_pairs:
            try:
                # Split by comma and convert to float
                x, y = pair.split(",")
                action = np.array([float(x), float(y)], dtype=np.float32)
                
                # Clip values to [-1, 1] range
                action = np.clip(action, -1.0, 1.0)
                action_sequence.append(action)
            except (ValueError, IndexError):
                # If parsing fails, use a random action
                action_sequence.append(self.random_action())
        
        # If we didn't get enough actions, pad with random actions
        while len(action_sequence) < self.planning_horizon:
            action_sequence.append(self.random_action())
            
        # If we got too many actions, truncate
        if len(action_sequence) > self.planning_horizon:
            action_sequence = action_sequence[:self.planning_horizon]
        
        return action_sequence

    def act(self):
        try:
            for attempt in tenacity.Retrying(
                stop=tenacity.stop_after_attempt(2),
                wait=tenacity.wait_none(),  # No waiting time between retries
                retry=tenacity.retry_if_exception_type(ValueError),
                before_sleep=lambda retry_state: print(
                    f"ValueError occurred: {retry_state.outcome.exception()}, retrying..."
                ),
            ):
                with attempt:
                    action_sequence = self._act()
                    if len(action_sequence) != self.planning_horizon:
                        raise ValueError(f"Expected {self.planning_horizon} actions, got {len(action_sequence)}")
                    return action_sequence
        except tenacity.RetryError as e:  # noqa: F841
            return [self.random_action() for _ in range(self.planning_horizon)]
    
    def plan(self):
        # Get a sequence of actions from the model
        action_sequence = self.act()
        self.plan_sequence = action_sequence
        self.current_step = 0
        return action_sequence

    def add_act_to_history(self):
        if self.plan_sequence and self.current_step < len(self.plan_sequence):
            action = self.plan_sequence[self.current_step]
            self.message_history.append(AIMessage(content=f"Action: [{action[0]:.4f}, {action[1]:.4f}]"))
            self.current_step += 1
    
    def reset_plan(self):
        """Reset planning state while preserving accumulated rewards."""
        self.plan_sequence = []
        self.current_step = 0

class WaterworldAgent(RawAgent):
    """
    Extend the RawAgent class to handle multi-agents setting in Waterworld.
    """
    @classmethod
    def get_docs(cls, env):
        return inspect.getmodule(env.unwrapped).__doc__

    def __init__(self, name, model, env, planning_horizon=5):
        super().__init__(model, env, planning_horizon)
        self.name = name

    def random_action(self):
        action = self.env.action_space(self.name).sample()
        return action