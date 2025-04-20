import tenacity
from langchain.output_parsers import RegexParser
from langchain.schema import HumanMessage, SystemMessage, AIMessage

import inspect
import numpy as np
import re

MAX_TURNS = 4 \



# -----------------------------------------------------------
#  Lightweight greedy step: points toward nearest food sensor
# -----------------------------------------------------------
def _greedy_direction(obs_array, n_sensors):
    """Return a tiny thrust vector toward the closest food sensor."""
    # food distances lie in indices [2*n_sensors : 3*n_sensors)
    start = 2 * n_sensors
    end   = 3 * n_sensors
    dists = obs_array[start:end]
    if dists.size == 0:
        return np.zeros(2)
    idx = int(np.argmin(dists))
    if dists[idx] >= 1.0:          # no food detected
        return np.zeros(2)
    angle = 2 * np.pi * (idx / n_sensors)
    direction = np.array([np.cos(angle), np.sin(angle)], dtype=np.float32)
    direction /= np.linalg.norm(direction)
    return 0.005 * direction       # small safe step (half max_thrust)









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
Replace x's and y's with the actual numerical values that satisfy the environment's action space while maximizing the return.

For example:
Action Sequence: [[-0.5, 0.7], [0.2, -0.3], [0.8, 0.1], [-0.4, -0.6], [0.1, 0.9]]  when planning_horizon = 5

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
        self.last_obs     = obs
        obs_message = f"""
Observation: {obs}
Reward: {reward}
Termination: {term}
Truncation: {trunc}
Return: {self.reward_total}
        """
        self.message_history.append(HumanMessage(content=obs_message))
        if len(self.message_history) > 2 + 2*MAX_TURNS:
            # preserve the two system messages and the K most‑recent pairs
            self.message_history = (
                self.message_history[:2] +          # docs + instructions
                self.message_history[-2*MAX_TURNS:]
            )       


        return obs_message

    def _format_messages_to_prompt(self):
        """Convert message history to a list of chat messages."""
        return self.message_history

    def _act(self):
        # Get completion from the model using chat format
        response = self.model.invoke(self.message_history)
        
        # Print the raw response from the model
        print("\n--- Raw LLM Response ---")
        print(response)  # response is already a string
        
        try:
            # Parse the action sequence from the response
            action_sequence_str = self.action_parser.parse(response)["action_sequence"]
            
            # Print the parsed action sequence string
            # print("\n--- Parsed Action Sequence String ---")
            # print(action_sequence_str)
            
            # Process the string to extract 2D actions
            # Use regex to find all coordinate pairs
            # This pattern looks for pairs of numbers inside brackets: [num, num]
            pattern = r'\[([-+]?[0-9]*\.?[0-9]+),\s*([-+]?[0-9]*\.?[0-9]+)\]'
            matches = re.findall(pattern, action_sequence_str)
            
            # print(f"Found {len(matches)} valid coordinate pairs")
            
            # Convert to numpy arrays
            action_sequence = []
            
            if matches:
                for x_str, y_str in matches:
                    try:
                        x = float(x_str)
                        y = float(y_str)
                        action = np.array([x, y], dtype=np.float32)
                        
                        # Clip values to [-1, 1] range
                        action = np.clip(action, -1.0, 1.0)
                        action_sequence.append(action)
                    except ValueError as e:
                        # print(f"Error converting to float: {x_str}, {y_str}. Error: {e}")
                        action_sequence.append(self.random_action())
            
            # If no valid pairs were found, fall back to old method
            if not action_sequence:
                # print("No valid coordinate pairs found. Trying fallback parsing method...")
                
                # Clean up and extract action pairs
                cleaned_str = action_sequence_str.replace(" ", "").replace("\n", "")
                # Replace placeholder text
                cleaned_str = re.sub(r'x\d+', '0', cleaned_str)  # Replace x1, x2, etc. with 0
                cleaned_str = re.sub(r'y\d+', '0', cleaned_str)  # Replace y1, y2, etc. with 0
                cleaned_str = re.sub(r'\.\.\.,', '', cleaned_str)  # Remove ellipsis
                
                # Split the string into individual action pairs
                action_pairs = cleaned_str.split("],[")
                # Clean up the brackets
                action_pairs = [pair.replace("[", "").replace("]", "") for pair in action_pairs]
                
                for pair in action_pairs:
                    try:
                        x, y = pair.split(",")
                        x = float(x)
                        y = float(y)
                        action = np.array([x, y], dtype=np.float32)
                        action = np.clip(action, -1.0, 1.0)
                        action_sequence.append(action)
                    except (ValueError, IndexError) as e:
                        # print(f"Fallback parsing failed for: {pair}. Error: {e}")
                        action_sequence.append(self.random_action())
        except Exception as e:
            print(f"Error parsing LLM response: {e}")
            # print(f"Full response that failed to parse: {response}")
            # Generate random actions if parsing completely fails
            action_sequence = []
        
        # change original If we didn't get enough actions, pad with random actions
        if len(action_sequence) < self.planning_horizon:
           obs_vec = np.asarray(self.last_obs).flatten()
    # heuristic sensor count from obs length (5 or 8 per sensor)
           for _ in range(self.planning_horizon - len(action_sequence)):
               n_feat = obs_vec.size
               n_s = max(n_feat // 8, n_feat // 5, 1)
               action_sequence.append(_greedy_direction(obs_vec, n_s))

            
        # If we got too many actions, truncate
        if len(action_sequence) > self.planning_horizon:
            # print(f"Too many actions ({len(action_sequence)}), truncating to {self.planning_horizon}")
            action_sequence = action_sequence[:self.planning_horizon]
        
        # Print final processed action sequence
        # print("\n--- Final Processed Actions ---")
        # for i, action in enumerate(action_sequence):
        #     print(f"Action {i+1}: [{action[0]:.4f}, {action[1]:.4f}]")
             
        # final safety: scale if norm > 0.01 and clip to [-1,1]
        for i, a in enumerate(action_sequence):
            norm = np.linalg.norm(a)
            if norm > 0.01:
               action_sequence[i] = (a / norm) * 0.01
            action_sequence[i] = np.clip(action_sequence[i], -1.0, 1.0)




        
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
                    if len(action_sequence) == 0:
                        raise ValueError("No valid actions returned")
                    # Return the first action from the sequence
                    return action_sequence[0]
        except tenacity.RetryError as e:  # noqa: F841
            # Return a random action if all attempts failed
            return self.random_action()
    
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