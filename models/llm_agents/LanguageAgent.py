import tenacity
from langchain.output_parsers import RegexParser
from langchain.schema import HumanMessage, SystemMessage, AIMessage

import inspect

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

Action Sequence: [<action1>, <action2>, ..., <action{planning_horizon}>]

where each <action> is a valid action for the environment.
Do nothing else but return the action sequence.
"""
        self.action_parser = RegexParser(
            regex=r"Action Sequence: \[(.*)\]", 
            output_keys=["action_sequence"], 
            default_output_key="action_sequence"
        )

        self.message_history = []

    def random_action(self):
        action = self.env.action_space.sample()
        return action

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

    def _act(self):
        act_message = self.model(self.message_history)
        action_sequence_str = self.action_parser.parse(act_message.content)["action_sequence"]
        # Convert the string of actions into a list of actions
        action_sequence = [int(action.strip()) for action in action_sequence_str.split(',')]
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
            self.message_history.append(AIMessage(content=f"Action: {action}"))
            self.current_step += 1
    
    def reset_plan(self):
        """Reset planning state while preserving accumulated rewards."""
        self.plan_sequence = []
        self.current_step = 0

class WaterworldAgent(GymnasiumAgent):
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