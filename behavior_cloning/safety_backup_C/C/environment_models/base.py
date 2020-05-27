import numpy as np


class BaseEnv:

    def __init__(
            self,
            initial_state,
            transition_function,
            reward_function,
            is_done=None,
            state_space_dimension=None,
            action_space_dimension=None,
    ):
        self.initial_state = initial_state
        self.state = initial_state
        self.transition_function = transition_function
        self.reward_function = reward_function
        self.is_done = is_done if is_done else lambda state: False
        self.state_space_dimension = state_space_dimension if state_space_dimension else len(self.state)
        self.action_space_dimension = action_space_dimension if action_space_dimension else len(self.state)

    def update(self, action):
        self.state = self.transition_function(state=self.state, action=action)
        done = self.is_done(self.state)
        reward = self.reward_function(state=self.state, action=action)
        return reward, done

    def set_state(self, state):
        self.state = state

    def reset(self):
        self.state = self.initial_state
