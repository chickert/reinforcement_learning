from pusher_goal import PusherEnv

import numpy as np


class HelperEnv:

    def __init__(self, initial_state, transition_function, reward_function,
                 is_done=None, state_space_dimension=None, action_space_dimension=None):
        self.initial_state = initial_state
        self.state = initial_state
        self.transition_function = transition_function
        self.reward_function = reward_function
        self.is_done = is_done if is_done else lambda state: False
        self.state_space_dimension = state_space_dimension if state_space_dimension else len(self.state)
        self.action_space_dimension = action_space_dimension if action_space_dimension else len(self.state)

    def set_state(self, state):
        self.state = state

    def reset(self):
        self.state = self.initial_state

    def update(self, action):
        self.state = self.transition_function(state=self.state, action=action)
        done = self.is_done(self.state)
        reward = self.reward_function(state=self.state, action=action)
        return reward, done


class PusherEnvModified(HelperEnv):
    def __init__(self):
        self.simulator = PusherEnv(render=False)

        def transition_function(state, action):
            self.simulator.apply_action(action)
            return self.simulator.get_obs()

        def reward_function(state, action):
            return self.simulator.compute_reward_push(state)

        HelperEnv.__init__(self, initial_state=self.simulator.get_obs(), transition_function=transition_function,
                           reward_function=reward_function, state_space_dimension=9, action_space_dimension=2)

    def reset(self):
        self.simulator.reset()
