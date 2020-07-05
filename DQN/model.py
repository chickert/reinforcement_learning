# Model under construction
import torch
import torch.nn as nn


# TODO: deepQnetwork (class nn.module?) --
#   FORWARD: takes in state and outputs q_vals for actions
#   LOSS?
#   needs to be able to take in many? or just one? ie How to batch?

class Qnetwork(nn.Module):
    def __init__(self, observation_shape,
                 num_actions,
                 layer_1_nodes=128,
                 layer_2_nodes=64):
        super(Qnetwork, self).__init__()
        self.observation_space = observation_shape
        self.num_actions = num_actions


    def forward(self):
        pass


class Agent():
    def __init__(self):
        pass

# TODO: agent (class) -- has agency (duh); interacts with environment and learns from it
#   ACT takes step based on q_vals returned by model, or randomly by epsilon
#   TRAINs models based on replay buffer and update schedule

# TODO: replay memory (class)
#   initialized to some fixed capacity
#   can FILL with experiences, but not overfill
#   can SAMPLE from

# TODO: epsilon
#   anneals by some schedule

import gym
env = gym.make('CartPole-v0')
for i_episode in range(20):
    observation = env.reset()
    for t in range(100):
        # env.render()
        print(observation)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()