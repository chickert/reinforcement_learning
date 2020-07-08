# Model under construction
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple, deque

#####HYPERPARAMS#####
layer_1_nodes = 128
layer_2_nodes = 64


#####################

# TODO: Qnetwork
#   Loss
#   Verify works with batching

class Qnetwork(nn.Module):
    def __init__(self, observation_shape,
                 num_actions,
                 layer_1_nodes,
                 layer_2_nodes):
        super(Qnetwork, self).__init__()
        self.fc1 = nn.Linear(observation_shape, layer_1_nodes)
        self.fc2 = nn.Linear(layer_1_nodes, layer_2_nodes)
        self.fc3 = nn.Linear(layer_2_nodes, num_actions)

    def forward(self, state):
        # takes in state and outputs q_vals for actions
        layer_1 = F.relu(self.fc1(state))
        layer_2 = F.relu(self.fc2(layer_1))
        q_vals = F.relu(self.fc3(layer_2))
        return q_vals


class Agent():
    def __init__(self):
        pass


SARS = namedtuple('SARS', ('state', 'action', 'reward', 'next_state'))


class ReplayMemory():
    def __init__(self, memory_size):
        self.memory = deque(maxlen=memory_size)

    def add(self, sars):
        self.memory.append(sars)

    def sample(self, batch_size):
        assert len(self.memory) > batch_size, "Batch size is greater than memory size"
        return random.sample(self.memory, batch_size)


# TODO: agent (class) -- interacts with environment and learns from it
#   ACT takes step based on q_vals returned by model, or randomly by epsilon
#   TRAINs models based on replay buffer and update schedule

# TODO: replay memory (class)
#   ensure it works well with namedtuple

# TODO: epsilon
#   anneals by some schedule

if __name__ == "__main__":
    import gym

    env = gym.make('CartPole-v0')
    rep_mem = ReplayMemory(memory_size=100)
    for i_episode in range(3):
        observation = env.reset()
        for t in range(10):
            # env.render()
            # print(observation)
            # import ipdb; ipdb.set_trace()
            action = env.action_space.sample()
            # import ipdb; ipdb.set_trace()
            observation, reward, done, info = env.step(action)




            if done:
                print("Episode finished after {} timesteps".format(t + 1))
                break
    env.close()
