# Model under construction
import random
import gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import namedtuple, deque
import wandb
import tqdm
import numpy as np

#####HYPERPARAMS#####
layer_1_nodes = 128
layer_2_nodes = 64
gamma = 0.999
eps_decay_rate = 0.99999
lr = 1e-4
batch_size = 128
num_episodes = 500
max_timesteps = 300
replay_memory_size = 5_000
episodes_before_target_network_update = 10
#####################


class Qnetwork(nn.Module):
    def __init__(self, observation_shape,
                 num_actions,
                 layer_1_nodes,
                 layer_2_nodes,
                 lr):
        super(Qnetwork, self).__init__()
        self.fc1 = nn.Linear(observation_shape, layer_1_nodes)
        self.fc2 = nn.Linear(layer_1_nodes, layer_2_nodes)
        self.fc3 = nn.Linear(layer_2_nodes, num_actions)
        self.deepnet = nn.Sequential(
            nn.Linear(observation_shape, layer_1_nodes),
            nn.ReLU(),
            nn.Linear(layer_1_nodes, layer_2_nodes),
            nn.ReLU(),
            nn.Linear(layer_2_nodes, num_actions)
        )
        self.optimizer = optim.Adam(self.deepnet.parameters(), lr=lr)

    def forward(self, state):
        # Takes in state and outputs q_vals for all actions
        return self.deepnet(state)


class Agent:
    def __init__(self,
                 q_network,
                 target_network,
                 replay_memory,
                 batch_size,
                 decay_rate):
        self.q_network = q_network
        self.target_network = target_network
        self.replay_memory = replay_memory
        self.batch_size = batch_size
        self.epsilon = 1
        self.eps_decay_rate = decay_rate

    def get_e_greedy_action(self, observation):
        self.epsilon = self.epsilon * self.eps_decay_rate
        if random.random() < self.epsilon:
            # Take random action
            action = env.action_space.sample()
        else:
            # Take action selected by deep q-network
            with torch.no_grad():
                q_vals = self.q_network(observation)
                assert len(q_vals) == 2, "Action selection issue; model outputs q_vals in wrong format"
                action = q_vals.argmax().item()
        return action


SARSD = namedtuple('SARSD', ('state', 'action', 'reward', 'next_state', 'done'))


class ReplayMemory:
    def __init__(self, memory_size):
        self.memory = deque(maxlen=memory_size)

    def add(self, sarsd):
        self.memory.append(sarsd)

    def sample(self, batch_size):
        assert len(self.memory) > batch_size, "Batch size is greater than memory size"
        return random.sample(self.memory, batch_size)


def train(agent, current_step_num, episode_rewards):
    if len(agent.replay_memory.memory) <= agent.batch_size:
        return

    transitions = agent.replay_memory.sample(agent.batch_size)

    cur_states = torch.stack([torch.Tensor(t.state) for t in transitions])
    actions_list = [t.action for t in transitions]
    rewards = torch.stack([torch.Tensor([t.reward]) for t in transitions])
    masks = torch.stack([torch.Tensor([0]) if t.done else torch.Tensor([1]) for t in transitions])
    next_states = torch.stack([torch.Tensor(t.next_state) for t in transitions])

    with torch.no_grad():
        # Use no_grad since we don't want to backprop through target network
        # Take the max since we are only interested in the q val for the action taken
        next_state_q_vals = agent.target_network(next_states).max(-1)[0] # (N, num_actions)

    agent.q_network.optimizer.zero_grad()
    q_vals = agent.q_network(cur_states) # (N, num_actions)
    actions_one_hot = F.one_hot(torch.LongTensor(actions_list), env.action_space.n)

    # Here is the TD error from the Bellman equation
    # The torch.sum() component is to select the q_vals ONLY for the action taken
    # without having to use a loop
    loss = ((rewards + gamma * masks[:, 0] * next_state_q_vals - torch.sum(q_vals * actions_one_hot, -1))**2).mean()
    wandb.log({'loss': loss.detach().item(), 'epsilon': agent.epsilon, 'avg_reward_per_episode': np.mean(episode_rewards)}, step=current_step_num)
    loss.backward()
    agent.q_network.optimizer.step()
    return loss


if __name__ == "__main__":

    wandb.init(project="dqn", name="dqn-cartpole")
    env = gym.make('CartPole-v0')

    replay_memory = ReplayMemory(memory_size=replay_memory_size)
    q_network = Qnetwork(
        observation_shape=env.observation_space.shape[0],
        num_actions=env.action_space.n,
        layer_1_nodes=layer_1_nodes,
        layer_2_nodes=layer_2_nodes,
        lr=lr
    )
    target_network = Qnetwork(
        observation_shape=env.observation_space.shape[0],
        num_actions=env.action_space.n,
        layer_1_nodes=layer_1_nodes,
        layer_2_nodes=layer_2_nodes,
        lr=lr
    )
    agent = Agent(
        q_network=q_network,
        target_network=target_network,
        replay_memory=replay_memory,
        batch_size=batch_size,
        decay_rate=eps_decay_rate
    )

    current_step_num = -1 * replay_memory_size

    tq = tqdm.tqdm()

    for i_episode in range(num_episodes):
        observation = env.reset()
        episode_rewards = []

        for t in range(max_timesteps):

            current_step_num += 1

            action = agent.get_e_greedy_action(observation=torch.Tensor(observation))
            next_observation, reward, done, _ = env.step(action)

            sarsd = SARSD(state=observation,
                          action=action,
                          reward=reward,
                          next_state=next_observation,
                          done=done)
            agent.replay_memory.add(sarsd=sarsd)
            episode_rewards.append(reward)

            train(agent=agent, current_step_num=current_step_num, episode_rewards=episode_rewards)

            observation = next_observation

            if done:
                # print("Episode finished after {} timesteps".format(t + 1))
                break

        if i_episode % episodes_before_target_network_update == 0:
            tq.update(1)
            print("Updating target network")
            agent.target_network.load_state_dict(agent.q_network.state_dict())

    env.close()
