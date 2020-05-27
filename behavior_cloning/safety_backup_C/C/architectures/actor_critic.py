import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal

from architectures.mlp import MultiLayerPerceptron, DEVICE
from algorithms.param_annealing import AnnealedParam


class ActorCritic(nn.Module):

    def __init__(
            self,
            state_space_dimension,
            action_space_dimension,
            actor_hidden_layer_units=(128, 64),
            critic_hidden_layer_units=(64, 32),
            action_map=None,
            actor_std=1e-2,
            activation=nn.ReLU,
            seed=0,
    ):
        torch.manual_seed(seed)
        super(ActorCritic, self).__init__()

        # Define policy as discrete or continuous
        self.action_map = action_map
        if self.action_map:
            self.actor_is_discrete = True
            self.inverse_action_map = {tuple(action): key for key, action in action_map.items()}
        else:
            self.actor_is_discrete = False
        self.actor_std = actor_std

        # Make actor network
        self.actor = MultiLayerPerceptron(
            in_features=state_space_dimension,
            out_features=action_space_dimension,
            hidden_layer_units=actor_hidden_layer_units,
            activation=activation,
            softmax_output=self.actor_is_discrete
        ).to(DEVICE)

        # Make critic network
        self.critic = MultiLayerPerceptron(
            in_features=state_space_dimension,
            out_features=1,
            hidden_layer_units=critic_hidden_layer_units,
            activation=activation,
            softmax_output=False
        ).to(DEVICE)

    def get_distribution(self, states):
        if self.actor_is_discrete:
            return Categorical(self.actor(states).to(DEVICE))
        else:
            return Normal(loc=self.actor(states).to(DEVICE), scale=self.actor_std)

    def get_distribution_argmax(self, states):
        if self.actor_is_discrete:
            return self.actor(states).to(DEVICE).argmax()
        else:
            return self.actor(states).to(DEVICE)

    def forward(self, states, actions):
        dist = self.get_distribution(states)
        log_probabilities = dist.log_prob(actions)
        entropy = dist.entropy()
        values = self.critic(states).to(DEVICE)
        return log_probabilities, values, entropy

    def sample_action(self, state):
        state = torch.tensor(state).float().to(DEVICE)
        action = self.get_distribution(state).sample()
        if DEVICE == 'cuda':
            action = action.cpu()
        if self.actor_is_discrete:
            return self.action_map[action.item()]
        else:
            return action.detach().numpy()

    def get_argmax_action(self, state):
        state = torch.tensor(state).float().to(DEVICE)
        action = self.get_distribution_argmax(state)
        if DEVICE == 'cuda':
            action = action.cpu()
        if self.actor_is_discrete:
            return self.action_map[action.item()]
        else:
            return action.detach().numpy()

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))

