import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class MLP(nn.Sequential):

    def __init__(self, in_dims, out_dims, hidden_layer_units, activation, softmax_output=False):

        # Build layers
        layers = [nn.Linear(in_dims, hidden_layer_units[0]).to(DEVICE), activation().to(DEVICE)]
        for i in range(1, len(hidden_layer_units)):
            layers += [nn.Linear(hidden_layer_units[i-1], hidden_layer_units[i]).to(DEVICE),
                       activation().to(DEVICE)]
        layers += [nn.Linear(hidden_layer_units[-1], out_dims).to(DEVICE)]
        if softmax_output:
            layers += [nn.Softmax(dim=-1).to(DEVICE)]

        super().__init__(*layers)

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))


# For this I used the ActorCritic we have been using in our project
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
        self.actor = MLP(
            in_dims=state_space_dimension,
            out_dims=action_space_dimension,
            hidden_layer_units=actor_hidden_layer_units,
            activation=activation,
            softmax_output=self.actor_is_discrete
        ).to(DEVICE)

        # Make critic network
        self.critic = MLP(
            in_dims=state_space_dimension,
            out_dims=1,
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

