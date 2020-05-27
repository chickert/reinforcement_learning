import logging

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from architectures.actor_critic import ActorCritic, DEVICE

# Set up logging
logger = logging.getLogger(__name__)


class BCLearner:

    def __init__(
            self,
            policy,
            n_epochs=50,
            batch_size=128,
            learning_rate=3e-4,
            seed=0
    ):
        torch.manual_seed(seed)
        self.policy = policy
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.training_loss = []

    def compute_bc_loss(self, states, actions):
        log_probabilities = self.policy.get_distribution(states).log_prob(actions).float().to(DEVICE)
        return -torch.mean(log_probabilities)

    def update_actor(self, states, actions):

        # Compute BC loss
        loss = self.compute_bc_loss(states=states, actions=actions)

        # Perform gradient update
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.training_loss.append(loss.item())

    def train(self, expert_data):
        steps_per_epoch = int(np.ceil(len(expert_data) / self.batch_size))
        for i in range(self.n_epochs):
            for states, actions in DataLoader(expert_data, batch_size=self.batch_size, shuffle=True):
                states = states.to(DEVICE)
                actions = actions.to(DEVICE)
                self.update_actor(states=states, actions=actions)
            logger.info(f"Epochs completed: {i+1}/{self.n_epochs}")
            logger.info(f"Mean loss: {'{0:.2f}'.format(np.mean(self.training_loss[-steps_per_epoch:]))}")
